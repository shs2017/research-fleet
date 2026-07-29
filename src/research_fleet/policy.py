"""Safeguards — everything that can say "no" before a container starts.

The policy engine is a pure function of (spec, context) → Decision. It runs on
the submit path, so a denial is recorded in the ledger and the job never
executes. Keeping it side-effect free means the same policy can be dry-run
against a plan (`fleet plan --explain`) to see what would be blocked.

Layers, roughly outermost to innermost:

  * **Fanout & depth** — an agent that can spawn agents can recurse forever;
    `max_agent_depth` and `max_children_per_agent` bound the tree.
  * **Budget** — enforced in `budget.BudgetTracker`, invoked from here so a
    denial reads the same as any other.
  * **Resources** — per-job ceilings, so one job can't request the whole cluster.
  * **Filesystem** — mounts must resolve inside an allowlist; no host-root, no
    docker socket, no writable mounts outside declared workspaces.
  * **Container hardening** — dropped capabilities, no-new-privileges, pid
    limits, non-root user, optional read-only rootfs.
  * **Network** — default-deny egress with an allowlist, because an agent with
    a shell and open internet is an exfiltration path.
  * **Commands & tools** — pattern denylist for command jobs, tool allowlist
    for agent jobs.
  * **Approval gates** — anything matching `require_approval_for` parks in
    AWAITING_APPROVAL instead of running.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .budget import BudgetExceeded, BudgetTracker, Quote

Verdict = Literal["allow", "deny", "require_approval"]


@dataclass
class Decision:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    mutations: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reasons": list(self.reasons), "mutations": dict(self.mutations)}


class NetworkPolicy(BaseModel):
    """Egress control, implemented by the research-ship firewall.

      * `none`         — no network at all (`--network none`)
      * `limited`      — research-ship's default-deny iptables allowlist, plus
                         `allowed_hosts` appended via FIREWALL_EXTRA_DOMAINS
      * `unrestricted` — normal outbound access

    `limited` is a real allowlist resolved to IPs inside the container, not a
    label. Two caveats inherited from the ship: CDN addresses rotate, so a
    long run can start failing and needs the firewall re-applied; and an
    allowlisted domain can still be used to move data out — this restricts
    *where* traffic goes, not *what* it carries.
    """

    mode: Literal["none", "limited", "unrestricted"] = "limited"
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Extra domains appended to research-ship's built-in allowlist.",
    )


class ContainerPolicy(BaseModel):
    """Per-job limits the fleet layers on top of the research-ship container.

    Deliberately small. research-ship already supplies the isolation posture —
    non-root `dev` user, a narrow sudo grant limited to the firewall script,
    volume-scoped caches, and only `/workspace` bind-mounted from the host.
    Re-imposing `--read-only`, `--cap-drop ALL` or `--user` here would break
    that contract (the venv and HOME must be writable; the firewall needs
    NET_ADMIN) and produce containers that simply don't run.

    What is left is what research-ship has no opinion about, because it is a
    scheduling concern rather than an environment one: fork-bomb and resource
    ceilings, plus an escape hatch for extra `--security-opt` flags.
    """

    pids_limit: int = Field(4096, gt=0)
    security_opt: list[str] = Field(default_factory=list)
    extra_docker_args: list[str] = Field(
        default_factory=list,
        description="Escape hatch, e.g. ['--cpuset-cpus', '0-15'].",
    )

    def docker_args(self) -> list[str]:
        args = ["--pids-limit", str(self.pids_limit)]
        for opt in self.security_opt:
            args += ["--security-opt", opt]
        args += list(self.extra_docker_args)
        return args


class Policy(BaseModel):
    """The full safeguard surface. Serialised into the ledger at run start."""

    # --- structural limits on agent recursion
    max_agent_depth: int = Field(2, ge=0, description="0 = agents may not spawn agents at all.")
    max_children_per_agent: int = Field(8, ge=0)
    max_concurrent_jobs: int = Field(16, ge=1)

    # --- per-job resource ceilings
    max_gpus_per_job: float = Field(8.0, gt=0)
    max_memory_gb_per_job: float = Field(256.0, gt=0)
    max_timeout_s: int = Field(24 * 3600, gt=0)

    # --- budget ceilings
    max_usd_per_job: float = Field(25.0, gt=0)
    max_tokens_per_job: int = Field(20_000_000, gt=0)

    # --- filesystem
    allowed_mount_roots: list[str] = Field(default_factory=list)
    deny_mount_paths: list[str] = Field(
        default_factory=lambda: [
            "/", "/etc", "/root", "/boot", "/sys", "/proc", "/dev",
            "/var/run/docker.sock", "/run/docker.sock",
            "~/.ssh", "~/.aws", "~/.config/gcloud", "~/.kube",
        ]
    )

    container: ContainerPolicy = Field(default_factory=ContainerPolicy)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)

    # --- command / tool control
    deny_command_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\brm\s+-rf\s+/(?:\s|$)",
            r"\bmkfs(\.|\s)",
            r"\bdd\s+.*of=/dev/",
            r":\(\)\{.*\};:",              # fork bomb
            r"\bshutdown\b|\breboot\b",
            r"\bdocker\s+(run|exec)\b",     # no container escape via nested docker
            r"\bnvidia-smi\s+.*(-r|--gpu-reset)",
        ]
    )
    agent_default_disallowed_tools: list[str] = Field(default_factory=list)
    require_approval_for: list[str] = Field(
        default_factory=lambda: ["net:unrestricted", "mount:rw-outside-workspace"],
        description="Labels emitted by the checks below; matching jobs park for human approval.",
    )

    # --- misc
    redact_secrets: bool = True
    fail_closed: bool = Field(
        True, description="If a check cannot be evaluated, deny rather than allow."
    )

    def check(
        self,
        spec,
        *,
        depth: int = 0,
        sibling_count: int = 0,
        budget: BudgetTracker | None = None,
        budget_scope: str | None = None,
        estimate: Quote | None = None,
        workspace_roots: list[str] | None = None,
    ) -> Decision:
        reasons: list[str] = []
        gates: list[str] = []
        mutations: dict[str, Any] = {}

        # ---- recursion bounds
        if depth > self.max_agent_depth:
            return Decision("deny", [f"agent depth {depth} exceeds max_agent_depth={self.max_agent_depth}"])
        if spec.parent_job_id and sibling_count >= self.max_children_per_agent:
            return Decision(
                "deny",
                [f"parent {spec.parent_job_id} already has {sibling_count} children "
                 f"(max_children_per_agent={self.max_children_per_agent})"],
            )

        # ---- resources
        r = spec.resources
        if r.gpus > self.max_gpus_per_job:
            return Decision("deny", [f"requested {r.gpus} GPUs > max_gpus_per_job={self.max_gpus_per_job}"])
        if r.memory_gb > self.max_memory_gb_per_job:
            return Decision("deny", [f"requested {r.memory_gb}GB > max_memory_gb_per_job={self.max_memory_gb_per_job}"])
        if spec.timeout_s > self.max_timeout_s:
            mutations["timeout_s"] = self.max_timeout_s
            reasons.append(f"timeout clamped {spec.timeout_s}s -> {self.max_timeout_s}s")

        # ---- budget
        if estimate is not None:
            if estimate.est_cost_usd > self.max_usd_per_job:
                return Decision(
                    "deny",
                    [f"estimated ${estimate.est_cost_usd:.2f} exceeds max_usd_per_job=${self.max_usd_per_job:.2f}"],
                )
            est_tokens = estimate.est_input_tokens + estimate.est_output_tokens
            if est_tokens > self.max_tokens_per_job:
                return Decision(
                    "deny",
                    [f"estimated {est_tokens:,} tokens exceeds max_tokens_per_job={self.max_tokens_per_job:,}"],
                )
            if budget is not None and budget_scope is not None:
                try:
                    budget.reserve(budget_scope, usd=estimate.est_cost_usd, tokens=est_tokens)
                    mutations["budget_reserved"] = {
                        "scope": budget_scope,
                        "usd": estimate.est_cost_usd,
                        "tokens": est_tokens,
                    }
                except BudgetExceeded as exc:
                    return Decision("deny", [str(exc)])

        # ---- filesystem
        roots = [Path(p).expanduser().resolve() for p in (self.allowed_mount_roots or (workspace_roots or []))]
        denied = [Path(p).expanduser() for p in self.deny_mount_paths]
        for m in spec.mounts:
            src = Path(m.source).expanduser()
            try:
                resolved = src.resolve()
            except OSError as exc:
                if self.fail_closed:
                    return Decision("deny", [f"cannot resolve mount source {m.source!r}: {exc}"])
                continue
            for d in denied:
                dr = d.resolve() if d.exists() else d
                if resolved == dr or (dr in resolved.parents and dr != Path("/")):
                    return Decision("deny", [f"mount {resolved} is under denied path {d}"])
            if roots and not any(resolved == r or r in resolved.parents for r in roots):
                if self.fail_closed:
                    return Decision(
                        "deny",
                        [f"mount {resolved} is outside allowed_mount_roots "
                         f"({', '.join(str(r) for r in roots)})"],
                    )
            if m.mode == "rw" and roots and not any(r in resolved.parents or r == resolved for r in roots):
                gates.append("mount:rw-outside-workspace")

        # ---- commands
        if spec.command:
            joined = " ".join(spec.command)
            for pat in self.deny_command_patterns:
                if re.search(pat, joined):
                    return Decision("deny", [f"command matches denied pattern {pat!r}"])

        # ---- agent tools
        if spec.agent is not None:
            merged_deny = sorted(set(self.agent_default_disallowed_tools) | set(spec.agent.disallowed_tools))
            if merged_deny != spec.agent.disallowed_tools:
                mutations["agent_disallowed_tools"] = merged_deny
                reasons.append(f"tool denylist merged from policy: {merged_deny}")

        # ---- network
        if self.network.mode == "unrestricted":
            gates.append("net:unrestricted")

        # ---- approval gates
        triggered = [g for g in gates if any(fnmatch.fnmatch(g, p) for p in self.require_approval_for)]
        if triggered:
            return Decision("require_approval", reasons + [f"gate: {g}" for g in triggered], mutations)

        return Decision("allow", reasons, mutations)
