"""Token and cost accounting.

Three things live here:

  1. **A price table** (`MODEL_COSTS`): per-model input/output rates plus the
     cache multipliers, so spend is computed from real `usage` fields rather
     than guessed.
  2. **Effort/process multipliers**: the same prompt on the same model costs
     very different amounts at `low` vs `max` effort, and an agentic loop costs
     a multiple of a single call. These are *estimation* heuristics used for
     pre-flight quotes; actual spend always comes from reported usage.
  3. **A hierarchical budget tree** (`BudgetTracker`): a run gets an
     allocation; each job reserves against it; an agent job that spawns
     children sub-allocates from *its own* remaining balance. A child can never
     spend budget its parent doesn't have, so a runaway sub-agent is bounded by
     construction rather than by a global limit it might race other jobs for.

Prices are a snapshot (see `PRICES_AS_OF`) and are overridable from config:
never let a stale table silently misreport spend.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

PRICES_AS_OF = "2026-08-05"

MTOK = 1_000_000


@dataclass(frozen=True)
class ModelCost:
    """USD per million tokens, plus cache multipliers relative to input price."""

    model: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int = 200_000
    max_output: int = 64_000
    cache_read_mult: float = 0.1
    cache_write_mult: float = 1.25      # 5-minute TTL; 1h TTL is 2.0
    cache_write_1h_mult: float = 2.0

    def cost_usd(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_ttl: str = "5m",
    ) -> float:
        write_mult = self.cache_write_1h_mult if cache_ttl == "1h" else self.cache_write_mult
        return (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
            + cache_read_tokens * self.input_per_mtok * self.cache_read_mult
            + cache_write_tokens * self.input_per_mtok * write_mult
        ) / MTOK


# Snapshot of published list prices. Override via config `budget.model_costs`.
MODEL_COSTS: dict[str, ModelCost] = {
    m.model: m
    for m in [
        ModelCost("claude-fable-5", 10.00, 50.00, 1_000_000, 128_000),
        ModelCost("claude-opus-5", 5.00, 25.00, 1_000_000, 128_000),
        ModelCost("claude-opus-4-8", 5.00, 25.00, 1_000_000, 128_000),
        ModelCost("claude-opus-4-7", 5.00, 25.00, 1_000_000, 128_000),
        ModelCost("claude-opus-4-6", 5.00, 25.00, 1_000_000, 128_000),
        ModelCost("claude-sonnet-5", 3.00, 15.00, 1_000_000, 128_000),
        ModelCost("claude-sonnet-4-6", 3.00, 15.00, 1_000_000, 128_000),
        ModelCost("claude-haiku-4-5", 1.00, 5.00, 200_000, 64_000),
        ModelCost("gpt-5.3-codex", 1.75, 14.00, 400_000, 128_000),
        ModelCost("gpt-5.6-sol", 5.00, 30.00, 1_050_000, 128_000),
        ModelCost("gpt-5.6", 5.00, 30.00, 1_050_000, 128_000),
        ModelCost("gpt-5.6-terra", 2.50, 15.00, 1_050_000, 128_000),
        ModelCost("gpt-5.6-luna", 1.00, 6.00, 1_050_000, 128_000),
        ModelCost("gpt-5.4", 2.50, 15.00, 1_050_000, 128_000),
        ModelCost("gpt-5.4-mini", 0.75, 4.50, 400_000, 128_000),
        ModelCost("gpt-5.4-nano", 0.20, 1.25, 400_000, 128_000),
    ]
}

ALIASES = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5",
    "default": "claude-opus-5",
    "codex": "gpt-5.3-codex",
}

# Relative output volume by reasoning effort, normalised to `high` = 1.0. Effort mostly
# changes how much the model thinks, so it scales output rather than input.
EFFORT_MULTIPLIER = {
    "low": 0.4,
    "medium": 0.7,
    "high": 1.0,
    "xhigh": 1.5,
    "max": 2.2,
}

# Token profiles per shape of work, as (fresh input, cached input, output).
#
# Calibrated against measured runs rather than derived from a single-call figure. Two
# things the earlier multiplier-based estimate got badly wrong: output tokens are far
# smaller than input (a long session still only writes a few thousand tokens), and most
# input is cache reads at a tenth of the price. Together those made it overestimate by
# more than an order of magnitude, which then caused legitimate jobs to be refused for
# breaching a budget they would never have reached.
PROCESS_PROFILES = {
    "single_call":    (4_000, 0, 800),
    "workflow":       (8_000, 20_000, 1_500),
    "agent_short":    (10_000, 60_000, 2_000),
    "agent_standard": (20_000, 150_000, 4_000),
    "agent_long":     (60_000, 600_000, 15_000),
}

# Kept for callers that still pass it; the profiles above supersede it.
PROCESS_MULTIPLIER = {name: 1.0 for name in PROCESS_PROFILES}


class BudgetExceeded(Exception):
    def __init__(self, scope: str, requested: float, remaining: float, unit: str):
        super().__init__(
            f"budget exceeded for {scope}: requested {requested:,.2f} {unit}, "
            f"{remaining:,.2f} {unit} remaining"
        )
        self.scope, self.requested, self.remaining, self.unit = scope, requested, remaining, unit


def resolve_model(model: str | None) -> str:
    if not model:
        return ALIASES["default"]
    return ALIASES.get(model, model)


def cost_for(model: str | None) -> ModelCost:
    key = resolve_model(model)
    if key not in MODEL_COSTS:
        raise KeyError(
            f"unknown model {key!r}; known: {', '.join(sorted(MODEL_COSTS))}. "
            "Add it under budget.model_costs in your config."
        )
    return MODEL_COSTS[key]


def register_model_cost(mc: ModelCost) -> None:
    MODEL_COSTS[mc.model] = mc


@dataclass
class Usage:
    """Normalised token counts. Backends map their own shapes onto this."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens + self.output_tokens
            + self.cache_read_tokens + self.cache_write_tokens
        )

    @property
    def priced(self) -> bool:
        """False when the model is absent from the price table, so spend is unknown."""
        try:
            return bool(self.model) and bool(cost_for(self.model))
        except KeyError:
            return False

    def cost_usd(self, model: str | None = None) -> float:
        """Zero for an unpriced model rather than raising.

        Harnesses report model names we do not price, including synthetic ones for
        locally generated messages. Charging nothing is wrong but survivable; taking
        down a running fleet over a label is not. `priced` exposes the difference.
        """
        key = model or self.model
        if not key:
            return 0.0
        try:
            mc = cost_for(key)
        except KeyError:
            return 0.0
        return mc.cost_usd(
            self.input_tokens, self.output_tokens,
            self.cache_read_tokens, self.cache_write_tokens,
        )

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            model=self.model or other.model,
            requests=self.requests + other.requests,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd(), 6) if self.model else None,
            "unpriced_model": bool(self.model) and not self.priced,
        }


@dataclass
class Quote:
    """A pre-flight estimate an agent can act on before spawning work."""

    model: str
    effort: str
    process: str
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    source: str = "estimated"       # or "measured over N past job(s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "process": self.process,
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_cost_usd": round(self.est_cost_usd, 4),
            "source": self.source,
        }


def quote(
    model: str | None = None,
    *,
    effort: str = "high",
    process: str = "agent_standard",
    observed: dict[str, Any] | None = None,
) -> Quote:
    """Estimate the cost of a unit of work before committing to it.

    `observed` is a summary of what work like this has actually cost, from the ledger.
    When there is enough history it wins: measurement beats a profile. Without it, the
    profile for `process` is scaled by `effort`.
    """
    mc = cost_for(model)

    if observed and observed.get("samples", 0) >= 3:
        return Quote(
            mc.model, effort, process,
            int(observed.get("input_tokens", 0)),
            int(observed.get("output_tokens", 0)),
            float(observed["cost_usd"]),
            source=f"measured over {observed['samples']} past job(s)",
        )

    fresh, cached, out = PROCESS_PROFILES.get(process, PROCESS_PROFILES["agent_standard"])
    out = int(out * EFFORT_MULTIPLIER.get(effort, 1.0))
    est = mc.cost_usd(input_tokens=fresh, output_tokens=out, cache_read_tokens=cached)
    return Quote(mc.model, effort, process, fresh + cached, out, est, source="estimated")


def cost_menu(
    models: Iterable[str] | None = None,
    *,
    efforts: Iterable[str] = ("low", "medium", "high", "xhigh"),
    process: str = "agent_standard",
) -> list[dict[str, Any]]:
    """The table handed to an agent so it can pick a model for a sub-task on price.

    Rendered into the agent's prompt by `render_cost_brief`.
    """
    models = list(models or ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"])
    rows = []
    for m in models:
        mc = cost_for(m)
        for e in efforts:
            q = quote(m, effort=e, process=process)
            rows.append(
                {
                    "model": mc.model,
                    "effort": e,
                    "input_per_mtok": mc.input_per_mtok,
                    "output_per_mtok": mc.output_per_mtok,
                    "est_cost_usd": round(q.est_cost_usd, 3),
                    "est_total_tokens": q.est_input_tokens + q.est_output_tokens,
                }
            )
    return rows


def render_cost_brief(
    remaining_usd: float,
    remaining_tokens: int,
    *,
    models: Iterable[str] | None = None,
    process: str = "agent_standard",
) -> str:
    """Human/agent-readable budget briefing injected into agent job prompts.

    An agent that can spawn sub-agents needs to know what they cost and what it
    has left, or it will either overspend or be uselessly conservative.
    """
    lines = [
        "## Budget",
        f"You have **${remaining_usd:,.2f}** and **{remaining_tokens:,} tokens** remaining "
        "for this task *including* any sub-agents you launch.",
        "",
        f"Estimated cost of one delegated sub-task ({process.replace('_', ' ')}):",
        "",
        "| model | effort | est. cost | est. tokens | $/Mtok in | $/Mtok out |",
        "|---|---|---|---|---|---|",
    ]
    for row in cost_menu(models, process=process):
        lines.append(
            f"| {row['model']} | {row['effort']} | ${row['est_cost_usd']:.3f} | "
            f"{row['est_total_tokens']:,} | ${row['input_per_mtok']:.2f} | ${row['output_per_mtok']:.2f} |"
        )
    lines += [
        "",
        "Guidance: delegate broad, parallel, or low-stakes work to a cheaper model at "
        "lower effort; reserve the expensive tier for the reasoning that actually needs it. "
        "Every sub-agent you launch debits the balance above: if a reservation would exceed "
        "it, the launch is rejected rather than silently truncated, so quote before you spawn.",
        f"(Prices as of {PRICES_AS_OF}.)",
    ]
    return "\n".join(lines)


@dataclass
class BudgetNode:
    scope: str
    max_usd: float
    max_tokens: int
    parent: str | None = None
    reserved_usd: float = 0.0
    reserved_tokens: int = 0
    spent_usd: float = 0.0
    spent_tokens: int = 0
    children: list[str] = field(default_factory=list)

    @property
    def committed_usd(self) -> float:
        """Reservations plus actual spend: what's already claimed."""
        return self.reserved_usd + self.spent_usd

    @property
    def committed_tokens(self) -> int:
        return self.reserved_tokens + self.spent_tokens

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.committed_usd)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.committed_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "parent": self.parent,
            "max_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 6),
            "reserved_usd": round(self.reserved_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "max_tokens": self.max_tokens,
            "spent_tokens": self.spent_tokens,
            "reserved_tokens": self.reserved_tokens,
            "remaining_tokens": self.remaining_tokens,
            "children": list(self.children),
        }


class BudgetTracker:
    """Hierarchical reserve/commit ledger for tokens and dollars.

    Flow for a job:  `reserve()` before it starts → `commit()` with real usage
    when it ends. Reservation is what stops a fleet of 20 agents from each
    individually fitting the budget while collectively blowing it.

    A child scope's ceiling is carved out of the parent's *remaining* balance,
    so spend rolls up: charging a sub-agent charges every ancestor too.
    """

    def __init__(self):
        self._nodes: dict[str, BudgetNode] = {}
        self._lock = threading.RLock()

    def open(
        self,
        scope: str,
        *,
        max_usd: float,
        max_tokens: int,
        parent: str | None = None,
    ) -> BudgetNode:
        with self._lock:
            if scope in self._nodes:
                return self._nodes[scope]
            if parent is not None:
                p = self._require(parent)
                # A child can only be granted what the parent still has.
                if max_usd > p.remaining_usd:
                    raise BudgetExceeded(f"{parent} -> {scope}", max_usd, p.remaining_usd, "USD")
                if max_tokens > p.remaining_tokens:
                    raise BudgetExceeded(f"{parent} -> {scope}", max_tokens, p.remaining_tokens, "tokens")
                # The grant is held against the parent until the child closes.
                p.reserved_usd += max_usd
                p.reserved_tokens += max_tokens
                p.children.append(scope)
            node = BudgetNode(scope=scope, max_usd=max_usd, max_tokens=max_tokens, parent=parent)
            self._nodes[scope] = node
            return node

    def _require(self, scope: str) -> BudgetNode:
        if scope not in self._nodes:
            raise KeyError(f"no budget scope {scope!r}")
        return self._nodes[scope]

    def reserve(self, scope: str, *, usd: float, tokens: int) -> None:
        with self._lock:
            node = self._require(scope)
            if usd > node.remaining_usd:
                raise BudgetExceeded(scope, usd, node.remaining_usd, "USD")
            if tokens > node.remaining_tokens:
                raise BudgetExceeded(scope, tokens, node.remaining_tokens, "tokens")
            node.reserved_usd += usd
            node.reserved_tokens += tokens

    def release(self, scope: str, *, usd: float, tokens: int) -> None:
        """Drop a reservation without spending it (job denied, cancelled, or finished cheaper)."""
        with self._lock:
            node = self._require(scope)
            node.reserved_usd = max(0.0, node.reserved_usd - usd)
            node.reserved_tokens = max(0, node.reserved_tokens - tokens)

    def commit(self, scope: str, usage: Usage, *, release_usd: float = 0.0, release_tokens: int = 0) -> float:
        """Record real spend and roll it up to every ancestor. Returns cost in USD."""
        cost = usage.cost_usd()
        with self._lock:
            if release_usd or release_tokens:
                self.release(scope, usd=release_usd, tokens=release_tokens)
            cur: str | None = scope
            while cur is not None:
                node = self._require(cur)
                node.spent_usd += cost
                node.spent_tokens += usage.total_tokens
                cur = node.parent
        return cost

    def close(self, scope: str) -> None:
        """Return a child's unused grant to its parent.

        While a child scope is open the parent holds *both* the full grant (as a
        reservation) and the child's rolled-up spend, so the parent transiently
        under-reports its own remaining balance. That error is in the safe
        direction: the parent refuses work it could technically afford rather
        than admitting work it cannot, and this call reconciles it, leaving the
        parent charged exactly the child's actual spend.

        Only call this once every descendant of `scope` is terminal; releasing
        the grant early would let a still-running child spend beyond the ceiling
        its parent was given.
        """
        with self._lock:
            node = self._require(scope)
            if node.parent is None:
                return
            # `open` reserved the whole grant against the parent; release exactly
            # that. The child's real spend was charged separately by `commit`, so
            # what remains on the parent is the spend and nothing else.
            parent = self._require(node.parent)
            parent.reserved_usd = max(0.0, parent.reserved_usd - node.max_usd)
            parent.reserved_tokens = max(0, parent.reserved_tokens - node.max_tokens)

    def get(self, scope: str) -> BudgetNode:
        with self._lock:
            return self._require(scope)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._nodes.items()}
