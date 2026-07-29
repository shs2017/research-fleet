<p align="center">
  <img src="logo.svg" width="620" alt="research-fleet — auditable, budgeted agent swarms across your GPUs">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-06B6D4?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/runtime-Docker%20%2B%20NVIDIA-8B5CF6?style=flat-square" alt="Docker + NVIDIA">
  <img src="https://img.shields.io/badge/built%20on-research--ship-64748B?style=flat-square" alt="built on research-ship">
</p>

<p align="center">
  Run fleets of coding agents across your GPUs — each in its own container,<br>
  under a budget it cannot exceed and an audit trail it cannot edit.
</p>

---

## What it is

One scheduler runs two kinds of job:

- **`command`** — a training run, an eval, a sweep point. No LLM, no token cost.
- **`agent`** — an LLM coding agent with a repo and a GPU.

Both get the same container, the same policy checks, the same budget accounting, and
the same tamper-evident ledger. Because an agent is just another job, **an agent can
submit jobs of its own** — and its sub-agents are bounded by the budget and policy of
the parent that spawned them.

Containers come from **[research-ship](https://github.com/shs2017/research-ship)**, which
owns what a GPU container *is* and is useful on its own for interactive work. This
project owns what a *fleet* is.

## Install

```bash
# 1. research-ship provides the container
git clone https://github.com/shs2017/research-ship ~/Code/research-ship
ln -s ~/Code/research-ship/ship ~/.local/bin/ship

# 2. research-fleet orchestrates it
git clone https://github.com/shs2017/research-fleet ~/Code/research-fleet
cd ~/Code/research-fleet && uv venv && uv pip install -e ".[ray,mcp]"

# 3. build the image for the project you want to research in
cd ~/my-project && ship build
```

Requires Docker with the NVIDIA container runtime:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```

## Quick start

```bash
fleet cost                       # what does this cost, before I run it?

fleet run "Compare RMSNorm vs LayerNorm on a 20M-param model" --agents 4 --max-usd 20
fleet run "..." --agents 4 --gpus 0.25          # pack four agents onto one card
fleet run "..." --agents 16 --executor ray      # multi-node

fleet sweep python train.py --lr {lr} -g lr=1e-3,3e-4    # no LLM involved
```

Then inspect what happened:

```bash
fleet runs                # every run
fleet ls <run_id>         # jobs in a run, including agent-spawned children
fleet trace <job_id>      # full reasoning + tool trace for one agent
fleet pending             # jobs parked waiting for approval
fleet audit verify        # prove the log has not been edited
```

### Three entry points, one implementation

```python
from research_fleet import Fleet

with Fleet(max_usd=20) as fleet:
    fleet.run_agents("Try 3 attention variants", n=4)
    print(fleet.wait().summary())
```

```bash
claude mcp add research-fleet -- fleet mcp    # drive it from Claude Code or Codex
```

## How it works

```
                        ┌──────────────┐
   CLI / Python / MCP ──▶│  Scheduler   │──▶ Policy ──▶ Budget ──▶ Ledger
                        └──────┬───────┘     (deny)     (reserve)   (append)
                               │
                     ┌─────────▼─────────┐
                     │     Executor      │   ship │ ray │ dry-run
                     └─────────┬─────────┘
                               │  `ship docker-args` + the fleet's limits
                   ┌───────────┴───────────┐
                   ▼                       ▼
         research-ship container    research-ship container
              (GPU 0)                   (GPU 1)
          agent: claude -p          python train.py
                   │
                   └── writes JSON to $FLEET_SUBMIT_DIR ──▶ back to Scheduler
```

| Module | Responsibility |
| --- | --- |
| `spec.py` | `JobSpec` / `JobResult` — the data model, with content fingerprints |
| `scheduler.py` | Dependency graph, fractional GPU slots, spool intake, state machine |
| `policy.py` | Every check that can say "no", as a pure function |
| `budget.py` | Price table, cost estimation, hierarchical reserve/commit tree |
| `ledger.py` | Hash-chained JSONL + SQLite index + secret redaction |
| `executors/` | `ship`, `ray`, `dry-run` — where a container is placed |
| `backends/` | `claude-cli`, `codex-cli` — how an agent is driven and parsed |

### Division of labour

| Concern | Owner |
| --- | --- |
| Base image, CUDA, Python, torch, agent CLIs | research-ship — `.ship.conf` |
| `/workspace` mount, model cache, venv volume | research-ship |
| Non-root user, egress firewall allowlist | research-ship |
| Which GPU, how much CPU/RAM, timeout | research-fleet — `fleet.yaml` |
| Recursion depth, fanout, budget, approval gates | research-fleet |
| Audit ledger, reasoning traces, cost accounting | research-fleet |

> [!NOTE]
> The fleet deliberately does **not** re-impose `--read-only`, `--user`, or
> `--cap-drop ALL`. Those would break research-ship's writable venv, its non-root
> user, and the `NET_ADMIN` its firewall needs. Isolation is research-ship's contract;
> the fleet adds only the limits that are a *scheduling* concern.

## Auditability

Every state change is appended to `<root>/ledger.jsonl` **before it takes effect**.
Each record carries `prev_hash` and `hash`, chaining back to genesis.

```console
$ fleet audit verify
✓ audit chain intact — 1,284 events verified
```

Editing a payload, deleting a record, or reordering the file all break the chain and
are reported with the exact sequence number where it broke. The JSONL is the source
of truth; the SQLite index is disposable (`fleet audit reindex`).

**Reasoning is traced, not just stdout.** Agent output is parsed *as it streams*, so
a job that crashes halfway still leaves a complete record up to the failure:

```console
$ fleet trace job_a1b2c3d4
14:22:01 agent: The baseline diverges above lr=1e-3, so I'll bisect downward.
14:22:03 tool  Bash            {"command": "python train.py --lr 5e-4"}
14:22:47 result
14:22:47 budget $0.0412 (18,204 tokens)
```

Secrets are scrubbed on the **write** path, by key name and by value shape
(`sk-ant-…`, `ghp_…`, `AKIA…`, JWTs). An append-only log cannot be cleaned up after
the fact, so redaction has to happen before the value is ever persisted. Secrets
reach containers as bare `--env KEY` names, so they never appear in a recorded argv.

## Token and cost awareness

Costs are computed from reported `usage` against a price table, not guessed. Before a
job runs, `quote()` estimates it from the model, the reasoning `effort`, and the
*shape* of the work — an agent loop re-reads its growing context every turn, a single
call does not.

```console
$ fleet cost
        Estimated cost per agent standard
┌──────────────────┬────────┬───────────┬─────────────┐
│ model            │ effort │ est. cost │ est. tokens │
├──────────────────┼────────┼───────────┼─────────────┤
│ claude-haiku-4-5 │ low    │ $0.309    │     515,000 │
│ claude-sonnet-5  │ high   │ $1.903    │     580,000 │
│ claude-opus-5    │ high   │ $3.172    │     580,000 │
└──────────────────┴────────┴───────────┴─────────────┘
```

**Budgets are hierarchical.** A run opens a root scope; each agent gets a child scope
carved from its parent's *remaining* balance; spend rolls up to every ancestor. A
sub-agent cannot spend money its parent does not have, so a runaway delegation tree is
bounded by construction rather than by a global limit it might win a race against. A
scope is released only once every descendant is terminal.

Reservation matters as much as accounting: without it, twenty agents each individually
fit the budget while collectively blowing it. A job reserves its estimate at submit
time and settles against real usage at exit.

## Agents launching agents

An agent receives a budget briefing in its prompt — remaining balance, the price of
each model it may delegate to, and its current depth:

> ## Budget
> You have **$14.20** and **8,400,000 tokens** remaining for this task *including*
> any sub-agents you launch.
>
> | model | effort | est. cost |
> |---|---|---|
> | claude-haiku-4-5 | low | $0.309 |
> | claude-sonnet-5 | high | $1.903 |
> | claude-opus-5 | high | $3.172 |
>
> Delegate broad, parallel, or low-stakes work to a cheaper model at lower effort;
> reserve the expensive tier for the reasoning that actually needs it.

To launch work, the agent writes a JSON spec into `$FLEET_SUBMIT_DIR`:

```bash
cat > "$FLEET_SUBMIT_DIR/sweep-lr.json" <<'EOF'
{ "kind": "command",
  "name": "sweep-lr",
  "command": ["python", "train.py", "--lr", "3e-4"],
  "resources": {"gpus": 1} }
EOF
```

**A spool directory, not a socket.** The scheduler picks the file up and runs it
through the same policy and budget path as any other job. This keeps
`network.mode: none` viable for agent containers, gives every child a durable
provenance record naming its parent, and means a compromised agent gets a queue slot
rather than a control channel. An agent cannot choose its own image, so it cannot
escape the ship the operator configured. Rejections come back as
`<name>.rejected.json` with a readable reason.

## Safeguards

Policy runs on the submit path and is a pure function of `(spec, context) → Decision`,
so `fleet policy` shows exactly what is enforced.

| Layer | Default |
| --- | --- |
| Recursion | `max_agent_depth: 2`, `max_children_per_agent: 8` |
| Budget | `$50` per run, `$25` per job; hierarchical, reserved up front |
| Resources | Per-job GPU / memory / timeout ceilings; timeouts clamped, not denied |
| Filesystem | Mounts must resolve inside an allowlist; `/`, `/etc`, `~/.ssh`, `~/.aws`, the Docker socket are refused |
| Container | Non-root user and cache isolation from research-ship; `--pids-limit` from the fleet |
| Network | `limited` by default — research-ship's default-deny egress allowlist |
| Commands | Denylist for destructive patterns (`rm -rf /`, nested `docker run`, fork bombs) |
| Approval | Unrestricted network or a writable mount outside the workspace parks the job |

`fail_closed: true` means a check that cannot be evaluated denies rather than allows.
`wait()` returns rather than hanging when the only remaining work needs a human, and
`fleet pending` shows what is blocked and why.

Approval is **in-process**: a parked job belongs to a live scheduler, so grant it with
`fleet.approve(job_id)` from the session that submitted it (or via MCP). A job whose
scheduler has exited cannot be resumed, only resubmitted.

> [!WARNING]
> **Known limits.** The egress allowlist resolves domains to IPs at container start,
> so CDN rotation can break a long run — and it restricts *where* traffic goes, not
> *what* it carries. The price table is a snapshot (`budget.PRICES_AS_OF`),
> overridable via `budget.model_costs`. `/workspace` is a read-write bind mount to
> your real project directory, so an agent can corrupt your working tree — keep it
> under version control.

## Configuration

Precedence, lowest to highest:

```
defaults < ~/.config/research-fleet/config.yaml < ./fleet.yaml < FLEET_* env < CLI flags
```

Env vars use `__` for nesting: `FLEET_EXECUTOR__KIND=ray`. See
[`fleet.yaml`](fleet.yaml) for the fully commented surface.

Container configuration is **not** here — it lives in the project's `.ship.conf`
and belongs to research-ship.

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
```

The `dry-run` executor validates specs, exercises policy, and writes a complete ledger
without starting containers, and the ship-executor tests use a fake `ship`
script — so the whole suite runs on a machine with no GPU and no Docker.
