# research-fleet

Run auditable agent workflows in containers. Fleet schedules stages and preserves
agent sessions; [research-ship](../research-ship) owns the container environment.

## Install

```bash
./fleet install
fleet doctor
```

Authenticate once per project:

```bash
fleet login --import
```

## Run

Run one agent:

```bash
fleet run "Investigate the failing experiment" --agents 1 --gpus 0
```

Run a YAML workflow:

```bash
fleet workflow discovery.yaml --plan
fleet workflow discovery.yaml
```

Inspect it:

```bash
fleet runs
fleet ls RUN_ID
fleet trace JOB_ID
fleet usage
```

## Workflow idea

An actor names an agent. With `persistent: true`, every later stage using that
actor resumes the same provider session. A stage receives only its own prompt,
after its dependencies finish.

```yaml
name: investigate
actors:
  researcher:
    backend: codex-cli
    persistent: true
  judge:
    backend: codex-cli
    persistent: true

stages:
  - name: research
    actor: researcher
    task: Investigate the question and save your evidence.
    gpus: 0
  - name: judge
    actor: judge
    needs: [research]
    task: Review the research output.
    gpus: 0
  - name: revise
    actor: researcher
    needs: [judge]
    task: Address the judge's feedback.
    gpus: 0
```

See [docs/workflows.md](docs/workflows.md) for the YAML reference and
[docs/development.md](docs/development.md) for architecture and testing.

## Design

- One local scheduler, one container executor (`ship`), and one validation-only
  executor (`dry-run`).
- One append-only ledger for events, costs, results, and resumable sessions.
- No MCP server, parameter sweeps, Ray, Slurm, or Fleet-level network policy.
- Networking and firewall behavior belong to research-ship.
