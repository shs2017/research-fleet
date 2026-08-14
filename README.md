<p align="center">
  <img src="logo.svg" alt="research-fleet" width="720">
</p>

# research-fleet

Run configurable, reproducible agent workflows in containers. Fleet schedules stages and preserves
agent sessions; research-ship owns the container environment.

## Install

```bash
./fleet install
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

## Workflows

Actors define who performs each stage and how they run. With `persistent: true`,
every later stage using an actor resumes the same provider session. This lets a
researcher pause while a judge reviews its work, then continue in its original
context.

Fleet releases a stage only after its dependencies finish. The actor receives
that stage's prompt at release time, so later instructions remain undisclosed
through the workflow interface until the workflow reaches them. Long prompts can
use `task_file`, resolved relative to the workflow YAML.

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
    task_file: prompts/01-research.md
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

- YAML defines actors, persistence, prompts, dependencies, models, effort, and
  resource requirements in one place.
- The scheduler runs ready stages, enforces budgets and resource limits, and
  checkpoints workflow progress.
- Research Ship provides the reproducible container environment, credentials,
  project workspace, and networking behavior.
- An append-only ledger records events, costs, results, and provider session IDs.
- `fleet trace` presents agent messages, tool activity, lifecycle events, and
  usage as a readable job history.
