# Workflows

A workflow is one YAML file with `actors` and ordered `stages`.

## Actors

An actor may set `backend`, `model`, `effort`, `system_prompt`, and `persistent`.
A persistent actor resumes the same Claude or Codex session each time it appears.
Fleet fails the workflow if the provider does not return a session ID; it never
silently starts a fresh agent.

## Stages

Stages use `task` for an agent or `command` for a normal process. `needs` controls
dependencies. Independent stages can run concurrently, except two stages cannot
use the same persistent actor concurrently.

Each stage is rendered only when it becomes runnable. Templates can reference
earlier output with `{{ steps.NAME.output }}`. This is also the key information
boundary: a later prompt is not sent to an actor until its dependencies finish.

Useful fields are:

```yaml
name: workflow-name
model: claude-sonnet-5       # optional workflow default
effort: high                 # optional workflow default
isolate: false
actors:
  researcher:
    backend: codex-cli
    persistent: true
stages:
  - name: first
    actor: researcher
    task: Do the first task.
    gpus: 0
  - name: second
    actor: researcher
    needs: [first]
    task: Now do the second task.
    gpus: 0
```

Stages may also contain a bounded `loop` with `until.output_contains`. Use
`fleet workflow FILE --plan` to validate and display the graph without running it.

Results live under `ROOT/results/RUN_ID/`; events and usage live in the ledger at
`ROOT`.
