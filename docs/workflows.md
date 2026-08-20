# Workflows

A workflow is one YAML file with `actors` and ordered `stages`.

## Actors

An actor may set `backend`, `model`, `effort`, `system_prompt`, and `persistent`.
A persistent actor resumes the same Claude or Codex session each time it appears.
Fleet fails the workflow if the provider does not return a session ID; it never
silently starts a fresh agent.

## Stages

Stages use `task` for a short agent prompt, `task_file` for a prompt stored beside
the workflow, or `command` for a normal process. A stage must set either `task` or
`task_file`, not both. Relative task-file paths are resolved from the workflow YAML.
`needs` controls dependencies. Independent stages can run concurrently, except two
stages cannot use the same persistent actor concurrently.

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
    task_file: prompts/01-first.md
    gpus: 0
  - name: second
    actor: researcher
    needs: [first]
    task: Now do the second task.
    gpus: 0
```

Stages may also contain a bounded `loop` with `until.output_contains`. Use
`fleet workflow FILE --plan` to validate and display the graph without running it.

## Files, iterations, and snapshots

A stage sees result directories only from its transitive `needs` dependencies under
`/inputs`. A repeated stage can therefore see an earlier iteration through its
rendered output and persistent actor conversation, while an isolated chain also
inherits tracked `/workspace` changes. Iteration result directories remain separate
(`review`, `review-2`, and so on). No stage can see a different run unless that run is
explicitly selected with `--from-run` or `--resume`.

With `isolate: true`, every finished job records `snapshot.json` and a binary-capable
`snapshot.patch` in its result directory. Fleet commits the stage's tracked workspace
state and retains it under the Git ref named in `snapshot.json`, so the state remains
inspectable after the temporary worktree is removed. `result.json` records the base
commit, snapshot commit, and ref.

Results live under `ROOT/results/<workflow>/<attempt>/`; events and usage live in the
ledger at `ROOT`.

The default `ship` executor runs stages in containers. The same workflow can run on
the host with `fleet workflow workflow.yaml --executor nono`; Fleet rewrites its
standard `/workspace`, `/results`, `/inputs`, and `/previous` paths to host paths.
With `isolate: true`, host mode also chains and snapshots Git worktrees.

Fleet checkpoints after every cycle stage. If a stage fails or is denied, later
cycle stages are not submitted. Re-run the unchanged workflow with
`--resume RUN_ID` to retry that stage and continue from there; completed stages are
not repeated.
