# Filesystem and tool boundaries

Fleet has three separate controls. Put them in the workflow/configuration file,
not in an agent prompt.

```yaml
executor:
  kind: nono                 # or ship for Docker
  mounts:
    - source: ./data
      target: /workspace/data
      mode: ro
policy:
  allowed_mount_roots: [./data, ./fleet-logs]
  deny_mount_paths: [/etc, /root, /var/run/docker.sock]
  deny_command_patterns: ['\\brm\\s+-rf\\s+/(?:\\s|$)']
  agent_default_disallowed_tools: [WebFetch]

stages:
  - name: inspect
    kind: agent
    allowed_tools: [Read, Bash]
    disallowed_tools: [Write]
```

`allowed_mount_roots` is the host-side allowlist. A mount source must resolve
under one of these roots; denied paths always win. `mode: ro` exposes a source
read-only. `mode: rw` permits changes and is normally used only for the stage's
own `/results` directory. Dependency directories under `/inputs` and prior
iteration directories under `/previous` are created read-only by Fleet.

The `allowed_tools` and `disallowed_tools` fields are passed to the selected
agent CLI (for example Claude's `--allowedTools` and `--disallowedTools`). The
policy denylist is a separate preflight check for command text. It rejects a job
before execution; it is not a shell command filter.

With `executor.kind: nono`, Fleet adds a kernel-enforced Landlock boundary. Only
the worktree, declared mounts, and the agent's own credential/session directory
are granted. Fleet uses a bundled profile that does not grant broad `/tmp`
writes, so a read-only mount remains read-only even when its source is on a
scratch filesystem. A writable directory can still be changed or deleted by the
agent; no sandbox can make a declared writable directory immutable. The source
directory of a read-only mount, undeclared host paths, prompts, and other runs
cannot be read, written, or deleted.

Fleet explicitly grants only the standard device files needed by normal command
execution (`/dev/null`, `/dev/zero`, `/dev/random`, `/dev/urandom`, and `/dev/tty`).
It does not grant general `/dev` access.

With `executor.kind: ship`, Docker provides the process boundary and the same
mount modes are passed as Docker bind mounts. With `executor.kind: direct`, Fleet
uses the Codex sandbox when available; `nono` is the recommended host executor.

These controls restrict filesystem and provider tools. They do not make an agent
safe to give arbitrary network credentials: configure network policy in the
container/runtime and pass only the credentials the workflow needs.
