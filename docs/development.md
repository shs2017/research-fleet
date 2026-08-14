# Development

The code path is deliberately short:

1. `workflow.py` validates YAML and releases stages when dependencies finish.
2. `fleet.py` turns stages into job specifications.
3. `scheduler.py` enforces budgets and policy, then records every transition.
4. `executors/ship_exec.py` asks research-ship for the Docker invocation.
5. `backends/` translates provider output into messages, usage, and session IDs.
6. `ledger.py` stores the append-only audit log and query index.

Run tests with:

```bash
.venv/bin/pytest -q
```

After changing the package, reinstall the command used outside this checkout:

```bash
./fleet install
```

Networking, images, credentials, and container caches are research-ship concerns.
Fleet owns workflow order, actor persistence, resource placement, budgets, results,
and audit history.
