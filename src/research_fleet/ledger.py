"""Append-only, hash-chained audit ledger.

Every state change in the fleet lands here before it takes effect. Three
properties are the point:

  1. **Append-only.** Events are written to a JSONL file opened in `"a"` mode
     and fsynced. Nothing rewrites history.
  2. **Tamper-evident.** Each event carries `prev_hash` and `hash`, forming a
     chain from the genesis record. `verify()` walks the chain and reports the
     first index where it breaks, so a silently edited log is detectable.
  3. **Queryable.** A SQLite index mirrors the JSONL for `fleet ls` / `fleet
     show` without parsing megabytes of transcript. The JSONL is the source of
     truth; the index is disposable and can be rebuilt with `reindex()`.

Agent reasoning is captured as `agent.message` / `agent.tool_use` /
`agent.tool_result` events, so a completed run replays as a full decision trace,
not just stdout.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

GENESIS_HASH = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY,
    ts         REAL NOT NULL,
    run_id     TEXT,
    job_id     TEXT,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    hash       TEXT NOT NULL,
    prev_hash  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);

CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    run_id      TEXT,
    name        TEXT,
    kind        TEXT,
    state       TEXT,
    fingerprint TEXT,
    parent      TEXT,
    created_at  REAL,
    updated_at  REAL,
    spec        TEXT,
    result      TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);

-- One row per finished job, flattened for reporting. Derived from the events, so it is
-- rebuilt by reindex() and can be dropped without losing anything.
CREATE TABLE IF NOT EXISTS usage (
    job_id             TEXT PRIMARY KEY,
    run_id             TEXT,
    name               TEXT,
    stage              TEXT,
    attempt            INTEGER,
    workflow           TEXT,
    kind               TEXT,
    state              TEXT,
    model              TEXT,
    backend            TEXT,
    started_at         REAL,
    ended_at           REAL,
    duration_s         REAL,
    agent_seconds      REAL,
    gpus               REAL,
    gpu_seconds        REAL,
    requests           INTEGER,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cache_read_tokens  INTEGER,
    cache_write_tokens INTEGER,
    total_tokens       INTEGER,
    cost_usd           REAL,
    unpriced           INTEGER
);
CREATE INDEX IF NOT EXISTS idx_usage_run ON usage(run_id);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model);
CREATE INDEX IF NOT EXISTS idx_usage_started ON usage(started_at);
CREATE INDEX IF NOT EXISTS idx_usage_stage ON usage(run_id, stage, attempt);
"""


def _number(value: Any, cast=int, default=0):
    """Coerce a ledger value to a number, tolerating anything odd.

    Historical logs can hold surprises, including counts that an earlier redactor
    masked. Reindexing one of those must not crash on the whole file.
    """
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    type: str
    run_id: str | None
    job_id: str | None
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    @staticmethod
    def compute_hash(seq: int, ts: float, type_: str, run_id, job_id, payload, prev_hash: str) -> str:
        body = _canonical(
            {
                "seq": seq,
                "ts": ts,
                "type": type_,
                "run_id": run_id,
                "job_id": job_id,
                "payload": payload,
                "prev_hash": prev_hash,
            }
        )
        return hashlib.sha256(body.encode()).hexdigest()

    def to_json(self) -> str:
        # vars() rather than a repeated field list, so the record written to disk
        # can never drift from the dataclass. (Not asdict(): that deep-copies the
        # payload on every append, and appends are the hot path here.)
        return json.dumps(vars(self), separators=(",", ":"), default=str)


class Ledger:
    """One ledger per fleet root. Thread-safe; safe to share across the scheduler."""

    def __init__(self, root: str | Path, redactor: "Redactor | None" = None):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "ledger.jsonl"
        self.db_path = self.root / "index.db"
        self._lock = threading.RLock()
        self._redactor = redactor or Redactor()

        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

        self._seq, self._head = self._recover_head()
        self._size = self.jsonl_path.stat().st_size if self.jsonl_path.exists() else 0
        self._specs: dict[str, dict] = {}      # scratch for reindex()

    # ---------------------------------------------------------------- writing

    def _recover_head(self) -> tuple[int, str]:
        """Read the tail of the JSONL to resume the chain across restarts."""
        if not self.jsonl_path.exists() or self.jsonl_path.stat().st_size == 0:
            return -1, GENESIS_HASH
        last = None
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            return -1, GENESIS_HASH
        rec = json.loads(last)
        return int(rec["seq"]), str(rec["hash"])

    def append(
        self,
        type_: str,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        job_id: str | None = None,
    ) -> Event:
        payload = self._redactor.scrub(payload or {})
        with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                # A hash chain has one head, so appending has to be exclusive across
                # processes as well as threads. `fleet kill` writes to the same ledger
                # as the running scheduler, and without this they both believed they
                # held the head and collided on `seq`.
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    if os.fstat(fh.fileno()).st_size != self._size:
                        # Someone else appended since we last did; adopt their head.
                        self._seq, self._head = self._recover_head()

                    seq = self._seq + 1
                    ts = time.time()
                    h = Event.compute_hash(seq, ts, type_, run_id, job_id, payload, self._head)
                    ev = Event(seq, ts, type_, run_id, job_id, payload, self._head, h)

                    fh.write(ev.to_json() + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    self._size = os.fstat(fh.fileno()).st_size
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

            self._db.execute(
                "INSERT OR REPLACE INTO events"
                "(seq, ts, run_id, job_id, type, payload, hash, prev_hash)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (seq, ts, run_id, job_id, type_, _canonical(payload), h, ev.prev_hash),
            )
            self._db.commit()

            self._seq, self._head = seq, h
            return ev

    # -------------------------------------------------------- job projections

    def upsert_job(self, spec, state: str, result=None) -> None:
        """Maintain the queryable job table. The events remain the source of truth."""
        with self._lock:
            self._upsert_job_locked(spec, state, result)
            self._db.commit()

    def _upsert_job_locked(self, spec, state: str, result=None) -> None:
        self._db.execute(
            """INSERT INTO jobs(job_id, run_id, name, kind, state, fingerprint, parent,
                                created_at, updated_at, spec, result)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(job_id) DO UPDATE SET
                   state=excluded.state,
                   updated_at=excluded.updated_at,
                   result=excluded.result""",
            (
                spec.id,
                spec.run_id,
                spec.name,
                spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind),
                state,
                spec.fingerprint(),
                spec.parent_job_id,
                spec.created_at,
                time.time(),
                _canonical(spec.model_dump(mode="json")),
                _canonical(result.model_dump(mode="json")) if result is not None else None,
            ),
        )
        if result is not None:
            self._record_usage_locked(spec, state, result)

    def _record_usage_locked(self, spec, state: str, result) -> None:
        """Flatten one job's cost and compute into the usage table.

        Called with the lock held, from upsert_job. Everything here is derivable from
        the ledger, which is what lets reindex() rebuild it.
        """
        usage = result.usage if isinstance(result.usage, dict) else {}
        duration = None
        if result.started_at and result.ended_at:
            duration = max(0.0, result.ended_at - result.started_at)
        gpus = float(getattr(spec.resources, "gpus", 0.0) or 0.0)
        agent = getattr(spec, "agent", None)
        labels = dict(getattr(spec, "labels", {}) or {})

        self._db.execute(
            """INSERT OR REPLACE INTO usage(
                   job_id, run_id, name, stage, attempt, workflow, kind, state, model, backend,
                   started_at, ended_at, duration_s, agent_seconds, gpus, gpu_seconds, requests,
                   input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                   total_tokens, cost_usd, unpriced)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                spec.id,
                spec.run_id,
                spec.name,
                labels.get("stage") or spec.name,
                int(labels.get("attempt") or 1),
                labels.get("workflow"),
                spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind),
                state,
                usage.get("model") or (agent.model if agent else None),
                agent.backend if agent else None,
                result.started_at,
                result.ended_at,
                duration,
                getattr(result, "agent_seconds", None),
                gpus,
                (duration or 0.0) * gpus,
                _number(usage.get("requests")),
                _number(usage.get("input_tokens")),
                _number(usage.get("output_tokens")),
                _number(usage.get("cache_read_tokens")),
                _number(usage.get("cache_write_tokens")),
                _number(usage.get("total_tokens")),
                _number(usage.get("cost_usd"), float, 0.0),
                1 if usage.get("unpriced_model") else 0,
            ),
        )

    # ---------------------------------------------------------------- reading

    def read_all(self) -> Iterator[Event]:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                yield Event(
                    r["seq"], r["ts"], r["type"], r.get("run_id"), r.get("job_id"),
                    r["payload"], r["prev_hash"], r["hash"],
                )

    def events(self, *, run_id=None, job_id=None, types=None, limit=1000) -> list[Event]:
        q = "SELECT seq, ts, run_id, job_id, type, payload, hash, prev_hash FROM events WHERE 1=1"
        args: list[Any] = []
        if run_id:
            q += " AND run_id = ?"
            args.append(run_id)
        if job_id:
            q += " AND job_id = ?"
            args.append(job_id)
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args.extend(types)
        q += " ORDER BY seq ASC LIMIT ?"
        args.append(limit)
        rows = self._db.execute(q, args).fetchall()
        return [
            Event(r[0], r[1], r[4], r[2], r[3], json.loads(r[5]), r[7], r[6])
            for r in rows
        ]

    def jobs(self, run_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT job_id, run_id, name, kind, state, parent, created_at, updated_at, result FROM jobs"
        args: list[Any] = []
        if run_id:
            q += " WHERE run_id = ?"
            args.append(run_id)
        q += " ORDER BY created_at ASC"
        cols = ["job_id", "run_id", "name", "kind", "state", "parent", "created_at", "updated_at", "result"]
        return [dict(zip(cols, row)) for row in self._db.execute(q, args).fetchall()]

    def runs(self) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT run_id, COUNT(*), MIN(created_at), MAX(updated_at),"
            " SUM(state='succeeded'), SUM(state='failed')"
            " FROM jobs WHERE run_id IS NOT NULL GROUP BY run_id ORDER BY MIN(created_at) DESC"
        ).fetchall()
        return [
            {"run_id": r[0], "jobs": r[1], "started_at": r[2], "updated_at": r[3],
             "succeeded": r[4] or 0, "failed": r[5] or 0}
            for r in rows
        ]

    def observed_cost(self, model: str | None = None, kind: str = "agent") -> dict[str, Any]:
        """What work like this has actually cost, for estimating the next one.

        Returns the 75th percentile rather than the mean: a little conservative, so a
        reservation has headroom, without inheriting the tail of one runaway job.
        """
        clauses = ["kind = ?", "cost_usd > 0", "state = 'succeeded'"]
        args: list[Any] = [kind]
        if model:
            clauses.append("model = ?")
            args.append(model)
        rows = self._db.execute(
            f"SELECT cost_usd, input_tokens + cache_read_tokens, output_tokens"
            f" FROM usage WHERE {' AND '.join(clauses)} ORDER BY cost_usd",
            args,
        ).fetchall()
        if not rows:
            return {"samples": 0}

        def percentile(values, fraction):
            index = min(len(values) - 1, int(len(values) * fraction))
            return values[index]

        costs = [r[0] for r in rows]
        return {
            "samples": len(rows),
            "cost_usd": percentile(costs, 0.75),
            "median_cost_usd": percentile(costs, 0.5),
            "input_tokens": percentile(sorted(r[1] for r in rows), 0.5),
            "output_tokens": percentile(sorted(r[2] for r in rows), 0.5),
        }

    def mark_cancelled(self, run_id: str, reason: str) -> list[str]:
        """Record that a run was killed from outside the process that started it.

        The scheduler that owned those jobs is gone, so nothing else will ever write
        their terminal state. Without this they sit in the index as `running` forever.
        """
        rows = self._db.execute(
            "SELECT job_id, name FROM jobs WHERE run_id = ? AND state NOT IN"
            " ('succeeded','failed','cancelled','denied')",
            (run_id,),
        ).fetchall()
        killed = []
        for job_id, name in rows:
            self.append(
                "job.cancelled", {"name": name, "reason": reason},
                run_id=run_id, job_id=job_id,
            )
            with self._lock:
                self._db.execute(
                    "UPDATE jobs SET state = 'cancelled', updated_at = ? WHERE job_id = ?",
                    (time.time(), job_id),
                )
                self._db.execute(
                    "UPDATE usage SET state = 'cancelled' WHERE job_id = ?", (job_id,)
                )
                self._db.commit()
            killed.append(job_id)
        if killed:
            self.append("run.cancelled", {"reason": reason, "jobs": len(killed)}, run_id=run_id)
        return killed

    def active_runs(self) -> list[str]:
        """Runs with at least one job that never reached a terminal state."""
        rows = self._db.execute(
            "SELECT DISTINCT run_id FROM jobs WHERE state NOT IN"
            " ('succeeded','failed','cancelled','denied') ORDER BY updated_at DESC"
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    # ------------------------------------------------------------------ usage

    USAGE_TOTALS = (
        "COUNT(*) AS jobs",
        "COALESCE(SUM(cost_usd), 0) AS cost_usd",
        "COALESCE(SUM(total_tokens), 0) AS total_tokens",
        "COALESCE(SUM(input_tokens), 0) AS input_tokens",
        "COALESCE(SUM(output_tokens), 0) AS output_tokens",
        "COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens",
        "COALESCE(SUM(requests), 0) AS requests",
        "COALESCE(SUM(duration_s), 0) AS duration_s",
        "COALESCE(SUM(agent_seconds), 0) AS agent_seconds",
        "COALESCE(SUM(gpu_seconds), 0) AS gpu_seconds",
        "SUM(unpriced) AS unpriced_jobs",
    )

    # What `group_by` accepts, mapped to SQL. `day` buckets by local date.
    USAGE_GROUPS = {
        "run": "run_id",
        "model": "model",
        "name": "name",
        "kind": "kind",
        "backend": "backend",
        "state": "state",
        "stage": "stage",
        "attempt": "attempt",
        "workflow": "workflow",
        "job": "job_id",
        "day": "date(started_at, 'unixepoch', 'localtime')",
    }

    def _usage_where(self, run_id=None, since=None, kind=None) -> tuple[str, list[Any]]:
        clauses, args = ["1=1"], []
        if run_id:
            clauses.append("run_id = ?")
            args.append(run_id)
        if since is not None:
            clauses.append("started_at >= ?")
            args.append(float(since))
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        return " AND ".join(clauses), args

    def usage_rows(self, *, run_id=None, since=None, kind=None, limit=1000) -> list[dict[str, Any]]:
        """Per-job usage, newest first."""
        where, args = self._usage_where(run_id, since, kind)
        cur = self._db.execute(
            f"SELECT * FROM usage WHERE {where} ORDER BY started_at DESC LIMIT ?",
            [*args, limit],
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def usage_totals(self, *, run_id=None, since=None, kind=None) -> dict[str, Any]:
        """One row of totals across everything matching the filters."""
        where, args = self._usage_where(run_id, since, kind)
        cur = self._db.execute(
            f"SELECT {', '.join(self.USAGE_TOTALS)} FROM usage WHERE {where}", args
        )
        cols = [c[0] for c in cur.description]
        return dict(zip(cols, cur.fetchone()))

    def usage_by(self, group_by: str, *, run_id=None, since=None, kind=None) -> list[dict[str, Any]]:
        """Totals grouped by one or more keys, comma separated.

        `run,stage,attempt` is the finest useful grain: a node that repeats in a cycle
        shows up once per attempt rather than being summed into a single row.
        """
        keys = [k.strip() for k in group_by.split(",") if k.strip()]
        unknown = [k for k in keys if k not in self.USAGE_GROUPS]
        if not keys or unknown:
            raise ValueError(
                f"cannot group usage by {group_by!r}; try one or more of "
                f"{', '.join(sorted(self.USAGE_GROUPS))}"
            )
        select = ", ".join(f"{self.USAGE_GROUPS[k]} AS {k}" for k in keys)
        where, args = self._usage_where(run_id, since, kind)
        cur = self._db.execute(
            f"SELECT {select}, {', '.join(self.USAGE_TOTALS)} FROM usage WHERE {where} "
            f"GROUP BY {', '.join(keys)} ORDER BY cost_usd DESC, jobs DESC",
            args,
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------ integrity

    def verify(self) -> tuple[bool, str | None]:
        """Walk the chain. Returns (ok, message)."""
        prev = GENESIS_HASH
        expected_seq = 0
        count = 0
        for ev in self.read_all():
            if ev.seq != expected_seq:
                return False, f"seq gap at {ev.seq} (expected {expected_seq}): records inserted or removed"
            if ev.prev_hash != prev:
                return False, f"prev_hash mismatch at seq={ev.seq}: history was rewritten"
            recomputed = Event.compute_hash(
                ev.seq, ev.ts, ev.type, ev.run_id, ev.job_id, ev.payload, ev.prev_hash
            )
            if recomputed != ev.hash:
                return False, f"payload edited at seq={ev.seq}: hash does not match content"
            prev = ev.hash
            expected_seq += 1
            count += 1
        return True, f"{count} events verified"

    def reindex(self) -> int:
        """Rebuild every derived table from the JSONL source of truth.

        Events, jobs and usage are all reconstructed, so a lost or stale index costs
        nothing: the append-only log has everything needed to recreate it.
        """
        from .spec import JobResult, JobSpec

        with self._lock:
            for table in ("events", "jobs", "usage"):
                self._db.execute(f"DELETE FROM {table}")

            n = 0
            for ev in self.read_all():
                self._db.execute(
                    "INSERT INTO events(seq, ts, run_id, job_id, type, payload, hash, prev_hash)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (ev.seq, ev.ts, ev.run_id, ev.job_id, ev.type,
                     _canonical(ev.payload), ev.hash, ev.prev_hash),
                )
                n += 1

                # job.submitted carries the spec; the terminal events carry the result.
                if ev.type == "job.submitted":
                    self._specs[ev.payload["spec"]["id"]] = ev.payload["spec"]
                elif ev.type.startswith("job.") and "result" in ev.payload:
                    raw = self._specs.get(ev.job_id)
                    if raw is None:
                        continue
                    state = ev.type.split(".", 1)[1]
                    self._upsert_job_locked(JobSpec(**raw), state, JobResult(**ev.payload["result"]))

            self._db.commit()
            self._specs.clear()
            return n

    def close(self) -> None:
        with self._lock:
            self._db.close()


class Redactor:
    """Keeps secrets out of the permanent record.

    The ledger is append-only by design, so a leaked key is unrecoverable, so
    scrubbing has to happen on the write path, before the value is ever
    persisted. Matches on key name for structured payloads and on value shape
    for free text (API keys embedded in a command line or an agent transcript).
    """

    DEFAULT_KEY_PATTERNS = (
        "api_key", "apikey", "token", "secret", "password", "passwd",
        "credential", "auth", "private_key", "session_key",
    )

    # Value-shaped matches for text that flows through logs and transcripts.
    DEFAULT_VALUE_PATTERNS = (
        r"sk-ant-[A-Za-z0-9_\-]{16,}",
        r"sk-[A-Za-z0-9]{32,}",
        r"gh[pousr]_[A-Za-z0-9]{16,}",
        r"AKIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z_\-]{35}",
        r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
    )

    MASK = "[REDACTED]"

    def __init__(self, key_patterns=None, value_patterns=None):
        import re

        self._keys = tuple(k.lower() for k in (key_patterns or self.DEFAULT_KEY_PATTERNS))
        self._values = [re.compile(p) for p in (value_patterns or self.DEFAULT_VALUE_PATTERNS)]

    def scrub(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # Key matching applies only to string values. A credential is always a
                # string, while `input_tokens` and friends are counts that happen to
                # contain a sensitive-looking word; masking those silently destroyed
                # the cost accounting in the log.
                if isinstance(v, str) and any(pat in str(k).lower() for pat in self._keys):
                    out[k] = self.MASK
                else:
                    out[k] = self.scrub(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [self.scrub(v) for v in obj]
        if isinstance(obj, str):
            return self.scrub_text(obj)
        return obj

    def scrub_text(self, text: str) -> str:
        for pat in self._values:
            text = pat.sub(self.MASK, text)
        return text
