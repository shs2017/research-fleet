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
"""


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
            seq = self._seq + 1
            ts = time.time()
            h = Event.compute_hash(seq, ts, type_, run_id, job_id, payload, self._head)
            ev = Event(seq, ts, type_, run_id, job_id, payload, self._head, h)

            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(ev.to_json() + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            self._db.execute(
                "INSERT INTO events(seq, ts, run_id, job_id, type, payload, hash, prev_hash)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (seq, ts, run_id, job_id, type_, _canonical(payload), h, self._head),
            )
            self._db.commit()

            self._seq, self._head = seq, h
            return ev

    # -------------------------------------------------------- job projections

    def upsert_job(self, spec, state: str, result=None) -> None:
        """Maintain the queryable job table. The events remain the source of truth."""
        with self._lock:
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
            self._db.commit()

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

    # ------------------------------------------------------------ integrity

    def verify(self) -> tuple[bool, str | None]:
        """Walk the chain. Returns (ok, message)."""
        prev = GENESIS_HASH
        expected_seq = 0
        count = 0
        for ev in self.read_all():
            if ev.seq != expected_seq:
                return False, f"seq gap at {ev.seq} (expected {expected_seq}) — records inserted or removed"
            if ev.prev_hash != prev:
                return False, f"prev_hash mismatch at seq={ev.seq} — history was rewritten"
            recomputed = Event.compute_hash(
                ev.seq, ev.ts, ev.type, ev.run_id, ev.job_id, ev.payload, ev.prev_hash
            )
            if recomputed != ev.hash:
                return False, f"payload edited at seq={ev.seq} — hash does not match content"
            prev = ev.hash
            expected_seq += 1
            count += 1
        return True, f"{count} events verified"

    def reindex(self) -> int:
        """Rebuild the SQLite index from the JSONL source of truth."""
        with self._lock:
            self._db.execute("DELETE FROM events")
            n = 0
            for ev in self.read_all():
                self._db.execute(
                    "INSERT INTO events(seq, ts, run_id, job_id, type, payload, hash, prev_hash)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (ev.seq, ev.ts, ev.run_id, ev.job_id, ev.type,
                     _canonical(ev.payload), ev.hash, ev.prev_hash),
                )
                n += 1
            self._db.commit()
            return n

    def close(self) -> None:
        with self._lock:
            self._db.close()


class Redactor:
    """Keeps secrets out of the permanent record.

    The ledger is append-only by design, so a leaked key is unrecoverable —
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
                if any(pat in str(k).lower() for pat in self._keys):
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
