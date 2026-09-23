"""SQLite persistence: the event log, snapshots, per-employee memory, and every LLM call.

One file per world (saves/<name>/world.db), WAL mode, single writer (the engine).
"""
from __future__ import annotations

import json
import sqlite3
import time
import zlib
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY, sim_ms INTEGER, wall REAL, type TEXT, actor TEXT, room TEXT,
    targets TEXT, payload TEXT, source TEXT, decision_id TEXT, visibility TEXT
);
CREATE INDEX IF NOT EXISTS events_type ON events(type);
CREATE INDEX IF NOT EXISTS events_sim ON events(sim_ms);
CREATE TABLE IF NOT EXISTS witnesses (seq INTEGER, emp_id TEXT, mode TEXT);
CREATE INDEX IF NOT EXISTS witnesses_emp ON witnesses(emp_id, seq);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY, seq INTEGER, sim_ms INTEGER, kind TEXT, engine_version TEXT, state BLOB
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY, emp_id TEXT, kind TEXT, tier TEXT, sim_ms INTEGER, day INTEGER, text TEXT,
    importance REAL, strength REAL, last_access_ms INTEGER, access_count INTEGER,
    people TEXT, provenance TEXT, event_seq INTEGER
);
CREATE INDEX IF NOT EXISTS memories_emp ON memories(emp_id, tier, sim_ms);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY, emp_id TEXT, about TEXT, statement TEXT, confidence REAL, source TEXT,
    adopted_ms INTEGER, abandoned_ms INTEGER, superseded_by INTEGER, evidence TEXT
);
CREATE INDEX IF NOT EXISTS beliefs_emp ON beliefs(emp_id, about);
CREATE TABLE IF NOT EXISTS relationships (
    emp_id TEXT, other_id TEXT, familiarity REAL, warmth REAL, trust REAL, notes TEXT, updated_ms INTEGER,
    PRIMARY KEY (emp_id, other_id)
);
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY, emp_id TEXT, version INTEGER, valid_from_ms INTEGER, valid_to_ms INTEGER,
    name TEXT, pronouns TEXT, look TEXT, self_concept TEXT, change_reason TEXT
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY, emp_id TEXT, text TEXT, status TEXT, created_ms INTEGER, updated_ms INTEGER
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY, key TEXT, task TEXT, emp_id TEXT, sim_ms INTEGER, wall REAL, model TEXT,
    provider TEXT, request TEXT, response TEXT, content TEXT, cost REAL, prompt_tokens INTEGER,
    completion_tokens INTEGER, latency_ms INTEGER, ok INTEGER, error TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_key ON llm_calls(key);
"""


def dumps(v) -> str:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


class Store:
    def __init__(self, path: str | Path | None):
        if path is None or str(path) == ":memory:":
            self.path = ":memory:"
        else:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --- meta ---------------------------------------------------------------------------
    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, dumps(value)))
        self.db.commit()

    # --- events -------------------------------------------------------------------------
    def append_event(self, ev) -> int:
        cur = self.db.execute(
            "INSERT INTO events(sim_ms, wall, type, actor, room, targets, payload, source, decision_id, visibility)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ev.sim_ms, time.time(), ev.type, ev.actor, ev.room, dumps(ev.targets), dumps(ev.payload),
             ev.source, ev.decision_id, ev.visibility))
        seq = cur.lastrowid
        if ev.witnesses:
            self.db.executemany("INSERT INTO witnesses(seq, emp_id, mode) VALUES (?,?,?)",
                                [(seq, e, m) for e, m in ev.witnesses.items()])
        return seq

    def commit(self) -> None:
        self.db.commit()

    def events(self, after: int = 0, limit: int = 500, types: list[str] | None = None) -> list[dict]:
        q = "SELECT * FROM events WHERE seq > ?"
        args: list = [after]
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args += types
        q += " ORDER BY seq LIMIT ?"
        args.append(limit)
        rows = self.db.execute(q, args).fetchall()
        return [self._event_row(r) for r in rows]

    def events_between(self, start_ms: int, end_ms: int) -> list[dict]:
        rows = self.db.execute("SELECT * FROM events WHERE sim_ms >= ? AND sim_ms < ? ORDER BY seq",
                               (start_ms, end_ms)).fetchall()
        return [self._event_row(r) for r in rows]

    def _event_row(self, r) -> dict:
        wit = {w["emp_id"]: w["mode"] for w in
               self.db.execute("SELECT emp_id, mode FROM witnesses WHERE seq=?", (r["seq"],))}
        return {"seq": r["seq"], "sim_ms": r["sim_ms"], "type": r["type"], "actor": r["actor"], "room": r["room"],
                "targets": json.loads(r["targets"] or "[]"), "payload": json.loads(r["payload"] or "{}"),
                "source": r["source"], "decision_id": r["decision_id"], "witnesses": wit}

    def last_seq(self) -> int:
        row = self.db.execute("SELECT MAX(seq) AS s FROM events").fetchone()
        return row["s"] or 0

    # --- snapshots ----------------------------------------------------------------------
    def save_snapshot(self, seq: int, sim_ms: int, kind: str, state: dict, engine_version: str) -> None:
        blob = zlib.compress(dumps(state).encode())
        self.db.execute("INSERT INTO snapshots(seq, sim_ms, kind, engine_version, state) VALUES (?,?,?,?,?)",
                        (seq, sim_ms, kind, engine_version, blob))
        self.db.commit()

    def latest_snapshot(self) -> dict | None:
        row = self.db.execute("SELECT * FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"seq": row["seq"], "sim_ms": row["sim_ms"], "kind": row["kind"],
                "state": json.loads(zlib.decompress(row["state"]))}

    def list_snapshots(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT id, seq, sim_ms, kind FROM snapshots ORDER BY id")]

    # --- llm calls ----------------------------------------------------------------------
    def record_llm_call(self, **row) -> None:
        cols = ", ".join(row)
        self.db.execute(f"INSERT INTO llm_calls({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        self.db.commit()

    def find_llm_call(self, key: str, replicate: int = 0) -> dict | None:
        rows = self.db.execute("SELECT * FROM llm_calls WHERE key=? AND ok=1 ORDER BY id", (key,)).fetchall()
        if len(rows) > replicate:
            return dict(rows[replicate])
        return None

    def llm_spend(self) -> float:
        row = self.db.execute("SELECT COALESCE(SUM(cost), 0) AS c FROM llm_calls").fetchone()
        return float(row["c"])

    def recent_llm_calls(self, limit: int = 50, emp_id: str | None = None) -> list[dict]:
        q = "SELECT id, task, emp_id, sim_ms, model, provider, cost, prompt_tokens, completion_tokens, latency_ms," \
            " ok, error, content, request FROM llm_calls"
        args: list = []
        if emp_id:
            q += " WHERE emp_id=?"
            args.append(emp_id)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.execute(q, args)]

    def close(self) -> None:
        self.db.commit()
        self.db.close()
