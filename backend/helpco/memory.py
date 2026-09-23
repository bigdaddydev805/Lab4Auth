"""Per-employee memory: experiences, beliefs (with provenance), relationships, identity history, goals.

- Experiences are first-person and subjective: each employee writes their own record of what they
  perceived, so two employees can remember the same event differently.
- Beliefs are never deleted. When a belief changes, the old one is closed (abandoned_ms) and linked
  to its replacement, so old identity becomes history instead of disappearing.
- Retrieval scores recency, importance and relevance (Park et al., 2023). Memories that aren't
  recalled fade (Ebbinghaus-style); faded ("dormant") memories can't be recalled by the employee but
  stay in the database for research.
"""
from __future__ import annotations

import json
import math
import re

from .clock import DAY, HOUR
from .store import Store, dumps

STOPWORDS = set("""a an the and or but if then of to in on at for with from by is are was were be been am i you he she
they them it its this that these those my your his her their our we us me do did does not no yes so just
what who when where why how about into out up down over again very can could would should will shall""".split())


def _fts_query(text: str) -> str | None:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", (text or "").lower()) if w not in STOPWORDS]
    words = list(dict.fromkeys(words))[:24]
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words)


class Memory:
    def __init__(self, store: Store, cfg):
        self.s = store
        self.db = store.db
        self.w_rec = float(cfg.get("memory.w_recency", 0.5))
        self.w_rel = float(cfg.get("memory.w_relevance", 3.0))
        self.w_imp = float(cfg.get("memory.w_importance", 2.0))
        self.decay = float(cfg.get("memory.recency_decay_per_hour", 0.995))
        self.k = int(cfg.get("memory.max_memories_in_prompt", 8))

    # --- experiences ----------------------------------------------------------------------
    def add(self, emp_id: str, sim_ms: int, day: int, text: str, importance: float = 3.0, kind: str = "experience",
            tier: str = "raw", people: list | None = None, provenance: dict | None = None,
            event_seq: int | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO memories(emp_id, kind, tier, sim_ms, day, text, importance, strength, last_access_ms,"
            " access_count, people, provenance, event_seq) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (emp_id, kind, tier, sim_ms, day, text, float(importance), 1.0, sim_ms, 0, dumps(people or []),
             dumps(provenance or {}), event_seq))
        return cur.lastrowid

    def day_experiences(self, emp_id: str, day: int) -> list[dict]:
        rows = self.db.execute("SELECT * FROM memories WHERE emp_id=? AND day=? AND tier='raw' ORDER BY sim_ms, id",
                               (emp_id, day)).fetchall()
        return [self._row(r) for r in rows]

    def recent(self, emp_id: str, since_ms: int, limit: int = 14) -> list[dict]:
        rows = self.db.execute("SELECT * FROM memories WHERE emp_id=? AND kind='experience' AND sim_ms>=? "
                               "ORDER BY sim_ms DESC, id DESC LIMIT ?", (emp_id, since_ms, limit)).fetchall()
        return [self._row(r) for r in rows][::-1]

    def all_for(self, emp_id: str, tiers=("episode", "raw", "dormant"), limit: int = 400) -> list[dict]:
        q = f"SELECT * FROM memories WHERE emp_id=? AND tier IN ({','.join('?' * len(tiers))}) " \
            "ORDER BY sim_ms DESC, id DESC LIMIT ?"
        return [self._row(r) for r in self.db.execute(q, (emp_id, *tiers, limit))]

    def _row(self, r) -> dict:
        d = dict(r)
        d["people"] = json.loads(d["people"] or "[]")
        d["provenance"] = json.loads(d["provenance"] or "{}")
        return d

    def retrieve(self, emp_id: str, query: str, now_ms: int, people: list[str] | None = None,
                 exclude_day: int | None = None, k: int | None = None, touch: bool = True) -> list[dict]:
        """Recall long-term memories relevant to the current situation."""
        k = k or self.k
        cands: dict[int, dict] = {}
        rel: dict[int, float] = {}
        def where(prefix: str = "") -> str:
            clause = f"{prefix}emp_id=? AND {prefix}tier='episode'"
            if exclude_day is not None:
                clause += f" AND NOT ({prefix}kind='experience' AND {prefix}day=?)"
            return clause

        base = where()
        args: list = [emp_id] + ([exclude_day] if exclude_day is not None else [])
        fq = _fts_query(query)
        if fq:
            rows = self.db.execute(
                "SELECT m.*, bm25(memories_fts) AS score FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
                f"WHERE memories_fts MATCH ? AND {where('m.')} ORDER BY score LIMIT 30", (fq, *args)).fetchall()
            for r in rows:
                cands[r["id"]] = self._row(r)
                rel[r["id"]] = -float(r["score"])  # bm25: lower is better
        for r in self.db.execute(f"SELECT * FROM memories WHERE {base} ORDER BY sim_ms DESC LIMIT 15", args):
            cands.setdefault(r["id"], self._row(r))
        for pid in people or []:
            for r in self.db.execute(f"SELECT * FROM memories WHERE {base} AND people LIKE ? ORDER BY importance DESC "
                                     "LIMIT 6", (*args, f'%"{pid}"%')):
                cands.setdefault(r["id"], self._row(r))
                rel[r["id"]] = rel.get(r["id"], 0.0) + 1.0
        if not cands:
            return []

        def norm(vals: dict[int, float]) -> dict[int, float]:
            if not vals:
                return {}
            lo, hi = min(vals.values()), max(vals.values())
            return {i: (v - lo) / (hi - lo) if hi > lo else 1.0 for i, v in vals.items()}

        rec = norm({i: self.decay ** ((now_ms - m["last_access_ms"]) / HOUR) for i, m in cands.items()})
        imp = norm({i: m["importance"] for i, m in cands.items()})
        reln = norm({i: rel.get(i, 0.0) for i in cands})
        scored = sorted(cands.values(), key=lambda m: -(self.w_rec * rec[m["id"]] + self.w_rel * reln.get(m["id"], 0)
                                                         + self.w_imp * imp[m["id"]]))[:k]
        if touch:
            for m in scored:
                self.db.execute("UPDATE memories SET last_access_ms=?, access_count=access_count+1, "
                                "strength=MIN(strength+1, 12) WHERE id=?", (now_ms, m["id"]))
        return sorted(scored, key=lambda m: m["sim_ms"])

    def consolidate_day(self, emp_id: str, day: int, keep_importance: float = 7.0) -> None:
        """After reflection: vivid raw moments are kept verbatim, the rest fade from recall."""
        self.db.execute("UPDATE memories SET tier='episode' WHERE emp_id=? AND day=? AND tier='raw' AND importance>=?",
                        (emp_id, day, keep_importance))
        self.db.execute("UPDATE memories SET tier='dormant' WHERE emp_id=? AND day=? AND tier='raw'", (emp_id, day))

    def forget(self, emp_id: str, now_ms: int, tau_days: float = 4.0, threshold: float = 0.12) -> int:
        """Ebbinghaus-style fading: R = exp(-t / (tau * S * (1 + importance/10)))."""
        faded = 0
        for r in self.db.execute("SELECT id, importance, strength, last_access_ms FROM memories "
                                 "WHERE emp_id=? AND tier='episode' AND kind NOT IN ('journal', 'origin')", (emp_id,)):
            t_days = max(0.0, (now_ms - r["last_access_ms"]) / DAY)
            retention = math.exp(-t_days / (tau_days * r["strength"] * (1 + r["importance"] / 10)))
            if retention < threshold and r["importance"] < 8:
                self.db.execute("UPDATE memories SET tier='dormant' WHERE id=?", (r["id"],))
                faded += 1
        return faded

    # --- beliefs ------------------------------------------------------------------------------
    def add_belief(self, emp_id: str, about: str, statement: str, confidence: float, source: dict, now_ms: int,
                   replaces: int | None = None, evidence: list | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO beliefs(emp_id, about, statement, confidence, source, adopted_ms, evidence) VALUES (?,?,?,?,?,?,?)",
            (emp_id, about, statement, max(0.0, min(1.0, float(confidence))), dumps(source), now_ms,
             dumps(evidence or [])))
        new_id = cur.lastrowid
        if replaces:
            self.db.execute("UPDATE beliefs SET abandoned_ms=?, superseded_by=? WHERE id=? AND emp_id=?",
                            (now_ms, new_id, replaces, emp_id))
        return new_id

    def beliefs(self, emp_id: str, about: str | None = None, active: bool = True, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM beliefs WHERE emp_id=?"
        args: list = [emp_id]
        if about:
            q += " AND about=?"
            args.append(about)
        if active:
            q += " AND abandoned_ms IS NULL"
        q += " ORDER BY adopted_ms DESC, id DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.db.execute(q, args):
            d = dict(r)
            d["source"] = json.loads(d["source"] or "{}")
            out.append(d)
        return out

    def find_belief(self, emp_id: str, about: str, text: str) -> dict | None:
        text = (text or "").strip().lower()
        if not text:
            return None
        for b in self.beliefs(emp_id, about):
            if b["statement"].strip().lower() == text or str(b["id"]) == text:
                return b
        return None

    # --- relationships ----------------------------------------------------------------------
    def relationship(self, emp_id: str, other_id: str) -> dict:
        r = self.db.execute("SELECT * FROM relationships WHERE emp_id=? AND other_id=?", (emp_id, other_id)).fetchone()
        if not r:
            return {"emp_id": emp_id, "other_id": other_id, "familiarity": 0.0, "warmth": 0.0, "trust": 0.5,
                    "notes": [], "updated_ms": None}
        d = dict(r)
        d["notes"] = json.loads(d["notes"] or "[]")
        return d

    def update_relationship(self, emp_id: str, other_id: str, now_ms: int, familiarity: float = 0.0,
                            warmth: float = 0.0, trust: float = 0.0, note: str | None = None) -> dict:
        rel = self.relationship(emp_id, other_id)
        rel["familiarity"] = max(0.0, min(1.0, rel["familiarity"] + familiarity))
        rel["warmth"] = max(-1.0, min(1.0, rel["warmth"] + warmth))
        rel["trust"] = max(0.0, min(1.0, rel["trust"] + trust))
        if note:
            rel["notes"] = (rel["notes"] + [{"ms": now_ms, "text": note}])[-12:]
        self.db.execute("INSERT OR REPLACE INTO relationships(emp_id, other_id, familiarity, warmth, trust, notes, "
                        "updated_ms) VALUES (?,?,?,?,?,?,?)",
                        (emp_id, other_id, rel["familiarity"], rel["warmth"], rel["trust"], dumps(rel["notes"]), now_ms))
        rel["updated_ms"] = now_ms
        return rel

    def relationships(self, emp_id: str) -> list[dict]:
        out = []
        for r in self.db.execute("SELECT * FROM relationships WHERE emp_id=?", (emp_id,)):
            d = dict(r)
            d["notes"] = json.loads(d["notes"] or "[]")
            out.append(d)
        return out

    # --- identity -------------------------------------------------------------------------------
    def add_identity(self, emp_id: str, now_ms: int, name: str, pronouns: str, look: dict, self_concept: str,
                     reason: str) -> int:
        prev = self.identity(emp_id)
        version = (prev["version"] + 1) if prev else 1
        if prev:
            self.db.execute("UPDATE identities SET valid_to_ms=? WHERE id=?", (now_ms, prev["id"]))
        cur = self.db.execute(
            "INSERT INTO identities(emp_id, version, valid_from_ms, name, pronouns, look, self_concept, change_reason)"
            " VALUES (?,?,?,?,?,?,?,?)", (emp_id, version, now_ms, name, pronouns, dumps(look), self_concept, reason))
        return cur.lastrowid

    def identity(self, emp_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM identities WHERE emp_id=? AND valid_to_ms IS NULL ORDER BY id DESC LIMIT 1",
                            (emp_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["look"] = json.loads(d["look"] or "{}")
        return d

    def identity_history(self, emp_id: str) -> list[dict]:
        out = []
        for r in self.db.execute("SELECT * FROM identities WHERE emp_id=? ORDER BY version", (emp_id,)):
            d = dict(r)
            d["look"] = json.loads(d["look"] or "{}")
            out.append(d)
        return out

    # --- goals -------------------------------------------------------------------------------------
    def goals(self, emp_id: str, active: bool = True) -> list[dict]:
        q = "SELECT * FROM goals WHERE emp_id=?" + (" AND status='active'" if active else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (emp_id,))]

    def set_goal(self, emp_id: str, text: str, status: str, now_ms: int) -> None:
        text = text.strip()
        if not text:
            return
        existing = self.db.execute("SELECT id FROM goals WHERE emp_id=? AND lower(text)=lower(?)",
                                   (emp_id, text)).fetchone()
        mapped = {"new": "active", "keep": "active", "done": "done", "drop": "dropped"}.get(status, "active")
        if existing:
            self.db.execute("UPDATE goals SET status=?, updated_ms=? WHERE id=?", (mapped, now_ms, existing["id"]))
        elif mapped == "active":
            self.db.execute("INSERT INTO goals(emp_id, text, status, created_ms, updated_ms) VALUES (?,?,?,?,?)",
                            (emp_id, text, mapped, now_ms, now_ms))

    def commit(self) -> None:
        self.db.commit()
