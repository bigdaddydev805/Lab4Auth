"""Nightly reflection: after work, each employee thinks back over their own day.

Input: only what *they* experienced (their raw memories of the day), plus their current beliefs,
relationships and goals. Output: a private journal entry, the few moments worth keeping, updated
beliefs (hearsay stays hearsay), relationship changes, goals, and — rarely — a change in how they
see themselves. Old beliefs and old self-concepts are closed, not erased: they become history.
"""
from __future__ import annotations

from collections import Counter

from .clock import DAY
from .events import Event
from .llm import LLMRequest
from .llm.schema import reflect_schema

SELF_CONCEPT_MIN_DAYS = 3   # after the first one, identity can't churn faster than this

PROMPT = """It's the evening of {date}. You've gone home, and you're thinking back over your day at HelpCo ({tenure}).

What happened today, as you experienced it:
{experiences}

What you currently believe:
{beliefs}

How things stand with people (your own sense of it):
{relationships}

Your goals: {goals}
{journals}
Reflect honestly. Most days don't change much, and that's fine — only record what actually matters to you.
Reply with only a JSON object:
- journal: a private diary entry about today, in your own voice.
- memories: the few moments worth remembering long-term (0–5), first person, each with importance 1–10 (1 = mundane, 10 = life-changing) and the names of the people involved.
- beliefs: things you now believe — or believe differently — about yourself, the world, or a specific person. about = "self", "world", or the person's name (or "owner"). Keep track of where it came from: if someone told you something you didn't see yourself, use source "told" with source_person, and word it that way ("Deven says Maya…") — don't turn hearsay into fact. replaces = the exact wording of an old belief this replaces, or null.
- relationships: for each person you dealt with, closer (-2..2), trust_change (-2..2) and a short note.
- self_concept: ONLY if today genuinely changed how you see yourself, a few sentences in the first person; otherwise null.
- goals: goals to add (new), keep, mark done, or drop.
- tomorrow: what you'd like to do tomorrow, briefly."""


class Reflection:
    def __init__(self, eng):
        self.eng = eng

    def run_night(self, workers: list, on_done) -> None:
        if not workers:
            on_done()
            return
        remaining = {w.id for w in workers}

        def finish(e, res):
            try:
                self.apply(e, res)
            finally:
                remaining.discard(e.id)
                if not remaining:
                    on_done()
        for e in workers:
            req = self.build(e)
            self.eng.spawn(self.eng.gateway.call(req), lambda res, e=e: finish(e, res), order_key=e.id)

    def build(self, e) -> LLMRequest:
        eng, mem, clock = self.eng, self.eng.memory, self.eng.clock
        day = eng.world.day_number
        exps = mem.day_experiences(e.id, day)
        if len(exps) > 120:  # keep the most important moments, in order
            keep = sorted(sorted(exps, key=lambda m: -m["importance"])[:120], key=lambda m: m["sim_ms"])
            exps = keep
        lines = "\n".join(f"- {clock.fmt_time(m['sim_ms'])} {m['text']}" for m in exps) or "- (a quiet day)"
        met_ids = Counter(p for m in exps for p in m["people"])
        blines = [f"- about yourself: {b['statement']}" for b in mem.beliefs(e.id, "self", limit=6)]
        for pid in list(met_ids)[:6]:
            for b in mem.beliefs(e.id, f"person:{pid}", limit=4):
                blines.append(f"- about {eng.name_of(pid)}: {b['statement']}"
                              + (f" ({b['source'].get('person')} told you)" if b["source"].get("channel") == "told"
                                 and b["source"].get("person") else ""))
        for b in mem.beliefs(e.id, "world", limit=4):
            blines.append(f"- about the world: {b['statement']}")
        rlines = []
        for pid in list(met_ids)[:6]:
            r = mem.relationship(e.id, pid)
            notes = "; ".join(n["text"] for n in r["notes"][-3:])
            rlines.append(f"- {eng.name_of(pid)}: familiarity {r['familiarity']:.1f}, warmth {r['warmth']:+.1f}, "
                          f"trust {r['trust']:.1f}" + (f" — {notes}" if notes else ""))
        goals = "; ".join(g["text"] for g in mem.goals(e.id)) or "none yet"
        journals = [m for m in mem.all_for(e.id, tiers=("episode",), limit=60) if m["kind"] == "journal"][:2]
        jtext = ("\nYour last journal entries:\n" + "\n".join(
            f"- ({clock.relative_day(j['sim_ms'])}) {j['text'][:500]}" for j in journals) + "\n") if journals else "\n"
        user = PROMPT.format(date=clock.fmt_date(), tenure=eng.cognition.tenure(e) if e.status == "present"
                             else f"day {e.days_worked} for you", experiences=lines,
                             beliefs="\n".join(blines) or "- nothing in particular yet",
                             relationships="\n".join(rlines) or "- you didn't really deal with anyone today",
                             goals=goals, journals=jtext)
        activities = Counter(m["text"].split(" for ")[0] for m in exps if m["text"].startswith("I worked")
                             or m["text"].startswith("I rested") or m["text"].startswith("I took"))
        talked = Counter()
        for m in exps:
            if m["text"].startswith("I said to") or " said to me" in m["text"]:
                for p in m["people"]:
                    talked[p] += 1
        hint = {"emp_id": e.id, "seed": eng.seed, "sim_ms": clock.ms, "day": day, "name": e.name,
                "experiences": [{"text": m["text"], "importance": m["importance"], "people": m["people"]} for m in exps],
                "people": {pid: eng.name_of(pid) for pid in met_ids}, "talked_with": dict(talked),
                "answered": e.stats.get("answers", 0) if day == 1 else sum(
                    1 for q in eng.world.questions.values() if q.answered_by == e.id and q.answered_ms
                    and clock.ms - q.answered_ms < DAY),
                "coffees": e.stats.get("coffees_today", 0),
                "top_activity": "working at my desk" if activities else None}
        return LLMRequest(task="reflect", system=[eng.cognition.rules.split("\n\nWhat you can do")[0],
                                                  eng.cognition.identity_block(e)],
                          user=user, schema=reflect_schema(), schema_name="nightly_reflection", max_tokens=3000,
                          emp_id=e.id, hire_no=e.hire_no, hint=hint, sim_ms=clock.ms)

    def _person(self, name: str | None):
        if not name:
            return None
        n = name.strip().lower()
        if n in ("owner", "the owner", "boss"):
            return "owner"
        e = self.eng.actions.find_person(n)
        return e.id if e else None

    def apply(self, e, res) -> None:
        eng, mem, clock = self.eng, self.eng.memory, self.eng.clock
        if isinstance(res, Exception) or not isinstance(getattr(res, "data", None), dict):
            mem.consolidate_day(e.id, eng.world.day_number)
            return
        d = res.data
        now, day = clock.ms, eng.world.day_number
        journal = (d.get("journal") or "").strip()
        if journal:
            mem.add(e.id, now, day, journal, 6, kind="journal", tier="episode",
                    provenance={"channel": "reflected"})
        for m in d.get("memories", [])[:6]:
            people = [p for p in (self._person(n) for n in m.get("people", [])) if p]
            mem.add(e.id, now, day, m["text"], m.get("importance", 5), kind="episode", tier="episode",
                    people=people, provenance={"channel": "reflected"})
        for b in d.get("beliefs", [])[:8]:
            about_raw = (b.get("about") or "").strip().lower()
            if about_raw in ("self", "me", "myself"):
                about = "self"
            elif about_raw in ("world", "office", "helpco", "work"):
                about = "world"
            else:
                pid = self._person(about_raw)
                if not pid:
                    continue
                about = f"person:{pid}"
            source_ch = b.get("source", "inferred")
            source_person = self._person(b.get("source_person"))
            conf = float(b.get("confidence", 0.5))
            if source_ch in ("told", "overheard") and source_person and source_person != "owner":
                conf = min(conf, 0.3 + 0.6 * mem.relationship(e.id, source_person)["trust"])  # can't exceed trust
            elif source_ch == "owner" or source_person == "owner":
                conf = min(conf, 0.3 + 0.6 * mem.relationship(e.id, "owner")["trust"])
            replaces = mem.find_belief(e.id, about, b.get("replaces") or "")
            mem.add_belief(e.id, about, b["statement"], conf,
                           {"channel": source_ch, "person": b.get("source_person"), "person_id": source_person},
                           now, replaces["id"] if replaces else None)
        for r in d.get("relationships", [])[:8]:
            pid = self._person(r.get("person"))
            if not pid or pid == e.id:
                continue
            mem.update_relationship(e.id, pid, now, familiarity=0.08, warmth=0.12 * int(r.get("closer", 0)),
                                    trust=0.08 * int(r.get("trust_change", 0)), note=r.get("note"))
        concept = (d.get("self_concept") or "").strip()
        if concept:
            hist = mem.identity_history(e.id)
            current = mem.identity(e.id) or {}
            last_change = max((h["valid_from_ms"] for h in hist if h["change_reason"] == "reflection"), default=None)
            if not current.get("self_concept") or last_change is None or \
                    now - last_change >= SELF_CONCEPT_MIN_DAYS * DAY:
                if concept != current.get("self_concept"):
                    mem.add_identity(e.id, now, e.name, e.pronouns, e.look, concept, "reflection")
                    eng.emit(Event("identity.changed", now, actor=e.id, payload={
                        "before": current.get("self_concept") or "", "after": concept}, visibility="system"),
                        remember=False)
        for g in d.get("goals", [])[:6]:
            mem.set_goal(e.id, g["text"], g.get("status", "new"), now)
        tomorrow = (d.get("tomorrow") or "").strip()
        if tomorrow:
            mem.add(e.id, now, day, f"My plan for tomorrow: {tomorrow}", 5, kind="plan", tier="episode",
                    provenance={"channel": "reflected"})
        mem.consolidate_day(e.id, day)
        faded = mem.forget(e.id, now)
        mem.commit()
        eng.emit(Event("reflection.completed", now, actor=e.id, payload={
            "journal": journal, "memories": len(d.get("memories", [])), "beliefs": len(d.get("beliefs", [])),
            "self_concept_changed": bool(concept), "faded": faded, "source": res.source, "model": res.model},
            visibility="system"), remember=False)
