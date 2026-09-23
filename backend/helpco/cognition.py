"""Cognition: when an employee thinks, and what they're told when they do.

The engine owns the clock and the facts; this module turns them into the employee's point of view:
the date and time, how long they've worked here, what they can see and hear right now, what just
happened to them today, what comes to mind from the past (retrieved memories), what they believe
about the people around them — and the list of things they could do. The model answers with one
intention; the world decides what happens.
"""
from __future__ import annotations

import itertools

from . import catalog
from .clock import DAY, HOUR, MINUTE
from .events import Event
from .llm import LLMRequest
from .llm.schema import VERBS, action_schema, onboard_schema
from .office import ROOM_NAMES

WORLD_RULES = """You work at HelpCo, a tiny office where people send in questions and the people who work here answer them. You're one of those people. This is your working life: it goes on day after day, and what happens here is yours to remember.

How this works:
- Each time you're asked, choose ONE next thing to do. The office — not you — decides what actually happens. If something isn't possible, you'll find out.
- You only know what you've seen, heard, read, or been told. People can be mistaken, and sometimes they lie.
- Talk like a coworker: short and casual, usually one line. Silence is fine. So is working, taking a break, wandering, or just sitting for a moment. Nobody expects you to be productive every minute.
- Questions sent to HelpCo are real work — someone is waiting for an answer. You can ask coworkers for help.
- Your "thought" is private. Everything you do or say can be noticed by people nearby.

What you can do (action — meaning):
{verbs}

Reply with only a JSON object: {{"thought": "...", "action": "...", "target": "..." or null, "item": "..." or null, "text": "..." or null, "minutes": number or null}}"""

DESK_NOTES = {"desk_1": "front row, by the meeting room door", "desk_2": "front row, under the question board",
              "desk_3": "back row, by the wall", "desk_4": "back row, a few steps from the printer"}

REFLECT_NOTE = """At night, after work, you think back over your day."""


def feel(needs: dict) -> str:
    en, hu, so = needs.get("energy", 1), needs.get("hunger", 0), needs.get("social", 0)
    energy = "rested" if en > 0.7 else "fine" if en > 0.45 else "a bit tired" if en > 0.25 else "exhausted"
    hunger = "not hungry" if hu < 0.4 else "getting hungry" if hu < 0.65 else "hungry"
    parts = [energy, hunger]
    if so > 0.7:
        parts.append("you could use some company")
    return ", ".join(parts)


class Cognition:
    def __init__(self, eng):
        self.eng = eng
        self._ids = itertools.count(1)
        self._burst: dict[str, tuple] = {}
        self._count: dict[str, int] = {}
        self.rules = WORLD_RULES.format(verbs="\n".join(f"- {k} — {v}" for k, v in VERBS.items()))

    # ================================================================ blocks
    def tenure(self, e) -> str:
        days = e.days_worked + (1 if e.status == "present" else 0)
        if days <= 1:
            return "today is your first day"
        return f"this is your day {days} at HelpCo (you started {self.eng.clock.fmt_span(self.eng.clock.ms - e.hired_ms)} ago)"

    def identity_block(self, e) -> str:
        mem = self.eng.memory
        ident = mem.identity(e.id) or {}
        goals = mem.goals(e.id)
        desk = self.eng.office.objects.get(e.desk_id)
        lines = [f"About you:",
                 f"- Your name is {e.name} ({e.pronouns}). You chose it yourself on your first day.",
                 f"- You look like this: {catalog.describe_look(e.look)}."]
        if desk:
            item = next((it for it in self.eng.world.items.values()
                         if it.owner == e.id and it.kind in catalog.STARTER_ITEMS), None)
            lines.append(f"- Your desk is {desk.name}" + (f"; you brought {item.label} for it." if item else "."))
        if e.intro:
            lines.append(f'- When you meet someone new, you introduce yourself: "{e.intro}"')
        concept = (ident.get("self_concept") or "").strip()
        lines.append(f"- How you see yourself right now: {concept}" if concept else
                     "- How you see yourself: you're still figuring that out.")
        if goals:
            lines.append("- Things you want: " + "; ".join(g["text"] for g in goals[:5]))
        return "\n".join(lines)

    def item_desc(self, it, viewer) -> str:
        label = it.label
        if it.owner == viewer.id:
            label = "your " + label.split(" ", 1)[1] if label.startswith(("a ", "an ")) else f"your {label}"
        elif it.owner and it.owner in self.eng.world.employees and it.kind in catalog.STARTER_ITEMS:
            label = f"{self.eng.name_of(it.owner)}'s " + (label.split(" ", 1)[1] if label.startswith(("a ", "an "))
                                                          else label)
        return f"{label} [{it.id}]"

    # ================================================================ situation
    def situation(self, e, reasons: list[dict]) -> tuple[str, dict]:
        eng, clock, world, mem = self.eng, self.eng.clock, self.eng.world, self.eng.memory
        now = clock.ms
        today = clock.today()
        room = eng.room_of(e)
        tile = eng.current_tile(e)
        L: list[str] = []

        L.append(f"It's {clock.fmt_date()}, {clock.fmt_time()} ({clock.season()}).")
        wd_end = eng._ms(today, "workday_end")
        L.append(f"Workday: {clock.fmt_time(eng._ms(today, 'workday_start'))}–{clock.fmt_time(wd_end)}; "
                 f"lunch is around {clock.fmt_time(eng._ms(today, 'lunch'))}; "
                 f"the office closes at {clock.fmt_time(eng._ms(today, 'closes'))}.")
        if e.arrived_ms:
            L.append(f"You arrived at {clock.fmt_time(e.arrived_ms)}; {self.tenure(e)}.")
        should_leave = now >= wd_end
        if should_leave:
            L.append("The workday is over. People are heading home.")
        L.append(f"You feel {feel(e.needs)}.")

        # where you are
        desk_here = next((o for o in eng.office.objects.values() if o.kind == "desk" and
                          any((s.x, s.y) == tile for s in o.spots)), None)
        at_own_desk = bool(desk_here and desk_here.id == e.desk_id)
        where = ROOM_NAMES.get(room, "the office")
        if at_own_desk:
            where_line = f"You're at your desk ({desk_here.name}) in {where}"
        elif desk_here:
            owner = next((x for x in world.employees.values() if x.desk_id == desk_here.id), None)
            where_line = f"You're at {desk_here.name}" + (f" ({owner.display}'s desk)" if owner else "") + f" in {where}"
        else:
            where_line = f"You're in {where}"
        if e.activity.kind == "idle":
            L.append(f"{where_line}, {'sitting' if e.activity.pose == 'sit' else 'standing'}.")
        else:
            L.append(f"{where_line}, {e.activity.label}.")
        held = world.items.get(e.holding) if e.holding else None
        if held:
            L.append(f"You're holding {self.item_desc(held, e)}.")

        # people
        here = eng.people_in(room, exclude=e.id)
        people_hint = []
        if here:
            L.append("People here:")
            for p in here:
                bits = [f"- {p.display} ({p.pronouns}) — {p.activity.label}"]
                if p.id not in e.met:
                    bits.append("you haven't met yet")
                if p.id in e.last_spoke:
                    bits.append(f"you last spoke {clock.fmt_ago(e.last_spoke[p.id])}")
                if p.holding and world.items.get(p.holding):
                    bits.append(f"holding {world.items[p.holding].label}")
                L.append("; ".join(bits))
                people_hint.append({"id": p.id, "name": p.display, "activity": p.activity.kind})
        else:
            L.append("Nobody else is here.")
        elsewhere = []
        for oid, (ms, r) in e.last_seen.items():
            o = world.employees.get(oid)
            if o and o not in here and o.id != e.id and ms >= clock.ms_of(today, eng.times["day_starts"]):
                elsewhere.append(f"{o.display} ({ROOM_NAMES.get(r, 'around')}, {clock.fmt_ago(ms)})")
        if elsewhere:
            L.append("Last you saw: " + "; ".join(elsewhere) + ".")

        # things around you
        nearby = []
        for it in world.items.values():
            loc = it.location
            if "on" in loc:
                obj = eng.office.objects.get(loc["on"])
                if obj and eng.office.room_of(obj.tiles[0]) == room:
                    place = "your desk" if obj.id == e.desk_id else obj.name
                    nearby.append(f"{self.item_desc(it, e)} on {place}")
            elif "at" in loc and eng.office.room_of(tuple(loc["at"])) == room:
                nearby.append(f"{self.item_desc(it, e)} on the floor")
        if nearby:
            L.append("Things around you: " + "; ".join(nearby[:10]) + ".")
        if room == "meeting_room" and "/whiteboard" in world.files:
            L.append(f'The whiteboard says: "{world.files["/whiteboard"].text[-300:]}"')

        # questions
        qhint = []
        if room == "work_area":
            self.saw_question_board(e)
            qs = [q for q in world.questions.values() if q.status != "answered" or
                  (q.answered_ms and now - q.answered_ms < DAY)]
            if qs:
                L.append("The question queue (on the board and your screen):")
                for q in qs[-8:]:
                    status = q.status
                    if q.status == "claimed":
                        status = "claimed by you" if q.claimed_by == e.id else f"claimed by {eng.name_of(q.claimed_by)}"
                    elif q.status == "answered":
                        status = f"answered by {'you' if q.answered_by == e.id else eng.name_of(q.answered_by)}"
                    draft = " — you have a draft" if e.id in q.drafts and q.status != "answered" else ""
                    L.append(f'- {q.id} ({status}{draft}), sent {clock.fmt_ago(q.submitted_ms)}: "{q.text[:220]}"')
            else:
                L.append("The question queue is empty.")
        elif e.known_questions:
            open_known = [qid for qid, st in e.known_questions.items() if st != "answered"]
            if open_known:
                L.append("Last time you looked, the question queue had: " + ", ".join(open_known[-6:]) +
                         " (you can only see the queue from the work area).")
        for q in world.questions.values():
            if q.id in e.known_questions or room == "work_area":
                qhint.append({"id": q.id, "status": q.status, "claimed_by_me": q.claimed_by == e.id,
                              "my_draft": e.id in q.drafts, "text": q.text[:80]})

        # conversation in progress
        conv = next((c for c in world.conversations.values() if c.ended_ms is None and e.id in c.participants), None)
        if conv and conv.turns:
            L.append("The conversation you're in:")
            for t in conv.turns[-8:]:
                who = "You" if t["speaker"] == e.id else eng.name_of(t["speaker"])
                L.append(f'  {clock.fmt_time(t["ms"])} {who}: {t["text"]}')

        # today so far
        day_start = clock.ms_of(today, eng.times["day_starts"])
        recent = mem.recent(e.id, day_start, limit=12)
        if recent:
            L.append("Earlier today:")
            for m in recent:
                L.append(f"- {clock.fmt_time(m['sim_ms'])} {m['text']}")

        # memories that come to mind
        query = " ".join([p.display for p in here] + [e.activity.label] +
                         [r.get("text", "") for r in reasons] + [q["text"] for q in qhint[:3]])
        recalled = mem.retrieve(e.id, query, now, people=[p.id for p in here], exclude_day=world.day_number)
        if recalled:
            L.append("What comes to mind:")
            for m in recalled:
                L.append(f"- ({clock.relative_day(m['sim_ms'])}) {m['text']}")

        # what you believe about the people here (and yourself)
        bl = []
        for p in here:
            for b in mem.beliefs(e.id, f"person:{p.id}", limit=3):
                bl.append(f"- about {p.display}: {b['statement']}{self._source_note(b)}")
        if bl:
            L.append("What you believe:")
            L.extend(bl)

        # why you're being asked now
        L.append("")
        addressed = None
        denied = None
        for r in reasons:
            kind = r["reason"]
            if kind == "addressed":
                if r.get("by") == "owner":
                    L.append(f'The owner (on the intercom, only you can hear) just said to you: "{r.get("text")}"')
                    addressed = {"owner": True, "by_id": "owner", "by_name": "owner", "text": r.get("text")}
                else:
                    L.append(f'{eng.name_of(r.get("by"))} just said to you: "{r.get("text")}"')
                    addressed = {"owner": False, "by_id": r.get("by"), "by_name": eng.name_of(r.get("by")),
                                 "text": r.get("text")}
            elif kind == "denied":
                L.append(f"What you just tried didn't work: {r.get('why')}")
                denied = r.get("why")
            elif kind == "new_question":
                L.append(f"A new question ({r.get('question')}) just showed up in the queue.")
            elif kind == "arrived":
                L.append("You just walked in through the front door.")
            elif kind == "workday_over":
                L.append("It's the end of the workday.")
            elif kind == "finished":
                L.append(f"You just finished what you were doing ({r.get('verb', 'your last task')}).")
        L.append("What do you do next?")

        hint = {
            "emp_id": e.id, "name": e.name, "seed": eng.seed, "sim_ms": now, "day": world.day_number,
            "room": room, "own_desk": e.desk_id, "at_own_desk": at_own_desk, "needs": dict(e.needs),
            "holding": {"id": held.id, "kind": held.kind} if held else None, "people_here": people_hint,
            "addressed": addressed, "denied": denied, "questions": qhint, "should_leave": should_leave,
            "just_arrived": any(r["reason"] == "arrived" for r in reasons),
            "conversation_turns": len(conv.turns) if conv else 0,
            "coffee_today": e.stats.get("coffees_today", 0),
        }
        return "\n".join(L), hint

    @staticmethod
    def _source_note(b: dict) -> str:
        src = b.get("source") or {}
        ch = src.get("channel")
        who = src.get("person")
        if ch == "told" and who:
            return f" ({who} told you)"
        if ch == "overheard":
            return " (you overheard it)"
        if ch == "owner":
            return " (the owner told you)"
        if ch == "inferred":
            return " (your impression)"
        return ""

    # ================================================================ deciding
    def decide(self, e) -> None:
        eng = self.eng
        reasons = eng.reasons.pop(e.id, [])
        # loop guard: nobody gets to make endless decisions without time passing
        last_ms, count = self._burst.get(e.id, (None, 0))
        count = count + 1 if last_ms == eng.clock.ms else 1
        self._burst[e.id] = (eng.clock.ms, count)
        if count > 4:
            eng.reasons[e.id] = reasons + eng.reasons.get(e.id, [])
            eng.decision_done(e)
            eng.at(eng.clock.ms + MINUTE, "nudge", emp=e.id, since=eng.clock.ms)
            return
        self._count[e.id] = self._count.get(e.id, 0) + 1
        text, hint = self.situation(e, reasons)
        hint["n"] = self._count[e.id]
        task = "converse" if hint["addressed"] else "decide"
        req = LLMRequest(task=task, system=[self.rules, self.identity_block(e)], user=text, schema=action_schema(),
                         schema_name="next_action", max_tokens=450, emp_id=e.id, hire_no=e.hire_no, hint=hint,
                         sim_ms=eng.clock.ms)
        eng.spawn(eng.gateway.call(req), lambda res: self._apply(e, req, reasons, res), order_key=e.id)

    def _apply(self, e, req, reasons, res) -> None:
        eng = self.eng
        eng.decision_done(e)
        if e.status != "present" or isinstance(res, Exception):
            if isinstance(res, Exception) and e.status == "present":
                eng.actions.autopilot(e, "I lost my train of thought for a bit.")
            return
        d = res.data or {}
        decision_id = f"d{next(self._ids)}"
        e.last_decision = {"id": decision_id, "ms": eng.clock.ms, "thought": d.get("thought"),
                           "action": d.get("action"), "target": d.get("target"), "item": d.get("item"),
                           "text": d.get("text"), "minutes": d.get("minutes"), "source": res.source,
                           "model": res.model, "reasons": [r["reason"] for r in reasons]}
        if res.source == "fallback":
            eng.emit(Event("llm.fallback", eng.clock.ms, actor=e.id, payload={"error": res.error or "",
                                                                              "task": req.task},
                           visibility="system"), remember=False)
        eng.broadcast({"type": "decision", "employee": e.id, "decision": e.last_decision})
        source = "llm" if res.source in ("openrouter", "cassette") else res.source
        eng.actions.execute(e, d, decision_id, source)
        if e.id not in eng.deciding and eng.reasons.get(e.id):
            eng.deciding.add(e.id)
            e.thinking = True
            self.decide(e)

    # ================================================================ questions
    def saw_question_board(self, e, remember: bool = False) -> None:
        world = self.eng.world
        new = []
        for q in world.questions.values():
            if e.known_questions.get(q.id) != q.status:
                if q.id not in e.known_questions and q.status != "answered":
                    new.append(q)
                e.known_questions[q.id] = q.status
        e.queue_seen_ms = self.eng.clock.ms
        if remember or new:
            opened = [q for q in world.questions.values() if q.status != "answered"]
            summary = ", ".join(f"{q.id} ({q.status})" for q in opened[-6:]) or "nothing open"
            self.eng.memory.add(e.id, self.eng.clock.ms, world.day_number,
                                f"I checked the question queue: {summary}.", 2 if not new else 3)

    def compose(self, e, q, notes: str = "") -> None:
        eng = self.eng
        plan = eng.plans.get(e.id)
        plan_id = plan.id if plan else None
        draft = q.drafts.get(e.id, {}).get("text")
        L = [f'You\'re at your desk working on question {q.id}, sent in {eng.clock.fmt_ago(q.submitted_ms)}:',
             f'"{q.text}"']
        if draft:
            L += ["", "Your current draft:", draft]
        if notes:
            L += ["", f"Your notes: {notes}"]
        related = eng.memory.retrieve(e.id, q.text + " " + q.id, eng.clock.ms, touch=False, k=5)
        if related:
            L += ["", "Things you remember that might be relevant:"] + [f"- {m['text']}" for m in related]
        L += ["", "Write the answer you'd send them now: plain text, friendly and clear, at most about 150 words. "
                  "If you're unsure about something, say so honestly instead of making it up. Reply with just the answer."]
        req = LLMRequest(task="compose", system=[self.rules.split("\n\nWhat you can do")[0], self.identity_block(e)],
                         user="\n".join(L), schema=None, max_tokens=600, emp_id=e.id, hire_no=e.hire_no,
                         hint={"emp_id": e.id, "seed": eng.seed, "sim_ms": eng.clock.ms, "question": q.text},
                         sim_ms=eng.clock.ms)

        def done(res):
            text = res.data if (not isinstance(res, Exception) and isinstance(res.data, str)) else None
            if text:
                q.drafts[e.id] = {"text": text.strip()[:2000], "ms": eng.clock.ms}
                path = f"/private/{(e.name or e.id).lower()}/drafts/{q.id}.txt"
                from .state import FileDoc
                eng.world.files[path] = FileDoc(path, e.id, text.strip()[:2000], eng.clock.ms, eng.clock.ms)
                eng.emit(Event("question.draft_updated", eng.clock.ms, actor=e.id, room=eng.room_of(e),
                               payload={"id": q.id, "chars": len(text)}), witnesses={e.id: "did"})
            p = eng.plans.get(e.id)
            if p is not None and p.id == plan_id:
                eng.advance_plan(e)
        eng.spawn(eng.gateway.call(req), done, order_key=e.id)

    # ================================================================ day one
    def onboard(self, e) -> None:
        eng = self.eng
        if getattr(e, "_onboarding", False):
            return
        world = eng.world
        taken_desks = {x.desk_id for x in world.employees.values() if x.desk_id}
        free = [d for d in ("desk_1", "desk_2", "desk_3", "desk_4") if d not in taken_desks]
        coworkers = [x for x in world.employees.values() if x.name and x.status != "departed"]
        desk_lines = []
        for d in free:
            neighbors = [x.display for x in coworkers if x.desk_id and
                         {d, x.desk_id} in ({"desk_1", "desk_2"}, {"desk_3", "desk_4"}, {"desk_1", "desk_3"},
                                            {"desk_2", "desk_4"})]
            desk_lines.append(f"  - {d} ({DESK_NOTES[d]})" + (f", next to {', '.join(neighbors)}" if neighbors else ""))
        people = "\n".join(f"- {x.display} ({x.pronouns}), at {eng.office.objects[x.desk_id].name}"
                           for x in coworkers if x.desk_id) or \
            "- Nobody has started yet. You're one of the first people here."
        user = f"""You have been hired by HelpCo, a small office that answers questions submitted by people.
Today is your first day: {eng.clock.fmt_date()}.

These are the rooms and resources available to you:
{eng.office.describe()}

These employees currently work here:
{people}

Before you walk in, decide for yourself:
- what you would like your coworkers to call you (any name you like),
- your pronouns,
- how you look — choose from what the office has available:
{chr(10).join("  " + line for line in catalog.catalog_text().splitlines())}
- which desk you'd like:
{chr(10).join(desk_lines)}
- one personal item to bring for your desk: {", ".join(f"{k} ({v})" for k, v in catalog.STARTER_ITEMS.items())},
- a short line you'll say when you first meet each coworker,
- and, privately, why you chose these things.

Reply with only a JSON object with keys: name, pronouns, appearance (skin, hair_style, hair_color, top_color, pants_color, shoes_color, accessories, accessory_color), desk, personal_item, introduction, why."""
        req = LLMRequest(task="onboard", system=[self.rules.split("\n\nWhat you can do")[0]], user=user,
                         schema=onboard_schema(free or ["desk_1"]), schema_name="first_day", max_tokens=900,
                         emp_id=e.id, hire_no=e.hire_no,
                         hint={"emp_id": e.id, "seed": eng.seed, "sim_ms": eng.clock.ms, "free_desks": free,
                               "taken_names": [x.name for x in coworkers]}, sim_ms=eng.clock.ms)
        e._onboarding = True
        eng.spawn(eng.gateway.call(req), lambda res: self._onboarded(e, res, free), order_key=e.id)

    def _onboarded(self, e, res, free: list[str]) -> None:
        eng = self.eng
        e._onboarding = False
        d = res.data if not isinstance(res, Exception) and isinstance(res.data, dict) else {}
        name = " ".join(str(d.get("name") or "").split())[:40] or f"Employee {e.hire_no}"
        pronouns = " ".join(str(d.get("pronouns") or "they/them").split())[:30]
        look, _errors = catalog.validate_appearance(d.get("appearance"))
        taken = {x.desk_id for x in eng.world.employees.values() if x.desk_id and x is not e}
        desk = d.get("desk") if d.get("desk") in free and d.get("desk") not in taken else \
            next((x for x in ("desk_1", "desk_2", "desk_3", "desk_4") if x not in taken), None)
        e.name, e.pronouns, e.look, e.desk_id = name, pronouns, look, desk
        e.intro = " ".join(str(d.get("introduction") or f"hi, i'm {name}").split())[:240]
        item_kind = d.get("personal_item") if d.get("personal_item") in catalog.STARTER_ITEMS else "mug"
        it = eng.actions.create_item(item_kind, catalog.ITEM_LABELS[item_kind], owner=e.id, created_by=e.id,
                                     color=look.get("accessory_color") if item_kind == "mug" else None)
        if desk:
            eng.actions.place_on(it, desk)
        e.personal_item = it.id
        e.status = "offsite"
        why = " ".join(str(d.get("why") or "").split())[:700]
        now = eng.clock.ms
        eng.memory.add_identity(e.id, now, name, pronouns, look, "", "chose who to be on the first day")
        eng.memory.add(e.id, now, eng.world.day_number,
                       f"My first day at HelpCo. I chose to be called {name} ({pronouns}), picked "
                       f"{eng.office.objects[desk].name if desk else 'no desk'}, and brought {it.label}. Why: {why}",
                       9, kind="origin", tier="episode")
        eng.emit(Event("employee.identity_chosen", now, actor=e.id,
                       payload={"hire_no": e.hire_no, "name": name, "pronouns": pronouns, "look": look, "desk": desk,
                                "personal_item": it.id, "introduction": e.intro, "source": res.source
                                if not isinstance(res, Exception) else "error"},
                       visibility="system"), remember=False)
        eng.push_item(it)
        eng.push_employee(e)
        eng.memory.commit()
