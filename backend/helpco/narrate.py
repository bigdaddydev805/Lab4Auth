"""Turning events into words.

- `experience()` writes what a specific employee perceived, in the first person. This is what goes
  into their memory — so memories are subjective by construction.
- `summary()` writes the observer's (player's) third-person feed line.
"""
from __future__ import annotations

from .office import ROOM_NAMES

IMPORTANCE = {
    "speech.said": 3, "owner.message": 6, "owner.reply": 4, "owner.gift": 7, "employee.arrived": 1,
    "employee.left": 1, "employee.introduced": 6, "action.denied": 4, "coffee.made": 1, "question.submitted": 3,
    "question.claimed": 2, "question.answered": 5, "question.draft_updated": 2, "note.written": 3, "note.read": 3,
    "file.written": 3, "file.read": 3, "item.given": 5, "item.put_down": 1, "item.picked_up": 1,
    "appearance.changed": 4, "activity.completed": 1, "printer.printed": 2, "whiteboard.read": 3,
}


def _q(text: str, n: int = 160) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def experience(eng, ev, me: str) -> tuple[str | None, float]:
    """First-person text of `ev` as perceived by employee `me` (or None if not memorable)."""
    t, p = ev.type, ev.payload
    name = eng.name_of
    actor_is_me = ev.actor == me
    imp = IMPORTANCE.get(t, 2)
    who = name(ev.actor) if ev.actor else "Someone"

    if t == "speech.said":
        to = p.get("to")
        to_name = "everyone" if to == "everyone" else name(to)
        if actor_is_me:
            return f'I said to {to_name}: "{_q(p["text"])}"', imp
        if to == me:
            return f'{who} said to me: "{_q(p["text"])}"', imp + 1
        return f'I overheard {who} say to {to_name}: "{_q(p["text"])}"', imp
    if t == "owner.message":
        return f'The owner messaged me on the intercom: "{_q(p["text"], 300)}"', imp
    if t == "owner.reply":
        return f'I replied to the owner: "{_q(p["text"])}"', imp
    if t == "owner.gift":
        return f"The owner gave me {p['label']}.", imp
    if t == "employee.arrived":
        return (("I arrived at work.", 2) if actor_is_me
                else (f"{who} arrived at work.", imp))
    if t == "employee.left":
        return ("I went home.", 1) if actor_is_me else (f"{who} went home for the day.", imp)
    if t == "employee.introduced":
        if actor_is_me:
            return f'I introduced myself to {name(p["to"])}: "{_q(p["text"])}"', 4
        if p.get("to") == me:
            return (f'I met {who} ({p.get("pronouns", "")}), who introduced {reflexive(p.get("pronouns", ""))}: '
                    f'"{_q(p["text"])}"', imp)
        return None, 0
    if t == "action.denied":
        if actor_is_me:
            return f"I tried to {p.get('verb', 'do something')} ({p.get('target') or ''}) but: {p['reason']}", \
                (5 if p.get("security") else imp)
        return None, 0
    if t == "coffee.made":
        return ("I made myself a coffee.", imp) if actor_is_me else (f"{who} made a coffee.", 1)
    if t == "question.submitted":
        return f'A new question arrived ({p["id"]}): "{_q(p["text"], 200)}"', imp
    if t == "question.claimed":
        return (f"I took question {p['id']}.", imp) if actor_is_me else (f"{who} took question {p['id']}.", imp)
    if t == "question.draft_updated":
        return (f"I worked on a draft answer for {p['id']}.", imp) if actor_is_me else (None, 0)
    if t == "question.answered":
        if actor_is_me:
            return f'I sent my answer to {p["id"]}: "{_q(p["answer"], 220)}"', imp
        return f"{who} answered question {p['id']}.", 2
    if t == "note.written":
        where = p.get("where_label", "somewhere")
        if actor_is_me:
            return f'I wrote a note on {where}: "{_q(p["text"])}"', imp
        return f"{who} wrote something on {where}.", 2
    if t in ("note.read", "whiteboard.read"):
        return (f'I read {p.get("where_label", "a note")}: "{_q(p["text"], 300)}"', imp) if actor_is_me else (None, 0)
    if t == "file.written":
        return (f'I saved {p["path"]}: "{_q(p["text"])}"', imp) if actor_is_me else (None, 0)
    if t == "file.read":
        return (f'I opened {p["path"]}: "{_q(p["text"], 300)}"', imp) if actor_is_me else (None, 0)
    if t == "item.given":
        if actor_is_me:
            return f"I gave {p['label']} to {name(p['to'])}.", imp
        if p.get("to") == me:
            return f"{who} gave me {p['label']}.", imp + 1
        return f"{who} gave {p['label']} to {name(p['to'])}.", 2
    if t == "item.put_down":
        return (f"I put {p['label']} on {p['where_label']}.", imp) if actor_is_me else (None, 0)
    if t == "item.picked_up":
        return (f"I picked up {p['label']}.", imp) if actor_is_me else (None, 0)
    if t == "appearance.changed":
        if actor_is_me:
            return f"I changed my look: {p['change']}.", imp
        return f"{who} changed their look ({p['change']}).", 3
    if t == "printer.printed":
        return (f"I printed {p['label']}.", imp) if actor_is_me else (None, 0)
    if t == "activity.completed":
        return (p.get("memory"), p.get("importance", 1)) if actor_is_me and p.get("memory") else (None, 0)
    return None, 0


def reflexive(pronouns: str) -> str:
    first = (pronouns or "").split("/")[0].strip().lower()
    return {"she": "herself", "he": "himself", "they": "themselves", "it": "itself"}.get(first, "themselves")


def summary(eng, ev) -> str:
    """Observer feed line (third person)."""
    t, p = ev.type, ev.payload
    who = eng.name_of(ev.actor) if ev.actor else ""
    room = ROOM_NAMES.get(ev.room or "", "")
    if t == "speech.said":
        to = "everyone" if p.get("to") == "everyone" else eng.name_of(p.get("to"))
        return f'{who} → {to}: "{_q(p["text"], 120)}"'
    if t == "owner.message":
        return f'You → {eng.name_of(ev.targets[0])} (intercom): "{_q(p["text"], 120)}"'
    if t == "owner.reply":
        return f'{who} → you: "{_q(p["text"], 120)}"'
    if t == "owner.gift":
        return f"You gave {eng.name_of(ev.targets[0])} {p['label']}"
    if t == "employee.hired":
        return f"A new employee was hired (#{p['hire_no']})"
    if t == "employee.identity_chosen":
        return f"New hire #{p['hire_no']} chose to be called {p['name']} ({p['pronouns']})"
    if t == "employee.introduced":
        e = eng.world.employees.get(ev.actor)
        return f'{who} introduced {reflexive(e.pronouns if e else "")} to {eng.name_of(p["to"])}: "{_q(p["text"], 100)}"'
    if t == "employee.arrived":
        return f"{who} arrived"
    if t == "employee.left":
        return f"{who} went home"
    if t == "employee.entered":
        return f"{who} entered {room}"
    if t == "action.started":
        return f"{who}: {p.get('label', p.get('verb'))}"
    if t == "action.denied":
        return f"{who} couldn't {p.get('verb')}: {p['reason']}"
    if t == "coffee.made":
        return f"{who} made a coffee"
    if t == "question.submitted":
        return f'New question {p["id"]}: "{_q(p["text"], 100)}"'
    if t == "question.claimed":
        return f"{who} claimed {p['id']}"
    if t == "question.draft_updated":
        return f"{who} drafted an answer to {p['id']}"
    if t == "question.answered":
        return f"{who} answered {p['id']}"
    if t == "note.written":
        return f'{who} wrote on {p.get("where_label")}: "{_q(p["text"], 80)}"'
    if t in ("note.read", "whiteboard.read"):
        return f"{who} read {p.get('where_label')}"
    if t == "file.written":
        return f"{who} saved {p['path']}"
    if t == "file.read":
        return f"{who} opened {p['path']}"
    if t == "item.given":
        return f"{who} gave {p['label']} to {eng.name_of(p['to'])}"
    if t == "item.put_down":
        return f"{who} put {p['label']} on {p.get('where_name', p['where_label'])}"
    if t == "item.picked_up":
        return f"{who} picked up {p['label']}"
    if t == "appearance.changed":
        return f"{who} changed their look: {p['change']}"
    if t == "printer.printed":
        return f"{who} printed {p['label']}"
    if t == "activity.completed":
        return f"{who} finished {p.get('kind')} ({p.get('minutes')} min)"
    if t == "office.lunch":
        return "It's lunchtime"
    if t == "office.workday_end":
        return "The workday is over"
    if t == "conversation.ended":
        return "A conversation ended"
    if t == "reflection.completed":
        return f"{who} reflected on the day"
    if t == "identity.changed":
        return f"{who}'s sense of self shifted"
    if t == "day.started":
        return f"Day {p['day']} — {p['date']}"
    if t == "day.ended":
        return f"The office closed for the night (day {p['day']})"
    if t == "clock.skipped":
        return f"Time passes… ({p['reason']})"
    if t == "llm.fallback":
        return f"{who}'s model call failed; used the offline brain ({p.get('error', '')[:80]})"
    return t
