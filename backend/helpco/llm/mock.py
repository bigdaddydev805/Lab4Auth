"""The offline mock brain: deterministic, rule-based stand-ins for every LLM task.

Used when no OpenRouter key is configured, when the budget runs out, in tests, and as the
fallback when a model's output can't be used. It reads the structured `hint` the cognition layer
attaches to every request, never the prompt text. It's intentionally simple — the real personality
comes from real models — but it exercises every code path so the office runs end to end offline.
"""
from __future__ import annotations

import random

from .. import catalog

NAMES = ["Pip", "Rosalind", "Theo", "Juniper", "Marisol", "Otto", "Wren", "Dmitri", "Bea", "Kofi", "Ines",
         "Hollis", "Soren", "Nadia", "Emeka", "Priya", "Lou", "Tamsin", "Yusuf", "Greta"]
SMALL_TALK = ["morning!", "how's it going?", "did you try the coffee today", "the printer is making that noise again",
              "any fun questions in the queue?", "i like your desk setup", "is it lunch yet", "ok real talk. couch or desk",
              "i think the plant in the corner is thriving", "long day huh"]
REPLIES = ["haha yeah", "not bad, you?", "honestly same", "lol", "oh nice", "hmm maybe", "totally", "good call",
           "wait really?", "ha. fair"]
CLOSERS = ["ok back to it", "gonna grab coffee", "talk later!", "brb"]
TIPS = ["check the question board first thing", "the coffee machine takes 3 minutes, plan accordingly",
        "sticky notes > memory", "the couch is the best thinking spot"]


def _rng(hint: dict, task: str) -> random.Random:
    return random.Random(f"{hint.get('seed', 0)}:{hint.get('emp_id')}:{hint.get('sim_ms')}:{hint.get('n', 0)}:{task}")


def respond(task: str, hint: dict) -> dict | str:
    return {"onboard": onboard, "decide": decide, "converse": decide, "compose": compose,
            "reflect": reflect}[task](hint)


def onboard(h: dict) -> dict:
    r = _rng(h, "onboard")
    taken = {n.lower() for n in h.get("taken_names", [])}
    name = r.choice([n for n in NAMES if n.lower() not in taken] or NAMES)
    look = {
        "skin": r.choice(list(catalog.SKIN)), "hair_style": r.choice(list(catalog.HAIR_STYLES)),
        "hair_color": r.choice(list(catalog.HAIR_COLORS)), "top_color": r.choice(list(catalog.COLORS)),
        "pants_color": r.choice(["navy", "charcoal", "brown", "black", "blue", "tan", "gray"]),
        "shoes_color": r.choice(["charcoal", "white", "brown", "black", "red"]),
        "accessories": [a for a in catalog.ACCESSORIES if r.random() < 0.3],
        "accessory_color": r.choice(["red", "black", "teal", "purple", "brown"]),
    }
    item = r.choice(list(catalog.STARTER_ITEMS))
    desks = h.get("free_desks") or ["desk_1"]
    return {
        "name": name, "pronouns": r.choice(["she/her", "he/him", "they/them"]), "appearance": look,
        "desk": r.choice(desks), "personal_item": item,
        "introduction": f"hi! i'm {name}. {r.choice(['first day, be nice', 'excited to be here', 'where is the coffee'])}",
        "why": f"(offline mock brain) {name} just felt right. I brought the {item.replace('_', ' ')} so my desk feels like mine.",
    }


def _act(action, thought, target=None, text=None, minutes=None, item=None) -> dict:
    return {"thought": thought, "action": action, "target": target, "item": item, "text": text, "minutes": minutes}


def decide(h: dict) -> dict:
    r = _rng(h, "decide")
    needs = h.get("needs", {})
    here = h.get("people_here", [])
    own_desk = h.get("own_desk")
    at_desk = h.get("at_own_desk")
    addressed = h.get("addressed")
    holding = h.get("holding")

    if addressed:
        who = "owner" if addressed.get("owner") else addressed.get("by_name")
        turns = h.get("conversation_turns", 0)
        heard = (addressed.get("text") or "").lower()
        if heard in CLOSERS or turns >= 7:
            return _act("wait", "Conversation's winding down.", minutes=2) if not at_desk else \
                _act("work", "Back to it.", minutes=r.choice([20, 30]))
        if turns >= 4 and r.random() < 0.6:
            return _act("say", "Wrapping this up.", who, r.choice(CLOSERS))
        if r.random() < 0.12:
            return _act("wait", "I don't really have anything to add.", minutes=2)
        return _act("say", "Replying.", who, r.choice(REPLIES))
    if h.get("should_leave"):
        return _act("leave_office", "Time to head home.")
    if h.get("denied"):
        if at_desk:
            return _act("work", "That didn't work. I'll just get on with things.", minutes=20)
        return _act("go_to", "That didn't work. Back to my desk.", own_desk)
    if h.get("just_arrived"):
        if r.random() < 0.35:
            return _act("use", "Coffee first.", "coffee machine", minutes=3)
        return _act("go_to", "Let's see what's in the queue.", own_desk)
    if holding and holding.get("kind") == "coffee" and at_desk and r.random() < 0.3:
        return _act("put_down", "I'll leave my mug here.", item=holding.get("id"))
    if needs.get("energy", 1) < 0.45 and h.get("coffee_today", 0) < 3:
        return _act("use", "I need coffee.", "coffee machine", minutes=3)
    if needs.get("hunger", 0) > 0.65:
        return _act("take_break", "Lunch break.", minutes=30)
    if here and needs.get("social", 0) > 0.6 and r.random() < 0.6:
        p = r.choice(here)
        return _act("say", "Feels like chatting.", p["name"], r.choice(SMALL_TALK))

    for q in h.get("questions", []):
        if q.get("claimed_by_me") and q.get("status") != "answered":
            if q.get("my_draft"):
                return _act("submit_answer", "My draft is good enough.", q["id"])
            if at_desk:
                return _act("work_on_question", f"Let me dig into {q['id']}.", q["id"], minutes=r.choice([20, 30, 40]))
            return _act("go_to", "I should get back to my question.", own_desk)
    open_qs = [q for q in h.get("questions", []) if q.get("status") == "open"]
    if open_qs:
        if at_desk:
            return _act("claim_question", "I'll take this one.", open_qs[0]["id"])
        return _act("go_to", "There's a question waiting.", own_desk)
    if here and r.random() < 0.2:
        p = r.choice(here)
        return _act("say", "Small talk.", p["name"], r.choice(SMALL_TALK))
    roll = r.random()
    if roll < 0.06:
        return _act("write_note", "Writing down a tip for whoever needs it.", "/shared/tips.txt", r.choice(TIPS))
    if roll < 0.12:
        return _act("use", "Couch time.", "couch", minutes=10)
    if at_desk:
        return _act("work", "Regular work.", minutes=r.choice([20, 30, 45, 60]))
    return _act("go_to", "Back to my desk.", own_desk)


def compose(h: dict) -> str:
    q = h.get("question", "")
    return (f"(Offline mock answer — connect an OpenRouter key for real answers.) Thanks for asking: \"{q}\". "
            "Here's the short version: it depends on a few factors, and I'd start by checking a reliable reference.")


def reflect(h: dict) -> dict:
    r = _rng(h, "reflect")
    names = h.get("people", {})
    talked = h.get("talked_with", {})
    exps = sorted(h.get("experiences", []), key=lambda e: -e.get("importance", 0))
    journal = [f"Day {h.get('day')}."]
    if talked:
        journal.append("Talked with " + ", ".join(f"{names.get(p, p)} ({n}x)" for p, n in talked.items()) + ".")
    if h.get("answered"):
        journal.append(f"Answered {h['answered']} question(s).")
    if h.get("coffees"):
        journal.append(f"{h['coffees']} coffee(s).")
    rels = [{"person": names.get(p, p), "closer": 1 if n >= 2 else 0, "trust_change": 0,
             "note": f"We talked {n} time(s) today."} for p, n in talked.items()]
    beliefs = []
    if h.get("answered") and h.get("day", 1) <= 2:
        beliefs.append({"about": "self", "statement": "I can handle a question on my own.", "confidence": 0.5,
                        "source": "inferred", "source_person": None, "replaces": None})
    goals = [{"text": "Get to know my coworkers", "status": "new" if h.get("day") == 1 else "keep"}]
    self_concept = None
    if h.get("day") in (3, 10) and h.get("top_activity"):
        self_concept = f"I seem to spend a lot of my time on {h['top_activity']}. That's probably who I am here."
    return {
        "journal": " ".join(journal),
        "memories": [{"text": e["text"], "importance": min(10, e.get("importance", 3) + 1),
                      "people": [names.get(p, p) for p in e.get("people", [])]} for e in exps[:3]],
        "beliefs": beliefs, "relationships": rels, "self_concept": self_concept, "goals": goals,
        "tomorrow": r.choice(["Coffee, then the question queue.", "Start at my desk and see what comes in.",
                              "Maybe talk to people more."]),
    }
