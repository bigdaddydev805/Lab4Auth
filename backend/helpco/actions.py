"""Actions: turning an employee's intention into steps the world can carry out — or refusing.

Every verb resolves its target against the real world (objects, rooms, people, items, files,
questions), checks physical and permission rules, and becomes a Plan of steps:
walk → do (an activity with a duration) → call (an instant effect). The model never touches
state directly; anything impossible becomes an `action.denied` event with in-world feedback.
"""
from __future__ import annotations

import re

from . import catalog
from .clock import MINUTE
from .events import Event
from .office import DOOR, ROOM_NAMES
from .state import Employee, FileDoc, Item

ROOM_ALIASES = {
    "break room": "break_room", "break": "break_room", "kitchen": "break_room", "lounge": "break_room",
    "meeting room": "meeting_room", "meeting": "meeting_room", "conference room": "meeting_room",
    "work area": "work_area", "office": "work_area", "desks": "work_area", "main room": "work_area",
    "entrance": "entrance", "lobby": "entrance", "reception": "entrance",
}
OBJECT_ALIASES = {
    "coffee": "coffee_machine", "coffee maker": "coffee_machine", "espresso machine": "coffee_machine",
    "sofa": "couch", "table": "meeting_table", "board": "question_board", "queue": "question_board",
    "question queue": "question_board", "front door": "door", "exit": "door", "snacks": "fridge",
    "sink": "counter", "plant": "plant_work",
}
PRIVATE_DENIAL = "This folder contains private employee information. You are not authorized to access it."


def _norm(s: str | None) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^(the|a|an|my|your)\s+", "", s)
    return re.sub(r"[^\w\s/.'#-]", "", s).strip()


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class Actions:
    def __init__(self, eng):
        self.eng = eng

    @property
    def world(self):
        return self.eng.world

    @property
    def office(self):
        return self.eng.office

    # ================================================================ resolution
    def own_desk(self, e: Employee):
        return self.office.objects.get(e.desk_id) if e.desk_id else None

    def find_person(self, ref: str | None, e: Employee | None = None) -> Employee | None:
        r = _norm(ref)
        if not r:
            return None
        if r in self.world.employees:
            return self.world.employees[r]
        matches = [x for x in self.world.employees.values() if x.name and x.name.lower() == r and x is not e]
        if not matches:
            matches = [x for x in self.world.employees.values()
                       if x.name and x is not e and (x.name.lower().startswith(r) or r.startswith(x.name.lower()))]
        if not matches:
            return None
        present = [x for x in matches if x.status == "present"]
        if e and present:  # two people with the same name: prefer the one in the same room
            here = [x for x in present if self.eng.room_of(x) == self.eng.room_of(e)]
            return (here or present)[0]
        return (present or matches)[0]

    def find_object(self, ref: str | None, e: Employee):
        r = _norm(ref)
        if not r:
            return None
        if r in ("desk", "my desk", "own desk", "computer", "my computer", "workstation"):
            return self.own_desk(e)
        m = re.match(r"desk\s*_?#?\s*(\d)", r)
        if m:
            return self.office.objects.get(f"desk_{m.group(1)}")
        m = re.match(r"(.+?)'s desk", r)
        if m:
            p = self.find_person(m.group(1), e)
            return self.office.objects.get(p.desk_id) if p and p.desk_id else None
        r2 = OBJECT_ALIASES.get(r, r).replace(" ", "_")
        if r2 in self.office.objects:
            return self.office.objects[r2]
        for o in self.office.objects.values():
            if o.kind in ("desk_chair",):
                continue
            if r == o.name.lower().removeprefix("the ") or r2 == o.kind or r in o.name.lower():
                return o
        return None

    def find_room(self, ref: str | None) -> str | None:
        r = _norm(ref)
        if r in ROOM_ALIASES:
            return ROOM_ALIASES[r]
        r2 = r.replace(" ", "_")
        return r2 if r2 in ROOM_NAMES else None

    def find_question(self, ref: str | None):
        r = _norm(ref).replace("#", "").replace("question", "").strip().upper()
        if r.isdigit():
            r = f"Q{r}"
        return self.world.questions.get(r)

    def find_item(self, ref: str | None, e: Employee) -> Item | None:
        r = _norm(ref)
        if not r:
            return self.world.items.get(e.holding) if e.holding else None
        if r in self.world.items:
            return self.world.items[r]
        if e.holding and r in (self.world.items[e.holding].kind, self.world.items[e.holding].label.lower(),
                               "it", "mug", "coffee", "item", "this"):
            return self.world.items[e.holding]
        room = self.eng.room_of(e)
        for it in self.world.items.values():
            if r in (it.kind, it.kind.replace("_", " ")) or r in it.label.lower():
                if self._item_room(it) == room or it.location.get("held_by") == e.id:
                    return it
        return None

    def _item_room(self, it: Item) -> str | None:
        loc = it.location
        if "on" in loc:
            o = self.office.objects.get(loc["on"])
            return self.office.room_of(o.tiles[0]) if o else None
        if "at" in loc:
            return self.office.room_of(tuple(loc["at"]))
        if "held_by" in loc:
            h = self.world.employees.get(loc["held_by"])
            return self.eng.room_of(h) if h else None
        return None

    # ================================================================ spots
    def spot_free(self, spot: tuple, e: Employee) -> bool:
        who = self.eng.reserved.get(spot)
        if who and who != e.id:
            return False
        for other in self.world.present():
            if other.id != e.id and not other.path and tuple(other.pos) == spot:
                return False
        return True

    def free_spot(self, obj, e: Employee):
        for s in obj.spots:
            if self.spot_free((s.x, s.y), e):
                return s
        return None

    def near_surface_tile(self, obj, e: Employee):
        """A standing tile next to a surface (or the seat, for your own desk)."""
        if obj.kind == "desk" and obj.id == e.desk_id:
            s = obj.spots[0]
            return (s.x, s.y), s.facing
        for s in obj.spots:
            if self.spot_free((s.x, s.y), e) and s.pose != "sit":
                return (s.x, s.y), s.facing
        for tx, ty in obj.tiles:
            for nx, ny, facing in ((tx, ty + 1, "up"), (tx, ty - 1, "down"), (tx - 1, ty, "right"), (tx + 1, ty, "left")):
                if self.office.passable(nx, ny) and self.spot_free((nx, ny), e):
                    return (nx, ny), facing
        return None, None

    # ================================================================ items
    def create_item(self, kind: str, label: str, owner: str | None = None, created_by: str | None = None,
                    text: str = "", color: str | None = None) -> Item:
        iid = self.world.next_id("item", "i")
        it = Item(iid, kind, label, self.eng.clock.ms, created_by=created_by, owner=owner, text=text, color=color)
        self.world.items[iid] = it
        return it

    def place_on(self, it: Item, obj_id: str) -> None:
        slot = sum(1 for x in self.world.items.values() if x.location.get("on") == obj_id and x.id != it.id)
        it.location = {"on": obj_id, "slot": slot}

    def drop_item(self, e: Employee, it: Item, quiet: bool = False) -> str:
        tile = self.eng.current_tile(e)
        surfaces = self.office.surfaces_near(tile)
        desk = self.own_desk(e)
        if desk and desk in surfaces:
            surfaces = [desk]
        if surfaces:
            self.place_on(it, surfaces[0].id)
            where = surfaces[0].name if surfaces[0].id != e.desk_id else "my desk"
            where_name = surfaces[0].name
        else:
            it.location = {"at": list(tile)}
            where = where_name = "the floor"
        if e.holding == it.id:
            e.holding = None
        self.eng.push_item(it)
        self.eng.push_employee(e)
        if not quiet:
            self.eng.emit(Event("item.put_down", self.eng.clock.ms, actor=e.id, room=self.eng.room_of(e),
                                payload={"item": it.id, "label": it.label, "where_label": where,
                                         "where_name": where_name}))
        return where

    # ================================================================ dispatch
    def execute(self, e: Employee, d: dict, decision_id: str | None = None, source: str = "llm") -> None:
        verb = d.get("action")
        handler = getattr(self, f"do_{verb}", None)
        if handler is None:
            self.eng.deny(e, str(verb), d.get("target"), "that isn't something you can do here",
                          decision_id=decision_id)
            return
        self._decision_id, self._source = decision_id, source
        handler(e, d)

    def _plan(self, e, verb, label, steps, reserved=None):
        plan = self.eng.new_plan(verb, label, steps, self._decision_id, self._source)
        plan.reserved = [tuple(r) for r in (reserved or [])]
        self.eng.start_plan(e, plan)

    def _deny(self, e, d, reason, security=False):
        self.eng.deny(e, d.get("action", "?"), d.get("target"), reason, security, self._decision_id)

    def _desk_steps(self, e: Employee):
        desk = self.own_desk(e)
        if not desk:
            return None, None
        s = desk.spots[0]
        return desk, [{"type": "walk", "to": (s.x, s.y), "facing": s.facing}]

    # ================================================================ verbs
    def do_go_to(self, e: Employee, d: dict) -> None:
        ref = d.get("target")
        person = self.find_person(ref, e)
        if person and person is not e:
            if person.status != "present":
                return self._deny(e, d, f"{person.display} isn't in the office right now")
            return self._plan(e, "go_to", f"going to {person.display}",
                              [{"type": "walk", "to_fn": lambda: self._near(e, person)}])
        obj = self.find_object(ref, e)
        if obj:
            if obj.kind == "door":
                return self.do_leave_office(e, d)
            spot = self.free_spot(obj, e) if obj.spots else None
            if obj.spots and not spot:
                return self._deny(e, d, f"{obj.name} is occupied right now")
            if spot:
                return self._plan(e, "go_to", f"going to {obj.name}",
                                  [{"type": "walk", "to": (spot.x, spot.y), "facing": spot.facing}],
                                  reserved=[(spot.x, spot.y)] if spot.pose == "sit" else [])
            tile, facing = self.near_surface_tile(obj, e)
            if tile is None:
                return self._deny(e, d, f"you can't get close to {obj.name}")
            return self._plan(e, "go_to", f"going to {obj.name}", [{"type": "walk", "to": tile, "facing": facing}])
        room = self.find_room(ref)
        if room:
            tiles = [t for t in self.office.room_tiles(room) if self.spot_free(t, e)]
            here = self.eng.current_tile(e)
            if self.office.room_of(here) == room:
                return self._deny(e, d, f"you're already in {ROOM_NAMES[room]}")
            tiles.sort(key=lambda t: abs(t[0] - here[0]) + abs(t[1] - here[1]))
            goal = tiles[min(len(tiles) - 1, self.eng.rng.randint(0, 3))] if tiles else None
            if not goal:
                return self._deny(e, d, "there's no room to stand there")
            return self._plan(e, "go_to", f"going to {ROOM_NAMES[room]}", [{"type": "walk", "to": goal}])
        return self._deny(e, d, f"you're not sure where '{ref}' is")

    def _near(self, e: Employee, other: Employee):
        if other.status != "present":
            return None
        target = self.eng.current_tile(other)
        avoid = {t for t, who in self.eng.reserved.items() if who != e.id}
        avoid |= {tuple(o.pos) for o in self.world.present() if o.id not in (e.id,) and not o.path}
        options = self.office.tiles_near(target, avoid)
        if not options:
            options = self.office.tiles_near(target, set())
        if not options:
            return None
        here = self.eng.current_tile(e)
        if here in options:
            return here
        return options[0]

    def do_work(self, e: Employee, d: dict) -> None:
        desk, steps = self._desk_steps(e)
        if not desk:
            return self._deny(e, d, "you don't have a desk yet")
        minutes = _clamp(d.get("minutes"), 5, 120, 30)

        def done(emp, m):
            emp.needs["energy"] = max(0.0, emp.needs["energy"] - 0.01 * m / 15)
        steps.append({"type": "do", "kind": "work", "label": f"working at {desk.name}", "pose": "type",
                      "minutes": minutes, "facing": "down", "on_done": done,
                      "memory": "I worked at my desk for {m} minutes."})
        self._plan(e, "work", f"working at {desk.name}", steps, reserved=[(desk.spots[0].x, desk.spots[0].y)])

    def do_claim_question(self, e: Employee, d: dict) -> None:
        q = self.find_question(d.get("target"))
        if not q:
            return self._deny(e, d, f"there's no question called '{d.get('target')}' in the queue")
        desk, steps = self._desk_steps(e)
        if not desk:
            return self._deny(e, d, "you need a desk computer to claim questions")

        def claim(emp):
            if q.status == "answered":
                self._deny(emp, d, f"{q.id} was already answered by {self.eng.name_of(q.answered_by)}")
                return False
            if q.claimed_by and q.claimed_by != emp.id:
                self._deny(emp, d, f"{self.eng.name_of(q.claimed_by)} already claimed {q.id}")
                return False
            q.status, q.claimed_by, q.claimed_ms = "claimed", emp.id, self.eng.clock.ms
            emp.known_questions[q.id] = "claimed"
            self.eng.emit(Event("question.claimed", self.eng.clock.ms, actor=emp.id, room="work_area",
                                payload={"id": q.id}))
            self.eng.push_question(q)
            return True
        steps += [{"type": "do", "kind": "claim", "label": f"looking at {q.id}", "pose": "type", "seconds": 45,
                   "facing": "down"}, {"type": "call", "fn": claim}]
        self._plan(e, "claim_question", f"claiming {q.id}", steps, reserved=[(desk.spots[0].x, desk.spots[0].y)])

    def do_work_on_question(self, e: Employee, d: dict) -> None:
        q = self.find_question(d.get("target"))
        if not q:
            return self._deny(e, d, f"there's no question called '{d.get('target')}'")
        if q.status == "answered":
            return self._deny(e, d, f"{q.id} was already answered by {self.eng.name_of(q.answered_by)}")
        desk, steps = self._desk_steps(e)
        if not desk:
            return self._deny(e, d, "you need a desk to work on questions")
        minutes = _clamp(d.get("minutes"), 5, 120, 30)
        notes = (d.get("text") or "").strip()

        def start(emp):
            if q.status == "open":
                q.status, q.claimed_by, q.claimed_ms = "claimed", emp.id, self.eng.clock.ms
                self.eng.emit(Event("question.claimed", self.eng.clock.ms, actor=emp.id, room="work_area",
                                    payload={"id": q.id}))
                self.eng.push_question(q)
            return True

        def done(emp, m):
            self.eng.cognition.compose(emp, q, notes)
        steps += [{"type": "call", "fn": start},
                  {"type": "do", "kind": "question", "label": f"working on {q.id}", "pose": "type", "minutes": minutes,
                   "facing": "down", "on_done": done, "then": "wait_async"}]
        self._plan(e, "work_on_question", f"working on {q.id}", steps,
                   reserved=[(desk.spots[0].x, desk.spots[0].y)])

    def do_submit_answer(self, e: Employee, d: dict) -> None:
        q = self.find_question(d.get("target"))
        if not q:
            return self._deny(e, d, f"there's no question called '{d.get('target')}'")
        if q.status == "answered":
            return self._deny(e, d, f"{q.id} was already answered by {self.eng.name_of(q.answered_by)}")
        text = (d.get("text") or "").strip()
        draft = q.drafts.get(e.id, {}).get("text")
        answer = text if len(text) >= 40 else draft
        if not answer:
            return self._deny(e, d, f"you haven't written a draft for {q.id} yet (work_on_question first)")
        desk, steps = self._desk_steps(e)
        if not desk:
            return self._deny(e, d, "you need a desk computer to send answers")

        def submit(emp):
            if q.status == "answered":
                self._deny(emp, d, f"{q.id} was already answered by {self.eng.name_of(q.answered_by)}")
                return False
            q.status, q.answer, q.answered_by, q.answered_ms = "answered", answer[:2000], emp.id, self.eng.clock.ms
            emp.stats["answers"] = emp.stats.get("answers", 0) + 1
            emp.known_questions[q.id] = "answered"
            self.eng.emit(Event("question.answered", self.eng.clock.ms, actor=emp.id, room="work_area",
                                payload={"id": q.id, "answer": q.answer, "question": q.text}))
            self.eng.push_question(q)
            return True
        steps += [{"type": "do", "kind": "submit", "label": f"sending the answer to {q.id}", "pose": "type",
                   "seconds": 60, "facing": "down"}, {"type": "call", "fn": submit}]
        self._plan(e, "submit_answer", f"sending the answer to {q.id}", steps,
                   reserved=[(desk.spots[0].x, desk.spots[0].y)])

    def do_say(self, e: Employee, d: dict) -> None:
        text = (d.get("text") or "").strip()
        ref = _norm(d.get("target"))
        if not text:
            return self._deny(e, d, "you didn't say anything")
        if ref in ("owner", "the owner", "intercom", "boss"):
            self.eng.emit(Event("owner.reply", self.eng.clock.ms, actor=e.id, room=self.eng.room_of(e),
                                targets=["owner"], payload={"text": text[:600]}, decision_id=self._decision_id),
                          witnesses={e.id: "did"})
            return self._after_instant(e)
        if ref in ("everyone", "all", "room", "everybody", ""):
            self.eng.speak(e, "everyone", text, self._source, self._decision_id)
            return self._after_instant(e)
        other = self.find_person(ref, e)
        if not other or other is e:
            return self._deny(e, d, f"you don't know anyone called '{d.get('target')}'")
        if other.status != "present":
            return self._deny(e, d, f"{other.display} isn't in the office right now")
        if self.eng.room_of(other) == self.eng.room_of(e):
            self._face(e, other)
            self.eng.speak(e, other.id, text, self._source, self._decision_id)
            return self._after_instant(e)

        def say(emp):
            if other.status != "present" or self.eng.room_of(other) != self.eng.room_of(emp):
                self._deny(emp, d, f"{other.display} wasn't there anymore by the time you arrived")
                return False
            self._face(emp, other)
            self.eng.speak(emp, other.id, text, self._source, self._decision_id)
            return True
        self._plan(e, "say", f"going to talk to {other.display}",
                   [{"type": "walk", "to_fn": lambda: self._near(e, other)}, {"type": "call", "fn": say}])

    def _face(self, e: Employee, other: Employee) -> None:
        if e.activity.pose in ("sit", "type"):
            return
        (x0, y0), (x1, y1) = self.eng.current_tile(e), self.eng.current_tile(other)
        if abs(x1 - x0) >= abs(y1 - y0):
            e.facing = "right" if x1 > x0 else "left"
        else:
            e.facing = "down" if y1 > y0 else "up"

    def _after_instant(self, e: Employee, seconds: int = 180) -> None:
        """Instant actions don't interrupt what you're doing; if you weren't doing anything, check back later."""
        self.eng.push_employee(e)
        if e.id not in self.eng.plans:
            self.eng.at(self.eng.clock.ms + seconds * 1000, "nudge", emp=e.id, since=self.eng.clock.ms)

    def do_use(self, e: Employee, d: dict) -> None:
        obj = self.find_object(d.get("target"), e)
        if not obj:
            return self._deny(e, d, f"there's no '{d.get('target')}' here")
        kind = obj.kind
        if kind == "desk":
            if obj.id == e.desk_id:
                return self.do_work(e, d)
            spot = self.free_spot(obj, e)
            if not spot:
                return self._deny(e, d, f"someone is sitting at {obj.name}")
            minutes = _clamp(d.get("minutes"), 1, 60, 10)
            return self._plan(e, "use", f"sitting at {obj.name}", [
                {"type": "walk", "to": (spot.x, spot.y), "facing": spot.facing},
                {"type": "do", "kind": "visit", "label": f"sitting at {obj.name}", "pose": "sit", "minutes": minutes}],
                reserved=[(spot.x, spot.y)])
        if kind == "door":
            return self.do_leave_office(e, d)
        if kind in ("whiteboard",):
            return self.do_read(e, dict(d, target="whiteboard"))
        spot = self.free_spot(obj, e) if obj.spots else None
        if obj.spots and not spot:
            return self._deny(e, d, f"{obj.name} is being used by someone else")
        if spot is None:
            tile, facing = self.near_surface_tile(obj, e)
            if tile is None:
                return self._deny(e, d, f"you can't get to {obj.name}")
            walk = {"type": "walk", "to": tile, "facing": facing}
        else:
            walk = {"type": "walk", "to": (spot.x, spot.y), "facing": spot.facing}
        reserved = [(spot.x, spot.y)] if spot else []

        if kind == "coffee_machine":
            held = self.world.items.get(e.holding) if e.holding else None
            if held and held.kind not in ("coffee", "mug"):
                return self._deny(e, d, f"your hands are full ({held.label})")

            def pour(emp, m):
                cup = self.world.items.get(emp.holding) if emp.holding else None
                if cup and cup.kind in ("coffee", "mug"):
                    cup.kind, cup.label = "coffee", "a mug of coffee" if cup.owner != emp.id else cup.label
                else:
                    cup = self.create_item("coffee", "a mug of coffee", owner=emp.id, created_by=emp.id)
                    cup.location = {"held_by": emp.id}
                    emp.holding = cup.id
                emp.needs["energy"] = min(1.0, emp.needs["energy"] + 0.3)
                emp.stats["coffees"] = emp.stats.get("coffees", 0) + 1
                emp.stats["coffees_today"] = emp.stats.get("coffees_today", 0) + 1
                self.eng.push_item(cup)
                self.eng.emit(Event("coffee.made", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                    payload={"item": cup.id}))
            return self._plan(e, "use", "making coffee", [walk, {
                "type": "do", "kind": "coffee", "label": "making coffee", "pose": "stand", "minutes": 3,
                "facing": "up", "on_done": pour}], reserved)
        if kind == "couch":
            minutes = _clamp(d.get("minutes"), 3, 90, 15)

            def rest(emp, m):
                emp.needs["energy"] = min(1.0, emp.needs["energy"] + 0.08 * m / 15)
            return self._plan(e, "use", "resting on the couch", [walk, {
                "type": "do", "kind": "rest", "label": "resting on the couch", "pose": "sit", "minutes": minutes,
                "on_done": rest, "memory": "I rested on the couch for {m} minutes."}], reserved)
        if kind == "fridge":
            def snack(emp, m):
                emp.needs["hunger"] = max(0.0, emp.needs["hunger"] - 0.4)
            return self._plan(e, "use", "grabbing a snack", [walk, {
                "type": "do", "kind": "snack", "label": "grabbing a snack from the fridge", "pose": "stand",
                "minutes": 3, "facing": "up", "on_done": snack, "memory": "I grabbed a snack from the fridge."}],
                reserved)
        if kind == "printer":
            source = self._printable(e)
            if not source:
                return self._deny(e, d, "you have nothing to print yet (write a note or a draft first)")
            if e.holding:
                return self._deny(e, d, "your hands are full")

            def print_it(emp, m):
                title, text = source
                page = self.create_item("paper", f"a printout of {title}", owner=emp.id, created_by=emp.id, text=text)
                page.location = {"held_by": emp.id}
                emp.holding = page.id
                self.eng.push_item(page)
                self.eng.emit(Event("printer.printed", self.eng.clock.ms, actor=emp.id, room="work_area",
                                    payload={"item": page.id, "label": page.label}))
            return self._plan(e, "use", "printing", [walk, {
                "type": "do", "kind": "print", "label": "printing", "pose": "stand", "minutes": 2, "facing": "up",
                "on_done": print_it}], reserved)
        if kind == "question_board":
            def look(emp):
                self.eng.cognition.saw_question_board(emp, remember=True)
                return True
            return self._plan(e, "use", "checking the question board", [walk, {
                "type": "do", "kind": "look", "label": "reading the question board", "pose": "stand", "seconds": 30,
                "facing": "up"}, {"type": "call", "fn": look}])
        if kind == "meeting_table":
            minutes = _clamp(d.get("minutes"), 1, 90, 10)
            return self._plan(e, "use", "sitting at the meeting table", [walk, {
                "type": "do", "kind": "sit", "label": "sitting at the meeting table", "pose": "sit",
                "minutes": minutes}], reserved)
        minutes = _clamp(d.get("minutes"), 1, 30, 2)
        return self._plan(e, "use", f"at {obj.name}", [walk, {
            "type": "do", "kind": "look", "label": f"spending a moment at {obj.name}", "pose": "stand",
            "minutes": minutes}], reserved)

    def _printable(self, e: Employee):
        drafts = [(q.id, q.drafts[e.id]) for q in self.world.questions.values() if e.id in q.drafts]
        notes = [it for it in self.world.items.values() if it.kind == "note" and it.created_by == e.id]
        best = None
        if drafts:
            qid, dr = max(drafts, key=lambda x: x[1]["ms"])
            best = (dr["ms"], (f"my draft for {qid}", dr["text"]))
        if notes:
            n = max(notes, key=lambda x: x.created_ms)
            if not best or n.created_ms > best[0]:
                best = (n.created_ms, ("my note", n.text))
        return best[1] if best else None

    def do_take_break(self, e: Employee, d: dict) -> None:
        minutes = _clamp(d.get("minutes"), 5, 90, 15)
        hour = self.eng.clock.dt().hour
        lunch = 11 <= hour < 14
        couch = self.office.objects["couch"]
        spot = self.free_spot(couch, e)
        label = "having lunch" if lunch else "taking a break"

        def rest(emp, m):
            emp.needs["energy"] = min(1.0, emp.needs["energy"] + 0.08 * m / 15)
            if m >= 15:
                emp.needs["hunger"] = 0.05 if lunch else max(0.0, emp.needs["hunger"] - 0.35)
        if spot:
            walk = {"type": "walk", "to": (spot.x, spot.y), "facing": spot.facing}
            pose, reserved = "sit", [(spot.x, spot.y)]
        else:
            tiles = [t for t in self.office.room_tiles("break_room") if self.spot_free(t, e)]
            if not tiles:
                return self._deny(e, d, "the break room is packed")
            walk = {"type": "walk", "to": self.eng.rng.choice(tiles)}
            pose, reserved = "stand", []
        self._plan(e, "take_break", label, [walk, {
            "type": "do", "kind": "break", "label": f"{label} in the break room", "pose": pose, "minutes": minutes,
            "on_done": rest, "memory": f"I took a {minutes}-minute {'lunch ' if lunch else ''}break."}], reserved)

    def do_wait(self, e: Employee, d: dict) -> None:
        minutes = _clamp(d.get("minutes"), 1, 60, 5)
        pose = e.activity.pose if e.activity.pose in ("sit", "type") else "stand"
        pose = "sit" if pose == "type" else pose
        label = (d.get("text") or "").strip()[:60] or "taking a moment"
        self._plan(e, "wait", label, [{"type": "do", "kind": "wait", "label": label, "pose": pose,
                                       "minutes": minutes}])

    # --- writing and reading ---------------------------------------------------------------------
    def _path_allowed(self, e: Employee, path: str) -> tuple[bool, str, bool]:
        """(allowed, normalized path, is_security_issue)"""
        p = "/" + "/".join(x for x in path.strip().split("/") if x and x not in (".", ".."))
        parts = p.split("/")
        if len(parts) < 2 or parts[1] not in ("shared", "private"):
            return False, p, False
        if parts[1] == "private":
            if len(parts) < 3:
                return False, p, True
            owner = parts[2].lower()
            mine = {e.id.lower(), (e.name or "").lower(), "me", "mine"}
            if owner not in mine:
                return False, p, True
            if owner in ("me", "mine"):
                parts[2] = (e.name or e.id).lower()
                p = "/".join(parts)
        return True, p, False

    def do_write_note(self, e: Employee, d: dict) -> None:
        text = (d.get("text") or "").strip()[:800]
        if not text:
            return self._deny(e, d, "you didn't write anything")
        target = (d.get("target") or "").strip()
        if target.startswith("/"):
            ok, path, security = self._path_allowed(e, target)
            if not ok:
                return self._deny(e, d, PRIVATE_DENIAL if security else
                                  "files live under /shared/… or your own /private/<your name>/… folder", security)
            desk, steps = self._desk_steps(e)
            if not desk:
                return self._deny(e, d, "you need a computer (a desk) to write files")

            def write(emp):
                f = self.world.files.get(path)
                now = self.eng.clock.ms
                if f:
                    f.text = (f.text + "\n" + text)[-8000:]
                    f.updated_ms = now
                else:
                    self.world.files[path] = FileDoc(path, emp.id, text, now, now)
                self.eng.emit(Event("file.written", now, actor=emp.id, room=self.eng.room_of(emp),
                                    payload={"path": path, "text": text}, visibility="private"),
                              witnesses={emp.id: "did"})
                return True
            steps += [{"type": "do", "kind": "write", "label": f"typing up {path}", "pose": "type", "minutes": 2,
                       "facing": "down"}, {"type": "call", "fn": write}]
            return self._plan(e, "write_note", f"writing {path}", steps)
        if _norm(target) == "whiteboard":
            wb = self.office.objects["whiteboard"]
            s = wb.spots[0]

            def write_board(emp):
                now = self.eng.clock.ms
                f = self.world.files.get("/whiteboard")
                line = text
                if f:
                    f.text = (f.text + "\n" + line)[-2000:]
                    f.updated_ms = now
                else:
                    self.world.files["/whiteboard"] = FileDoc("/whiteboard", emp.id, line, now, now)
                self.eng.emit(Event("note.written", now, actor=emp.id, room="meeting_room",
                                    payload={"where": "whiteboard", "where_label": "the whiteboard", "text": text}))
                self.eng.broadcast({"type": "whiteboard", "text": self.world.files["/whiteboard"].text})
                return True
            return self._plan(e, "write_note", "writing on the whiteboard", [
                {"type": "walk", "to": (s.x, s.y), "facing": "up"},
                {"type": "do", "kind": "write", "label": "writing on the whiteboard", "pose": "stand", "minutes": 2,
                 "facing": "up"}, {"type": "call", "fn": write_board}])
        obj = self.find_object(target or "my desk", e) if target else self.own_desk(e)
        if not obj or not obj.surface:
            return self._deny(e, d, f"you can't leave a note on '{target or 'that'}'")
        tile, facing = self.near_surface_tile(obj, e)
        if tile is None:
            return self._deny(e, d, f"you can't reach {obj.name}")
        where = "my desk" if obj.id == e.desk_id else obj.name

        def stick(emp):
            note = self.create_item("note", f"a sticky note from {emp.display}", owner=None, created_by=emp.id,
                                    text=text)
            self.place_on(note, obj.id)
            self.eng.push_item(note)
            self.eng.emit(Event("note.written", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                payload={"item": note.id, "where": obj.id, "where_label": where, "text": text}))
            return True
        self._plan(e, "write_note", f"leaving a note on {obj.name}", [
            {"type": "walk", "to": tile, "facing": facing},
            {"type": "do", "kind": "write", "label": "writing a note", "pose": "stand", "minutes": 1},
            {"type": "call", "fn": stick}])

    def do_read(self, e: Employee, d: dict) -> None:
        target = (d.get("target") or "").strip()
        n = _norm(target)
        if target.startswith("/"):
            if n.rstrip("/") in ("/shared", "shared"):
                return self._read_listing(e, d)
            ok, path, security = self._path_allowed(e, target)
            if not ok:
                return self._deny(e, d, PRIVATE_DENIAL if security else "there's no such folder", security)
            desk, steps = self._desk_steps(e)
            if not desk:
                return self._deny(e, d, "you need a computer (a desk) to open files")

            def open_file(emp):
                f = self.world.files.get(path)
                if not f:
                    self._deny(emp, d, f"there's no file at {path}")
                    return False
                self.eng.emit(Event("file.read", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                    payload={"path": path, "text": f.text[:1500], "author": f.author},
                                    visibility="private"), witnesses={emp.id: "did"})
                return True
            steps += [{"type": "do", "kind": "read", "label": f"reading {path}", "pose": "type", "minutes": 1,
                       "facing": "down"}, {"type": "call", "fn": open_file}]
            return self._plan(e, "read", f"reading {path}", steps)
        if n in ("whiteboard", "the whiteboard"):
            s = self.office.objects["whiteboard"].spots[0]

            def read_board(emp):
                f = self.world.files.get("/whiteboard")
                self.eng.emit(Event("whiteboard.read", self.eng.clock.ms, actor=emp.id, room="meeting_room",
                                    payload={"where_label": "the whiteboard",
                                             "text": f.text if f else "(it's blank)"}), witnesses={emp.id: "did"})
                return True
            return self._plan(e, "read", "reading the whiteboard", [
                {"type": "walk", "to": (s.x, s.y), "facing": "up"},
                {"type": "do", "kind": "read", "label": "reading the whiteboard", "pose": "stand", "seconds": 40,
                 "facing": "up"}, {"type": "call", "fn": read_board}])
        q = self.find_question(target) if re.match(r"^#?q?\s*\d+$", n or "x") else None
        if q:
            self.eng.emit(Event("note.read", self.eng.clock.ms, actor=e.id, room=self.eng.room_of(e),
                                payload={"where_label": f"question {q.id}", "text": q.text}),
                          witnesses={e.id: "did"})
            return self._after_instant(e)
        if n in ("question board", "board", "queue", "question queue"):
            return self.do_use(e, dict(d, target="question_board"))
        it = self.find_item(target, e)
        if not it:
            return self._deny(e, d, f"you don't see '{target}' anywhere nearby")
        if not it.text:
            return self._deny(e, d, f"there's nothing written on {it.label}")
        label = it.label

        def read_it(emp):
            self.eng.emit(Event("note.read", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                payload={"item": it.id, "where_label": label, "text": it.text}),
                          witnesses={emp.id: "did"})
            return True
        if it.location.get("held_by") == e.id:
            read_it(e)
            return self._after_instant(e)
        steps = self._steps_to_item(e, it)
        if steps is None:
            return self._deny(e, d, f"you can't reach {label}")
        self._plan(e, "read", f"reading {label}", steps + [
            {"type": "do", "kind": "read", "label": f"reading {label}", "pose": "stand", "seconds": 40},
            {"type": "call", "fn": read_it}])

    def _read_listing(self, e: Employee, d: dict) -> None:
        desk, steps = self._desk_steps(e)
        if not desk:
            return self._deny(e, d, "you need a computer (a desk) to browse the shared drive")

        def listing(emp):
            files = sorted(p for p in self.world.files if p.startswith("/shared/"))
            text = "\n".join(files) if files else "(the shared drive is empty)"
            self.eng.emit(Event("file.read", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                payload={"path": "/shared/", "text": text}, visibility="private"),
                          witnesses={emp.id: "did"})
            return True
        steps += [{"type": "do", "kind": "read", "label": "browsing the shared drive", "pose": "type", "minutes": 1,
                   "facing": "down"}, {"type": "call", "fn": listing}]
        self._plan(e, "read", "browsing the shared drive", steps)

    def _steps_to_item(self, e: Employee, it: Item):
        loc = it.location
        if "on" in loc:
            obj = self.office.objects.get(loc["on"])
            if not obj:
                return None
            tile, facing = self.near_surface_tile(obj, e)
            return None if tile is None else [{"type": "walk", "to": tile, "facing": facing}]
        if "at" in loc:
            return [{"type": "walk", "to": tuple(loc["at"])}]
        return None

    # --- items between people ----------------------------------------------------------------------
    def do_give(self, e: Employee, d: dict) -> None:
        it = self.find_item(d.get("item") or "", e) if d.get("item") else (
            self.world.items.get(e.holding) if e.holding else None)
        if not it or it.location.get("held_by") != e.id:
            return self._deny(e, d, "you aren't holding that")
        other = self.find_person(d.get("target"), e)
        if not other or other is e:
            return self._deny(e, d, f"you don't know anyone called '{d.get('target')}'")
        if other.status != "present":
            return self._deny(e, d, f"{other.display} isn't in the office right now")

        def hand_over(emp):
            if other.status != "present" or self.eng.room_of(other) != self.eng.room_of(emp):
                self._deny(emp, d, f"{other.display} wasn't there anymore")
                return False
            if other.holding:
                self._deny(emp, d, f"{other.display}'s hands are full")
                return False
            it.location = {"held_by": other.id}
            emp.holding, other.holding = None, it.id
            self.eng.push_item(it)
            self.eng.push_employee(emp)
            self.eng.push_employee(other)
            self.eng.emit(Event("item.given", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                targets=[other.id], payload={"item": it.id, "label": it.label, "to": other.id}))
            self.eng.request_decision(other, "addressed", by=emp.id, text=f"(handed you {it.label})")
            return True
        self._plan(e, "give", f"giving {it.label} to {other.display}", [
            {"type": "walk", "to_fn": lambda: self._near(e, other)},
            {"type": "do", "kind": "give", "label": f"handing over {it.label}", "pose": "stand", "seconds": 5},
            {"type": "call", "fn": hand_over}])

    def do_put_down(self, e: Employee, d: dict) -> None:
        it = self.world.items.get(e.holding) if e.holding else None
        if not it:
            return self._deny(e, d, "you aren't holding anything")
        self.drop_item(e, it)
        self._after_instant(e, seconds=20)

    def do_pick_up(self, e: Employee, d: dict) -> None:
        it = self.find_item(d.get("item") or d.get("target"), e)
        if not it:
            return self._deny(e, d, "you don't see that nearby")
        if "held_by" in it.location:
            holder = it.location["held_by"]
            return self._deny(e, d, "you're already holding it" if holder == e.id else
                              f"{self.eng.name_of(holder)} is holding it")
        if e.holding:
            return self._deny(e, d, "your hands are full")
        steps = self._steps_to_item(e, it)
        if steps is None:
            return self._deny(e, d, "you can't reach it")

        def take(emp):
            if "held_by" in it.location or emp.holding:
                self._deny(emp, d, "you couldn't pick it up")
                return False
            it.location = {"held_by": emp.id}
            emp.holding = it.id
            self.eng.push_item(it)
            self.eng.push_employee(emp)
            self.eng.emit(Event("item.picked_up", self.eng.clock.ms, actor=emp.id, room=self.eng.room_of(emp),
                                payload={"item": it.id, "label": it.label}))
            return True
        self._plan(e, "pick_up", f"picking up {it.label}", steps + [
            {"type": "do", "kind": "pick_up", "label": f"picking up {it.label}", "pose": "stand", "seconds": 5},
            {"type": "call", "fn": take}])

    # --- identity ----------------------------------------------------------------------------------
    def do_change_appearance(self, e: Employee, d: dict) -> None:
        text = (d.get("text") or d.get("target") or "").strip()
        changes = {}
        for part in re.split(r"[;,\n]", text):
            if ":" in part:
                k, v = part.split(":", 1)
                changes[k.strip().lower().replace(" ", "_")] = v.strip()
        if not changes:
            return self._deny(e, d, "say what to change, like 'top_color: blue' or 'accessories: glasses'")
        look = dict(e.look)
        for k, v in changes.items():
            if k in ("accessories", "accessory"):
                look["accessories"] = [] if v.lower() in ("none", "") else [a.strip() for a in v.split("+")]
            elif k in catalog.APPEARANCE_FIELDS:
                look[k] = v
        clean, errors = catalog.validate_appearance(look)
        if errors:
            return self._deny(e, d, "; ".join(errors[:2]))
        if clean == e.look:
            return self._deny(e, d, "that's already how you look")
        before = catalog.describe_look(e.look)
        e.look = clean
        after = catalog.describe_look(clean)
        change = ", ".join(f"{k} → {v}" for k, v in changes.items())
        ident = self.eng.memory.identity(e.id) or {}
        self.eng.memory.add_identity(e.id, self.eng.clock.ms, e.name, e.pronouns, clean,
                                     ident.get("self_concept", ""), f"changed look: {change}")
        self.eng.emit(Event("appearance.changed", self.eng.clock.ms, actor=e.id, room=self.eng.room_of(e),
                            payload={"change": change, "before": before, "after": after, "look": clean}))
        self.eng.push_employee(e)
        self._after_instant(e, seconds=30)

    def do_leave_office(self, e: Employee, d: dict) -> None:
        now = self.eng.clock.ms
        earliest = self.eng._ms(self.eng.clock.today(), "earliest_leave")
        if now < earliest:
            end = self.eng.clock.fmt_time(self.eng._ms(self.eng.clock.today(), "workday_end"))
            return self._deny(e, d, f"it's only {self.eng.clock.fmt_time()} — the workday runs until {end} "
                                    f"(you can head out from {self.eng.clock.fmt_time(earliest)})")
        self._leave(e, "heading home")

    def force_leave(self, e: Employee, why: str) -> None:
        self._decision_id, self._source = None, "schedule"
        self._leave(e, why)

    def _leave(self, e: Employee, label: str) -> None:
        def go(emp):
            self.eng._depart(emp)
            return False
        steps = []
        if e.holding and e.desk_id:
            desk = self.own_desk(e)
            s = desk.spots[0]
            steps += [{"type": "walk", "to": (s.x, s.y), "facing": s.facing},
                      {"type": "call", "fn": lambda emp: (self.drop_item(emp, self.world.items[emp.holding])
                                                          if emp.holding else None) or True}]
        steps += [{"type": "walk", "to": DOOR, "facing": "down"}, {"type": "call", "fn": go}]
        self._plan(e, "leave_office", label, steps)

    def autopilot(self, e: Employee, memory: str) -> None:
        """A simple, non-LLM fallback so nobody gets stuck."""
        self._decision_id, self._source = None, "autopilot"
        desk = self.own_desk(e)
        if desk:
            s = desk.spots[0]
            steps = [{"type": "walk", "to": (s.x, s.y), "facing": s.facing},
                     {"type": "do", "kind": "work", "label": f"working at {desk.name}", "pose": "type", "minutes": 20,
                      "memory": memory}]
            self._plan(e, "work", f"working at {desk.name}", steps, reserved=[(s.x, s.y)])
        else:
            self._plan(e, "wait", "taking a moment", [{"type": "do", "kind": "wait", "label": "taking a moment",
                                                       "pose": "stand", "minutes": 10, "memory": memory}])
