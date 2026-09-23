"""World state: everything that physically exists. Only the engine mutates it."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Activity:
    kind: str = "idle"             # idle | walk | work | question | coffee | rest | talk | wait | ...
    label: str = "standing around"
    pose: str = "stand"            # stand | sit | type | sip | talk
    started_ms: int = 0
    until_ms: int | None = None
    target: str | None = None
    data: dict = field(default_factory=dict)


@dataclass
class Employee:
    id: str
    hire_no: int
    hired_ms: int
    name: str | None = None
    pronouns: str = "they/them"
    look: dict = field(default_factory=dict)
    desk_id: str | None = None
    intro: str = ""
    personal_item: str | None = None
    status: str = "hired"          # hired (not onboarded) | offsite | present | departed
    pos: tuple[int, int] = (20, 13)
    facing: str = "down"
    path: list = field(default_factory=list)
    path_t0: int = 0
    ms_per_tile: int = 0
    activity: Activity = field(default_factory=Activity)
    holding: str | None = None
    needs: dict = field(default_factory=lambda: {"energy": 1.0, "hunger": 0.1, "social": 0.3})
    arrived_ms: int | None = None
    left_ms: int | None = None
    last_spoke: dict = field(default_factory=dict)       # other id -> ms
    last_seen: dict = field(default_factory=dict)        # other id -> [ms, room]
    met: list = field(default_factory=list)              # ids of coworkers they've been introduced to
    thinking: bool = False
    last_decision: dict = field(default_factory=dict)
    days_worked: int = 0
    stats: dict = field(default_factory=lambda: {"conversations": 0, "answers": 0, "coffees": 0})
    queue_seen_ms: int = 0
    known_questions: dict = field(default_factory=dict)  # qid -> status as last seen
    llm_spend_usd: float = 0.0

    @property
    def display(self) -> str:
        return self.name or f"New hire #{self.hire_no}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pos"] = list(self.pos)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Employee":
        d = dict(d)
        d["pos"] = tuple(d["pos"])
        d["activity"] = Activity(**d.get("activity", {}))
        return cls(**d)

    def public(self) -> dict:
        """What clients render."""
        a = self.activity
        look_key = hashlib.sha1(json.dumps(self.look, sort_keys=True).encode()).hexdigest()[:10] if self.look else ""
        return {"id": self.id, "hire_no": self.hire_no, "name": self.display, "pronouns": self.pronouns,
                "look": self.look, "look_key": look_key, "status": self.status, "pos": list(self.pos), "facing": self.facing,
                "path": self.path, "path_t0": self.path_t0, "ms_per_tile": self.ms_per_tile,
                "activity": {"kind": a.kind, "label": a.label, "pose": a.pose, "until_ms": a.until_ms,
                             "target": a.target},
                "holding": self.holding, "thinking": self.thinking, "desk_id": self.desk_id,
                "needs": {k: round(v, 2) for k, v in self.needs.items()}}


@dataclass
class Item:
    id: str
    kind: str
    label: str
    created_ms: int
    created_by: str | None = None
    owner: str | None = None
    location: dict = field(default_factory=dict)   # {"on": obj_id} | {"held_by": emp_id} | {"at": [x, y]}
    text: str = ""
    color: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    id: str
    text: str
    submitted_ms: int
    status: str = "open"            # open | claimed | answered
    claimed_by: str | None = None
    claimed_ms: int | None = None
    drafts: dict = field(default_factory=dict)    # emp_id -> {"text", "ms"}
    answer: str | None = None
    answered_by: str | None = None
    answered_ms: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def public(self) -> dict:
        d = self.to_dict()
        d.pop("drafts")
        return d


@dataclass
class FileDoc:
    path: str
    author: str
    text: str
    created_ms: int
    updated_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Conversation:
    id: str
    room: str
    started_ms: int
    last_ms: int
    participants: list = field(default_factory=list)
    turns: list = field(default_factory=list)       # {"speaker", "to", "text", "ms"}
    ended_ms: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class World:
    employees: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    questions: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    conversations: dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {"emp": 0, "item": 0, "q": 0, "conv": 0})
    day_number: int = 0
    phase: str = "night"            # morning | workday | evening | night
    budget_exhausted: bool = False

    def next_id(self, kind: str, prefix: str) -> str:
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return f"{prefix}{self.counters[kind]}"

    def present(self) -> list[Employee]:
        return [e for e in self.employees.values() if e.status == "present"]

    def by_name(self, name: str) -> list[Employee]:
        n = (name or "").strip().lower()
        return [e for e in self.employees.values() if e.name and e.name.lower() == n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "employees": {k: v.to_dict() for k, v in self.employees.items()},
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "questions": {k: v.to_dict() for k, v in self.questions.items()},
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "conversations": {k: v.to_dict() for k, v in self.conversations.items() if v.ended_ms is None},
            "counters": self.counters, "day_number": self.day_number, "phase": self.phase,
            "budget_exhausted": self.budget_exhausted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        w = cls()
        w.employees = {k: Employee.from_dict(v) for k, v in d["employees"].items()}
        w.items = {k: Item(**v) for k, v in d["items"].items()}
        w.questions = {k: Question(**v) for k, v in d["questions"].items()}
        w.files = {k: FileDoc(**v) for k, v in d["files"].items()}
        w.conversations = {k: Conversation(**v) for k, v in d.get("conversations", {}).items()}
        w.counters = d["counters"]
        w.day_number = d["day_number"]
        w.phase = d.get("phase", "night")
        w.budget_exhausted = d.get("budget_exhausted", False)
        return w
