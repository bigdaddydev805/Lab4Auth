"""Events: the append-only record of everything meaningful that happens.

Every event records who did it, where, and who perceived it (witnesses). Employees only ever learn
about events they witnessed, were told about, or later found evidence of.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    type: str
    sim_ms: int
    actor: str | None = None
    room: str | None = None
    targets: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    witnesses: dict = field(default_factory=dict)   # emp_id -> "did" | "saw" | "heard" | "told" | "read"
    source: str = "engine"                          # engine | llm | autopilot | schedule | player
    decision_id: str | None = None
    visibility: str = "public"                      # public | private (only witnesses) | system
    seq: int = 0

    def to_dict(self) -> dict:
        return {"seq": self.seq, "type": self.type, "sim_ms": self.sim_ms, "actor": self.actor, "room": self.room,
                "targets": self.targets, "payload": self.payload, "witnesses": self.witnesses,
                "source": self.source, "decision_id": self.decision_id, "visibility": self.visibility}
