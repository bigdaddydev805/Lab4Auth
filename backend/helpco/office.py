"""The physical office: a tile grid, rooms, furniture, and pathfinding.

The map is authored as ASCII. Each floor character names the room a tile belongs to; '#' is wall.
Furniture is listed separately with its footprint and the spots where people stand or sit to use it.
Earshot and sight are per room: you hear and see what happens in the room you're in.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

TILE = 16

MAP = [
    "########################",
    "#mmmmmmm#bbbbbbbbbbbbbb#",
    "#mmmmmmm#bbbbbbbbbbbbbb#",
    "#mmmmmmm#bbbbbbbbbbbbbb#",
    "#mmmmmmm#bbbbbbbbbbbbbb#",
    "#mmmmmmm#bbbbbbbbbbbbbb#",
    "####w#######ww#####e####",
    "#wwwwwwwwwwwwwwww#eeeee#",
    "#wwwwwwwwwwwwwwww#eeeee#",
    "#wwwwwwwwwwwwwwwwweeeee#",
    "#wwwwwwwwwwwwwwwwweeeee#",
    "#wwwwwwwwwwwwwwww#eeeee#",
    "#wwwwwwwwwwwwwwww#eeeee#",
    "####################e###",
]
WIDTH = len(MAP[0])
HEIGHT = len(MAP)

ROOMS = {
    "m": {"id": "meeting_room", "name": "the meeting room"},
    "b": {"id": "break_room", "name": "the break room"},
    "w": {"id": "work_area", "name": "the work area"},
    "e": {"id": "entrance", "name": "the entrance"},
}
ROOM_NAMES = {r["id"]: r["name"] for r in ROOMS.values()}

DOOR = (20, 13)
SPAWN = (20, 12)


@dataclass
class Spot:
    x: int
    y: int
    facing: str = "down"     # which way a person faces while using it
    pose: str = "stand"      # stand | sit


@dataclass
class OfficeObject:
    id: str
    kind: str
    name: str
    x: int
    y: int
    w: int = 1
    h: int = 1
    blocking: bool = True
    surface: bool = False            # items can be left on it
    wall: bool = False               # mounted on a wall (drawn on the wall face)
    spots: list[Spot] = field(default_factory=list)
    description: str = ""

    @property
    def tiles(self):
        return [(x, y) for y in range(self.y, self.y + self.h) for x in range(self.x, self.x + self.w)]

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "x": self.x, "y": self.y,
                "w": self.w, "h": self.h, "surface": self.surface, "wall": self.wall,
                "spots": [s.__dict__ for s in self.spots]}


def _desk(n: int, x: int, y: int) -> list[OfficeObject]:
    return [
        OfficeObject(f"desk_{n}", "desk", f"Desk {n}", x, y, 3, 1, surface=True,
                     spots=[Spot(x + 1, y - 1, "down", "sit")],
                     description="a desk with a computer that shows the question queue"),
        OfficeObject(f"chair_{n}", "desk_chair", f"Desk {n} chair", x + 1, y - 1, blocking=False),
    ]


def build_objects() -> list[OfficeObject]:
    objs: list[OfficeObject] = []
    for n, (x, y) in enumerate([(2, 9), (7, 9), (2, 12), (7, 12)], start=1):
        objs += _desk(n, x, y)
    objs += [
        OfficeObject("printer", "printer", "the printer", 13, 7, spots=[Spot(13, 8, "up")],
                     description="prints notes and drafts onto paper"),
        OfficeObject("question_board", "question_board", "the question board", 6, 6, 2, 1, blocking=False,
                     wall=True, spots=[Spot(6, 7, "up"), Spot(7, 7, "up")],
                     description="a wall screen showing questions people have sent to HelpCo"),
        OfficeObject("plant_work", "plant_big", "a tall plant", 16, 7),
        OfficeObject("fridge", "fridge", "the fridge", 10, 1, spots=[Spot(10, 2, "up")]),
        OfficeObject("coffee_machine", "coffee_machine", "the coffee machine", 11, 1,
                     spots=[Spot(11, 2, "up")], description="makes coffee; mugs come out full"),
        OfficeObject("counter", "counter", "the kitchen counter", 12, 1, 3, 1, surface=True,
                     spots=[Spot(12, 2, "up"), Spot(13, 2, "up"), Spot(14, 2, "up")]),
        OfficeObject("couch", "couch", "the couch", 17, 1, 3, 1, surface=False,
                     spots=[Spot(17, 1, "down", "sit"), Spot(18, 1, "down", "sit"), Spot(19, 1, "down", "sit")],
                     description="a comfy couch for breaks"),
        OfficeObject("coffee_table", "coffee_table", "the coffee table", 17, 3, 3, 1, surface=True,
                     spots=[Spot(17, 4, "up"), Spot(19, 4, "up")]),
        OfficeObject("plant_break", "plant_big", "a tall plant", 22, 1),
        OfficeObject("meeting_table", "meeting_table", "the meeting table", 2, 2, 4, 2, surface=True,
                     spots=[Spot(3, 1, "down", "sit"), Spot(4, 1, "down", "sit"),
                            Spot(3, 4, "up", "sit"), Spot(4, 4, "up", "sit")]),
        OfficeObject("whiteboard", "whiteboard", "the whiteboard", 5, 0, 2, 1, blocking=False, wall=True,
                     spots=[Spot(6, 1, "up")], description="anyone in the meeting room can read it"),
        OfficeObject("coat_rack", "coat_rack", "the coat rack", 18, 7),
        OfficeObject("plant_lobby", "plant_small", "a small plant", 18, 12),
        OfficeObject("door", "door", "the front door", DOOR[0], DOOR[1], blocking=False, wall=True,
                     spots=[Spot(SPAWN[0], SPAWN[1], "down")]),
    ]
    return objs


class Office:
    def __init__(self):
        self.objects: dict[str, OfficeObject] = {o.id: o for o in build_objects()}
        self.blocked: set[tuple[int, int]] = set()
        self.goal_only: set[tuple[int, int]] = {DOOR}
        for o in self.objects.values():
            if o.blocking and not o.wall:
                self.blocked.update(o.tiles)
            for s in o.spots:
                if s.pose == "sit":
                    self.goal_only.add((s.x, s.y))
                    self.blocked.discard((s.x, s.y))  # e.g. couch seats: you can sit there, not walk through

    # --- geometry -----------------------------------------------------------------------
    def room_char(self, x: int, y: int) -> str:
        if 0 <= y < HEIGHT and 0 <= x < WIDTH:
            return MAP[y][x]
        return "#"

    def room_of(self, pos: tuple[int, int]) -> str | None:
        c = self.room_char(*pos)
        return ROOMS[c]["id"] if c in ROOMS else None

    def walkable(self, x: int, y: int) -> bool:
        return self.room_char(x, y) in ROOMS and (x, y) not in self.blocked

    def passable(self, x: int, y: int) -> bool:
        return self.walkable(x, y) and (x, y) not in self.goal_only

    def neighbors(self, x: int, y: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yield x + dx, y + dy

    def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """4-directional A*. Seats and the door can be a destination but never a shortcut."""
        if start == goal:
            return [start]
        if not self.walkable(*goal):
            return None
        frontier = [(0, 0, start)]
        came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost = {start: 0}
        tie = 0
        while frontier:
            _, _, cur = heapq.heappop(frontier)
            if cur == goal:
                break
            for nxt in self.neighbors(*cur):
                if nxt != goal and not self.passable(*nxt):
                    continue
                new_cost = cost[cur] + 1
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    tie += 1
                    h = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                    heapq.heappush(frontier, (new_cost + h, tie, nxt))
                    came[nxt] = cur
        if goal not in came:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        return path[::-1]

    def tiles_near(self, pos: tuple[int, int], avoid: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """Standing tiles next to a position (for walking up to someone), closest first."""
        x, y = pos
        ring = [(x, y + 1), (x - 1, y), (x + 1, y), (x, y - 1),
                (x - 1, y + 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]
        return [t for t in ring if self.passable(*t) and t not in avoid]

    def room_tiles(self, room_id: str) -> list[tuple[int, int]]:
        return [(x, y) for y in range(HEIGHT) for x in range(WIDTH)
                if self.room_of((x, y)) == room_id and self.passable(x, y)]

    def surfaces_near(self, pos: tuple[int, int]) -> list[OfficeObject]:
        """Surfaces you could put something on from where you stand (adjacent or your own seat)."""
        x, y = pos
        near = []
        for o in self.objects.values():
            if not o.surface:
                continue
            if any(abs(tx - x) + abs(ty - y) <= 1 for tx, ty in o.tiles):
                near.append(o)
        return near

    def describe(self) -> str:
        """What a new hire is told about the office."""
        lines = [
            "- The work area: four desks (Desk 1–4), each with a computer that shows the question queue; "
            "the printer; a wall screen (the question board).",
            "- The break room: the coffee machine, the fridge, a kitchen counter, a couch and a coffee table.",
            "- The meeting room: a meeting table with four seats and a whiteboard.",
            "- The entrance: the front door and a coat rack.",
            "- The shared drive (/shared/…) is readable and writable by everyone. "
            "Each employee also has a private folder (/private/<name>/…) that only they can open.",
        ]
        return "\n".join(lines)

    def layout(self) -> dict:
        return {"tile": TILE, "width": WIDTH, "height": HEIGHT, "map": MAP,
                "rooms": {c: r for c, r in ROOMS.items()}, "door": DOOR, "spawn": SPAWN,
                "objects": [o.to_dict() for o in self.objects.values()]}
