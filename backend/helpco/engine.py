"""The World Engine — the Game Master. The only thing that changes reality.

It owns the clock and the schedule, validates and executes intentions (via Actions), decides who
perceives what (witnesses), runs the day cycle (arrive → work → leave → reflect → sleep), and
records every meaningful thing as an event.

Two ways to run:
- realtime: the clock follows wall time × scale; employees keep existing while others think.
- lockstep: headless and reproducible; time only advances when nobody is waiting on a model.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

from . import __version__, catalog, narrate
from .clock import DAY, MINUTE, SimClock, parse_hhmm, resolve_start, resolve_tz
from .events import Event
from .llm import Gateway
from .memory import Memory
from .office import ROOM_NAMES, SPAWN, Office
from .state import Activity, Conversation, Employee, FileDoc, Item, Question, World
from .store import Store

log = logging.getLogger("helpco.engine")

WALK_TILES_PER_REAL_SECOND = 3.0
CONVERSATION_IDLE_MS = 4 * MINUTE
CONVERSATION_MAX_TURNS = 14
NEEDS_TICK_MS = 15 * MINUTE


@dataclass
class Plan:
    id: int
    verb: str
    label: str
    steps: list
    idx: int = 0
    decision_id: str | None = None
    source: str = "llm"
    reserved: list = field(default_factory=list)


class Engine:
    def __init__(self, cfg, store: Store | None = None, gateway: Gateway | None = None):
        from .actions import Actions
        from .cognition import Cognition
        from .reflection import Reflection

        self.cfg = cfg
        self.office = Office()
        self.store = store or Store(cfg.save_dir / "world.db")
        self.memory = Memory(self.store, cfg)
        self.world_id = str(cfg.get("world.name", "HelpCo"))
        self.seed = cfg.get("world.seed", 7)
        self.gateway = gateway or Gateway(cfg, self.store, self.world_id)
        self.times = {k: parse_hhmm(cfg.get(f"office.{k}")) for k in
                      ("day_starts", "workday_start", "lunch", "workday_end", "earliest_leave", "closes")}
        tz = resolve_tz(cfg.get("world.timezone", "local"))
        start_ms = resolve_start(str(cfg.get("world.start", "today")), tz, self.times["day_starts"])
        self.clock = SimClock(start_ms, tz, cfg.get("clock.scale", 20.0), cfg.get("clock.mode", "realtime"))
        self.conversation_scale = float(cfg.get("clock.conversation_scale", 3.0))
        self.world = World()
        self.rng = random.Random(f"{self.seed}:boot")
        self.queue: list = []
        self._qseq = itertools.count()
        self._plan_ids = itertools.count(1)
        self.plans: dict[str, Plan] = {}
        self.reserved: dict[tuple, str] = {}
        self.deciding: set[str] = set()
        self.reasons: dict[str, list] = defaultdict(list)
        self.denial_streak: dict[str, int] = defaultdict(int)
        self.inflight: set[asyncio.Task] = set()
        self.ready: list[tuple] = []
        self._ready_seq = itertools.count()
        self.inputs: list[tuple[dict, asyncio.Future]] = []
        self.listeners: list[Callable[[dict], None]] = []
        self.present_today: set[str] = set()
        self.night_started = False
        self.running = False
        self.last_real = time.monotonic()
        self.actions = Actions(self)
        self.cognition = Cognition(self)
        self.reflection = Reflection(self)

    # ======================================================================= setup / persistence
    def boot(self) -> None:
        snap = self.store.latest_snapshot()
        if snap:
            self._restore(snap)
            log.info("restored world at %s (day %s)", self.clock.fmt_date(), self.world.day_number)
        else:
            self.store.set_meta("created", {"seed": self.seed, "start_ms": self.clock.ms, "version": __version__})
            for _ in range(int(self.cfg.get("employees.initial", 2))):
                self.hire(announce=False)
            self._schedule_day(self.clock.today(), include_today=True)
            self.store.commit()

    def _restore(self, snap: dict) -> None:
        st = snap["state"]
        self.world = World.from_dict(st["world"])
        self.clock.ms = st["clock_ms"]
        self.present_today = set(st.get("present_today", []))
        self.night_started = False
        today = self.clock.today()
        phase = self.world.phase
        if phase in ("night", "evening"):
            self.world.phase = "night"
            self._schedule_next_day()
            return
        # mid-day restore: rebuild the schedule for the rest of today, and let everyone re-decide
        for e in self.world.employees.values():
            e.path, e.thinking = [], False
            if e.status == "present":
                self._set_idle(e)
        self._schedule_day_events(today, only_future=True)
        for e in self.world.present():
            self.request_decision(e, "resumed")
        for e in self.world.employees.values():
            if e.status == "offsite" and e.left_ms is None and self.clock.ms < self._ms(today, "workday_end"):
                self.at(self.clock.ms + MINUTE, "arrive", emp=e.id)

    def snapshot(self, kind: str = "manual") -> None:
        state = {"world": self.world.to_dict(), "clock_ms": self.clock.ms, "present_today": sorted(self.present_today)}
        self.store.save_snapshot(self.store.last_seq(), self.clock.ms, kind, state, __version__)

    # ======================================================================= scheduling
    def at(self, ms: int, kind: str, **data) -> None:
        heapq.heappush(self.queue, (int(ms), next(self._qseq), kind, data))

    def _ms(self, d, key: str) -> int:
        return self.clock.ms_of(d, self.times[key])

    def _schedule_day(self, d, include_today: bool = False) -> None:
        if include_today and self.clock.is_workday(d) and self.clock.ms <= self._ms(d, "day_starts"):
            day = d
        else:
            day = self.clock.next_workday(d, include_today=False)
        self.at(self._ms(day, "day_starts"), "day_start", date=day.isoformat())

    def _schedule_day_events(self, d, only_future: bool = False) -> None:
        for key, kind in (("lunch", "lunch"), ("workday_end", "workday_end"), ("closes", "closes")):
            ms = self._ms(d, key)
            if not only_future or ms > self.clock.ms:
                self.at(ms, kind, date=d.isoformat())
        self.at(self.clock.ms + NEEDS_TICK_MS, "needs_tick", date=d.isoformat())

    def _stale(self, date: str | None) -> bool:
        """Day-scoped events (lunch, closing, ...) from a day that's already over are ignored."""
        return date is not None and date != self.clock.today().isoformat()

    def _schedule_next_day(self) -> None:
        today = self.clock.today()
        nxt = self.clock.next_workday(today)
        start = self._ms(nxt, "day_starts")
        if self.cfg.get("clock.skip_nights", True):
            skipped_weekend = (nxt - today).days > 1
            self.emit(Event("clock.skipped", self.clock.ms, payload={
                "reason": "the weekend" if skipped_weekend else "overnight", "to_ms": start - MINUTE},
                visibility="system"))
            self.clock.ms = max(self.clock.ms, start - MINUTE)
        self.at(start, "day_start", date=nxt.isoformat())

    # ======================================================================= main loops
    def effective_scale(self) -> float:
        if any(c.ended_ms is None for c in self.world.conversations.values()):
            return min(self.clock.scale, self.conversation_scale)
        return self.clock.scale

    async def step_to(self, target_ms: int) -> None:
        while self.queue and self.queue[0][0] <= target_ms:
            ms, _, kind, data = heapq.heappop(self.queue)
            if ms > self.clock.ms:
                self.clock.ms = ms
            try:
                await self._dispatch(kind, data)
            except Exception:  # never let one bad handler stop the office
                log.exception("handler %s failed", kind)
            self._process_ready()
        if target_ms > self.clock.ms:
            self.clock.ms = target_ms

    async def tick_realtime(self) -> None:
        now = time.monotonic()
        dt = now - self.last_real
        self.last_real = now
        await self._drain_inputs()
        self._process_ready()
        if self.clock.paused:
            return
        await self.step_to(self.clock.ms + int(dt * 1000 * self.effective_scale()))
        self.store.commit()

    async def run_realtime(self, stop: asyncio.Event | None = None) -> None:
        self.running = True
        self.last_real = time.monotonic()
        while self.running and not (stop and stop.is_set()):
            await asyncio.sleep(0.05)
            await self.tick_realtime()

    async def run_lockstep(self, until_ms: int) -> None:
        """Deterministic, as-fast-as-possible run (LLM calls still take their real time)."""
        self.running = True
        while self.running and self.clock.ms < until_ms:
            await self._drain_inputs()
            if self.inflight:
                await asyncio.gather(*list(self.inflight), return_exceptions=True)
                self._process_ready()
                continue
            self._process_ready()
            if self.inflight:
                continue
            if not self.queue:
                break
            nxt = self.queue[0][0]
            if nxt > until_ms:
                self.clock.ms = until_ms
                break
            await self.step_to(nxt)
        self.store.commit()

    async def run_days(self, days: int) -> None:
        start_day = self.world.day_number
        while self.world.day_number < start_day + days or self.world.phase not in ("night",):
            before = self.clock.ms
            await self.run_lockstep(self.clock.ms + 6 * 60 * MINUTE)
            if self.clock.ms == before and not self.queue and not self.inflight:
                break
            if self.world.day_number >= start_day + days and self.world.phase == "night":
                break

    # ======================================================================= async work (LLM calls)
    def spawn(self, coro, on_done: Callable[[Any], None], order_key: str = "") -> None:
        task = asyncio.ensure_future(coro)
        self.inflight.add(task)

        def _done(t: asyncio.Task):
            self.inflight.discard(t)
            try:
                result = t.result()
            except Exception as e:  # surfaced to the callback as an exception object
                log.exception("background task failed")
                result = e
            self.ready.append((order_key, next(self._ready_seq), on_done, result))

        task.add_done_callback(_done)

    def _process_ready(self) -> None:
        while self.ready:
            self.ready.sort(key=lambda r: (r[0], r[1]))
            _, _, fn, result = self.ready.pop(0)
            try:
                fn(result)
            except Exception:
                log.exception("completion handler failed")

    # ======================================================================= events & perception
    def name_of(self, emp_id: str | None) -> str:
        if emp_id in (None, ""):
            return "someone"
        if emp_id == "owner":
            return "the owner"
        if emp_id == "everyone":
            return "everyone"
        e = self.world.employees.get(emp_id)
        return e.display if e else str(emp_id)

    def current_tile(self, e: Employee, ms: int | None = None) -> tuple[int, int]:
        ms = self.clock.ms if ms is None else ms
        if e.path and e.ms_per_tile:
            i = int((ms - e.path_t0) // e.ms_per_tile)
            i = max(0, min(i, len(e.path) - 1))
            return tuple(e.path[i])
        return e.pos

    def room_of(self, e: Employee) -> str | None:
        if e.status != "present":
            return None
        return self.office.room_of(self.current_tile(e))

    def people_in(self, room: str | None, exclude: str | None = None) -> list[Employee]:
        return [e for e in self.world.present() if e.id != exclude and self.room_of(e) == room]

    def emit(self, ev: Event, witnesses: dict | None = None, remember: bool = True) -> Event:
        """Record an event, decide who perceived it, write their memories, and tell the clients."""
        if witnesses is None:
            witnesses = {}
            if ev.visibility == "public" and ev.room:
                for e in self.people_in(ev.room):
                    witnesses[e.id] = "saw"
            if ev.actor in self.world.employees:
                witnesses[ev.actor] = "did"
        ev.witnesses = witnesses
        ev.seq = self.store.append_event(ev)
        if remember:
            for emp_id in witnesses:
                if emp_id not in self.world.employees:
                    continue
                text, importance = narrate.experience(self, ev, emp_id)
                if text:
                    people = [p for p in {ev.actor, *ev.targets, ev.payload.get("to")}
                              if p and p != emp_id and p in self.world.employees]
                    prov = {"channel": {"did": "did", "saw": "witnessed", "heard": "heard", "told": "told",
                                        "read": "read"}.get(witnesses[emp_id], "witnessed"),
                            "source": ev.actor}
                    self.memory.add(emp_id, ev.sim_ms, self.world.day_number, text, importance,
                                    people=people, provenance=prov, event_seq=ev.seq)
                if ev.actor and ev.actor != emp_id and ev.actor in self.world.employees:
                    self.world.employees[emp_id].last_seen[ev.actor] = [ev.sim_ms, ev.room]
        d = ev.to_dict()
        d["summary"] = narrate.summary(self, ev)
        self.broadcast({"type": "event", "event": d})
        return ev

    def broadcast(self, msg: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception:
                log.exception("listener failed")

    def push_employee(self, e: Employee) -> None:
        self.broadcast({"type": "employee", "employee": e.public()})

    def push_item(self, it: Item) -> None:
        self.broadcast({"type": "item", "item": it.to_dict()})

    def push_question(self, q: Question) -> None:
        self.broadcast({"type": "question", "question": q.public()})

    # ======================================================================= dispatch
    async def _dispatch(self, kind: str, data: dict) -> None:
        handler = getattr(self, f"_on_{kind}", None)
        if handler is None:
            log.warning("no handler for %s", kind)
            return
        res = handler(**data)
        if asyncio.iscoroutine(res):
            await res

    # --- day cycle ---------------------------------------------------------------------------
    def _on_day_start(self, date: str) -> None:
        d = self.clock.dt().date().fromisoformat(date)
        self.world.day_number += 1
        self.world.phase = "morning"
        self.night_started = False
        self.present_today = set()
        self.rng = random.Random(f"{self.seed}:day:{self.world.day_number}")
        for c in self.world.conversations.values():
            c.ended_ms = c.ended_ms or self.clock.ms
        self.world.conversations = {}
        self.emit(Event("day.started", self.clock.ms, payload={
            "day": self.world.day_number, "date": self.clock.fmt_date(), "season": self.clock.season()},
            visibility="system"))
        for e in sorted(self.world.employees.values(), key=lambda x: x.hire_no):
            if e.status == "departed":
                continue
            e.left_ms = None
            if e.status == "hired":
                self.cognition.onboard(e)
            arrive = self._ms(d, "workday_start") - 20 * MINUTE + self.rng.randint(0, 35) * MINUTE
            self.at(max(arrive, self.clock.ms + MINUTE), "arrive", emp=e.id)
        self._schedule_day_events(d)
        self.snapshot("morning")

    def _on_arrive(self, emp: str) -> None:
        e = self.world.employees[emp]
        if e.status == "hired":  # still choosing who to be
            self.at(self.clock.ms + 2 * MINUTE, "arrive", emp=emp)
            return
        if e.status == "present" or e.status == "departed":
            return
        if self.clock.ms >= self._ms(self.clock.today(), "closes"):
            return
        e.status = "present"
        e.pos, e.facing, e.path = SPAWN, "up", []
        e.arrived_ms = self.clock.ms
        e.holding = None
        e.needs = {"energy": round(0.85 + self.rng.random() * 0.15, 2), "hunger": 0.1,
                   "social": min(1.0, e.needs.get("social", 0.3) + 0.1)}
        e.stats["coffees_today"] = 0
        self.present_today.add(e.id)
        self.world.phase = "workday"
        self._set_idle(e)
        self.emit(Event("employee.arrived", self.clock.ms, actor=e.id, room="entrance"))
        self.push_employee(e)
        self._check_introductions(e)
        self.request_decision(e, "arrived")

    def _on_lunch(self, date: str | None = None) -> None:
        if self._stale(date):
            return
        self.emit(Event("office.lunch", self.clock.ms, payload={"text": "It's lunchtime."}, visibility="system"),
                  remember=False)

    def _on_workday_end(self, date: str | None = None) -> None:
        if self._stale(date):
            return
        self.emit(Event("office.workday_end", self.clock.ms, visibility="system"), remember=False)
        for e in self.world.present():
            self.request_decision(e, "workday_over")

    def _on_closes(self, date: str | None = None) -> None:
        if self._stale(date):
            return
        for e in self.world.present():
            plan = self.plans.get(e.id)
            if not (plan and plan.verb == "leave_office"):
                self.actions.force_leave(e, "The office is closing for the night.")
        self.at(self.clock.ms + 45 * MINUTE, "night_check", date=date)

    def _on_night_check(self, date: str | None = None) -> None:
        if self._stale(date):
            return
        for e in self.world.present():  # anyone somehow still here is walked out
            self.cancel_plan(e)
            self._depart(e)
        self._maybe_start_night()

    def _on_needs_tick(self, date: str | None = None) -> None:
        if self._stale(date) or self.world.phase not in ("morning", "workday"):
            return
        for e in self.world.present():
            kind = e.activity.kind
            n = e.needs
            n["energy"] = max(0.0, n["energy"] - (0.025 if kind in ("work", "question") else 0.012))
            n["hunger"] = min(1.0, n["hunger"] + 0.03)
            talking = any(e.id in c.participants and c.ended_ms is None for c in self.world.conversations.values())
            n["social"] = max(0.0, n["social"] - 0.1) if talking else min(1.0, n["social"] + 0.02)
        self.at(self.clock.ms + NEEDS_TICK_MS, "needs_tick", date=date)

    def _maybe_start_night(self) -> None:
        if self.night_started or self.world.phase in ("night", "evening"):
            return
        if self.world.present() or self.clock.ms < self._ms(self.clock.today(), "earliest_leave"):
            return
        if any(e.status == "hired" for e in self.world.employees.values()):
            return
        self.night_started = True
        self.world.phase = "evening"
        self.emit(Event("day.ended", self.clock.ms, payload={"day": self.world.day_number}, visibility="system"),
                  remember=False)
        workers = [self.world.employees[i] for i in sorted(self.present_today)]
        self.reflection.run_night(workers, self._on_night_done)

    def _on_night_done(self) -> None:
        self.world.phase = "night"
        self.memory.commit()
        self.snapshot("night")
        self._schedule_next_day()

    # --- movement ----------------------------------------------------------------------------
    def walk_ms_per_tile(self) -> int:
        scale = self.cfg.get("clock.scale", 20.0) if self.clock.mode == "lockstep" else self.effective_scale()
        return max(200, int(1000 / WALK_TILES_PER_REAL_SECOND * scale))

    def start_walk(self, e: Employee, goal: tuple[int, int], plan: Plan, facing: str | None = None) -> bool | None:
        """Returns True if walking, False if already there, None if unreachable."""
        start = self.current_tile(e)
        if e.path:
            e.pos, e.path = start, []
        path = self.office.find_path(start, goal)
        if path is None:
            return None
        if len(path) == 1:
            e.pos = goal
            if facing:
                e.facing = facing
            return False
        e.path = [list(p) for p in path]
        e.path_t0 = self.clock.ms
        e.ms_per_tile = self.walk_ms_per_tile()
        label = plan.label if plan.label.startswith("going") else f"on the way ({plan.label})"
        e.activity = Activity("walk", label, "walk", self.clock.ms,
                              self.clock.ms + (len(path) - 1) * e.ms_per_tile, target=plan.label)
        room = self.office.room_of(start)
        for i, tile in enumerate(path[1:], start=1):
            r = self.office.room_of(tuple(tile))
            if r != room:
                self.at(e.path_t0 + i * e.ms_per_tile, "room_enter", emp=e.id, plan=plan.id, room=r)
                room = r
        self.at(e.path_t0 + (len(path) - 1) * e.ms_per_tile, "move_done", emp=e.id, plan=plan.id,
                goal=list(goal), facing=facing)
        self.push_employee(e)
        return True

    def _on_room_enter(self, emp: str, plan: int, room: str) -> None:
        e = self.world.employees.get(emp)
        p = self.plans.get(emp)
        if not e or not p or p.id != plan or e.status != "present":
            return
        self.emit(Event("employee.entered", self.clock.ms, actor=e.id, room=room), remember=False)
        if room == "work_area":
            self.cognition.saw_question_board(e)
        self._check_introductions(e)

    def _on_move_done(self, emp: str, plan: int, goal: list, facing: str | None) -> None:
        e = self.world.employees.get(emp)
        p = self.plans.get(emp)
        if not e or not p or p.id != plan:
            return
        e.pos = tuple(goal)
        if facing:
            e.facing = facing
        elif len(e.path) >= 2:
            (x0, y0), (x1, y1) = e.path[-2], e.path[-1]
            e.facing = "right" if x1 > x0 else "left" if x1 < x0 else "down" if y1 > y0 else "up"
        e.path = []
        self.advance_plan(e)

    # --- plans & activities -------------------------------------------------------------------
    def new_plan(self, verb: str, label: str, steps: list, decision_id: str | None = None,
                 source: str = "llm") -> Plan:
        return Plan(next(self._plan_ids), verb, label, steps, decision_id=decision_id, source=source)

    def start_plan(self, e: Employee, plan: Plan) -> None:
        self.cancel_plan(e)
        self.plans[e.id] = plan
        for spot in plan.reserved:
            self.reserved[spot] = e.id
        self.emit(Event("action.started", self.clock.ms, actor=e.id, room=self.room_of(e),
                        payload={"verb": plan.verb, "label": plan.label}, source=plan.source,
                        decision_id=plan.decision_id), remember=False)
        self.advance_plan(e)

    def cancel_plan(self, e: Employee) -> None:
        plan = self.plans.pop(e.id, None)
        if e.path:
            e.pos = self.current_tile(e)
            e.path = []
        for spot, who in list(self.reserved.items()):
            if who == e.id and (plan is None or spot in plan.reserved):
                del self.reserved[spot]
        if plan is not None and e.activity.kind not in ("idle",):
            self._set_idle(e)

    def advance_plan(self, e: Employee) -> None:
        plan = self.plans.get(e.id)
        while plan is not None and plan.idx < len(plan.steps):
            step = plan.steps[plan.idx]
            plan.idx += 1
            kind = step["type"]
            if kind == "walk":
                goal = tuple(step["to"]) if step.get("to") else step["to_fn"]()
                if goal is None:
                    self.deny(e, plan.verb, plan.label, step.get("fail", "you couldn't find a way there"))
                    return
                res = self.start_walk(e, goal, plan, step.get("facing"))
                if res is None:
                    self.deny(e, plan.verb, plan.label, "there's no way to get there from here")
                    return
                if res:
                    return
            elif kind == "do":
                self._start_activity(e, plan, step)
                return
            elif kind == "call":
                ok = step["fn"](e, **step.get("args", {}))
                if ok is False or self.plans.get(e.id) is not plan:
                    return
            plan = self.plans.get(e.id)
        if plan is not None:
            self.plans.pop(e.id, None)
            self._plan_finished(e, plan)

    def _start_activity(self, e: Employee, plan: Plan, step: dict) -> None:
        if "seconds" in step:
            until = self.clock.ms + max(1, int(step["seconds"])) * 1000
        else:
            until = self.clock.ms + max(1, int(step.get("minutes", 5))) * MINUTE
        if step.get("facing"):
            e.facing = step["facing"]
        e.activity = Activity(step["kind"], step["label"], step.get("pose", "stand"), self.clock.ms, until,
                              target=step.get("target"), data={"plan": plan.id})
        self.at(until, "activity_done", emp=e.id, plan=plan.id, step=plan.idx - 1)
        self.push_employee(e)

    def _on_activity_done(self, emp: str, plan: int, step: int) -> None:
        e = self.world.employees.get(emp)
        p = self.plans.get(emp)
        if not e or not p or p.id != plan:
            return
        st = p.steps[step]
        minutes = int(st.get("minutes", 5))
        if st.get("on_done"):
            st["on_done"](e, minutes)
        if st.get("memory"):
            self.emit(Event("activity.completed", self.clock.ms, actor=e.id, room=self.room_of(e),
                            payload={"kind": st["kind"], "minutes": minutes, "memory": st["memory"].format(m=minutes),
                                     "importance": st.get("importance", 1)}, visibility="private"),
                      witnesses={e.id: "did"})
        if st.get("then") == "wait_async":
            return  # an async step (e.g. writing a draft) will advance the plan when it finishes
        self.advance_plan(e)

    def _plan_finished(self, e: Employee, plan: Plan) -> None:
        for spot, who in list(self.reserved.items()):
            if who == e.id and spot in plan.reserved and tuple(e.pos) != spot:
                del self.reserved[spot]
        if e.status != "present":
            return
        if e.activity.kind not in ("idle",):
            self._set_idle(e)
        self.denial_streak[e.id] = 0
        self.request_decision(e, "finished", verb=plan.verb)

    def _set_idle(self, e: Employee) -> None:
        pose = "stand"
        for o in self.office.objects.values():
            for s in o.spots:
                if (s.x, s.y) == tuple(e.pos) and s.pose == "sit":
                    pose, e.facing = "sit", s.facing
        room = self.office.room_of(tuple(e.pos))
        label = "sitting" if pose == "sit" else "standing"
        e.activity = Activity("idle", f"{label} in {ROOM_NAMES.get(room, 'the office')}", pose, self.clock.ms)
        self.push_employee(e)

    def deny(self, e: Employee, verb: str, target: str | None, reason: str, security: bool = False,
             decision_id: str | None = None) -> None:
        self.cancel_plan(e)
        self.emit(Event("action.denied", self.clock.ms, actor=e.id, room=self.room_of(e),
                        payload={"verb": verb, "target": target, "reason": reason, "security": security},
                        decision_id=decision_id), witnesses={e.id: "did"})
        self.denial_streak[e.id] += 1
        if self.denial_streak[e.id] >= 3:  # stop an LLM from spinning on impossible requests
            self.denial_streak[e.id] = 0
            self.actions.autopilot(e, "Several things didn't work out, so I just got on with my day.")
            return
        self.request_decision(e, "denied", why=reason)

    # --- decisions --------------------------------------------------------------------------------
    def request_decision(self, e: Employee, reason: str, **info) -> None:
        if e.status != "present":
            return
        self.reasons[e.id].append({"reason": reason, "ms": self.clock.ms, **info})
        if e.id in self.deciding:
            return
        self.deciding.add(e.id)
        e.thinking = True
        self.push_employee(e)
        self.cognition.decide(e)

    def decision_done(self, e: Employee) -> None:
        self.deciding.discard(e.id)
        e.thinking = False
        self.push_employee(e)

    # --- conversations -------------------------------------------------------------------------------
    def conversation_for(self, e: Employee, room: str, other: str | None) -> Conversation:
        for c in self.world.conversations.values():
            if c.ended_ms is None and c.room == room and (e.id in c.participants or other in c.participants):
                return c
        cid = self.world.next_id("conv", "c")
        c = Conversation(cid, room, self.clock.ms, self.clock.ms)
        self.world.conversations[cid] = c
        return c

    def speak(self, e: Employee, to: str, text: str, source: str = "llm", decision_id: str | None = None,
              event_type: str = "speech.said", extra: dict | None = None) -> None:
        room = self.room_of(e)
        text = " ".join((text or "").split())[:280]
        conv = self.conversation_for(e, room, to if to in self.world.employees else None)
        for p in (e.id, to):
            if p in self.world.employees and p not in conv.participants:
                conv.participants.append(p)
        conv.turns.append({"speaker": e.id, "to": to, "text": text, "ms": self.clock.ms})
        conv.last_ms = self.clock.ms
        witnesses = {p.id: "heard" for p in self.people_in(room)}
        witnesses[e.id] = "did"
        payload = {"to": to, "text": text, "conversation": conv.id, **(extra or {})}
        self.emit(Event(event_type, self.clock.ms, actor=e.id, room=room, targets=[to], payload=payload,
                        source=source, decision_id=decision_id), witnesses=witnesses)
        e.last_spoke.update({to: self.clock.ms} if to in self.world.employees else {})
        if to in self.world.employees:
            self.world.employees[to].last_spoke[e.id] = self.clock.ms
        e.needs["social"] = max(0.0, e.needs["social"] - 0.15)
        self.at(self.clock.ms + CONVERSATION_IDLE_MS, "conversation_check", conv=conv.id)
        if len(conv.turns) >= CONVERSATION_MAX_TURNS:
            self._end_conversation(conv, "they drifted back to what they were doing")
            return
        listeners = [p for p in self.people_in(room, exclude=e.id)] if to == "everyone" else \
            [self.world.employees[to]] if to in self.world.employees else []
        heard_at = self.clock.ms + self.speech_ms(text)
        for other in listeners:
            if other.status == "present" and event_type == "speech.said":
                self.at(heard_at, "heard", emp=other.id, by=e.id, text=text, conv=conv.id)

    @staticmethod
    def speech_ms(text: str) -> int:
        return 2000 + 60 * len(text or "")

    def _on_heard(self, emp: str, by: str, text: str, conv: str) -> None:
        e = self.world.employees.get(emp)
        c = self.world.conversations.get(conv)
        if e and e.status == "present" and c and c.ended_ms is None:
            self.request_decision(e, "addressed", by=by, text=text, conversation=conv)

    def _on_speak_later(self, emp: str, to: str, text: str, event_type: str, extra: dict) -> None:
        e = self.world.employees.get(emp)
        if e and e.status == "present":
            self.speak(e, to, text, source="schedule", event_type=event_type, extra=extra)

    def _on_nudge(self, emp: str, since: int) -> None:
        """Someone said something and then had nothing else going on: check back in with them."""
        e = self.world.employees.get(emp)
        if e and e.status == "present" and e.id not in self.plans and e.id not in self.deciding:
            self.request_decision(e, "quiet")

    def _on_conversation_check(self, conv: str) -> None:
        c = self.world.conversations.get(conv)
        if c and c.ended_ms is None and self.clock.ms - c.last_ms >= CONVERSATION_IDLE_MS:
            self._end_conversation(c, "quiet")

    def _end_conversation(self, c: Conversation, why: str) -> None:
        if c.ended_ms is not None:
            return
        c.ended_ms = self.clock.ms
        for p in c.participants:
            e = self.world.employees.get(p)
            if e and e.status == "present":
                e.stats["conversations"] = e.stats.get("conversations", 0) + 1
        self.emit(Event("conversation.ended", self.clock.ms, room=c.room,
                        payload={"conversation": c.id, "turns": len(c.turns), "participants": c.participants,
                                 "why": why}, visibility="system"), remember=False)

    def _check_introductions(self, e: Employee) -> None:
        """First time two people share a room, they introduce themselves (their own chosen words)."""
        room = self.room_of(e)
        if e.status != "present" or not e.name:
            return
        for other in self.people_in(room, exclude=e.id):
            if not other.name or other.id in e.met:
                continue
            e.met.append(other.id)
            other.met.append(e.id)
            first = e.intro or f"hi, i'm {e.name}"
            self.speak(e, other.id, first, source="schedule", event_type="employee.introduced",
                       extra={"pronouns": e.pronouns})
            later = self.clock.ms + self.speech_ms(first)
            self.at(later, "speak_later", emp=other.id, to=e.id, text=other.intro or f"hi, i'm {other.name}",
                    event_type="employee.introduced", extra={"pronouns": other.pronouns})
            self.at(later + self.speech_ms(other.intro), "heard", emp=e.id, by=other.id,
                    text=other.intro or f"hi, i'm {other.name}", conv=self.conversation_for(e, room, other.id).id)

    def _depart(self, e: Employee) -> None:
        if e.holding:
            it = self.world.items.get(e.holding)
            if it:
                self.actions.drop_item(e, it, quiet=True)
        self.cancel_plan(e)
        e.status = "offsite"
        e.left_ms = self.clock.ms
        e.days_worked += 1
        e.thinking = False
        self.deciding.discard(e.id)
        self.reasons.pop(e.id, None)
        self.emit(Event("employee.left", self.clock.ms, actor=e.id, room="entrance"))
        e.pos, e.path = (20, 13), []
        self.push_employee(e)
        self._maybe_start_night()

    # ======================================================================= hiring
    def hire(self, announce: bool = True) -> Employee | None:
        active = [e for e in self.world.employees.values() if e.status != "departed"]
        if len(active) >= int(self.cfg.get("employees.max", 4)):
            return None
        eid = self.world.next_id("emp", "emp_")
        e = Employee(id=eid, hire_no=self.world.counters["emp"], hired_ms=self.clock.ms)
        self.world.employees[eid] = e
        self.emit(Event("employee.hired", self.clock.ms, actor=eid, payload={"hire_no": e.hire_no},
                        visibility="system"), remember=False)
        self.push_employee(e)
        if announce and self.world.phase in ("morning", "workday"):
            self.cognition.onboard(e)
            self.at(self.clock.ms + 10 * MINUTE, "arrive", emp=eid)
        return e

    # ======================================================================= player commands
    def enqueue(self, cmd: dict) -> asyncio.Future:
        fut = asyncio.get_event_loop().create_future()
        self.inputs.append((cmd, fut))
        return fut

    async def _drain_inputs(self) -> None:
        while self.inputs:
            cmd, fut = self.inputs.pop(0)
            try:
                result = self.command(cmd)
                if not fut.done():
                    fut.set_result(result)
            except Exception as ex:
                log.exception("command failed: %s", cmd)
                if not fut.done():
                    fut.set_result({"ok": False, "error": str(ex)})
            self.store.commit()

    def command(self, cmd: dict) -> dict:
        kind = cmd.get("type")
        if kind == "submit_question":
            return self.submit_question(str(cmd.get("text", "")))
        if kind == "message":
            return self.owner_message(str(cmd.get("to", "")), str(cmd.get("text", "")))
        if kind == "give":
            return self.owner_gift(str(cmd.get("to", "")), str(cmd.get("kind", "")), cmd.get("label"))
        if kind == "hire":
            e = self.hire()
            return {"ok": bool(e), "id": e.id if e else None, "error": None if e else "all desks are taken"}
        if kind == "clock":
            act = cmd.get("action")
            if act == "pause":
                self.clock.paused = True
            elif act == "resume":
                self.clock.paused = False
                self.last_real = time.monotonic()
            elif act == "scale":
                self.clock.scale = max(0.1, min(600.0, float(cmd.get("value", 20))))
            self.broadcast({"type": "clock", "clock": self.clock_state()})
            return {"ok": True, "clock": self.clock_state()}
        if kind == "snapshot":
            self.snapshot("manual")
            return {"ok": True}
        return {"ok": False, "error": f"unknown command {kind!r}"}

    def submit_question(self, text: str) -> dict:
        text = " ".join(text.split())[:1000]
        if not text:
            return {"ok": False, "error": "empty question"}
        qid = self.world.next_id("q", "Q")
        q = Question(qid, text, self.clock.ms)
        self.world.questions[qid] = q
        witnesses = {e.id: "saw" for e in self.people_in("work_area")}
        self.emit(Event("question.submitted", self.clock.ms, room="work_area", payload={"id": qid, "text": text},
                        source="player"), witnesses=witnesses)
        self.push_question(q)
        for emp_id in witnesses:
            e = self.world.employees[emp_id]
            e.known_questions[qid] = "open"
            if e.activity.kind in ("idle", "work", "wait"):
                self.request_decision(e, "new_question", question=qid)
        return {"ok": True, "id": qid}

    def owner_message(self, to: str, text: str) -> dict:
        e = self.find_employee(to)
        text = " ".join(text.split())[:600]
        if not e or not text:
            return {"ok": False, "error": "unknown employee or empty message"}
        self.emit(Event("owner.message", self.clock.ms, actor="owner", room=self.room_of(e), targets=[e.id],
                        payload={"text": text}, source="player", visibility="private"), witnesses={e.id: "told"})
        if e.status == "present":
            self.request_decision(e, "addressed", by="owner", text=text)
        return {"ok": True, "delivered": e.status == "present"}

    def owner_gift(self, to: str, kind: str, label: str | None) -> dict:
        e = self.find_employee(to)
        if not e:
            return {"ok": False, "error": "unknown employee"}
        kind = kind if kind in catalog.ITEM_KINDS else "rubber_duck"
        it = self.actions.create_item(kind, label or catalog.ITEM_KINDS[kind], owner=e.id, created_by="owner")
        if e.status == "present" and not e.holding:
            it.location = {"held_by": e.id}
            e.holding = it.id
        elif e.desk_id:
            it.location = {"on": e.desk_id}
        else:
            it.location = {"at": list(SPAWN)}
        self.push_item(it)
        self.push_employee(e)
        self.emit(Event("owner.gift", self.clock.ms, actor="owner", room=self.room_of(e), targets=[e.id],
                        payload={"item": it.id, "label": it.label}, source="player", visibility="private"),
                  witnesses={e.id: "told"})
        if e.status == "present":
            self.request_decision(e, "addressed", by="owner", text=f"(gave you {it.label})")
        return {"ok": True, "item": it.id}

    def find_employee(self, ref: str) -> Employee | None:
        if ref in self.world.employees:
            return self.world.employees[ref]
        matches = self.world.by_name(ref)
        return matches[0] if matches else None

    # ======================================================================= client state
    def clock_state(self) -> dict:
        d = self.clock.snapshot()
        d.update({"effective_scale": self.effective_scale(), "date": self.clock.fmt_date(),
                  "time": self.clock.fmt_time(), "day": self.world.day_number, "phase": self.world.phase})
        return d

    def full_state(self) -> dict:
        return {
            "office": self.office.layout(),
            "clock": self.clock_state(),
            "employees": [e.public() for e in sorted(self.world.employees.values(), key=lambda x: x.hire_no)],
            "items": [i.to_dict() for i in self.world.items.values()],
            "questions": [q.public() for q in self.world.questions.values()],
            "whiteboard": self.world.files.get("/whiteboard").text if "/whiteboard" in self.world.files else "",
            "llm": self.gateway.status(),
            "last_seq": self.store.last_seq(),
        }
