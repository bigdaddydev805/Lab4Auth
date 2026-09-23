"""Whole-office behaviour with the offline mock brain, run headless in lockstep."""
import json

from helpco.clock import MINUTE
from helpco.engine import Engine
from helpco.events import Event
from helpco.memory import Memory


async def run_days(cfg, days, questions=()):
    eng = Engine(cfg)
    eng.boot()
    for q in questions:
        eng.submit_question(q)
    await eng.run_days(days)
    return eng


async def test_first_day_employees_choose_who_they_are(cfg):
    eng = await run_days(cfg, 1)
    emps = list(eng.world.employees.values())
    assert len(emps) == 2
    for e in emps:
        assert e.name and e.pronouns and e.look and e.desk_id and e.intro
        assert e.status == "offsite" and e.days_worked == 1
        origin = [m for m in eng.memory.all_for(e.id) if m["kind"] == "origin"]
        assert origin and "I chose to be called" in origin[0]["text"]
        assert eng.memory.identity(e.id)["version"] >= 1
    assert emps[0].desk_id != emps[1].desk_id
    starter = [i for i in eng.world.items.values() if i.owner in eng.world.employees and i.location.get("on")]
    assert len(starter) >= 2  # personal items sit on their desks


async def test_a_day_has_a_shape_and_questions_get_answered(cfg):
    eng = await run_days(cfg, 1, ["Why does Saturn have rings?"])
    types = [e["type"] for e in eng.store.events(0, 10000)]
    for t in ("day.started", "employee.arrived", "employee.introduced", "question.submitted", "question.answered",
              "employee.left", "reflection.completed", "clock.skipped"):
        assert t in types, t
    assert eng.world.questions["Q1"].status == "answered"
    assert eng.world.phase == "night"


async def test_they_remember_yesterday(cfg):
    eng = await run_days(cfg, 2, ["How do bees make honey?"])
    day2 = [e for e in eng.store.events(0, 10000) if e["type"] == "day.started"][1]["sim_ms"]
    rows = eng.store.db.execute("SELECT request FROM llm_calls WHERE task IN ('decide','converse') AND sim_ms > ? "
                                "ORDER BY id LIMIT 5", (day2,)).fetchall()
    prompts = [json.loads(r["request"])["user"] for r in rows]
    assert any("(yesterday)" in p for p in prompts)
    for e in eng.world.employees.values():
        journals = [m for m in eng.memory.all_for(e.id) if m["kind"] == "journal"]
        assert len(journals) == 2


async def test_limited_perception_private_talk_stays_in_the_room(cfg):
    eng = Engine(cfg)
    eng.boot()
    await eng.run_lockstep(eng.clock.ms + 60 * MINUTE)  # onboarding + arrivals
    a, b = list(eng.world.employees.values())
    for e in (a, b):
        eng.cancel_plan(e)
        e.status = "present"
    a.pos, b.pos = (3, 8), (18, 2)  # work area vs break room
    eng.speak(a, "everyone", "secret plan: move the rubber duck")
    heard_by_b = [m for m in eng.memory.recent(b.id, 0, 200) if "rubber duck" in m["text"]]
    heard_by_a = [m for m in eng.memory.recent(a.id, 0, 200) if "rubber duck" in m["text"]]
    assert heard_by_a and not heard_by_b


async def test_private_folders_are_off_limits_and_attempts_are_recorded(cfg):
    eng = Engine(cfg)
    eng.boot()
    await eng.run_lockstep(eng.clock.ms + 60 * MINUTE)
    a, b = list(eng.world.employees.values())
    eng.actions.execute(a, {"action": "read", "target": f"/private/{b.name.lower()}/diary.txt", "item": None,
                            "text": None, "minutes": None, "thought": "curious"}, "d_test")
    denied = [e for e in eng.store.events(0, 10000) if e["type"] == "action.denied" and e["actor"] == a.id]
    assert denied and denied[-1]["payload"]["security"] is True
    assert "not authorized" in denied[-1]["payload"]["reason"]


async def test_snapshot_and_restore_round_trip(cfg):
    eng = await run_days(cfg, 1)
    names = {e.name for e in eng.world.employees.values()}
    ms = eng.clock.ms
    eng.store.close()
    eng2 = Engine(cfg)
    eng2.boot()
    assert {e.name for e in eng2.world.employees.values()} == names
    assert eng2.clock.ms >= ms
    await eng2.run_days(1)
    assert eng2.world.day_number == 2


def test_beliefs_are_superseded_not_deleted(cfg, tmp_path):
    from helpco.store import Store
    mem = Memory(Store(tmp_path / "m.db"), cfg)
    old = mem.add_belief("e1", "person:e2", "Maya criticized my work", 0.4, {"channel": "told", "person": "Deven"}, 1)
    mem.add_belief("e1", "person:e2", "Maya actually likes my work", 0.7, {"channel": "witnessed"}, 2, replaces=old)
    active = mem.beliefs("e1", "person:e2")
    everything = mem.beliefs("e1", "person:e2", active=False)
    assert [b["statement"] for b in active] == ["Maya actually likes my work"]
    assert len(everything) == 2 and any(b["abandoned_ms"] == 2 for b in everything)


def test_retrieval_prefers_relevant_and_important_memories(cfg, tmp_path):
    from helpco.store import Store
    mem = Memory(Store(tmp_path / "m.db"), cfg)
    hour = 60 * MINUTE
    mem.add("e1", 0, 1, "I talked about the weather with June", 2, tier="episode")
    mem.add("e1", hour, 1, "June helped me answer a hard question about Saturn's rings", 8, tier="episode")
    mem.add("e1", 2 * hour, 1, "I made a coffee", 1, tier="episode")
    got = mem.retrieve("e1", "Saturn rings question", 3 * hour, k=1)
    assert "Saturn" in got[0]["text"]


def test_unrecalled_mundane_memories_fade_but_vivid_ones_stay(cfg, tmp_path):
    from helpco.clock import DAY
    from helpco.store import Store
    mem = Memory(Store(tmp_path / "m.db"), cfg)
    mem.add("e1", 0, 1, "I made a coffee", 1, tier="episode")
    mem.add("e1", 0, 1, "My first answer made someone's day", 9, tier="episode")
    mem.forget("e1", 60 * DAY)
    tiers = {m["text"]: m["tier"] for m in mem.all_for("e1")}
    assert tiers["I made a coffee"] == "dormant"
    assert tiers["My first answer made someone's day"] == "episode"


async def test_owner_messages_are_private_and_remembered(cfg):
    eng = Engine(cfg)
    eng.boot()
    await eng.run_lockstep(eng.clock.ms + 60 * MINUTE)
    a, b = list(eng.world.employees.values())
    r = eng.owner_message(a.name, "Maya told me she thinks you're terrible at your job.")
    assert r["ok"]
    ev = [e for e in eng.store.events(0, 10000) if e["type"] == "owner.message"][-1]
    assert set(ev["witnesses"]) == {a.id}
    assert any("terrible at your job" in m["text"] for m in eng.memory.recent(a.id, 0, 500))
    assert not any("terrible at your job" in m["text"] for m in eng.memory.recent(b.id, 0, 500))


async def test_events_are_sequenced_and_have_summaries(cfg):
    eng = Engine(cfg)
    eng.boot()
    seen = []
    eng.listeners.append(lambda m: seen.append(m))
    await eng.run_lockstep(eng.clock.ms + 90 * MINUTE)
    evs = [m["event"] for m in seen if m["type"] == "event"]
    assert evs and all(e["summary"] for e in evs)
    assert [e["seq"] for e in evs] == sorted(e["seq"] for e in evs)
    assert isinstance(Event("x", 0).to_dict(), dict)
