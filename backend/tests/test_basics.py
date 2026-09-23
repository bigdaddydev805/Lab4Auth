from datetime import date, time
from zoneinfo import ZoneInfo

from helpco import catalog
from helpco.clock import MINUTE, SimClock, resolve_start
from helpco.llm.schema import action_schema, extract_json, reflect_schema, validate, wire_schema
from helpco.office import DOOR, SPAWN, Office

TZ = ZoneInfo("America/Los_Angeles")


def test_clock_formats_like_a_person_would_say_it():
    start = resolve_start("2027-03-16", TZ, time(10, 38))
    c = SimClock(start, TZ)
    assert c.fmt_date() == "Tuesday, March 16, 2027"
    assert c.fmt_time() == "10:38 AM"
    assert c.fmt_ago(c.ms - 47 * MINUTE) == "47 minutes ago"
    assert c.relative_day(c.ms - 24 * 60 * MINUTE) == "yesterday"
    assert c.season() == "spring"


def test_calendar_skips_weekends():
    c = SimClock(0, TZ)
    friday = date(2027, 3, 19)
    assert c.next_workday(friday) == date(2027, 3, 22)
    assert not c.is_workday(date(2027, 3, 20))


def test_pathfinding_reaches_every_use_spot_and_seats_are_not_shortcuts():
    o = Office()
    for obj in o.objects.values():
        for s in obj.spots:
            assert o.find_path(SPAWN, (s.x, s.y)) is not None, (obj.id, s)
    assert o.find_path(SPAWN, DOOR) is not None
    path = o.find_path((17, 4), (19, 4))  # around the coffee table, not across couch seats
    assert all(t not in o.goal_only for t in path[1:-1])


def test_rooms_define_earshot():
    o = Office()
    assert o.room_of((3, 8)) == "work_area"
    assert o.room_of((18, 1)) == "break_room"
    assert o.room_of((3, 4)) == "meeting_room"
    assert o.room_of(SPAWN) == "entrance"


def test_appearance_is_validated_against_the_catalog():
    look, errors = catalog.validate_appearance({"skin": "tan", "hair_style": "bob", "hair_color": "Grey",
                                               "top_color": "teal", "pants_color": "navy", "shoes_color": "white",
                                               "accessories": ["frog hat", "jetpack"]})
    assert look["hair_color"] == "gray"
    assert look["accessories"] == ["frog_hat"]
    assert any("jetpack" in e for e in errors)


def test_action_validation_is_forgiving_but_strict_about_enums():
    ok, errs = validate({"thought": "x" * 999, "action": "USE", "target": "coffee machine", "item": None,
                         "text": None, "minutes": "500", "extra": 1}, action_schema())
    assert not errs
    assert ok["action"] == "use" and ok["minutes"] == 120 and len(ok["thought"]) == 400 and "extra" not in ok
    _, errs = validate({"thought": "", "action": "teleport", "target": None, "item": None, "text": None,
                        "minutes": None}, action_schema())
    assert errs


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}


def test_wire_schema_drops_limits_but_keeps_structure():
    w = wire_schema(reflect_schema())
    assert "maxLength" not in str(w) and "maximum" not in str(w)
    assert w["required"] == list(w["properties"])
