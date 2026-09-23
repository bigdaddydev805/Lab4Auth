extends RefCounted
## The client's copy of the world. A `snapshot` replaces everything; `employee`, `item` and
## `question` messages replace one record; events and decisions are passed on for the UI.
## All ids are strings (Godot parses JSON numbers as floats).

const SimClock := preload("res://scripts/model/sim_clock.gd")

signal snapshot_applied
signal employee_changed(emp_id: String)
signal items_changed
signal questions_changed
signal event_added(ev: Dictionary, live: bool)
signal decision_changed(emp_id: String)
signal whiteboard_changed
signal clock_changed
signal ack_received(id: String, result: Dictionary)

var world_name: String = "HelpCo"
var has_snapshot: bool = false
var office: Dictionary = {}
## {"character": {...}, "office": {...}} from the snapshot.
var art_meta: Dictionary = {}
var employees: Dictionary = {}
var items: Dictionary = {}
var questions: Dictionary = {}
## emp_id -> last `decision` dict (thought, action, target, text, ...).
var decisions: Dictionary = {}
var whiteboard: String = ""
var llm: Dictionary = {}
var last_event_seq: int = 0
var clock: SimClock = SimClock.new()


## Applies one server message.
func apply(type: String, payload: Variant) -> void:
	if typeof(payload) != TYPE_DICTIONARY:
		return
	var p: Dictionary = payload
	match type:
		"hello":
			world_name = str(p.get("world", world_name))
		"snapshot":
			_apply_snapshot(p)
		"clock":
			clock.apply(p)
			clock_changed.emit()
		"employee":
			_put_employee(p)
		"item":
			items[str(p.get("id", ""))] = p
			items_changed.emit()
		"question":
			questions[str(p.get("id", ""))] = p
			questions_changed.emit()
		"event":
			add_event(p, true)
		"decision":
			var emp_id: String = str(p.get("employee", ""))
			if typeof(p.get("decision")) == TYPE_DICTIONARY:
				decisions[emp_id] = p["decision"]
				decision_changed.emit(emp_id)
		"whiteboard":
			whiteboard = str(p.get("text", ""))
			whiteboard_changed.emit()
		"ack":
			var result: Variant = p.get("result")
			ack_received.emit(str(p.get("id", "")), result if typeof(result) == TYPE_DICTIONARY else {})


## Records an event (live from the socket, or backfilled over REST) and tells listeners.
func add_event(ev: Dictionary, live: bool) -> void:
	last_event_seq = maxi(last_event_seq, int(ev.get("seq", 0)))
	event_added.emit(ev, live)


## Employees sorted by hire number.
func employee_list() -> Array:
	var out: Array = employees.values()
	out.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return int(a.get("hire_no", 0)) < int(b.get("hire_no", 0)))
	return out


func employee_name(emp_id: String) -> String:
	if emp_id == "owner":
		return "you"
	var e: Dictionary = employees.get(emp_id, {})
	var n: String = str(e.get("name", ""))
	return n if n != "" and n != "<null>" else emp_id


## The item an employee holds, or {}.
func held_item(emp_id: String) -> Dictionary:
	var e: Dictionary = employees.get(emp_id, {})
	var item_id: Variant = e.get("holding")
	if item_id == null:
		return {}
	return items.get(str(item_id), {})


func _apply_snapshot(p: Dictionary) -> void:
	office = p.get("office", {})
	art_meta = p.get("art", {})
	whiteboard = str(p.get("whiteboard", ""))
	llm = p.get("llm", {})
	last_event_seq = int(p.get("last_seq", 0))
	employees.clear()
	for e: Variant in p.get("employees", []):
		if typeof(e) == TYPE_DICTIONARY:
			employees[str(e.get("id", ""))] = e
	items.clear()
	for it: Variant in p.get("items", []):
		if typeof(it) == TYPE_DICTIONARY:
			items[str(it.get("id", ""))] = it
	questions.clear()
	for q: Variant in p.get("questions", []):
		if typeof(q) == TYPE_DICTIONARY:
			questions[str(q.get("id", ""))] = q
	if typeof(p.get("clock")) == TYPE_DICTIONARY:
		clock.apply(p["clock"])
	has_snapshot = true
	snapshot_applied.emit()


func _put_employee(p: Dictionary) -> void:
	var emp_id: String = str(p.get("id", ""))
	if emp_id == "":
		return
	employees[emp_id] = p
	employee_changed.emit(emp_id)
