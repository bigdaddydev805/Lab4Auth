extends Node3D
## The office as a little 3D diorama, built from the server's office layout (map + objects).
## Tile (x, y) is the world square [x, x+1] × [y, y+1] on the floor plane (y up); the camera
## looks from the +x/+z corner, so the top and left outer walls are tall "back" walls and the
## rest are cut low (a cutaway dollhouse). Employees are 3D chibis; items rest on surfaces.
## API mirrors the old 2D office view so main.gd and the overlay stay simple.

const Kit := preload("res://scripts/world3d/kit.gd")
const Props := preload("res://scripts/world3d/props.gd")
const Chibi := preload("res://scripts/world3d/chibi.gd")
const CameraRig := preload("res://scripts/world3d/camera_rig.gd")
const Atmosphere := preload("res://scripts/world3d/atmosphere.gd")
const WorldState := preload("res://scripts/model/world_state.gd")

const BACK_WALL_H := 2.6
const HALF_WALL_H := 0.72
const FRONT_WALL_H := 0.3
const FLOORS := {"w": ["#c8925c", "#bb8552"], "b": ["#efe3cc", "#dcc9a8"], "m": ["#6f7ca3", "#6a769c"], "e": ["#b9b1a8", "#aea69c"]}
const PAPER := {"m": "#d8d0ee", "b": "#cfe8dc", "w": "#f0dcc0", "e": "#f0d9cf"}
const WAINSCOT := {"m": "#8a82b0", "b": "#8fb8a4", "w": "#c49a70", "e": "#c49a90"}

var state: WorldState
var rig: CameraRig
var atmosphere: Atmosphere
## emp_id -> Chibi for everyone currently present.
var characters: Dictionary = {}
var selected_id: String = ""
var width: int = 24
var height: int = 14

var _static: Node3D
var _items: Node3D
var _people: Node3D
var _layout_key: String = ""
var _objects: Dictionary = {}     # object id -> {obj, node}
var _fx: Dictionary = Props.fx_dummy()
var _t: float = 0.0
var _whiteboard_text: String = ""


func setup(world_state: WorldState) -> void:
	state = world_state
	_static = Kit.node(self)
	_items = Kit.node(self)
	_people = Kit.node(self)
	atmosphere = Atmosphere.new()
	add_child(atmosphere)
	rig = CameraRig.new()
	add_child(rig)


func is_ready_to_draw() -> bool:
	return state.has_snapshot and not state.office.is_empty() and _layout_key != ""


## Office size in "pixels" for the corner window (keeps a pleasant aspect).
func office_size() -> Vector2:
	return Vector2(width, height) * 22.0


## Rebuilds the static office (only when the layout changed), then items and people.
func rebuild() -> void:
	var office: Dictionary = state.office
	if office.is_empty():
		return
	var key: String = str(hash(JSON.stringify([office.get("map", []), office.get("objects", [])])))
	if key != _layout_key:
		_layout_key = key
		_build_static(office)
		rig.set_bounds(Vector3(0, 0, 0), Vector3(width, BACK_WALL_H, height))
		rig.frame_all(true)
	rebuild_items()
	sync_all_employees()
	set_whiteboard(state.whiteboard)


func set_whiteboard(text: String) -> void:
	_whiteboard_text = text
	var label: Label3D = _fx.get("whiteboard")
	if label:
		label.text = text.substr(0, 220)


func sync_all_employees() -> void:
	for emp_id: String in characters.keys():
		if not state.employees.has(emp_id):
			_remove_character(emp_id)
	for emp_id: String in state.employees.keys():
		sync_employee(emp_id)


func sync_employee(emp_id: String) -> void:
	var emp: Dictionary = state.employees.get(emp_id, {})
	var present: bool = str(emp.get("status", "")) == "present" and typeof(emp.get("look")) == TYPE_DICTIONARY
	if not present:
		_remove_character(emp_id)
		return
	var ch: Chibi = characters.get(emp_id)
	if ch == null:
		ch = Chibi.new()
		ch.setup(emp_id)
		_people.add_child(ch)
		characters[emp_id] = ch
	ch.emp = emp
	ch.apply_look(emp["look"], str(emp.get("look_key", JSON.stringify(emp["look"]))))
	var held: Dictionary = state.held_item(emp_id)
	ch.holding_kind = str(held.get("kind", ""))
	ch.holding_color = held.get("color")


func rebuild_items() -> void:
	for c: Node in _items.get_children():
		c.queue_free()
	var used: Dictionary = {}
	var items: Array = state.items.values()
	items.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return float(a.get("created_ms", 0)) < float(b.get("created_ms", 0)))
	var pending: Array = []
	for it: Dictionary in items:
		var loc: Dictionary = it.get("location", {}) if typeof(it.get("location")) == TYPE_DICTIONARY else {}
		if loc.has("on") and _objects.has(str(loc["on"])):
			if loc.get("slot") == null:
				pending.append(it)
				continue
			_place_on(it, str(loc["on"]), int(loc["slot"]), used)
		elif loc.has("at"):
			var at: Vector2 = Chibi._vec(loc["at"])
			var n: Node3D = Props.item(_items, str(it.get("kind", "")), it.get("color"), 1.4)
			n.position = Vector3(at.x + 0.5 + _jitter(it, 1) * 0.25, 0.0, at.y + 0.5 + _jitter(it, 2) * 0.25)
			n.rotation.y = _jitter(it, 3) * PI
	for it: Dictionary in pending:
		var obj_id: String = str(it["location"]["on"])
		var slot: int = 0
		while used.has(obj_id + "#" + str(slot)) and slot < 8:
			slot += 1
		_place_on(it, obj_id, slot, used)


func _place_on(it: Dictionary, obj_id: String, slot: int, used: Dictionary) -> void:
	used[obj_id + "#" + str(slot)] = true
	var entry: Dictionary = _objects[obj_id]
	var slots: Dictionary = Props.surface_slots(entry["obj"])
	var xs: Array = slots["xs"]
	if xs.is_empty():
		return
	var i: int = slot % xs.size()
	var base: Vector3 = (entry["node"] as Node3D).position
	var n: Node3D = Props.item(_items, str(it.get("kind", "")), it.get("color"), 1.4)
	n.position = base + Vector3(float(xs[i]), float(slots["y"]), float(slots["zs"][i]))
	n.rotation.y = _jitter(it, 3) * 0.6 - 0.3


## Deterministic -1..1 wobble per item so things look set down, not aligned.
static func _jitter(it: Dictionary, salt: int) -> float:
	return float(hash(str(it.get("id", "")) + str(salt)) % 2001) / 1000.0 - 1.0


## Picks the front-most employee under a screen point, or "".
func character_at(screen_pos: Vector2) -> String:
	var cam: Camera3D = rig.camera
	var best: String = ""
	var best_d: float = INF
	for emp_id: String in characters.keys():
		var ch: Chibi = characters[emp_id]
		var feet: Vector2 = cam.unproject_position(ch.feet())
		var head: Vector2 = cam.unproject_position(ch.head_top())
		var half_w: float = maxf(absf(feet.y - head.y) * 0.32, 14.0)
		var rect: Rect2 = Rect2(feet.x - half_w, head.y, half_w * 2.0, feet.y - head.y + 6.0)
		if rect.has_point(screen_pos):
			var d: float = cam.global_position.distance_to(ch.feet())   # nearer to the camera wins
			if d < best_d:
				best_d = d
				best = emp_id
	return best


## Screen anchors for the overlay: feet and head-top points.
func anchor_of(emp_id: String) -> Dictionary:
	var ch: Chibi = characters.get(emp_id)
	if ch == null or not visible:
		return {}
	var cam: Camera3D = rig.camera
	return {"feet": cam.unproject_position(ch.feet()), "head": cam.unproject_position(ch.head_top())}


## Per-frame update of people, ambient motion and lighting.
func tick(now_ms: float, delta: float, talking: Dictionary) -> void:
	_t += delta
	var sitting_at: Dictionary = {}
	var coffee_making: bool = false
	for emp_id: String in characters.keys():
		var ch: Chibi = characters[emp_id]
		ch.talking = talking.has(emp_id)
		ch.selected = emp_id == selected_id
		ch.tick(now_ms, delta)
		if ch.seated:
			sitting_at[Vector2i(int(round(ch.position.x - 0.5 - sin(ch.rotation.y) * Chibi.TYPE_LEAN)),
				int(round(ch.position.z - 0.5 - cos(ch.rotation.y) * Chibi.TYPE_LEAN)))] = true
		var act: Variant = ch.emp.get("activity")
		if typeof(act) == TYPE_DICTIONARY and str(act.get("kind", "")) == "coffee" and not ch.walking:
			coffee_making = true
	var sel: Chibi = characters.get(selected_id)
	rig.follow_target = sel.feet() + Vector3(0, 0.6, 0) if sel else Vector3.INF
	_animate_fx(delta, sitting_at, coffee_making)
	if state.clock.has_time():
		atmosphere.update_time(now_ms + state.clock.tz_offset_s * 1000.0, delta)
		_update_clock(now_ms + state.clock.tz_offset_s * 1000.0)


func _animate_fx(delta: float, sitting_at: Dictionary, coffee_making: bool) -> void:
	for s: Dictionary in _fx["sway"]:
		var n: Node3D = s["node"]
		var a: float = float(s["amp"])
		n.rotation_degrees = Vector3(sin(_t * 0.9 + s["phase"]) * a, 0.0, sin(_t * 1.3 + s["phase"] * 1.7) * a)
	for b: Dictionary in _fx["blink"]:
		var on: bool = fmod(_t, float(b["period"])) < float(b["period"]) * float(b["duty"])
		(b["mat"] as StandardMaterial3D).emission_energy_multiplier = float(b["energy"]) if on else 0.2
	for sc: Dictionary in _fx["screens"]:
		var target: float = 1.0 if sitting_at.has(sc["seat"]) else 0.0
		var cur: float = float(sc["on"])
		cur = target if cur < 0.0 else move_toward(cur, target, delta * 2.0)
		sc["on"] = cur
		var m: StandardMaterial3D = sc["mat"]
		var flicker: float = 1.0 + 0.05 * sin(_t * 7.0 + float(hash(sc["seat"]) % 10))
		m.albedo_color = Color("#1e2233").lerp(Color("#7fb8ff"), cur)
		m.emission = m.albedo_color
		m.emission_energy_multiplier = (0.15 + 2.1 * cur) * flicker
		(sc["light"] as OmniLight3D).light_energy = 0.9 * cur
		(sc["light"] as OmniLight3D).visible = cur > 0.02
	for st: Dictionary in _fx["steam"]:
		if str(st.get("where", "")) == "coffee_machine":
			(st["node"] as CPUParticles3D).emitting = coffee_making
	var open_q: int = 0
	for q: Dictionary in state.questions.values():
		if str(q.get("status", "")) != "answered":
			open_q += 1
	Props.board_cards(_fx, open_q)


func _update_clock(local_ms: float) -> void:
	var c: Dictionary = _fx.get("clock", {})
	if c.is_empty():
		return
	var secs: float = fmod(local_ms / 1000.0, 86400.0)
	var minutes: float = secs / 60.0
	(c["minute"] as Node3D).rotation.z = -fmod(minutes, 60.0) / 60.0 * TAU
	(c["hour"] as Node3D).rotation.z = -fmod(minutes / 60.0, 12.0) / 12.0 * TAU


func _remove_character(emp_id: String) -> void:
	var ch: Chibi = characters.get(emp_id)
	if ch:
		ch.queue_free()
		characters.erase(emp_id)


# ------------------------------------------------------------------ static scene

func _build_static(office: Dictionary) -> void:
	for c: Node in _static.get_children():
		c.queue_free()
	_objects.clear()
	_fx = Props.fx_dummy()
	atmosphere.clear_fixtures()
	var map: Array = office.get("map", [])
	height = map.size()
	width = str(map[0]).length() if height > 0 else 0
	_build_base()
	_build_floors(map)
	_build_walls(map)
	var occupied: Dictionary = {}
	for o: Variant in office.get("objects", []):
		var obj: Dictionary = o
		var node: Node3D = Props.build(_static, obj, _fx)
		_objects[str(obj.get("id", ""))] = {"obj": obj, "node": node}
		for dx: int in int(obj.get("w", 1)):
			for dy: int in int(obj.get("h", 1)):
				occupied[Vector2i(int(obj["x"]) + dx, int(obj["y"]) + dy)] = true
		for s: Variant in obj.get("spots", []):
			occupied[Vector2i(int(s["x"]), int(s["y"]))] = true
	_build_clutter(map, occupied)
	_build_windows_and_decor(map)
	_build_lights()
	set_whiteboard(_whiteboard_text)


func _build_base() -> void:
	# The diorama plinth the building sits on, and the wet street around it.
	Kit.box(_static, Vector3(width + 60, 0.05, height + 60), "#141020", Vector3(width * 0.5, -0.62, height * 0.5))
	Kit.box(_static, Vector3(width + 1.2, 0.5, height + 1.2), "#2a2238", Vector3(width * 0.5, -0.6, height * 0.5))
	Kit.box(_static, Vector3(width + 1.3, 0.06, height + 1.3), "#3b2e4a", Vector3(width * 0.5, -0.16, height * 0.5))


func _build_floors(map: Array) -> void:
	for y: int in height:
		var row: String = map[y]
		var x: int = 0
		while x < width:
			var ch: String = row[x]
			if not FLOORS.has(ch):
				x += 1
				continue
			var start: int = x
			while x < width and row[x] == ch:
				x += 1
			var colors: Array = FLOORS[ch]
			if ch == "b":
				Kit.box(_static, Vector3(x - start, 0.1, 1), colors[0], Vector3((start + x) * 0.5, -0.1, y + 0.5))
				for tx: int in range(start, x):
					if (tx + y) % 2 == 1:
						Kit.box(_static, Vector3(1, 0.004, 1), colors[1], Vector3(tx + 0.5, 0.0, y + 0.5))
			else:
				# Planks / carpet rows alternate shade for a little texture.
				Kit.box(_static, Vector3(x - start, 0.1, 1), colors[y % 2], Vector3((start + x) * 0.5, -0.1, y + 0.5))
				if ch == "w":
					for tx: int in range(start + (y % 3), x, 3):
						Kit.box(_static, Vector3(0.02, 0.004, 1.0), "#a87447", Vector3(tx, 0.0, y + 0.5))
	# Rugs.
	Kit.box(_static, Vector3(4.6, 0.02, 3.0), "#4e5a86", Vector3(4.0, 0.0, 3.0))
	Kit.box(_static, Vector3(4.2, 0.022, 2.6), "#5d6a9a", Vector3(4.0, 0.0, 3.0))
	Kit.box(_static, Vector3(3.8, 0.02, 2.0), "#5d8f73", Vector3(18.5, 0.0, 3.4))
	Kit.box(_static, Vector3(4.2, 0.02, 2.6), "#b8584a", Vector3(13.6, 0.0, 10.6))
	Kit.box(_static, Vector3(3.8, 0.022, 2.2), "#c9705f", Vector3(13.6, 0.0, 10.6))


func _is_floor(map: Array, x: int, y: int) -> bool:
	return x >= 0 and y >= 0 and x < width and y < height and FLOORS.has(str(map[y])[x])


func _room_near(map: Array, x: int, y: int) -> String:
	for d: Vector2i in [Vector2i(0, 1), Vector2i(1, 0), Vector2i(0, -1), Vector2i(-1, 0)]:
		if _is_floor(map, x + d.x, y + d.y):
			return str(map[y + d.y])[x + d.x]
	return ""


func _build_walls(map: Array) -> void:
	for y: int in height:
		for x: int in width:
			if FLOORS.has(str(map[y])[x]):
				continue
			var back: bool = y == 0 or x == 0
			var border: bool = back or y == height - 1 or x == width - 1
			var room: String = _room_near(map, x, y)
			if back:
				Kit.box(_static, Vector3(1, BACK_WALL_H, 1), PAPER.get(room, "#4c405e"), Vector3(x + 0.5, 0, y + 0.5))
				Kit.box(_static, Vector3(1.02, 0.1, 1.02), "#4c405e", Vector3(x + 0.5, BACK_WALL_H, y + 0.5))
				if room != "":
					# Wainscot and a skirting board on the inner face.
					var inner: Vector3 = Vector3(x + 0.5, 0, y + 1.0) if y == 0 else Vector3(x + 1.0, 0, y + 0.5)
					var sz: Vector3 = Vector3(1.0, 0.9, 0.04) if y == 0 else Vector3(0.04, 0.9, 1.0)
					Kit.box(_static, sz, WAINSCOT.get(room, "#8a82b0"), inner)
					var rail: Vector3 = Vector3(1.0, 0.05, 0.07) if y == 0 else Vector3(0.07, 0.05, 1.0)
					Kit.box(_static, rail, "#fff3de", inner + Vector3(0, 0.9, 0))
			elif border:
				Kit.box(_static, Vector3(1, FRONT_WALL_H, 1), "#5b4c6e", Vector3(x + 0.5, 0, y + 0.5))
				Kit.box(_static, Vector3(1.02, 0.05, 1.02), "#76648c", Vector3(x + 0.5, FRONT_WALL_H, y + 0.5))
			else:
				Kit.box(_static, Vector3(1, HALF_WALL_H, 1), WAINSCOT.get(room, "#8a82b0"), Vector3(x + 0.5, 0, y + 0.5))
				Kit.box(_static, Vector3(1.02, 0.06, 1.02), "#8d5f3d", Vector3(x + 0.5, HALF_WALL_H, y + 0.5))
				_glass(map, x, y)


## A glass partition on top of an interior half wall, with slim posts: the office stays
## open to the camera but rooms still read as rooms.
func _glass(map: Array, x: int, y: int) -> void:
	var along_x: bool = not _is_floor(map, x - 1, y) or not _is_floor(map, x + 1, y)
	if _is_floor(map, x, y - 1) and _is_floor(map, x, y + 1):
		along_x = true
	elif _is_floor(map, x - 1, y) and _is_floor(map, x + 1, y):
		along_x = false
	var size: Vector3 = Vector3(1.0, 0.95, 0.04) if along_x else Vector3(0.04, 0.95, 1.0)
	var pane: MeshInstance3D = Kit.box(_static, size, _glass_mat(), Vector3(x + 0.5, HALF_WALL_H + 0.06, y + 0.5))
	pane.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	Kit.box(_static, Vector3(0.05, 0.95, 0.05), "#4c405e", Vector3(x + 0.5 + (0.48 if along_x else 0.0), HALF_WALL_H + 0.06, y + 0.5 + (0.0 if along_x else 0.48)))
	Kit.box(_static, Vector3(1.0, 0.04, 0.06) if along_x else Vector3(0.06, 0.04, 1.0), "#4c405e", Vector3(x + 0.5, HALF_WALL_H + 1.0, y + 0.5))


var _glass_material: StandardMaterial3D


func _glass_mat() -> StandardMaterial3D:
	if _glass_material == null:
		_glass_material = StandardMaterial3D.new()
		_glass_material.albedo_color = Color(0.75, 0.85, 1.0, 0.16)
		_glass_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		_glass_material.roughness = 0.05
		_glass_material.metallic_specular = 0.9
	return _glass_material


func _build_clutter(map: Array, occupied: Dictionary) -> void:
	var free := func(x: int, y: int) -> bool:
		return _is_floor(map, x, y) and not occupied.has(Vector2i(x, y))
	if free.call(7, 1):
		Props.bookshelf(Kit.node(_static, Vector3(7.5, 0, 1.5)), _fx)
	if free.call(9, 1):
		Props.trash_bin(Kit.node(_static, Vector3(9.35, 0, 1.35)))
	if free.call(15, 7):
		Props.water_cooler(Kit.node(_static, Vector3(15.5, 0, 7.4)))
	if free.call(1, 7):
		Props.filing_cabinet(Kit.node(_static, Vector3(1.45, 0, 7.45)))
	if free.call(1, 12):
		Props.plant(Kit.node(_static, Vector3(1.5, 0, 12.5)), 1.1, _fx)
	if free.call(1, 5):
		Props.plant(Kit.node(_static, Vector3(1.5, 0, 5.4)), 1.2, _fx)
	if free.call(22, 11):
		Props.armchair(Kit.node(_static, Vector3(22.4, 0, 11.0), -90.0), "#8e6cc9")
	if free.call(22, 7):
		atmosphere.add_lamp(Props.floor_lamp(Kit.node(_static, Vector3(22.4, 0, 7.5))), 2.0, 1.0)


func _build_windows_and_decor(map: Array) -> void:
	var pane_mat: ShaderMaterial = atmosphere.window_material()
	var back_windows: Array = [[2.5, 1], [3.5, 1], [15.5, 1], [17.5, 1], [18.5, 1], [19.5, 1]]
	for wv: Array in back_windows:
		var wx: float = wv[0]
		if not _is_floor(map, int(wx), 1):
			continue
		_window(Vector3(wx, 1.05, 1.0), 0.0, pane_mat)
	for wz: float in [2.5, 3.5, 9.0, 10.0, 11.0]:
		if _is_floor(map, 1, int(wz)):
			_window(Vector3(1.0, 1.05, wz), 90.0, pane_mat)
	# Clock, sign and posters.
	Props.wall_clock(Kit.node(_static, Vector3(9.5, 2.05, 1.03)), _fx)
	Props.neon_sign(Kit.node(_static, Vector3(21.3, 2.1, 1.03)), "HelpCo")
	_poster(Vector3(1.03, 1.5, 7.6), 90.0, "#f2c14e", "#e8893a")
	_poster(Vector3(1.03, 1.45, 5.0), 90.0, "#6cb4e4", "#3f6fc4")
	_poster(Vector3(13.2, 1.45, 1.03), 0.0, "#f06fa4", "#8e6cc9")


func _window(pos: Vector3, yaw: float, pane_mat: ShaderMaterial) -> void:
	var w: Node3D = Kit.node(_static, pos, yaw)
	Kit.box(w, Vector3(0.94, 1.14, 0.05), "#fff3de", Vector3(0, -0.02, 0.0))
	var q: QuadMesh = QuadMesh.new()
	q.size = Vector2(0.8, 1.0)
	var pane: MeshInstance3D = Kit.mesh(w, q, pane_mat, Vector3(0, 0.55, 0.03))
	pane.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	Kit.box(w, Vector3(0.03, 1.0, 0.02), "#fff3de", Vector3(0, 0.05, 0.035))
	Kit.box(w, Vector3(0.8, 0.03, 0.02), "#fff3de", Vector3(0, 0.55, 0.035))
	Kit.box(w, Vector3(1.0, 0.05, 0.14), "#fff3de", Vector3(0, -0.04, 0.06))    # sill
	atmosphere.add_window_light(Kit.omni(w, Vector3(0, 0.5, 0.7), "#7d95e0", 0.6, 2.6))


func _poster(pos: Vector3, yaw: float, bg: String, fg: String) -> void:
	var p: Node3D = Kit.node(_static, pos, yaw)
	Kit.box(p, Vector3(0.62, 0.8, 0.02), "#fff7e8", Vector3(0, -0.4, 0))
	Kit.box(p, Vector3(0.54, 0.6, 0.01), bg, Vector3(0, -0.28, 0.012))
	Kit.sphere(p, 0.14, fg, Vector3(0, 0.02, 0.02), 0.02, 12).rotation_degrees.x = 90.0


func _build_lights() -> void:
	# Warm light from above (the ceiling isn't drawn, so neither are its fixtures: nothing hangs
	# between the camera and the people), plus string lights along the tall walls.
	for p: Array in [[Vector3(4.0, 2.4, 3.0), true, 2.3, 5.5], [Vector3(18.5, 2.4, 3.2), false, 2.0, 5.5],
			[Vector3(5.5, 2.5, 10.5), true, 2.0, 6.5], [Vector3(13.5, 2.5, 10.0), false, 1.6, 6.0],
			[Vector3(20.5, 2.4, 10.0), false, 1.2, 5.0]]:
		atmosphere.add_lamp(Kit.omni(_static, p[0], "#ffb070", p[2], p[3], p[1]), p[2], 1.0)
	var bulb: StandardMaterial3D = Kit.mat("#ffd79a", 0.5, 4.5)
	_string_lights(Vector3(1.2, 2.35, 1.05), Vector3(width - 1.2, 2.35, 1.05), bulb)
	_string_lights(Vector3(1.05, 2.35, 1.2), Vector3(1.05, 2.35, height - 1.2), bulb)


## A sagging strand of warm fairy lights between two points.
func _string_lights(a: Vector3, b: Vector3, bulb: StandardMaterial3D) -> void:
	var length: float = a.distance_to(b)
	var spans: int = maxi(1, int(length / 3.0))
	var count: int = int(length * 3.0)
	for i: int in count + 1:
		var t: float = float(i) / float(count)
		var span_t: float = fmod(t * spans, 1.0)
		var sag: float = 4.0 * span_t * (1.0 - span_t) * 0.22
		var p: Vector3 = a.lerp(b, t) - Vector3(0, sag, 0)
		var s: MeshInstance3D = Kit.sphere(_static, 0.035, bulb, p, -1.0, 6)
		s.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
