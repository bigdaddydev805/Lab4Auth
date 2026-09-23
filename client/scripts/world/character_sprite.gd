extends Node2D
## One employee in the office. Its origin is the feet anchor in world pixels; it follows the
## server's `path` with the local sim clock and plays sheet animations per PROTOCOL.md.

const TILE := 16
const SEATED_DROP := 2
const SIP_MIN_SEC := 5.0
const SIP_MAX_SEC := 11.0

var emp_id: String = ""
var emp: Dictionary = {}
## Kind of the held item ("coffee", ...), set by the office view.
var holding_kind: String = ""
## True while this employee's speech bubble is up (drives `talk_down`).
var talking: bool = false
var seated: bool = false
var walking: bool = false
var anim: String = ""

var _meta: Dictionary = {}
var _sprite: Sprite2D
var _anim_t: float = 0.0
var _sip_wait: float = 0.0
var _sip_left: float = 0.0


func setup(id: String, character_meta: Dictionary) -> void:
	emp_id = id
	_meta = character_meta
	_sprite = Sprite2D.new()
	_sprite.centered = false
	_sprite.region_enabled = true
	_sprite.visible = false
	add_child(_sprite)
	_sip_wait = randf_range(SIP_MIN_SEC * 0.5, SIP_MAX_SEC)


func set_sheet(tex: Texture2D) -> void:
	_sprite.texture = tex
	_sprite.visible = tex != null


func has_sheet() -> bool:
	return _sprite.texture != null


func sheet() -> Texture2D:
	return _sprite.texture


## Feet position in world pixels.
func feet() -> Vector2:
	return position


## Screen-independent rectangle (world pixels) used for click picking.
func pick_rect() -> Rect2:
	return Rect2(position + Vector2(-9, -28 + (SEATED_DROP if seated else 0)), Vector2(18, 30))


## Per-frame update: position along the path, animation choice and frame.
func tick(now_ms: float, delta: float) -> void:
	var place: Dictionary = _place(now_ms)
	var tile: Vector2 = place["tile"]
	walking = place["walking"]
	position = Vector2(tile.x * TILE + TILE * 0.5, (tile.y + 1.0) * TILE)
	var pose: String = str((emp.get("activity", {}) as Dictionary).get("pose", "stand"))
	seated = not walking and (pose == "sit" or pose == "type")
	var next: String = _choose_anim(place["facing"], pose, delta)
	if next != anim:
		anim = next
		_anim_t = 0.0
	else:
		_anim_t += delta
	_show_frame()
	z_index = int(round(position.y)) * 3 + 1


## Where the employee is: {"tile": Vector2 (fractional while walking), "walking", "facing"}.
func _place(now_ms: float) -> Dictionary:
	var facing: String = str(emp.get("facing", "down"))
	var path: Array = emp.get("path", []) if typeof(emp.get("path")) == TYPE_ARRAY else []
	var ms_per_tile: float = float(emp.get("ms_per_tile", 0))
	if path.size() >= 2 and ms_per_tile > 0.0:
		var last: int = path.size() - 1
		var t: float = maxf((now_ms - float(emp.get("path_t0", 0))) / ms_per_tile, 0.0)
		if t >= float(last):
			# Arrived locally; the server's final `employee` update is on its way.
			return {"tile": _vec(path[last]), "walking": false, "facing": _dir(_vec(path[last - 1]), _vec(path[last]))}
		var i: int = int(floor(t))
		var a: Vector2 = _vec(path[i])
		var b: Vector2 = _vec(path[i + 1])
		return {"tile": a.lerp(b, t - float(i)), "walking": true, "facing": _dir(a, b)}
	return {"tile": _vec(emp.get("pos", [0, 0])), "walking": false, "facing": facing}


func _choose_anim(facing: String, pose: String, delta: float) -> String:
	if walking:
		return "walk_" + facing
	if pose == "type":
		return "type"
	if pose == "sit":
		return "sit_up" if facing == "up" else "sit_down"
	if facing == "down":
		if holding_kind == "coffee":
			return _coffee_anim(delta)
		if talking:
			return "talk_down"
	return "idle_" + facing


## `hold_down`, with a `sip` now and then.
func _coffee_anim(delta: float) -> String:
	if _sip_left > 0.0:
		_sip_left -= delta
		return "sip"
	_sip_wait -= delta
	if _sip_wait <= 0.0:
		var sip: Dictionary = _anim_info("sip")
		_sip_left = float(sip.get("frames", 8)) / maxf(float(sip.get("fps", 4)), 0.1)
		_sip_wait = randf_range(SIP_MIN_SEC, SIP_MAX_SEC)
		return "sip"
	return "hold_down"


func _show_frame() -> void:
	var info: Dictionary = _anim_info(anim)
	if info.is_empty():
		info = _anim_info("idle_down")
	var fw: float = float(_meta.get("frame_w", 34))
	var fh: float = float(_meta.get("frame_h", 34))
	var frames: int = maxi(int(info.get("frames", 1)), 1)
	var frame: int = int(_anim_t * float(info.get("fps", 4))) % frames
	_sprite.region_rect = Rect2(frame * fw, float(info.get("row", 0)) * fh, fw, fh)
	var anchor: Array = _meta.get("anchor", [17, 32])
	_sprite.position = Vector2(-float(anchor[0]), -float(anchor[1]) + (SEATED_DROP if seated else 0))


func _anim_info(name: String) -> Dictionary:
	var anims: Dictionary = _meta.get("animations", {})
	return anims.get(name, {})


static func _vec(v: Variant) -> Vector2:
	if typeof(v) == TYPE_ARRAY and (v as Array).size() >= 2:
		return Vector2(float(v[0]), float(v[1]))
	return Vector2.ZERO


static func _dir(a: Vector2, b: Vector2) -> String:
	var d: Vector2 = b - a
	if d == Vector2.ZERO:
		return "down"
	if absf(d.x) >= absf(d.y):
		return "right" if d.x > 0.0 else "left"
	return "down" if d.y > 0.0 else "up"
