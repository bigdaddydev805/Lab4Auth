extends Node3D
## One employee as a little low-poly 3D chibi, built from the look they chose. Its origin is the
## feet; it follows the server's `path` with the local sim clock and animates procedurally:
## walk, sit, type, sip, hold, talk, idle breathing, glances and blinks.

const Kit := preload("res://scripts/world3d/kit.gd")
const Props := preload("res://scripts/world3d/props.gd")

## Top of the head (hair and hats included) above the feet, in world units.
const HEAD_TOP := 1.5
const TURN_SPEED := 10.0
const TYPE_LEAN := 0.16

var emp_id: String = ""
var emp: Dictionary = {}
## Kind and color of the held item ("coffee", ...), set by the office.
var holding_kind: String = ""
var holding_color: Variant = null
var talking: bool = false
var seated: bool = false
var walking: bool = false
var selected: bool = false

var _look_key: String = ""
var _rig: Node3D
var _hips: Array[Node3D] = []
var _knees: Array[Node3D] = []
var _shoulders: Array[Node3D] = []
var _neck: Node3D
var _eyes: Array[Node3D] = []
var _mouth: Node3D
var _hand: Node3D
var _held: Node3D
var _held_kind: String = "-"
var _ring: MeshInstance3D
var _yaw: float = 0.0
var _t: float = 0.0
var _phase: float = 0.0
var _blink_in: float = 2.0
var _blink_left: float = 0.0
var _glance_in: float = 4.0
var _glance_left: float = 0.0
var _glance_to: float = 0.0
var _sip_in: float = 6.0
var _sip_left: float = 0.0
var _gesture: float = 0.0


func setup(id: String) -> void:
	emp_id = id
	_t = randf() * 10.0
	_blink_in = randf_range(1.0, 4.0)


## Rebuilds the body if the look changed.
func apply_look(look: Dictionary, key: String) -> void:
	if key == _look_key and _rig != null:
		return
	_look_key = key
	for c: Node in get_children():
		c.queue_free()
	_hips.clear()
	_knees.clear()
	_shoulders.clear()
	_eyes.clear()
	_held = null
	_held_kind = "-"
	_build(look)


## Feet position (world).
func feet() -> Vector3:
	return global_position


func head_top() -> Vector3:
	return global_position + Vector3(0, HEAD_TOP, 0)


## Per-frame update: position along the path, pose and animation.
func tick(now_ms: float, delta: float) -> void:
	if _rig == null:
		return
	_t += delta
	var place: Dictionary = _place(now_ms)
	var tile: Vector2 = place["tile"]
	walking = place["walking"]
	var pose: String = str((emp.get("activity", {}) as Dictionary).get("pose", "stand")) if typeof(emp.get("activity")) == TYPE_DICTIONARY else "stand"
	seated = not walking and (pose == "sit" or pose == "type")
	var typing: bool = seated and pose == "type"
	var target_yaw: float = _yaw_of(str(place["facing"]))
	_yaw = lerp_angle(_yaw, target_yaw, clampf(delta * TURN_SPEED, 0.0, 1.0))
	rotation.y = _yaw
	var pos: Vector3 = Vector3(tile.x + 0.5, 0.0, tile.y + 0.5)
	if typing:
		pos += Vector3(sin(_yaw), 0, cos(_yaw)) * TYPE_LEAN
	position = pos
	_sync_held()
	_animate(delta, typing)
	if _ring:
		_ring.visible = selected and not seated
		if _ring.visible:
			var s: float = 1.0 + 0.06 * sin(_t * 4.0)
			_ring.scale = Vector3(s, 1.0, s)


# ------------------------------------------------------------------ animation

func _animate(delta: float, typing: bool) -> void:
	var hip: Array[float] = [0.0, 0.0]
	var knee: Array[float] = [0.0, 0.0]
	var arm_x: Array[float] = [0.0, 0.0]
	var arm_z: Array[float] = [0.0, 0.0]
	var bob: float = 0.0
	var head_x: float = 0.0
	var head_y: float = 0.0
	var head_z: float = 0.0
	var lean: float = 0.0
	var breathe: float = 1.0 + 0.018 * sin(_t * 2.2)

	if walking:
		_phase += delta * 9.0
		var s: float = sin(_phase)
		hip = [s * 30.0, -s * 30.0]
		knee = [maxf(0.0, -s) * 35.0, maxf(0.0, s) * 35.0]
		arm_x = [-s * 26.0, s * 26.0]
		bob = absf(cos(_phase)) * 0.05
		lean = 5.0
		head_z = sin(_phase) * 2.0
	elif seated:
		hip = [-88.0, -88.0]
		knee = [88.0, 88.0]
		if typing:
			var k: float = _t * 16.0
			arm_x = [-62.0 + sin(k) * 7.0, -62.0 + sin(k + 1.7) * 7.0]
			arm_z = [-8.0, 8.0]
			head_x = 8.0 + sin(_t * 0.7) * 3.0
			lean = 8.0
		else:
			arm_x = [-20.0, -20.0]
			head_x = sin(_t * 0.5) * 3.0
	else:
		arm_z = [4.0, -4.0]

	# Holding something: carry it in front; coffee gets sipped now and then.
	if holding_kind != "" and not typing:
		arm_x[0] = -48.0 if not walking else -48.0 + arm_x[0] * 0.2
		arm_z[0] = -12.0
		if holding_kind == "coffee" or holding_kind == "mug":
			_sip_in -= delta
			if _sip_left > 0.0:
				_sip_left -= delta
				var u: float = sin(clampf(_sip_left / 2.0, 0.0, 1.0) * PI)
				arm_x[0] = lerpf(-48.0, -128.0, u)
				arm_z[0] = lerpf(-12.0, -24.0, u)
				head_x = lerpf(head_x, -10.0, u)
			elif _sip_in <= 0.0 and not walking:
				_sip_left = 2.0
				_sip_in = randf_range(6.0, 12.0)

	# Talking: mouth flaps, little nods and the odd hand gesture.
	var mouth_open: float = 0.25
	if talking:
		mouth_open = 0.35 + 0.65 * absf(sin(_t * 11.0)) * (0.5 + 0.5 * sin(_t * 3.1))
		head_x += sin(_t * 4.0) * 4.0
		head_z += sin(_t * 1.7) * 3.0
		_gesture += delta
		if not typing and not walking:
			var g: float = maxf(0.0, sin(_gesture * 2.3)) * (0.5 + 0.5 * sin(_gesture * 0.7))
			arm_x[1] = lerpf(arm_x[1], -55.0, g)
			arm_z[1] = lerpf(arm_z[1], 18.0, g)
	else:
		_gesture = 0.0

	# Idle glances around the room.
	if not walking and not typing:
		_glance_in -= delta
		if _glance_left > 0.0:
			_glance_left -= delta
			head_y = _glance_to * sin(clampf(_glance_left / 1.6, 0.0, 1.0) * PI)
		elif _glance_in <= 0.0:
			_glance_left = 1.6
			_glance_to = randf_range(-35.0, 35.0)
			_glance_in = randf_range(4.0, 10.0)

	# Blinks.
	_blink_in -= delta
	var eye_s: float = 1.0
	if _blink_left > 0.0:
		_blink_left -= delta
		eye_s = 0.15
	elif _blink_in <= 0.0:
		_blink_left = 0.12
		_blink_in = randf_range(2.0, 5.0)

	var k_smooth: float = clampf(delta * 14.0, 0.0, 1.0)
	for i: int in 2:
		_hips[i].rotation_degrees.x = lerpf(_hips[i].rotation_degrees.x, hip[i], k_smooth)
		_knees[i].rotation_degrees.x = lerpf(_knees[i].rotation_degrees.x, knee[i], k_smooth)
		_shoulders[i].rotation_degrees.x = lerpf(_shoulders[i].rotation_degrees.x, arm_x[i], k_smooth)
		_shoulders[i].rotation_degrees.z = lerpf(_shoulders[i].rotation_degrees.z, arm_z[i], k_smooth)
		_eyes[i].scale.y = eye_s
	_neck.rotation_degrees = _neck.rotation_degrees.lerp(Vector3(head_x, head_y, head_z), k_smooth)
	_rig.position.y = bob
	_rig.rotation_degrees.x = lerpf(_rig.rotation_degrees.x, lean, k_smooth)
	_rig.scale = Vector3(1.0, breathe, 1.0)
	_mouth.scale.y = mouth_open


func _sync_held() -> void:
	var key: String = holding_kind + "|" + str(holding_color)
	if key == _held_kind:
		return
	_held_kind = key
	if _held:
		_held.queue_free()
		_held = null
	if holding_kind != "":
		_held = Props.item(_hand, holding_kind, holding_color, 0.8)


# ------------------------------------------------------------------ placement

func _place(now_ms: float) -> Dictionary:
	var facing: String = str(emp.get("facing", "down"))
	var path: Array = emp.get("path", []) if typeof(emp.get("path")) == TYPE_ARRAY else []
	var ms_per_tile: float = float(emp.get("ms_per_tile", 0))
	if path.size() >= 2 and ms_per_tile > 0.0:
		var last: int = path.size() - 1
		var t: float = maxf((now_ms - float(emp.get("path_t0", 0))) / ms_per_tile, 0.0)
		if t >= float(last):
			return {"tile": _vec(path[last]), "walking": false, "facing": _dir(_vec(path[last - 1]), _vec(path[last]))}
		var i: int = int(floor(t))
		var a: Vector2 = _vec(path[i])
		var b: Vector2 = _vec(path[i + 1])
		return {"tile": a.lerp(b, t - float(i)), "walking": true, "facing": _dir(a, b)}
	return {"tile": _vec(emp.get("pos", [0, 0])), "walking": false, "facing": facing}


static func _yaw_of(facing: String) -> float:
	match facing:
		"up":
			return PI
		"right":
			return PI * 0.5
		"left":
			return -PI * 0.5
	return 0.0


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


# ------------------------------------------------------------------ body

func _build(look: Dictionary) -> void:
	var skin: Color = Kit.color_of(Kit.SKIN, look.get("skin"), "#e8b88f")
	var top: Color = Kit.color_of(Kit.COLORS, look.get("top_color"), "#5aa36a")
	var pants: Color = Kit.color_of(Kit.COLORS, look.get("pants_color"), "#2f3f6e")
	var shoes: Color = Kit.color_of(Kit.COLORS, look.get("shoes_color"), "#4a4a5a")
	var hair: Color = Kit.color_of(Kit.HAIR, look.get("hair_color"), "#6b3f2a")
	var acc: Color = Kit.color_of(Kit.COLORS, look.get("accessory_color"), "#d9534f")
	var accessories: Array = look.get("accessories", []) if typeof(look.get("accessories")) == TYPE_ARRAY else []
	var style: String = str(look.get("hair_style", "short"))

	Kit.blob_shadow(self, 0.42, 0.4)
	_ring = Kit.torus(self, 0.44, 0.52, Kit.unique_mat("#f06fa4", 0.5, 1.6), Vector3(0, 0.03, 0))
	_ring.scale = Vector3(1, 0.3, 1)
	_ring.visible = false
	_ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	_rig = Kit.node(self)
	# Legs: hip → thigh → knee → shin + shoe. Rest pose hangs straight down.
	for dx: float in [-0.1, 0.1]:
		var hip: Node3D = Kit.node(_rig, Vector3(dx, 0.47, 0))
		Kit.box(hip, Vector3(0.15, 0.22, 0.17), pants, Vector3(0, -0.22, 0))
		var knee: Node3D = Kit.node(hip, Vector3(0, -0.21, 0))
		Kit.box(knee, Vector3(0.13, 0.2, 0.14), pants, Vector3(0, -0.2, 0))
		Kit.box(knee, Vector3(0.15, 0.07, 0.22), shoes, Vector3(0, -0.26, 0.03))
		_hips.append(hip)
		_knees.append(knee)
	# Sweater body with a rolled collar and a hem stripe.
	Kit.box(_rig, Vector3(0.44, 0.4, 0.3), top, Vector3(0, 0.44, 0))
	Kit.box(_rig, Vector3(0.46, 0.05, 0.32), top.darkened(0.15), Vector3(0, 0.43, 0))
	Kit.box(_rig, Vector3(0.24, 0.05, 0.2), top.darkened(0.1), Vector3(0, 0.83, 0))
	# Arms (the first one is the carrying hand).
	for dx: float in [-0.28, 0.28]:
		var sh: Node3D = Kit.node(_rig, Vector3(dx, 0.8, 0))
		Kit.box(sh, Vector3(0.12, 0.3, 0.13), top, Vector3(0, -0.3, 0))
		Kit.sphere(sh, 0.065, skin, Vector3(0, -0.33, 0), -1.0, 10)
		_shoulders.append(sh)
	_hand = Kit.node(_shoulders[0], Vector3(0, -0.4, 0.02))
	_hand.rotation_degrees.x = 90.0

	# Head.
	_neck = Kit.node(_rig, Vector3(0, 0.85, 0))
	var hy: float = 0.28
	Kit.sphere(_neck, 0.3, skin, Vector3(0, hy, 0), -1.0, 18)
	for dx: float in [-0.1, 0.1]:
		var eye: Node3D = Kit.node(_neck, Vector3(dx, hy - 0.02, 0.28))
		Kit.box(eye, Vector3(0.055, 0.085, 0.03), "#2b2135", Vector3(0, -0.04, 0))
		Kit.box(eye, Vector3(0.02, 0.025, 0.01), Kit.mat("#ffffff", 0.3, 0.6), Vector3(0.012, 0.0, 0.016))
		_eyes.append(eye)
		Kit.sphere(_neck, 0.045, Kit.mat("#f29a9a", 0.9, 0.15), Vector3(dx * 1.8, hy - 0.1, 0.24), -1.0, 8)
	_mouth = Kit.node(_neck, Vector3(0, hy - 0.13, 0.285))
	Kit.box(_mouth, Vector3(0.07, 0.04, 0.02), "#7a2e3a", Vector3(0, -0.02, 0))

	# Hair.
	match style:
		"buzz":
			Kit.sphere(_neck, 0.305, hair, Vector3(0, hy + 0.02, -0.01), 0.3, 18)
		"short":
			Kit.sphere(_neck, 0.32, hair, Vector3(0, hy + 0.04, -0.02), 0.32, 18)
			Kit.box(_neck, Vector3(0.36, 0.08, 0.1), hair, Vector3(0.04, hy + 0.16, 0.22))
		"bob":
			Kit.sphere(_neck, 0.33, hair, Vector3(0, hy + 0.03, -0.02), 0.33, 18)
			Kit.box(_neck, Vector3(0.66, 0.4, 0.34), hair, Vector3(0, hy - 0.3, -0.1))
			Kit.box(_neck, Vector3(0.46, 0.09, 0.1), hair, Vector3(0, hy + 0.12, 0.23))
		"long":
			Kit.sphere(_neck, 0.33, hair, Vector3(0, hy + 0.03, -0.02), 0.33, 18)
			Kit.box(_neck, Vector3(0.64, 0.72, 0.3), hair, Vector3(0, hy - 0.62, -0.13))
			Kit.box(_neck, Vector3(0.4, 0.08, 0.1), hair, Vector3(-0.05, hy + 0.14, 0.23))
		"bun":
			Kit.sphere(_neck, 0.32, hair, Vector3(0, hy + 0.04, -0.02), 0.32, 18)
			Kit.sphere(_neck, 0.14, hair, Vector3(0, hy + 0.34, -0.1), -1.0, 12)
		_:
			Kit.sphere(_neck, 0.32, hair, Vector3(0, hy + 0.04, -0.02), 0.32, 18)

	if "glasses" in accessories:
		for dx: float in [-0.1, 0.1]:
			Kit.torus(_neck, 0.045, 0.068, acc, Vector3(dx, hy - 0.06, 0.3), Vector3(90, 0, 0))
		Kit.box(_neck, Vector3(0.08, 0.015, 0.015), acc, Vector3(0, hy - 0.05, 0.3))
	if "frog_hat" in accessories:
		var green: String = "#6cc24a"
		Kit.sphere(_neck, 0.345, green, Vector3(0, hy + 0.06, -0.02), 0.345, 18)
		for dx: float in [-0.14, 0.14]:
			Kit.sphere(_neck, 0.095, green, Vector3(dx, hy + 0.34, 0.08), -1.0, 10)
			Kit.sphere(_neck, 0.06, "#ffffff", Vector3(dx, hy + 0.36, 0.14), -1.0, 8)
			Kit.sphere(_neck, 0.03, "#1c1624", Vector3(dx, hy + 0.36, 0.19), -1.0, 6)
