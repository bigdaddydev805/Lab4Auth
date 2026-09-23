extends Node3D
## Isometric orthographic camera over the diorama.
##   drag (mouse or one finger)  pan          wheel / pinch / + -   zoom
##   WASD / arrows               pan          Home                  frame the whole office
## A click or tap without dragging emits `tapped` (main uses it to pick employees). While
## `following`, the view glides after `follow_target` (the selected employee); panning stops it.

signal tapped(screen_pos: Vector2)

const DIR := Vector3(16.0, 17.0, 20.0)
const DISTANCE := 60.0
const DRAG_THRESHOLD := 8.0
const MIN_SIZE := 3.5
const KEY_PAN_SPEED := 0.9   # viewport heights per second

var camera: Camera3D
var target: Vector3 = Vector3(12, 0.5, 7)
var size: float = 16.0
## Pixels of the screen covered by HUD on each side (top, right, bottom, left).
var insets: Vector4 = Vector4.ZERO
var follow_target: Vector3 = Vector3.INF
var following: bool = false

var _bounds: AABB = AABB(Vector3.ZERO, Vector3(24, 2.6, 14))
var _goal_target: Vector3 = target
var _goal_size: float = size
var _fit_size: float = 16.0
var _press_pos: Vector2 = Vector2.ZERO
var _pressed: bool = false
var _dragging: bool = false
var _touches: Dictionary = {}
var _pinch_d: float = 0.0
var _t: float = 0.0


func _ready() -> void:
	camera = Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.near = 1.0
	camera.far = 200.0
	add_child(camera)
	camera.current = true
	_apply()


func set_bounds(origin: Vector3, end: Vector3) -> void:
	_bounds = AABB(origin, end - origin)


## Frames the whole office inside the area the HUD leaves free.
func frame_all(instant: bool = false) -> void:
	var fit: Dictionary = _fit()
	_fit_size = fit["size"]
	_goal_size = _fit_size
	_goal_target = fit["target"]
	following = false
	if instant:
		size = _goal_size
		target = _goal_target
		_apply()


func zoom_by(factor: float, around: Vector2 = Vector2(-1, -1)) -> void:
	var new_size: float = clampf(_goal_size * factor, MIN_SIZE, _fit_size * 1.25)
	if around.x >= 0.0 and not following:
		# Keep the point under the cursor still.
		var vp: Vector2 = get_viewport().get_visible_rect().size
		var d: Vector2 = around - vp * 0.5
		var upp_old: float = _goal_size / vp.y
		var upp_new: float = new_size / vp.y
		_goal_target += (_right() * d.x - _up() * d.y) * (upp_old - upp_new)
	_goal_size = new_size
	_clamp_goal()


func pan_pixels(delta_px: Vector2) -> void:
	var vp: Vector2 = get_viewport().get_visible_rect().size
	var upp: float = size / vp.y
	_goal_target += (-_right() * delta_px.x + _up() * delta_px.y) * upp
	target = _goal_target
	following = false
	_clamp_goal()


func _process(delta: float) -> void:
	_t += delta
	var pan: Vector2 = Vector2(
		float(Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT))
			- float(Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT)),
		float(Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN))
			- float(Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP)))
	if pan != Vector2.ZERO and get_viewport().gui_get_focus_owner() == null:
		var vp: Vector2 = get_viewport().get_visible_rect().size
		pan_pixels(-pan * vp.y * KEY_PAN_SPEED * delta)
	if following and follow_target != Vector3.INF:
		_goal_target = follow_target
	var k: float = clampf(delta * 6.0, 0.0, 1.0)
	target = target.lerp(_goal_target, k)
	size = lerpf(size, _goal_size, k)
	_apply()


func _apply() -> void:
	if camera == null:
		return
	# A very slow drift keeps the scene feeling alive even when nothing moves.
	var drift: Vector3 = Vector3(sin(_t * 0.07), 0.0, cos(_t * 0.05)) * 0.06
	camera.size = size
	camera.position = target + drift + DIR.normalized() * DISTANCE
	camera.look_at(target + drift, Vector3.UP)
	var vp: Vector2 = get_viewport().get_visible_rect().size
	var upp: float = size / maxf(vp.y, 1.0)
	camera.h_offset = (insets.y - insets.w) * 0.5 * upp
	camera.v_offset = (insets.x - insets.z) * 0.5 * upp


func _unhandled_input(event: InputEvent) -> void:
	var st: InputEventScreenTouch = event as InputEventScreenTouch
	if st:
		if st.pressed:
			_touches[st.index] = st.position
		else:
			_touches.erase(st.index)
		_pinch_d = _touch_distance()
		if _touches.size() >= 2:
			_pressed = false
			_dragging = false
		return
	var sd: InputEventScreenDrag = event as InputEventScreenDrag
	if sd:
		_touches[sd.index] = sd.position
		if _touches.size() >= 2:
			var d: float = _touch_distance()
			if _pinch_d > 0.0 and d > 0.0:
				zoom_by(_pinch_d / d)
				size = _goal_size
			_pinch_d = d
			get_viewport().set_input_as_handled()
		return
	var mg: InputEventMagnifyGesture = event as InputEventMagnifyGesture
	if mg:
		zoom_by(1.0 / maxf(mg.factor, 0.01), mg.position)
		get_viewport().set_input_as_handled()
		return
	var pg: InputEventPanGesture = event as InputEventPanGesture
	if pg:
		pan_pixels(-pg.delta * 12.0)
		get_viewport().set_input_as_handled()
		return
	var mb: InputEventMouseButton = event as InputEventMouseButton
	if mb:
		match mb.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				if mb.pressed:
					zoom_by(0.88, mb.position)
			MOUSE_BUTTON_WHEEL_DOWN:
				if mb.pressed:
					zoom_by(1.0 / 0.88, mb.position)
			MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE:
				if mb.pressed:
					_pressed = true
					_dragging = mb.button_index != MOUSE_BUTTON_LEFT
					_press_pos = mb.position
				else:
					if _pressed and not _dragging and mb.button_index == MOUSE_BUTTON_LEFT and _touches.size() < 2:
						tapped.emit(mb.position)
					_pressed = false
					_dragging = false
			_:
				return
		get_viewport().set_input_as_handled()
		return
	var mm: InputEventMouseMotion = event as InputEventMouseMotion
	if mm and _pressed and _touches.size() < 2:
		if not _dragging and mm.position.distance_to(_press_pos) > DRAG_THRESHOLD:
			_dragging = true
		if _dragging:
			pan_pixels(mm.relative)
		get_viewport().set_input_as_handled()
		return
	var key: InputEventKey = event as InputEventKey
	if key and key.pressed and not key.echo:
		match key.keycode:
			KEY_EQUAL, KEY_PLUS, KEY_KP_ADD:
				zoom_by(0.8)
			KEY_MINUS, KEY_KP_SUBTRACT:
				zoom_by(1.25)
			KEY_HOME:
				frame_all()
			_:
				return
		get_viewport().set_input_as_handled()


func _touch_distance() -> float:
	if _touches.size() < 2:
		return 0.0
	var pts: Array = _touches.values()
	return (pts[0] as Vector2).distance_to(pts[1])


func _right() -> Vector3:
	return Vector3.UP.cross(DIR.normalized()).normalized()


func _up() -> Vector3:
	return DIR.normalized().cross(_right()).normalized()


func _clamp_goal() -> void:
	var c: Vector3 = _bounds.get_center()
	var half: Vector3 = _bounds.size * 0.5 + Vector3(2, 0, 2)
	_goal_target.x = clampf(_goal_target.x, c.x - half.x, c.x + half.x)
	_goal_target.z = clampf(_goal_target.z, c.z - half.z, c.z + half.z)
	_goal_target.y = clampf(_goal_target.y, -2.0, 3.0)


## Ortho size and target that fit the bounds into the free screen area.
func _fit() -> Dictionary:
	var vp: Vector2 = get_viewport().get_visible_rect().size
	var r: Vector3 = _right()
	var u: Vector3 = _up()
	var lo: Vector2 = Vector2(INF, INF)
	var hi: Vector2 = Vector2(-INF, -INF)
	var o: Vector3 = _bounds.position
	for i: int in 8:
		var p: Vector3 = _bounds.get_endpoint(i) - o
		var q: Vector2 = Vector2(p.dot(r), p.dot(u))
		lo = lo.min(q)
		hi = hi.max(q)
	var ext: Vector2 = hi - lo
	var free: Vector2 = Vector2(vp.x - insets.y - insets.w, vp.y - insets.x - insets.z) - Vector2(24, 24)
	free = free.max(Vector2(64, 64))
	var s: float = maxf(ext.y * vp.y / free.y, ext.x * vp.y / free.x)
	var mid: Vector2 = (lo + hi) * 0.5
	var tgt: Vector3 = o + r * mid.x + u * mid.y
	# Move the target onto the floor plane along the view direction (same picture, nicer pivot).
	var d: Vector3 = DIR.normalized()
	tgt -= d * ((tgt.y - 0.5) / d.y)
	return {"size": s, "target": tgt}
