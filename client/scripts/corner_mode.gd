extends Node
## "Corner of the monitor" mode (F2): a small borderless, always-on-top window showing only the
## office at an integer scale in the bottom-right of the screen. Drag it with the left mouse
## button; the mouse wheel changes its size. Toggling again restores the previous window.

signal changed(active: bool)

var active: bool = false
## Office size in art pixels; the owner keeps this in sync with the server's layout.
var office_px: Vector2i = Vector2i(384, 224)
var corner_scale: int = 0

var _saved: Dictionary = {}
var _dragging: bool = false
var _drag_offset: Vector2i = Vector2i.ZERO


func toggle() -> void:
	set_active(not active)


func set_active(on: bool) -> void:
	if on == active or DisplayServer.get_name() == "headless":
		return
	var win: Window = get_window()
	if on:
		_saved = {
			"mode": win.mode, "size": win.size, "position": win.position, "borderless": win.borderless,
			"on_top": win.always_on_top, "scale_mode": win.content_scale_mode, "min_size": win.min_size,
		}
		if win.mode != Window.MODE_WINDOWED:
			win.mode = Window.MODE_WINDOWED
		win.content_scale_mode = Window.CONTENT_SCALE_MODE_DISABLED
		win.borderless = true
		win.always_on_top = true
		win.min_size = office_px
		if corner_scale <= 0:
			var usable: Rect2i = DisplayServer.screen_get_usable_rect(win.current_screen)
			corner_scale = 2 if usable.size.y >= 900 else 1
		_apply_size(true)
	else:
		_dragging = false
		win.always_on_top = bool(_saved.get("on_top", false))
		win.borderless = bool(_saved.get("borderless", false))
		win.content_scale_mode = _saved.get("scale_mode", Window.CONTENT_SCALE_MODE_CANVAS_ITEMS)
		win.min_size = _saved.get("min_size", Vector2i.ZERO)
		win.size = _saved.get("size", Vector2i(1280, 720))
		win.position = _saved.get("position", Vector2i(64, 64))
		win.mode = _saved.get("mode", Window.MODE_WINDOWED)
	active = on
	changed.emit(on)


func _input(event: InputEvent) -> void:
	if not active:
		return
	var win: Window = get_window()
	var mb: InputEventMouseButton = event as InputEventMouseButton
	if mb:
		if mb.button_index == MOUSE_BUTTON_LEFT:
			_dragging = mb.pressed
			_drag_offset = DisplayServer.mouse_get_position() - win.position
		elif mb.pressed and mb.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
			corner_scale = clampi(corner_scale + (1 if mb.button_index == MOUSE_BUTTON_WHEEL_UP else -1), 1, 4)
			_apply_size(false)
		get_viewport().set_input_as_handled()
	elif event is InputEventMouseMotion and _dragging:
		win.position = DisplayServer.mouse_get_position() - _drag_offset
		get_viewport().set_input_as_handled()


## Resizes to office × corner_scale, keeping the bottom-right corner in place (or snapping to
## the screen's bottom-right corner when `to_corner`).
func _apply_size(to_corner: bool) -> void:
	var win: Window = get_window()
	var usable: Rect2i = DisplayServer.screen_get_usable_rect(win.current_screen)
	var sz: Vector2i = office_px * corner_scale
	var bottom_right: Vector2i = win.position + win.size
	if to_corner:
		bottom_right = usable.position + usable.size - Vector2i(24, 24)
	win.size = sz
	win.position = bottom_right - sz
