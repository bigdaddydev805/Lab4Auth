extends CanvasLayer
## The HUD at native resolution: top bar, collapsible right side panel, bottom bar and toasts.
## Children expose their own signals; main.gd wires them to commands.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")
const TopBar := preload("res://scripts/ui/top_bar.gd")
const BottomBar := preload("res://scripts/ui/bottom_bar.gd")
const SidePanel := preload("res://scripts/ui/side_panel.gd")
const Toasts := preload("res://scripts/ui/toasts.gd")

signal panel_visibility_changed(open: bool)

const MARGIN := 8

var top_bar: TopBar
var bottom_bar: BottomBar
var side_panel: SidePanel
var toasts: Toasts
var panel_open: bool = true

var _root: Control
var _panel_tween: Tween


func _init() -> void:
	layer = 10
	_root = Control.new()
	_root.set_anchors_preset(Control.PRESET_FULL_RECT)
	_root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_root.theme = UiTheme.build()
	add_child(_root)

	side_panel = SidePanel.new()
	side_panel.anchor_left = 1.0
	side_panel.anchor_right = 1.0
	side_panel.anchor_top = 0.0
	side_panel.anchor_bottom = 1.0
	side_panel.offset_top = TopBar.HEIGHT + MARGIN
	side_panel.offset_bottom = -(BottomBar.HEIGHT + MARGIN)
	side_panel.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	_root.add_child(side_panel)
	_apply_panel_offsets(1.0 if panel_open else 0.0)

	top_bar = TopBar.new()
	top_bar.set_anchors_preset(Control.PRESET_TOP_WIDE)
	_root.add_child(top_bar)
	top_bar.panel_toggle_pressed.connect(func() -> void: set_panel_open(not panel_open))

	bottom_bar = BottomBar.new()
	bottom_bar.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	bottom_bar.offset_top = -BottomBar.HEIGHT
	_root.add_child(bottom_bar)

	toasts = Toasts.new()
	toasts.set_anchors_preset(Control.PRESET_CENTER_TOP)
	toasts.grow_horizontal = Control.GROW_DIRECTION_BOTH
	toasts.offset_top = TopBar.HEIGHT + 10
	_root.add_child(toasts)


func top_height() -> float:
	return TopBar.HEIGHT


func bottom_height() -> float:
	return BottomBar.HEIGHT


## Width the open panel takes from the right edge (including margins).
func panel_width() -> float:
	return SidePanel.WIDTH + MARGIN * 2.0


func set_panel_open(open: bool, animate: bool = true) -> void:
	if open == panel_open and animate:
		return
	panel_open = open
	if _panel_tween:
		_panel_tween.kill()
	var target: float = 1.0 if open else 0.0
	if animate:
		_panel_tween = create_tween()
		_panel_tween.tween_method(_apply_panel_offsets, 1.0 - target, target, 0.22) \
			.set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
	else:
		_apply_panel_offsets(target)
	panel_visibility_changed.emit(open)


## Is a GUI control (not the world) under this screen point?
func is_over_ui(point: Vector2) -> bool:
	if not visible:
		return false
	for c: Control in [top_bar, bottom_bar, side_panel]:
		if c.visible and c.get_global_rect().has_point(point):
			return true
	return false


func _apply_panel_offsets(t: float) -> void:
	var shown: float = SidePanel.WIDTH + MARGIN
	side_panel.offset_right = -MARGIN + (1.0 - t) * shown
	side_panel.offset_left = side_panel.offset_right - SidePanel.WIDTH
	side_panel.visible = t > 0.001
