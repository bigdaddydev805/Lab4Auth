extends PanelContainer
## Top bar: world name, date, time, day, phase, connection state, and time controls
## (pause/resume and speed buttons) plus the side-panel toggle.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

signal pause_pressed(pause: bool)
signal speed_chosen(value: float)
signal panel_toggle_pressed

const HEIGHT := 32
const SPEEDS: Array[float] = [5.0, 20.0, 60.0, 240.0]
const PHASE_COLORS := {
	"morning": Color("f2c14e"), "workday": Color("7bc47f"), "evening": Color("e8893a"), "night": Color("6c7fd9"),
}

var _world: Label
var _date: Label
var _time: Label
var _day: Label
var _phase_chip: PanelContainer
var _phase: Label
var _slow: Label
var _conn_dot: Panel
var _conn: Label
var _pause: Button
var _speed_buttons: Array[Button] = []
var _paused: bool = false


func _init() -> void:
	theme_type_variation = "BarPanel"
	custom_minimum_size.y = HEIGHT
	mouse_filter = Control.MOUSE_FILTER_STOP
	var row: HBoxContainer = HBoxContainer.new()
	row.add_theme_constant_override("separation", 12)
	add_child(row)

	_world = _label(row, "HelpCo", "BarLabel")
	_world.add_theme_font_override("font", UiTheme.bold_font())
	_world.add_theme_color_override("font_color", UiTheme.PINK)
	_world.add_theme_font_size_override("font_size", 16)
	_time = _label(row, "--:--", "BarLabel")
	_time.add_theme_font_override("font", UiTheme.bold_font())
	_time.add_theme_font_size_override("font_size", 17)
	_time.custom_minimum_size.x = 76
	_date = _label(row, "", "BarLabel")
	_day = _label(row, "", "BarDim")
	_phase_chip = PanelContainer.new()
	_phase_chip.theme_type_variation = "Chip"
	_phase_chip.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(_phase_chip)
	_phase = _label(_phase_chip, "", "Small")
	_slow = _label(row, "", "BarDim")
	_slow.tooltip_text = "Time slows down while a conversation is going on."
	_slow.mouse_filter = Control.MOUSE_FILTER_PASS

	var spacer: Control = Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(spacer)

	_conn_dot = Panel.new()
	_conn_dot.custom_minimum_size = Vector2(10, 10)
	_conn_dot.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(_conn_dot)
	_conn = _label(row, "connecting…", "BarDim")
	row.add_child(VSeparator.new())

	_pause = _button(row, "Pause", "Pause or resume the simulation clock (Space)")
	_pause.custom_minimum_size.x = 76
	_pause.pressed.connect(func() -> void: pause_pressed.emit(not _paused))
	for value: float in SPEEDS:
		var b: Button = _button(row, "%d×" % int(value), "Run the office at %d× real time" % int(value))
		b.toggle_mode = true
		b.pressed.connect(func() -> void: speed_chosen.emit(value))
		_speed_buttons.append(b)
	row.add_child(VSeparator.new())
	var panel_btn: Button = _button(row, "Panel", "Show or hide the side panel (F3)")
	panel_btn.pressed.connect(func() -> void: panel_toggle_pressed.emit())
	set_connection(0, "")


## Updates the date/day/phase/speed widgets from a clock payload.
func set_clock(info: Dictionary, paused: bool, scale: float, effective_scale: float) -> void:
	_date.text = str(info.get("date", ""))
	_day.text = "Day %d" % int(info.get("day", 0)) if info.has("day") else ""
	var phase: String = str(info.get("phase", ""))
	_phase.text = phase.capitalize()
	_phase_chip.visible = phase != ""
	var chip: StyleBoxFlat = UiTheme.box(PHASE_COLORS.get(phase, UiTheme.PLUM_LIGHT), 9, 0, UiTheme.INK, 8, 1)
	_phase_chip.add_theme_stylebox_override("panel", chip)
	_paused = paused
	_pause.text = "Resume" if paused else "Pause"
	var slowed: bool = not paused and effective_scale < scale - 0.01
	_slow.text = "chatting · %s×" % _fmt_scale(effective_scale) if slowed else ""
	for i: int in range(SPEEDS.size()):
		_speed_buttons[i].set_pressed_no_signal(absf(SPEEDS[i] - scale) < 0.01)


func set_time_text(text: String) -> void:
	if _time.text != text:
		_time.text = text


func set_world_name(name: String) -> void:
	_world.text = name


## state: 0 offline, 1 connecting, 2 open (ServerLink.LinkState).
func set_connection(state: int, detail: String) -> void:
	var color: Color = [UiTheme.RED, UiTheme.YELLOW, UiTheme.GREEN][clampi(state, 0, 2)]
	_conn_dot.add_theme_stylebox_override("panel", UiTheme.box(color, 5))
	_conn.text = ["offline", "connecting…", "live"][clampi(state, 0, 2)]
	_conn.tooltip_text = detail
	_pause.disabled = state != 2
	for b: Button in _speed_buttons:
		b.disabled = state != 2


func _label(parent: Node, text: String, variation: String) -> Label:
	var l: Label = Label.new()
	l.text = text
	l.theme_type_variation = variation
	l.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	l.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	parent.add_child(l)
	return l


func _button(parent: Node, text: String, tip: String) -> Button:
	var b: Button = Button.new()
	b.text = text
	b.tooltip_text = tip
	b.theme_type_variation = "BarButton"
	b.focus_mode = Control.FOCUS_NONE
	b.add_theme_font_size_override("font_size", 13)
	b.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	parent.add_child(b)
	return b


static func _fmt_scale(v: float) -> String:
	return str(int(round(v))) if v >= 10.0 else String.num(v, 1)
