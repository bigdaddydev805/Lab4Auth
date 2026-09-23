extends ScrollContainer
## The employee list: status dot, name and current activity. Clicking a row selects them.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

signal employee_selected(emp_id: String)

const STATUS := {
	"present": ["in the office", Color("7bc47f")],
	"offsite": ["at home", Color("8a7f96")],
	"hired": ["choosing who to be…", Color("f2c14e")],
	"departed": ["left HelpCo", Color("d9534f")],
}

var _list: VBoxContainer
var _selected: String = ""


func _init() -> void:
	horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	_list = VBoxContainer.new()
	_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_list.add_theme_constant_override("separation", 4)
	add_child(_list)


## Rebuilds the rows from a list of employee dicts (sorted by hire number).
func refresh(employees: Array, selected_id: String) -> void:
	_selected = selected_id
	for child: Node in _list.get_children():
		child.queue_free()
	if employees.is_empty():
		var empty: Label = Label.new()
		empty.text = "Nobody works here yet. Press Hire!"
		empty.theme_type_variation = "Muted"
		_list.add_child(empty)
		return
	for emp: Dictionary in employees:
		_list.add_child(_row(emp))


func _row(emp: Dictionary) -> Control:
	var emp_id: String = str(emp.get("id", ""))
	var status: String = str(emp.get("status", ""))
	var info: Array = STATUS.get(status, [status, UiTheme.MUTED])
	var btn: Button = Button.new()
	btn.theme_type_variation = "FlatRow"
	btn.toggle_mode = true
	btn.button_pressed = emp_id == _selected
	btn.focus_mode = Control.FOCUS_NONE
	btn.custom_minimum_size = Vector2(0, 46)
	btn.pressed.connect(func() -> void: employee_selected.emit(emp_id))
	var row: HBoxContainer = HBoxContainer.new()
	row.set_anchors_preset(Control.PRESET_FULL_RECT)
	row.offset_left = 10
	row.offset_right = -8
	row.add_theme_constant_override("separation", 10)
	row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	btn.add_child(row)
	var dot: Panel = Panel.new()
	dot.custom_minimum_size = Vector2(10, 10)
	dot.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	dot.add_theme_stylebox_override("panel", UiTheme.box(info[1], 5))
	dot.mouse_filter = Control.MOUSE_FILTER_IGNORE
	row.add_child(dot)
	var col: VBoxContainer = VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	col.alignment = BoxContainer.ALIGNMENT_CENTER
	col.add_theme_constant_override("separation", 0)
	col.mouse_filter = Control.MOUSE_FILTER_IGNORE
	row.add_child(col)
	var name_label: Label = Label.new()
	var pronouns: String = str(emp.get("pronouns", ""))
	name_label.text = str(emp.get("name", emp_id)) + ("  ·  " + pronouns if pronouns != "" else "")
	name_label.add_theme_font_override("font", UiTheme.bold_font())
	name_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	col.add_child(name_label)
	var act: Label = Label.new()
	var activity: Dictionary = emp.get("activity", {}) if typeof(emp.get("activity")) == TYPE_DICTIONARY else {}
	act.text = str(activity.get("label", "")) if status == "present" else str(info[0])
	act.theme_type_variation = "Muted"
	act.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	act.clip_text = true
	act.mouse_filter = Control.MOUSE_FILTER_IGNORE
	col.add_child(act)
	return btn
