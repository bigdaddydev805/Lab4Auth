extends PanelContainer
## Bottom bar: ask the office a question, talk to one employee over the intercom (or give them
## a gift), and hire someone new.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

signal question_submitted(text: String)
signal intercom_sent(emp_id: String, text: String)
signal gift_sent(emp_id: String, kind: String)
signal hire_pressed
signal recipient_changed(emp_id: String)

const HEIGHT := 48
const GIFTS := [["rubber_duck", "a rubber duck"], ["succulent", "a succulent"], ["coffee", "a coffee"],
	["snow_globe", "a snow globe"], ["books", "some books"], ["photo_frame", "a photo frame"],
	["mug", "a mug"], ["note", "a note"]]

var question_edit: LineEdit
var say_edit: LineEdit
var _recipient: OptionButton
var _gift: MenuButton
var _ids: Array[String] = []


func _init() -> void:
	theme_type_variation = "BarPanel"
	custom_minimum_size.y = HEIGHT
	mouse_filter = Control.MOUSE_FILTER_STOP
	var row: HBoxContainer = HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	add_child(row)

	question_edit = _edit(row, "Ask the office a question…", 1.3)
	question_edit.max_length = 1000
	question_edit.text_submitted.connect(func(_t: String) -> void: _send_question())
	_bar_button(row, "Send question", "Adds it to the question board; whoever is in the work area sees it").pressed.connect(_send_question)

	row.add_child(VSeparator.new())
	var lbl: Label = Label.new()
	lbl.text = "Intercom"
	lbl.theme_type_variation = "BarDim"
	row.add_child(lbl)
	_recipient = OptionButton.new()
	_recipient.focus_mode = Control.FOCUS_NONE
	_recipient.custom_minimum_size.x = 118
	_recipient.fit_to_longest_item = false
	_recipient.item_selected.connect(func(i: int) -> void:
		if i >= 0 and i < _ids.size():
			recipient_changed.emit(_ids[i]))
	row.add_child(_recipient)
	say_edit = _edit(row, "Say something privately…", 1.0)
	say_edit.max_length = 600
	say_edit.text_submitted.connect(func(_t: String) -> void: _send_say())
	_bar_button(row, "Say", "Speak to them over the intercom (private; they'll remember it)").pressed.connect(_send_say)
	_gift = MenuButton.new()
	_gift.text = "Gift"
	_gift.tooltip_text = "Give them something"
	_gift.theme_type_variation = "BarButton"
	_gift.flat = false
	_gift.focus_mode = Control.FOCUS_NONE
	for i: int in range(GIFTS.size()):
		_gift.get_popup().add_item(str(GIFTS[i][1]).capitalize(), i)
	_gift.get_popup().id_pressed.connect(func(id: int) -> void:
		var to: String = selected_recipient()
		if to != "":
			gift_sent.emit(to, str(GIFTS[id][0])))
	row.add_child(_gift)

	row.add_child(VSeparator.new())
	var hire: Button = _bar_button(row, "Hire", "Hire a new employee (up to 4); they choose who to be")
	hire.add_theme_stylebox_override("normal", UiTheme.box(UiTheme.PINK, 8, 0, UiTheme.INK, 12, 4))
	hire.add_theme_color_override("font_color", UiTheme.INK)
	hire.pressed.connect(func() -> void: hire_pressed.emit())
	set_employees([])


## Refreshes the intercom dropdown (keeps the current choice when possible).
func set_employees(employees: Array) -> void:
	var keep: String = selected_recipient()
	_recipient.clear()
	_ids.clear()
	for emp: Dictionary in employees:
		if str(emp.get("status", "")) == "departed" or str(emp.get("name", "")) == "":
			continue
		_ids.append(str(emp["id"]))
		_recipient.add_item(str(emp["name"]))
	if _ids.is_empty():
		_recipient.add_item("(nobody)")
		_recipient.disabled = true
	else:
		_recipient.disabled = false
		select_recipient(keep if keep in _ids else _ids[0])
	_gift.disabled = _ids.is_empty()


func select_recipient(emp_id: String) -> void:
	var i: int = _ids.find(emp_id)
	if i >= 0:
		_recipient.select(i)


func selected_recipient() -> String:
	var i: int = _recipient.selected
	return _ids[i] if i >= 0 and i < _ids.size() else ""


func _send_question() -> void:
	var text: String = question_edit.text.strip_edges()
	if text != "":
		question_submitted.emit(text)
		question_edit.clear()


func _send_say() -> void:
	var text: String = say_edit.text.strip_edges()
	var to: String = selected_recipient()
	if text != "" and to != "":
		intercom_sent.emit(to, text)
		say_edit.clear()


func _edit(parent: Node, placeholder: String, stretch: float) -> LineEdit:
	var e: LineEdit = LineEdit.new()
	e.placeholder_text = placeholder
	e.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	e.size_flags_stretch_ratio = stretch
	e.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	e.custom_minimum_size.x = 140
	e.clear_button_enabled = false
	parent.add_child(e)
	return e


func _bar_button(parent: Node, text: String, tip: String) -> Button:
	var b: Button = Button.new()
	b.text = text
	b.tooltip_text = tip
	b.theme_type_variation = "BarButton"
	b.focus_mode = Control.FOCUS_NONE
	b.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	b.custom_minimum_size.y = 32
	parent.add_child(b)
	return b
