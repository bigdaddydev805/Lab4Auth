extends ScrollContainer
## The Questions tab: every question (newest first) with its status and, once answered, the answer.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const STATUS_COLORS := {
	"open": Color("f2c14e"), "claimed": Color("6cb4e4"), "answered": Color("7bc47f"),
}

var _list: VBoxContainer


func _init() -> void:
	horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	_list = VBoxContainer.new()
	_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_list.add_theme_constant_override("separation", 8)
	add_child(_list)


## Rebuilds from the questions dict; `name_of` maps an employee id to a display name.
func refresh(questions: Dictionary, name_of: Callable) -> void:
	for child: Node in _list.get_children():
		child.queue_free()
	var qs: Array = questions.values()
	qs.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return float(a.get("submitted_ms", 0)) > float(b.get("submitted_ms", 0)))
	if qs.is_empty():
		var empty: Label = Label.new()
		empty.text = "No questions yet. Ask one below!"
		empty.theme_type_variation = "Muted"
		_list.add_child(empty)
		return
	for q: Dictionary in qs:
		_list.add_child(_card(q, name_of))


func _card(q: Dictionary, name_of: Callable) -> Control:
	var status: String = str(q.get("status", "open"))
	var card: PanelContainer = PanelContainer.new()
	card.theme_type_variation = "Card"
	var col: VBoxContainer = VBoxContainer.new()
	col.add_theme_constant_override("separation", 4)
	card.add_child(col)

	var head: HBoxContainer = HBoxContainer.new()
	col.add_child(head)
	var id_label: Label = Label.new()
	id_label.text = str(q.get("id", "?"))
	id_label.add_theme_font_override("font", UiTheme.bold_font())
	head.add_child(id_label)
	var chip: PanelContainer = PanelContainer.new()
	chip.add_theme_stylebox_override("panel", UiTheme.box(STATUS_COLORS.get(status, UiTheme.MUTED), 8, 0, UiTheme.INK, 7, 0))
	chip.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	var chip_label: Label = Label.new()
	chip_label.text = status
	chip_label.theme_type_variation = "Small"
	chip.add_child(chip_label)
	head.add_child(chip)
	var who: Label = Label.new()
	who.theme_type_variation = "Muted"
	who.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	who.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	if status == "answered" and q.get("answered_by") != null:
		who.text = "by " + str(name_of.call(str(q["answered_by"])))
	elif q.get("claimed_by") != null:
		who.text = str(name_of.call(str(q["claimed_by"]))) + " is on it"
	head.add_child(who)

	col.add_child(_wrapped(str(q.get("text", "")), "Label"))
	if status == "answered" and q.get("answer") != null:
		var ans_box: PanelContainer = PanelContainer.new()
		ans_box.add_theme_stylebox_override("panel", UiTheme.box(UiTheme.CREAM, 8, 0, UiTheme.INK, 8, 6))
		ans_box.add_child(_wrapped(str(q["answer"]), "Small"))
		col.add_child(ans_box)
	return card


func _wrapped(text: String, variation: String) -> Label:
	var l: Label = Label.new()
	l.text = text
	l.theme_type_variation = variation
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	l.custom_minimum_size.x = 100
	l.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	return l
