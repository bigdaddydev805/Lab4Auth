extends PanelContainer
## The inspector card for the selected employee: portrait, name, pronouns, current activity,
## last private thought and action (from `decision` messages), needs bars and what they hold.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

signal closed

const NEEDS := [["energy", "Energy", Color("f2c14e")], ["hunger", "Hunger", Color("e8893a")],
	["social", "Social", Color("6cb4e4")]]

var emp_id: String = ""

var _portrait: TextureRect
var _portrait_tex: AtlasTexture = AtlasTexture.new()
var _name: Label
var _sub: Label
var _activity: Label
var _thinking: Label
var _thought: Label
var _action: Label
var _holding: Label
var _bars: Dictionary = {}


func _init() -> void:
	theme_type_variation = "Card"
	var col: VBoxContainer = VBoxContainer.new()
	col.add_theme_constant_override("separation", 5)
	add_child(col)

	var head: HBoxContainer = HBoxContainer.new()
	head.add_theme_constant_override("separation", 8)
	col.add_child(head)
	var frame: PanelContainer = PanelContainer.new()
	frame.add_theme_stylebox_override("panel", UiTheme.box(UiTheme.CREAM, 10, 2, UiTheme.INK, 0, 0))
	head.add_child(frame)
	_portrait = TextureRect.new()
	_portrait.texture = _portrait_tex
	_portrait.custom_minimum_size = Vector2(68, 68)
	_portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_portrait.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	frame.add_child(_portrait)
	var names: VBoxContainer = VBoxContainer.new()
	names.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	names.alignment = BoxContainer.ALIGNMENT_CENTER
	names.add_theme_constant_override("separation", 0)
	head.add_child(names)
	_name = Label.new()
	_name.theme_type_variation = "Heading"
	_name.add_theme_font_size_override("font_size", 18)
	_name.clip_text = true
	_name.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	names.add_child(_name)
	_sub = _small(names, "Muted")
	_sub.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_activity = _small(names, "Small")
	_activity.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	var close: Button = Button.new()
	close.text = "×"
	close.tooltip_text = "Close (Esc)"
	close.focus_mode = Control.FOCUS_NONE
	close.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	close.pressed.connect(func() -> void: closed.emit())
	head.add_child(close)

	_thinking = _small(col, "Small")
	_thinking.text = "thinking…"
	_thinking.add_theme_color_override("font_color", UiTheme.PINK)
	var caption: Label = _small(col, "Muted")
	caption.text = "Last private thought"
	caption.add_theme_font_size_override("font_size", 11)
	_thought = _small(col, "Small")
	_thought.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_thought.add_theme_font_override("font", UiTheme.italic_font())
	_thought.add_theme_color_override("font_color", Color("5b4a6b"))
	_action = _small(col, "Muted")
	_action.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART

	var grid: GridContainer = GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 8)
	grid.add_theme_constant_override("v_separation", 3)
	col.add_child(grid)
	for need: Array in NEEDS:
		var l: Label = _small(grid, "Small")
		l.text = need[1]
		l.custom_minimum_size.x = 54
		var bar: ProgressBar = ProgressBar.new()
		bar.max_value = 1.0
		bar.step = 0.01
		bar.show_percentage = false
		bar.custom_minimum_size = Vector2(0, 10)
		bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		bar.add_theme_stylebox_override("fill", UiTheme.box(need[2], 5, 0, UiTheme.INK, 0, 0))
		grid.add_child(bar)
		_bars[need[0]] = bar
	_holding = _small(col, "Small")


## Fills the card. `decision` may be {}; `sheet` may be null until the art arrives.
func show_employee(emp: Dictionary, decision: Dictionary, held: Dictionary, sheet: Texture2D,
		character_meta: Dictionary, name_of: Callable) -> void:
	emp_id = str(emp.get("id", ""))
	_name.text = str(emp.get("name", emp_id))
	var status: String = str(emp.get("status", ""))
	var bits: PackedStringArray = PackedStringArray()
	if str(emp.get("pronouns", "")) != "":
		bits.append(str(emp["pronouns"]))
	bits.append({"present": "in the office", "offsite": "at home", "hired": "choosing who to be",
		"departed": "left HelpCo"}.get(status, status))
	if emp.get("desk_id") != null:
		bits.append(str(emp["desk_id"]).replace("desk_", "desk "))
	_sub.text = "  ·  ".join(bits)
	var activity: Dictionary = emp.get("activity", {}) if typeof(emp.get("activity")) == TYPE_DICTIONARY else {}
	_activity.text = _sentence(str(activity.get("label", ""))) if status == "present" else ""
	_thinking.visible = bool(emp.get("thinking", false))

	var thought: String = str(decision.get("thought", "")) if decision.get("thought") != null else ""
	_thought.text = "“%s”" % thought if thought != "" else "No private thoughts seen yet."
	_action.text = _describe_action(decision, name_of)
	_action.visible = _action.text != ""

	var needs: Dictionary = emp.get("needs", {}) if typeof(emp.get("needs")) == TYPE_DICTIONARY else {}
	for key: String in _bars.keys():
		var bar: ProgressBar = _bars[key]
		bar.value = float(needs.get(key, 0.0))
		bar.tooltip_text = "%s: %d%%" % [key.capitalize(), int(round(bar.value * 100.0))]
	_holding.text = "Holding: " + (str(held.get("label", held.get("kind", "something"))) if not held.is_empty() else "nothing")

	_portrait_tex.atlas = sheet
	var fw: float = float(character_meta.get("frame_w", 34))
	var fh: float = float(character_meta.get("frame_h", 34))
	_portrait_tex.region = Rect2(0, 0, fw, fh)
	_portrait.visible = sheet != null


func _describe_action(d: Dictionary, name_of: Callable) -> String:
	if d.is_empty() or d.get("action") == null:
		return ""
	var action: String = str(d["action"]).replace("_", " ")
	var out: String = "Last action: " + action
	if d.get("target") != null and str(d["target"]) != "":
		var target: String = str(d["target"])
		out += " → " + str(name_of.call(target))
	if d.get("text") != null and str(d["text"]) != "":
		var text: String = str(d["text"])
		out += ": “%s”" % (text.substr(0, 90) + "…" if text.length() > 90 else text)
	return out


func _small(parent: Node, variation: String) -> Label:
	var l: Label = Label.new()
	l.theme_type_variation = variation
	l.custom_minimum_size.x = 40
	parent.add_child(l)
	return l


static func _sentence(text: String) -> String:
	return text.substr(0, 1).to_upper() + text.substr(1) if text != "" else ""
