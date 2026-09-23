extends CanvasLayer
## Connection/loading status: a full-screen card before the first snapshot, then a small pill
## while reconnecting. Stays visible in corner mode too.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

var _cover: ColorRect
var _card: PanelContainer
var _title: Label
var _detail: Label
var _pill: PanelContainer
var _pill_label: Label


func _init() -> void:
	layer = 20
	var root: Control = Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.theme = UiTheme.build()
	add_child(root)

	_cover = ColorRect.new()
	_cover.color = Color("3b2e4a")
	_cover.set_anchors_preset(Control.PRESET_FULL_RECT)
	_cover.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(_cover)
	var center: CenterContainer = CenterContainer.new()
	center.set_anchors_preset(Control.PRESET_FULL_RECT)
	center.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_cover.add_child(center)
	_card = PanelContainer.new()
	center.add_child(_card)
	var col: VBoxContainer = VBoxContainer.new()
	col.alignment = BoxContainer.ALIGNMENT_CENTER
	_card.add_child(col)
	_title = Label.new()
	_title.text = "HelpCo"
	_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_title.add_theme_font_override("font", UiTheme.bold_font())
	_title.add_theme_font_size_override("font_size", 28)
	_title.add_theme_color_override("font_color", UiTheme.PINK)
	col.add_child(_title)
	_detail = Label.new()
	_detail.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	col.add_child(_detail)

	_pill = PanelContainer.new()
	_pill.add_theme_stylebox_override("panel", UiTheme.box(Color(UiTheme.INK, 0.9), 12, 2, UiTheme.YELLOW, 14, 6))
	_pill.set_anchors_preset(Control.PRESET_CENTER_TOP)
	_pill.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_pill.offset_top = 44
	_pill.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(_pill)
	_pill_label = Label.new()
	_pill_label.theme_type_variation = "BarLabel"
	_pill.add_child(_pill_label)
	_pill.visible = false


## `ready_to_draw`: we have a snapshot and the office art. `connected`: the socket is open.
func update_status(ready_to_draw: bool, connected: bool, text: String) -> void:
	_cover.visible = not ready_to_draw
	_detail.text = text
	_pill.visible = ready_to_draw and not connected
	_pill_label.text = text
