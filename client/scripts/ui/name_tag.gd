extends Control
## A small name pill drawn over a character (screen space, native resolution).

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const PAD := Vector2(6, 2)

var font_size: int = 12
var selected: bool = false:
	set(value):
		if selected != value:
			selected = value
			queue_redraw()

var _text: String = ""


func _init() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func set_text(text: String) -> void:
	if text == _text and size != Vector2.ZERO:
		return
	_text = text
	var font: Font = UiTheme.bold_font()
	var w: float = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x
	size = Vector2(ceilf(w) + PAD.x * 2.0, ceilf(font.get_height(font_size)) + PAD.y * 2.0)
	queue_redraw()


func _draw() -> void:
	var bg: Color = UiTheme.PINK if selected else Color(UiTheme.INK, 0.82)
	draw_style_box(UiTheme.box(bg, int(size.y / 2.0)), Rect2(Vector2.ZERO, size))
	var font: Font = UiTheme.bold_font()
	var baseline: float = PAD.y + font.get_ascent(font_size)
	draw_string(font, Vector2(PAD.x, baseline), _text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, UiTheme.CREAM)
