extends Control
## A short coworker chat bubble: auto-sized, width-capped, at most 3 lines (trimmed with "…"),
## pops in, stays ~2.5 s + 60 ms per character (real time), then fades. The overlay positions it;
## `tail_x` is where the tail points (local x), `flipped` puts the tail on top.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const MAX_TEXT_W := 200.0
const MAX_LINES := 3
const PAD := Vector2(10, 6)
const TAIL_H := 7.0
const FONT_SIZE := 14
const FADE_SEC := 0.3

var emp_id: String = ""
var born_msec: int = 0
var lifetime: float = 3.0
var flipped: bool = false:
	set(value):
		if flipped != value:
			flipped = value
			_update_pivot()
			queue_redraw()
var tail_x: float = 0.0:
	set(value):
		if not is_equal_approx(tail_x, value):
			tail_x = value
			queue_redraw()

var _para: TextParagraph = TextParagraph.new()
var _bg: Color = UiTheme.CREAM
var _border: Color = UiTheme.INK
var _text_color: Color = UiTheme.INK
var _age: float = 0.0


## `kind` is "speech" or "intercom" (the owner's voice, drawn in pink).
func setup(id: String, text: String, kind: String = "speech") -> void:
	emp_id = id
	born_msec = Time.get_ticks_msec()
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	if kind == "intercom":
		_bg = Color("ffe6f0")
		_border = Color("c2417a")
	var clean: String = " ".join(text.strip_edges().split(" ", false))
	lifetime = 2.5 + 0.06 * clean.length()
	_set_text(_fit(clean))
	var w: float = 0.0
	for i: int in range(_para.get_line_count()):
		w = maxf(w, _para.get_line_width(i))
	size = Vector2(ceilf(w) + PAD.x * 2.0, ceilf(_para.get_size().y) + PAD.y * 2.0)
	tail_x = size.x * 0.5
	_update_pivot()
	scale = Vector2(0.5, 0.5)
	modulate.a = 0.0


func _ready() -> void:
	var tw: Tween = create_tween().set_parallel(true)
	tw.tween_property(self, "scale", Vector2.ONE, 0.22).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tw.tween_property(self, "modulate:a", 1.0, 0.12)


## Full height including the tail.
func total_height() -> float:
	return size.y + TAIL_H


func is_expired() -> bool:
	return _age >= lifetime


func _process(delta: float) -> void:
	_age += delta
	if _age > lifetime - FADE_SEC:
		modulate.a = clampf((lifetime - _age) / FADE_SEC, 0.0, 1.0)


func _draw() -> void:
	var sb: StyleBoxFlat = UiTheme.box(_bg, 9, 2, _border)
	draw_style_box(sb, Rect2(Vector2.ZERO, size))
	var tx: float = clampf(tail_x, 12.0, size.x - 12.0)
	var base_y: float = 0.0 if flipped else size.y
	var dir: float = -1.0 if flipped else 1.0
	var a: Vector2 = Vector2(tx - 6.0, base_y - dir * 2.0)
	var b: Vector2 = Vector2(tx + 6.0, base_y - dir * 2.0)
	var tip: Vector2 = Vector2(tx - 1.0, base_y + dir * TAIL_H)
	draw_colored_polygon(PackedVector2Array([a, b, tip]), _bg)
	draw_line(Vector2(a.x, base_y - dir * 0.5), tip, _border, 2.0, true)
	draw_line(Vector2(b.x, base_y - dir * 0.5), tip, _border, 2.0, true)
	_para.draw(get_canvas_item(), PAD, _text_color)


func _set_text(text: String) -> void:
	_para.clear()
	_para.width = MAX_TEXT_W
	_para.break_flags = TextServer.BREAK_MANDATORY | TextServer.BREAK_WORD_BOUND | TextServer.BREAK_ADAPTIVE
	_para.add_string(text, UiTheme.regular_font(), FONT_SIZE)


## Trims whole words until the text fits in MAX_LINES lines.
func _fit(text: String) -> String:
	_set_text(text)
	if _para.get_line_count() <= MAX_LINES:
		return text
	var words: PackedStringArray = text.split(" ")
	while words.size() > 1:
		words.remove_at(words.size() - 1)
		var candidate: String = " ".join(words).rstrip(",.;:!?") + "…"
		_set_text(candidate)
		if _para.get_line_count() <= MAX_LINES:
			return candidate
	return text.substr(0, 40) + "…"


func _update_pivot() -> void:
	pivot_offset = Vector2(tail_x, -TAIL_H if flipped else size.y + TAIL_H)
