extends Control
## A tiny thought bubble with three bouncing dots, shown while an employee waits on its model.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const BUBBLE := Vector2(28, 15)

var _t: float = 0.0


func _init() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	size = BUBBLE + Vector2(0, 8)


func _process(delta: float) -> void:
	if visible:
		_t += delta
		queue_redraw()


func _draw() -> void:
	# Two trailing puffs toward the head (bottom-left), then the bubble itself.
	draw_circle(Vector2(2.5, BUBBLE.y + 5.5), 2.0, UiTheme.CREAM)
	draw_arc(Vector2(2.5, BUBBLE.y + 5.5), 2.0, 0.0, TAU, 12, UiTheme.INK, 1.0, true)
	draw_circle(Vector2(6.5, BUBBLE.y + 1.5), 2.5, UiTheme.CREAM)
	draw_arc(Vector2(6.5, BUBBLE.y + 1.5), 2.5, 0.0, TAU, 12, UiTheme.INK, 1.0, true)
	draw_style_box(UiTheme.box(UiTheme.CREAM, 7, 2, UiTheme.INK), Rect2(Vector2(3, 0), BUBBLE))
	for i: int in range(3):
		var phase: float = _t * 6.0 - float(i) * 0.9
		var lift: float = maxf(0.0, sin(phase)) * 2.0
		var c: Color = UiTheme.INK.lerp(UiTheme.PINK, maxf(0.0, sin(phase)))
		draw_circle(Vector2(3.0 + 8.0 + float(i) * 6.0, BUBBLE.y * 0.5 + 0.5 - lift), 1.8, c)
