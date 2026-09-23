extends Node2D
## A soft pulsing ellipse drawn under the selected employee's feet (world pixels).

const COLOR := Color("f06fa4")

var _t: float = 0.0


func _process(delta: float) -> void:
	if visible:
		_t += delta
		queue_redraw()


func _draw() -> void:
	var pulse: float = 0.5 + 0.5 * sin(_t * 4.0)
	var rx: float = 8.0 + pulse
	var pts: PackedVector2Array = PackedVector2Array()
	for i: int in range(24):
		var a: float = TAU * float(i) / 24.0
		pts.append(Vector2(cos(a) * rx, sin(a) * rx * 0.4 - 0.5))
	draw_colored_polygon(pts, Color(COLOR, 0.35 + 0.2 * pulse))
	pts.append(pts[0])
	draw_polyline(pts, Color(COLOR, 0.9), 1.0)
