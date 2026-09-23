extends VBoxContainer
## Small notifications stacked under the top bar (command results, answered questions, errors).

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const LIFETIME := 3.5
const MAX_TOASTS := 4


func _init() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	alignment = BoxContainer.ALIGNMENT_BEGIN
	add_theme_constant_override("separation", 6)


## kind: "info", "good" or "error".
func show_toast(text: String, kind: String = "info") -> void:
	var colors: Dictionary = {"info": UiTheme.CREAM, "good": Color("dff3d8"), "error": Color("fbd9d6")}
	var chip: PanelContainer = PanelContainer.new()
	chip.mouse_filter = Control.MOUSE_FILTER_IGNORE
	chip.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	chip.add_theme_stylebox_override("panel", UiTheme.box(colors.get(kind, UiTheme.CREAM), 10, 2, UiTheme.INK, 12, 6))
	var l: Label = Label.new()
	l.text = text
	chip.add_child(l)
	add_child(chip)
	while get_child_count() > MAX_TOASTS:
		var old: Node = get_child(0)
		remove_child(old)
		old.queue_free()
	chip.modulate.a = 0.0
	var tw: Tween = chip.create_tween()
	tw.tween_property(chip, "modulate:a", 1.0, 0.15)
	tw.tween_interval(LIFETIME)
	tw.tween_property(chip, "modulate:a", 0.0, 0.4)
	tw.tween_callback(chip.queue_free)
