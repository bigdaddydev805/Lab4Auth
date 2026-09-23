extends RefCounted
## Palette and the shared cozy Theme for all UI, built in code (no theme assets).
## Colors come from the office art: plum outlines, cream paper, pink/yellow accents.

const INK := Color("2a1f33")
const PLUM := Color("3b2e4a")
const PLUM_LIGHT := Color("54446a")
const CREAM := Color("fff7e8")
const PAPER := Color("f6e9d2")
const PAPER_DARK := Color("e6d5b8")
const MUTED := Color("8a7f96")
const PINK := Color("f06fa4")
const YELLOW := Color("f2c14e")
const GREEN := Color("7bc47f")
const BLUE := Color("6cb4e4")
const ORANGE := Color("e8893a")
const RED := Color("d9534f")

const BAR_BG := Color(0.165, 0.121, 0.2, 0.8)

static var _bold: FontVariation
static var _italic: FontVariation


## A rounded flat box.
static func box(bg: Color, radius: int = 8, border: int = 0, border_color: Color = INK,
		pad_h: float = 8.0, pad_v: float = 4.0) -> StyleBoxFlat:
	var sb: StyleBoxFlat = StyleBoxFlat.new()
	sb.bg_color = bg
	sb.set_corner_radius_all(radius)
	sb.set_border_width_all(border)
	sb.border_color = border_color
	sb.content_margin_left = pad_h
	sb.content_margin_right = pad_h
	sb.content_margin_top = pad_v
	sb.content_margin_bottom = pad_v
	sb.anti_aliasing = true
	return sb


static func bold_font() -> Font:
	if _bold == null:
		_bold = FontVariation.new()
		_bold.base_font = ThemeDB.fallback_font
		_bold.variation_embolden = 0.8
	return _bold


static func italic_font() -> Font:
	if _italic == null:
		_italic = FontVariation.new()
		_italic.base_font = ThemeDB.fallback_font
		_italic.variation_transform = Transform2D(Vector2(1.0, 0.0), Vector2(0.2, 1.0), Vector2.ZERO)
	return _italic


static func regular_font() -> Font:
	return ThemeDB.fallback_font


## The Theme used by the HUD and the world overlay.
static func build() -> Theme:
	var t: Theme = Theme.new()
	t.default_font = regular_font()
	t.default_font_size = 14

	t.set_color("font_color", "Label", INK)
	_label_variation(t, "BarLabel", CREAM, 14)
	_label_variation(t, "BarDim", Color(CREAM, 0.7), 13)
	_label_variation(t, "Muted", MUTED, 12)
	_label_variation(t, "Heading", INK, 16, true)
	_label_variation(t, "Small", INK, 12)

	# Light buttons (panels) and dark buttons (bars).
	_button_styles(t, "Button", PAPER, Color("fff1cf"), YELLOW, INK, 2)
	t.set_type_variation("BarButton", "Button")
	_button_styles(t, "BarButton", PLUM_LIGHT, Color("6a5784"), PINK, CREAM, 0)
	t.set_color("font_disabled_color", "BarButton", Color(CREAM, 0.4))
	t.set_type_variation("FlatRow", "Button")
	_button_styles(t, "FlatRow", Color(1, 1, 1, 0), Color(PAPER_DARK, 0.6), Color(YELLOW, 0.5), INK, 0)

	for type_name: String in ["LineEdit"]:
		t.set_stylebox("normal", type_name, box(CREAM, 8, 2, INK, 10, 5))
		t.set_stylebox("focus", type_name, box(Color(0, 0, 0, 0), 8, 2, PINK, 10, 5))
		t.set_stylebox("read_only", type_name, box(PAPER_DARK, 8, 2, MUTED, 10, 5))
		t.set_color("font_color", type_name, INK)
		t.set_color("font_placeholder_color", type_name, MUTED)
		t.set_color("caret_color", type_name, INK)
		t.set_color("selection_color", type_name, Color(PINK, 0.35))

	_button_styles(t, "OptionButton", PAPER, Color("fff1cf"), YELLOW, INK, 2)
	t.set_constant("arrow_margin", "OptionButton", 6)

	t.set_stylebox("panel", "PopupMenu", box(CREAM, 8, 2, INK, 6, 6))
	t.set_stylebox("hover", "PopupMenu", box(YELLOW, 6, 0, INK, 6, 2))
	t.set_color("font_color", "PopupMenu", INK)
	t.set_color("font_hover_color", "PopupMenu", INK)

	t.set_stylebox("panel", "PanelContainer", box(CREAM, 12, 2, INK, 10, 10))
	t.set_type_variation("BarPanel", "PanelContainer")
	t.set_stylebox("panel", "BarPanel", box(BAR_BG, 0, 0, INK, 12, 4))
	t.set_type_variation("Card", "PanelContainer")
	t.set_stylebox("panel", "Card", box(PAPER, 10, 0, INK, 10, 8))
	t.set_type_variation("Chip", "PanelContainer")
	t.set_stylebox("panel", "Chip", box(PLUM_LIGHT, 9, 0, INK, 8, 1))

	t.set_stylebox("tab_selected", "TabContainer", _tab_box(CREAM))
	t.set_stylebox("tab_unselected", "TabContainer", _tab_box(PAPER_DARK))
	t.set_stylebox("tab_hovered", "TabContainer", _tab_box(Color("fff1cf")))
	t.set_stylebox("tab_focus", "TabContainer", StyleBoxEmpty.new())
	t.set_stylebox("panel", "TabContainer", box(Color(0, 0, 0, 0), 0, 0, INK, 2, 6))
	t.set_color("font_selected_color", "TabContainer", INK)
	t.set_color("font_unselected_color", "TabContainer", MUTED)
	t.set_color("font_hovered_color", "TabContainer", INK)
	t.set_font_size("font_size", "TabContainer", 13)

	t.set_stylebox("background", "ProgressBar", box(PAPER_DARK, 5, 0, INK, 0, 0))
	t.set_stylebox("fill", "ProgressBar", box(GREEN, 5, 0, INK, 0, 0))
	t.set_color("font_color", "ProgressBar", INK)

	t.set_color("default_color", "RichTextLabel", INK)
	t.set_font_size("normal_font_size", "RichTextLabel", 13)
	t.set_font_size("bold_font_size", "RichTextLabel", 13)
	t.set_font("bold_font", "RichTextLabel", bold_font())
	t.set_font("italics_font", "RichTextLabel", italic_font())
	t.set_constant("line_separation", "RichTextLabel", 3)

	t.set_stylebox("scroll", "VScrollBar", box(Color(0, 0, 0, 0.06), 4, 0, INK, 3, 3))
	t.set_stylebox("grabber", "VScrollBar", box(Color(PLUM, 0.45), 4, 0, INK, 3, 3))
	t.set_stylebox("grabber_highlight", "VScrollBar", box(Color(PLUM, 0.65), 4, 0, INK, 3, 3))
	t.set_stylebox("grabber_pressed", "VScrollBar", box(PLUM, 4, 0, INK, 3, 3))

	t.set_stylebox("panel", "TooltipPanel", box(INK, 6, 0, INK, 8, 4))
	t.set_color("font_color", "TooltipLabel", CREAM)
	t.set_stylebox("separator", "VSeparator", box(Color(CREAM, 0.15), 0, 0, INK, 1, 0))
	t.set_constant("separation", "VSeparator", 12)
	return t


static func _label_variation(t: Theme, name: String, color: Color, size: int, bold: bool = false) -> void:
	t.set_type_variation(name, "Label")
	t.set_color("font_color", name, color)
	t.set_font_size("font_size", name, size)
	if bold:
		t.set_font("font", name, bold_font())


static func _button_styles(t: Theme, type_name: String, bg: Color, hover: Color, pressed: Color,
		text: Color, border: int) -> void:
	t.set_stylebox("normal", type_name, box(bg, 8, border, INK, 10, 4))
	t.set_stylebox("hover", type_name, box(hover, 8, border, INK, 10, 4))
	t.set_stylebox("pressed", type_name, box(pressed, 8, border, INK, 10, 4))
	t.set_stylebox("hover_pressed", type_name, box(pressed, 8, border, INK, 10, 4))
	t.set_stylebox("disabled", type_name, box(Color(bg, 0.5), 8, border, Color(INK, 0.4), 10, 4))
	t.set_stylebox("focus", type_name, StyleBoxEmpty.new())
	t.set_color("font_color", type_name, text)
	t.set_color("font_hover_color", type_name, text)
	t.set_color("font_focus_color", type_name, text)
	t.set_color("font_pressed_color", type_name, INK)
	t.set_color("font_hover_pressed_color", type_name, INK)
	t.set_font("font", type_name, bold_font())


static func _tab_box(bg: Color) -> StyleBoxFlat:
	var sb: StyleBoxFlat = box(bg, 8, 0, INK, 10, 4)
	sb.corner_radius_bottom_left = 0
	sb.corner_radius_bottom_right = 0
	return sb
