extends CanvasLayer
## Crisp screen-space UI over the world: name tags, "thinking" dots and speech bubbles.
## Each frame they are placed over their character via the camera's projection, flipped
## below the character when there is no room above, kept inside `safe_rect`, and stacked so
## bubbles don't overlap.

const Office3D := preload("res://scripts/world3d/office3d.gd")
const WorldState := preload("res://scripts/model/world_state.gd")
const NameTag := preload("res://scripts/ui/name_tag.gd")
const ThinkingDots := preload("res://scripts/ui/thinking_dots.gd")
const SpeechBubble := preload("res://scripts/ui/speech_bubble.gd")
const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const GAP := 3.0

var office: Office3D
var state: WorldState
## Screen area overlay elements must stay inside (excludes the HUD bars and side panel).
var safe_rect: Rect2 = Rect2(0, 0, 1280, 720)

var _root: Control
var _tags: Dictionary = {}      # emp_id -> NameTag
var _thinking: Dictionary = {}  # emp_id -> ThinkingDots
var _bubbles: Array = []        # SpeechBubble, oldest first


func setup(office_view: Office3D, world_state: WorldState) -> void:
	office = office_view
	state = world_state
	layer = 5
	_root = Control.new()
	_root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_root.set_anchors_preset(Control.PRESET_FULL_RECT)
	_root.theme = UiTheme.build()
	add_child(_root)


## Shows a bubble over `emp_id`, replacing their previous bubble of the same kind.
func show_bubble(emp_id: String, text: String, kind: String = "speech") -> void:
	if text.strip_edges() == "" or not office.characters.has(emp_id):
		return
	for b: SpeechBubble in _bubbles.duplicate():
		if b.emp_id == emp_id and b.kind == kind:
			_remove_bubble(b)
	var bubble: SpeechBubble = SpeechBubble.new()
	bubble.setup(emp_id, text, kind)
	_root.add_child(bubble)
	_bubbles.append(bubble)
	_place_all()


## Employees with a bubble up (drives the talk animation).
func talking() -> Dictionary:
	var out: Dictionary = {}
	for b: SpeechBubble in _bubbles:
		if b.kind == "speech":
			out[b.emp_id] = true
	return out


func clear() -> void:
	for b: SpeechBubble in _bubbles.duplicate():
		_remove_bubble(b)


## Called every frame after the characters moved.
func refresh() -> void:
	for b: SpeechBubble in _bubbles.duplicate():
		if b.is_expired() or not office.characters.has(b.emp_id):
			_remove_bubble(b)
	_sync_nodes()
	_place_all()


func _sync_nodes() -> void:
	for emp_id: String in _tags.keys():
		if not office.characters.has(emp_id):
			_tags[emp_id].queue_free()
			_thinking[emp_id].queue_free()
			_tags.erase(emp_id)
			_thinking.erase(emp_id)
	for emp_id: String in office.characters.keys():
		if not _tags.has(emp_id):
			var tag: NameTag = NameTag.new()
			_root.add_child(tag)
			_tags[emp_id] = tag
			var dots: ThinkingDots = ThinkingDots.new()
			_root.add_child(dots)
			_thinking[emp_id] = dots


func _place_all() -> void:
	var anchors: Dictionary = {}  # emp_id -> {x, above_y, below_y}
	for emp_id: String in office.characters.keys():
		var tag: NameTag = _tags.get(emp_id)
		var dots: ThinkingDots = _thinking.get(emp_id)
		if tag == null:
			continue
		var a: Dictionary = office.anchor_of(emp_id)
		var shown: bool = not a.is_empty()
		tag.visible = shown
		if not shown:
			dots.visible = false
			continue
		var emp: Dictionary = state.employees.get(emp_id, {})
		var feet: Vector2 = a["feet"]
		var head: Vector2 = a["head"]
		tag.font_size = 12 if feet.y - head.y > 60.0 else 11
		tag.set_text(str(emp.get("name", emp_id)))
		tag.selected = emp_id == office.selected_id
		var pos: Vector2 = Vector2(head.x - tag.size.x * 0.5, head.y - tag.size.y - GAP)
		var below: bool = pos.y < safe_rect.position.y
		if below:
			pos.y = feet.y + GAP
		pos.x = clampf(pos.x, safe_rect.position.x, safe_rect.end.x - tag.size.x)
		tag.position = pos.round()
		dots.visible = bool(emp.get("thinking", false))
		if dots.visible:
			dots.position = (pos + Vector2(tag.size.x + 1.0, -dots.size.y + tag.size.y * 0.5 + 2.0)).round()
			if below:
				dots.position.y = pos.y + tag.size.y * 0.5 - dots.size.y * 0.5
		anchors[emp_id] = {"x": head.x, "above": pos.y - GAP, "below": pos.y + tag.size.y + GAP if below else feet.y + GAP, "tag_below": below}
	_place_bubbles(anchors)


func _place_bubbles(anchors: Dictionary) -> void:
	var placed: Array = []  # Rect2 of bubbles already placed (incl. tails)
	for b: SpeechBubble in _bubbles:
		var a: Dictionary = anchors.get(b.emp_id, {})
		if a.is_empty():
			b.visible = false
			continue
		b.visible = true
		var h: float = b.total_height()
		var flip: bool = bool(a["tag_below"]) or float(a["above"]) - h < safe_rect.position.y
		var x: float = clampf(float(a["x"]) - b.size.x * 0.5, safe_rect.position.x, safe_rect.end.x - b.size.x)
		var rect: Rect2 = Rect2(x, float(a["below"]) if flip else float(a["above"]) - h, b.size.x, h)
		for _i: int in range(6):
			var hit: bool = false
			for other: Rect2 in placed:
				if rect.grow(2.0).intersects(other):
					rect.position.y = other.end.y + 2.0 if flip else other.position.y - h - 2.0
					hit = true
			if not hit:
				break
		placed.append(rect)
		b.flipped = flip
		b.position = Vector2(rect.position.x, rect.position.y + (SpeechBubble.TAIL_H if flip else 0.0)).round()
		b.tail_x = float(a["x"]) - b.position.x


func _remove_bubble(b: SpeechBubble) -> void:
	_bubbles.erase(b)
	b.queue_free()
