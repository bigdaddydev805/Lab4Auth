extends Node2D
## The office in world pixels (16 px per tile), scaled by an integer factor by the owner.
## Background at (0, 0); furniture, items and characters share one layer ordered by
## z_index = sort_y * 3 + order (furniture 0, characters 1, items 2), like the server render.

const CharacterSprite := preload("res://scripts/world/character_sprite.gd")
const ArtStore := preload("res://scripts/net/art_store.gd")
const WorldState := preload("res://scripts/model/world_state.gd")

const TILE := 16
const SURFACE_SORT_BIAS := 2   # Items on a surface draw just after that surface.

var state: WorldState
var art: ArtStore
## emp_id -> CharacterSprite for everyone currently present.
var characters: Dictionary = {}
var selected_id: String = ""

var _background: Sprite2D
var _entities: Node2D
var _furniture: Node2D
var _items: Node2D
var _marker: Node2D
var _placement_sort: Dictionary = {}   # placement id -> sort_y


func setup(world_state: WorldState, art_store: ArtStore) -> void:
	state = world_state
	art = art_store
	_background = Sprite2D.new()
	_background.centered = false
	_background.z_index = -10
	add_child(_background)
	_entities = Node2D.new()
	add_child(_entities)
	_furniture = Node2D.new()
	_entities.add_child(_furniture)
	_items = Node2D.new()
	_entities.add_child(_items)
	_marker = preload("res://scripts/world/selection_marker.gd").new()
	_marker.visible = false
	_entities.add_child(_marker)


## Office size in world pixels.
func office_size() -> Vector2:
	var meta: Dictionary = _office_meta()
	var tile: int = int(meta.get("tile", TILE))
	return Vector2(int(meta.get("width", 24)) * tile, int(meta.get("height", 14)) * tile)


func is_ready_to_draw() -> bool:
	return art.has_office_art() and not _office_meta().is_empty()


## Rebuilds everything static (after a snapshot or when the office art arrives).
func rebuild() -> void:
	_background.texture = art.background
	_build_furniture()
	rebuild_items()
	sync_all_employees()


func sync_all_employees() -> void:
	for emp_id: String in characters.keys():
		if not state.employees.has(emp_id):
			_remove_character(emp_id)
	for emp_id: String in state.employees.keys():
		sync_employee(emp_id)


## Creates, updates or removes the character for one employee.
func sync_employee(emp_id: String) -> void:
	var emp: Dictionary = state.employees.get(emp_id, {})
	var present: bool = str(emp.get("status", "")) == "present" and emp.get("look") != null
	if not present:
		_remove_character(emp_id)
		return
	var ch: CharacterSprite = characters.get(emp_id)
	if ch == null:
		ch = CharacterSprite.new()
		ch.setup(emp_id, state.art_meta.get("character", {}))
		_entities.add_child(ch)
		characters[emp_id] = ch
	ch.emp = emp
	ch.holding_kind = str(state.held_item(emp_id).get("kind", ""))
	art.ensure_sheet(emp)
	var tex: Texture2D = art.sheet_for(emp_id)
	if tex != ch.sheet():
		ch.set_sheet(tex)


func on_sheet_ready(emp_id: String) -> void:
	var ch: CharacterSprite = characters.get(emp_id)
	if ch:
		ch.set_sheet(art.sheet_for(emp_id))


## Redraws every item on surfaces and floors (held items are part of the character art).
func rebuild_items() -> void:
	for child: Node in _items.get_children():
		child.queue_free()
	if not art.has_office_art():
		return
	var meta: Dictionary = _office_meta()
	var surfaces: Dictionary = meta.get("surfaces", {})
	var item_sprites: Dictionary = meta.get("items", {})
	var used: Dictionary = {}   # surface id -> {slot: true}
	var pending: Array = []     # items on a surface without a slot
	for it: Dictionary in _sorted_items():
		var loc: Dictionary = it.get("location", {}) if typeof(it.get("location")) == TYPE_DICTIONARY else {}
		var sprite_name: String = str(item_sprites.get(str(it.get("kind", "")), ""))
		if sprite_name == "":
			continue
		if loc.has("on") and surfaces.has(str(loc["on"])):
			if loc.get("slot") == null:
				pending.append(it)
				continue
			var obj: String = str(loc["on"])
			var slot: int = int(loc["slot"])
			if not used.has(obj):
				used[obj] = {}
			used[obj][slot] = true
			_place_on_surface(it, sprite_name, obj, slot)
		elif loc.has("at"):
			var at: Vector2 = CharacterSprite._vec(loc["at"])
			var pos: Vector2 = Vector2(at.x * TILE + 4, at.y * TILE + 8)
			_add_atlas_sprite(_items, sprite_name, pos, int(at.y) * TILE + 15, 2)
	for it: Dictionary in pending:
		var obj: String = str(it["location"]["on"])
		var slots: Array = surfaces[obj]
		if not used.has(obj):
			used[obj] = {}
		var slot: int = 0
		while used[obj].has(slot) and slot < slots.size():
			slot += 1
		used[obj][slot] = true
		_place_on_surface(it, str(item_sprites.get(str(it.get("kind", "")), "")), obj, slot)


## Picks the front-most character under a world-pixel point, or "".
func character_at(world_pos: Vector2) -> String:
	var best: String = ""
	var best_z: int = -1000000
	for emp_id: String in characters.keys():
		var ch: CharacterSprite = characters[emp_id]
		if ch.has_sheet() and ch.pick_rect().has_point(world_pos) and ch.z_index > best_z:
			best = emp_id
			best_z = ch.z_index
	return best


## Per-frame update of every character.
func tick(now_ms: float, delta: float, talking: Dictionary) -> void:
	for emp_id: String in characters.keys():
		var ch: CharacterSprite = characters[emp_id]
		ch.talking = talking.has(emp_id)
		ch.tick(now_ms, delta)
	var sel: CharacterSprite = characters.get(selected_id)
	_marker.visible = sel != null and sel.has_sheet()
	if _marker.visible:
		_marker.position = sel.position
		_marker.z_index = sel.z_index - 1


func _office_meta() -> Dictionary:
	return state.art_meta.get("office", {}) if state else {}


func _build_furniture() -> void:
	for child: Node in _furniture.get_children():
		child.queue_free()
	_placement_sort.clear()
	if not art.has_office_art():
		return
	for p: Variant in _office_meta().get("placements", []):
		var pl: Dictionary = p
		var sort_y: int = int(pl.get("sort_y", 0))
		_placement_sort[str(pl.get("id", ""))] = sort_y
		_add_atlas_sprite(_furniture, str(pl.get("sprite", "")), Vector2(float(pl.get("x", 0)), float(pl.get("y", 0))), sort_y, 0)


func _place_on_surface(it: Dictionary, sprite_name: String, obj: String, slot: int) -> void:
	var slots: Array = _office_meta()["surfaces"][obj]
	if slots.is_empty():
		return
	var sp: Vector2 = CharacterSprite._vec(slots[slot % slots.size()])
	var rect: Dictionary = _sprite_rect(sprite_name)
	var h: float = float(rect.get("h", 8))
	var sort_y: int = int(_placement_sort.get(obj, int(sp.y)))
	_add_atlas_sprite(_items, sprite_name, Vector2(sp.x, sp.y - h + 4.0), sort_y, SURFACE_SORT_BIAS)


func _add_atlas_sprite(parent: Node2D, sprite_name: String, top_left: Vector2, sort_y: int, order: int) -> void:
	var rect: Dictionary = _sprite_rect(sprite_name)
	if rect.is_empty():
		return
	var tex: AtlasTexture = AtlasTexture.new()
	tex.atlas = art.atlas
	tex.region = Rect2(float(rect["x"]), float(rect["y"]), float(rect["w"]), float(rect["h"]))
	var s: Sprite2D = Sprite2D.new()
	s.texture = tex
	s.centered = false
	s.position = top_left
	s.z_index = sort_y * 3 + order
	parent.add_child(s)


func _sprite_rect(sprite_name: String) -> Dictionary:
	var sprites: Dictionary = _office_meta().get("sprites", {})
	return sprites.get(sprite_name, {})


func _sorted_items() -> Array:
	var out: Array = state.items.values()
	out.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return float(a.get("created_ms", 0)) < float(b.get("created_ms", 0)))
	return out


func _remove_character(emp_id: String) -> void:
	var ch: CharacterSprite = characters.get(emp_id)
	if ch:
		ch.queue_free()
		characters.erase(emp_id)
