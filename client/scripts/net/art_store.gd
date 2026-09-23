extends Node
## Downloads the server-drawn art and keeps it as textures: the office background, the
## furniture/item atlas, and one animation sheet per employee (refetched when `look_key` changes).

const HttpUtil := preload("res://scripts/net/http_util.gd")

signal office_art_ready
signal sheet_ready(emp_id: String)

const SHEET_RETRY_SEC := 10.0

var base_url: String = ""
var background: Texture2D
var atlas: Texture2D
## emp_id -> Texture2D for the look currently shown.
var sheets: Dictionary = {}

var _office_loading: bool = false
var _office_base: String = ""
## emp_id -> look_key of the sheet we have or are fetching.
var _sheet_keys: Dictionary = {}
## emp_id -> ticks (ms) of the last failed fetch, to avoid hammering the server.
var _sheet_failed_at: Dictionary = {}


func has_office_art() -> bool:
	return background != null and atlas != null and _office_base == base_url


## Fetches the background and atlas unless we already have them from this server.
func ensure_office_art() -> void:
	if has_office_art() or _office_loading or base_url == "":
		return
	_office_loading = true
	var from: String = base_url
	var parts: Dictionary = {}
	var finish := func() -> void:
		if parts.size() < 2:
			return
		_office_loading = false
		if parts["bg"] == null or parts["atlas"] == null or from != base_url:
			push_warning("Could not load the office art from %s" % from)
			return
		background = parts["bg"]
		atlas = parts["atlas"]
		_office_base = from
		office_art_ready.emit()
	HttpUtil.get_texture(self, from + "/art/office_bg.png", func(tex: Variant) -> void:
		parts["bg"] = tex
		finish.call()
	)
	HttpUtil.get_texture(self, from + "/art/office_atlas.png", func(tex: Variant) -> void:
		parts["atlas"] = tex
		finish.call()
	)


## Makes sure we have (or are fetching) the sheet for this employee's current look.
func ensure_sheet(emp: Dictionary) -> void:
	var emp_id: String = str(emp.get("id", ""))
	var look_key: String = str(emp.get("look_key", ""))
	if emp_id == "" or look_key == "" or emp.get("look") == null:
		return
	if _sheet_keys.get(emp_id, "") == look_key:
		return
	var failed: int = int(_sheet_failed_at.get(emp_id + look_key, -100000))
	if Time.get_ticks_msec() - failed < SHEET_RETRY_SEC * 1000.0:
		return
	_sheet_keys[emp_id] = look_key
	var url: String = "%s/art/employee/%s.png?v=%s" % [base_url, emp_id.uri_encode(), look_key.uri_encode()]
	HttpUtil.get_texture(self, url, func(tex: Variant) -> void:
		if _sheet_keys.get(emp_id, "") != look_key:
			return  # A newer look was requested meanwhile.
		if tex == null:
			_sheet_keys.erase(emp_id)
			_sheet_failed_at[emp_id + look_key] = Time.get_ticks_msec()
			return
		sheets[emp_id] = tex
		sheet_ready.emit(emp_id)
	)


func sheet_for(emp_id: String) -> Texture2D:
	return sheets.get(emp_id) as Texture2D


## Forgets everything (used when switching servers).
func reset(new_base_url: String) -> void:
	base_url = new_base_url
	background = null
	atlas = null
	_office_base = ""
	sheets.clear()
	_sheet_keys.clear()
	_sheet_failed_at.clear()
