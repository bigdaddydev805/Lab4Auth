extends Node
## HelpCo client entry point. Builds the scene in code and wires the server link, world state,
## art, office view, world overlay and HUD together. The server is the only source of truth:
## this client renders what it says and sends commands.

const CliArgs := preload("res://scripts/cli_args.gd")
const ClientSettings := preload("res://scripts/client_settings.gd")
const WorldState := preload("res://scripts/model/world_state.gd")
const ServerLink := preload("res://scripts/net/server_link.gd")
const ArtStore := preload("res://scripts/net/art_store.gd")
const HttpUtil := preload("res://scripts/net/http_util.gd")
const OfficeView := preload("res://scripts/world/office_view.gd")
const WorldOverlay := preload("res://scripts/ui/world_overlay.gd")
const Hud := preload("res://scripts/ui/hud.gd")
const StatusLayer := preload("res://scripts/ui/status_layer.gd")
const CornerMode := preload("res://scripts/corner_mode.gd")
const CaptureTool := preload("res://scripts/capture_tool.gd")
const Commands := preload("res://scripts/commands.gd")

const DEFAULT_SERVER := "127.0.0.1:8765"
const BACKFILL_EVENTS := 80
const BG_COLOR := Color("2a1f33")

var args: Dictionary = {}
var settings: ClientSettings
var state: WorldState = WorldState.new()
var link: ServerLink
var art: ArtStore
var office: OfficeView
var overlay: WorldOverlay
var hud: Hud
var status_layer: StatusLayer
var corner: CornerMode
var commands: Commands
var selected_id: String = ""

var _pending_select: String = ""
var _world_tween: Tween
var _lists_dirty: bool = false
var _lists_timer: float = 0.0
var _status_timer: float = 0.0
var _hover_cursor: int = -1


func _ready() -> void:
	args = CliArgs.parse()
	settings = ClientSettings.new()
	RenderingServer.set_default_clear_color(BG_COLOR)
	_build()
	_wire()
	var server: String = _resolve_server()
	art.reset("http://" + server)
	link.start(server)
	_pending_select = str(args.get("select", ""))
	var panel_open: bool = bool(settings.get_value("panel_open", true))
	if args.has("panel"):
		panel_open = str(args["panel"]) not in ["off", "false", "0"]
	hud.set_panel_open(panel_open, false)
	if args.has("tab"):
		hud.side_panel.show_tab(str(args["tab"]).capitalize())
	get_viewport().size_changed.connect(func() -> void: _relayout(false))
	_relayout(false)
	_update_status()
	if args.has("corner"):
		corner.set_active.call_deferred(true)


func _build() -> void:
	link = ServerLink.new()
	add_child(link)
	art = ArtStore.new()
	add_child(art)
	office = OfficeView.new()
	office.setup(state, art)
	add_child(office)
	overlay = WorldOverlay.new()
	overlay.setup(office, state)
	add_child(overlay)
	hud = Hud.new()
	add_child(hud)
	status_layer = StatusLayer.new()
	add_child(status_layer)
	corner = CornerMode.new()
	add_child(corner)
	commands = Commands.new()
	commands.setup(link, state, hud.toasts)
	add_child(commands)
	var capture: CaptureTool = CaptureTool.new()
	add_child(capture)
	capture.configure(args)


func _wire() -> void:
	link.message_received.connect(state.apply)
	link.link_state_changed.connect(_on_link_state)
	art.office_art_ready.connect(_on_office_art)
	art.sheet_ready.connect(func(emp_id: String) -> void:
		office.on_sheet_ready(emp_id)
		if emp_id == selected_id:
			_refresh_inspector())

	state.snapshot_applied.connect(_on_snapshot)
	state.employee_changed.connect(_on_employee_changed)
	state.items_changed.connect(func() -> void:
		office.rebuild_items()
		office.sync_all_employees()
		_refresh_inspector())
	state.questions_changed.connect(_refresh_questions)
	state.event_added.connect(_on_event)
	state.decision_changed.connect(func(emp_id: String) -> void:
		if emp_id == selected_id:
			_refresh_inspector())
	state.whiteboard_changed.connect(func() -> void: hud.side_panel.set_whiteboard(state.whiteboard))
	state.clock_changed.connect(_refresh_clock)
	state.ack_received.connect(commands.on_ack)

	var top := hud.top_bar
	top.pause_pressed.connect(func(pause: bool) -> void: commands.clock("pause" if pause else "resume"))
	top.speed_chosen.connect(func(value: float) -> void: commands.clock("scale", value))
	hud.panel_visibility_changed.connect(func(open: bool) -> void:
		settings.set_value("panel_open", open)
		_relayout(true))
	hud.side_panel.employee_selected.connect(_select)
	hud.side_panel.inspector_closed.connect(func() -> void: _select(""))
	var bottom := hud.bottom_bar
	bottom.question_submitted.connect(commands.submit_question)
	bottom.intercom_sent.connect(commands.message)
	bottom.gift_sent.connect(commands.give)
	bottom.hire_pressed.connect(commands.hire)
	corner.changed.connect(func(active: bool) -> void:
		hud.visible = not active
		_relayout(false))


# ------------------------------------------------------------------ server → client

func _on_link_state(link_state: int, detail: String) -> void:
	hud.top_bar.set_connection(link_state, detail)
	match link_state:
		ServerLink.LinkState.OPEN:
			print("HelpCo: connected to %s" % link.ws_url())
		ServerLink.LinkState.OFFLINE:
			print("HelpCo: offline (%s); retrying in %ds" % [detail, ceili(link.retry_in)])
	_update_status()


func _on_snapshot() -> void:
	print("HelpCo: snapshot with %d employees, %d items, %d questions" % [
		state.employees.size(), state.items.size(), state.questions.size()])
	if hud.side_panel.feed.max_seq() > state.last_event_seq:
		hud.side_panel.feed.reset()  # A different (or reset) world.
	art.ensure_office_art()
	if art.has_office_art():
		office.rebuild()
	var osz: Vector2 = office.office_size()
	corner.office_px = Vector2i(osz)
	_refresh_clock()
	_refresh_lists()
	_refresh_questions()
	hud.side_panel.set_whiteboard(state.whiteboard)
	hud.top_bar.set_world_name(state.world_name)
	_backfill_feed()
	if _pending_select != "" and state.employees.has(_pending_select):
		_select(_pending_select)
		_pending_select = ""
	elif selected_id != "" and not state.employees.has(selected_id):
		_select("")
	_refresh_inspector()
	_relayout(false)
	_update_status()


func _on_office_art() -> void:
	office.rebuild()
	_relayout(false)
	_update_status()


func _on_employee_changed(emp_id: String) -> void:
	office.sync_employee(emp_id)
	_lists_dirty = true
	if emp_id == selected_id:
		_refresh_inspector()


func _on_event(ev: Dictionary, live: bool) -> void:
	hud.side_panel.feed.add_event(ev, state.clock.format_time(float(ev.get("sim_ms", 0))))
	if not live:
		return
	var type: String = str(ev.get("type", ""))
	var payload: Dictionary = ev.get("payload", {}) if typeof(ev.get("payload")) == TYPE_DICTIONARY else {}
	var actor: String = str(ev.get("actor", ""))
	match type:
		"speech.said", "employee.introduced", "owner.reply":
			overlay.show_bubble(actor, str(payload.get("text", "")))
		"owner.message":
			var targets: Array = ev.get("targets", []) if typeof(ev.get("targets")) == TYPE_ARRAY else []
			if not targets.is_empty():
				overlay.show_bubble(str(targets[0]), "(intercom) " + str(payload.get("text", "")), "intercom")
		"question.answered":
			hud.toasts.show_toast("%s answered %s!" % [state.employee_name(actor), str(payload.get("id", ""))], "good")


func _backfill_feed() -> void:
	var after: int = maxi(0, state.last_event_seq - BACKFILL_EVENTS)
	var url: String = "%s/api/events?after=%d&limit=%d" % [link.http_base(), after, BACKFILL_EVENTS + 20]
	HttpUtil.get_json(self, url, func(rows: Variant) -> void:
		if typeof(rows) != TYPE_ARRAY:
			return
		for r: Variant in rows:
			if typeof(r) == TYPE_DICTIONARY:
				state.add_event(r, false))


# ------------------------------------------------------------------ selection & panels

func _select(emp_id: String) -> void:
	selected_id = emp_id
	office.selected_id = emp_id
	if emp_id != "":
		hud.bottom_bar.select_recipient(emp_id)
		if not hud.panel_open and not corner.active:
			hud.set_panel_open(true)
		_fetch_last_decision(emp_id)
	_refresh_inspector()
	_lists_dirty = true


## The snapshot has no decisions; fetch the latest one over REST so the card isn't empty.
func _fetch_last_decision(emp_id: String) -> void:
	if state.decisions.has(emp_id):
		return
	HttpUtil.get_json(self, "%s/api/employees/%s" % [link.http_base(), emp_id.uri_encode()], func(data: Variant) -> void:
		if typeof(data) != TYPE_DICTIONARY or state.decisions.has(emp_id):
			return
		var d: Variant = (data as Dictionary).get("last_decision")
		if typeof(d) == TYPE_DICTIONARY and not (d as Dictionary).is_empty():
			state.decisions[emp_id] = d
			if emp_id == selected_id:
				_refresh_inspector())


func _refresh_inspector() -> void:
	var card := hud.side_panel.inspector
	if selected_id == "" or not state.employees.has(selected_id):
		card.visible = false
		return
	card.visible = true
	card.show_employee(state.employees[selected_id], state.decisions.get(selected_id, {}),
		state.held_item(selected_id), art.sheet_for(selected_id), state.art_meta.get("character", {}),
		state.employee_name)


func _refresh_lists() -> void:
	_lists_dirty = false
	var emps: Array = state.employee_list()
	hud.side_panel.team.refresh(emps, selected_id)
	hud.bottom_bar.set_employees(emps)
	if selected_id != "":
		hud.bottom_bar.select_recipient(selected_id)


func _refresh_questions() -> void:
	hud.side_panel.questions.refresh(state.questions, state.employee_name)


func _refresh_clock() -> void:
	var c := state.clock
	hud.top_bar.set_clock(c.info, c.paused, c.scale, c.effective_scale)


# ------------------------------------------------------------------ frame loop & input

func _process(delta: float) -> void:
	state.clock.update(delta)
	var now: float = state.clock.now_ms()
	if state.clock.has_time():
		hud.top_bar.set_time_text(state.clock.format_time(now))
	office.tick(now, delta, overlay.talking())
	overlay.refresh()
	_lists_timer -= delta
	if _lists_dirty and _lists_timer <= 0.0:
		_lists_timer = 0.3
		_refresh_lists()
	_status_timer -= delta
	if _status_timer <= 0.0:
		_status_timer = 0.25
		_update_status()
	_update_hover()


func _unhandled_input(event: InputEvent) -> void:
	var key: InputEventKey = event as InputEventKey
	if key and key.pressed and not key.echo:
		match key.keycode:
			KEY_F2:
				corner.toggle()
			KEY_F3:
				if not corner.active:
					hud.set_panel_open(not hud.panel_open)
			KEY_ESCAPE:
				_select("")
			KEY_SPACE:
				commands.clock("resume" if state.clock.paused else "pause")
			_:
				return
		get_viewport().set_input_as_handled()
		return
	var mb: InputEventMouseButton = event as InputEventMouseButton
	if mb and mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT and not corner.active:
		get_viewport().gui_release_focus()
		var hit: String = office.character_at(office.get_local_mouse_position())
		if hit != "" or selected_id != "":
			_select(hit)
		get_viewport().set_input_as_handled()


func _update_hover() -> void:
	var shape: Input.CursorShape = Input.CURSOR_ARROW
	if not corner.active and office.visible:
		var mouse: Vector2 = get_viewport().get_mouse_position()
		if not hud.is_over_ui(mouse) and office.character_at(office.get_local_mouse_position()) != "":
			shape = Input.CURSOR_POINTING_HAND
	if int(shape) != _hover_cursor:
		_hover_cursor = int(shape)
		Input.set_default_cursor_shape(shape)


# ------------------------------------------------------------------ layout & status

## Scales the office by the largest integer that fits and places it between the bars (shifted
## left when the side panel is open). The bars may cover the outer walls' top edges only.
func _relayout(animate: bool) -> void:
	var vp: Vector2 = get_viewport().get_visible_rect().size
	var osz: Vector2 = office.office_size()
	var top: float = 0.0
	var bottom: float = 0.0
	var right: float = 0.0
	if not corner.active:
		top = hud.top_height()
		bottom = hud.bottom_height()
		right = hud.panel_width() if hud.panel_open else 0.0
	var px: int = maxi(1, int(floor(minf(vp.x / osz.x, vp.y / osz.y))))
	var world: Vector2 = osz * px
	var y: float = round((top + vp.y - bottom - world.y) * 0.5)
	if not corner.active:
		y = minf(maxf(y, top - 4.0 * px), vp.y - world.y)  # Hide at most the wall's top cap.
	var x: float = round((vp.x - right - world.x) * 0.5)
	if right > 0.0:
		x = maxf(x, -(16.0 * px - 8.0))  # May tuck the outer left wall under the screen edge.
	x = minf(x, round((vp.x - world.x) * 0.5))
	var target: Vector2 = Vector2(x, y)
	office.scale = Vector2(px, px)
	if _world_tween:
		_world_tween.kill()
	if animate:
		_world_tween = create_tween()
		_world_tween.tween_property(office, "position", target, 0.22) \
			.set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
	else:
		office.position = target
	var margin: float = 8.0 if not corner.active else 2.0
	overlay.safe_rect = Rect2(margin, top + 4.0, vp.x - right - margin * 2.0, vp.y - top - bottom - 8.0)


func _update_status() -> void:
	var ready: bool = state.has_snapshot and office.is_ready_to_draw()
	office.visible = ready
	var text: String = ""
	match link.state:
		ServerLink.LinkState.OPEN:
			text = "Loading the office…" if not ready else ""
		ServerLink.LinkState.CONNECTING:
			text = "Connecting to %s…" % link.host_port
		_:
			text = "Can't reach the HelpCo server at %s. Retrying in %ds…" % [link.host_port, ceili(link.retry_in)]
	status_layer.update_status(ready, link.is_open(), text)


func _resolve_server() -> String:
	var addr: String = str(args.get("server", "")).strip_edges()
	if addr != "":
		addr = addr.trim_prefix("ws://").trim_prefix("http://").trim_suffix("/").trim_suffix("/ws")
		settings.set_value("server", addr)
		return addr
	return str(settings.get_value("server", DEFAULT_SERVER))
