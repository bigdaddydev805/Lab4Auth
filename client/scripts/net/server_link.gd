extends Node
## The WebSocket link to the HelpCo backend (raw WebSocketPeer, not the multiplayer API).
## Connects, reconnects with exponential backoff (1 s to 30 s), parses JSON envelopes and
## emits them by type. Commands go out as {"type": "command", "id", "command"}.

signal message_received(type: String, payload: Variant)
signal link_state_changed(state: int, detail: String)

enum LinkState { OFFLINE, CONNECTING, OPEN }

const MIN_BACKOFF := 1.0
const MAX_BACKOFF := 30.0
const CONNECT_TIMEOUT := 8.0
const INBOUND_BUFFER := 4 * 1024 * 1024

var host_port: String = "127.0.0.1:8765"
var state: int = LinkState.OFFLINE
## Seconds until the next reconnect attempt while OFFLINE.
var retry_in: float = 0.0
## Envelope seq gaps seen (other clients' acks/snapshots also consume seq numbers, so gaps are informational).
var seq_gaps: int = 0

var _ws: WebSocketPeer
var _backoff: float = MIN_BACKOFF
var _connect_left: float = 0.0
var _last_seq: int = -1
var _next_cmd: int = 1


func ws_url() -> String:
	return "ws://%s/ws" % host_port


func http_base() -> String:
	return "http://%s" % host_port


## Starts (or restarts) the connection to `addr` ("host:port").
func start(addr: String) -> void:
	host_port = addr
	_backoff = MIN_BACKOFF
	if _ws:
		_ws.close()
		_ws = null
	_open()


func is_open() -> bool:
	return state == LinkState.OPEN


## Sends a command; returns its id, or "" when not connected.
func send_command(command: Dictionary) -> String:
	if not is_open():
		return ""
	var id: String = "c%d" % _next_cmd
	_next_cmd += 1
	_send({"type": "command", "id": id, "command": command})
	return id


## Asks the server for a fresh snapshot.
func request_resync() -> void:
	if is_open():
		_send({"type": "resync"})


func _send(data: Dictionary) -> void:
	var err: Error = _ws.send_text(JSON.stringify(data))
	if err != OK:
		push_warning("WebSocket send failed: %s" % error_string(err))


func _open() -> void:
	_ws = WebSocketPeer.new()
	_ws.inbound_buffer_size = INBOUND_BUFFER
	_ws.max_queued_packets = 8192
	_last_seq = -1
	var err: Error = _ws.connect_to_url(ws_url())
	if err != OK:
		_ws = null
		_schedule_retry("could not connect (%s)" % error_string(err))
		return
	_connect_left = CONNECT_TIMEOUT
	_set_state(LinkState.CONNECTING, "")


func _process(delta: float) -> void:
	if _ws == null:
		if state == LinkState.OFFLINE:
			retry_in -= delta
			if retry_in <= 0.0:
				_open()
		return
	_ws.poll()
	match _ws.get_ready_state():
		WebSocketPeer.STATE_CONNECTING:
			_connect_left -= delta
			if _connect_left <= 0.0:
				_ws.close()
				_ws = null
				_schedule_retry("timed out")
		WebSocketPeer.STATE_OPEN:
			if state != LinkState.OPEN:
				_backoff = MIN_BACKOFF
				_set_state(LinkState.OPEN, "")
			_drain()
		WebSocketPeer.STATE_CLOSING:
			_drain()
		WebSocketPeer.STATE_CLOSED:
			_drain()
			var code: int = _ws.get_close_code()
			_ws = null
			_schedule_retry("connection closed" if code == -1 else "closed (%d)" % code)


func _drain() -> void:
	while _ws and _ws.get_available_packet_count() > 0:
		_handle_packet(_ws.get_packet())


func _handle_packet(packet: PackedByteArray) -> void:
	var parsed: Variant = JSON.parse_string(packet.get_string_from_utf8())
	if typeof(parsed) != TYPE_DICTIONARY:
		push_warning("Ignoring a non-JSON message from the server")
		return
	var env: Dictionary = parsed
	var seq: int = int(env.get("seq", -1))
	if _last_seq >= 0 and seq > _last_seq + 1:
		seq_gaps += 1
	_last_seq = seq
	message_received.emit(str(env.get("type", "")), env.get("payload"))


func _schedule_retry(why: String) -> void:
	retry_in = _backoff
	_backoff = minf(_backoff * 2.0, MAX_BACKOFF)
	_set_state(LinkState.OFFLINE, why)


func _set_state(new_state: int, detail: String) -> void:
	state = new_state
	link_state_changed.emit(new_state, detail)
