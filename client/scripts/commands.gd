extends Node
## Sends owner commands to the server and turns their `ack` replies into friendly toasts.

const ServerLink := preload("res://scripts/net/server_link.gd")
const WorldState := preload("res://scripts/model/world_state.gd")
const Toasts := preload("res://scripts/ui/toasts.gd")

var _link: ServerLink
var _state: WorldState
var _toasts: Toasts
## command id -> {"kind": ..., plus context for the toast}
var _pending: Dictionary = {}


func setup(link: ServerLink, state: WorldState, toasts: Toasts) -> void:
	_link = link
	_state = state
	_toasts = toasts


func submit_question(text: String) -> void:
	_send({"type": "submit_question", "text": text}, {"kind": "question"})


func message(emp_id: String, text: String) -> void:
	_send({"type": "message", "to": emp_id, "text": text}, {"kind": "message", "to": emp_id})


func give(emp_id: String, kind: String) -> void:
	_send({"type": "give", "to": emp_id, "kind": kind}, {"kind": "give", "to": emp_id})


func hire() -> void:
	_send({"type": "hire"}, {"kind": "hire"})


## action: "pause" | "resume" | "scale" (with value).
func clock(action: String, value: float = 0.0) -> void:
	var cmd: Dictionary = {"type": "clock", "action": action}
	if action == "scale":
		cmd["value"] = value
	_send(cmd, {"kind": "clock"})


func on_ack(id: String, result: Dictionary) -> void:
	var ctx: Dictionary = _pending.get(id, {})
	_pending.erase(id)
	if ctx.is_empty():
		return  # Not ours.
	var ok: bool = bool(result.get("ok", false))
	var err: String = str(result.get("error", "")) if result.get("error") != null else ""
	var who: String = _state.employee_name(str(ctx.get("to", "")))
	match str(ctx["kind"]):
		"question":
			_toast("Question %s is on the board" % str(result.get("id", "")) if ok else "Couldn't send: " + err, ok)
		"message":
			if ok and bool(result.get("delivered", true)):
				_toasts.show_toast("%s heard you over the intercom" % who)
			elif ok:
				_toasts.show_toast("%s isn't in right now; they'll remember it" % who)
			else:
				_toast("Couldn't reach them: " + err, false)
		"give":
			_toast("Gift sent to %s" % who if ok else "Couldn't give it: " + err, ok)
		"hire":
			_toast("Hired someone new! They're choosing who to be." if ok else "Can't hire: " + err, ok)
		_:
			if not ok:
				_toast("The server said no: " + err, false)


func _send(cmd: Dictionary, ctx: Dictionary) -> void:
	var id: String = _link.send_command(cmd)
	if id == "":
		_toasts.show_toast("Not connected to the server", "error")
		return
	_pending[id] = ctx


func _toast(text: String, ok: bool) -> void:
	_toasts.show_toast(text, "good" if ok else "error")
