extends RefCounted
## Local simulation clock. The server ticks twice a second; between ticks we extrapolate with
## `sim_now = tick.sim_ms + real_ms_since_tick * effective_scale` (unless paused), and smooth
## small corrections so movement never jumps backwards.

## The last clock payload from the server (date, time, day, phase, scale, ...).
var info: Dictionary = {}
var effective_scale: float = 1.0
var scale: float = 1.0
var paused: bool = false
## Seconds east of UTC for the office's timezone, parsed from `iso`.
var tz_offset_s: int = 0

var _tick_sim_ms: float = 0.0
var _tick_real_ms: int = 0
var _display_ms: float = 0.0
var _has_tick: bool = false


func has_time() -> bool:
	return _has_tick


## Applies a server clock payload (from a `clock` message or a snapshot).
func apply(clock: Dictionary) -> void:
	info = clock
	_tick_sim_ms = float(clock.get("sim_ms", 0))
	_tick_real_ms = Time.get_ticks_msec()
	scale = float(clock.get("scale", 1.0))
	effective_scale = float(clock.get("effective_scale", scale))
	paused = bool(clock.get("paused", false))
	tz_offset_s = _parse_offset(str(clock.get("iso", "")))
	var far: bool = absf(_target() - _display_ms) > maxf(effective_scale * 2000.0, 5000.0)
	if not _has_tick or paused or far:
		_display_ms = _target()
	_has_tick = true


## Advances the smoothed clock; call once per frame.
func update(delta: float) -> void:
	if not _has_tick:
		return
	var target: float = _target()
	if paused:
		_display_ms = target
		return
	var next: float = _display_ms + delta * 1000.0 * effective_scale
	next += (target - next) * clampf(delta * 4.0, 0.0, 1.0)
	_display_ms = maxf(_display_ms, next)


## The current (smoothed) simulation time in Unix epoch milliseconds.
func now_ms() -> float:
	return _display_ms


## "9:05 AM" in the office's timezone.
func format_time(sim_ms: float) -> String:
	var dt: Dictionary = Time.get_datetime_dict_from_unix_time(int(sim_ms / 1000.0) + tz_offset_s)
	var hour: int = int(dt.get("hour", 0))
	var h12: int = hour % 12
	if h12 == 0:
		h12 = 12
	return "%d:%02d %s" % [h12, int(dt.get("minute", 0)), "AM" if hour < 12 else "PM"]


func _target() -> float:
	if paused:
		return _tick_sim_ms
	return _tick_sim_ms + float(Time.get_ticks_msec() - _tick_real_ms) * effective_scale


## Parses the "+HH:MM" / "-HH:MM" / "Z" suffix of an ISO timestamp into seconds.
static func _parse_offset(iso: String) -> int:
	if iso.length() < 6 or iso.ends_with("Z"):
		return 0
	var tail: String = iso.substr(iso.length() - 6)
	var sign_ch: String = tail.substr(0, 1)
	if (sign_ch != "+" and sign_ch != "-") or tail.substr(3, 1) != ":":
		return 0
	var secs: int = int(tail.substr(1, 2)) * 3600 + int(tail.substr(4, 2)) * 60
	return -secs if sign_ch == "-" else secs
