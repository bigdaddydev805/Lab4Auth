extends Node
## Soft procedural ambience, synthesized live (no audio files): rain on the windows, a quiet
## room tone, keyboard clicks while people type, the coffee machine gurgling and a little
## chime when a question gets answered. Off by default; M toggles it.

const RATE := 22050.0

var enabled: bool = false
## Inputs the office updates every frame.
var rain: float = 1.0
var typists: int = 0
var coffee: bool = false

var _player: AudioStreamPlayer
var _playback: AudioStreamGeneratorPlayback
var _lp_rain: float = 0.0
var _lp_rain2: float = 0.0
var _lp_room: float = 0.0
var _lp_coffee: float = 0.0
var _click: float = 0.0
var _click_lp: float = 0.0
var _drop: float = 0.0
var _chimes: Array = []      # {f, t, amp}
var _gain: float = 0.0
var _t: float = 0.0


func _ready() -> void:
	var gen: AudioStreamGenerator = AudioStreamGenerator.new()
	gen.mix_rate = RATE
	gen.buffer_length = 0.25
	_player = AudioStreamPlayer.new()
	_player.stream = gen
	_player.volume_db = -6.0
	add_child(_player)


func set_enabled(on: bool) -> void:
	enabled = on
	if on and not _player.playing:
		_player.play()
		_playback = _player.get_stream_playback()


func toggle() -> bool:
	set_enabled(not enabled)
	return enabled


## A soft two-note chime (a question answered, someone arriving...).
func chime(high: bool = true) -> void:
	var base: float = 1046.5 if high else 784.0
	_chimes.append({"f": base, "t": 0.0, "amp": 0.12})
	_chimes.append({"f": base * 1.26, "t": -0.12, "amp": 0.1})


func _process(_delta: float) -> void:
	if _playback == null:
		return
	if not enabled and _gain < 0.001:
		_player.stop()
		_playback = null
		return
	var frames: int = _playback.get_frames_available()
	var dt: float = 1.0 / RATE
	var click_rate: float = float(typists) * 6.0 / RATE
	for i: int in frames:
		_t += dt
		var target_gain: float = 1.0 if enabled else 0.0
		_gain += (target_gain - _gain) * 0.0005
		var n: float = randf() * 2.0 - 1.0
		# Rain: filtered noise with a slow swell, plus sparse drips on the glass.
		_lp_rain += (n - _lp_rain) * 0.35
		_lp_rain2 += (_lp_rain - _lp_rain2) * 0.08
		var swell: float = 0.75 + 0.25 * sin(_t * 0.23) * sin(_t * 0.071)
		var s: float = (_lp_rain - _lp_rain2) * 0.22 * rain * swell
		if randf() < 0.0009 * rain:
			_drop = 0.35
		_drop *= 0.992
		s += _drop * sin(_t * 2600.0) * 0.12
		# Room tone: very low, very quiet.
		_lp_room += (n - _lp_room) * 0.01
		s += _lp_room * 0.05
		# Keyboards.
		if randf() < click_rate:
			_click = 0.5 + randf() * 0.4
		_click *= 0.965
		_click_lp += (n * _click - _click_lp) * 0.5
		s += (n * _click - _click_lp) * 0.16
		# Coffee machine: bubbling low noise.
		if coffee:
			_lp_coffee += (n - _lp_coffee) * 0.04
			s += _lp_coffee * 0.25 * (0.6 + 0.4 * sin(_t * 23.0 + sin(_t * 3.0) * 4.0))
		# Chimes.
		for c: Dictionary in _chimes:
			c["t"] = float(c["t"]) + dt
			var ct: float = float(c["t"])
			if ct > 0.0:
				s += sin(TAU * float(c["f"]) * ct) * float(c["amp"]) * exp(-ct * 3.0)
		s = clampf(s * _gain, -1.0, 1.0)
		_playback.push_frame(Vector2(s, s))
	_chimes = _chimes.filter(func(c: Dictionary) -> bool: return float(c["t"]) < 2.0)
