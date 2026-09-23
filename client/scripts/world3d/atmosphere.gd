extends Node3D
## Mood and weather: sky/ambient color and the sun (or moon) follow the office's local time;
## lamps glow brighter as it gets dark; windows show the sky with rain running down the glass;
## rain falls around the building; dust drifts in the lamplight.
## Weather is ambience only (the simulation doesn't know about it): each calendar day is rainy
## or clear, picked from the date so every client sees the same thing. R cycles auto/rain/clear.

const Kit := preload("res://scripts/world3d/kit.gd")

const DAY_MS := 86400000.0
const WINDOW_SHADER := """
shader_type spatial;
render_mode unshaded, cull_disabled;
uniform vec3 sky_top : source_color = vec3(0.12, 0.16, 0.33);
uniform vec3 sky_bottom : source_color = vec3(0.35, 0.28, 0.48);
uniform vec3 city : source_color = vec3(1.0, 0.75, 0.45);
uniform float city_lights = 1.0;
uniform float rain = 1.0;
uniform float energy = 1.2;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void fragment() {
	vec2 uv = UV;
	vec3 col = mix(sky_top, sky_bottom, smoothstep(0.0, 1.0, uv.y));
	// Distant windows of the city, low on the glass.
	vec2 cg = uv * vec2(14.0, 20.0);
	vec2 cid = floor(cg);
	float lit = step(0.72, hash(cid)) * step(13.0, cid.y) * city_lights;
	vec2 cf = fract(cg) - 0.5;
	col += city * lit * (1.0 - smoothstep(0.18, 0.32, max(abs(cf.x), abs(cf.y)))) * 0.8;
	// Rain running down the glass.
	float cols = 16.0;
	float cx = floor(uv.x * cols);
	float r = hash(vec2(cx, 3.0));
	float t = fract(uv.y * 0.8 - TIME * (0.25 + r * 0.5) + r * 7.0);
	float streak = smoothstep(0.0, 0.03, t) * (1.0 - smoothstep(0.03, 0.35, t));
	float xw = abs(fract(uv.x * cols) - 0.5 + (r - 0.5) * 0.3);
	streak *= 1.0 - smoothstep(0.02, 0.07, xw);
	col += vec3(0.65, 0.75, 1.0) * streak * rain * 0.55;
	// Beads of water.
	vec2 g = uv * vec2(9.0, 11.0);
	vec2 id = floor(g);
	vec2 c = fract(g) - 0.5 - (vec2(hash(id + 3.1), hash(id + 7.7)) - 0.5) * 0.6;
	float bead = (1.0 - smoothstep(0.035, 0.07, length(c))) * step(0.5, hash(id + 1.3));
	col += vec3(0.7, 0.8, 1.0) * bead * rain * 0.45;
	ALBEDO = col * energy;
}
"""

## Keyframes over the local day: hour → mood.
const KEYS := [
	{"h": 0.0, "bg": "#0f0c1a", "amb": "#4a4a88", "amb_e": 0.22, "sun": "#8ea0e8", "sun_e": 0.28, "top": "#0b1028", "bot": "#1b2244", "lamps": 1.0, "city": 1.0},
	{"h": 5.5, "bg": "#15112a", "amb": "#50508e", "amb_e": 0.24, "sun": "#8ea0e8", "sun_e": 0.25, "top": "#141a3a", "bot": "#2c2a52", "lamps": 1.0, "city": 1.0},
	{"h": 7.0, "bg": "#3a2c48", "amb": "#b08aa0", "amb_e": 0.34, "sun": "#ffb88a", "sun_e": 0.55, "top": "#5a5a8e", "bot": "#e6a88a", "lamps": 0.85, "city": 0.4},
	{"h": 10.0, "bg": "#262a3e", "amb": "#a8b0d0", "amb_e": 0.36, "sun": "#e8e4ff", "sun_e": 0.55, "top": "#7488b0", "bot": "#b0b8cc", "lamps": 0.7, "city": 0.0},
	{"h": 15.0, "bg": "#262a3e", "amb": "#b0b0d0", "amb_e": 0.36, "sun": "#fff0dc", "sun_e": 0.55, "top": "#7488b0", "bot": "#b8b8c8", "lamps": 0.72, "city": 0.0},
	{"h": 17.8, "bg": "#2a1f3a", "amb": "#9a74b0", "amb_e": 0.34, "sun": "#ff9a6a", "sun_e": 0.5, "top": "#3c4a78", "bot": "#e88a6a", "lamps": 0.95, "city": 0.5},
	{"h": 19.5, "bg": "#1b1628", "amb": "#6a6aa8", "amb_e": 0.28, "sun": "#8ea0e8", "sun_e": 0.35, "top": "#1a2250", "bot": "#4a3a6a", "lamps": 1.0, "city": 1.0},
	{"h": 24.0, "bg": "#0f0c1a", "amb": "#4a4a88", "amb_e": 0.22, "sun": "#8ea0e8", "sun_e": 0.28, "top": "#0b1028", "bot": "#1b2244", "lamps": 1.0, "city": 1.0},
]
const WEATHER_MODES := ["auto", "rain", "clear"]

var weather_mode: String = "auto"
## For screenshots/testing: pin the local hour (negative = follow the clock).
var force_hour: float = -1.0
## 0..1, eased towards today's weather.
var rain: float = 1.0
var hour: float = 18.5

var env: Environment
var sun: DirectionalLight3D
var _window_mat: ShaderMaterial
var _lamps: Array = []          # {light, base, flicker}
var _window_lights: Array = []
var _rain_emitters: Array[CPUParticles3D] = []
var _dust: CPUParticles3D
var _t: float = 0.0


func _ready() -> void:
	env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color("#1b1628")
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color("#6a6aa8")
	env.ambient_light_energy = 0.28
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.tonemap_exposure = 1.08
	env.glow_enabled = true
	env.glow_intensity = 0.85
	env.glow_bloom = 0.22
	env.glow_hdr_threshold = 0.95
	env.adjustment_enabled = true
	env.adjustment_saturation = 1.08
	var we: WorldEnvironment = WorldEnvironment.new()
	we.environment = env
	add_child(we)
	sun = DirectionalLight3D.new()
	sun.shadow_enabled = true
	sun.shadow_blur = 1.5
	sun.directional_shadow_max_distance = 60.0
	add_child(sun)
	_build_rain()
	_build_dust()


func window_material() -> ShaderMaterial:
	if _window_mat == null:
		var sh: Shader = Shader.new()
		sh.code = WINDOW_SHADER
		_window_mat = ShaderMaterial.new()
		_window_mat.shader = sh
	return _window_mat


func clear_fixtures() -> void:
	_lamps.clear()
	_window_lights.clear()


func add_lamp(light: OmniLight3D, base_energy: float, flicker: float = 0.0) -> void:
	_lamps.append({"light": light, "base": base_energy, "flicker": flicker, "seed": randf() * 100.0})


func add_window_light(light: OmniLight3D) -> void:
	_window_lights.append(light)


func cycle_weather() -> String:
	weather_mode = WEATHER_MODES[(WEATHER_MODES.find(weather_mode) + 1) % WEATHER_MODES.size()]
	return weather_mode


## True when it's raining (or about to) today.
func raining_today(local_ms: float) -> bool:
	if weather_mode != "auto":
		return weather_mode == "rain"
	var day: int = int(floor(local_ms / DAY_MS))
	return (hash(day * 7919) % 10) < 7   # mostly rainy: the lo-fi mood


## Called every frame with the office's local wall-clock time in ms.
func update_time(local_ms: float, delta: float) -> void:
	_t += delta
	hour = fmod(local_ms, DAY_MS) / 3600000.0 if force_hour < 0.0 else force_hour
	var target_rain: float = 1.0 if raining_today(local_ms) else 0.0
	rain = move_toward(rain, target_rain, delta * 0.25)
	var k: Dictionary = _sample(hour)
	var clear: float = 1.0 - rain
	var daylight: float = clampf((1.0 - float(k["lamps"])) / 0.3, 0.0, 1.0)
	env.background_color = (k["bg"] as Color).lerp(Color("#3a4a6a"), clear * daylight * 0.4)
	env.ambient_light_color = k["amb"]
	env.ambient_light_energy = float(k["amb_e"]) * (1.0 + clear * 0.25 * daylight)
	sun.light_color = k["sun"]
	sun.light_energy = float(k["sun_e"]) * (1.0 + clear * 0.6 * daylight) * lerpf(0.7, 1.0, clear)
	# Sun path: rises in the east (behind the left wall), sets west; the moon at night.
	var day_t: float = clampf((hour - 6.0) / 13.0, 0.0, 1.0)
	var elev: float = lerpf(18.0, 58.0, sin(day_t * PI)) if hour > 6.0 and hour < 19.0 else 45.0
	var az: float = lerpf(100.0, 200.0, day_t) if hour > 6.0 and hour < 19.0 else 150.0
	sun.rotation_degrees = Vector3(-elev, az, 0.0)
	var lamps: float = float(k["lamps"])
	for l: Dictionary in _lamps:
		var light: OmniLight3D = l["light"]
		var f: float = 1.0 + 0.03 * sin(_t * 3.0 + float(l["seed"])) * float(l["flicker"])
		light.light_energy = float(l["base"]) * lamps * f
	for wl: OmniLight3D in _window_lights:
		wl.light_color = (k["top"] as Color).lerp(k["bot"], 0.4).lightened(0.2)
		wl.light_energy = 0.5 + daylight * (0.6 + clear * 0.5)
	if _window_mat:
		_window_mat.set_shader_parameter("sky_top", (k["top"] as Color).lerp(Color("#6fa8e8"), clear * daylight * 0.6))
		_window_mat.set_shader_parameter("sky_bottom", (k["bot"] as Color).lerp(Color("#bfe0ff"), clear * daylight * 0.5))
		_window_mat.set_shader_parameter("city_lights", float(k["city"]))
		_window_mat.set_shader_parameter("rain", rain)
		_window_mat.set_shader_parameter("energy", 1.1 + daylight * 0.1)
	for e: CPUParticles3D in _rain_emitters:
		e.emitting = rain > 0.3
	_dust.emitting = true


func _sample(h: float) -> Dictionary:
	for i: int in range(KEYS.size() - 1):
		var a: Dictionary = KEYS[i]
		var b: Dictionary = KEYS[i + 1]
		if h >= float(a["h"]) and h <= float(b["h"]):
			var t: float = (h - float(a["h"])) / maxf(float(b["h"]) - float(a["h"]), 0.001)
			t = t * t * (3.0 - 2.0 * t)
			var out: Dictionary = {}
			for key: String in a.keys():
				if key == "h":
					continue
				var va: Variant = a[key]
				if va is String:
					out[key] = Color(va).lerp(Color(str(b[key])), t)
				else:
					out[key] = lerpf(float(va), float(b[key]), t)
			return out
	return _sample(0.0)


func _build_rain() -> void:
	var drop: BoxMesh = BoxMesh.new()
	drop.size = Vector3(0.015, 0.45, 0.015)
	var m: StandardMaterial3D = StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color = Color(0.6, 0.7, 1.0, 0.35)
	drop.material = m
	# Four slabs of rain around the building (never inside: there's a roof we don't draw).
	# Behind the building rain falls from high up; in front (between the camera and the office)
	# it only falls low and further out, so it never streaks across the interior.
	for zone: Array in [[Vector3(12, 5, -4.5), Vector3(22, 0.2, 3.5), 0.42], [Vector3(-3.5, 5, 7), Vector3(2.5, 0.2, 9), 0.42],
			[Vector3(12, 1.6, 21.5), Vector3(24, 0.1, 4.0), 0.16], [Vector3(31.0, 1.6, 7), Vector3(4.0, 0.1, 16), 0.16]]:
		var p: CPUParticles3D = CPUParticles3D.new()
		p.position = zone[0]
		p.lifetime = zone[2]
		p.amount = 260 if p.lifetime > 0.3 else 140
		p.preprocess = 1.0
		p.emission_shape = CPUParticles3D.EMISSION_SHAPE_BOX
		p.emission_box_extents = zone[1]
		p.direction = Vector3(0.08, -1, 0.04)
		p.spread = 2.0
		p.gravity = Vector3.ZERO
		p.initial_velocity_min = 13.0
		p.initial_velocity_max = 15.0
		p.mesh = drop
		p.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(p)
		_rain_emitters.append(p)


func _build_dust() -> void:
	_dust = CPUParticles3D.new()
	_dust.position = Vector3(12, 1.3, 7)
	_dust.amount = 70
	_dust.lifetime = 14.0
	_dust.preprocess = 14.0
	_dust.emission_shape = CPUParticles3D.EMISSION_SHAPE_BOX
	_dust.emission_box_extents = Vector3(11, 1.0, 6)
	_dust.direction = Vector3(1, 0.2, 0.3)
	_dust.spread = 180.0
	_dust.gravity = Vector3(0, -0.004, 0)
	_dust.initial_velocity_min = 0.02
	_dust.initial_velocity_max = 0.07
	var g: Gradient = Gradient.new()
	g.set_color(0, Color(1, 0.9, 0.7, 0.0))
	g.add_point(0.5, Color(1, 0.9, 0.7, 0.55))
	g.set_color(g.get_point_count() - 1, Color(1, 0.9, 0.7, 0.0))
	_dust.color_ramp = g
	var sm: SphereMesh = SphereMesh.new()
	sm.radius = 0.012
	sm.height = 0.024
	sm.radial_segments = 4
	sm.rings = 2
	var m: StandardMaterial3D = StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.vertex_color_use_as_albedo = true
	m.emission_enabled = true
	m.emission = Color(1, 0.85, 0.6)
	m.emission_energy_multiplier = 1.5
	sm.material = m
	_dust.mesh = sm
	_dust.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_dust)
