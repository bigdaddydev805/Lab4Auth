extends RefCounted
## Small persisted client preferences (last server address, panel state) in user://helpco.cfg.

const PATH := "user://helpco.cfg"
const SECTION := "client"

var _cfg: ConfigFile = ConfigFile.new()


func _init() -> void:
	_cfg.load(PATH)  # A missing file just means defaults.


func get_value(key: String, default: Variant = null) -> Variant:
	return _cfg.get_value(SECTION, key, default)


func set_value(key: String, value: Variant) -> void:
	_cfg.set_value(SECTION, key, value)
	var err: Error = _cfg.save(PATH)
	if err != OK:
		push_warning("Could not save settings to %s (%s)" % [PATH, error_string(err)])
