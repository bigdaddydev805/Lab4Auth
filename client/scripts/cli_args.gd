extends RefCounted
## Parses user arguments given after `--` on the command line, e.g.
## `godot --path client -- --server=127.0.0.1:8790 --select=emp_1`.
## `--key=value` becomes {"key": "value"}; a bare `--flag` becomes {"flag": "true"}.


static func parse() -> Dictionary:
	var out: Dictionary = {}
	for raw: String in OS.get_cmdline_user_args():
		if not raw.begins_with("--"):
			continue
		var body: String = raw.substr(2)
		var eq: int = body.find("=")
		if eq == -1:
			out[body] = "true"
		else:
			out[body.substr(0, eq)] = body.substr(eq + 1)
	return out
