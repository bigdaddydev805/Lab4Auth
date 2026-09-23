extends Node
## Test/automation helpers driven by user args:
##   --screenshot=out.png      save the rendered frame as PNG
##   --screenshot-at=SECONDS   when to take it (default: 0.5 s before quitting, or 10 s)
##   --screenshot-every=S      instead, save out-001.png, out-002.png, ... every S seconds
##   --quit-after-seconds=N    quit after N seconds (real time)

var _shot_path: String = ""
var _shot_at: float = -1.0
var _quit_after: float = -1.0
var _every: float = -1.0
var _next_every: float = 0.0
var _count: int = 0
var _elapsed: float = 0.0
var _shot_state: int = 0   # 0 pending, 1 capturing, 2 done


func configure(args: Dictionary) -> void:
	_shot_path = str(args.get("screenshot", ""))
	_quit_after = float(args.get("quit-after-seconds", -1.0))
	_shot_at = float(args.get("screenshot-at", -1.0))
	_every = float(args.get("screenshot-every", -1.0))
	_next_every = maxf(_shot_at, _every) if _shot_at >= 0.0 else _every
	if _shot_at < 0.0:
		_shot_at = maxf(_quit_after - 0.5, 0.0) if _quit_after > 0.0 else 10.0
	if _shot_path == "":
		_shot_state = 2
	elif _shot_path.is_relative_path() and not _shot_path.begins_with("res://") and not _shot_path.begins_with("user://"):
		_shot_path = OS.get_environment("PWD").path_join(_shot_path)
	set_process(_shot_path != "" or _quit_after > 0.0)


func _process(delta: float) -> void:
	_elapsed += delta
	if _every > 0.0 and _shot_path != "":
		if _elapsed >= _next_every and _shot_state != 1:
			_next_every += _every
			_count += 1
			_shot_state = 1
			_capture(_shot_path.get_basename() + "-%03d.png" % _count)
		if _quit_after > 0.0 and _elapsed >= _quit_after and _shot_state != 1:
			get_tree().quit()
		return
	if _shot_state == 0 and _elapsed >= _shot_at:
		_shot_state = 1
		_capture(_shot_path)
	if _quit_after > 0.0 and _elapsed >= _quit_after and _shot_state == 2:
		get_tree().quit()


func _capture(path: String) -> void:
	await RenderingServer.frame_post_draw
	var img: Image = get_viewport().get_texture().get_image()
	if img == null or img.is_empty():
		push_warning("Screenshot skipped: nothing was rendered (headless?)")
	else:
		var err: Error = img.save_png(path)
		print("Screenshot %s: %s" % ["saved to " + path if err == OK else "failed", error_string(err)])
	_shot_state = 2
