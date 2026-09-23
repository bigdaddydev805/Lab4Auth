extends RefCounted
## One-shot HTTP GET helpers built on HTTPRequest nodes (created under `parent`, freed when done).
## Callbacks receive null on any failure, so callers only need one code path.

const TIMEOUT := 15.0


## GETs `url` and calls `done(body: PackedByteArray or null)`.
static func get_bytes(parent: Node, url: String, done: Callable) -> void:
	var req: HTTPRequest = HTTPRequest.new()
	req.timeout = TIMEOUT
	parent.add_child(req)
	req.request_completed.connect(
		func(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
			req.queue_free()
			if result != HTTPRequest.RESULT_SUCCESS or code != 200:
				done.call(null)
			else:
				done.call(body)
	)
	var err: Error = req.request(url)
	if err != OK:
		req.queue_free()
		done.call(null)


## GETs a PNG and calls `done(texture: ImageTexture or null)`.
static func get_texture(parent: Node, url: String, done: Callable) -> void:
	get_bytes(parent, url, func(body: Variant) -> void:
		if body == null:
			done.call(null)
			return
		var img: Image = Image.new()
		if img.load_png_from_buffer(body) != OK:
			done.call(null)
			return
		done.call(ImageTexture.create_from_image(img))
	)


## GETs JSON and calls `done(data: Variant or null)`.
static func get_json(parent: Node, url: String, done: Callable) -> void:
	get_bytes(parent, url, func(body: Variant) -> void:
		if body == null:
			done.call(null)
			return
		var bytes: PackedByteArray = body
		done.call(JSON.parse_string(bytes.get_string_from_utf8()))
	)
