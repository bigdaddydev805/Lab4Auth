extends RefCounted
## Low-poly furniture and small items. `build(parent, obj, fx)` makes the model for one server
## office object (desk, couch, printer, ...) at its tile position; `item()` makes a small item
## (mug, duck, ...). Anything that moves or glows is registered in `fx` for the office to
## animate: "sway" (plants), "blink" (LEDs), "screens" (monitors), "steam" (emitters).

const Kit := preload("res://scripts/world3d/kit.gd")

const WOOD := "#b98357"
const WOOD_DARK := "#8d5f3d"
const PLUM := "#3a3548"
const METAL := "#2d2a38"

## Heights of each kind's top surface, for items placed on it.
const SURFACE_Y := {"desk": 0.76, "counter": 0.94, "coffee_table": 0.37, "meeting_table": 0.76}


static func build(parent: Node3D, obj: Dictionary, fx: Dictionary) -> Node3D:
	var x: float = float(obj.get("x", 0))
	var y: float = float(obj.get("y", 0))
	var w: float = float(obj.get("w", 1))
	var h: float = float(obj.get("h", 1))
	var root: Node3D = Kit.node(parent, Vector3(x + w * 0.5, 0, y + h * 0.5))
	root.name = str(obj.get("id", "obj"))
	match str(obj.get("kind", "")):
		"desk":
			_desk(root, obj, w, fx)
		"desk_chair":
			_chair(root, 0.0, "#4a4458")
		"printer":
			_printer(root, fx)
		"question_board":
			_question_board(root, w, fx)
		"plant_big":
			plant(root, 1.4, fx)
		"plant_small":
			plant(root, 0.9, fx)
		"fridge":
			_fridge(root)
		"coffee_machine":
			_coffee_machine(root, fx)
		"counter":
			_counter(root, w)
		"couch":
			_couch(root, w)
		"coffee_table":
			_coffee_table(root, w)
		"meeting_table":
			_meeting_table(root, obj, w, h)
		"whiteboard":
			_whiteboard(root, w, fx)
		"coat_rack":
			_coat_rack(root)
		"door":
			_door(root)
		_:
			Kit.box(root, Vector3(0.7, 0.7, 0.7), "#9a8fb0", Vector3.ZERO)
	return root


## Local (x offsets from the object's center) slots where items rest, and the surface height.
static func surface_slots(obj: Dictionary) -> Dictionary:
	var kind: String = str(obj.get("kind", ""))
	var w: float = float(obj.get("w", 1))
	var xs: Array = []
	var zs: Array = []
	match kind:
		"desk":
			xs = [-1.1, -0.72, -0.36, 1.15, -0.1]
			zs = [0.0, 0.1, -0.05, -0.2, 0.25]
		"meeting_table":
			xs = [-1.2, -0.4, 0.4, 1.2, -0.8, 0.8]
			zs = [-0.3, 0.3, -0.3, 0.3, 0.1, -0.1]
		_:
			var n: int = maxi(2, int(w * 2.0))
			for i: int in n:
				xs.append(-w * 0.5 + (i + 0.5) * w / float(n))
				zs.append(0.0 if i % 2 == 0 else 0.08)
	return {"y": float(SURFACE_Y.get(kind, 0.76)), "xs": xs, "zs": zs}


# ------------------------------------------------------------------ furniture

static func _desk(r: Node3D, obj: Dictionary, w: float, fx: Dictionary) -> void:
	Kit.box(r, Vector3(w - 0.2, 0.06, 0.86), WOOD, Vector3(0, 0.7, 0))
	for dx: float in [-(w * 0.5 - 0.18), w * 0.5 - 0.18]:
		Kit.box(r, Vector3(0.07, 0.7, 0.76), WOOD_DARK, Vector3(dx, 0, 0))
	Kit.box(r, Vector3(0.62, 0.52, 0.7), WOOD_DARK, Vector3(-w * 0.5 + 0.55, 0.16, 0.02))   # drawers
	for k: int in 2:
		Kit.box(r, Vector3(0.16, 0.03, 0.02), "#e2c39a", Vector3(-w * 0.5 + 0.55, 0.3 + k * 0.22, -0.34))
	Kit.box(r, Vector3(w - 0.3, 0.3, 0.03), WOOD_DARK, Vector3(0, 0.38, 0.38))                 # modesty panel
	# Monitor on the far side, screen facing the chair (towards -z).
	# Off to one side and angled, so the camera still sees the face of whoever sits here.
	var mon: Node3D = Kit.node(r, Vector3(0.72, 0.76, 0.12), -28.0)
	Kit.box(mon, Vector3(0.3, 0.02, 0.2), METAL, Vector3.ZERO)
	Kit.box(mon, Vector3(0.06, 0.2, 0.06), METAL, Vector3(0, 0.02, 0.02))
	Kit.box(mon, Vector3(0.86, 0.52, 0.06), METAL, Vector3(0, 0.18, 0.02))
	var screen_mat: StandardMaterial3D = Kit.unique_mat("#7fb8ff", 0.4, 2.2)
	Kit.box(mon, Vector3(0.78, 0.44, 0.01), screen_mat, Vector3(0, 0.22, -0.015))
	Kit.box(mon, Vector3(0.12, 0.1, 0.005), Kit.mat("#f2e27a", 0.9, 0.2), Vector3(0.36, 0.52, -0.02))  # sticky note
	var glow: OmniLight3D = Kit.omni(r, Vector3(0.45, 1.05, -0.3), "#8fc4ff", 0.8, 1.7)
	Kit.box(r, Vector3(0.58, 0.025, 0.18), "#4a4458", Vector3(0.1, 0.76, -0.2))              # keyboard
	Kit.box(r, Vector3(0.08, 0.02, 0.12), "#4a4458", Vector3(0.55, 0.76, -0.22))             # mouse
	Kit.cyl(r, 0.05, 0.12, "#6b5a80", Vector3(0.2, 0.76, 0.3))                               # pen cup
	for k: int in 3:
		Kit.box(r, Vector3(0.012, 0.09, 0.012), ["#d9534f", "#3f6fc4", "#f2c14e"][k], Vector3(0.18 + k * 0.02, 0.86, 0.3))
	# A little desk lamp at the far corner.
	var lamp: Node3D = Kit.node(r, Vector3(-w * 0.5 + 0.3, 0.76, 0.28))
	Kit.cyl(lamp, 0.08, 0.03, METAL, Vector3.ZERO)
	var arm: MeshInstance3D = Kit.box(lamp, Vector3(0.025, 0.34, 0.025), METAL, Vector3(0, 0.02, 0))
	arm.rotation_degrees.x = -12.0
	var shade: MeshInstance3D = Kit.cyl(lamp, 0.1, 0.1, "#f2c14e", Vector3(0, 0.3, -0.08), 0.05)
	shade.rotation_degrees.x = -30.0
	Kit.sphere(lamp, 0.04, Kit.mat("#ffe2a8", 0.5, 5.0), Vector3(0, 0.28, -0.1), -1.0, 8)
	var seat: Vector2i = Vector2i(int(obj.get("x", 0)) + 1, int(obj.get("y", 0)) - 1)
	var spots: Array = obj.get("spots", [])
	if not spots.is_empty():
		seat = Vector2i(int(spots[0].get("x", seat.x)), int(spots[0].get("y", seat.y)))
	fx["screens"].append({"mat": screen_mat, "light": glow, "seat": seat, "on": -1.0})


static func _chair(r: Node3D, yaw: float, c: String) -> void:
	var ch: Node3D = Kit.node(r, Vector3.ZERO, yaw)
	Kit.box(ch, Vector3(0.5, 0.07, 0.5), c, Vector3(0, 0.4, 0))
	Kit.box(ch, Vector3(0.5, 0.55, 0.07), c, Vector3(0, 0.47, -0.25))
	Kit.cyl(ch, 0.035, 0.4, METAL, Vector3.ZERO)
	for k: int in 5:
		var a: float = k * TAU / 5.0
		Kit.box(ch, Vector3(0.05, 0.04, 0.05), METAL, Vector3(cos(a) * 0.22, 0, sin(a) * 0.22))


static func _printer(r: Node3D, fx: Dictionary) -> void:
	Kit.box(r, Vector3(0.8, 0.45, 0.6), WOOD_DARK, Vector3(0, 0, -0.1))
	Kit.box(r, Vector3(0.72, 0.36, 0.56), "#dcd8e4", Vector3(0, 0.45, -0.1))
	Kit.box(r, Vector3(0.6, 0.04, 0.2), "#b8b3c4", Vector3(0, 0.62, 0.22))
	Kit.box(r, Vector3(0.44, 0.02, 0.3), "#fbfbff", Vector3(0, 0.82, -0.12))
	Kit.box(r, Vector3(0.18, 0.08, 0.02), "#3a3548", Vector3(0.18, 0.66, 0.19))
	var led: StandardMaterial3D = Kit.unique_mat("#5aff8a", 0.3, 3.0)
	Kit.box(r, Vector3(0.04, 0.03, 0.02), led, Vector3(-0.2, 0.7, 0.19))
	fx["blink"].append({"mat": led, "period": 2.4, "duty": 0.5, "energy": 3.0})


static func _question_board(r: Node3D, w: float, fx: Dictionary) -> void:
	# Cork board sitting on the half wall, facing the work area (+z).
	var b: Node3D = Kit.node(r, Vector3(0, 0.72, 0.45))
	Kit.box(b, Vector3(w - 0.3, 0.8, 0.06), WOOD_DARK, Vector3.ZERO)
	Kit.box(b, Vector3(w - 0.42, 0.68, 0.02), "#c99a62", Vector3(0, 0.06, 0.03))
	var cards: Node3D = Kit.node(b, Vector3(0, 0, 0.045))
	fx["board"] = {"node": cards, "w": w - 0.5, "count": -1}
	var title: Label3D = Label3D.new()
	title.text = "QUESTIONS"
	title.font_size = 40
	title.pixel_size = 0.004
	title.modulate = Color("#fff3d6")
	title.outline_size = 8
	title.outline_modulate = Color("#3b2e4a")
	title.position = Vector3(0, 0.88, 0.06)
	b.add_child(title)


## Cards pinned on the question board, one per open question (max 12).
static func board_cards(fx: Dictionary, open_count: int) -> void:
	var board: Dictionary = fx.get("board", {})
	if board.is_empty() or int(board["count"]) == open_count:
		return
	board["count"] = open_count
	var cards: Node3D = board["node"]
	for c: Node in cards.get_children():
		c.queue_free()
	var colors: Array = ["#fff6c8", "#ffd6e6", "#d6f0ff", "#e0ffd6"]
	var n: int = mini(open_count, 12)
	for i: int in n:
		var col: int = i % 4
		var row: int = i / 4
		var px: float = -float(board["w"]) * 0.5 + 0.18 + col * (float(board["w"]) - 0.36) / 3.0
		var card: MeshInstance3D = Kit.box(cards, Vector3(0.2, 0.16, 0.01), colors[(i * 7) % 4], Vector3(px, 0.52 - row * 0.2, 0))
		card.rotation_degrees.z = float((i * 37) % 11) - 5.0
		Kit.sphere(cards, 0.02, "#d9534f", Vector3(px, 0.66 - row * 0.2, 0.01), -1.0, 6)


static func plant(r: Node3D, s: float, fx: Dictionary) -> void:
	Kit.cyl(r, 0.2 * s, 0.34 * s, "#c9652e", Vector3.ZERO, 0.24 * s)
	Kit.cyl(r, 0.21 * s, 0.03, "#5b3a26", Vector3(0, 0.33 * s, 0), -1.0)
	var leaves: Node3D = Kit.node(r, Vector3(0, 0.34 * s, 0))
	for i: int in 7:
		var a: float = i * TAU / 7.0 + 0.4
		var lift: float = 0.12 * s + float(i % 3) * 0.13 * s
		Kit.sphere(leaves, (0.17 + 0.03 * float(i % 2)) * s, "#4f9a3a" if i % 2 else "#5aa36a",
			Vector3(cos(a) * 0.13 * s, lift, sin(a) * 0.13 * s), -1.0, 10)
	Kit.sphere(leaves, 0.16 * s, "#6cbf5a", Vector3(0, 0.5 * s, 0), -1.0, 10)
	fx["sway"].append({"node": leaves, "phase": randf() * TAU, "amp": 2.5 / s})


static func _fridge(r: Node3D) -> void:
	Kit.box(r, Vector3(0.8, 1.9, 0.7), "#e8f1f0", Vector3(0, 0, -0.1), 0.0, 0.4)
	Kit.box(r, Vector3(0.82, 0.02, 0.72), "#c7d4d3", Vector3(0, 1.25, -0.1))
	Kit.box(r, Vector3(0.04, 0.3, 0.04), "#9aa8a7", Vector3(0.3, 1.35, 0.26))
	Kit.box(r, Vector3(0.04, 0.4, 0.04), "#9aa8a7", Vector3(0.3, 0.7, 0.26))
	for k: int in 4:
		Kit.box(r, Vector3(0.08, 0.08, 0.02), ["#d9534f", "#f2c14e", "#3f6fc4", "#5aa36a"][k], Vector3(-0.2 + k * 0.1, 1.45 + (k % 2) * 0.12, 0.25))
	Kit.box(r, Vector3(0.18, 0.22, 0.01), "#fbfbff", Vector3(-0.1, 0.9, 0.255))   # a drawing


static func _coffee_machine(r: Node3D, fx: Dictionary) -> void:
	Kit.box(r, Vector3(0.9, 0.9, 0.7), "#a8744d", Vector3(0, 0, -0.1))
	Kit.box(r, Vector3(0.94, 0.05, 0.74), "#e8e2d6", Vector3(0, 0.9, -0.1))
	Kit.box(r, Vector3(0.46, 0.58, 0.44), "#3a3548", Vector3(0, 0.95, -0.14), 0.0, 0.35)
	Kit.box(r, Vector3(0.3, 0.12, 0.08), "#26222f", Vector3(0, 1.28, 0.1))
	Kit.cyl(r, 0.07, 0.12, "#f4ead8", Vector3(0, 0.95, 0.05))
	var led: StandardMaterial3D = Kit.unique_mat("#ff5a4a", 0.3, 3.0)
	Kit.box(r, Vector3(0.06, 0.05, 0.02), led, Vector3(0.14, 1.36, 0.09))
	fx["blink"].append({"mat": led, "period": 3.2, "duty": 0.8, "energy": 3.0})
	fx["steam"].append({"node": steam(r, Vector3(0, 1.1, 0.06), 10), "where": "coffee_machine"})
	Kit.omni(r, Vector3(0, 1.4, 0.4), "#ff9a5a", 0.5, 1.4)


static func _counter(r: Node3D, w: float) -> void:
	Kit.box(r, Vector3(w, 0.9, 0.7), "#a8744d", Vector3(0, 0, -0.1))
	Kit.box(r, Vector3(w + 0.06, 0.05, 0.74), "#e8e2d6", Vector3(0, 0.9, -0.1))
	for i: int in int(w):
		var cx: float = -w * 0.5 + 0.5 + i
		Kit.box(r, Vector3(0.9, 0.78, 0.02), "#b7815a", Vector3(cx, 0.06, 0.255))
		Kit.box(r, Vector3(0.2, 0.03, 0.03), "#e2c39a", Vector3(cx, 0.7, 0.27))
	# Sink with a tap, a kettle and a fruit bowl.
	Kit.box(r, Vector3(0.5, 0.02, 0.36), "#9aa8b7", Vector3(w * 0.5 - 0.5, 0.95, -0.12))
	Kit.cyl(r, 0.025, 0.25, "#c7d4d3", Vector3(w * 0.5 - 0.5, 0.95, -0.35))
	Kit.cyl(r, 0.11, 0.2, "#d9534f", Vector3(-w * 0.5 + 0.3, 0.95, -0.25), 0.08)
	Kit.cyl(r, 0.17, 0.07, "#e8e2d6", Vector3(0.1, 0.95, -0.25), 0.2)
	for k: int in 4:
		Kit.sphere(r, 0.06, ["#f2c14e", "#d9534f", "#9bcf53", "#e8893a"][k], Vector3(0.02 + (k % 2) * 0.12, 1.06, -0.3 + (k / 2) * 0.1), -1.0, 8)
	# Wall shelf with jars above the counter.
	Kit.box(r, Vector3(w - 0.4, 0.04, 0.24), WOOD, Vector3(0, 1.6, -0.36))
	for k: int in 5:
		Kit.cyl(r, 0.06, 0.14 + (k % 2) * 0.05, ["#efe6d2", "#c8a27a", "#f06fa4", "#6cb4e4", "#efe6d2"][k], Vector3(-w * 0.5 + 0.5 + k * 0.45, 1.64, -0.36))


static func _couch(r: Node3D, w: float) -> void:
	var c: String = "#c96c26"
	Kit.box(r, Vector3(w, 0.28, 0.8), "#b85f22", Vector3(0, 0.08, -0.05))
	for i: int in int(w):
		Kit.box(r, Vector3(0.94, 0.12, 0.72), c, Vector3(-w * 0.5 + 0.5 + i, 0.34, 0.0))
	Kit.box(r, Vector3(w, 0.62, 0.24), "#b85f22", Vector3(0, 0.3, -0.38))
	for dx: float in [-w * 0.5 - 0.1, w * 0.5 + 0.1]:
		Kit.box(r, Vector3(0.22, 0.62, 0.8), c, Vector3(dx, 0.0, -0.05))
	var p: MeshInstance3D = Kit.box(r, Vector3(0.36, 0.34, 0.12), "#f2c14e", Vector3(-w * 0.5 + 0.35, 0.44, -0.2))
	p.rotation_degrees = Vector3(-10, 18, 8)
	var p2: MeshInstance3D = Kit.box(r, Vector3(0.34, 0.32, 0.12), "#8e6cc9", Vector3(w * 0.5 - 0.35, 0.44, -0.2))
	p2.rotation_degrees = Vector3(-10, -15, -8)
	Kit.box(r, Vector3(0.7, 0.05, 0.8), "#3a9e97", Vector3(w * 0.5 - 0.55, 0.46, 0.0))   # a folded blanket


static func _coffee_table(r: Node3D, w: float) -> void:
	Kit.box(r, Vector3(w - 0.6, 0.05, 0.76), WOOD, Vector3(0, 0.32, 0))
	for dx: float in [-(w * 0.5 - 0.4), w * 0.5 - 0.4]:
		for dz: float in [-0.3, 0.3]:
			Kit.box(r, Vector3(0.06, 0.32, 0.06), WOOD_DARK, Vector3(dx, 0, dz))
	for k: int in 3:
		Kit.box(r, Vector3(0.36, 0.035, 0.26), ["#3f6fc4", "#f06fa4", "#efe6d2"][k], Vector3(0.6, 0.37 + k * 0.035, -0.1)).rotation_degrees.y = k * 12.0


static func _meeting_table(r: Node3D, obj: Dictionary, w: float, h: float) -> void:
	Kit.box(r, Vector3(w - 0.3, 0.06, h - 0.3), WOOD, Vector3(0, 0.7, 0))
	for dx: float in [-(w * 0.5 - 0.3), w * 0.5 - 0.3]:
		for dz: float in [-(h * 0.5 - 0.3), h * 0.5 - 0.3]:
			Kit.box(r, Vector3(0.08, 0.7, 0.08), WOOD_DARK, Vector3(dx, 0, dz))
	Kit.cyl(r, 0.14, 0.14, "#efe6d2", Vector3(0, 0.76, 0), 0.1)
	var leaves: MeshInstance3D = Kit.sphere(r, 0.12, "#5aa36a", Vector3(0, 0.95, 0), -1.0, 10)
	leaves.scale = Vector3(1.0, 0.8, 1.0)
	var cx: float = float(obj.get("x", 0)) + w * 0.5
	var cz: float = float(obj.get("y", 0)) + h * 0.5
	for s: Variant in obj.get("spots", []):
		var sp: Dictionary = s
		if str(sp.get("pose", "")) != "sit":
			continue
		var yaw: float = 0.0 if str(sp.get("facing", "down")) == "down" else 180.0
		var holder: Node3D = Kit.node(r, Vector3(float(sp["x"]) + 0.5 - cx, 0, float(sp["y"]) + 0.5 - cz))
		_chair(holder, yaw, "#6f7ca3")


static func _whiteboard(r: Node3D, w: float, fx: Dictionary) -> void:
	# Mounted on the back wall, facing the room (+z).
	var b: Node3D = Kit.node(r, Vector3(0, 0.95, 0.52))
	Kit.box(b, Vector3(w - 0.2, 0.95, 0.04), "#c7c3d2", Vector3.ZERO)
	Kit.box(b, Vector3(w - 0.3, 0.85, 0.02), Kit.mat("#fbfbff", 0.3, 0.05), Vector3(0, 0.05, 0.02))
	Kit.box(b, Vector3(w - 0.4, 0.04, 0.08), "#c7c3d2", Vector3(0, -0.02, 0.04))
	for k: int in 3:
		Kit.box(b, Vector3(0.1, 0.025, 0.025), ["#d9534f", "#3f6fc4", "#2b2833"][k], Vector3(-0.3 + k * 0.15, 0.02, 0.06))
	var text: Label3D = Label3D.new()
	text.font_size = 28
	text.pixel_size = 0.0035
	text.modulate = Color("#2f3f6e")
	text.width = (w - 0.45) / 0.0035
	text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	text.vertical_alignment = VERTICAL_ALIGNMENT_TOP
	text.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	text.position = Vector3(0, 0.84, 0.035)
	text.shaded = false
	text.double_sided = false
	b.add_child(text)
	fx["whiteboard"] = text


static func _coat_rack(r: Node3D) -> void:
	Kit.cyl(r, 0.18, 0.04, WOOD_DARK, Vector3.ZERO)
	Kit.cyl(r, 0.035, 1.7, WOOD_DARK, Vector3.ZERO)
	for k: int in 4:
		var a: float = k * TAU / 4.0
		Kit.box(r, Vector3(0.03, 0.03, 0.18), WOOD_DARK, Vector3(cos(a) * 0.08, 1.6, sin(a) * 0.08)).rotation_degrees.y = rad_to_deg(-a) + 90.0
	var coat: MeshInstance3D = Kit.box(r, Vector3(0.36, 0.7, 0.14), "#b8584a", Vector3(0.08, 0.9, 0.1))
	coat.rotation_degrees.z = -6.0
	var scarf: MeshInstance3D = Kit.box(r, Vector3(0.1, 0.5, 0.06), "#f2c14e", Vector3(-0.12, 1.1, -0.06))
	scarf.rotation_degrees.z = 8.0
	# A wet umbrella leaning on it (it's often raining here).
	var um: Node3D = Kit.node(r, Vector3(0.28, 0, 0.12))
	um.rotation_degrees.z = -12.0
	Kit.cyl(um, 0.1, 0.7, "#3f6fc4", Vector3(0, 0.12, 0), 0.02)
	Kit.cyl(um, 0.012, 0.95, METAL, Vector3.ZERO)


static func _door(r: Node3D) -> void:
	Kit.box(r, Vector3(0.9, 0.012, 0.6), "#8e3b24", Vector3(0, 0.0, -0.8))   # doormat (inside)
	Kit.box(r, Vector3(0.7, 0.014, 0.4), "#c9652e", Vector3(0, 0.0, -0.8))
	for dx: float in [-0.5, 0.5]:
		Kit.box(r, Vector3(0.1, 0.34, 1.0), "#4c405e", Vector3(dx, 0, 0))
	var leaf: Node3D = Kit.node(r, Vector3(0.45, 0.0, -0.45), -70.0)   # open door leaf
	Kit.box(leaf, Vector3(0.9, 0.34, 0.06), "#a8744d", Vector3(-0.45, 0, 0))


# ------------------------------------------------------------------ clutter & fx

static func bookshelf(r: Node3D, fx: Dictionary) -> void:
	Kit.box(r, Vector3(0.95, 1.7, 0.4), WOOD_DARK, Vector3(0, 0, -0.25))
	var cols: Array = ["#d9534f", "#3f6fc4", "#f2c14e", "#5aa36a", "#8e6cc9", "#efe6d2", "#e8893a"]
	for shelf: int in 3:
		var sy: float = 0.18 + shelf * 0.52
		Kit.box(r, Vector3(0.88, 0.03, 0.36), WOOD, Vector3(0, sy - 0.03, -0.23))
		var px: float = -0.38
		var i: int = shelf * 3
		while px < 0.3:
			var bw: float = 0.07 + float((i * 5) % 3) * 0.02
			var bh: float = 0.28 + float((i * 7) % 4) * 0.04
			Kit.box(r, Vector3(bw, bh, 0.26), cols[i % cols.size()], Vector3(px + bw * 0.5, sy, -0.2))
			px += bw + 0.01
			i += 1
	plant(Kit.node(r, Vector3(0.2, 1.7, -0.25)), 0.5, fx)


static func floor_lamp(r: Node3D) -> OmniLight3D:
	Kit.cyl(r, 0.16, 0.04, METAL, Vector3.ZERO)
	Kit.cyl(r, 0.025, 1.45, METAL, Vector3.ZERO)
	Kit.cyl(r, 0.26, 0.3, Kit.mat("#f2d9a0", 0.9, 0.9), Vector3(0, 1.4, 0), 0.16)
	return Kit.omni(r, Vector3(0, 1.45, 0), "#ffcf8a", 2.0, 4.5, true)


static func water_cooler(r: Node3D) -> void:
	Kit.box(r, Vector3(0.4, 0.9, 0.4), "#d9d6df", Vector3.ZERO)
	var bottle: MeshInstance3D = Kit.cyl(r, 0.17, 0.42, "#8fd0f0", Vector3(0, 0.9, 0))
	var m: StandardMaterial3D = Kit.unique_mat("#8fd0f0", 0.1, 0.25)
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color.a = 0.75
	bottle.material_override = m
	Kit.box(r, Vector3(0.06, 0.05, 0.04), "#3f6fc4", Vector3(-0.08, 0.6, 0.2))
	Kit.box(r, Vector3(0.06, 0.05, 0.04), "#d9534f", Vector3(0.08, 0.6, 0.2))


static func filing_cabinet(r: Node3D) -> void:
	Kit.box(r, Vector3(0.5, 1.0, 0.6), "#8a8793", Vector3.ZERO, 0.0, 0.5)
	for k: int in 3:
		Kit.box(r, Vector3(0.46, 0.28, 0.02), "#9a98a3", Vector3(0.0, 0.06 + k * 0.32, 0.3))
		Kit.box(r, Vector3(0.14, 0.03, 0.03), "#d9d6df", Vector3(0.0, 0.26 + k * 0.32, 0.31))
	plant(Kit.node(r, Vector3(0.0, 1.0, 0.0)), 0.45, fx_dummy())


static func armchair(r: Node3D, c: String) -> void:
	Kit.box(r, Vector3(0.8, 0.36, 0.75), c, Vector3.ZERO)
	Kit.box(r, Vector3(0.8, 0.5, 0.2), Color(c).darkened(0.12), Vector3(0, 0.36, -0.28))
	for dx: float in [-0.4, 0.4]:
		Kit.box(r, Vector3(0.14, 0.52, 0.75), Color(c).darkened(0.06), Vector3(dx, 0.0, 0))
	Kit.box(r, Vector3(0.3, 0.3, 0.1), "#efe6d2", Vector3(0.12, 0.38, -0.12)).rotation_degrees.z = 12.0


static func trash_bin(r: Node3D) -> void:
	Kit.cyl(r, 0.16, 0.36, "#5aa36a", Vector3.ZERO, 0.19)
	Kit.box(r, Vector3(0.14, 0.1, 0.12), "#fbfbff", Vector3(0.02, 0.33, 0.0)).rotation_degrees = Vector3(20, 30, 10)


## Wall clock whose hands show the office's time (fx "clock": {hour, minute}).
static func wall_clock(r: Node3D, fx: Dictionary) -> void:
	var face: Node3D = Kit.node(r)
	var body: MeshInstance3D = Kit.cyl(face, 0.3, 0.06, "#3b2e4a", Vector3.ZERO)
	body.rotation_degrees.x = 90.0
	var dial: MeshInstance3D = Kit.cyl(face, 0.26, 0.02, Kit.mat("#fff7e8", 0.6, 0.15), Vector3(0, 0, 0.04))
	dial.rotation_degrees.x = 90.0
	for k: int in 12:
		var a: float = k * TAU / 12.0
		Kit.box(face, Vector3(0.02, 0.04 if k % 3 else 0.06, 0.01), "#3b2e4a", Vector3(sin(a) * 0.21, cos(a) * 0.21 - 0.02, 0.05))
	var hour: Node3D = Kit.node(face, Vector3(0, 0, 0.06))
	Kit.box(hour, Vector3(0.03, 0.13, 0.01), "#2b2833", Vector3.ZERO)
	var minute: Node3D = Kit.node(face, Vector3(0, 0, 0.07))
	Kit.box(minute, Vector3(0.02, 0.2, 0.01), "#d9534f", Vector3.ZERO)
	fx["clock"] = {"hour": hour, "minute": minute}


## Glowing "HelpCo" sign on a wall.
static func neon_sign(r: Node3D, text: String) -> void:
	Kit.box(r, Vector3(1.9, 0.6, 0.04), "#2a1f33", Vector3(0, -0.3, 0))
	var l: Label3D = Label3D.new()
	l.text = text
	l.font_size = 96
	l.pixel_size = 0.0045
	l.modulate = Color(2.6, 0.9, 1.6)
	l.outline_size = 18
	l.outline_modulate = Color(0.9, 0.25, 0.55, 0.9)
	l.shaded = false
	l.position = Vector3(0, 0, 0.04)
	l.font = _bold_font()
	r.add_child(l)
	Kit.omni(r, Vector3(0, 0, 0.5), "#f06fa4", 1.2, 2.6)


static func _bold_font() -> Font:
	var f: FontVariation = FontVariation.new()
	f.base_font = ThemeDB.fallback_font
	f.variation_embolden = 1.0
	return f


## Small puffs of steam rising from `pos`.
static func steam(parent: Node3D, pos: Vector3, amount: int = 6) -> CPUParticles3D:
	var p: CPUParticles3D = CPUParticles3D.new()
	p.position = pos
	p.amount = amount
	p.lifetime = 2.2
	p.local_coords = false
	p.emission_shape = CPUParticles3D.EMISSION_SHAPE_SPHERE
	p.emission_sphere_radius = 0.03
	p.direction = Vector3.UP
	p.spread = 12.0
	p.gravity = Vector3(0.02, 0.12, 0.0)
	p.initial_velocity_min = 0.08
	p.initial_velocity_max = 0.16
	p.scale_amount_min = 0.6
	p.scale_amount_max = 1.2
	var curve: Curve = Curve.new()
	curve.add_point(Vector2(0.0, 0.3))
	curve.add_point(Vector2(0.5, 1.0))
	curve.add_point(Vector2(1.0, 1.4))
	p.scale_amount_curve = curve
	var grad: Gradient = Gradient.new()
	grad.set_color(0, Color(1, 1, 1, 0.0))
	grad.add_point(0.2, Color(1, 1, 1, 0.35))
	grad.set_color(grad.get_point_count() - 1, Color(1, 1, 1, 0.0))
	p.color_ramp = grad
	var sm: SphereMesh = SphereMesh.new()
	sm.radius = 0.035
	sm.height = 0.07
	sm.radial_segments = 6
	sm.rings = 3
	var m: StandardMaterial3D = StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.vertex_color_use_as_albedo = true
	m.albedo_color = Color(1, 1, 1, 1)
	sm.material = m
	p.mesh = sm
	p.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	parent.add_child(p)
	return p


static func fx_dummy() -> Dictionary:
	return {"sway": [], "blink": [], "screens": [], "steam": []}


# ------------------------------------------------------------------ items

## A small item model, bottom-centered on `parent`'s origin.
static func item(parent: Node3D, kind: String, color: Variant, s: float = 1.0) -> Node3D:
	var r: Node3D = Kit.node(parent)
	r.scale = Vector3(s, s, s)
	var tint: String = str(Kit.COLORS.get(str(color), "")) if color != null else ""
	match kind:
		"mug", "coffee":
			var c: String = tint if tint != "" else ("#f4ead8" if kind == "coffee" else "#f06fa4")
			Kit.cyl(r, 0.065, 0.13, c, Vector3.ZERO)
			Kit.torus(r, 0.025, 0.045, c, Vector3(0.075, 0.065, 0), Vector3(90, 0, 0))
			if kind == "coffee":
				Kit.cyl(r, 0.058, 0.01, "#5b3a26", Vector3(0, 0.12, 0))
				steam(r, Vector3(0, 0.15, 0), 4)
		"succulent":
			Kit.cyl(r, 0.07, 0.09, "#e8893a", Vector3.ZERO, 0.08)
			for k: int in 5:
				var a: float = k * TAU / 5.0
				Kit.sphere(r, 0.04, "#7bc47f", Vector3(cos(a) * 0.035, 0.12, sin(a) * 0.035), -1.0, 6)
			Kit.sphere(r, 0.035, "#9bcf53", Vector3(0, 0.15, 0), -1.0, 6)
		"photo_frame":
			var f: MeshInstance3D = Kit.box(r, Vector3(0.2, 0.16, 0.025), "#8d5f3d", Vector3.ZERO)
			f.rotation_degrees.x = -12.0
			var ph: MeshInstance3D = Kit.box(r, Vector3(0.16, 0.12, 0.01), Kit.mat("#6cb4e4", 0.6, 0.2), Vector3(0, 0.02, 0.012))
			ph.rotation_degrees.x = -12.0
		"rubber_duck":
			Kit.sphere(r, 0.07, "#f2c14e", Vector3(0, 0.06, 0), -1.0, 10)
			Kit.sphere(r, 0.045, "#f2c14e", Vector3(0, 0.14, 0.03), -1.0, 10)
			Kit.box(r, Vector3(0.05, 0.02, 0.04), "#e8893a", Vector3(0, 0.13, 0.08))
			for dx: float in [-0.02, 0.02]:
				Kit.sphere(r, 0.01, "#1c1624", Vector3(dx, 0.155, 0.068), -1.0, 4)
		"books":
			for k: int in 3:
				Kit.box(r, Vector3(0.22 - k * 0.02, 0.045, 0.16), ["#3f6fc4", "#d9534f", "#5aa36a"][k], Vector3(0, k * 0.045, 0)).rotation_degrees.y = k * 9.0 - 6.0
		"snow_globe":
			Kit.cyl(r, 0.07, 0.05, "#7a4b2a", Vector3.ZERO, 0.06)
			var g: MeshInstance3D = Kit.sphere(r, 0.075, "#d6f0ff", Vector3(0, 0.11, 0), -1.0, 12)
			var gm: StandardMaterial3D = Kit.unique_mat("#d6f0ff", 0.05, 0.2)
			gm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
			gm.albedo_color.a = 0.5
			g.material_override = gm
			Kit.cyl(r, 0.018, 0.06, "#2f6e3a", Vector3(0, 0.07, 0), 0.0)
		"note":
			Kit.box(r, Vector3(0.1, 0.008, 0.1), "#f2e27a", Vector3.ZERO)
		"paper":
			Kit.box(r, Vector3(0.18, 0.006, 0.24), "#fbfbff", Vector3.ZERO)
		_:
			Kit.box(r, Vector3(0.1, 0.1, 0.1), tint if tint != "" else "#b9a3e3", Vector3.ZERO)
	return r
