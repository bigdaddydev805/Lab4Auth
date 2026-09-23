extends RefCounted
## Tiny low-poly construction kit shared by the office, props and chibis: cached materials and
## one-line helpers that add boxes, spheres and cylinders to a parent node.
## Sizes are in tiles (1 tile = 1 world unit). Boxes and cylinders sit ON `pos` (pos is the
## bottom-center); spheres are centered on `pos`.

const SKIN := {"ivory": "#ffe0c7", "peach": "#f3c49b", "sand": "#e8b88f", "golden": "#d9a066",
	"tan": "#c98c5f", "caramel": "#a86f47", "brown": "#8d5a3b", "umber": "#6e4430", "deep": "#4f3024"}
const COLORS := {"red": "#d9534f", "coral": "#f07a63", "orange": "#e8893a", "mustard": "#d9a531",
	"yellow": "#f2c14e", "lime": "#9bcf53", "green": "#5aa36a", "teal": "#3a9e97", "sky": "#6cb4e4",
	"blue": "#3f6fc4", "navy": "#2f3f6e", "purple": "#8e6cc9", "lavender": "#b9a3e3", "pink": "#f06fa4",
	"brown": "#7a4b2a", "tan": "#c8a27a", "cream": "#efe6d2", "white": "#f4f1ec", "gray": "#8a8793",
	"charcoal": "#4a4a5a", "black": "#2b2833"}
const HAIR := {"black": "#241c2c", "dark brown": "#4a2f22", "brown": "#6b3f2a", "auburn": "#8e3b24",
	"ginger": "#c9652e", "blonde": "#e8b64c", "platinum": "#efe3b8", "gray": "#9a98a3", "white": "#eeeef2",
	"pink": "#f06fa4", "blue": "#4f7fd9", "green": "#4fae6a", "purple": "#8e5cc9"}

static var _mats: Dictionary = {}
static var _meshes: Dictionary = {}


static func color_of(table: Dictionary, key: Variant, fallback: String) -> Color:
	return Color(str(table.get(str(key), fallback)))


## A shared material per (color, roughness, emission). Use `unique_mat` for animated ones.
static func mat(c: Variant, rough: float = 0.85, emit: float = 0.0) -> StandardMaterial3D:
	var col: Color = c if c is Color else Color(str(c))
	var key: String = "%s|%.2f|%.2f" % [col.to_html(), rough, emit]
	var m: StandardMaterial3D = _mats.get(key)
	if m == null:
		m = unique_mat(col, rough, emit)
		_mats[key] = m
	return m


static func unique_mat(c: Variant, rough: float = 0.85, emit: float = 0.0) -> StandardMaterial3D:
	var col: Color = c if c is Color else Color(str(c))
	var m: StandardMaterial3D = StandardMaterial3D.new()
	m.albedo_color = col
	m.roughness = rough
	if emit > 0.0:
		m.emission_enabled = true
		m.emission = col
		m.emission_energy_multiplier = emit
	return m


static func node(parent: Node, pos: Vector3 = Vector3.ZERO, yaw_deg: float = 0.0) -> Node3D:
	var n: Node3D = Node3D.new()
	n.position = pos
	n.rotation_degrees.y = yaw_deg
	parent.add_child(n)
	return n


static func mesh(parent: Node, m: Mesh, material: Material, pos: Vector3, rot: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	var mi: MeshInstance3D = MeshInstance3D.new()
	mi.mesh = m
	mi.material_override = material
	mi.position = pos
	mi.rotation_degrees = rot
	parent.add_child(mi)
	return mi


static func box(parent: Node, size: Vector3, c: Variant, pos: Vector3, emit: float = 0.0, rough: float = 0.85) -> MeshInstance3D:
	var key: String = "box%s" % size
	var bm: BoxMesh = _meshes.get(key)
	if bm == null:
		bm = BoxMesh.new()
		bm.size = size
		_meshes[key] = bm
	return mesh(parent, bm, c if c is Material else mat(c, rough, emit), pos + Vector3(0, size.y * 0.5, 0))


static func sphere(parent: Node, r: float, c: Variant, pos: Vector3, h: float = -1.0, segments: int = 14) -> MeshInstance3D:
	var key: String = "sph%.3f|%.3f|%d" % [r, h, segments]
	var sm: SphereMesh = _meshes.get(key)
	if sm == null:
		sm = SphereMesh.new()
		sm.radius = r
		sm.height = r * 2.0 if h < 0.0 else h
		sm.is_hemisphere = h >= 0.0 and h <= r * 1.01
		sm.radial_segments = segments
		sm.rings = maxi(4, segments / 2)
		_meshes[key] = sm
	return mesh(parent, sm, c if c is Material else mat(c), pos)


static func cyl(parent: Node, r: float, h: float, c: Variant, pos: Vector3, top_r: float = -1.0, segments: int = 12) -> MeshInstance3D:
	var key: String = "cyl%.3f|%.3f|%.3f|%d" % [r, h, top_r, segments]
	var cm: CylinderMesh = _meshes.get(key)
	if cm == null:
		cm = CylinderMesh.new()
		cm.bottom_radius = r
		cm.top_radius = r if top_r < 0.0 else top_r
		cm.height = h
		cm.radial_segments = segments
		cm.rings = 1
		_meshes[key] = cm
	return mesh(parent, cm, c if c is Material else mat(c), pos + Vector3(0, h * 0.5, 0))


static func torus(parent: Node, inner: float, outer: float, c: Variant, pos: Vector3, rot: Vector3 = Vector3.ZERO) -> MeshInstance3D:
	var tm: TorusMesh = TorusMesh.new()
	tm.inner_radius = inner
	tm.outer_radius = outer
	tm.rings = 12
	tm.ring_segments = 6
	return mesh(parent, tm, c if c is Material else mat(c), pos, rot)


static func omni(parent: Node, pos: Vector3, c: Variant, energy: float, rng: float, shadows: bool = false) -> OmniLight3D:
	var o: OmniLight3D = OmniLight3D.new()
	o.position = pos
	o.light_color = c if c is Color else Color(str(c))
	o.light_energy = energy
	o.omni_range = rng
	o.omni_attenuation = 1.4
	o.shadow_enabled = shadows
	parent.add_child(o)
	return o


## A soft round contact shadow on the floor (a transparent disc), for grounding things.
static func blob_shadow(parent: Node, radius: float, alpha: float = 0.35) -> MeshInstance3D:
	var key: String = "blob%.2f" % alpha
	var m: StandardMaterial3D = _mats.get(key)
	if m == null:
		m = StandardMaterial3D.new()
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.albedo_texture = _blob_texture()
		m.albedo_color = Color(0.05, 0.03, 0.1, alpha)
		m.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
		_mats[key] = m
	var q: PlaneMesh = PlaneMesh.new()
	q.size = Vector2(radius * 2.0, radius * 2.0)
	var mi: MeshInstance3D = mesh(parent, q, m, Vector3(0, 0.012, 0))
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	return mi


static func _blob_texture() -> Texture2D:
	var g: Gradient = Gradient.new()
	g.set_color(0, Color(1, 1, 1, 1))
	g.set_color(1, Color(1, 1, 1, 0))
	var t: GradientTexture2D = GradientTexture2D.new()
	t.gradient = g
	t.fill = GradientTexture2D.FILL_RADIAL
	t.fill_from = Vector2(0.5, 0.5)
	t.fill_to = Vector2(0.5, 0.0)
	t.width = 64
	t.height = 64
	return t
