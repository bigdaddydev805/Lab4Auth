"""Research sample (not production art). Run: pip install pillow && python sprite_sample.py

Quick demo: modular, palette-ramped pixel-art employees drawn entirely in code.

Each layer is a 32x32 mask. Digits are ramp indices (1=darkest .. 4=highlight);
letters are fixed colors (eyes, blush, etc). Layers are recolored from a catalog
color, composited, then given an automatic 1px outer outline.
"""
from pathlib import Path

from PIL import Image, ImageDraw
import colorsys

W, H = 32, 32

def grid(rows):
    g = {}
    for y, row in enumerate(rows):
        row = row.ljust(W, ".")
        assert len(row) == W, (y, row, len(row))
        for x, ch in enumerate(row):
            if ch != ".":
                g[(x, y)] = ch
    return g

def blank():
    return ["." * W for _ in range(H)]

def paint(rows, y, x0, x1, ch):
    r = list(rows[y])
    for x in range(x0, x1 + 1):
        r[x] = ch
    rows[y] = "".join(r)

# ---------- body (skin + face) ----------
body = blank()
for y, (a, b) in {5: (11, 20), 6: (10, 21), 7: (9, 22)}.items():
    paint(body, y, a, b, "3")
for y in range(8, 15):
    paint(body, y, 8, 23, "3")
paint(body, 15, 9, 22, "3")
paint(body, 16, 10, 21, "2")
paint(body, 16, 12, 19, "3")
paint(body, 17, 12, 19, "2")
# right-side soft shadow on the face
for y in range(8, 15):
    paint(body, y, 23, 23, "2")
paint(body, 15, 22, 22, "2")
# eyes (2x3), blush, mouth
for (ex, ey), ch in {(11, 11): "e", (12, 11): "e", (11, 12): "e", (12, 12): "w", (11, 13): "e", (12, 13): "e",
                     (19, 11): "e", (20, 11): "e", (19, 12): "e", (20, 12): "w", (19, 13): "e", (20, 13): "e"}.items():
    paint(body, ey, ex, ex, ch)
paint(body, 14, 9, 10, "b")
paint(body, 14, 21, 22, "b")
paint(body, 15, 15, 16, "m")
# hands
paint(body, 24, 8, 9, "3"); paint(body, 24, 22, 23, "3")
paint(body, 25, 8, 9, "2"); paint(body, 25, 22, 23, "2")

# ---------- shirt / sweater ----------
shirt = blank()
paint(shirt, 18, 9, 22, "3")
paint(shirt, 18, 13, 18, "2")  # neckline shadow
for y in range(19, 24):
    paint(shirt, y, 8, 23, "3")
    paint(shirt, y, 10, 10, "2"); paint(shirt, y, 21, 21, "2")   # arm/torso separation
    paint(shirt, y, 22, 23, "2")                                 # right sleeve in shade
for y in (24, 25):
    paint(shirt, y, 10, 21, "3")
paint(shirt, 25, 10, 21, "2")  # hem
paint(shirt, 19, 11, 13, "4")  # chest highlight
paint(shirt, 20, 11, 12, "4")

# ---------- pants + shoes ----------
pants = blank()
paint(pants, 26, 10, 21, "3")
for y in (27, 28):
    paint(pants, y, 10, 14, "3"); paint(pants, y, 17, 21, "3")
    paint(pants, y, 14, 14, "2"); paint(pants, y, 21, 21, "2")
paint(pants, 26, 15, 16, "2")
shoes = blank()
paint(shoes, 29, 9, 14, "3"); paint(shoes, 29, 17, 22, "3")
paint(shoes, 30, 9, 14, "2"); paint(shoes, 30, 17, 22, "2")
paint(shoes, 29, 10, 11, "4"); paint(shoes, 29, 18, 19, "4")

# ---------- hair styles ----------
def hair_bob():
    h = blank()
    paint(h, 2, 12, 19, "3"); paint(h, 2, 13, 16, "4")
    paint(h, 3, 10, 21, "3"); paint(h, 3, 11, 15, "4")
    paint(h, 4, 9, 22, "3"); paint(h, 4, 10, 12, "4")
    for y in range(5, 10):
        paint(h, y, 7, 24, "3")
        paint(h, y, 23, 24, "2")
    paint(h, 5, 9, 10, "4")
    # bangs: jagged tips over the forehead
    paint(h, 9, 12, 12, "2"); paint(h, 9, 16, 16, "2"); paint(h, 9, 20, 20, "2")
    for x0, x1 in ((7, 10), (13, 14), (17, 18), (21, 24)):
        paint(h, 10, x0, x1, "3")
    paint(h, 10, 10, 10, "2"); paint(h, 10, 14, 14, "2"); paint(h, 10, 18, 18, "2"); paint(h, 10, 23, 24, "2")
    for y in range(11, 16):
        paint(h, y, 7, 9, "3"); paint(h, y, 22, 24, "2")
        paint(h, y, 9, 9, "2")
    paint(h, 16, 7, 9, "2"); paint(h, 16, 22, 24, "1")
    return h

def hair_short():
    h = blank()
    paint(h, 2, 11, 13, "3"); paint(h, 2, 17, 19, "3")  # tufts
    paint(h, 3, 10, 21, "3"); paint(h, 3, 11, 14, "4")
    paint(h, 4, 9, 22, "3"); paint(h, 4, 10, 12, "4")
    for y in range(5, 9):
        paint(h, y, 8, 23, "3")
        paint(h, y, 22, 23, "2")
    paint(h, 5, 9, 10, "4")
    paint(h, 9, 8, 13, "3"); paint(h, 9, 17, 23, "3"); paint(h, 9, 13, 13, "2"); paint(h, 9, 22, 23, "2")
    paint(h, 10, 8, 9, "3"); paint(h, 10, 22, 23, "2")
    paint(h, 11, 8, 8, "2"); paint(h, 11, 23, 23, "2")
    return h

def hair_long():
    h = hair_bob()
    for y in range(16, 23):
        paint(h, y, 6, 8, "3"); paint(h, y, 23, 25, "2")
        paint(h, y, 8, 8, "2")
    paint(h, 23, 6, 8, "2"); paint(h, 23, 23, 25, "1")
    for y in range(11, 16):
        paint(h, y, 6, 6, "3"); paint(h, y, 25, 25, "2")
    return h

def hair_bun():
    h = hair_short()
    paint(h, 0, 14, 17, "3")
    paint(h, 1, 13, 18, "3"); paint(h, 1, 14, 15, "4")
    paint(h, 2, 13, 18, "2")
    return h

# ---------- accessories ----------
def glasses():
    g = blank()
    paint(g, 11, 14, 17, "k")                      # bridge
    for y in (11, 12, 13):
        paint(g, y, 10, 10, "k"); paint(g, y, 13, 13, "k")
        paint(g, y, 18, 18, "k"); paint(g, y, 21, 21, "k")
    paint(g, 14, 11, 12, "k"); paint(g, 14, 19, 20, "k")   # rounded bottoms
    return g

def frog_hat():
    f = blank()
    paint(f, 0, 9, 12, "3"); paint(f, 0, 19, 22, "3")
    for x0 in (9, 19):
        paint(f, 1, x0, x0 + 3, "3"); paint(f, 1, x0 + 1, x0 + 2, "W")
        paint(f, 2, x0, x0 + 3, "3"); paint(f, 2, x0 + 1, x0 + 2, "W")
    paint(f, 2, 11, 11, "K"); paint(f, 2, 20, 20, "K")  # pupils look inward
    paint(f, 3, 8, 23, "3"); paint(f, 3, 10, 12, "4")
    for y in (4, 5):
        paint(f, y, 7, 24, "3")
    paint(f, 4, 9, 11, "4")
    paint(f, 5, 12, 19, "M")   # little frog smile on the hat
    paint(f, 6, 7, 24, "2")
    paint(f, 4, 23, 24, "2"); paint(f, 5, 23, 24, "2")
    return f

# ---------- color ramps ----------
def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def ramp(base):
    """4-step ramp with hue shifting: darker shades drift toward purple, highlights toward warm."""
    r, g, b = [c / 255 for c in hexrgb(base)]
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    def shade(dl, dh, ds):
        h2 = (hh + dh) % 1.0
        return tuple(round(c * 255) for c in colorsys.hls_to_rgb(h2, max(0, min(1, ll + dl)), max(0, min(1, ss + ds))))
    toward_purple = lambda amt: (0.75 - hh) * amt if abs(0.75 - hh) < 0.5 else 0
    def shade_rel(mult, floor, dh, ds):
        h2 = (hh + dh) % 1.0
        l2 = max(ll * mult, floor)
        return tuple(round(c * 255) for c in colorsys.hls_to_rgb(h2, l2, max(0, min(1, ss + ds))))
    return {
        "1": shade_rel(0.52, 0.07, toward_purple(0.12), 0.05),
        "2": shade_rel(0.76, 0.11, toward_purple(0.06), 0.03),
        "3": hexrgb(base),
        "4": shade(max(0.10, (1 - ll) * 0.22), -0.02 if hh > 0.1 else 0.0, -0.05),
    }

FIXED = {
    "e": hexrgb("#2b2135"), "w": hexrgb("#ffffff"), "b": hexrgb("#f29a9a"), "m": hexrgb("#8c3b4a"),
    "k": hexrgb("#b0413e"), "g": hexrgb("#dff6ff"), "W": hexrgb("#ffffff"), "K": hexrgb("#1c1624"),
}
OUTLINE = hexrgb("#2a1f33")

def compose(look):
    layers = [
        (grid(body), ramp(look["skin"]), "skin"),
        (grid(pants), ramp(look["pants"]), "pants"),
        (grid(shoes), ramp(look["shoes"]), "shoes"),
        (grid(shirt), ramp(look["top"]), "top"),
        (grid(look["hair_style"]()), ramp(look["hair"]), "hair"),
    ]
    for acc in look.get("accessories", []):
        colors = ramp(look.get("hat", "#6cc24a"))
        colors["M"] = ramp(look.get("hat", "#6cc24a"))["1"]
        layers.append((grid(acc()), colors, acc.__name__))
    px, owner = {}, {}
    for mask, colors, name in layers:
        for xy, ch in mask.items():
            px[xy] = colors.get(ch) or FIXED[ch]
            owner[xy] = name
    # internal edges: a layer pixel sitting on top of a *different* layer gets its shadow tone at the boundary
    edged = dict(px)
    for (x, y), name in owner.items():
        if name in ("hair", "top", "frog_hat"):
            below = owner.get((x, y + 1))
            if below and below != name and below not in ("glasses",):
                layer_colors = next(c for m, c, n in layers if n == name)
                if px[(x, y)] in (layer_colors["3"], layer_colors["4"]):
                    edged[(x, y)] = layer_colors["2"]
    img = Image.new("RGBA", (W + 2, H + 2), (0, 0, 0, 0))
    for (x, y), c in edged.items():
        img.putpixel((x + 1, y + 1), c + (255,))
    # automatic outer outline
    out = img.copy()
    for y in range(H + 2):
        for x in range(W + 2):
            if img.getpixel((x, y))[3] == 0:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W + 2 and 0 <= ny < H + 2 and img.getpixel((nx, ny))[3]:
                        out.putpixel((x, y), OUTLINE + (255,))
                        break
    return out

LOOKS = [
    dict(name="bob + glasses", skin="#f3c49b", hair="#6b3f2a", hair_style=hair_bob, top="#5aa36a", pants="#3f5f9e", shoes="#3a2e3f", accessories=[glasses]),
    dict(name="frog hat", skin="#c98c5f", hair="#f06fa4", hair_style=hair_short, top="#f2c14e", pants="#4a4a6a", shoes="#e8e4dc", accessories=[frog_hat], hat="#6cc24a"),
    dict(name="long hair", skin="#8d5a3b", hair="#241c2c", hair_style=hair_long, top="#d9534f", pants="#2f3440", shoes="#7a4b2a"),
    dict(name="bun", skin="#ffe0c7", hair="#e8b64c", hair_style=hair_bun, top="#8e6cc9", pants="#6e5a4a", shoes="#3a2e3f", accessories=[glasses]),
]

SCALE = 8
pad = 24
cell_w = (W + 2) * SCALE
sheet = Image.new("RGBA", (pad + len(LOOKS) * (cell_w + pad), (H + 2) * SCALE + pad * 2 + 20), (246, 239, 228, 255))
draw = ImageDraw.Draw(sheet)
for i, look in enumerate(LOOKS):
    spr = compose(look).resize(((W + 2) * SCALE, (H + 2) * SCALE), Image.NEAREST)
    x = pad + i * (cell_w + pad)
    sheet.alpha_composite(spr, (x, pad))
    draw.text((x + 8, pad + (H + 2) * SCALE + 4), look["name"], fill=(80, 60, 70, 255))
sheet.save(Path(__file__).with_name("sprite-sample-claude.png"))
print("wrote sprite-sample-claude.png")
