"""The office, drawn in code: a background (floors, walls, wall decor) plus a sprite atlas for
furniture and items. Everything is 16 px per tile, 3/4 top-down; furniture extends upward from its
footprint and is y-sorted with the characters by the clients.
"""
from __future__ import annotations

import io
from functools import lru_cache

from PIL import Image, ImageDraw

from ..office import DOOR, HEIGHT, MAP, TILE, WIDTH, Office

O = (42, 31, 51)  # outline


def rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


FLOORS = {
    "w": ("#c8925c", "#b27a48", "#d6a36c"),   # warm wood planks
    "b": ("#efe3cc", "#dfceb0", "#f7eedd"),   # checker tiles
    "m": ("#7482a8", "#6a789f", "#7f8db2"),   # carpet
    "e": ("#b9b1a8", "#a69e95", "#c7c0b8"),   # stone
}
WALLPAPER = {"w": "#f3e2c4", "b": "#cfe8dc", "m": "#dcd6f2", "e": "#f0d9cf"}
WALL_TOP = rgb("#4c405e")


def box(d: ImageDraw.ImageDraw, x0, y0, x1, y1, fill, outline=O):
    d.rectangle((x0, y0, x1, y1), fill=fill, outline=outline)


# ================================================================================== background
def _floor(d, x, y, ch):
    base, dark, light = (rgb(c) for c in FLOORS[ch])
    px, py = x * TILE, y * TILE
    d.rectangle((px, py, px + 15, py + 15), fill=base)
    if ch == "w":
        for i, oy in enumerate((0, 5, 10)):
            d.line((px, py + oy, px + 15, py + oy), fill=dark)
            off = ((x * 7 + y * 3 + i * 5) % 16)
            d.line((px + off, py + oy, px + off, py + oy + 4), fill=dark)
        d.point((px + 3, py + 2), fill=light)
    elif ch == "b":
        if (x + y) % 2:
            d.rectangle((px, py, px + 15, py + 15), fill=dark)
        d.line((px, py, px + 15, py), fill=light)
    elif ch == "m":
        for i in range(0, 16, 4):
            d.point((px + i + (y % 2) * 2, py + i), fill=dark)
            d.point((px + (i + 7) % 16, py + (i + 3) % 16), fill=light)
    elif ch == "e":
        d.rectangle((px, py, px + 7, py + 7), outline=dark)
        d.rectangle((px + 8, py + 8, px + 15, py + 15), outline=dark)
        d.point((px + 12, py + 3), fill=light)


def _wall(d, x, y):
    px, py = x * TILE, y * TILE
    below = MAP[y + 1][x] if y + 1 < HEIGHT else "#"
    if below != "#":  # a wall we see the face of: cap + wallpaper + baseboard
        paper = rgb(WALLPAPER.get(below, "#eee"))
        d.rectangle((px, py, px + 15, py + 3), fill=WALL_TOP)
        d.line((px, py + 3, px + 15, py + 3), fill=shade(WALL_TOP, 0.7))
        d.rectangle((px, py + 4, px + 15, py + 13), fill=paper)
        for sx in range(px + 2, px + 16, 5):
            d.point((sx, py + 7 + (sx // 5) % 3), fill=shade(paper, 0.93))
        d.rectangle((px, py + 14, px + 15, py + 15), fill=rgb("#a9805f"))
        d.line((px, py + 14, px + 15, py + 14), fill=rgb("#c49a74"))
    else:
        d.rectangle((px, py, px + 15, py + 15), fill=WALL_TOP)
        d.line((px, py, px + 15, py), fill=shade(WALL_TOP, 1.35))


def _window(d, x, y, w=2):
    px, py = x * TILE + 2, y * TILE + 5
    x1 = px + w * TILE - 5
    box(d, px, py, x1, py + 7, rgb("#9fd3ef"))
    d.line((px + 1, py + 1, x1 - 1, py + 1), fill=rgb("#d8f0fb"))
    d.line(((px + x1) // 2, py, (px + x1) // 2, py + 7), fill=O)
    d.line((px + 2, py + 5, px + 4, py + 3), fill=rgb("#e6f6fd"))


def _whiteboard(d, x, y):
    px, py = x * TILE + 2, y * TILE + 4
    box(d, px, py, px + 27, py + 9, rgb("#f7f7fb"))
    d.line((px + 3, py + 3, px + 12, py + 3), fill=rgb("#4f7fd9"))
    d.line((px + 3, py + 6, px + 9, py + 6), fill=rgb("#d9534f"))
    d.rectangle((px + 4, py + 10, px + 23, py + 10), fill=rgb("#8a8793"))


def _question_board(d, x, y):
    px, py = x * TILE + 2, y * TILE + 4
    box(d, px, py, px + 27, py + 9, rgb("#2d3350"))
    for i, col in enumerate(("#f2c14e", "#9bcf53", "#6cb4e4")):
        d.line((px + 3, py + 2 + i * 2, px + 3 + 14 - i * 4, py + 2 + i * 2), fill=rgb(col))
    for gx, gy in ((22, 2), (23, 2), (24, 3), (23, 4), (23, 6)):   # a tiny "?"
        d.point((px + gx, py + gy), fill=rgb("#f2c14e"))


def _clock(d, x, y):
    cx, cy = x * TILE + 8, y * TILE + 8
    d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=rgb("#f7f7fb"), outline=O)
    d.line((cx, cy, cx, cy - 3), fill=O)
    d.line((cx, cy, cx + 2, cy), fill=O)


PIXEL_FONT = {  # 3x5 glyphs for tiny signs
    "H": ["1.1", "1.1", "111", "1.1", "1.1"], "e": ["...", "111", "1.1", "11.", "111"],
    "l": ["1", "1", "1", "1", "1"], "p": ["...", "111", "1.1", "111", "1.."],
    "C": ["111", "1..", "1..", "1..", "111"], "o": ["...", "111", "1.1", "1.1", "111"],
}


def pixel_text(d, x, y, text, fill):
    for ch in text:
        glyph = PIXEL_FONT.get(ch)
        if glyph:
            for gy, row in enumerate(glyph):
                for gx, c in enumerate(row):
                    if c == "1":
                        d.point((x + gx, y + gy), fill=fill)
        x += (len(glyph[0]) if glyph else 3) + 1


def _sign(d, x, y):
    px, py = x * TILE + 2, y * TILE + 4
    box(d, px, py, px + 27, py + 8, rgb("#f06fa4"))
    pixel_text(d, px + 4, py + 2, "HelpCo", rgb("#fff7e8"))


def _door(d, x, y):
    px, py = x * TILE, y * TILE
    d.rectangle((px, py, px + 15, py + 15), fill=rgb("#8a5a3a"))
    d.rectangle((px + 1, py, px + 14, py + 3), fill=rgb("#a8744d"))
    d.point((px + 12, py + 8), fill=rgb("#f2c14e"))
    # welcome mat on the tile inside
    d.rectangle((px + 2, py - 12, px + 13, py - 4), fill=rgb("#d9534f"), outline=shade(rgb("#d9534f"), 0.7))


@lru_cache(maxsize=1)
def background_png() -> bytes:
    img = Image.new("RGBA", (WIDTH * TILE, HEIGHT * TILE), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    for y, row in enumerate(MAP):
        for x, ch in enumerate(row):
            if ch == "#":
                _wall(d, x, y)
            else:
                _floor(d, x, y, ch)
    # soft shadow along the base of walls
    for y, row in enumerate(MAP):
        for x, ch in enumerate(row):
            if ch != "#" and y > 0 and MAP[y - 1][x] == "#":
                d.line((x * TILE, y * TILE, x * TILE + 15, y * TILE), fill=(0, 0, 0, 255))
                img.paste(Image.new("RGBA", (16, 2), (40, 25, 45, 70)), (x * TILE, y * TILE + 1),
                          Image.new("L", (16, 2), 70))
    _whiteboard(d, 5, 0)
    _window(d, 2, 0, 2)
    _window(d, 12, 0, 2)
    _window(d, 19, 0, 2)
    _question_board(d, 6, 6)
    _clock(d, 10, 6)
    _window(d, 15, 6, 2)
    _sign(d, 21, 6)
    _door(d, *DOOR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ================================================================================== sprites
def _spr(w, h):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def spr_desk():
    img, d = _spr(48, 26)
    wood, dark, light = rgb("#b98357"), rgb("#8d5f3d"), rgb("#d19f70")
    box(d, 0, 10, 47, 17, wood)                      # top surface
    d.line((1, 11, 46, 11), fill=light)
    box(d, 1, 17, 46, 25, dark)                      # front panel
    box(d, 4, 19, 17, 24, shade(dark, 0.85))         # drawer
    d.line((9, 21, 12, 21), fill=light)
    box(d, 31, 1, 44, 11, rgb("#4a4458"))            # monitor (back)
    d.rectangle((32, 2, 43, 10), fill=rgb("#5b566b"))
    d.line((33, 3, 38, 3), fill=rgb("#716b82"))
    box(d, 36, 11, 39, 13, rgb("#3a3548"))
    return img, 16 + 10 - 16  # returns (image, pixels above footprint top)


def spr_chair():
    img, d = _spr(16, 22)
    c, dark = rgb("#5b566b"), rgb("#3f3a4d")
    box(d, 3, 0, 12, 10, c)
    d.line((4, 1, 11, 1), fill=rgb("#766f88"))
    box(d, 2, 11, 13, 15, dark)
    d.line((7, 16, 7, 19), fill=O)
    d.line((4, 20, 11, 20), fill=O)
    return img


def spr_printer():
    img, d = _spr(16, 22)
    box(d, 1, 6, 14, 21, rgb("#d9d6df"))
    d.rectangle((2, 7, 13, 9), fill=rgb("#efedf3"))
    box(d, 3, 2, 12, 7, rgb("#f7f7fb"))              # paper
    box(d, 3, 13, 12, 16, rgb("#8a8793"))
    d.point((12, 11), fill=rgb("#9bcf53"))
    return img


def spr_plant(big=True):
    h = 28 if big else 18
    img, d = _spr(16, h)
    pot = rgb("#c9652e")
    box(d, 4, h - 8, 11, h - 1, pot)
    d.line((5, h - 7, 10, h - 7), fill=shade(pot, 1.2))
    leaf, ldark = rgb("#5aa36a"), rgb("#3f7f52")
    blobs = [(3, 2, 9, 12), (7, 0, 13, 10), (1, 8, 8, 17), (8, 7, 15, 17), (5, 10, 11, 20)] if big else \
        [(3, 2, 9, 9), (7, 1, 13, 8), (4, 5, 12, 11)]
    for x0, y0, x1, y1 in blobs:
        d.ellipse((x0, y0, x1, y1), fill=leaf, outline=O)
    for x0, y0, x1, y1 in blobs:
        d.ellipse((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=leaf)
        d.point(((x0 + x1) // 2, y0 + 2), fill=rgb("#8fd08e"))
        d.point(((x0 + x1) // 2 + 1, y1 - 3), fill=ldark)
    return img


def spr_fridge():
    img, d = _spr(16, 30)
    box(d, 1, 0, 14, 29, rgb("#e8f1f0"))
    d.line((1, 11, 14, 11), fill=O)
    d.line((12, 4, 12, 8), fill=rgb("#8a8793"))
    d.line((12, 14, 12, 20), fill=rgb("#8a8793"))
    d.point((4, 16), fill=rgb("#f06fa4"))   # a magnet
    d.point((6, 18), fill=rgb("#f2c14e"))
    return img


def spr_coffee_machine():
    img, d = _spr(16, 28)
    box(d, 0, 14, 15, 27, rgb("#a8744d"))                    # counter segment
    d.line((1, 15, 14, 15), fill=rgb("#c49a74"))
    box(d, 3, 2, 12, 15, rgb("#3a3548"))                     # machine
    d.rectangle((4, 3, 11, 5), fill=rgb("#5b566b"))
    d.point((10, 4), fill=rgb("#d9534f"))
    box(d, 6, 10, 9, 14, rgb("#f4ead8"))                     # cup
    return img


def spr_counter():
    img, d = _spr(48, 22)
    box(d, 0, 6, 47, 21, rgb("#a8744d"))
    box(d, 0, 4, 47, 8, rgb("#e8e2d6"))                      # countertop
    for x in (2, 18, 34):
        box(d, x, 10, x + 12, 19, shade(rgb("#a8744d"), 0.88))
        d.point((x + 6, 14), fill=rgb("#e8e2d6"))
    box(d, 30, 4, 42, 7, rgb("#9fb4c4"))                    # sink
    d.line((36, 1, 36, 4), fill=rgb("#8a8793"))
    return img


def spr_couch():
    img, d = _spr(48, 24)
    c, dark, light = rgb("#e8893a"), rgb("#c96c26"), rgb("#f2a563")
    box(d, 2, 0, 45, 11, dark)                               # backrest
    d.line((3, 1, 44, 1), fill=c)
    box(d, 0, 6, 5, 22, c)                                   # arms
    box(d, 42, 6, 47, 22, c)
    box(d, 5, 11, 42, 21, c)                                 # seat
    for x in (17, 30):
        d.line((x, 12, x, 20), fill=dark)
    d.line((6, 12, 41, 12), fill=light)
    box(d, 5, 21, 42, 23, dark)
    return img


def spr_coffee_table():
    img, d = _spr(48, 16)
    box(d, 1, 2, 46, 9, rgb("#8d5f3d"))
    d.line((2, 3, 45, 3), fill=rgb("#b98357"))
    d.line((4, 10, 4, 14), fill=O)
    d.line((43, 10, 43, 14), fill=O)
    return img


def spr_meeting_table():
    img, d = _spr(64, 38)
    wood, dark, light = rgb("#b98357"), rgb("#8d5f3d"), rgb("#d19f70")
    box(d, 1, 4, 62, 29, wood)
    d.line((2, 5, 61, 5), fill=light)
    box(d, 1, 29, 62, 35, dark)
    box(d, 12, 12, 21, 18, rgb("#f7f7fb"))                   # papers
    box(d, 40, 10, 45, 15, rgb("#f2c14e"))                   # sticky pad
    return img


def spr_coat_rack():
    img, d = _spr(16, 30)
    d.line((8, 3, 8, 28), fill=rgb("#5b3a24"), width=2)
    d.line((4, 29, 12, 29), fill=O)
    d.ellipse((2, 5, 9, 17), fill=rgb("#3f6fc4"), outline=O)  # a coat
    d.line((9, 3, 13, 5), fill=rgb("#5b3a24"))
    return img


def spr_items():
    out = {}

    def item(name, w, h, draw):
        img, d = _spr(w, h)
        draw(d)
        out[name] = img
    item("mug", 8, 8, lambda d: (box(d, 1, 2, 5, 7, rgb("#f4ead8")), d.point((6, 4), fill=O), d.point((7, 4), fill=O)))
    item("coffee", 8, 9, lambda d: (box(d, 1, 3, 5, 8, rgb("#f4ead8")), d.line((2, 4, 4, 4), fill=rgb("#6b4226")),
                                    d.point((6, 5), fill=O), d.point((3, 1), fill=rgb("#e9e4f0")),
                                    d.point((2, 0), fill=rgb("#e9e4f0"))))
    item("note", 7, 7, lambda d: (box(d, 0, 0, 6, 6, rgb("#f2e05e")), d.line((1, 2, 5, 2), fill=rgb("#b9a83a"))))
    item("paper", 7, 9, lambda d: (box(d, 0, 0, 6, 8, rgb("#fbfbff")), d.line((1, 2, 5, 2), fill=rgb("#8a8793")),
                                   d.line((1, 4, 4, 4), fill=rgb("#8a8793"))))
    item("succulent", 8, 9, lambda d: (box(d, 1, 5, 6, 8, rgb("#c9652e")), d.ellipse((0, 0, 7, 6), fill=rgb("#7bc47f"),
                                                                                         outline=O)))
    item("photo_frame", 8, 8, lambda d: (box(d, 0, 0, 7, 7, rgb("#8d5f3d")), d.rectangle((2, 2, 5, 5),
                                                                                           fill=rgb("#9fd3ef")),
                                         d.point((3, 4), fill=rgb("#5aa36a"))))
    item("rubber_duck", 8, 8, lambda d: (d.ellipse((0, 3, 7, 7), fill=rgb("#f2d14e"), outline=O),
                                         d.ellipse((3, 0, 7, 4), fill=rgb("#f2d14e"), outline=O),
                                         d.point((6, 2), fill=rgb("#e8893a"))))
    item("books", 10, 8, lambda d: (box(d, 0, 4, 9, 7, rgb("#3f6fc4")), box(d, 1, 1, 8, 4, rgb("#d9534f"))))
    item("snow_globe", 8, 9, lambda d: (d.ellipse((0, 0, 7, 7), fill=rgb("#cfeaf7"), outline=O),
                                        box(d, 1, 6, 6, 8, rgb("#8d5f3d")), d.point((3, 3), fill=(255, 255, 255))))
    return out


SPRITE_FUNCS = {
    "desk": lambda: spr_desk()[0], "desk_chair": spr_chair, "printer": spr_printer, "plant_big": lambda: spr_plant(True),
    "plant_small": lambda: spr_plant(False), "fridge": spr_fridge, "coffee_machine": spr_coffee_machine,
    "counter": spr_counter, "couch": spr_couch, "coffee_table": spr_coffee_table,
    "meeting_table": spr_meeting_table, "coat_rack": spr_coat_rack,
}


@lru_cache(maxsize=1)
def _atlas():
    sprites = {k: f() for k, f in SPRITE_FUNCS.items()}
    sprites.update({f"item_{k}": v for k, v in spr_items().items()})
    x, y, row_h, max_w = 0, 0, 0, 256
    rects = {}
    for name, img in sprites.items():
        if x + img.width > max_w:
            x, y, row_h = 0, y + row_h + 1, 0
        rects[name] = (x, y, img.width, img.height)
        x += img.width + 1
        row_h = max(row_h, img.height)
    atlas = Image.new("RGBA", (max_w, y + row_h + 1), (0, 0, 0, 0))
    for name, img in sprites.items():
        rx, ry, _, _ = rects[name]
        atlas.alpha_composite(img, (rx, ry))
    buf = io.BytesIO()
    atlas.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), rects


def atlas_png() -> bytes:
    return _atlas()[0]


def office_meta() -> dict:
    """Where every sprite goes. Positions are world pixels (top-left); sort_y is the y-sort key."""
    _, rects = _atlas()
    office = Office()
    placements = []
    surfaces = {}
    for o in office.objects.values():
        if o.kind not in SPRITE_FUNCS:
            continue
        _, _, w, h = rects[o.kind]
        foot_bottom = (o.y + o.h) * TILE
        px = o.x * TILE + (o.w * TILE - w) // 2
        py = foot_bottom - h
        sort_y = foot_bottom
        if o.kind == "desk_chair":
            sort_y = o.y * TILE + 2           # behind whoever sits in it
        elif o.kind == "couch":
            sort_y = foot_bottom - 2          # people sit on it
        elif o.kind == "desk":
            py += 0
        placements.append({"id": o.id, "sprite": o.kind, "x": px, "y": py, "sort_y": sort_y})
        if o.surface:
            top = py + {"desk": 11, "counter": 4, "coffee_table": 3, "meeting_table": 8}.get(o.kind, 2)
            xs = {"desk": [3, 13, 22, 25], "counter": [3, 14, 22, 44 - 8], "coffee_table": [6, 20, 34],
                  "meeting_table": [26, 34, 48, 6]}.get(o.kind, [4, 12, 20])
            surfaces[o.id] = [[px + dx, top] for dx in xs]
    for o in office.objects.values():   # chairs at the meeting table's seats
        if o.kind == "meeting_table":
            for s in o.spots:
                placements.append({"id": f"{o.id}_chair_{s.x}_{s.y}", "sprite": "desk_chair",
                                   "x": s.x * TILE, "y": (s.y + 1) * TILE - 22,
                                   "sort_y": s.y * TILE + (2 if s.facing == "down" else 17)})
    return {
        "tile": TILE, "width": WIDTH, "height": HEIGHT,
        "background": "/art/office_bg.png", "atlas": "/art/office_atlas.png",
        "sprites": {k: {"x": r[0], "y": r[1], "w": r[2], "h": r[3]} for k, r in rects.items()},
        "placements": placements, "surfaces": surfaces,
        "items": {k: f"item_{k}" for k in ("mug", "coffee", "note", "paper", "succulent", "photo_frame",
                                            "rubber_duck", "books", "snow_globe")},
    }
