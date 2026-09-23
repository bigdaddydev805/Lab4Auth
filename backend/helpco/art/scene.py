"""Server-side rendering of the whole office as a still image (used by /art/snapshot.png and previews).

The Godot client and the browser inspector do their own rendering from the same art; this is the
reference composition: background → y-sorted furniture and people → items on surfaces.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from . import character, office

POSE_ANIM = {"type": "type", "sit": "sit_down", "stand": "idle_down", "walk": "walk_down", "sip": "sip",
             "talk": "talk_down"}


def _frame(sheet: Image.Image, anim: str, col: int = 0) -> Image.Image:
    meta = character.meta()
    a = meta["animations"][anim]
    f = meta["frame_w"]
    col = col % a["frames"]
    return sheet.crop((col * f, a["row"] * f, col * f + f, a["row"] * f + f))


def anim_for(emp: dict) -> str:
    act = emp.get("activity", {})
    pose = act.get("pose", "stand")
    facing = emp.get("facing", "down")
    if pose == "walk" or act.get("kind") == "walk":
        return f"walk_{facing}"
    if pose in ("sit", "type"):
        if facing == "up":
            return "sit_up"
        return "type" if pose == "type" else "sit_down"
    if facing in ("up", "left", "right"):
        return f"idle_{facing}"
    return "hold_down" if emp.get("holding_kind") == "coffee" else "idle_down"


def render(state: dict, scale: int = 3, tick: int = 0) -> bytes:
    meta = office.office_meta()
    bg = Image.open(io.BytesIO(office.background_png())).convert("RGBA")
    atlas = Image.open(io.BytesIO(office.atlas_png())).convert("RGBA")
    canvas = bg.copy()
    sprites = meta["sprites"]

    def sprite(name):
        r = sprites[name]
        return atlas.crop((r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]))

    drawables = []  # (sort_y, order, image, x, y)
    for p in meta["placements"]:
        drawables.append((p["sort_y"], 0, sprite(p["sprite"]), p["x"], p["y"]))

    items = state.get("items", [])
    by_surface: dict[str, int] = {}
    for it in items:
        loc = it.get("location", {})
        name = meta["items"].get(it["kind"])
        if not name:
            continue
        img = sprite(name)
        if "on" in loc and loc["on"] in meta["surfaces"]:
            slots = meta["surfaces"][loc["on"]]
            i = by_surface.get(loc["on"], 0)
            by_surface[loc["on"]] = i + 1
            sx, sy = slots[i % len(slots)]
            drawables.append((sy + 40, 2, img, sx, sy - img.height + 4))
        elif "at" in loc:
            x, y = loc["at"]
            drawables.append((y * 16 + 15, 2, img, x * 16 + 4, y * 16 + 8))

    held = {it["location"].get("held_by"): it for it in items if "held_by" in it.get("location", {})}
    for emp in state.get("employees", []):
        if emp.get("status") != "present" or not emp.get("look"):
            continue
        sheet = Image.open(io.BytesIO(character.sheet_png(emp["look"]))).convert("RGBA")
        e = dict(emp)
        if emp["id"] in held:
            e["holding_kind"] = held[emp["id"]]["kind"]
        frame = _frame(sheet, anim_for(e), tick)
        x, y = emp["pos"]
        px = x * 16 + 8 - character.ANCHOR[0]
        py = (y + 1) * 16 - character.ANCHOR[1] + (2 if e.get("activity", {}).get("pose") in ("sit", "type") else 0)
        drawables.append(((y + 1) * 16, 1, frame, px, py))

    for _, _, img, x, y in sorted(drawables, key=lambda d: (d[0], d[1])):
        canvas.alpha_composite(img, (int(x), int(y)))

    out = canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(out)
    for emp in state.get("employees", []):
        if emp.get("status") == "present" and emp.get("name"):
            x, y = emp["pos"]
            label = emp["name"]
            l, t, r, b = d.textbbox((0, 0), label)
            cx, top = (x * 16 + 8) * scale, (y * 16 - 17) * scale
            d.rounded_rectangle((cx - (r - l) // 2 - 4, top - 2, cx + (r - l) // 2 + 4, top + (b - t) + 4),
                                radius=4, fill=(42, 31, 51, 210))
            d.text((cx - (r - l) // 2, top), label, fill=(255, 247, 232))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
