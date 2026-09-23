"""Modular, palette-ramped character sprites drawn in code.

Every body part is a mask whose digits are shades of a color ramp (1 = darkest … 4 = highlight).
A look is just data from the catalog (part ids + color names), so an employee's own choice becomes
a full animated sprite sheet: idle, walk ×4 directions, typing, sitting, sipping coffee, talking.
"""
from __future__ import annotations

import colorsys
import hashlib
import io
import json
from functools import lru_cache

from PIL import Image

from .. import catalog

SPR = 32              # the character is drawn on a 32×32 grid
FRAME = SPR + 2       # +1px border for the outline
ANCHOR = (17, 32)     # feet position inside a frame (client aligns this with the tile)

ANIMATIONS = [        # name, frame count, fps — row order in the sheet
    ("idle_down", 8, 5), ("walk_down", 4, 8), ("walk_up", 4, 8), ("walk_right", 4, 8), ("walk_left", 4, 8),
    ("idle_up", 4, 4), ("idle_right", 4, 4), ("idle_left", 4, 4), ("type", 8, 6), ("sit_down", 8, 4),
    ("sit_up", 4, 3), ("sip", 8, 4), ("talk_down", 4, 6), ("hold_down", 8, 4),
]
COLUMNS = 8


class Mask(dict):
    def rect(self, x0, y0, x1, y1, ch):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self[(x, y)] = ch
        return self

    def row(self, y, x0, x1, ch):
        return self.rect(x0, y, x1, y, ch)

    def dot(self, x, y, ch):
        self[(x, y)] = ch
        return self

    def erase(self, x0, y0, x1, y1):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.pop((x, y), None)
        return self

    def shifted(self, dx, dy):
        return Mask({(x + dx, y + dy): c for (x, y), c in self.items()})


# ----------------------------------------------------------------------------- heads
def head_front(eyes="open", mouth="closed"):
    m = Mask()
    for y, (a, b) in {5: (11, 20), 6: (10, 21), 7: (9, 22)}.items():
        m.row(y, a, b, "3")
    m.rect(8, 8, 23, 14, "3").row(15, 9, 22, "3")
    m.row(16, 10, 21, "2").row(16, 12, 19, "3").row(17, 12, 19, "2")
    m.rect(23, 8, 23, 14, "2").dot(22, 15, "2")
    for ex in (11, 19):
        if eyes == "open":
            m.rect(ex, 11, ex + 1, 13, "e").dot(ex + 1, 12, "w")
        else:
            m.row(12, ex, ex + 1, "e")
    m.row(14, 9, 10, "b").row(14, 21, 22, "b")
    if mouth == "open":
        m.rect(15, 15, 16, 16, "m")
    else:
        m.row(15, 15, 16, "m")
    return m


def head_back():
    m = Mask()
    for y, (a, b) in {5: (11, 20), 6: (10, 21), 7: (9, 22)}.items():
        m.row(y, a, b, "3")
    return m.rect(8, 8, 23, 14, "3").row(15, 9, 22, "3").row(16, 10, 21, "2").row(17, 12, 19, "2")


def head_side(eyes="open"):
    m = Mask()
    for y, (a, b) in {5: (12, 19), 6: (11, 20), 7: (10, 21)}.items():
        m.row(y, a, b, "3")
    m.rect(9, 8, 22, 14, "3").row(15, 10, 21, "3").row(16, 11, 20, "2").row(17, 13, 17, "2")
    m.rect(9, 8, 9, 14, "2").rect(13, 12, 13, 13, "2")
    if eyes == "open":
        m.rect(19, 11, 20, 13, "e").dot(20, 12, "w")
    else:
        m.row(12, 19, 20, "e")
    return m.row(14, 20, 21, "b").dot(21, 15, "m")


# ----------------------------------------------------------------------------- body
def shirt_front(highlight=True, right_arm="down", hands_forward=False):
    m = Mask()
    m.row(18, 9, 22, "3").row(18, 13, 18, "2").rect(8, 19, 23, 23, "3")
    m.rect(10, 19, 10, 23, "2").rect(21, 19, 21, 23, "2").rect(22, 19, 23, 23, "2")
    m.rect(10, 24, 21, 25, "3").row(25, 10, 21, "2")
    if highlight:
        m.row(19, 11, 13, "4").row(20, 11, 12, "4")
    if right_arm == "mid":
        m.erase(22, 21, 23, 23).rect(21, 18, 22, 20, "2")
    elif right_arm == "up":
        m.erase(22, 20, 23, 23).rect(21, 17, 22, 19, "2")
    if hands_forward:
        m.erase(8, 22, 9, 23).erase(22, 22, 23, 23)
        m.rect(10, 22, 11, 23, "2").rect(20, 22, 21, 23, "2")
    return m


def hands_front(left_dy=0, right_dy=0, right="down", forward=False):
    m = Mask()
    if forward:
        return m.rect(11, 24 + left_dy, 12, 24 + left_dy, "3").rect(19, 24 + right_dy, 20, 24 + right_dy, "3")
    m.rect(8, 24 + left_dy, 9, 24 + left_dy, "3").rect(8, 25 + left_dy, 9, 25 + left_dy, "2")
    if right == "down":
        m.rect(22, 24 + right_dy, 23, 24 + right_dy, "3").rect(22, 25 + right_dy, 23, 25 + right_dy, "2")
    elif right == "mid":
        m.rect(22, 20, 23, 21, "3")
    elif right == "up":
        m.rect(20, 15, 21, 16, "3")
    return m


def shirt_side(arm_dx=0):
    m = Mask()
    m.row(18, 12, 19, "3").rect(11, 19, 20, 24, "3").row(25, 12, 19, "2").rect(11, 19, 11, 24, "2")
    ax = 14 + arm_dx
    m.rect(ax, 19, ax + 2, 23, "3").rect(ax, 19, ax, 23, "2").rect(ax + 3, 20, ax + 3, 23, "2")
    return m.row(19, ax + 1, ax + 2, "4")


def hands_side(arm_dx=0):
    ax = 14 + arm_dx
    return Mask().rect(ax + 1, 24, ax + 2, 24, "3").rect(ax + 1, 25, ax + 2, 25, "2")


def legs_front(lift_left=0, lift_right=0):
    pants, shoes = Mask(), Mask()
    pants.row(26, 10, 21, "3").row(26, 15, 16, "2")
    for x0, x1, lift in ((10, 14, lift_left), (17, 21, lift_right)):
        pants.rect(x0, 27, x1, 28 - lift, "3").rect(x1, 27, x1, 28 - lift, "2")
        sy = 29 - lift
        shoes.row(sy, x0 - 1, x1, "3").row(sy + 1, x0 - 1, x1, "2").row(sy, x0, x0 + 1, "4")
    return pants, shoes


def legs_sit():
    """Seated, knees toward the viewer: short legs, feet just below."""
    pants, shoes = Mask(), Mask()
    pants.row(26, 10, 21, "3").row(26, 15, 16, "2").row(27, 10, 14, "2").row(27, 17, 21, "2")
    shoes.row(28, 10, 13, "3").row(28, 18, 21, "3").row(29, 10, 13, "2").row(29, 18, 21, "2")
    return pants, shoes


def legs_side(front_c=16, back_c=15):
    pants, shoes = Mask(), Mask()
    pants.row(26, 12, 19, "3")
    for c, near in ((back_c, False), (front_c, True)):
        tone = "3" if near else "2"
        mid = round((15.5 + c) / 2)
        pants.row(27, mid - 1, mid + 1, tone).row(28, c - 1, c + 1, tone)
        shoes.row(29, c - 1, c + 3, "3" if near else "2").row(30, c - 1, c + 3, "2" if near else "1")
        if near:
            shoes.dot(c, 29, "4")
    return pants, shoes


# ----------------------------------------------------------------------------- hair
def hair(style, view):
    if view == "side":
        return hair_side(style)
    m = Mask()
    if style == "buzz":
        m.row(4, 10, 21, "3").rect(9, 5, 22, 7, "3").row(8, 8, 23, "3").row(4, 11, 13, "4").row(5, 10, 11, "4")
        m.rect(21, 5, 22, 8, "2")
        if view == "back":
            m.rect(8, 9, 23, 13, "3").rect(22, 9, 23, 13, "2").row(14, 9, 22, "2")
        return m
    top_rows = {2: (12, 19), 3: (10, 21), 4: (9, 22)}
    if style in ("short", "bun"):
        top_rows[2] = None
    for y, span in top_rows.items():
        if span:
            m.row(y, *span, "3")
    if style in ("short", "bun"):
        m.row(2, 11, 13, "3").row(2, 17, 19, "3").rect(8, 5, 23, 8, "3").rect(22, 5, 23, 8, "2")
    else:
        m.rect(7, 5, 24, 9, "3").rect(23, 5, 24, 9, "2")
    m.row(3, 11, 14, "4").row(4, 10, 12, "4").row(5, 9, 10, "4")
    if view == "back":
        if style in ("short", "bun"):
            m.rect(8, 9, 23, 14, "3").rect(22, 9, 23, 14, "2").row(15, 9, 22, "2")
        else:
            m.rect(7, 10, 24, 16, "3").rect(23, 10, 24, 16, "2").row(17, 8, 23, "2")
        if style == "long":
            m.rect(7, 17, 24, 22, "3").rect(23, 17, 24, 22, "2").row(23, 8, 23, "2")
        bottom = {"long": 22, "bob": 15}.get(style, 13)
        start = 10 if style in ("bob", "long") else 9
        for x in (11, 16, 20):
            m.rect(x, start, x, bottom, "2")
        m.row(6, 10, 13, "4").row(7, 9, 11, "4").row(6, 18, 20, "2")
    else:
        if style in ("short", "bun"):
            m.row(9, 8, 13, "3").row(9, 17, 23, "3").dot(13, 9, "2").row(9, 22, 23, "2")
            m.row(10, 8, 9, "3").row(10, 22, 23, "2").dot(8, 11, "2").dot(23, 11, "2")
        else:
            m.dot(12, 9, "2").dot(16, 9, "2").dot(20, 9, "2")
            for x0, x1 in ((7, 10), (13, 14), (17, 18), (21, 24)):
                m.row(10, x0, x1, "3")
            m.dot(10, 10, "2").dot(14, 10, "2").dot(18, 10, "2").row(10, 23, 24, "2")
            m.rect(7, 11, 9, 15, "3").rect(9, 11, 9, 15, "2").rect(22, 11, 24, 15, "2")
            m.row(16, 7, 9, "2").row(16, 22, 24, "1")
            if style == "long":
                m.rect(6, 11, 6, 22, "3").rect(7, 16, 8, 22, "3").rect(8, 16, 8, 22, "2")
                m.rect(23, 16, 25, 22, "2").rect(25, 11, 25, 15, "2")
                m.row(23, 6, 8, "2").row(23, 23, 25, "1")
    if style == "bun":
        m.row(0, 14, 17, "3").row(1, 13, 18, "3").row(1, 14, 15, "4").row(2, 13, 18, "2")
    return m


def hair_side(style):
    m = Mask()
    if style == "buzz":
        return m.row(4, 11, 19, "3").rect(9, 5, 21, 7, "3").row(8, 9, 12, "3").row(4, 12, 15, "4").rect(9, 9, 11, 10, "2")
    m.row(2, 11, 18, "3").row(3, 9, 20, "3").row(4, 8, 21, "3").rect(8, 5, 22, 8, "3")
    m.row(3, 12, 16, "4").row(4, 14, 18, "4")
    if style in ("short", "bun"):
        m.row(9, 8, 12, "3").row(9, 18, 22, "3").rect(8, 10, 11, 11, "2").dot(22, 10, "2")
    else:
        m.row(9, 8, 22, "3").row(10, 8, 13, "3").row(10, 18, 22, "3").dot(20, 10, "2")
        m.rect(8, 11, 12, 15, "3").rect(8, 11, 8, 15, "2").row(16, 8, 12, "2")
        if style == "long":
            m.rect(7, 16, 12, 22, "3").rect(7, 16, 7, 22, "2").row(23, 7, 12, "2")
    if style == "bun":
        m.row(0, 9, 12, "3").row(1, 8, 12, "3").row(1, 9, 10, "4").row(2, 8, 11, "2")
    return m


# ----------------------------------------------------------------------------- accessories & props
def glasses(view):
    m = Mask()
    if view == "front":
        m.row(11, 14, 17, "2")
        for x in (10, 13, 18, 21):
            m.rect(x, 11, x, 13, "2")
        m.row(14, 11, 12, "2").row(14, 19, 20, "2")
    elif view == "side":
        m.rect(18, 11, 18, 13, "2").rect(21, 11, 21, 13, "2").row(14, 19, 20, "2").row(11, 15, 17, "2")
    return m


def frog_hat(view, blink=False):
    m = Mask()
    if view == "side":
        m.row(3, 9, 21, "3").rect(8, 4, 22, 5, "3").row(6, 8, 22, "2").row(4, 10, 12, "4")
        m.row(0, 15, 18, "3").rect(15, 1, 18, 2, "3")
        if blink:
            m.row(2, 16, 17, "M")
        else:
            m.rect(16, 1, 17, 2, "W").dot(17, 2, "K")
        return m.row(1, 10, 12, "2").row(2, 10, 13, "3").row(5, 19, 22, "M")
    m.row(0, 9, 12, "3").row(0, 19, 22, "3")
    for x0 in (9, 19):
        m.rect(x0, 1, x0 + 3, 2, "3")
        if view == "front":
            if blink:
                m.row(2, x0 + 1, x0 + 2, "M")
            else:
                m.rect(x0 + 1, 1, x0 + 2, 2, "W")
    if view == "front" and not blink:
        m.dot(11, 2, "K").dot(20, 2, "K")
    m.row(3, 8, 23, "3").row(3, 10, 12, "4").rect(7, 4, 24, 5, "3").row(4, 9, 11, "4")
    m.rect(23, 4, 24, 5, "2").row(6, 7, 24, "2")
    if view == "front":
        m.row(5, 12, 19, "M")
    return m


def mug(pos):
    x, y = {"low": (23, 21), "mid": (22, 17), "mouth": (16, 14)}[pos]
    m = Mask().rect(x, y, x + 3, y + 3, "3").rect(x + 3, y, x + 3, y + 3, "2").row(y + 3, x, x + 3, "2")
    if pos != "mouth":
        m.row(y, x, x + 3, "c")
    return m.dot(x - 1, y + 1, "2").dot(x - 1, y + 2, "2")


def steam(pos, t):
    if pos != "low":
        return Mask()
    x = 24 + (t % 2)
    return Mask().dot(x, 19 - (t % 2), "s").dot(x + 1 - 2 * (t % 2), 17, "s")


# ----------------------------------------------------------------------------- color
def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=512)
def ramp(base):
    r, g, b = [c / 255 for c in hexrgb(base)]
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    purple = (0.75 - hh) if abs(0.75 - hh) < 0.5 else 0

    def rel(mult, floor, dh, ds):
        return tuple(round(c * 255) for c in colorsys.hls_to_rgb((hh + dh) % 1, max(ll * mult, floor), min(1, ss + ds)))

    hi_l = min(1, ll + max(0.10, (1 - ll) * 0.22))
    return {"1": rel(0.52, 0.07, purple * 0.12, 0.05), "2": rel(0.76, 0.11, purple * 0.06, 0.03),
            "3": hexrgb(base), "4": tuple(round(c * 255) for c in colorsys.hls_to_rgb(hh, hi_l, max(0, ss - 0.05)))}


FIXED = {k: hexrgb(v) for k, v in {"e": "#2b2135", "w": "#ffffff", "b": "#f29a9a", "m": "#8c3b4a", "W": "#ffffff",
                                    "K": "#1c1624", "c": "#6b4226", "s": "#e9e4f0"}.items()}
OUTLINE = hexrgb("#2a1f33")


# ----------------------------------------------------------------------------- poses → frames
def layers_for(look: dict, pose: dict):
    pal = catalog.palette_for(look)
    view = pose.get("view", "front")
    bob = pose.get("bob", 0)
    eyes = pose.get("eyes", "open")
    skin, top = ramp(pal["skin"]), ramp(pal["top"])
    if view == "side":
        head = head_side(eyes)
        pants, shoes = legs_side(*pose.get("stride", (16, 15)))
        shirt, hands = shirt_side(pose.get("arm_dx", 0)), hands_side(pose.get("arm_dx", 0))
    else:
        head = head_front(eyes, pose.get("mouth", "closed")) if view == "front" else head_back()
        pants, shoes = legs_sit() if pose.get("sit") else legs_front(*pose.get("lift", (0, 0)))
        right = pose.get("right_arm", "down")
        fwd = pose.get("hands_forward", False)
        shirt = shirt_front(highlight=(view == "front"), right_arm=right, hands_forward=fwd)
        hl, hr = pose.get("hand_dy", (0, 0))
        hands = hands_front(hl, hr, right, fwd)
    up = lambda m: m.shifted(0, bob)  # noqa: E731
    out = [(up(head), skin, "skin"), (pants, ramp(pal["pants"]), "pants"), (shoes, ramp(pal["shoes"]), "shoes"),
           (up(shirt), top, "top"), (up(hands), skin, "hands"), (up(hair(look["hair_style"], view)),
                                                                  ramp(pal["hair"]), "hair")]
    for acc in look.get("accessories", []):
        if acc == "glasses":
            out.append((up(glasses(view)), ramp(pal["accessory"]), "glasses"))
        elif acc == "frog_hat":
            hat = dict(ramp(pal["hat"]))
            hat["M"] = hat["1"]
            out.append((up(frog_hat(view, pose.get("frog_blink", False))), hat, "frog_hat"))
    if "mug" in pose:
        out.append((up(mug(pose["mug"])), ramp("#f4ead8"), "mug"))
        out.append((up(steam(pose["mug"], pose.get("t", 0))), {}, "steam"))
    if pose.get("mirror"):
        out = [(Mask({(SPR - 1 - x, y): c for (x, y), c in m.items()}), col, n) for m, col, n in out]
    return out


def render_frame(look: dict, pose: dict) -> Image.Image:
    layers = layers_for(look, pose)
    px, owner = {}, {}
    for mask, colors, name in layers:
        for xy, ch in mask.items():
            px[xy] = colors.get(ch) or FIXED[ch]
            owner[xy] = name
    lookup = {n: c for _, c, n in layers}
    for (x, y), name in list(owner.items()):
        if name in ("hair", "top", "frog_hat") and owner.get((x, y + 1)) not in (None, name, "glasses", "steam"):
            if px[(x, y)] in (lookup[name]["3"], lookup[name]["4"]):
                px[(x, y)] = lookup[name]["2"]
    img = Image.new("RGBA", (FRAME, FRAME), (0, 0, 0, 0))
    pix = img.load()
    for (x, y), c in px.items():
        if 0 <= x < SPR and 0 <= y < SPR and owner[(x, y)] != "steam":
            pix[x + 1, y + 1] = c + (255,)
    out = img.copy()
    opix = out.load()
    for y in range(FRAME):
        for x in range(FRAME):
            if pix[x, y][3]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < FRAME and 0 <= ny < FRAME and pix[nx, ny][3]:
                    opix[x, y] = OUTLINE + (255,)
                    break
    for (x, y), name in owner.items():
        if name == "steam" and 0 <= x < SPR and 0 <= y < SPR and not opix[x + 1, y + 1][3]:
            opix[x + 1, y + 1] = FIXED["s"] + (255,)
    return out


def _walk(view):
    if view == "side":
        cyc = [dict(stride=(19, 12), arm_dx=-2, bob=1), dict(stride=(16, 15)),
               dict(stride=(12, 19), arm_dx=2, bob=1), dict(stride=(15, 16))]
    else:
        cyc = [dict(lift=(1, 0), hand_dy=(0, -1), bob=1), dict(), dict(lift=(0, 1), hand_dy=(-1, 0), bob=1), dict()]
    return [dict(view=view, t=t, **c) for t, c in enumerate(cyc)]


def poses() -> dict[str, list[dict]]:
    breathe = lambda t: 1 if t in (2, 3, 6, 7) else 0  # noqa: E731
    side_idle = [dict(view="side", t=t, bob=t % 2) for t in range(4)]
    sip_seq = ["low", "low", "mid", "mouth", "mouth", "mouth", "mid", "low"]
    arm = {"low": "down", "mid": "mid", "mouth": "up"}
    return {
        "idle_down": [dict(t=t, bob=breathe(t), eyes="closed" if t == 5 else "open", frog_blink=t == 1)
                      for t in range(8)],
        "walk_down": _walk("front"), "walk_up": _walk("back"), "walk_right": _walk("side"),
        "walk_left": [dict(p, mirror=True) for p in _walk("side")],
        "idle_up": [dict(view="back", t=t, bob=t % 2) for t in range(4)],
        "idle_right": side_idle, "idle_left": [dict(p, mirror=True) for p in side_idle],
        "type": [dict(t=t, sit=True, hands_forward=True, hand_dy=(0, 1) if t % 2 == 0 else (1, 0),
                      eyes="closed" if t == 6 else "open", frog_blink=t == 3) for t in range(8)],
        "sit_down": [dict(t=t, sit=True, bob=1 if t in (4, 5, 6, 7) else 0, eyes="closed" if t == 2 else "open",
                          frog_blink=t == 6) for t in range(8)],
        "sit_up": [dict(view="back", sit=True, t=t, bob=t // 2) for t in range(4)],
        "sip": [dict(t=t, mug=p, right_arm=arm[p], eyes="closed" if p == "mouth" else "open", frog_blink=t == 4)
                for t, p in enumerate(sip_seq)],
        "talk_down": [dict(t=t, mouth="open" if t % 2 == 0 else "closed", hand_dy=(0, -1) if t == 1 else (0, 0))
                      for t in range(4)],
        "hold_down": [dict(t=t, mug="low", bob=breathe(t), eyes="closed" if t == 5 else "open") for t in range(8)],
    }


def look_key(look: dict) -> str:
    return hashlib.sha1(json.dumps(look, sort_keys=True).encode()).hexdigest()[:12]


def meta() -> dict:
    return {"frame_w": FRAME, "frame_h": FRAME, "columns": COLUMNS, "anchor": list(ANCHOR),
            "animations": {name: {"row": i, "frames": n, "fps": fps} for i, (name, n, fps) in enumerate(ANIMATIONS)}}


@lru_cache(maxsize=64)
def _sheet(look_json: str) -> bytes:
    look = json.loads(look_json)
    all_poses = poses()
    sheet = Image.new("RGBA", (COLUMNS * FRAME, len(ANIMATIONS) * FRAME), (0, 0, 0, 0))
    for row, (name, n, _fps) in enumerate(ANIMATIONS):
        for col, pose in enumerate(all_poses[name][:n]):
            sheet.alpha_composite(render_frame(look, pose), (col * FRAME, row * FRAME))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sheet_png(look: dict) -> bytes:
    """The full animation sheet for a look (cached)."""
    clean, _ = catalog.validate_appearance(look)
    return _sheet(json.dumps(clean, sort_keys=True))


def portrait_png(look: dict, scale: int = 4) -> bytes:
    clean, _ = catalog.validate_appearance(look)
    img = render_frame(clean, {"t": 0}).resize((FRAME * scale, FRAME * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
