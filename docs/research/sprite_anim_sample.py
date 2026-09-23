"""Research sample (not production art): animated, modular pixel-art employees drawn in code.

Run:  pip install pillow && python sprite_anim_sample.py

Produces:
  anim-showcase.gif      4 employees x 6 animations (idle, walk down/right/up, typing, coffee)
  anim-walk-sheet.png    a Godot-ready sprite sheet (rows: down, right, up, left; 4 frames each)

Same idea as sprite_sample.py: every body part is a mask whose digits are shades of a
color ramp (1 = darkest .. 4 = highlight). A "look" is just data (part ids + colors), so an
AI employee can choose it and the engine can validate it. Poses are parameters to the
part functions, so animation = a list of poses.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent
SPR = 32  # sprite canvas (the character fits in 32x32; +1px border for the outline)


# --------------------------------------------------------------------------- masks
class Mask(dict):
    """(x, y) -> shade character."""

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


# ----- heads / faces ---------------------------------------------------------
def head_front(eyes="open"):
    m = Mask()
    for y, (a, b) in {5: (11, 20), 6: (10, 21), 7: (9, 22)}.items():
        m.row(y, a, b, "3")
    m.rect(8, 8, 23, 14, "3").row(15, 9, 22, "3")
    m.row(16, 10, 21, "2").row(16, 12, 19, "3").row(17, 12, 19, "2")
    m.rect(23, 8, 23, 14, "2").dot(22, 15, "2")
    for ex in (11, 19):
        if eyes == "open":
            m.rect(ex, 11, ex + 1, 13, "e").dot(ex + 1, 12, "w")
        else:  # closed / blinking
            m.row(12, ex, ex + 1, "e")
    m.row(14, 9, 10, "b").row(14, 21, 22, "b").row(15, 15, 16, "m")
    return m


def head_back():
    m = Mask()
    for y, (a, b) in {5: (11, 20), 6: (10, 21), 7: (9, 22)}.items():
        m.row(y, a, b, "3")
    m.rect(8, 8, 23, 14, "3").row(15, 9, 22, "3").row(16, 10, 21, "2")
    m.row(17, 12, 19, "2")
    return m


def head_side(eyes="open"):  # facing right
    m = Mask()
    for y, (a, b) in {5: (12, 19), 6: (11, 20), 7: (10, 21)}.items():
        m.row(y, a, b, "3")
    m.rect(9, 8, 22, 14, "3").row(15, 10, 21, "3").row(16, 11, 20, "2")
    m.row(17, 13, 17, "2")
    m.rect(9, 8, 9, 14, "2")                       # back of the head in shade
    m.rect(13, 12, 13, 13, "2")                      # ear (subtle)
    if eyes == "open":
        m.rect(19, 11, 20, 13, "e").dot(20, 12, "w")
    else:
        m.row(12, 19, 20, "e")
    m.row(14, 20, 21, "b").dot(21, 15, "m")
    return m


# ----- torso / arms / hands --------------------------------------------------
def shirt_front(highlight=True, right_arm="down", hands_forward=False):
    m = Mask()
    m.row(18, 9, 22, "3").row(18, 13, 18, "2")
    m.rect(8, 19, 23, 23, "3")
    m.rect(10, 19, 10, 23, "2").rect(21, 19, 21, 23, "2").rect(22, 19, 23, 23, "2")
    m.rect(10, 24, 21, 25, "3").row(25, 10, 21, "2")
    if highlight:
        m.row(19, 11, 13, "4").row(20, 11, 12, "4")
    if right_arm == "mid":
        m.erase(22, 21, 23, 23).rect(21, 18, 22, 20, "2")
    elif right_arm == "up":
        m.erase(22, 20, 23, 23).rect(21, 17, 22, 19, "2")
    if hands_forward:  # forearms reach toward a keyboard
        m.erase(8, 22, 9, 23).erase(22, 22, 23, 23)
        m.rect(10, 22, 11, 23, "2").rect(20, 22, 21, 23, "2")
    return m


def hands_front(left_dy=0, right_dy=0, right="down", forward=False):
    m = Mask()
    if forward:
        m.rect(11, 24 + left_dy, 12, 24 + left_dy, "3")
        m.rect(19, 24 + right_dy, 20, 24 + right_dy, "3")
        return m
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
    m.row(18, 12, 19, "3").rect(11, 19, 20, 24, "3").row(25, 12, 19, "2")
    m.rect(11, 19, 11, 24, "2")                          # back edge in shade
    # near arm (swings): a sleeve strip drawn over the torso
    ax = 14 + arm_dx
    m.rect(ax, 19, ax + 2, 23, "3").rect(ax, 19, ax, 23, "2").rect(ax + 3, 20, ax + 3, 23, "2")
    m.row(19, ax + 1, ax + 2, "4")
    return m


def hands_side(arm_dx=0):
    ax = 14 + arm_dx
    return Mask().rect(ax + 1, 24, ax + 2, 24, "3").rect(ax + 1, 25, ax + 2, 25, "2")


# ----- legs / shoes ----------------------------------------------------------
def legs_front(lift_left=0, lift_right=0):
    pants, shoes = Mask(), Mask()
    pants.row(26, 10, 21, "3").row(26, 15, 16, "2")
    for x0, x1, lift in ((10, 14, lift_left), (17, 21, lift_right)):
        pants.rect(x0, 27, x1, 28 - lift, "3").rect(x1, 27, x1, 28 - lift, "2")
        sy = 29 - lift
        shoes.row(sy, x0 - 1, x1, "3").row(sy + 1, x0 - 1, x1, "2").row(sy, x0, x0 + 1, "4")
    return pants, shoes


def legs_side(front_c=16, back_c=15):
    """Stride pose: foot centers for the near (front_c) and far (back_c) leg."""
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


# ----- hair styles (front / back / side) --------------------------------------
def hair(style, view):
    m = Mask()
    if view == "side":
        return hair_side(style)
    top_rows = {2: (12, 19), 3: (10, 21), 4: (9, 22)}
    if style in ("short", "bun"):
        top_rows[2] = None
    for y, span in top_rows.items():
        if span:
            m.row(y, *span, "3")
    if style in ("short", "bun"):
        m.row(2, 11, 13, "3").row(2, 17, 19, "3")
        m.rect(8, 5, 23, 8, "3").rect(22, 5, 23, 8, "2")
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
        # strands + sheen so the back of the head isn't a flat block
        bottom = {"long": 22, "bob": 15}.get(style, 13)
        start = 10 if style in ("bob", "long") else 9
        for x in (11, 16, 20):
            m.rect(x, start, x, bottom, "2")
        m.row(6, 10, 13, "4").row(7, 9, 11, "4").row(6, 18, 20, "2")
    else:  # front: bangs + sides
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
    m.row(2, 11, 18, "3").row(3, 9, 20, "3").row(4, 8, 21, "3")
    m.rect(8, 5, 22, 8, "3")
    m.row(3, 12, 16, "4").row(4, 14, 18, "4")
    if style in ("short", "bun"):
        m.row(9, 8, 12, "3").row(9, 18, 22, "3").rect(8, 10, 11, 11, "2")
        m.dot(22, 10, "2")
    else:
        m.row(9, 8, 22, "3").row(10, 8, 13, "3").row(10, 18, 22, "3").dot(20, 10, "2")
        m.rect(8, 11, 12, 15, "3").rect(8, 11, 8, 15, "2").row(16, 8, 12, "2")
        if style == "long":
            m.rect(7, 16, 12, 22, "3").rect(7, 16, 7, 22, "2").row(23, 7, 12, "2")
    if style == "bun":
        m.row(0, 9, 12, "3").row(1, 8, 12, "3").row(1, 9, 10, "4").row(2, 8, 11, "2")
    return m


# ----- accessories ------------------------------------------------------------
def glasses(view):
    m = Mask()
    if view == "front":
        m.row(11, 14, 17, "k")
        for x in (10, 13, 18, 21):
            m.rect(x, 11, x, 13, "k")
        m.row(14, 11, 12, "k").row(14, 19, 20, "k")
    elif view == "side":
        m.rect(18, 11, 18, 13, "k").rect(21, 11, 21, 13, "k").row(14, 19, 20, "k").row(11, 15, 17, "k")
    return m


def frog_hat(view, frog_blink=False):
    m = Mask()
    if view == "side":
        m.row(3, 9, 21, "3").rect(8, 4, 22, 5, "3").row(6, 8, 22, "2").row(4, 10, 12, "4")
        m.row(0, 15, 18, "3").rect(15, 1, 18, 2, "3")
        if frog_blink:
            m.row(2, 16, 17, "M")
        else:
            m.rect(16, 1, 17, 2, "W").dot(17, 2, "K")
        m.row(1, 10, 12, "2").row(2, 10, 13, "3")     # far eye bump, just green
        m.row(5, 19, 22, "M")
        return m
    m.row(0, 9, 12, "3").row(0, 19, 22, "3")
    for x0 in (9, 19):
        m.rect(x0, 1, x0 + 3, 2, "3")
        if view == "front":
            if frog_blink:
                m.row(2, x0 + 1, x0 + 2, "M")
            else:
                m.rect(x0 + 1, 1, x0 + 2, 2, "W")
    if view == "front" and not frog_blink:
        m.dot(11, 2, "K").dot(20, 2, "K")
    m.row(3, 8, 23, "3").row(3, 10, 12, "4").rect(7, 4, 24, 5, "3").row(4, 9, 11, "4")
    m.rect(23, 4, 24, 5, "2").row(6, 7, 24, "2")
    if view == "front":
        m.row(5, 12, 19, "M")
    return m


# ----- props --------------------------------------------------------------------
def mug(pos):
    """A cream mug with coffee; pos = 'low' | 'mid' | 'mouth'."""
    x, y = {"low": (23, 21), "mid": (22, 17), "mouth": (16, 14)}[pos]
    m = Mask()
    m.rect(x, y, x + 3, y + 3, "3").rect(x + 3, y, x + 3, y + 3, "2").row(y + 3, x, x + 3, "2")
    if pos != "mouth":
        m.row(y, x, x + 3, "c")       # coffee surface visible from above
    m.dot(x - 1, y + 1, "2").dot(x - 1, y + 2, "2")  # handle
    return m


def steam(pos, t):
    if pos != "low":
        return Mask()
    x = 24 + (t % 2)
    return Mask().dot(x, 19 - (t % 2), "s").dot(x + 1 - 2 * (t % 2), 17, "s")


# --------------------------------------------------------------------------- color
def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp(base):
    r, g, b = [c / 255 for c in hexrgb(base)]
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    purple = (0.75 - hh) if abs(0.75 - hh) < 0.5 else 0

    def rel(mult, floor, dh, ds):
        return tuple(round(c * 255) for c in colorsys.hls_to_rgb((hh + dh) % 1, max(ll * mult, floor), min(1, ss + ds)))

    hi_l = min(1, ll + max(0.10, (1 - ll) * 0.22))
    return {
        "1": rel(0.52, 0.07, purple * 0.12, 0.05),
        "2": rel(0.76, 0.11, purple * 0.06, 0.03),
        "3": hexrgb(base),
        "4": tuple(round(c * 255) for c in colorsys.hls_to_rgb(hh, hi_l, max(0, ss - 0.05))),
    }


FIXED = {
    "e": "#2b2135", "w": "#ffffff", "b": "#f29a9a", "m": "#8c3b4a", "k": "#a53a3a",
    "W": "#ffffff", "K": "#1c1624", "c": "#6b4226", "s": "#e9e4f0",
}
FIXED = {k: hexrgb(v) for k, v in FIXED.items()}
OUTLINE = hexrgb("#2a1f33")


# --------------------------------------------------------------------------- poses
def build_layers(look, pose):
    """Return an ordered list of (mask, colors, name) for one pose."""
    view = pose.get("view", "front")
    bob = pose.get("bob", 0)
    eyes = pose.get("eyes", "open")
    skin, top = ramp(look["skin"]), ramp(look["top"])
    layers = []

    if view == "side":
        head = head_side(eyes)
        pants, shoes = legs_side(*pose.get("stride", (16, 15)))
        shirt, hands = shirt_side(pose.get("arm_dx", 0)), hands_side(pose.get("arm_dx", 0))
    else:
        head = head_front(eyes) if view == "front" else head_back()
        pants, shoes = legs_front(*pose.get("lift", (0, 0)))
        right = pose.get("right_arm", "down")
        fwd = pose.get("hands_forward", False)
        shirt = shirt_front(highlight=(view == "front"), right_arm=right, hands_forward=fwd)
        hl, hr = pose.get("hand_dy", (0, 0))
        hands = hands_front(hl, hr, right, fwd)

    upper = lambda m: m.shifted(0, bob)
    layers += [(upper(head), skin, "skin"), (pants, ramp(look["pants"]), "pants"),
               (shoes, ramp(look["shoes"]), "shoes"), (upper(shirt), top, "top"),
               (upper(hands), skin, "hands"), (upper(hair(look["hair_style"], view)), ramp(look["hair"]), "hair")]
    for acc in look.get("accessories", []):
        if acc == "glasses":
            layers.append((upper(glasses(view)), {}, "glasses"))
        elif acc == "frog_hat":
            hat = ramp(look.get("hat", "#6cc24a"))
            hat["M"] = hat["1"]
            layers.append((upper(frog_hat(view, pose.get("frog_blink", False))), hat, "frog_hat"))
    if "mug" in pose:
        cream = ramp("#f4ead8")
        layers.append((upper(mug(pose["mug"])), cream, "mug"))
        layers.append((upper(steam(pose["mug"], pose.get("t", 0))), {}, "steam"))
    if pose.get("mirror"):
        layers = [(Mask({(SPR - 1 - x, y): c for (x, y), c in m.items()}), col, n) for m, col, n in layers]
    return layers


def render(look, pose):
    layers = build_layers(look, pose)
    px, owner = {}, {}
    for mask, colors, name in layers:
        for xy, ch in mask.items():
            px[xy] = colors.get(ch) or FIXED[ch]
            owner[xy] = name
    # soft internal edges: hair/top/hat pixels resting on another layer get their shadow tone
    lookup = {n: c for _, c, n in layers}
    for (x, y), name in list(owner.items()):
        if name in ("hair", "top", "frog_hat") and owner.get((x, y + 1)) not in (None, name, "glasses", "steam"):
            if px[(x, y)] in (lookup[name]["3"], lookup[name]["4"]):
                px[(x, y)] = lookup[name]["2"]
    img = Image.new("RGBA", (SPR + 2, SPR + 2), (0, 0, 0, 0))
    for (x, y), c in px.items():
        if 0 <= x < SPR and 0 <= y < SPR and owner[(x, y)] != "steam":
            img.putpixel((x + 1, y + 1), c + (255,))
    out = img.copy()
    for y in range(SPR + 2):
        for x in range(SPR + 2):
            if not img.getpixel((x, y))[3] and any(
                    0 <= x + dx < SPR + 2 and 0 <= y + dy < SPR + 2 and img.getpixel((x + dx, y + dy))[3]
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out.putpixel((x, y), OUTLINE + (255,))
    for (x, y), name in owner.items():  # steam floats without an outline
        if name == "steam" and 0 <= x < SPR and 0 <= y < SPR and not out.getpixel((x + 1, y + 1))[3]:
            out.putpixel((x + 1, y + 1), FIXED["s"] + (255,))
    return out


# --------------------------------------------------------------------------- animations (8-frame loops)
def anim_idle(look):
    frames = []
    for t in range(8):
        frames.append(dict(t=t, bob=1 if t in (2, 3, 6, 7) else 0, eyes="closed" if t == 5 else "open",
                           frog_blink=(t == 1)))
    return frames


def anim_walk(view):
    if view == "side":
        cycle = [dict(stride=(19, 12), arm_dx=-2, bob=1), dict(stride=(16, 15), arm_dx=0),
                 dict(stride=(12, 19), arm_dx=2, bob=1), dict(stride=(15, 16), arm_dx=0)]
    else:
        cycle = [dict(lift=(1, 0), hand_dy=(0, -1), bob=1), dict(lift=(0, 0)),
                 dict(lift=(0, 1), hand_dy=(-1, 0), bob=1), dict(lift=(0, 0))]
    return [dict(view=view, t=t, frog_blink=(t == 6), **cycle[t % 4]) for t in range(8)]


def anim_type():
    return [dict(t=t, hands_forward=True, hand_dy=(0, 1) if t % 2 == 0 else (1, 0),
                 eyes="closed" if t == 6 else "open", frog_blink=(t == 3)) for t in range(8)]


def anim_coffee():
    seq = ["low", "low", "mid", "mouth", "mouth", "mouth", "mid", "low"]
    arm = {"low": "down", "mid": "mid", "mouth": "up"}
    return [dict(t=t, mug=p, right_arm=arm[p], eyes="closed" if p == "mouth" else "open",
                 frog_blink=(t == 4)) for t, p in enumerate(seq)]


# --------------------------------------------------------------------------- scenes
def desk_scene(char_img):
    """Character sitting behind a desk with a monitor (we see its back) and a keyboard."""
    W, H = 48, 44
    s = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(s)
    s.alpha_composite(char_img.crop((0, 0, SPR + 2, 27)), (3, 5))    # upper body only
    wood, wood_d, wood_l = hexrgb("#b07a4f"), hexrgb("#7d5236"), hexrgb("#cf9a6a")
    d.rectangle((1, 30, 46, 31), fill=wood_l)
    d.rectangle((1, 32, 46, 41), fill=wood)
    d.rectangle((1, 41, 46, 42), fill=wood_d)
    d.rectangle((4, 34, 20, 39), fill=wood_d)                          # drawer
    d.rectangle((11, 36, 13, 36), fill=wood_l)
    d.rectangle((10, 29, 26, 30), fill=hexrgb("#4a4458"))              # keyboard
    d.rectangle((11, 29, 25, 29), fill=hexrgb("#6e6880"))
    d.rectangle((30, 14, 43, 26), fill=hexrgb("#5b566b"))              # monitor back
    d.rectangle((31, 15, 42, 25), fill=hexrgb("#6e6880"))
    d.rectangle((35, 27, 38, 29), fill=hexrgb("#4a4458"))
    d.rectangle((33, 29, 40, 29), fill=hexrgb("#4a4458"))
    d.point((44, 28), fill=hexrgb("#6cc24a"))                           # tiny plant
    d.rectangle((43, 29, 45, 29), fill=hexrgb("#4f9a3a"))
    return s


# --------------------------------------------------------------------------- looks (AI-chosen data)
LOOKS = [
    dict(name="bob + glasses", skin="#f3c49b", hair="#6b3f2a", hair_style="bob", top="#5aa36a",
         pants="#3f5f9e", shoes="#3a2e3f", accessories=["glasses"]),
    dict(name="frog hat", skin="#c98c5f", hair="#f06fa4", hair_style="short", top="#f2c14e",
         pants="#4a4a6a", shoes="#e8e4dc", accessories=["frog_hat"], hat="#6cc24a"),
    dict(name="long hair", skin="#8d5a3b", hair="#241c2c", hair_style="long", top="#d9534f",
         pants="#2f3440", shoes="#7a4b2a"),
    dict(name="bun", skin="#ffe0c7", hair="#e8b64c", hair_style="bun", top="#8e6cc9",
         pants="#6e5a4a", shoes="#3a2e3f", accessories=["glasses"]),
]

COLUMNS = [
    ("idle", lambda lk: anim_idle(lk)),
    ("walk down", lambda lk: anim_walk("front")),
    ("walk right", lambda lk: anim_walk("side")),
    ("walk up", lambda lk: anim_walk("back")),
    ("typing", lambda lk: anim_type()),
    ("coffee", lambda lk: anim_coffee()),
]


def main():
    scale, cell_w, cell_h, top = 4, 50, 46, 14
    bg, ink = (246, 239, 228, 255), (90, 70, 80, 255)
    frames = []
    for t in range(8):
        sheet = Image.new("RGBA", (len(COLUMNS) * cell_w, top + len(LOOKS) * cell_h), bg)
        for r, look in enumerate(LOOKS):
            for c, (label, anim) in enumerate(COLUMNS):
                spr = render(look, anim(look)[t])
                if label == "typing":
                    spr = desk_scene(spr)
                    sheet.alpha_composite(spr, (c * cell_w + 1, top + r * cell_h))
                else:
                    sheet.alpha_composite(spr, (c * cell_w + 8, top + r * cell_h + 8))
        big = sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST)
        d = ImageDraw.Draw(big)
        for c, (label, _) in enumerate(COLUMNS):
            d.text((c * cell_w * scale + 12, 16), label, fill=ink)
        frames.append(big.convert("RGB"))
    frames[0].save(OUT / "anim-showcase.gif", save_all=True, append_images=frames[1:], duration=140, loop=0,
                   disposal=1, optimize=False)

    # Godot-ready walk sheet for one employee: rows = down, right, up, left; 4 frames each
    look = LOOKS[1]
    rows = [anim_walk("front"), anim_walk("side"), anim_walk("back"),
            [dict(p, mirror=True) for p in anim_walk("side")]]
    sheet = Image.new("RGBA", (4 * (SPR + 2), 4 * (SPR + 2)), (0, 0, 0, 0))
    for r, poses in enumerate(rows):
        for f in range(4):
            sheet.alpha_composite(render(look, poses[f]), (f * (SPR + 2), r * (SPR + 2)))
    sheet.save(OUT / "anim-walk-sheet.png")
    print("wrote anim-showcase.gif and anim-walk-sheet.png")


if __name__ == "__main__":
    main()
