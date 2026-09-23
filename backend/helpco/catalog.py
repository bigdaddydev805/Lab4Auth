"""What exists in the world to choose from: appearance parts, colors, and starter items.

Employees pick their look from this catalog (the engine validates the choice), so everything an
employee can wear is something the art pipeline can actually draw.
"""
from __future__ import annotations

SKIN = {
    "ivory": "#ffe0c7", "peach": "#f3c49b", "sand": "#e8b88f", "golden": "#d9a066", "tan": "#c98c5f",
    "caramel": "#a86f47", "brown": "#8d5a3b", "umber": "#6e4430", "deep": "#4f3024",
}

COLORS = {
    "red": "#d9534f", "coral": "#f07a63", "orange": "#e8893a", "mustard": "#d9a531", "yellow": "#f2c14e",
    "lime": "#9bcf53", "green": "#5aa36a", "teal": "#3a9e97", "sky": "#6cb4e4", "blue": "#3f6fc4",
    "navy": "#2f3f6e", "purple": "#8e6cc9", "lavender": "#b9a3e3", "pink": "#f06fa4", "brown": "#7a4b2a",
    "tan": "#c8a27a", "cream": "#efe6d2", "white": "#f4f1ec", "gray": "#8a8793", "charcoal": "#4a4a5a",
    "black": "#2b2833",
}

HAIR_COLORS = {
    "black": "#241c2c", "dark brown": "#4a2f22", "brown": "#6b3f2a", "auburn": "#8e3b24",
    "ginger": "#c9652e", "blonde": "#e8b64c", "platinum": "#efe3b8", "gray": "#9a98a3", "white": "#eeeef2",
    "pink": "#f06fa4", "blue": "#4f7fd9", "green": "#4fae6a", "purple": "#8e5cc9",
}

HAIR_STYLES = {
    "bob": "chin-length bob with bangs",
    "short": "short and tidy",
    "long": "long, past the shoulders",
    "bun": "tied up in a bun",
    "buzz": "buzzed very short",
}

TOPS = {"sweater": "a sweater"}

ACCESSORIES = {
    "glasses": "round glasses (frame color from the palette)",
    "frog_hat": "a green frog hat with eyes on top",
}

STARTER_ITEMS = {
    "mug": "a mug in your favorite color",
    "succulent": "a tiny succulent in a pot",
    "photo_frame": "a framed photo",
    "rubber_duck": "a rubber duck",
    "books": "a small stack of books",
    "snow_globe": "a snow globe",
}

# Short labels used for items once they exist in the world.
ITEM_LABELS = {"mug": "a mug", "succulent": "a tiny succulent", "photo_frame": "a framed photo",
               "rubber_duck": "a rubber duck", "books": "a stack of books", "snow_globe": "a snow globe",
               "coffee": "a mug of coffee", "note": "a sticky note", "paper": "a printed page"}

# Items that can exist in the office (starter items plus things made in the office).
ITEM_KINDS = dict(STARTER_ITEMS, coffee="a mug of coffee", note="a sticky note", paper="a printed page")

APPEARANCE_FIELDS = ("skin", "hair_style", "hair_color", "top", "top_color", "pants_color", "shoes_color",
                     "accessories", "accessory_color")

SYNONYMS = {"grey": "gray", "dark-brown": "dark brown", "darkbrown": "dark brown", "blond": "blonde",
            "none": "", "no": ""}


def _norm(v) -> str:
    s = str(v or "").strip().lower().replace("_", " ")
    return SYNONYMS.get(s, s)


def _pick(value, options: dict, label: str, errors: list[str], default: str) -> str:
    v = _norm(value)
    keys = {k.replace("_", " "): k for k in options}
    if v in keys:
        return keys[v]
    errors.append(f"{label} '{value}' isn't available (options: {', '.join(options)})")
    return default


def validate_appearance(raw: dict | None) -> tuple[dict, list[str]]:
    """Clamp a requested look to what exists. Returns (clean look, list of problems)."""
    raw = raw or {}
    errors: list[str] = []
    look = {
        "skin": _pick(raw.get("skin"), SKIN, "skin", errors, "sand"),
        "hair_style": _pick(raw.get("hair_style"), HAIR_STYLES, "hair_style", errors, "short"),
        "hair_color": _pick(raw.get("hair_color"), HAIR_COLORS, "hair_color", errors, "brown"),
        "top": _pick(raw.get("top", "sweater"), TOPS, "top", errors, "sweater"),
        "top_color": _pick(raw.get("top_color"), COLORS, "top_color", errors, "green"),
        "pants_color": _pick(raw.get("pants_color"), COLORS, "pants_color", errors, "navy"),
        "shoes_color": _pick(raw.get("shoes_color"), COLORS, "shoes_color", errors, "charcoal"),
        "accessory_color": _pick(raw.get("accessory_color", "red"), COLORS, "accessory_color", errors, "red"),
    }
    acc = raw.get("accessories") or []
    if isinstance(acc, str):
        acc = [a for a in acc.replace(",", " ").split() if a]
    clean_acc = []
    for a in acc:
        k = _norm(a).replace(" ", "_")
        if k in ACCESSORIES and k not in clean_acc:
            clean_acc.append(k)
        elif k:
            errors.append(f"accessory '{a}' isn't available (options: {', '.join(ACCESSORIES)})")
    look["accessories"] = clean_acc
    return look, errors


def describe_look(look: dict) -> str:
    parts = [f"{look['hair_color']} {HAIR_STYLES[look['hair_style']].split(',')[0]} hair",
             f"a {look['top_color']} {look['top']}", f"{look['pants_color']} pants"]
    if "glasses" in look.get("accessories", []):
        parts.append(f"{look.get('accessory_color', 'red')} glasses")
    if "frog_hat" in look.get("accessories", []):
        parts.append("a frog hat")
    return ", ".join(parts)


def catalog_text() -> str:
    """The catalog as shown to an employee choosing their look."""
    return "\n".join([
        f"- skin: {', '.join(SKIN)}",
        f"- hair_style: " + "; ".join(f"{k} ({v})" for k, v in HAIR_STYLES.items()),
        f"- hair_color: {', '.join(HAIR_COLORS)}",
        f"- top: sweater (more clothes may exist someday)",
        f"- top_color / pants_color / shoes_color / accessory_color: {', '.join(COLORS)}",
        f"- accessories (any number, or none): " + "; ".join(f"{k} ({v})" for k, v in ACCESSORIES.items()),
    ])


def palette_for(look: dict) -> dict:
    """Hex colors for the art pipeline."""
    return {
        "skin": SKIN[look["skin"]], "hair": HAIR_COLORS[look["hair_color"]], "top": COLORS[look["top_color"]],
        "pants": COLORS[look["pants_color"]], "shoes": COLORS[look["shoes_color"]],
        "accessory": COLORS[look.get("accessory_color", "red")], "hat": "#6cc24a",
    }
