"""Structured-output schemas and a small, forgiving validator.

Models from different families follow JSON schemas with different fidelity, so every response is
validated locally: strings are truncated to their limits, numbers clamped, unknown keys dropped.
Only genuinely unusable output (wrong types, missing fields, bad enum values) fails validation.
"""
from __future__ import annotations

import json
import re

from .. import catalog

VERBS = {
    "go_to": "walk somewhere. target = a place, object or person (e.g. 'break room', 'coffee machine', 'June')",
    "work": "work at your desk for `minutes` (general work, reading, tidying up)",
    "claim_question": "take ownership of a question from the queue. target = question id (e.g. 'Q3'). Needs a desk computer.",
    "work_on_question": "research/write an answer at your desk for `minutes`. target = question id",
    "submit_answer": "send your latest draft answer to the person who asked. target = question id",
    "say": "say something short out loud. target = a person here, 'everyone', or 'owner' (the intercom). text = what you say",
    "use": "use an object for `minutes` (coffee machine, couch, printer, meeting table, fridge...). target = object",
    "take_break": "take a break for `minutes` (you'll go somewhere comfy)",
    "wait": "stay where you are and do nothing in particular for `minutes` (think, look around, be quiet)",
    "write_note": "write something. target = a surface (your desk, someone's desk, meeting table, counter), "
                  "'whiteboard', or a file path like /shared/tips.txt or /private/<you>/diary.txt. text = content",
    "read": "read something. target = a note id, 'whiteboard', or a file path (/shared/... or a private folder)",
    "give": "give an item you're holding to someone. item = item id, target = person",
    "put_down": "put down what you're holding (on the nearest surface). item = item id (optional)",
    "pick_up": "pick up an item nearby. item = item id",
    "change_appearance": "change how you look. text = 'field: value' (e.g. 'top_color: blue', 'accessories: glasses')",
    "leave_office": "go home for the day",
}


def action_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "maxLength": 400},
            "action": {"type": "string", "enum": list(VERBS)},
            "target": {"type": ["string", "null"], "maxLength": 160},
            "item": {"type": ["string", "null"], "maxLength": 40},
            "text": {"type": ["string", "null"], "maxLength": 900},
            "minutes": {"type": ["integer", "null"], "minimum": 1, "maximum": 120},
        },
        "required": ["thought", "action", "target", "item", "text", "minutes"],
        "additionalProperties": False,
    }


def onboard_schema(free_desks: list[str]) -> dict:
    colors = list(catalog.COLORS)
    s = lambda **kw: dict(type="string", **kw)  # noqa: E731
    return {
        "type": "object",
        "properties": {
            "name": s(maxLength=40),
            "pronouns": s(maxLength=30),
            "appearance": {
                "type": "object",
                "properties": {
                    "skin": s(enum=list(catalog.SKIN)),
                    "hair_style": s(enum=list(catalog.HAIR_STYLES)),
                    "hair_color": s(enum=list(catalog.HAIR_COLORS)),
                    "top_color": s(enum=colors),
                    "pants_color": s(enum=colors),
                    "shoes_color": s(enum=colors),
                    "accessories": {"type": "array", "items": s(enum=list(catalog.ACCESSORIES))},
                    "accessory_color": s(enum=colors),
                },
                "required": ["skin", "hair_style", "hair_color", "top_color", "pants_color", "shoes_color",
                             "accessories", "accessory_color"],
                "additionalProperties": False,
            },
            "desk": s(enum=free_desks),
            "personal_item": s(enum=list(catalog.STARTER_ITEMS)),
            "introduction": s(maxLength=240),
            "why": s(maxLength=700),
        },
        "required": ["name", "pronouns", "appearance", "desk", "personal_item", "introduction", "why"],
        "additionalProperties": False,
    }


def reflect_schema() -> dict:
    s = lambda **kw: dict(type="string", **kw)  # noqa: E731
    obj = lambda props, req=None: {"type": "object", "properties": props,  # noqa: E731
                                   "required": req or list(props), "additionalProperties": False}
    return obj({
        "journal": s(maxLength=2000),
        "memories": {"type": "array", "maxItems": 8, "items": obj({
            "text": s(maxLength=400), "importance": {"type": "integer", "minimum": 1, "maximum": 10},
            "people": {"type": "array", "items": s(maxLength=40)}})},
        "beliefs": {"type": "array", "maxItems": 8, "items": obj({
            "about": s(maxLength=40), "statement": s(maxLength=300),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "source": s(enum=["witnessed", "told", "overheard", "read", "owner", "inferred"]),
            "source_person": {"type": ["string", "null"], "maxLength": 40},
            "replaces": {"type": ["string", "null"], "maxLength": 300}})},
        "relationships": {"type": "array", "maxItems": 8, "items": obj({
            "person": s(maxLength=40), "closer": {"type": "integer", "minimum": -2, "maximum": 2},
            "trust_change": {"type": "integer", "minimum": -2, "maximum": 2}, "note": s(maxLength=300)})},
        "self_concept": {"type": ["string", "null"], "maxLength": 900},
        "goals": {"type": "array", "maxItems": 6, "items": obj({
            "text": s(maxLength=200), "status": s(enum=["new", "keep", "done", "drop"])})},
        "tomorrow": s(maxLength=400),
    })


UNSUPPORTED_WIRE_KEYS = {"maxLength", "minLength", "minimum", "maximum", "maxItems", "minItems", "pattern", "format"}


def wire_schema(schema):
    """The schema as sent to providers: limits are enforced locally, since strict modes vary."""
    if isinstance(schema, dict):
        return {k: wire_schema(v) for k, v in schema.items() if k not in UNSUPPORTED_WIRE_KEYS}
    if isinstance(schema, list):
        return [wire_schema(v) for v in schema]
    return schema


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating code fences and chatter around the object."""
    if text is None:
        raise ValueError("empty reply")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise


def validate(value, schema: dict, path: str = "$"):
    """Returns (clean_value, errors)."""
    errors: list[str] = []
    types = schema.get("type")
    types = types if isinstance(types, list) else [types] if types else []

    if value is None:
        if "null" in types:
            return None, errors
        if "string" in types:
            errors.append(f"{path} is required")
            return "", errors
        errors.append(f"{path} is required")
        return None, errors

    if "object" in types:
        if not isinstance(value, dict):
            return None, [f"{path} should be an object"]
        out = {}
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key not in value:
                if "null" in (sub.get("type") if isinstance(sub.get("type"), list) else [sub.get("type")]):
                    out[key] = None
                    continue
                if sub.get("type") == "array":
                    out[key] = []
                    continue
                if key in schema.get("required", []):
                    errors.append(f"{path}.{key} is missing")
                continue
            out[key], errs = validate(value[key], sub, f"{path}.{key}")
            errors += errs
        return out, errors

    if "array" in types:
        if not isinstance(value, list):
            value = [value]
        items = schema.get("items", {})
        out = []
        for i, v in enumerate(value[: schema.get("maxItems", len(value))]):
            cv, errs = validate(v, items, f"{path}[{i}]")
            if errs:
                errors += errs
            else:
                out.append(cv)
        return out, errors

    if "integer" in types or "number" in types:
        try:
            num = float(value)
        except (TypeError, ValueError):
            if "string" in types:
                pass
            else:
                return None, [f"{path} should be a number"]
        else:
            if "minimum" in schema:
                num = max(num, schema["minimum"])
            if "maximum" in schema:
                num = min(num, schema["maximum"])
            return (int(round(num)) if "integer" in types else num), errors

    if "boolean" in types:
        if isinstance(value, bool):
            return value, errors
        return str(value).lower() in ("true", "yes", "1"), errors

    if "string" in types:
        s = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        s = s.strip()
        if "enum" in schema:
            options = schema["enum"]
            match = next((o for o in options if o.lower() == s.lower().replace(" ", "_")
                          or o.lower() == s.lower()), None)
            if match is None:
                return None, [f"{path} must be one of {options}, got {s!r}"]
            s = match
        if "maxLength" in schema and len(s) > schema["maxLength"]:
            s = s[: schema["maxLength"]].rstrip()
        return s, errors

    return value, errors
