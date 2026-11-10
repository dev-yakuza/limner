"""Templates (spec §7.6): layout JSON with `{{key}}` placeholders, a small schema checker."""

from __future__ import annotations

import os
import re
from importlib import resources

from .. import util
from ..util import bad

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")
WHOLE = re.compile(r"^\{\{\s*([A-Za-z0-9_.]+)\s*\}\}$")
BUILTIN = ("og-title", "card-cover", "card-body", "card-end", "strip-2x2", "strip-3x2")


def find(name: str, root, templates_dir: str | None) -> tuple[dict, dict, str]:
    """(template, schema, where) — a --templates dir first, then the built-in ones."""
    if not re.match(r"^[A-Za-z0-9_-]+$", name or ""):
        raise bad("--template: a name of letters, digits, - and _")
    if templates_dir:
        d = root.inside(templates_dir, "--templates")
        p = os.path.join(d, name + ".json")
        if os.path.isfile(p):
            s = os.path.join(d, name + ".schema.json")
            return _load(p), (_load(s) if os.path.isfile(s) else {}), p
    pkg = resources.files("limner").joinpath("templates")
    if pkg.joinpath(name + ".json").is_file():
        import json

        t = json.loads(pkg.joinpath(name + ".json").read_text())
        sc = pkg.joinpath(name + ".schema.json")
        return t, (json.loads(sc.read_text()) if sc.is_file() else {}), "builtin:" + name
    raise bad("no template %r (built in: %s)" % (name, ", ".join(BUILTIN)))


def _load(p):
    try:
        return util.load_json(p)
    except (OSError, ValueError) as e:
        raise bad("%s: %s" % (p, e))


# --------------------------------------------------------------------------- schema


TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _type_ok(v, t) -> bool:
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, TYPES.get(t, object))


def validate(value, schema: dict, where: str = "data") -> None:
    """type, required, properties, items, maxLength, minLength, maxItems, enum,
    additionalProperties: false — nothing else (no jsonschema dependency)."""
    if not isinstance(schema, dict):
        return
    t = schema.get("type")
    if t:
        ts = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in ts):
            raise bad("%s: expected %s" % (where, " or ".join(ts)))
    if "enum" in schema and value not in schema["enum"]:
        raise bad("%s: must be one of %s" % (where, schema["enum"]))
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise bad("%s: longer than %d" % (where, schema["maxLength"]))
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise bad("%s: shorter than %d" % (where, schema["minLength"]))
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise bad("%s: more than %d items" % (where, schema["maxItems"]))
        if "items" in schema:
            for i, v in enumerate(value):
                validate(v, schema["items"], "%s[%d]" % (where, i))
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                raise bad("%s: %s is required" % (where, k))
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                raise bad("%s: unknown keys %s" % (where, extra))
        for k, v in value.items():
            if k in props:
                validate(v, props[k], "%s.%s" % (where, k))


# --------------------------------------------------------------------------- substitution


_MISSING = object()


def lookup(data, key: str):
    cur = data
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


def _sub(v, data):
    if isinstance(v, str):
        m = WHOLE.match(v)
        if m:
            got = lookup(data, m.group(1))
            if got is _MISSING:
                raise bad("template: {{%s}} is not in the data" % m.group(1))
            return got  # the data's own type

        def rep(m2):
            got = lookup(data, m2.group(1))
            if got is _MISSING:
                raise bad("template: {{%s}} is not in the data" % m2.group(1))
            if isinstance(got, (dict, list)) or got is None:
                raise bad("template: {{%s}} inside a string must be a string or number" % m2.group(1))
            return str(got)

        return PLACEHOLDER.sub(rep, v)
    if isinstance(v, list):
        return [_sub(x, data) for x in v]
    if isinstance(v, dict):
        return {k: _sub(x, data) for k, x in v.items()}
    return v


def _present(cond, data) -> bool:
    if isinstance(cond, str):
        m = WHOLE.match(cond)
        if m:
            got = lookup(data, m.group(1))
            return not (got is _MISSING or got is None or got == "")
        return cond != ""
    return bool(cond)


def apply(template: dict, data: dict) -> dict:
    """Substitute inside the parsed JSON's string values only; drop layers whose `when` is
    missing / null / empty."""
    out = {k: v for k, v in template.items() if k != "layers"}
    out = _sub(out, data)
    layers = []
    for layer in template.get("layers", []):
        if "when" in layer:
            if not _present(layer["when"], data):
                continue
            layer = {k: v for k, v in layer.items() if k != "when"}
        layers.append(_sub(layer, data))
    out["layers"] = layers
    return out
