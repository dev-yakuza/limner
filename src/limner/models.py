"""The model store shared with script2video (spec §3).

Folder rule and marker rule follow script2video `imagegen/models.py` (MIT, same author):
`<models dir>/<name>__<revision[:12]>`, complete only with a marker file. limner reads both
`.complete` and `.s2v-complete` (partial match: other keys such as `note` are ignored) and always
writes both with the same content, so s2v never takes a limner-made model for a broken download.
"""

from __future__ import annotations

import json
import os
from importlib import resources

from . import util

MARKERS = (".complete", ".s2v-complete")
PARTIAL = ".limner-partial"


def load_lock() -> dict:
    return json.loads(resources.files("limner").joinpath("limner.lock.json").read_text())


def candidates() -> list[tuple[str, str]]:
    """(source, absolute dir) in the order of spec §3 (relative env values are ignored)."""
    home = os.path.expanduser("~")
    out = []
    for env in ("LIMNER_MODELS_DIR", "S2V_MODELS_DIR"):
        v = os.environ.get(env)
        if v and os.path.isabs(v):
            out.append((env, v))
    out.append(("default", os.path.join(home, ".cache", "models")))
    out.append(("s2v-default", os.path.join(home, ".cache", "script2video", "models")))
    return out


def models_dir() -> tuple[str, str]:
    """The first candidate that exists; for new downloads the first one (else ~/.cache/models)."""
    c = candidates()
    for src, d in c:
        if os.path.isdir(d):
            return src, d
    return c[0]  # an env value if one is set, else ~/.cache/models


def folder(lock: dict, name: str) -> str:
    _, d = models_dir()
    if name == lock["quantized"]["name"]:
        rev = lock["models"][lock["quantized"]["from"]]["revision"]
    else:
        rev = lock["models"][name]["revision"]
    return os.path.join(d, "%s__%s" % (name, rev[:12]))


def expected_marker(lock: dict, name: str) -> dict:
    if name == lock["quantized"]["name"]:
        q = lock["quantized"]
        return {"from": q["from"], "revision": lock["models"][q["from"]]["revision"], "bits": q["bits"]}
    e = lock["models"][name]
    return {"repo": e["repo"], "revision": e["revision"]}


def read_markers(path: str) -> list[dict]:
    out = []
    for m in MARKERS:
        d = util.read_json_or_none(os.path.join(path, m))
        if isinstance(d, dict):
            out.append(d)
        elif os.path.exists(os.path.join(path, m)):
            out.append({})
    return out


def partial_match(marker: dict, expected: dict) -> str | None:
    """None when every expected key matches, else the first key that does not."""
    for k, v in expected.items():
        if marker.get(k) != v:
            return k
    return None


def model_names(lock: dict) -> list[str]:
    q = lock["quantized"]["name"]
    return [q] + [n for n in lock["models"] if n != q]


def required(lock: dict, name: str) -> bool:
    if name == lock["quantized"]["name"]:
        return bool(lock["quantized"].get("required", True))
    return bool(lock["models"][name].get("required", True))


def status(lock: dict, name: str) -> dict:
    """{"name", "path", "required", "state": ready|missing|mismatch:<key>|foreign_partial|limner_partial,
    "marker"}."""
    path = folder(lock, name)
    st = {"name": name, "path": path, "required": required(lock, name)}
    if os.path.isdir(path):
        markers = read_markers(path)
        if not markers:
            st["state"] = "foreign_partial"
            return st
        exp = expected_marker(lock, name)
        miss = None
        for m in markers:
            miss = partial_match(m, exp)
            if miss is None:
                st["marker"] = m
                break
        if miss is not None:
            st["state"] = "mismatch:%s" % miss
            return st
        if name == lock["lora"]["model"] and not os.path.isfile(os.path.join(path, lock["lora"]["file"])):
            st["state"] = "mismatch:file"
            return st
        st["state"] = "ready"
        return st
    if os.path.isdir(path + PARTIAL):
        st["state"] = "limner_partial"
        st["partial"] = path + PARTIAL
        return st
    st["state"] = "missing"
    return st


def all_status(lock: dict) -> list[dict]:
    return [status(lock, n) for n in model_names(lock)]


def generation_ready(lock: dict) -> list[str]:
    """Problems that stop generation (empty = ready): required models not `ready`."""
    return ["%s: %s (%s)" % (s["name"], s["state"], s["path"])
            for s in all_status(lock) if s["required"] and s["state"] != "ready"]


def quantized_path(lock: dict) -> str:
    return folder(lock, lock["quantized"]["name"])


def lora_file(lock: dict) -> str:
    return os.path.join(folder(lock, lock["lora"]["model"]), lock["lora"]["file"])


def mflux_warning(lock: dict, installed: str | None) -> str | None:
    s = status(lock, lock["quantized"]["name"])
    m = s.get("marker") or {}
    rec = m.get("mflux")
    if rec and installed and rec != installed:
        return "the 4-bit model was made with mflux %s; installed is %s (doctor --smoke decides)" % (rec, installed)
    return None


def write_markers(path: str, info: dict) -> None:
    """Both markers, same content (spec §3)."""
    body = json.dumps(info, indent=2) + "\n"
    for m in MARKERS:
        with open(os.path.join(path, m), "w") as f:
            f.write(body)
