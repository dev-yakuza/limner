"""`lmn doctor [--json] [--smoke]` and `lmn gc` (spec §9, §5)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time

from . import dirlock, fonts, lock, models, util
from .util import Fail


def _version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def internals() -> dict:
    """Do the mflux internals limner uses exist? (imports mflux in a subprocess: no model load)"""
    import importlib.util

    if importlib.util.find_spec("mflux") is None:
        return {"ok": False, "missing": ["mflux not installed"]}
    code = (
        "import json\n"
        "from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit as Q\n"
        "from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil as U\n"
        "from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage as I\n"
        "from limner.child import INTERNAL_METHODS\n"
        "m=[n for n in INTERNAL_METHODS if not callable(getattr(Q,n,None))]\n"
        "m+=[] if callable(getattr(U,'create_image_conditioning_latents',None)) else ['QwenEditUtil.create_image_conditioning_latents']\n"
        "m+=[] if callable(getattr(I,'compute_guided_noise',None)) else ['QwenImage.compute_guided_noise']\n"
        "print(json.dumps(m))\n"
    )
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, HF_HUB_OFFLINE="1"))
        import json

        missing = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "missing": ["cannot import mflux: %s" % e]}
    return {"ok": not missing, "missing": missing}


def report() -> dict:
    from PIL import features

    lock_ = models.load_lock()
    src, mdir = models.models_dir()
    sts = models.all_status(lock_)
    warns = []
    w = models.mflux_warning(lock_, _version("mflux"))
    if w:
        warns.append(w)
    for s in sts:
        if s["state"] == "ready" and not s["required"]:
            warns.append("%s is only needed to convert; it uses disk at %s (remove it yourself if unneeded)"
                         % (s["name"], s["path"]))
    owner = dirlock.read_owner(lock.machine_lock_path())
    from .runner import memory_limit_gib

    info = {
        "contract": util.CONTRACT,
        "platform": "%s %s" % (sys.platform, platform.machine()),
        "apple_silicon": util.is_apple_silicon(),
        "ram_gib": round(util.ram_gib(), 1),
        "free_gib": util.free_gib(),
        "memory_limit_gib": round(memory_limit_gib(lock_), 1),
        "models_dir": {"source": src, "path": mdir, "candidates": [d for _, d in models.candidates()]},
        "disk_free_gib": util.disk_free_gib(mdir),
        "versions": {"limner": __import__("limner").__version__, "mflux": _version("mflux"), "mlx": _version("mlx"),
                     "pillow": _version("pillow"), "raqm": features.check("raqm"),
                     "freetype": features.version("freetype2"), "resvg_py": _version("resvg-py")},
        "models": sts,
        "fonts": {k: fonts.installed(k) for k in fonts.FONTS},
        "fonts_dir": fonts.fonts_dir(),
        "lock": {"path": lock.machine_lock_path(), "owner": lock.describe_holder(owner)},
        "internals": internals(),
        "warnings": warns,
    }
    info["ready"] = bool(info["apple_silicon"] and not models.generation_ready(lock_) and info["internals"]["ok"])
    return info


def text(info: dict) -> str:
    lines = ["limner %s on %s, RAM %s GiB (free ~%s GiB), memory limit %s GiB"
             % (info["versions"]["limner"], info["platform"], info["ram_gib"], info["free_gib"],
                info["memory_limit_gib"]),
             "models dir: %s (%s), disk free %s GiB" % (info["models_dir"]["path"], info["models_dir"]["source"],
                                                       info["disk_free_gib"])]
    for s in info["models"]:
        lines.append("  %-34s %-16s %s%s" % (s["name"], s["state"], "required " if s["required"] else "optional ",
                                             s["path"]))
    lines.append("versions: " + ", ".join("%s %s" % kv for kv in info["versions"].items()))
    lines.append("fonts (%s): %s" % (info["fonts_dir"], ", ".join(
        "%s %s" % (k, "ok" if v else "MISSING") for k, v in info["fonts"].items())))
    o = info["lock"]["owner"]
    lines.append("machine lock %s: %s" % (info["lock"]["path"], "free" if not o else o))
    lines.append("mflux internals: %s" % ("ok" if info["internals"]["ok"] else info["internals"]["missing"]))
    for w in info["warnings"]:
        lines.append("warning: " + w)
    if "smoke" in info:
        lines.append("smoke: %s" % info["smoke"])
    lines.append("ready to generate: %s" % ("yes" if info["ready"] else "no"))
    return "\n".join(lines) + "\n"


def smoke() -> tuple[dict, int]:
    """64x64, 2 steps, under the machine lock: the internals and the weights really work."""
    if os.environ.get("S2V_HEAVY_LOCK_TOKEN"):
        raise Fail(util.BAD, "S2V_HEAVY_LOCK_TOKEN is set: not supported")
    lock_ = models.load_lock()
    me = util.me()
    tok, busy = lock.try_machine({"kind": "gen", "pid": me["pid"], "pid_start": me["pid_start"],
                                  "holder": "limner doctor --smoke"}, util.free_gib())
    if not tok:
        return {"ok": False, "busy": busy}, util.BUSY
    try:
        with tempfile.TemporaryDirectory(prefix="limner-smoke-") as d:
            from . import jobs

            blank = os.path.join(d, "blank.png")
            jobs.blank_canvas(64, 64, lock_["generation"]["blank"], blank)
            gen = dict(lock_["generation"], steps=2)
            spec = {"fake": os.environ.get("LIMNER_FAKE_GENERATOR") == "1", "generation": gen,
                    "jobs": [{"id": "smoke", "prompt": "A small red circle.", "width": 64, "height": 64, "seed": 1,
                              "images": [blank], "tmp_out": os.path.join(d, "smoke.png"), "job_sha": "00" * 32}]}
            sf, ef = os.path.join(d, "stage.json"), os.path.join(d, "events.jsonl")
            util.write_json_atomic(sf, spec)
            t = time.time()
            r, w = os.pipe()
            try:
                p = subprocess.run([sys.executable, "-m", "limner.child", sf, ef], stdin=r, stdout=sys.stderr,
                                   stderr=sys.stderr, timeout=1800)
            finally:
                os.close(r)
                os.close(w)
            from .worker import read_events

            _, evs = read_events(ef, 0)
            done = any(e.get("event") == "done" for e in evs)
            errs = [e.get("message") for e in evs if e.get("event") in ("error", "environment")]
            ok = p.returncode == 0 and done and os.path.isfile(os.path.join(d, "smoke.png"))
            res = {"ok": ok, "seconds": round(time.time() - t, 1), "exit": p.returncode, "errors": errs}
            if ok:
                return res, util.OK
            return res, (util.BAD if p.returncode == 2 else util.FAILED)
    finally:
        dirlock.release(lock.machine_lock_path(), tok)


def cmd_doctor(args) -> tuple[dict, int]:
    info = report()
    code = util.OK
    if args.smoke:
        info["smoke"], code = smoke()
    info["text"] = text(info)
    return info, code


def _size(path: str) -> int:
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def cmd_gc(args) -> tuple[dict, int]:
    sd = os.path.abspath(args.state_dir)
    if not os.path.isdir(sd):
        raise Fail(util.BAD, "--state-dir %s is not a directory" % sd)
    out = {"contract": util.CONTRACT, "state_dir": sd}
    for name in ("prev", "runs", "tmp"):
        p = os.path.join(sd, name)
        entries = []
        if os.path.isdir(p):
            for dp, _dn, fn in os.walk(p):
                for f in fn:
                    fp = os.path.join(dp, f)
                    try:
                        entries.append({"path": os.path.relpath(fp, sd), "bytes": os.lstat(fp).st_size})
                    except OSError:
                        pass
        out[name] = {"bytes": _size(p) if os.path.isdir(p) else 0, "files": sorted(entries, key=lambda e: e["path"])}
    return out, util.OK
