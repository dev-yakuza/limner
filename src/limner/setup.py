"""`lmn setup [--yes]` (spec §3, §9): make the required models ready, then install the fonts.

Models that script2video already made are used as they are (nothing is downloaded). limner writes
only into `<folder>.limner-partial/` (with owner.json), then both markers, then renames it into
place; it never touches a folder without its own partial suffix and never deletes the source model.
Steps follow script2video `imagegen/setup_models.py` (MIT, same author).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from . import dirlock, fonts, lock, models, util
from .util import Fail, bad


def owner_alive(partial: str) -> bool:
    o = util.read_json_or_none(os.path.join(partial, "owner.json"))
    return isinstance(o, dict) and util.alive(o.get("pid"), o.get("pid_start"))


def claim(partial: str, step: str) -> None:
    """Own a `.limner-partial` folder: new, or taken over from a dead owner (a download resumes;
    a conversion cannot, so it starts over)."""
    if os.path.isdir(partial):
        if owner_alive(partial):
            raise bad("%s is being written by another limner process" % partial)
        o = util.read_json_or_none(os.path.join(partial, "owner.json")) or {}
        if step == "quantize" or o.get("step") == "quantize":
            shutil.rmtree(partial)
    os.makedirs(partial, exist_ok=True)
    me = util.me()
    util.write_json_atomic(os.path.join(partial, "owner.json"),
                           {"tool": "limner", "step": step, "pid": me["pid"], "pid_start": me["pid_start"],
                            "started_at": util.utc_now()})


def finish(partial: str, target: str, info: dict) -> None:
    os.unlink(os.path.join(partial, "owner.json"))
    models.write_markers(partial, info)
    os.rename(partial, target)


def snapshot_download(**kw):  # replaced in tests
    from huggingface_hub import snapshot_download as sd

    return sd(**kw)


def download(lock_: dict, name: str) -> None:
    e = lock_["models"][name]
    target = models.folder(lock_, name)
    partial = target + models.PARTIAL
    claim(partial, "download")
    print("[lmn setup] download %s@%s -> %s" % (e["repo"], e["revision"][:12], target), file=sys.stderr, flush=True)
    snapshot_download(repo_id=e["repo"], revision=e["revision"], local_dir=partial,
                      allow_patterns=e.get("allow_patterns"))
    finish(partial, target, {"repo": e["repo"], "revision": e["revision"]})


def quantize(lock_: dict) -> None:
    """The 4-bit conversion in a child process under a memory watch of min(44, 0.7 x RAM) GiB."""
    q = lock_["quantized"]
    src = models.folder(lock_, q["from"])
    target = models.quantized_path(lock_)
    partial = target + models.PARTIAL
    claim(partial, "quantize")
    staged = os.path.join(partial, "weights")
    m = lock_["setup"]["quantize_memory_limit"]
    limit = min(m["max_gib"], m["ram_fraction"] * util.ram_gib())
    print("[lmn setup] quantize %s -> %s (%d-bit, memory limit %.0f GiB)" % (src, target, q["bits"], limit),
          file=sys.stderr, flush=True)
    code = ("import sys, mlx.core as mx\n"
            "from limner.child import exit_with_parent\n"
            "exit_with_parent()\n"
            "mx.set_cache_limit(1_000_000_000)\n"
            "from mflux.models.common.cli.save import main\n"
            "sys.argv=['mflux-save','--model',sys.argv[1],'--base-model','qwen-image-edit-2511','--quantize',"
            "sys.argv[2],'--path',sys.argv[3]]\n"
            "sys.exit(main() or 0)\n")
    r, w = os.pipe()
    from .worker import footprint_gib

    p = subprocess.Popen([sys.executable, "-c", code, src, str(q["bits"]), staged], stdin=r,
                         stdout=sys.stderr, stderr=sys.stderr, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    os.close(r)
    killed = False
    while p.poll() is None:
        fp = footprint_gib(p.pid)
        if fp is not None and fp > limit:
            p.kill()
            killed = True
        time.sleep(3)
    os.close(w)
    if killed:
        raise Fail(util.FAILED, "the conversion went over %.0f GiB and was stopped" % limit)
    if p.returncode != 0:
        raise Fail(util.FAILED, "the conversion failed (exit %s)" % p.returncode)
    for n in os.listdir(staged):
        os.rename(os.path.join(staged, n), os.path.join(partial, n))
    os.rmdir(staged)
    from importlib.metadata import version

    finish(partial, target, {"from": q["from"], "revision": lock_["models"][q["from"]]["revision"],
                             "bits": q["bits"], "mflux": version("mflux")})


def confirm(text: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise bad("setup needs confirmation: %s — rerun with --yes" % text)
    sys.stderr.write("%s\nContinue? [y/N] " % text)
    sys.stderr.flush()
    if sys.stdin.readline().strip().lower() not in ("y", "yes"):
        raise bad("setup cancelled")


def cmd_setup(args) -> tuple[dict, int]:
    if os.environ.get("LIMNER_FAKE_GENERATOR") != "1" and not util.is_apple_silicon():
        raise bad("this machine cannot generate (macOS on Apple silicon only)")
    lock_ = models.load_lock()
    sts = {s["name"]: s for s in models.all_status(lock_)}
    req = [s for s in sts.values() if s["required"]]
    done = []
    if all(s["state"] == "ready" for s in req):
        res_models = "ready"
    else:
        for s in req:
            if s["state"] == "foreign_partial":
                raise bad("%s exists without a marker (another tool is writing it, or it stopped): "
                          "left alone; finish or remove it with that tool" % s["path"])
            if s["state"].startswith("mismatch"):
                raise bad("%s does not match limner's lock (%s): decide by hand (script2video may use it)"
                          % (s["path"], s["state"]))
        q = lock_["quantized"]
        steps = []
        for s in req:
            if s["state"] == "ready":
                continue
            if s["name"] == q["name"]:
                src = sts[q["from"]]
                if src["state"] == "foreign_partial" or src["state"].startswith("mismatch"):
                    raise bad("the source model %s is %s: left alone" % (src["path"], src["state"]))
                if src["state"] != "ready":
                    steps.append(("download", q["from"]))
                steps.append(("quantize", q["name"]))
            else:
                steps.append(("download", s["name"]))
        ram = util.ram_gib()
        if ram < lock_["setup"]["min_ram_gb"]:
            raise bad("setup needs %d GB RAM (the conversion peaks at 30-32 GB); this machine has %.0f"
                      % (lock_["setup"]["min_ram_gb"], ram))
        need = sum(lock_["models"][n]["size_gb"] for st, n in steps if st == "download")
        need += q["size_gb"] if any(st == "quantize" for st, _ in steps) else 0
        _, mdir = models.models_dir()
        free = util.disk_free_gib(mdir) or 0
        if free < need * 1.1:
            raise bad("needs about %.0f GB free under %s; %.0f GB free" % (need, mdir, free))
        confirm("limner setup will: %s (about %.0f GB into %s)" % (
            ", ".join("%s %s" % s for s in steps), need, mdir), args.yes)
        os.makedirs(mdir, exist_ok=True)
        me = util.me()
        tok, busy = lock.try_machine({"kind": "gen", "pid": me["pid"], "pid_start": me["pid_start"],
                                      "holder": "limner setup"}, util.free_gib())
        if not tok:
            raise Fail(util.BUSY, "the machine is busy", busy=busy)
        try:
            for step, name in steps:
                if step == "download":
                    download(lock_, name)
                else:
                    quantize(lock_)
                done.append("%s %s" % (step, name))
        finally:
            dirlock.release(lock.machine_lock_path(), tok)
        res_models = "installed"
    f = fonts.install()
    return {"contract": util.CONTRACT, "models": res_models, "steps": done,
            "models_dir": models.models_dir()[1], "fonts": f}, util.OK
