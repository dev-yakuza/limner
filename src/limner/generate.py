"""`lmn generate`: for people. Starts `lmn run` as a SUBPROCESS (so the worker is never our child —
no zombie that looks alive — and Ctrl-C is not swallowed by run's flag-only handler), then polls
`lmn status` and shows progress on stderr; Ctrl-C asks `lmn cancel`."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from . import util


def _lmn(*argv: str) -> tuple[dict, int]:
    """Run `lmn …` and read its JSON. A Ctrl-C reaches the child too (same process group): wait
    for it to finish its own handling instead of killing it, then raise."""
    p = subprocess.Popen([sys.executable, "-m", "limner", *argv], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    interrupted = False
    while True:
        try:
            out, _ = p.communicate()
            break
        except KeyboardInterrupt:
            interrupted = True
    try:
        obj = json.loads(out.decode("utf-8").strip().splitlines()[-1])
    except (ValueError, IndexError):
        obj = {"error": "no JSON from lmn %s" % argv[0]}
    if interrupted:
        raise Interrupted(obj, p.returncode)
    return obj, p.returncode


class Interrupted(Exception):
    def __init__(self, obj, code):
        super().__init__("interrupted")
        self.obj, self.code = obj, code


def cmd_generate(args) -> tuple[dict, int]:
    base = ["--root", args.root] + (["--state-dir", args.state_dir] if args.state_dir else [])
    argv = ["run", "--job", args.job] + base
    for flag in ("force", "accept"):
        v = getattr(args, flag)
        if v:
            argv += ["--%s" % flag, v]
    if args.wait is not None:
        argv += ["--wait", str(args.wait)]
    try:
        obj, code = _lmn(*argv)
    except Interrupted as e:
        if e.obj.get("state") != "starting":
            return e.obj, e.code  # run handled the signal itself (signal:2, exit 11)
        obj, code = e.obj, e.code
        print("[lmn] cancelling…", file=sys.stderr, flush=True)
        return _cancel(obj, base)
    if obj.get("state") != "starting" or code != 0:
        return obj, code
    ident = ["--run-id", obj["run_id"], "--attempt", obj["attempt"]] + base
    last = None
    try:
        while True:
            st, code = _lmn("status", *ident, "--wait-seconds", "5")
            line = "[lmn] %s stage %s/%s %s done %d pending %d" % (
                st.get("state"), st.get("stage"), st.get("stages"), st.get("current") or "",
                len(st.get("done") or []), len(st.get("pending") or []))
            if line != last:
                print(line, file=sys.stderr, flush=True)
                last = line
            if code != util.RUNNING:
                return st, code
    except (KeyboardInterrupt, Interrupted):
        print("[lmn] cancelling…", file=sys.stderr, flush=True)
        return _cancel(obj, base)


def _cancel(obj: dict, base: list) -> tuple[dict, int]:
    ident = ["--run-id", obj["run_id"], "--attempt", obj["attempt"]] + base
    while True:
        try:
            return _lmn("cancel", *ident, "--wait-seconds", "60")
        except Interrupted as e:
            return e.obj, e.code
