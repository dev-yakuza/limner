"""`lmn` — one JSON line on stdout and an exit code per command (spec §10)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from . import util
from .util import Fail


class Parser(argparse.ArgumentParser):
    def error(self, message):  # usage errors are exit 2 with JSON like any bad input
        raise Fail(util.BAD, "usage: %s" % message)


def _common(p, state=True):
    p.add_argument("--root", required=True)
    if state:
        p.add_argument("--state-dir")
    p.add_argument("--result", help="also write the result JSON to this file (atomically)")


def build() -> Parser:
    ap = Parser(prog="lmn", description="limner: a local image engine")
    ap.add_argument("--version", action="store_true")
    sub = ap.add_subparsers(dest="cmd", parser_class=Parser)

    p = sub.add_parser("run", help="start a detached generation (returns at once)")
    p.add_argument("--job", required=True)
    _common(p)
    p.add_argument("--force")
    p.add_argument("--accept")
    p.add_argument("--wait", type=int, default=0)
    p.add_argument("--check", action="store_true")

    p = sub.add_parser("generate", help="run, then follow it until it ends (for people)")
    p.add_argument("--job", required=True)
    _common(p)
    p.add_argument("--force")
    p.add_argument("--accept")
    p.add_argument("--wait", type=int)

    for name in ("status", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--attempt")
        _common(p)
        p.add_argument("--wait-seconds", type=int)

    p = sub.add_parser("compose", help="draw text and layout over images")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--layout")
    g.add_argument("--template")
    p.add_argument("--data")
    p.add_argument("--templates")
    p.add_argument("--out", required=True)
    _common(p)

    p = sub.add_parser("rasterize", help="SVG to PNG")
    p.add_argument("--svg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int)
    p.add_argument("--background")
    _common(p)

    p = sub.add_parser("doctor")
    p.add_argument("--json", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--result")

    p = sub.add_parser("setup")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--result")

    p = sub.add_parser("fonts")
    fs = p.add_subparsers(dest="fonts_cmd", parser_class=Parser)
    fi = fs.add_parser("install")
    fi.add_argument("--result")

    p = sub.add_parser("gc", help="list what prev/, runs/ and tmp/ hold (removes nothing)")
    p.add_argument("--state-dir", required=True)
    p.add_argument("--result")
    return ap


def dispatch(args) -> tuple[dict, int]:
    c = args.cmd
    if c == "run":
        from .runner import cmd_run
        return cmd_run(args)
    if c == "generate":
        from .generate import cmd_generate
        return cmd_generate(args)
    if c == "status":
        from .status import cmd_status
        return cmd_status(args)
    if c == "cancel":
        from .status import cmd_cancel
        return cmd_cancel(args)
    if c == "compose":
        from .compose.command import cmd_compose
        return cmd_compose(args)
    if c == "rasterize":
        from .rasterize import cmd_rasterize
        return cmd_rasterize(args)
    if c == "doctor":
        from .doctor import cmd_doctor
        return cmd_doctor(args)
    if c == "setup":
        from .setup import cmd_setup
        return cmd_setup(args)
    if c == "fonts":
        if args.fonts_cmd != "install":
            raise Fail(util.BAD, "usage: lmn fonts install")
        from .fonts import cmd_install
        return cmd_install(args)
    if c == "gc":
        from .doctor import cmd_gc
        return cmd_gc(args)
    raise Fail(util.BAD, "usage: lmn {run,generate,status,cancel,compose,rasterize,doctor,setup,fonts,gc} …")


def emit(obj: dict, args) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    out = getattr(args, "result", None) if args is not None else None
    if out:
        try:
            util.write_json_atomic(os.path.abspath(out), obj)
        except OSError as e:
            print("[lmn] cannot write --result %s: %s" % (out, e), file=sys.stderr)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = None
    try:
        args = build().parse_args(argv)
        if args.version:
            import limner
            emit({"contract": util.CONTRACT, "limner": limner.__version__}, args)
            return 0
        obj, code = dispatch(args)
    except Fail as e:
        obj = {"contract": util.CONTRACT, "error": e.message}
        obj.update(e.extra)
        code = e.code
    except SystemExit as e:  # argparse --help
        return int(e.code or 0)
    except Exception as e:  # noqa: BLE001 - unexpected: a bug, exit 70
        traceback.print_exc()
        obj = {"contract": util.CONTRACT, "error": "internal error: %s" % e}
        code = util.INTERNAL
    if getattr(args, "cmd", None) == "doctor" and not getattr(args, "json", False) and "text" in obj:
        sys.stdout.write(obj["text"])
        sys.stdout.flush()
        return code
    obj.pop("text", None)
    emit(obj, args)
    return code


if __name__ == "__main__":
    sys.exit(main())
