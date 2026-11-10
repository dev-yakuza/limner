"""`lmn status` and `lmn cancel` (spec §6.1)."""

from __future__ import annotations

import os
import signal
import time

from . import paths, result, runs, util
from .util import Fail

MAX_WAIT = 540
KILL_GRACE = 10.0


def locate(args) -> runs.Attempt:
    root = paths.Root(args.root, args.state_dir)
    run = runs.Run(root.state_dir, args.run_id)
    if not os.path.isdir(run.dir):
        raise Fail(util.NOT_FOUND, "no run %s under %s" % (args.run_id, root.state_dir))
    a = args.attempt or run.current_attempt()
    if not a:
        raise Fail(util.NOT_FOUND, "run %s has no current attempt" % args.run_id)
    att = runs.Attempt(run, a)
    if not os.path.isdir(att.dir):
        raise Fail(util.NOT_FOUND, "no attempt %s in run %s" % (a, args.run_id))
    return att


def actors(att: runs.Attempt) -> dict:
    """Who may still write this attempt: the worker and its child (after spawn.json), else the
    `run` that made it. {"worker": bool, "child": bool, "run": bool}."""
    r = att.read("run.json") or {}
    run_alive = util.alive(r.get("pid"), r.get("pid_start"))  # until go it writes; then it exits at once
    spawn = att.read("spawn.json")
    if spawn:
        child = att.read("child.json")
        return {"worker": util.alive(spawn.get("pid"), spawn.get("pid_start")),
                "child": bool(child) and util.alive(child.get("pid"), child.get("pid_start")),
                "run": run_alive, "spawned": True}
    return {"worker": False, "child": False, "run": run_alive, "spawned": False}


def judge(att: runs.Attempt) -> dict:
    """The attempt's status, with the verdict for a live state whose writers are all gone
    (judged only, never written)."""
    st = att.read_status()
    if st is None:
        raise Fail(util.NOT_FOUND, "attempt %s has no status" % att.id)
    if st.get("state") not in result.LIVE:
        return st
    a = actors(att)
    if a["worker"] or a["run"]:
        return st
    again = att.read_status()  # the last writer may have closed it and exited since the first read
    if again and again.get("state") not in result.LIVE:
        return again
    st = dict(again or st)
    if a["child"]:
        st["state"] = "orphan_child"
    elif att.cancel_requested():
        st["state"] = "cancelled"
    else:
        st["state"] = "crashed"
    st["judged"] = True
    st["actors"] = {k: a[k] for k in ("worker", "child", "run")}
    return st


def cmd_status(args) -> tuple[dict, int]:
    att = locate(args)
    wait = args.wait_seconds or 0
    if not 0 <= wait <= MAX_WAIT:
        raise util.bad("--wait-seconds must be 0 to %d" % MAX_WAIT)
    end = time.monotonic() + wait
    while True:
        st = judge(att)
        if st.get("state") not in result.LIVE or time.monotonic() >= end:
            return st, result.exit_code(st)
        time.sleep(min(1.0, max(0.05, end - time.monotonic())))


def _kill_checked(pid, pid_start, group: bool, sig) -> None:
    if not util.alive(pid, pid_start):
        return
    try:
        if group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop_worker(att: runs.Attempt) -> None:
    """SIGTERM to the worker's group and to the child (each only when its pid_start matches),
    SIGKILL after 10 s; returns when both are dead."""
    spawn = att.read("spawn.json") or {}
    child = att.read("child.json") or {}
    procs = [(spawn.get("pid"), spawn.get("pid_start"), True), (child.get("pid"), child.get("pid_start"), False)]
    for pid, ps, group in procs:
        if pid and ps:
            _kill_checked(pid, ps, group, signal.SIGTERM)
    end = time.monotonic() + KILL_GRACE
    while time.monotonic() < end:
        if not any(pid and ps and util.alive(pid, ps) for pid, ps, _ in procs):
            return
        time.sleep(0.1)
    for pid, ps, group in procs:
        if pid and ps:
            _kill_checked(pid, ps, group, signal.SIGKILL)
    end = time.monotonic() + 5
    while time.monotonic() < end and any(pid and ps and util.alive(pid, ps) for pid, ps, _ in procs):
        time.sleep(0.05)


def _write_cancelled(att: runs.Attempt) -> dict:
    """Re-read; leave a final state alone; else the job in progress failed: cancelled."""
    st = att.read_status()
    if st is None or st.get("state") not in result.LIVE:
        return st
    cur = st.get("current")
    if cur and cur in (st.get("pending") or []):
        st["pending"].remove(cur)
        st["failed"].append({"id": cur, "reason": "cancelled", "message": "cancelled by lmn cancel"})
    st["state"] = "cancelled"
    st["current"] = None
    att.write_status(st)
    return st


def cmd_cancel(args) -> tuple[dict, int]:
    att = locate(args)
    wait = 30 if args.wait_seconds is None else args.wait_seconds
    if not 0 <= wait <= MAX_WAIT:
        raise util.bad("--wait-seconds must be 0 to %d" % MAX_WAIT)
    st = judge(att)
    if st.get("state") in result.FINAL and st.get("state") != "orphan_child":
        if st.get("judged") and st.get("state") == "cancelled":  # an earlier request, all writers gone
            st = _write_cancelled(att)
            return st, (util.OK if st.get("state") == "cancelled" else result.exit_code(st))
        return st, result.exit_code(st)  # also `crashed`: not turned into a cancel
    if not att.cancel_requested():
        util.write_text_atomic(att.p("cancel_requested"), util.utc_now() + "\n")
    end = time.monotonic() + wait
    while True:
        st = att.read_status()
        if st is None:
            raise Fail(util.NOT_FOUND, "attempt %s has no status" % att.id)
        state = st.get("state")
        if state in result.FINAL:
            return st, (util.OK if state == "cancelled" else result.exit_code(st))
        a = actors(att)
        if state == "running" and (a["worker"] or a["child"]):
            _stop_worker(att)
            st = _write_cancelled(att)
            return st, (util.OK if st.get("state") == "cancelled" else result.exit_code(st))
        if not (a["worker"] or a["run"] or a["child"]):
            st = _write_cancelled(att)
            return st, (util.OK if st.get("state") == "cancelled" else result.exit_code(st))
        if a["child"] and not a["worker"] and a["spawned"]:
            _stop_worker(att)  # orphan child
            continue
        if time.monotonic() >= end:
            return st, util.RUNNING
        time.sleep(0.2)
