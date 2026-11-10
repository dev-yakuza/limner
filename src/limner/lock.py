"""The machine lock shared with script2video and Studio (spec §4), and the hand-off of a lock
from `lmn run` to its worker (spec §6.1 step 7)."""

from __future__ import annotations

import os

from . import dirlock, util

HOLDER_PREFIX = "limner run "


def state_dir() -> str:
    """$S2V_STATE_DIR (absolute) or ~/.s2v; created 0700 when missing."""
    v = os.environ.get("S2V_STATE_DIR")
    d = v if v and os.path.isabs(v) else os.path.join(os.path.expanduser("~"), ".s2v")
    if not os.path.isdir(d):
        os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def machine_lock_path() -> str:
    return os.path.join(state_dir(), "heavy.lock.d")


def holder_text(run_id: str, status_file: str) -> str:
    return "%s%s %s" % (HOLDER_PREFIX, run_id, status_file)


def describe_holder(owner: dict | None) -> dict | None:
    if not owner:
        return None
    h = dirlock.display_text(owner.get("holder"))
    d = {"kind": dirlock.display_text(owner.get("kind")), "pid": owner.get("pid"), "holder": h}
    if h.startswith(HOLDER_PREFIX):
        rest = h[len(HOLDER_PREFIX):].split(" ", 1)
        d["run_id"] = rest[0]
        if len(rest) > 1:
            d["status_file"] = rest[1]
    return d


def resident_busy() -> dict | None:
    """s2v's resident TTS server (runs without the lock): busy when its owner is alive.
    A dead resident.json is left alone (s2v cleans it up)."""
    p = os.path.join(state_dir(), "resident.json")
    d = util.read_json_or_none(p)
    if not isinstance(d, dict):
        return None
    if util.alive(d.get("pid"), d.get("pid_start")):
        return {"reason": "resident", "holder": {"pid": d.get("pid"), "file": p}}
    return None


def _port(env: str, default: int) -> int:
    try:
        return int(os.environ.get(env, default))
    except ValueError:
        return default


def _health(port: int, timeout: float, tick) -> bool:
    """Does something answer `GET /health` on 127.0.0.1:port within `timeout`? Waits in 0.1 s
    slices and calls `tick()` between them (the caller's signal check)."""
    import errno
    import select
    import socket
    import time

    end = time.monotonic() + timeout
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        s.setblocking(False)
        err = s.connect_ex(("127.0.0.1", port))
        connected = err == 0
        while not connected:
            if err not in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EAGAIN, errno.EALREADY):
                return False  # refused
            tick()
            if time.monotonic() >= end:
                return False
            _, w, _ = select.select([], [s], [], 0.1)
            if w:
                err = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if err:
                    return False
                connected = True
        s.setblocking(True)
        s.settimeout(max(0.1, end - time.monotonic()))
        s.sendall(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        s.setblocking(False)

        while time.monotonic() < end:
            tick()
            r, _, _ = select.select([s], [], [], 0.1)
            if r:
                try:
                    return bool(s.recv(16))  # any answer at all
                except OSError:
                    return False
        return False
    except OSError:
        return False
    finally:
        s.close()


def server_busy(timeout: float = 2.0, tick=lambda: None) -> dict | None:
    """An s2v TTS / aligner server answering /health (s2v genPreflight): busy (`orphan`)."""
    for env, default in (("S2V_TTS_PORT", 47861), ("S2V_ALIGNER_PORT", 47862)):
        port = _port(env, default)
        if _health(port, timeout, tick):
            return {"reason": "orphan", "holder": {"port": port}}
    return None


def memory_busy(free_gib: float | None) -> dict | None:
    v = os.environ.get("LIMNER_MIN_FREE_GIB")
    if not v or free_gib is None:
        return None
    try:
        need = float(v)
    except ValueError:
        return None
    if free_gib < need:
        return {"reason": "memory", "holder": {"free_gib": free_gib, "min_free_gib": need}}
    return None


def try_machine(fields: dict, free_gib: float | None, env=None, tick=lambda: None) -> tuple[str | None, dict | None]:
    """Take the machine lock, then check what runs without it. (token, None) or (None, busy)."""
    lock = machine_lock_path()
    r = dirlock.acquire(lock, fields, env=env)
    if r.status == "breaking":
        return None, {"reason": "breaking", "holder": None}
    if r.status != "acquired":
        return None, {"reason": "lock", "holder": describe_holder(r.owner)}
    try:
        busy = resident_busy() or server_busy(tick=tick) or memory_busy(free_gib)
    except BaseException:  # tick() ended the caller (a signal): never keep the lock
        dirlock.release(lock, r.token)
        raise
    if busy:
        dirlock.release(lock, r.token)
        return None, busy
    return r.token, None


def hand_over(lock: str, expect_token: str, fields: dict) -> str | None:
    """Give `lock` to another process: only while `expect_token` still holds it, with a NEW
    token, by an atomic rename of owner.json (never rewritten in place: a reader of an empty file
    would break the lock as unreadable). The new token matters: a breaker that read the old owner
    re-checks the token under the mutex, sees it changed, reads again and finds the live worker
    (review R5-1). Returns the new token, or None when the lock is not ours any more."""
    o = dirlock.read_owner(lock)
    if not o or o.get("token") != expect_token:
        return None
    new = dirlock.new_token()
    tmp = os.path.join(lock, ".owner.%s.tmp" % dirlock.rand())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(dirlock.owner_body(fields, new))
    os.rename(tmp, os.path.join(lock, "owner.json"))
    return new


def owned_by(lock: str, token: str, pid: int, pid_start: str) -> bool:
    o = dirlock.read_owner(lock)
    return bool(o) and o.get("token") == token and o.get("pid") == pid and o.get("pid_start") == pid_start
