# Ported from Studio (sdd-plugin) `plugins/studio-plugin/skills/std/scripts/stdo/dirlock.py`,
# itself the Python port of script2video's `dir-lock.ts` (s2v 0.12.1 rules); same author.
# The algorithm is kept unchanged so limner, script2video and Studio can share one lock
# (`~/.s2v/heavy.lock.d`). When the lock format changes there, change it here too.
"""A lock that is a directory — the Python port of script2video's `dir-lock.ts` (S8).

script2video and Studio's lock.py take the same locks (`<state>/heavy.lock.d`, a character's
`.work/op.lock.d`), so every rule here follows s2v 0.12.1 exactly (m0c-plan §1):

- take: write owner.json inside a private `<lock>.new.<rand>/`, then rename it onto `<lock>`.
  The rename fails (ENOTEMPTY / EEXIST) while someone holds it, so `<lock>` only ever appears
  with its owner inside. A rename onto an EMPTY directory succeeds: never take it with mkdir.
- a dead owner (pid + process start time) is broken only under `<lock>.break` (mkdir mutex):
  the folder's inode and the owner's token are checked again there, and after the move.
- release: only by the token's holder (rename away, then delete).
"""

import errno
import json
import os
import re
import secrets
import shutil
import subprocess
import time

MUTEX_STALE_MS = 30000
BREAKING_TRIES = 5
UNREADABLE_TRIES = 5
TRIES = 50
LEFTOVER_MS = 60000

_LEFTOVER = re.compile(r"^\.(new|stale|gone)\.[0-9a-f]+$")
_LSTART = re.compile(r"^[A-Z][a-z]{2} ([A-Z][a-z]{2}) +(\d{1,2}) (\d{2}):(\d{2}):(\d{2}) (\d{4})$",
                     re.ASCII)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class CannotCheck(Exception):
    """`ps` could not run: a holder is then taken as alive."""


def rand():
    return secrets.token_hex(4)


def new_token():
    return secrets.token_hex(16)


def sleep_ms(ms):
    time.sleep(ms / 1000.0)


def valid_pid(pid):
    """s2v: Number.isInteger(pid) && pid > 0 (a float with an integer value counts)."""
    if isinstance(pid, bool):
        return False
    if isinstance(pid, int):
        return pid > 0
    if isinstance(pid, float):
        return pid.is_integer() and pid > 0
    return False


def proc_start(pid):
    """`LC_ALL=C TZ=UTC ps -o lstart= -p <pid>` (both ends trimmed), or None when no such process."""
    if not valid_pid(pid):
        return None
    env = dict(os.environ, LC_ALL="C", TZ="UTC")
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    except OSError as e:
        raise CannotCheck("cannot check process %s (ps: %s)" % (pid, e))
    if out.returncode < 0:  # ps itself was killed by a signal: like s2v, cannot check
        raise CannotCheck("cannot check process %s (ps ended by signal %d)" % (pid, -out.returncode))
    if out.returncode != 0:
        return None  # ps ran and found nothing
    text = out.stdout.decode("utf-8", "replace").strip()
    return text or None


def parse_lstart(s):
    """`Mon Oct  5 20:06:47 2026` read as UTC (ms), or None (also for dates that do not exist —
    JS Date.UTC would roll them over; ps never prints them, so the verdicts agree)."""
    m = _LSTART.match(s) if isinstance(s, str) else None
    if not m:
        return None
    if m.group(1) not in _MONTHS:
        return None
    import calendar
    import datetime
    try:
        d = datetime.datetime(int(m.group(6)), _MONTHS.index(m.group(1)) + 1, int(m.group(2)),
                              int(m.group(3)), int(m.group(4)), int(m.group(5)))
    except ValueError:
        return None
    return calendar.timegm(d.timetuple()) * 1000


def same_start(a, b, platform=None):
    """Two readings of one start time: the exact string, or on Linux within 2 s."""
    if a == b:
        return True
    if (platform or _platform()) != "linux":
        return False
    x, y = parse_lstart(a), parse_lstart(b)
    return x is not None and y is not None and abs(x - y) <= 2000


def _platform():
    import sys
    return "linux" if sys.platform.startswith("linux") else sys.platform


class LockEnv(object):
    """What the algorithm asks of the world; tests replace parts (ps, clock, hooks)."""

    def __init__(self, proc_start=proc_start, platform=None, now=None, hooks=None, warn=None,
                 sleep=sleep_ms):
        self.proc_start = proc_start
        self.platform = platform or _platform()
        self.now = now or (lambda: time.time() * 1000)
        self.hooks = hooks or {}
        self.warn = warn
        self.sleep = sleep

    def hook(self, name, *args):
        f = self.hooks.get(name)
        if f:
            f(*args)


def liveness(owner, env):
    """(alive, cannot_check) of an owner."""
    try:
        start = env.proc_start(owner["pid"])
    except CannotCheck:
        return True, True
    return (start is not None and same_start(start, owner["pid_start"], env.platform)), False


def _reject_constant(name):
    raise ValueError("not JSON: %s" % name)


def read_owner(lock):
    """owner.json of a lock directory, or None when missing or not an owner."""
    try:
        with open(os.path.join(lock, "owner.json"), encoding="utf-8") as f:
            o = json.loads(f.read(), parse_constant=_reject_constant)
    except (OSError, ValueError):
        return None
    if not isinstance(o, dict):
        return None
    pid = o.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, (int, float)):
        return None
    if not isinstance(o.get("pid_start"), str) or not isinstance(o.get("token"), str):
        return None
    return o


def _token(o):
    return o.get("token") if o else None


def inode(path):
    try:
        return os.stat(path).st_ino
    except OSError:
        return None


class Result(object):
    def __init__(self, status, token=None, owner=None, cannot_check=False, note=None):
        self.status = status  # acquired | reentered | held | breaking
        self.token = token
        self.owner = owner
        self.cannot_check = cannot_check
        self.note = note

    def __repr__(self):
        return "Result(%s)" % self.status


def owner_body(fields, token):
    d = dict(fields)
    d["token"] = token
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n"


def acquire(lock, fields, env_token=None, token=None, env=None):
    """Take `lock` for `fields` (written with `token`, new if None), or say why not."""
    env = env or LockEnv()
    token = token or new_token()
    body = owner_body(fields, token)
    parent = os.path.dirname(lock)
    os.makedirs(parent, exist_ok=True)
    seen = {}

    def live(o):
        key = (o.get("pid"), o.get("pid_start"), o.get("token"))
        if key not in seen:
            seen[key] = liveness(o, env)
        return seen[key]

    breaking = 0
    unreadable = 0
    for _ in range(TRIES):
        priv = "%s.new.%s" % (lock, rand())
        os.mkdir(priv)
        try:
            with open(os.path.join(priv, "owner.json"), "w", encoding="utf-8") as f:
                f.write(body)
            os.rename(priv, lock)
            return Result("acquired", token=token)
        except OSError as e:
            shutil.rmtree(priv, ignore_errors=True)
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
        # the lock as judged: its folder (inode) and its owner
        ino = inode(lock)
        if ino is None:
            continue  # released meanwhile
        env.hook("before_owner_read")
        cur = read_owner(lock)
        if cur:
            unreadable = 0
        else:
            unreadable += 1
            if unreadable <= UNREADABLE_TRIES:
                env.sleep(10)
                continue
        if cur and env_token and cur.get("token") == env_token:
            return Result("reentered", owner=cur)
        if cur:
            alive, cannot = live(cur)
            if alive:
                return Result("held", owner=cur, cannot_check=cannot)
        # dead, or no readable owner: break it, under the mutex only
        mutex = lock + ".break"
        try:
            os.mkdir(mutex)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            try:
                if abs(env.now() - os.stat(mutex).st_mtime * 1000) > MUTEX_STALE_MS:
                    os.rmdir(mutex)
                    continue
            except OSError:
                continue  # gone meanwhile
            breaking += 1
            if breaking > BREAKING_TRIES:
                return Result("breaking")
            env.sleep(10)
            continue
        try:
            env.hook("in_mutex")
            if inode(lock) != ino:
                continue
            if _token(read_owner(lock)) != _token(cur):
                continue
            if inode(lock) != ino:
                continue
            env.hook("before_rename")
            stale = "%s.stale.%s" % (lock, rand())
            try:
                os.rename(lock, stale)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
                continue
            env.hook("after_break_rename", stale)
            moved = read_owner(stale)
            if inode(stale) == ino and _token(moved) == _token(cur):
                shutil.rmtree(stale, ignore_errors=True)
                continue
            # another lock took its place between the check and the rename: put it back
            back = False
            try:
                os.rename(stale, lock)
                back = True
            except OSError:
                pass
            note = ("moved a lock that was not the judged one; put it back" if back else
                    "moved a lock that was not the judged one; it is left at %s" % stale)
            if env.warn:
                env.warn("warning: " + note)
            return Result("held", owner=moved, note=note)
        finally:
            try:
                os.rmdir(mutex)
            except OSError:
                pass  # removed as stale by another process
    return Result("held", owner=read_owner(lock), note="contended")


def release(lock, token):
    """Release `lock` when `token` holds it; False when it is not ours (any more)."""
    if _token(read_owner(lock)) != token:
        return False
    gone = "%s.gone.%s" % (lock, rand())
    try:
        os.rename(lock, gone)
    except OSError:
        return False
    shutil.rmtree(gone, ignore_errors=True)
    return True


def clean_leftovers(lock, env=None):
    """Leftovers of `lock` (`.new.*`, `.stale.*`, `.gone.*`, lowercase hex) older than 60 s whose
    owner is dead or missing: removed; their names are returned."""
    env = env or LockEnv()
    parent, base = os.path.dirname(lock), os.path.basename(lock)
    out = []
    try:
        names = os.listdir(parent)
    except OSError:
        return out
    for name in sorted(names):
        if not name.startswith(base) or not _LEFTOVER.match(name[len(base):]):
            continue
        p = os.path.join(parent, name)
        try:
            if env.now() - os.stat(p).st_mtime * 1000 <= LEFTOVER_MS:
                continue
            o = read_owner(p)
            if o and liveness(o, env)[0]:
                continue
            shutil.rmtree(p)
            out.append(name)
        except OSError:
            pass
    return out


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def display_text(s, limit=200):
    """Control characters out, length capped (holder strings come from other programs).
    Counts code points (JS counts UTF-16 units): display only."""
    t = _CONTROL.sub(" ", "" if s is None else str(s))
    return t[: limit - 1] + "…" if len(t) > limit else t
