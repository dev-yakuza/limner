"""Ported from Studio `tests/py/test_lock_lib.py` (DirLock, L1–L10), same author; plus a race of
real processes (Studio's I2 race, Python takers only)."""

import json
import os
import subprocess
import sys
import time

from limner import dirlock

LIVE_START = "Mon Oct 19 20:06:47 2026"


def env_with(alive=None, cannot=False, now=None, hooks=None, platform="darwin"):
    """A LockEnv whose ps says: pids in `alive` run with LIVE_START."""
    alive = set(alive or [])

    def ps(pid):
        if cannot:
            raise dirlock.CannotCheck("no ps")
        return LIVE_START if pid in alive else None

    return dirlock.LockEnv(proc_start=ps, platform=platform, now=now, hooks=hooks, sleep=lambda ms: None)


def owner(lock, pid=4242, start=LIVE_START, token="a" * 32, **extra):
    os.makedirs(lock, exist_ok=True)
    d = {"kind": "tts", "pid": pid, "pid_start": start, "holder": "x", "token": token}
    d.update(extra)
    with open(os.path.join(lock, "owner.json"), "w") as f:
        json.dump(d, f)


def test_l1_take_hold_release(tmp_path):
    lock = str(tmp_path / "state" / "heavy.lock.d")
    e = env_with(alive={os.getpid()})
    r = dirlock.acquire(lock, {"kind": "tts", "pid": os.getpid(), "pid_start": LIVE_START, "holder": "h"}, env=e)
    assert r.status == "acquired"
    body = open(os.path.join(lock, "owner.json")).read()
    assert body.endswith("\n")
    assert list(json.loads(body)) == ["kind", "pid", "pid_start", "holder", "token"]
    assert len(r.token) == 32 and all(c in "0123456789abcdef" for c in r.token)
    r2 = dirlock.acquire(lock, {"kind": "tts", "pid": 1, "pid_start": "x", "holder": "h"}, env=e)
    assert r2.status == "held"
    assert not dirlock.release(lock, "b" * 32)
    assert dirlock.release(lock, r.token)
    assert not os.path.exists(lock)


def test_l2_rename_onto_an_empty_dir_succeeds(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    os.makedirs(lock)
    assert dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with()).status == "acquired"


def test_l3_unreadable_owner_is_retried_then_broken(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    os.makedirs(lock)
    with open(os.path.join(lock, "owner.json"), "w") as f:
        f.write("{broken")
    reads = []
    e = env_with(hooks={"before_owner_read": lambda: reads.append(1)})
    assert dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=e).status == "acquired"
    assert len(reads) == dirlock.UNREADABLE_TRIES + 1


def test_l3_owner_appearing_meanwhile_is_not_broken(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    os.makedirs(lock)
    with open(os.path.join(lock, "owner.json"), "w") as f:
        f.write("{being written")
    calls = []

    def fix():
        calls.append(1)
        if len(calls) == 2:
            owner(lock, pid=7)

    r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(alive={7}, hooks={"before_owner_read": fix}))
    assert r.status == "held"


def test_l4_dead_and_live_owners(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    owner(lock, pid=7)
    assert dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with()).status == "acquired"
    dirlock.release(lock, dirlock.read_owner(lock)["token"])
    owner(lock, pid=7, start="Mon Oct 19 20:06:48 2026")
    assert dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(alive={7})).status == "acquired"
    dirlock.release(lock, dirlock.read_owner(lock)["token"])
    owner(lock, pid=7)
    r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(cannot=True))
    assert (r.status, r.cannot_check) == ("held", True)


def test_l5_lock_replaced_inside_the_mutex_is_not_broken(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    owner(lock, pid=7, token="1" * 32)

    def replace():
        dirlock.release(lock, "1" * 32)
        owner(lock, pid=8, token="2" * 32)

    r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(alive={8}, hooks={"in_mutex": replace}))
    assert r.status == "held"
    assert dirlock.read_owner(lock)["token"] == "2" * 32


def test_l5_owner_rewritten_in_place_inside_the_mutex_is_not_moved(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    owner(lock, pid=7, token="1" * 32)

    def rewrite():
        with open(os.path.join(lock, "owner.json"), "w") as f:
            json.dump({"pid": 8, "pid_start": LIVE_START, "token": "2" * 32}, f)

    r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(alive={8}, hooks={"in_mutex": rewrite}))
    assert r.status == "held"
    assert r.note is None
    assert dirlock.read_owner(lock)["token"] == "2" * 32


def test_l5_moved_lock_not_the_judged_one_is_put_back(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    owner(lock, pid=7, token="1" * 32)
    warned = []

    def swap(stale):
        with open(os.path.join(stale, "owner.json"), "w") as f:
            json.dump({"pid": 8, "pid_start": LIVE_START, "token": "3" * 32}, f)

    e = env_with(alive={8}, hooks={"after_break_rename": swap})
    e.warn = warned.append
    r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=e)
    assert r.status == "held"
    assert "put it back" in r.note
    assert dirlock.read_owner(lock)["token"] == "3" * 32
    assert warned


def test_l6_mutex_rules(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    owner(lock, pid=7)
    os.mkdir(lock + ".break")
    now = time.time() * 1000
    assert dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(now=lambda: now)).status == "breaking"
    for later in (now + 31000, now - 31000):
        r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(now=lambda: later))
        assert r.status == "acquired"
        dirlock.release(lock, r.token)
        owner(lock, pid=7)
        os.mkdir(lock + ".break")


def test_l7_contended(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    n = []

    def keep_changing():
        n.append(1)
        dirlock.release(lock, dirlock.read_owner(lock)["token"])
        owner(lock, pid=9, token="%032x" % len(n))

    owner(lock, pid=9, token="0" * 32)
    r = dirlock.acquire(lock, {"pid": 1, "pid_start": "s"}, env=env_with(hooks={"in_mutex": keep_changing}))
    assert (r.status, r.note) == ("held", "contended")
    assert len(n) == dirlock.TRIES


def test_l8_leftovers(tmp_path):
    lock = str(tmp_path / "state" / "heavy.lock.d")
    parent = os.path.dirname(lock)
    os.makedirs(parent)
    names = {"heavy.lock.d.new.0a1b2c3d": True, "heavy.lock.d.stale.ff": True,
             "heavy.lock.d.gone.12345678": True, "heavy.lock.d.new.0A1B": False,
             "heavy.lock.d.old.12": False, "heavy.lock.d.break": False,
             "heavy.lock.d.stale.live0": False}
    for n in names:
        os.makedirs(os.path.join(parent, n))
    owner(os.path.join(parent, "heavy.lock.d.stale.live0"), pid=5)
    old = time.time() - 120
    for n in names:
        os.utime(os.path.join(parent, n), (old, old))
    os.makedirs(os.path.join(parent, "heavy.lock.d.gone.abcd"))
    removed = dirlock.clean_leftovers(lock, env_with(alive={5}))
    assert sorted(removed) == sorted(n for n, gone in names.items() if gone)
    assert os.path.exists(os.path.join(parent, "heavy.lock.d.gone.abcd"))


def test_l9_owner_parsing(tmp_path):
    cases = [({"pid": True, "pid_start": "s", "token": "t"}, False),
             ({"pid": "1", "pid_start": "s", "token": "t"}, False),
             ({"pid": 1, "pid_start": "s"}, False),
             ({"pid": 4242.0, "pid_start": "s", "token": "t"}, True),
             ({"pid": 1, "pid_start": "s", "token": "t", "extra": {"k": 1}}, True)]
    for i, (o, ok) in enumerate(cases):
        d = tmp_path / ("o%d" % i)
        d.mkdir()
        (d / "owner.json").write_text(json.dumps(o))
        got = dirlock.read_owner(str(d))
        assert (got is not None) == ok, o
        if ok and "extra" in o:
            assert got["extra"] == {"k": 1}
    d = tmp_path / "nan"
    d.mkdir()
    (d / "owner.json").write_text('{"pid": NaN, "pid_start": "s", "token": "t"}')
    assert dirlock.read_owner(str(d)) is None
    for pid in (0, -3, 1.5, 4242.5):
        assert dirlock.proc_start(pid) is None
    assert dirlock.liveness({"pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid())}, dirlock.LockEnv())[0]


def test_l10_lstart():
    assert dirlock.parse_lstart("Mon Oct  5 01:02:03 2026") == 1791162123000
    assert dirlock.parse_lstart("Mon Foo  5 01:02:03 2026") is None
    assert dirlock.parse_lstart("Mon Feb 30 01:02:03 2026") is None
    assert dirlock.parse_lstart("Mon Oct ٥ 01:02:03 2026") is None
    a = "Mon Oct 19 20:06:47 2026"
    assert dirlock.same_start(a, "Mon Oct 19 20:06:49 2026", "linux")
    assert not dirlock.same_start(a, "Mon Oct 19 20:06:50 2026", "linux")
    assert not dirlock.same_start(a, "Mon Oct 19 20:06:48 2026", "darwin")


def test_owner_json_has_the_s2v_shape(tmp_path):
    """The same owner.json fields and order as s2v's acquireDir writes (kind, pid, pid_start,
    holder, token), one line with a newline."""
    lock = str(tmp_path / "heavy.lock.d")
    me = {"kind": "gen", "pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid()), "holder": "limner run x y"}
    r = dirlock.acquire(lock, me)
    body = open(os.path.join(lock, "owner.json")).read()
    assert body.count("\n") == 1 and body.endswith("\n")
    assert list(json.loads(body)) == ["kind", "pid", "pid_start", "holder", "token"]
    dirlock.release(lock, r.token)


RACER = r"""
import os, sys, time
from limner import dirlock
lock, barrier, log, loops, name = sys.argv[1:]
me = {"kind": "gen", "pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid()), "holder": name}
open(log, "a").write("ready %s\n" % name)
while not os.path.exists(barrier):
    time.sleep(0.005)
for _ in range(int(loops)):
    while True:
        r = dirlock.acquire(lock, me)
        if r.status == "acquired":
            break
        time.sleep(0.002)
    with open(log, "a") as f:
        f.write("enter %s\n" % name); f.flush()
    time.sleep(0.02)
    still = dirlock.read_owner(lock)["token"] == r.token
    with open(log, "a") as f:
        f.write("exit %s stillmine=%s\n" % (name, still)); f.flush()
    dirlock.release(lock, r.token)
"""


def test_race_of_real_processes_never_overlaps(tmp_path):
    lock = str(tmp_path / "heavy.lock.d")
    barrier, log = str(tmp_path / "go"), str(tmp_path / "race.log")
    kids = [subprocess.Popen([sys.executable, "-c", RACER, lock, barrier, log, "5", "p%d" % i]) for i in range(6)]
    end = time.time() + 30
    while time.time() < end:
        if os.path.exists(log) and open(log).read().count("ready ") == 6:
            break
        time.sleep(0.02)
    open(barrier, "w").close()
    for k in kids:
        assert k.wait(timeout=120) == 0
    inside = 0
    for line in open(log).read().splitlines():
        if line.startswith("enter "):
            inside += 1
            assert inside == 1, "two holders at once"
        elif line.startswith("exit "):
            assert line.endswith("stillmine=True")
            inside -= 1
