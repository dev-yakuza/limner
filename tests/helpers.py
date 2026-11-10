import json
import os
import signal
import subprocess
import sys
import time

from limner import dirlock


def write_job(root, jobs, name="job.json"):
    p = os.path.join(str(root), name)
    with open(p, "w") as f:
        json.dump({"version": 1, "model": "qwen-image-edit-2511", "jobs": jobs}, f)
    return p


def job(jid, out=None, refs=(), w=64, h=64, prompt=None, seed=7):
    return {"id": jid, "out": out or "out/%s.png" % jid, "width": w, "height": h,
            "prompt": prompt or "picture %s" % jid, "refs": list(refs), "seed": seed}


def wait_for(cond, timeout=20, step=0.02, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting for %s" % what)


def popen_lmn(*argv, env=None, preexec_fn=None):
    return subprocess.Popen([sys.executable, "-m", "limner", *map(str, argv)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env or os.environ, preexec_fn=preexec_fn)


def result_of(p, timeout=60):
    out, err = p.communicate(timeout=timeout)
    lines = out.strip().splitlines()
    return (json.loads(lines[-1]) if lines else None), p.returncode, err


def run_dir(root):
    runs = os.path.join(str(root), ".limner", "runs")
    names = os.listdir(runs) if os.path.isdir(runs) else []
    return os.path.join(runs, names[0]) if len(names) == 1 else None


def attempt_dirs(root):
    rd = run_dir(root)
    if not rd or not os.path.isdir(os.path.join(rd, "attempts")):
        return []
    return [os.path.join(rd, "attempts", a) for a in sorted(os.listdir(os.path.join(rd, "attempts")))]


def read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def alive(pid, start):
    from limner import util
    return util.alive(pid, start)


def kill(pid, sig=signal.SIGKILL):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def hold_machine_lock(state, kind="tts", holder="s2v tts"):
    """Hold the machine lock as this (live) test process."""
    lock = os.path.join(str(state), "heavy.lock.d")
    me = {"kind": kind, "pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid()), "holder": holder}
    r = dirlock.acquire(lock, me)
    assert r.status == "acquired"
    return lock, r.token


class Gates:
    """LIMNER_TEST_GATES: `lmn run` stops at a named point until released."""

    def __init__(self, tmp, names):
        self.dir = os.path.join(str(tmp), "gates")
        os.makedirs(self.dir, exist_ok=True)
        self.names = names

    def env(self, base=None):
        e = dict(base or os.environ)
        e["LIMNER_TEST_GATES"] = self.dir
        e["LIMNER_TEST_GATE_NAMES"] = ",".join(self.names)
        return e

    def reached(self, name, timeout=20):
        wait_for(lambda: os.path.exists(os.path.join(self.dir, name + ".reached")), timeout, what=name)

    def release(self, name):
        open(os.path.join(self.dir, name + ".go"), "w").close()
