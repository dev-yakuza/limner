"""The run / status / cancel process model with the fake generator (spec §6.1, §6.5, §12)."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import lmn
from helpers import (Gates, alive, attempt_dirs, hold_machine_lock, job, kill, popen_lmn, read, result_of,
                     wait_for, write_job)
from limner import dirlock


def start(env, jobs, *extra, check=0):
    jf = write_job(env["root"], jobs)
    obj, code = lmn("run", "--job", jf, "--root", env["root"], *extra)
    assert code == check, (code, obj)
    return obj


def status(env, obj, wait=0):
    return lmn("status", "--run-id", obj["run_id"], "--attempt", obj["attempt"], "--root", env["root"],
               "--wait-seconds", wait)


def machine_lock(env):
    return os.path.join(str(env["state"]), "heavy.lock.d")


# --------------------------------------------------------------------------- basics


def test_run_status_done_then_nothing_to_do(env):
    obj = start(env, [job("ref", w=64, h=80), job("p1", refs=["out/ref.png"])])
    assert obj["state"] == "starting" and obj["status_file"].endswith("status.json")
    st, code = status(env, obj, wait=30)
    assert (st["state"], code) == ("done", 0), st
    assert [d["id"] for d in st["done"]] == ["ref", "p1"]
    assert st["generator"] == "fake" and st["stages"] == 2
    rec = read(os.path.join(env["root"], ".limner", "records", "out", "p1.png.json"))
    assert rec["kind"] == "generate" and rec["generator"] == "fake"
    assert rec["refs"][0]["path"] == "out/ref.png" and rec["letterbox"] == 1 and rec["blank"] == "#E0E0E0"
    assert not os.path.exists(machine_lock(env))
    again, code = lmn("run", "--job", os.path.join(env["root"], "job.json"), "--root", env["root"])
    assert code == 0 and again["attempt"] is None and again["state"] == "done"
    assert sorted(again["skipped"]) == ["p1", "ref"]
    assert len(attempt_dirs(env["root"])) == 1  # nothing to do: no new attempt folder


def test_check_validates_only(env):
    jf = write_job(env["root"], [job("a")])
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--check")
    assert code == 0 and obj["check"] == "ok" and obj["jobs"] == 1
    assert not os.path.exists(os.path.join(env["root"], ".limner"))


@pytest.mark.parametrize("bad,needle", [
    ({"width": 65}, "multiple of 16"),
    ({"width": 1040}, "multiple of 16"),
    ({"seed": -1}, "seed"),
    ({"prompt": ""}, "prompt"),
    ({"prompt": "x" * 1501}, "prompt"),
    ({"refs": ["a.png", "b.png", "c.png"]}, "refs"),
    ({"out": "out/a.jpg"}, ".png"),
    ({"out": "../a.png"}, "outside"),
    ({"out": ".git/a.png"}, ".git"),
    ({"refs": ["nope.png"]}, "not a file"),
    ({"extra": 1}, "unknown"),
])
def test_bad_jobs_are_exit_2(env, bad, needle):
    j = job("a")
    j.update(bad)
    jf = write_job(env["root"], [j])
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--check")
    assert code == 2 and needle in obj["error"], obj


def test_force_accept_overlap_and_token_are_exit_2(env, monkeypatch):
    jf = write_job(env["root"], [job("a")])
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--force", "a", "--accept", "a")
    assert code == 2
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--wait", "541")
    assert code == 2
    monkeypatch.setenv("S2V_HEAVY_LOCK_TOKEN", "x" * 32)
    obj, code = lmn("run", "--job", jf, "--root", env["root"])
    assert code == 2 and "S2V_HEAVY_LOCK_TOKEN" in obj["error"]


def test_status_and_cancel_of_unknown_run_are_12(env):
    obj, code = lmn("status", "--run-id", "0" * 16, "--root", env["root"])
    assert code == 12
    obj, code = lmn("cancel", "--run-id", "0" * 16, "--root", env["root"])
    assert code == 12


def test_run_writes_result_file(env):
    jf = write_job(env["root"], [job("a")])
    res = os.path.join(env["root"], "res.json")
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--result", res)
    assert code == 0 and read(res) == obj
    status(env, obj, wait=30)


# --------------------------------------------------------------------------- busy


def test_busy_lock_is_73_with_holder_and_one_attempt(env):
    lock, tok = hold_machine_lock(env["state"])
    try:
        obj = start(env, [job("a")], check=73)
        assert obj["state"] == "busy" and obj["busy"]["reason"] == "lock"
        assert obj["busy"]["holder"]["holder"] == "s2v tts"
        st, code = status(env, obj)
        assert (st["state"], code) == ("busy", 73)
        start(env, [job("a")], check=73)
        assert len(attempt_dirs(env["root"])) == 2
    finally:
        dirlock.release(lock, tok)
    obj = start(env, [job("a")])
    assert status(env, obj, wait=30)[1] == 0


def test_busy_holder_names_a_limner_run(env):
    lock, tok = hold_machine_lock(env["state"], kind="gen", holder="limner run abcdef0123456789 /x/status.json")
    try:
        obj = start(env, [job("a")], check=73)
        h = obj["busy"]["holder"]
        assert h["run_id"] == "abcdef0123456789" and h["status_file"] == "/x/status.json"
    finally:
        dirlock.release(lock, tok)


def test_resident_server_is_busy_and_the_lock_is_released(env):
    os.makedirs(env["state"], exist_ok=True)
    with open(os.path.join(env["state"], "resident.json"), "w") as f:
        json.dump({"pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid())}, f)
    obj = start(env, [job("a")], check=73)
    assert obj["busy"]["reason"] == "resident"
    assert not os.path.exists(machine_lock(env))
    with open(os.path.join(env["state"], "resident.json"), "w") as f:
        json.dump({"pid": 999999, "pid_start": "Mon Oct 19 20:06:47 2026"}, f)  # dead: not busy, kept
    obj = start(env, [job("a")])
    assert status(env, obj, wait=30)[1] == 0
    assert os.path.exists(os.path.join(env["state"], "resident.json"))


def test_orphan_server_is_busy(env, monkeypatch):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("S2V_ALIGNER_PORT", str(srv.server_address[1]))
        obj = start(env, [job("a")], check=73)
        assert obj["busy"]["reason"] == "orphan"
    finally:
        srv.shutdown()


def test_min_free_memory_is_busy_only_when_asked(env, monkeypatch):
    monkeypatch.setenv("LIMNER_MIN_FREE_GIB", "1000000")
    obj = start(env, [job("a")], check=73)
    assert obj["busy"]["reason"] == "memory"


# --------------------------------------------------------------------------- failures


def test_child_abort_is_metal_and_the_rest_pending(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_ABORT_ID", "a")
    obj = start(env, [job("a"), job("b"), job("c", refs=["out/a.png"])])
    st, code = status(env, obj, wait=30)
    assert (st["state"], code) == ("failed", 11), st
    assert st["failed"] == [{"id": "a", "reason": "metal", "message": st["failed"][0]["message"]}]
    assert set(st["pending"]) == {"b", "c"}
    assert not os.path.exists(machine_lock(env))
    monkeypatch.delenv("LIMNER_FAKE_ABORT_ID")
    obj = start(env, [job("a"), job("b"), job("c", refs=["out/a.png"])])  # resumes
    st, code = status(env, obj, wait=30)
    assert code == 0 and len(st["done"]) == 3


def test_memory_watch_kills_the_child(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_FOOTPRINT_GIB", "1000")
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "8")
    obj = start(env, [job("a")])
    st, code = status(env, obj, wait=40)
    assert code == 11 and st["failed"][0]["reason"] == "memory", st


def test_job_error_fails_that_job_and_blocks_its_dependents(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_ERROR_ID", "a")
    obj = start(env, [job("a"), job("b"), job("c", refs=["out/a.png"])])
    st, code = status(env, obj, wait=30)
    assert code == 11 and st["state"] == "done"
    assert [f["id"] for f in st["failed"]] == ["a"] and [d["id"] for d in st["done"]] == ["b"]
    assert st["blocked"] == [{"id": "c", "by": "a", "cause": "failed"}]


def test_environment_failure_is_2(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_ENV_ERROR", "1")
    obj = start(env, [job("a")])
    st, code = status(env, obj, wait=30)
    assert code == 2 and st["failed"][0]["reason"] == "environment"


# --------------------------------------------------------------------------- cancel


def test_cancel_running_kills_worker_and_child(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "60")
    obj = start(env, [job("a"), job("b")])
    att = os.path.dirname(obj["status_file"])
    wait_for(lambda: (read(obj["status_file"]) or {}).get("state") == "running" and
             os.path.exists(os.path.join(att, "child.json")), what="running")
    t = time.time()
    st, code = lmn("cancel", "--run-id", obj["run_id"], "--root", env["root"])
    assert (st["state"], code) == ("cancelled", 0), st
    assert time.time() - t < 8  # SIGTERM was enough (signals reset in worker and child)
    assert st["failed"] == [{"id": "a", "reason": "cancelled", "message": "cancelled by lmn cancel"}]
    assert st["pending"] == ["b"]
    w, c = read(os.path.join(att, "spawn.json")), read(os.path.join(att, "child.json"))
    assert not alive(w["pid"], w["pid_start"]) and not alive(c["pid"], c["pid_start"])
    st2, code = lmn("cancel", "--run-id", obj["run_id"], "--root", env["root"])
    assert (st2["state"], code) == ("cancelled", 11)  # already final: its own code (cancel step 1)
    # the dead worker's locks are broken by the next taker
    monkeypatch.delenv("LIMNER_FAKE_SECONDS")
    obj = start(env, [job("a"), job("b")])
    assert status(env, obj, wait=30)[1] == 0


def test_cancel_during_busy_wait(env):
    lock, tok = hold_machine_lock(env["state"])
    try:
        jf = write_job(env["root"], [job("a")])
        p = popen_lmn("run", "--job", jf, "--root", env["root"], "--wait", "120")
        wait_for(lambda: attempt_dirs(env["root"]) and
                 (read(os.path.join(attempt_dirs(env["root"])[0], "status.json")) or {}).get("state") == "starting",
                 what="starting")
        rid = os.path.basename(os.path.dirname(os.path.dirname(attempt_dirs(env["root"])[0])))
        st, code = lmn("cancel", "--run-id", rid, "--root", env["root"])
        assert (st["state"], code) == ("cancelled", 0), st
        obj, rc, _ = result_of(p)
        assert (obj["state"], rc) == ("cancelled", 11)
        assert len(attempt_dirs(env["root"])) == 1
    finally:
        dirlock.release(lock, tok)


def test_request_then_run_sigkilled_in_busy_wait_is_cancelled(env, tmp_path):
    """§12: a request during the busy wait, then `run` SIGKILLed: `status` says cancelled."""
    lock, tok = hold_machine_lock(env["state"])
    g = Gates(tmp_path, ["busy_wait"])
    try:
        jf = write_job(env["root"], [job("a")])
        p = popen_lmn("run", "--job", jf, "--root", env["root"], "--wait", "120", env=g.env())
        g.reached("busy_wait")
        rid = os.path.basename(os.path.dirname(os.path.dirname(attempt_dirs(env["root"])[0])))
        st, code = lmn("cancel", "--run-id", rid, "--root", env["root"], "--wait-seconds", "0")
        assert code == 10  # the request stays
        p.kill()
        p.wait()
        st, code = lmn("status", "--run-id", rid, "--root", env["root"])
        assert (st["state"], code) == ("cancelled", 11)
        st, code = lmn("cancel", "--run-id", rid, "--root", env["root"])
        assert (st["state"], code) == ("cancelled", 0)
        assert read(os.path.join(attempt_dirs(env["root"])[0], "status.json"))["state"] == "cancelled"
    finally:
        dirlock.release(lock, tok)


def test_run_sigkilled_without_request_is_crashed(env, tmp_path):
    lock, tok = hold_machine_lock(env["state"])
    g = Gates(tmp_path, ["busy_wait"])
    try:
        jf = write_job(env["root"], [job("a")])
        p = popen_lmn("run", "--job", jf, "--root", env["root"], "--wait", "120", env=g.env())
        g.reached("busy_wait")
        p.kill()
        p.wait()
        rid = os.path.basename(os.path.dirname(os.path.dirname(attempt_dirs(env["root"])[0])))
        st, code = lmn("status", "--run-id", rid, "--root", env["root"])
        assert (st["state"], code) == ("crashed", 11)
    finally:
        dirlock.release(lock, tok)


def test_generate_supervisor_sigkill_is_crashed_not_a_live_zombie(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "60")
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("generate", "--job", jf, "--root", env["root"])
    sp = wait_for(lambda: attempt_dirs(env["root"]) and read(os.path.join(attempt_dirs(env["root"])[0], "child.json"))
                  and read(os.path.join(attempt_dirs(env["root"])[0], "spawn.json")), what="spawn")
    kill(sp["pid"])
    obj, rc, err = result_of(p, timeout=60)
    assert (obj["state"], rc) == ("crashed", 11), (obj, err)
    c = read(os.path.join(attempt_dirs(env["root"])[0], "child.json"))
    wait_for(lambda: not alive(c["pid"], c["pid_start"]), what="child exits by its lifeline")


def test_orphan_child_is_reported_and_cancel_kills_it(env, monkeypatch):
    """The supervisor dies but its child lives (lifeline not yet closed): orphan_child."""
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "60")
    obj = start(env, [job("a")])
    att = os.path.dirname(obj["status_file"])
    c = wait_for(lambda: read(os.path.join(att, "child.json")), what="child")
    w = read(os.path.join(att, "spawn.json"))
    # keep the child alive past its supervisor: stop it so it cannot see the lifeline close
    os.kill(c["pid"], signal.SIGSTOP)
    kill(w["pid"])
    wait_for(lambda: not alive(w["pid"], w["pid_start"]), what="worker gone")
    st, code = status(env, obj)
    assert (st["state"], code) == ("orphan_child", 11)
    st, code = lmn("cancel", "--run-id", obj["run_id"], "--root", env["root"])
    assert (st["state"], code) == ("cancelled", 0)
    assert not alive(c["pid"], c["pid_start"])


# --------------------------------------------------------------------------- signals


def _blocked_parent():
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def test_worker_and_child_reset_signals_even_from_a_blocking_parent(env, monkeypatch, tmp_path):
    rep = tmp_path / "sig"
    rep.mkdir()
    monkeypatch.setenv("LIMNER_TEST_SIGNAL_REPORT", str(rep))
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "60")
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], preexec_fn=_blocked_parent)
    obj, rc, err = result_of(p)
    assert rc == 0, err
    att = os.path.dirname(obj["status_file"])
    wait_for(lambda: read(os.path.join(att, "child.json")), what="child")
    reports = wait_for(lambda: len(os.listdir(rep)) >= 2 and [read(os.path.join(rep, n)) for n in os.listdir(rep)],
                       what="reports")
    kinds = {os.path.basename(r["argv"][0]) for r in reports}
    assert kinds == {"worker.py", "child.py"}, kinds
    for r in reports:
        assert r["blocked"] == [] and r["term_default"] and r["int_default"], r
    t = time.time()
    st, code = lmn("cancel", "--run-id", obj["run_id"], "--root", env["root"])
    assert code == 0 and time.time() - t < 8


def test_signal_during_popen_kills_the_new_worker(env, monkeypatch):
    """A SIGTERM while Popen is still forking (preexec_fn delay widens the window)."""
    monkeypatch.setenv("LIMNER_TEST_PREEXEC_SLEEP", "1.5")
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"])
    me_lock = machine_lock(env)
    wait_for(lambda: (dirlock.read_owner(me_lock) or {}).get("pid") == p.pid, what="run holds the machine lock")
    time.sleep(0.4)
    p.send_signal(signal.SIGTERM)
    obj, rc, err = result_of(p)
    assert rc == 11 and obj["state"] == "failed" and obj["failed"][0]["reason"] == "signal:15", (obj, err)
    att = attempt_dirs(env["root"])[0]
    # the worker's argv carries the attempt id: no process with it may survive (spawn.json may not exist)
    left = subprocess.run(["pgrep", "-f", "limner.worker .* %s" % os.path.basename(att)], capture_output=True,
                          text=True).stdout.split()
    assert left == [], left
    assert not os.path.exists(me_lock)
    st, code = lmn("status", "--run-id", obj["run_id"], "--root", env["root"])
    assert (st["state"], code) == ("failed", 11)


def test_signal_before_go_closes_signal_and_kills_worker(env, tmp_path):
    g = Gates(tmp_path, ["before_go"])
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=g.env())
    g.reached("before_go")
    sp = read(os.path.join(attempt_dirs(env["root"])[0], "spawn.json"))
    p.send_signal(signal.SIGINT)
    obj, rc, err = result_of(p)
    assert (rc, obj["failed"][0]["reason"]) == (11, "signal:2"), (obj, err)
    assert not alive(sp["pid"], sp["pid_start"])
    # the handed locks belong to the dead worker: the next run breaks them
    obj = start(env, [job("a")])
    assert status(env, obj, wait=30)[1] == 0


# --------------------------------------------------------------------------- hand-off races


def test_handoff_new_token_beats_a_breaker_that_read_run(env, tmp_path):
    """§12: the breaker reads owner = run; then the hand-off and run's exit happen; then the
    breaker's ps. It must end `held` by the live worker, for both locks."""
    for lockname in ("machine", "run"):
        g = Gates(tmp_path / lockname, ["after_spawn"])
        root = env["root"] / lockname
        root.mkdir()
        jf = write_job(root, [job("a")])
        e = g.env()
        e["LIMNER_FAKE_SECONDS"] = "3"
        p = popen_lmn("run", "--job", jf, "--root", root, env=e)
        g.reached("after_spawn")
        att = attempt_dirs(root)[0]
        sp = read(os.path.join(att, "spawn.json"))
        path = machine_lock(env) if lockname == "machine" else os.path.join(os.path.dirname(os.path.dirname(att)),
                                                                            "run.lock.d")
        fired = []

        def ps(pid):
            if pid == p.pid and not fired:
                fired.append(1)
                g.release("after_spawn")
                p.wait()  # hand-off done, run gone
            return dirlock.proc_start(pid)

        me = {"kind": "tts", "pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid()), "holder": "breaker"}
        b = dirlock.acquire(path, me, env=dirlock.LockEnv(proc_start=ps))
        assert fired and b.status == "held", (lockname, b.status)
        assert b.owner["pid"] == sp["pid"]
        st, code = lmn("status", "--run-id", os.path.basename(os.path.dirname(os.path.dirname(att))),
                       "--root", root, "--wait-seconds", "30")
        assert (st["state"], code) == ("done", 0), st


def _sleeper(new_session=False):
    return subprocess.Popen(["sleep", "60"], start_new_session=new_session)


@pytest.mark.parametrize("variant", ["limner", "same_token", "in_place"])
def test_handoff_variants_against_a_racing_breaker(tmp_path, variant):
    """The race test catches broken hand-offs: same token (R5-1) or an in-place rewrite."""
    from limner import lock as lk

    lock = str(tmp_path / "heavy.lock.d")
    R, W = _sleeper(), _sleeper(True)
    try:
        time.sleep(0.1)
        rs, ws = dirlock.proc_start(R.pid), dirlock.proc_start(W.pid)
        r = dirlock.acquire(lock, {"kind": "gen", "pid": R.pid, "pid_start": rs, "holder": "limner run"})
        wf = {"kind": "gen", "pid": W.pid, "pid_start": ws, "holder": "limner run"}
        fired = []

        def ps(pid):
            if pid == R.pid and not fired:
                fired.append(1)
                if variant == "limner":
                    assert lk.hand_over(lock, r.token, wf)
                elif variant == "same_token":
                    tmp = os.path.join(lock, "o.tmp")
                    with open(tmp, "w") as f:
                        f.write(dirlock.owner_body(wf, r.token))
                    os.rename(tmp, os.path.join(lock, "owner.json"))
                else:  # in place, and a reader sees it empty for a moment
                    with open(os.path.join(lock, "owner.json"), "w") as f:
                        pass
                R.kill()
                R.wait()
            return dirlock.proc_start(pid)

        me = {"kind": "tts", "pid": os.getpid(), "pid_start": dirlock.proc_start(os.getpid()), "holder": "s2v"}
        b = dirlock.acquire(lock, me, env=dirlock.LockEnv(proc_start=ps, sleep=lambda ms: None))
        owner = dirlock.read_owner(lock)
        worker_owns = b.status == "held" and owner and owner["pid"] == W.pid
        assert worker_owns == (variant == "limner"), (variant, b.status, owner)
    finally:
        for p in (R, W):
            p.kill()
            p.wait()


def test_run_dies_between_handoffs(env, tmp_path):
    g = Gates(tmp_path, ["between_handoffs"])
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=g.env())
    g.reached("between_handoffs")
    att = attempt_dirs(env["root"])[0]
    sp = read(os.path.join(att, "spawn.json"))
    p.kill()
    p.wait()
    wait_for(lambda: not alive(sp["pid"], sp["pid_start"]), what="worker ends on EOF")
    rid = os.path.basename(os.path.dirname(os.path.dirname(att)))
    st, code = lmn("status", "--run-id", rid, "--root", env["root"])
    assert (st["state"], code) == ("crashed", 11)
    assert read(os.path.join(att, "status.json"))["state"] == "starting"  # the worker wrote nothing
    obj = start(env, [job("a")])  # both stale locks are broken
    assert status(env, obj, wait=30)[1] == 0


def test_run_dies_before_go(env, tmp_path):
    g = Gates(tmp_path, ["before_go"])
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=g.env())
    g.reached("before_go")
    att = attempt_dirs(env["root"])[0]
    sp = read(os.path.join(att, "spawn.json"))
    p.kill()
    p.wait()
    wait_for(lambda: not alive(sp["pid"], sp["pid_start"]), what="worker ends on EOF")
    rid = os.path.basename(os.path.dirname(os.path.dirname(att)))
    assert lmn("status", "--run-id", rid, "--root", env["root"])[0]["state"] == "crashed"
    obj = start(env, [job("a")])
    assert status(env, obj, wait=30)[1] == 0


def test_worker_dies_before_go_epipe(env, tmp_path):
    g = Gates(tmp_path, ["before_go"])
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=g.env())
    g.reached("before_go")
    att = attempt_dirs(env["root"])[0]
    sp = read(os.path.join(att, "spawn.json"))
    kill(sp["pid"])
    wait_for(lambda: not alive(sp["pid"], sp["pid_start"]), what="worker gone")
    g.release("before_go")
    obj, rc, err = result_of(p)
    assert (obj["state"], rc) == ("starting", 0), err
    assert lmn("status", "--run-id", obj["run_id"], "--root", env["root"])[0]["state"] == "crashed"


@pytest.mark.parametrize("gate", ["after_spawn", "between_handoffs"])
def test_cancel_while_handing_over(env, tmp_path, gate):
    """`cancel` reads the attempt while `run` is mid hand-off; run then finishes and exits. The
    request file decides: the worker closes the attempt as cancelled."""
    g = Gates(tmp_path, [gate])
    jf = write_job(env["root"], [job("a")])
    e = g.env()
    e["LIMNER_FAKE_SECONDS"] = "30"
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=e)
    g.reached(gate)
    att = attempt_dirs(env["root"])[0]
    rid = os.path.basename(os.path.dirname(os.path.dirname(att)))
    c = popen_lmn("cancel", "--run-id", rid, "--root", env["root"], "--wait-seconds", "60")
    wait_for(lambda: os.path.exists(os.path.join(att, "cancel_requested")), what="request")
    g.release(gate)
    obj, rc, _ = result_of(p)
    assert rc == 0 and obj["state"] == "starting"
    st, code, err = result_of(c, timeout=90)
    assert (st["state"], code) == ("cancelled", 0), (st, err)
    sp = read(os.path.join(att, "spawn.json"))
    wait_for(lambda: not alive(sp["pid"], sp["pid_start"]), what="worker ends")
    assert not os.path.exists(machine_lock(env))


def test_second_run_of_the_same_run_id_reports_the_first(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "5")
    obj = start(env, [job("a")])
    again, code = lmn("run", "--job", os.path.join(env["root"], "job.json"), "--root", env["root"])
    assert code == 0 and again["attempt"] == obj["attempt"] and again["state"] in ("starting", "running")
    assert status(env, obj, wait=30)[1] == 0


# --------------------------------------------------------------------------- review round 1


def test_child_dying_after_its_last_report_is_not_success(env, monkeypatch):
    """R1-1: the child reports `done` for the last job of its stage, then dies by a signal."""
    monkeypatch.setenv("LIMNER_FAKE_ABORT_AFTER_ID", "a")
    obj = start(env, [job("a"), job("b", refs=["out/a.png"])])
    st, code = status(env, obj, wait=30)
    assert (st["state"], code) == ("failed", 11), st
    assert [d["id"] for d in st["done"]] == ["a"] and st["pending"] == ["b"]
    assert st["failed"] and st["failed"][0]["id"] is None and st["failed"][0]["reason"] == "metal"


def test_cancel_does_not_close_while_run_is_alive_before_go(env, tmp_path):
    """R1-5: the worker died before go, `run` still lives: cancel must not write over run."""
    g = Gates(tmp_path, ["before_go"])
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=g.env())
    g.reached("before_go")
    att = attempt_dirs(env["root"])[0]
    rid = os.path.basename(os.path.dirname(os.path.dirname(att)))
    sp = read(os.path.join(att, "spawn.json"))
    kill(sp["pid"])
    wait_for(lambda: not alive(sp["pid"], sp["pid_start"]), what="worker gone")
    st, code = lmn("cancel", "--run-id", rid, "--root", env["root"], "--wait-seconds", "1")
    assert code == 10, st  # run is still a writer
    p.send_signal(signal.SIGTERM)
    obj, rc, _ = result_of(p)
    assert rc == 11 and obj["failed"][0]["reason"] == "signal:15"
    st, code = lmn("status", "--run-id", rid, "--root", env["root"])
    assert st["state"] == "failed"


def test_cancel_leaves_a_crashed_attempt_crashed(env, tmp_path):
    """R1-6: all writers gone without a request: cancel reports crashed, not cancelled."""
    g = Gates(tmp_path, ["before_go"])
    jf = write_job(env["root"], [job("a")])
    p = popen_lmn("run", "--job", jf, "--root", env["root"], env=g.env())
    g.reached("before_go")
    att = attempt_dirs(env["root"])[0]
    rid = os.path.basename(os.path.dirname(os.path.dirname(att)))
    sp = read(os.path.join(att, "spawn.json"))
    p.kill()
    p.wait()
    wait_for(lambda: not alive(sp["pid"], sp["pid_start"]), what="worker ends on EOF")
    st, code = lmn("cancel", "--run-id", rid, "--root", env["root"])
    assert (st["state"], code) == ("crashed", 11)
    assert read(os.path.join(att, "status.json"))["state"] == "starting"


def test_state_dir_under_git_or_claude_is_refused(env):
    """R1-2."""
    jf = write_job(env["root"], [job("a")])
    for sd in (".git/hooks", ".Claude/x", "sub/.git"):
        obj, code = lmn("run", "--job", jf, "--root", env["root"], "--state-dir", os.path.join(env["root"], sd))
        assert code == 2 and ".git" in obj["error"], obj
    assert not os.path.exists(os.path.join(env["root"], ".git"))


def test_job_and_result_paths_stay_in_root(env):
    """R1-13."""
    jf = write_job(env["tmp"], [job("a")])
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--check")
    assert code == 2 and "outside" in obj["error"]
    jf = write_job(env["root"], [job("a")])
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--check", "--result", os.path.join(env["tmp"], "r.json"))
    assert code == 2 and not os.path.exists(os.path.join(env["tmp"], "r.json"))


def test_signal_during_a_slow_health_check_is_seen_at_once(env, monkeypatch):
    """R1-11: a port that accepts but never answers; the health wait is split, so SIGTERM ends
    `run` quickly and the machine lock is released."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    try:
        monkeypatch.setenv("S2V_TTS_PORT", str(srv.getsockname()[1]))
        jf = write_job(env["root"], [job("a")])
        p = popen_lmn("run", "--job", jf, "--root", env["root"])
        wait_for(lambda: (dirlock.read_owner(machine_lock(env)) or {}).get("pid") == p.pid, what="lock")
        t = time.time()
        p.send_signal(signal.SIGTERM)
        obj, rc, err = result_of(p)
        assert rc == 11 and obj["failed"][0]["reason"] == "signal:15", (obj, err)
        assert time.time() - t < 1.0
        assert not os.path.exists(machine_lock(env))
    finally:
        srv.close()


def test_generate_ctrl_c_cancels(env, monkeypatch):
    """R1-12: Ctrl-C (SIGINT to the whole group) while generating ends in cancelled with JSON."""
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "60")
    jf = write_job(env["root"], [job("a")])
    p = subprocess.Popen([sys.executable, "-m", "limner", "generate", "--job", jf, "--root", env["root"]],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    wait_for(lambda: attempt_dirs(env["root"]) and read(os.path.join(attempt_dirs(env["root"])[0], "child.json")),
             what="child")
    os.killpg(p.pid, signal.SIGINT)
    obj, rc, err = result_of(p, timeout=90)
    assert (obj["state"], rc) == ("cancelled", 0), (obj, err)
