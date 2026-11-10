"""`lmn run` (spec §6.1 steps 1–9): validate, preview, take the run lock and the machine lock,
start the detached worker and hand both locks over to it, then return at once."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time

from . import dirlock, jobs, lock, models, paths, plan, result, runs, util
from .util import Fail, bad

MAX_WAIT = 540


class Stop(Exception):
    """Ends `run` with this result and code (signal handling, busy, cancel)."""

    def __init__(self, obj: dict, code: int):
        super().__init__(obj.get("state"))
        self.obj = obj
        self.code = code


def fake() -> bool:
    return os.environ.get("LIMNER_FAKE_GENERATOR") == "1"


def parse_ids(v: str | None) -> list[str]:
    if not v:
        return []
    return [x for x in (s.strip() for s in v.split(",")) if x]


def normalize(obj):
    if isinstance(obj, str):
        return util.nfc(obj)
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {util.nfc(k): normalize(v) for k, v in obj.items()}
    return obj


def load_job_file(path: str) -> dict:
    try:
        return util.load_json(path)
    except (OSError, ValueError) as e:
        raise bad("job file %s: %s" % (path, e))


def check_environment(lock_: dict) -> None:
    """What can be known without importing mflux (spec §6.1 step 1): exit 2 when it fails."""
    if fake():
        return
    if not util.is_apple_silicon():
        raise bad("this machine cannot generate (macOS on Apple silicon only)")
    gen = lock_["generation"]
    ram = util.ram_gib()
    if ram < gen["min_ram_gib"] or memory_limit_gib(lock_) < gen["peak_gib"]:
        raise bad("this machine's RAM (%.0f GiB) cannot generate (needs %d GiB)" % (ram, gen["min_ram_gib"]))
    import importlib.util

    if importlib.util.find_spec("mflux") is None:
        raise bad("mflux is not installed: install limner with the [generate] extra")
    problems = models.generation_ready(lock_)
    if problems:
        raise bad("models are not ready (run `lmn setup`): " + "; ".join(problems))


def memory_limit_gib(lock_: dict) -> float:
    m = lock_["generation"]["memory_limit"]
    limit = max(m["min_gib"], m["ram_fraction"] * util.ram_gib())
    v = os.environ.get("LIMNER_MEMORY_LIMIT_GIB")
    if v:
        try:
            limit = min(limit, float(v))
        except ValueError:
            pass
    return limit


def generator_name() -> str:
    return "fake" if fake() else "mflux"


# --------------------------------------------------------------------------- preview


def preview(root: paths.Root, jl: list[jobs.Job], lock_: dict, force: set, accept: set) -> tuple[bool, dict]:
    """Spec §6.1 step 3: the §6.4 verdicts read-only. (work to do?, verdict lists)."""
    st = {k: [] for k in result.LISTS}
    outcome: dict[str, str] = {}
    work = False
    for stage in jobs.stages(jl):
        for job in stage:
            b = plan.blocked_by(job, outcome)
            if b:
                st["blocked"].append(b)
                outcome[job.id] = "blocked"
                continue
            if any(r["producer"] and outcome.get(r["producer"]) == "generate" for r in job.refs):
                work = True
                outcome[job.id] = "generate"
                continue
            shas = [util.sha256_file_or_none(r["path"]) for r in job.refs]
            if None in shas:
                work = True
                outcome[job.id] = "generate"
                continue
            d = plan.decide(root, job, jobs.job_sha(job, shas, lock_), force, accept, adopt=False)
            if d.adoptable:
                work = True
            if d.action == "generate":
                work = True
                outcome[job.id] = "generate"
            elif d.action == "accept":
                work = True
                outcome[job.id] = "accepted"
            else:
                st[d.action].append(job.id)
                outcome[job.id] = d.action
    return work, st


# --------------------------------------------------------------------------- run


class RunCommand:
    def __init__(self, args):
        self.args = args
        self.pending_signal = None
        self.worker = None
        self.tokens: dict[str, str] = {}  # lock name -> our token (while still ours)
        self.locks: dict[str, str] = {}
        self.att: runs.Attempt | None = None
        self.st: dict | None = None
        self.job_ids: list[str] = []

    # ------------------------------------------------------------------ signals

    def _handler(self, signum, _frame):
        if self.pending_signal is None:
            self.pending_signal = signum

    def check_signal(self) -> None:
        """Before `go`: kill the worker (it has not started), release what is still ours, close
        the attempt as failed: signal:<n>, exit 11 (spec §6.1 `run`'s signal handling)."""
        if self.pending_signal is None:
            return
        n = self.pending_signal
        if self.worker is not None:
            try:
                self.worker.kill()
            except OSError:
                pass
            self.worker.wait()
        for name in ("machine", "run"):
            tok = self.tokens.get(name)
            if tok:
                dirlock.release(self.locks[name], tok)
        obj = self.close("failed", failed=[{"id": None, "reason": "signal:%d" % n,
                                            "message": "lmn run got signal %d before handing over" % n}])
        raise Stop(obj, util.FAILED)

    def close(self, state: str, **kw) -> dict:
        if self.att is None:
            st = result.new_status(getattr(self, "run_id", None), None, state, generator_name())
            st.update(kw)
            return st
        st = self.st
        st["state"] = state
        st["pending"] = list(self.job_ids)
        st.update(kw)
        self.att.write_status(st)
        return st

    def sleep(self, seconds: float) -> None:
        """Short slices so a signal or a cancel request is seen within about 0.1 s / 1 s."""
        end = time.monotonic() + seconds
        last_cancel = 0.0
        while True:
            self.check_signal()
            now = time.monotonic()
            if now - last_cancel >= 1.0:
                last_cancel = now
                self.check_cancel()
            if now >= end:
                return
            time.sleep(min(0.1, end - now))

    def check_cancel(self) -> None:
        if self.att is not None and self.att.cancel_requested():
            tok = self.tokens.get("run")
            obj = self.close("cancelled")
            if tok:
                dirlock.release(self.locks["run"], tok)
            raise Stop(obj, util.FAILED)

    def gate(self, name: str) -> None:
        """Test hook: with LIMNER_TEST_GATES=<dir>, stop at <name> until <dir>/<name>.go exists."""
        d = os.environ.get("LIMNER_TEST_GATES")
        if not d:
            return
        names = os.environ.get("LIMNER_TEST_GATE_NAMES", "")
        if names and name not in names.split(","):
            return
        open(os.path.join(d, name + ".reached"), "w").close()
        while not os.path.exists(os.path.join(d, name + ".go")):
            self.check_signal()
            time.sleep(0.01)

    # ------------------------------------------------------------------ main

    def execute(self) -> tuple[dict, int]:
        old = {s: signal.signal(s, self._handler) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            return self._execute()
        except Stop as s:
            return s.obj, s.code
        finally:
            for s, h in old.items():
                try:
                    signal.signal(s, h)
                except (ValueError, OSError):
                    pass

    def _execute(self) -> tuple[dict, int]:
        a = self.args
        if os.environ.get("S2V_HEAVY_LOCK_TOKEN"):
            raise bad("S2V_HEAVY_LOCK_TOKEN is set: running limner under another lock holder is not supported")
        if a.wait is not None and not 0 <= a.wait <= MAX_WAIT:
            raise bad("--wait must be 0 to %d seconds" % MAX_WAIT)
        root = paths.Root(a.root, a.state_dir)
        lock_ = models.load_lock()
        spec = load_job_file(a.job)
        jl = jobs.validate(spec, root, lock_)
        ids = {j.id for j in jl}
        force, accept = parse_ids(a.force), parse_ids(a.accept)
        for i in force + accept:
            if i not in ids:
                raise bad("--force/--accept: no job %r" % i)
        if set(force) & set(accept):
            raise bad("--force and --accept name the same job: %s" % sorted(set(force) & set(accept)))
        for i in accept:
            j = next(j for j in jl if j.id == i)
            if not os.path.isfile(j.out):
                raise bad("--accept %s: its out %s does not exist" % (i, j.out_rel))
        check_environment(lock_)
        self.run_id = runs.run_id_of(normalize(spec), force, accept)
        if a.check:
            return {"contract": util.CONTRACT, "check": "ok", "run_id": self.run_id, "jobs": len(jl),
                    "stages": max(j.stage for j in jl)}, util.OK
        self.job_ids = [j.id for j in jl]
        warnings = root.state_dir_warnings([j.out for j in jl])
        work, verdicts = preview(root, jl, lock_, set(force), set(accept))
        if not work:
            st = result.new_status(self.run_id, None, "done", generator_name(), warnings=warnings)
            st.update(verdicts)
            st["stages"] = max(j.stage for j in jl)
            return st, result.exit_code(st)
        self.check_signal()
        run = runs.Run(root.state_dir, self.run_id)
        paths.safe_makedirs(root, run.dir)
        me = util.me()
        self.locks = {"run": run.lock, "machine": lock.machine_lock_path()}
        r = dirlock.acquire(run.lock, {"kind": "limner-run", "pid": me["pid"], "pid_start": me["pid_start"],
                                       "holder": "limner run %s" % self.run_id})
        if r.status != "acquired":
            return self.existing(run)
        self.tokens["run"] = r.token
        attempt = util.rand_hex(8)
        att = runs.Attempt(run, attempt)
        os.makedirs(att.dir)
        util.write_json_atomic(att.p("job.json"), {"job": spec, "force": force, "accept": accept,
                                                   "root": root.real, "state_dir": root.state_dir,
                                                   "job_file": os.path.abspath(a.job)})
        util.write_json_atomic(att.p("run.json"), me)
        free = util.free_gib()
        lock_gen = lock_["generation"]
        if free is not None and free < lock_gen["peak_gib"]:
            warnings.append("low_free_memory")
        st = result.new_status(self.run_id, attempt, "starting", generator_name(), warnings=warnings,
                               free_gib_at_start=free, stages=max(j.stage for j in jl),
                               status_file=att.status_file)
        st["pending"] = list(self.job_ids)
        self.att, self.st = att, st
        att.write_status(st)
        util.write_json_atomic(run.current, {"attempt": attempt})
        self.check_signal()

        # step 5: the machine lock, waiting up to --wait while busy
        holder = lock.holder_text(self.run_id, att.status_file)
        mfields = {"kind": "gen", "pid": me["pid"], "pid_start": me["pid_start"], "holder": holder}
        deadline = time.monotonic() + (a.wait or 0)
        while True:
            self.check_signal()
            self.check_cancel()
            tok, busy = lock.try_machine(mfields, free)
            if tok:
                self.tokens["machine"] = tok
                break
            self.check_signal()
            left = deadline - time.monotonic()
            if left <= 0:
                obj = self.close("busy", busy=busy)
                dirlock.release(run.lock, self.tokens.pop("run"))
                return obj, util.BUSY
            self.gate("busy_wait")
            self.sleep(min(left, random.uniform(5, 15)))
        self.check_signal()

        # step 6: the worker, in its own session; argv carries no token
        env = dict(os.environ)
        cmd = [sys.executable, "-m", "limner.worker", root.state_dir, self.run_id, attempt]
        log = open(att.p("worker.log"), "ab")
        delay = float(os.environ.get("LIMNER_TEST_PREEXEC_SLEEP", "0") or 0)
        try:
            self.worker = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                           start_new_session=True, env=env, close_fds=True,
                                           preexec_fn=(lambda: time.sleep(delay)) if delay else None)
        finally:
            log.close()
        self.check_signal()
        ws = util.proc_start(self.worker.pid)
        if not ws:
            return self.spawn_failed("could not read the worker's start time (it ended at once?)")
        util.write_json_atomic(att.p("spawn.json"), {"pid": self.worker.pid, "pid_start": ws})
        self.check_signal()
        self.gate("after_spawn")

        # step 7: hand both locks over, each with a NEW token, by atomic rename
        wfields = {"kind": "gen", "pid": self.worker.pid, "pid_start": ws, "holder": holder}
        new_tokens = {}
        for name in ("run", "machine"):
            fields = dict(wfields, kind="limner-run") if name == "run" else wfields
            new = lock.hand_over(self.locks[name], self.tokens[name], fields)
            if new is None:
                return self.spawn_failed("the %s lock is not ours any more" % name)
            new_tokens[name] = new
            del self.tokens[name]
            self.check_signal()
            if name == "run":
                self.gate("between_handoffs")
        self.check_signal()
        self.gate("before_go")
        self.check_signal()

        # step 8: go. From here a signal is ignored: the worker owns the attempt.
        line = json.dumps({"go": True, "tokens": new_tokens}) + "\n"
        try:
            self.worker.stdin.write(line.encode())
            self.worker.stdin.close()
        except BrokenPipeError:
            pass  # the worker died before go: `status` judges it
        except OSError:
            pass
        self.worker = None
        return {"contract": util.CONTRACT, "run_id": self.run_id, "attempt": attempt, "state": "starting",
                "status_file": att.status_file}, util.OK

    def spawn_failed(self, message: str) -> tuple[dict, int]:
        if self.worker is not None:
            try:
                self.worker.kill()
            except OSError:
                pass
            self.worker.wait()
            self.worker = None
        for name in ("machine", "run"):
            tok = self.tokens.pop(name, None)
            if tok:
                dirlock.release(self.locks[name], tok)
        obj = self.close("failed", failed=[{"id": None, "reason": "environment", "message": message}])
        return obj, util.BAD

    def existing(self, run: runs.Run) -> tuple[dict, int]:
        """Another `run` of this run_id holds the run lock: report its attempt."""
        end = time.monotonic() + 3
        while True:
            a = run.current_attempt()
            st = runs.Attempt(run, a).read_status() if a else None
            if st and st.get("state") in result.LIVE:
                return st, util.OK
            if time.monotonic() > end:
                break
            self.check_signal()
            time.sleep(0.05)
        if st:
            code = result.exit_code(st)
            return st, (util.OK if code == util.RUNNING else code)
        obj = result.new_status(self.run_id, a, "starting", generator_name())
        return obj, util.OK


def cmd_run(args) -> tuple[dict, int]:
    return RunCommand(args).execute()
