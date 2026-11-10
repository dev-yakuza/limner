"""The detached worker = the supervisor (spec §6.1 "작업자 구조").

    python -m limner.worker <state_dir> <run_id> <attempt>

It waits for `lmn run`'s `go` line on stdin, checks that both locks were handed to it, then runs
the stages: for each, the §6.4 verdicts under the locks, then a NEW child process that loads the
model, encodes every prompt of the stage, drops the text encoder and denoises job by job. The
supervisor watches the child's memory, writes every output (record, prev/, atomic rename) and the
attempt's status.json, and releases both locks at the end. It has no SIGTERM handler on purpose
(`lmn cancel` kills the process group and writes the final state).
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback

from . import dirlock, jobs, lock, models, paths, plan, result, runs, util

POLL = 0.2
MEMORY_EVERY = 3.0
CHILD_CANCELLED = 3
CHILD_ENVIRONMENT = 2


def log(msg: str) -> None:
    print("[limner worker %s] %s" % (util.utc_now(), msg), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- memory


def footprint_gib(pid: int) -> float | None:
    """ri_phys_footprint of a child (proc_pid_rusage, RUSAGE_INFO_V2), GiB."""
    fake = os.environ.get("LIMNER_FAKE_FOOTPRINT_GIB")
    if fake:
        return float(fake)
    if sys.platform != "darwin":
        try:
            with open("/proc/%d/status" % pid) as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 2**20
        except OSError:
            return None
        return None
    import ctypes
    import ctypes.util

    class RU(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
            (n, ctypes.c_uint64) for n in (
                "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
                "ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime").split()
        ] + [("rest", ctypes.c_uint64 * 64)]

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    r = RU()
    if libc.proc_pid_rusage(pid, 2, ctypes.byref(r)) != 0:
        return None
    return r.ri_phys_footprint / 2**30


def machine_text() -> str:
    brand = ""
    if sys.platform == "darwin":
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True).stdout.strip()
        except OSError:
            pass
    return "%s %s %s %.0fGiB" % (platform.system(), platform.machine(), brand, util.ram_gib())


def version_of(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


# --------------------------------------------------------------------------- supervisor


class Supervisor:
    def __init__(self, att: runs.Attempt, snap: dict, st: dict):
        self.att = att
        self.snap = snap
        self.st = st
        self.root = paths.Root(snap["root"], snap["state_dir"])
        self.lock = models.load_lock()
        self.fake = os.environ.get("LIMNER_FAKE_GENERATOR") == "1"
        self.outcome: dict[str, str] = {}

    def write(self) -> None:
        self.att.write_status(self.st)

    def remove_pending(self, jid: str) -> None:
        if jid in self.st["pending"]:
            self.st["pending"].remove(jid)

    def finish(self, state: str) -> None:
        self.st["state"] = state
        self.st["current"] = None
        self.write()

    def run(self) -> None:
        try:
            jl = jobs.validate(self.snap["job"], self.root, self.lock)
        except util.Fail as e:
            self.st["failed"].append({"id": None, "reason": "error", "message": e.message})
            self.finish("failed")
            return
        force, accept = set(self.snap.get("force") or []), set(self.snap.get("accept") or [])
        stale = paths.stale_tmp(self.root)
        if stale:
            self.st["warnings"].append("stale_tmp")
            self.st["stale_tmp"] = stale
        st_list = jobs.stages(jl)
        self.st["stages"] = len(st_list)
        self.st["pending"] = [j.id for j in jl]
        for si, stage in enumerate(st_list, 1):
            if self.att.cancel_requested():
                self.finish("cancelled")
                return
            self.st["stage"] = si
            todo = []
            for job in stage:
                b = plan.blocked_by(job, self.outcome)
                if b:
                    self.st["blocked"].append(b)
                    self.outcome[job.id] = "blocked"
                    self.remove_pending(job.id)
                    continue
                try:
                    shas = [util.sha256_file(r["path"]) for r in job.refs]
                    jsha = jobs.job_sha(job, shas, self.lock)
                    d = plan.decide(self.root, job, jsha, force, accept, adopt=True)
                except OSError as e:
                    self.fail(job.id, "error", "cannot read an input: %s" % e)
                    continue
                if d.adopted:
                    self.st["adopted"].append(job.id)
                if d.action == "generate":
                    todo.append((job, jsha, d, shas))
                    continue
                self.remove_pending(job.id)
                if d.action == "accept":
                    plan.write_record(self.root, job.out, plan.human_record(job, jsha, d.out_sha, d.record))
                    self.st["accepted"].append(job.id)
                    self.outcome[job.id] = "accepted"
                else:
                    self.st[d.action].append(job.id)
                    self.outcome[job.id] = d.action
            self.write()
            if todo:
                ended = self.run_stage(si, todo)
                if ended:
                    self.finish(ended)
                    return
        self.finish("done")

    def fail(self, jid: str, reason: str, message: str) -> None:
        self.st["failed"].append({"id": jid, "reason": reason, "message": message})
        self.outcome[jid] = "failed"
        self.remove_pending(jid)

    # ------------------------------------------------------------------ one stage

    def run_stage(self, si: int, todo: list) -> str | None:
        """Generate one stage in a new child. None = go on; else the attempt's final state."""
        gen = self.lock["generation"]
        made: list[str] = []
        entries = []
        tdir = self.root.tmp_dir()
        try:
            for job, jsha, d, shas in todo:
                images = []
                if job.refs:
                    for r in job.refs:
                        p = os.path.join(tdir, paths.tmp_name(".ref.png"))
                        jobs.letterbox(r["path"], job.width, job.height, gen["blank"], p)
                        made.append(p)
                        images.append(p)
                else:
                    p = os.path.join(tdir, paths.tmp_name(".blank.png"))
                    jobs.blank_canvas(job.width, job.height, gen["blank"], p)
                    made.append(p)
                    images.append(p)
                tmp_out = os.path.join(tdir, paths.tmp_name(".png"))
                made.append(tmp_out)
                entries.append({"id": job.id, "prompt": job.prompt, "width": job.width, "height": job.height,
                                "seed": job.seed, "images": images, "tmp_out": tmp_out, "job_sha": jsha})
            spec = {"fake": self.fake, "generation": gen, "jobs": entries,
                    "cancel_file": self.att.p("cancel_requested")}
            stage_file = self.att.p("stage-%d.json" % si)
            events_file = self.att.p("stage-%d.events.jsonl" % si)
            util.write_json_atomic(stage_file, spec)
            return self._child(si, todo, entries, stage_file, events_file)
        finally:
            for p in made:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _child(self, si, todo, entries, stage_file, events_file) -> str | None:
        by_id = {job.id: (job, jsha, d, shas) for job, jsha, d, shas in todo}
        open(events_file, "a").close()
        rfd, wfd = os.pipe()  # the lifeline: the child exits when this closes
        logf = open(self.att.p("worker.log"), "ab")
        try:
            child = subprocess.Popen([sys.executable, "-m", "limner.child", stage_file, events_file],
                                     stdin=rfd, stdout=logf, stderr=logf, close_fds=True)
        finally:
            os.close(rfd)
            logf.close()
        cs = util.proc_start(child.pid)
        util.write_json_atomic(self.att.p("child.json"), {"pid": child.pid, "pid_start": cs, "stage": si})
        limit = memory_limit()
        current = entries[0]["id"]
        self.st["current"] = current
        self.write()
        reported: set[str] = set()
        killed_for_memory = False
        env_message = None
        pos = 0
        last_mem = 0.0
        rc = None
        while True:
            pos, events = read_events(events_file, pos)
            for ev in events:
                kind, jid = ev.get("event"), ev.get("id")
                if kind == "start":
                    current = jid
                    self.st["current"] = jid
                    self.write()
                elif kind == "done" and jid in by_id:
                    reported.add(jid)
                    self.commit(by_id[jid], ev)
                    self.write()
                elif kind == "error" and jid in by_id:
                    reported.add(jid)
                    self.fail(jid, "error", str(ev.get("message")))
                    self.write()
                elif kind == "environment":
                    env_message = str(ev.get("message"))
            if rc is not None:
                break
            rc = child.poll()
            if rc is not None:
                continue  # read the last events once more
            now = time.monotonic()
            if now - last_mem >= MEMORY_EVERY:
                last_mem = now
                fp = footprint_gib(child.pid)
                if fp is not None:
                    self.st["peak_gib"] = round(max(self.st.get("peak_gib") or 0, fp), 1)
                    if fp > limit:
                        log("child footprint %.1f GiB > limit %.1f GiB: killing it" % (fp, limit))
                        killed_for_memory = True
                        child.kill()
            time.sleep(POLL)
        os.close(wfd)
        rest = [e["id"] for e in entries if e["id"] not in reported]
        if rc == 0:
            for jid in rest:
                self.fail(jid, "error", "the generator ended without a result for this job")
            self.st["current"] = None
            return None
        if rc == CHILD_CANCELLED:
            return "cancelled"
        if rc == CHILD_ENVIRONMENT:
            reason, message = "environment", env_message or "the generator could not start"
        elif rc < 0:
            sig = -rc
            if killed_for_memory:
                reason, message = "memory", "the child went over %.1f GiB" % limit
            elif sig == signal.SIGABRT:
                reason, message = "metal", "the child aborted (SIGABRT; Metal allocation failure?)"
            else:
                reason, message = "signal:%d" % sig, "the child ended by signal %d" % sig
        else:
            reason, message = "error", "the child exited with code %d" % rc
        if current in rest:
            self.fail(current, reason, message)
        return "failed"

    def commit(self, item, ev: dict) -> None:
        job, jsha, d, shas = item
        tmp = ev.get("tmp_out")
        now = util.sha256_file_or_none(job.out)
        if now != d.out_sha:
            # changed since the verdict (a person or compose wrote it): keep theirs, keep ours aside
            dst = self.root.prev_base(job.out, util.utc_stamp()) + ".rejected.png"
            paths.safe_makedirs(self.root, os.path.dirname(dst))
            paths.rename_into(tmp, dst)
            self.st["modified"].append(job.id)
            self.outcome[job.id] = "modified"
            self.remove_pending(job.id)
            return
        out_sha = util.sha256_file(tmp)
        gen = self.lock["generation"]
        q, lo = self.lock["quantized"], self.lock["lora"]
        import limner

        rec = {
            "limner": limner.__version__, "contract": util.CONTRACT, "kind": "generate",
            "generator": "fake" if self.fake else "mflux",
            "mflux": version_of("mflux"), "mlx": version_of("mlx"), "machine": machine_text(),
            "model": {"name": q["name"], "revision": self.lock["models"][q["from"]]["revision"], "bits": q["bits"]},
            "lora": {"name": lo["model"], "revision": self.lock["models"][lo["model"]]["revision"], "file": lo["file"]},
            "model_license": self.lock["model_license"], "steps": gen["steps"], "guidance": gen["guidance"],
            "multi_ref": gen["multi_ref"], "skip_negative": gen["skip_negative"],
            "width": job.width, "height": job.height, "seed": job.seed, "prompt": job.prompt,
            "refs": [{"path": r["rel"], "sha256": s} for r, s in zip(job.refs, shas)],
            "blank": gen["blank"], "letterbox": gen["letterbox"],
            "out_sha256": out_sha, "job_sha": jsha, "started_at": ev.get("started_at"),
            "seconds": ev.get("seconds"), "peak_gb": ev.get("peak_gib"),
        }
        res = paths.commit(self.root, job.out, tmp, rec)
        self.st["done"].append({"id": job.id, "out": job.out_rel, "out_sha256": out_sha, "record": res["record"],
                                "seconds": ev.get("seconds")})
        if d.reason in ("job_changed", "forced"):
            self.st["regenerated"].append({"id": job.id, "reason": d.reason,
                                           "prev": self.root.rel(res["prev"]) if res["prev"] else None})
        self.outcome[job.id] = "done"
        self.remove_pending(job.id)


def memory_limit() -> float:
    from .runner import memory_limit_gib

    return memory_limit_gib(models.load_lock())


def read_events(path: str, pos: int) -> tuple[int, list[dict]]:
    try:
        with open(path, "rb") as f:
            f.seek(pos)
            data = f.read()
    except OSError:
        return pos, []
    if not data:
        return pos, []
    end = data.rfind(b"\n")
    if end < 0:
        return pos, []
    out = []
    for line in data[: end + 1].splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return pos + end + 1, out


# --------------------------------------------------------------------------- entry


def main(argv: list[str]) -> int:
    util.reset_signals()
    state_dir, run_id, attempt = argv
    run = runs.Run(state_dir, run_id)
    att = runs.Attempt(run, attempt)
    line = sys.stdin.readline()
    if not line:
        log("no go line: lmn run ended before handing the locks over; nothing written")
        return 0
    if os.environ.get("LIMNER_TEST_GATES") and "worker_after_go" in os.environ.get("LIMNER_TEST_GATE_NAMES", ""):
        d = os.environ["LIMNER_TEST_GATES"]
        open(os.path.join(d, "worker_after_go.reached"), "w").close()
        while not os.path.exists(os.path.join(d, "worker_after_go.go")):
            time.sleep(0.01)
    go = json.loads(line)
    tokens = go.get("tokens") or {}
    me = util.me()
    snap = att.read("job.json")
    st = att.read_status()
    locks = {"run": run.lock, "machine": lock.machine_lock_path()}
    ours = {k: lock.owned_by(p, tokens.get(k, ""), me["pid"], me["pid_start"]) for k, p in locks.items()}

    def release_all():
        for k, p in locks.items():
            if tokens.get(k):
                dirlock.release(p, tokens[k])

    if not all(ours.values()) or snap is None or st is None:
        log("the locks were not handed to this worker: %s" % ours)
        st = st or result.new_status(run_id, attempt, "failed", "fake")
        st["state"] = "failed"
        st["failed"].append({"id": None, "reason": "lock_lost",
                             "message": "lock owners after go: %s" % {k: v for k, v in ours.items()}})
        att.write_status(st)
        release_all()
        return 0
    if att.cancel_requested():
        release_all()
        st["state"] = "cancelled"
        att.write_status(st)
        return 0
    st["state"] = "running"
    st["worker"] = {"pid": me["pid"], "pid_start": me["pid_start"]}
    att.write_status(st)
    sup = Supervisor(att, snap, st)
    try:
        sup.run()
    except Exception as e:  # noqa: BLE001 - recorded, then the locks are released
        log(traceback.format_exc())
        st["failed"].append({"id": st.get("current"), "reason": "error", "message": "worker: %s" % e})
        st["state"] = "failed"
        att.write_status(st)
    finally:
        release_all()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
