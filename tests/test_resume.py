"""Resume rules (spec §6.4) end to end with the fake generator, and the write order."""

import glob
import json
import os
import shutil
import time

from conftest import lmn
from helpers import job, read, wait_for, write_job
from limner import paths


def run(env, jobs, *extra):
    jf = write_job(env["root"], jobs)
    obj, code = lmn("run", "--job", jf, "--root", env["root"], *extra)
    if obj.get("state") == "starting":
        obj, code = lmn("status", "--run-id", obj["run_id"], "--attempt", obj["attempt"], "--root", env["root"],
                        "--wait-seconds", 30)
    return obj, code


def out(env, name="a"):
    return os.path.join(env["root"], "out", "%s.png" % name)


def record(env, name="a"):
    return read(os.path.join(env["root"], ".limner", "records", "out", "%s.png.json" % name))


def scribble(path):
    from PIL import Image
    Image.new("RGB", (64, 64), (1, 2, 3)).save(path)


def test_modified_untracked_force_accept(env):
    assert run(env, [job("a")])[1] == 0
    scribble(out(env))
    st, code = run(env, [job("a")])
    assert (code, st["modified"], st["attempt"]) == (5, ["a"], None)
    st, code = run(env, [job("a")], "--accept", "a")
    assert code == 0 and st["accepted"] == ["a"]
    rec = record(env)
    assert rec["generator"] == "human" and rec["accepted_from"]["generator"] == "fake"
    st, code = run(env, [job("a")])
    assert (code, st["skipped"]) == (0, ["a"])
    scribble_bytes = open(out(env), "rb").read()
    os.unlink(os.path.join(env["root"], ".limner", "records", "out", "a.png.json"))
    st, code = run(env, [job("a")])
    assert (code, st["untracked"]) == (5, ["a"])
    st, code = run(env, [job("a")], "--force", "a")
    assert code == 0 and st["regenerated"][0]["reason"] == "forced"
    prev = os.path.join(env["root"], st["regenerated"][0]["prev"])
    assert open(prev, "rb").read() == scribble_bytes  # the person's file is kept in prev/


def test_job_changed_regenerates_and_keeps_prev(env):
    run(env, [job("a")])
    old = open(out(env), "rb").read()
    st, code = run(env, [job("a", prompt="something else")])
    assert code == 0 and st["regenerated"][0]["reason"] == "job_changed"
    assert open(os.path.join(env["root"], st["regenerated"][0]["prev"]), "rb").read() == old
    prevs = glob.glob(os.path.join(env["root"], ".limner", "prev", "out", "a.*.json"))
    assert len(prevs) == 1  # the old record next to the old file


def test_other_kind_is_modified(env):
    run(env, [job("a")])
    p = os.path.join(env["root"], ".limner", "records", "out", "a.png.json")
    rec = read(p)
    rec["kind"] = "compose"
    with open(p, "w") as f:
        json.dump(rec, f)
    st, code = run(env, [job("a")])
    assert (code, st["modified"]) == (5, ["a"])


def test_blocked_by_modified_reference(env):
    jobs = [job("ref"), job("p1", refs=["out/ref.png"]), job("p2", refs=["out/p1.png"])]
    run(env, jobs)
    scribble(out(env, "ref"))
    os.unlink(out(env, "p1"))
    st, code = run(env, jobs)
    assert code == 5
    assert st["blocked"] == [{"id": "p1", "by": "ref", "cause": "modified"},
                             {"id": "p2", "by": "p1", "cause": "blocked"}]


def test_reference_change_regenerates_dependents(env):
    jobs = [job("ref"), job("p1", refs=["out/ref.png"])]
    run(env, jobs)
    jobs[0]["seed"] = 8
    st, code = run(env, jobs)
    assert code == 0 and [r["id"] for r in st["regenerated"]] == ["ref", "p1"]


def test_accept_without_out_is_2(env):
    jf = write_job(env["root"], [job("a")])
    obj, code = lmn("run", "--job", jf, "--root", env["root"], "--accept", "a")
    assert code == 2


def test_orphan_record_is_adopted(env):
    run(env, [job("a")])
    rpath = os.path.join(env["root"], ".limner", "records", "out", "a.png.json")
    tmp = os.path.join(env["root"], ".limner", "tmp")
    os.makedirs(tmp, exist_ok=True)
    shutil.move(rpath, os.path.join(tmp, "999999-1-abc.record.json"))  # its writer is dead
    st, code = run(env, [job("a")])
    assert code == 0 and st["adopted"] == ["a"] and st["skipped"] == ["a"]
    assert os.path.exists(rpath) and not os.listdir(tmp)


def test_out_changed_during_generation_is_not_overwritten(env, monkeypatch):
    monkeypatch.setenv("LIMNER_FAKE_SECONDS", "2")
    jf = write_job(env["root"], [job("a")])
    obj, code = lmn("run", "--job", jf, "--root", env["root"])
    wait_for(lambda: (read(obj["status_file"]) or {}).get("current") == "a", what="generating")
    os.makedirs(os.path.dirname(out(env)), exist_ok=True)
    scribble(out(env))
    mine = open(out(env), "rb").read()
    st, code = lmn("status", "--run-id", obj["run_id"], "--root", env["root"], "--wait-seconds", 30)
    assert code == 5 and st["modified"] == ["a"]
    assert open(out(env), "rb").read() == mine
    assert glob.glob(os.path.join(env["root"], ".limner", "prev", "out", "a.*.rejected.png"))


def test_commit_order_never_leaves_out_missing(env, tmp_path):
    """prev hard link before the atomic rename; the old record goes next to the prev file."""
    root = paths.Root(str(env["root"]))
    o = os.path.join(root.real, "x.png")
    with open(o, "wb") as f:
        f.write(b"old")
    os.makedirs(os.path.dirname(root.record_path(o)), exist_ok=True)
    with open(root.record_path(o), "w") as f:
        json.dump({"kind": "compose"}, f)
    t = os.path.join(root.tmp_dir(), paths.tmp_name(".png"))
    with open(t, "wb") as f:
        f.write(b"new")
    res = paths.commit(root, o, t, {"kind": "compose", "out_sha256": "x"})
    assert open(o, "rb").read() == b"new" and open(res["prev"], "rb").read() == b"old"
    assert read(res["prev"][:-4] + ".json") == {"kind": "compose"}
    assert read(res["record"])["prev"] == root.rel(res["prev"])
    assert os.stat(res["prev"]).st_nlink == 1 and not os.listdir(root.tmp_dir())
