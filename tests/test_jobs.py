"""Job validation, stages, job_sha, letterboxing, path rules (spec §5, §6.2, §6.4)."""

import os
import unicodedata

import pytest

from helpers import job
from limner import jobs, models, paths, util
from limner.util import Fail

LOCK = models.load_lock()


def spec(*js):
    return {"version": 1, "model": "qwen-image-edit-2511", "jobs": list(js)}


def root_of(tmp_path):
    (tmp_path / "r").mkdir(exist_ok=True)
    return paths.Root(str(tmp_path / "r"))


def test_stages_and_cycle(tmp_path):
    r = root_of(tmp_path)
    jl = jobs.validate(spec(job("c", refs=["out/b.png"]), job("b", refs=["out/a.png"]), job("a")), r, LOCK)
    assert {j.id: j.stage for j in jl} == {"a": 1, "b": 2, "c": 3}
    assert [[j.id for j in s] for s in jobs.stages(jl)] == [["a"], ["b"], ["c"]]
    with pytest.raises(Fail, match="cycle"):
        jobs.validate(spec(job("a", refs=["out/b.png"]), job("b", refs=["out/a.png"])), r, LOCK)


def test_duplicate_ids_and_outs_fold_case_and_nfc(tmp_path):
    r = root_of(tmp_path)
    with pytest.raises(Fail, match="id appears twice"):
        jobs.validate(spec(job("A"), job("a", out="out/x.png")), r, LOCK)
    with pytest.raises(Fail, match="also the out"):
        jobs.validate(spec(job("a", out="out/Ä.png"), job("b", out=unicodedata.normalize("NFD", "out/ä.png"))),
                      r, LOCK)


def test_paths_symlink_git_claude(tmp_path):
    r = root_of(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), os.path.join(r.real, "link"))
    with pytest.raises(Fail, match="outside"):
        r.writable("link/a.png")
    for p in (".git/a.png", "x/.GIT/a.png", ".Claude/a.png"):
        with pytest.raises(Fail, match=".git/ or .claude"):
            r.writable(p)
    with pytest.raises(Fail, match="only .png"):
        r.writable("a.jpg")
    assert r.writable("Sub/A.PNG").endswith("Sub/A.PNG")
    with pytest.raises(Fail, match="outside"):
        paths.Root(str(tmp_path / "r"), str(tmp_path / "elsewhere"))


def test_reference_rules(tmp_path):
    from PIL import Image

    r = root_of(tmp_path)
    Image.new("RGB", (64, 64)).save(os.path.join(r.real, "ok.png"))
    Image.new("RGB", (64, 64)).save(os.path.join(r.real, "bad.gif"))
    Image.new("RGB", (4097, 16)).save(os.path.join(r.real, "wide.png"))
    jobs.validate(spec(job("a", refs=["ok.png"])), r, LOCK)
    with pytest.raises(Fail, match="not PNG"):
        jobs.validate(spec(job("a", refs=["bad.gif"])), r, LOCK)
    with pytest.raises(Fail, match="long side"):
        jobs.validate(spec(job("a", refs=["wide.png"])), r, LOCK)
    with pytest.raises(Fail) as e:
        jobs.validate(spec(job("a", refs=["none.png"])), r, LOCK)
    assert e.value.extra["missing_input"] == "none.png"


def test_job_sha(tmp_path):
    r = root_of(tmp_path)
    a, = jobs.validate(spec(job("a", prompt="café")), r, LOCK)
    b, = jobs.validate(spec(job("b", out="out/other.png", prompt="café")), r, LOCK)
    assert jobs.job_sha(a, [], LOCK) == jobs.job_sha(b, [], LOCK)  # NFC; id and out do not count
    assert jobs.job_sha(a, ["x"], LOCK) != jobs.job_sha(a, ["y"], LOCK)
    c, = jobs.validate(spec(job("a", seed=8)), r, LOCK)
    assert jobs.job_sha(c, [], LOCK) != jobs.job_sha(a, [], LOCK)
    lock2 = dict(LOCK, generation=dict(LOCK["generation"], skip_negative=not LOCK["generation"]["skip_negative"]))
    assert jobs.job_sha(a, [], lock2) != jobs.job_sha(a, [], LOCK)


def test_letterbox_keeps_aspect(tmp_path):
    from PIL import Image

    src = str(tmp_path / "ref.png")
    Image.new("RGB", (640, 800), (255, 0, 0)).save(src)
    dst = str(tmp_path / "lb.png")
    jobs.letterbox(src, 896, 896, "#E0E0E0", dst)
    im = Image.open(dst)
    assert im.size == (896, 896)
    assert im.getpixel((5, 448)) == (224, 224, 224) and im.getpixel((448, 448)) == (255, 0, 0)
    assert im.getpixel((448 - 358 + 2, 448)) == (255, 0, 0)  # 640*1.12=717 wide, centered


def test_run_id_depends_on_force_and_accept():
    from limner import runs

    s = spec(job("a"))
    assert runs.run_id_of(s, [], []) != runs.run_id_of(s, ["a"], [])
    assert runs.run_id_of(s, ["a"], []) != runs.run_id_of(s, [], ["a"])
    assert len(runs.run_id_of(s, [], [])) == 16


def test_exit_code_priority():
    from limner import result

    st = result.new_status("r", "a", "done", "fake")
    assert result.exit_code(st) == 0
    st["modified"] = ["x"]
    assert result.exit_code(st) == 5
    st["failed"] = [{"id": "y", "reason": "memory"}]
    assert result.exit_code(st) == 11
    st["failed"].append({"id": "z", "reason": "environment"})
    assert result.exit_code(st) == 2
    st2 = result.new_status("r", "a", "done", "fake", blocked=[{"id": "b", "by": "c", "cause": "blocked"},
                                                               {"id": "c", "by": "d", "cause": "untracked"}])
    assert result.exit_code(st2) == 5
    assert result.exit_code(result.new_status("r", "a", "running", "fake")) == 10
    assert result.exit_code(result.new_status("r", "a", "busy", "fake")) == 73
