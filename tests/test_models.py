"""The model store shared with script2video (spec §3) and `lmn setup` (spec §9).

Every test here works on a temporary models dir (LIMNER_MODELS_DIR from the `env` fixture); the
guard below refuses to run when the store would resolve anywhere else (the real store is shared
with script2video and must never be written by tests).
"""

import json
import os

import pytest

from conftest import lmn
from limner import cli, fonts, models, setup, util

LOCK = models.load_lock()
Q = "qwen-image-edit-2511-q4"
L = "qwen-image-edit-2511-lightning"
SRC = "qwen-image-edit-2511"


@pytest.fixture(autouse=True)
def private_store(env):
    _, d = models.models_dir()
    assert d.startswith(str(env["tmp"])), "refusing: the model store is not a temporary dir (%s)" % d
    yield


def folder(name):
    p = models.folder(LOCK, name)
    assert "pytest" in p or p.startswith("/private/var/folders") or p.startswith("/tmp"), p
    return p


def make(name, marker=".s2v-complete", body=None, lora_file=True):
    p = folder(name)
    os.makedirs(p, exist_ok=True)
    if marker:
        with open(os.path.join(p, marker), "w") as f:
            json.dump(body if body is not None else models.expected_marker(LOCK, name), f)
    if name == L and lora_file:
        open(os.path.join(p, LOCK["lora"]["file"]), "w").close()
    return p


def ready_store():
    make(Q, body=dict(models.expected_marker(LOCK, Q), mflux="0.22.0", note="made by s2v"))
    make(L)


def test_store_order(env, monkeypatch):
    tmp = env["tmp"]
    monkeypatch.setenv("LIMNER_MODELS_DIR", "relative/ignored")
    monkeypatch.setenv("S2V_MODELS_DIR", str(tmp / "s2v-models"))
    monkeypatch.setenv("HOME", str(tmp / "home"))
    assert models.models_dir() == ("S2V_MODELS_DIR", str(tmp / "s2v-models"))  # none exist: the first
    os.makedirs(tmp / "home" / ".cache" / "script2video" / "models")
    assert models.models_dir()[0] == "s2v-default"
    os.makedirs(tmp / "s2v-models")
    assert models.models_dir()[0] == "S2V_MODELS_DIR"


def test_marker_rules(env):
    assert models.status(LOCK, Q)["state"] == "missing"
    ready_store()
    assert models.status(LOCK, Q)["state"] == "ready"  # partial match: note/mflux ignored
    assert models.status(LOCK, L)["state"] == "ready"
    assert models.generation_ready(LOCK) == []
    os.rename(os.path.join(folder(L), ".s2v-complete"), os.path.join(folder(L), ".complete"))
    assert models.status(LOCK, L)["state"] == "ready"  # limner's own marker counts too
    make(Q, body={"from": SRC, "revision": "0" * 40, "bits": 4})
    assert models.status(LOCK, Q)["state"] == "mismatch:revision"
    os.rename(os.path.join(folder(L), LOCK["lora"]["file"]), os.path.join(folder(L), "moved"))
    assert models.status(LOCK, L)["state"] == "mismatch:file"


def test_foreign_and_limner_partial(env):
    os.makedirs(folder(Q))
    assert models.status(LOCK, Q)["state"] == "foreign_partial"
    os.makedirs(folder(L) + ".limner-partial")
    assert models.status(LOCK, L)["state"] == "limner_partial"
    assert models.status(LOCK, SRC)["required"] is False


def test_mflux_version_is_a_warning_only(env):
    make(Q, body=dict(models.expected_marker(LOCK, Q), mflux="0.21.0"))
    make(L)
    assert models.status(LOCK, Q)["state"] == "ready"
    assert "0.21.0" in models.mflux_warning(LOCK, "0.22.0")


def test_write_markers_writes_both_alike(env):
    d = env["tmp"] / "m"
    d.mkdir()
    models.write_markers(str(d), {"repo": "r", "revision": "x"})
    a = (d / ".complete").read_text()
    assert a == (d / ".s2v-complete").read_text() and json.loads(a)["repo"] == "r"


@pytest.fixture
def no_net(monkeypatch):
    def boom(**kw):
        raise AssertionError("downloaded %s" % kw)

    monkeypatch.setattr(setup, "snapshot_download", boom)
    monkeypatch.setattr(fonts, "install", lambda: {"fonts": "stub"})
    monkeypatch.setattr(util, "ram_gib", lambda: 64.0)
    monkeypatch.setattr(util, "disk_free_gib", lambda p: 1000.0)


def main_json(capsys, *argv):
    code = cli.main(list(argv))
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1]), code


def test_setup_with_s2v_models_downloads_nothing(env, no_net, capsys):
    ready_store()
    before = {p: sorted(os.listdir(folder(p))) for p in (Q, L)}
    obj, code = main_json(capsys, "setup", "--yes")
    assert code == 0 and obj["models"] == "ready" and obj["steps"] == []
    assert {p: sorted(os.listdir(folder(p))) for p in (Q, L)} == before  # nothing written into them


def test_setup_downloads_into_partial_and_marks_both(env, no_net, monkeypatch, capsys):
    make(Q)
    calls = []

    def fake_download(repo_id, revision, local_dir, allow_patterns):
        calls.append(repo_id)
        assert local_dir.endswith(".limner-partial")
        assert json.load(open(os.path.join(local_dir, "owner.json")))["tool"] == "limner"
        open(os.path.join(local_dir, LOCK["lora"]["file"]), "w").close()

    monkeypatch.setattr(setup, "snapshot_download", fake_download)
    obj, code = main_json(capsys, "setup", "--yes")
    assert code == 0 and obj["steps"] == ["download %s" % L] and calls == [LOCK["models"][L]["repo"]]
    p = folder(L)
    assert models.status(LOCK, L)["state"] == "ready"
    assert open(os.path.join(p, ".complete")).read() == open(os.path.join(p, ".s2v-complete")).read()
    assert not os.path.exists(p + ".limner-partial") and not os.path.exists(os.path.join(p, "owner.json"))
    assert not os.path.exists(os.path.join(env["state"], "heavy.lock.d"))


def test_setup_takes_over_a_dead_download_and_refuses_a_live_one(env, no_net, monkeypatch, capsys):
    make(Q)
    part = folder(L) + ".limner-partial"
    os.makedirs(part)
    with open(os.path.join(part, "owner.json"), "w") as f:
        json.dump({"tool": "limner", "step": "download", "pid": 999999, "pid_start": "Mon Oct 19 20:06:47 2026"}, f)
    open(os.path.join(part, "half.bin"), "w").close()
    seen = []

    def resume(**kw):
        seen.append(os.listdir(kw["local_dir"]))
        open(os.path.join(kw["local_dir"], LOCK["lora"]["file"]), "w").close()

    monkeypatch.setattr(setup, "snapshot_download", resume)
    obj, code = main_json(capsys, "setup", "--yes")
    assert code == 0 and "half.bin" in seen[0]  # resumed, not restarted
    os.rename(folder(L), part)
    os.rename(os.path.join(part, ".complete"), os.path.join(part, "x1"))
    os.rename(os.path.join(part, ".s2v-complete"), os.path.join(part, "x2"))
    me = util.me()
    with open(os.path.join(part, "owner.json"), "w") as f:
        json.dump({"tool": "limner", "step": "download", **me}, f)
    obj, code = main_json(capsys, "setup", "--yes")
    assert code == 2 and "another limner" in obj["error"]


def test_setup_leaves_foreign_partial_alone(env, no_net, capsys):
    make(Q)
    os.makedirs(folder(L))
    obj, code = main_json(capsys, "setup", "--yes")
    assert code == 2 and "without a marker" in obj["error"]
    assert os.listdir(folder(L)) == []


def test_setup_without_yes_unattended_is_2(env, no_net, capsys, monkeypatch):
    make(Q)
    monkeypatch.setattr("sys.stdin", open(os.devnull))
    obj, code = main_json(capsys, "setup")
    assert code == 2 and "--yes" in obj["error"]


def test_doctor_json(env):
    ready_store()
    obj, code = lmn("doctor", "--json", timeout=180)
    assert code == 0
    assert {m["name"]: m["state"] for m in obj["models"]}[Q] == "ready"
    assert obj["models_dir"]["source"] == "LIMNER_MODELS_DIR"
    assert obj["versions"]["pillow"] and "raqm" in obj["versions"]


def test_doctor_smoke_with_fake_generator(env):
    obj, code = lmn("doctor", "--json", "--smoke", timeout=180)
    assert code == 0 and obj["smoke"]["ok"], obj["smoke"]
    assert not os.path.exists(os.path.join(env["state"], "heavy.lock.d"))
