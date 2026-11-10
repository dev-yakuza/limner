import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A private S2V state dir, the fake generator, s2v server ports that answer nothing."""
    state = tmp_path / "s2v"
    root = tmp_path / "root"
    root.mkdir()
    for k in ("S2V_HEAVY_LOCK_TOKEN", "LIMNER_MIN_FREE_GIB", "LIMNER_FAKE_SECONDS", "LIMNER_FAKE_ABORT_ID",
              "LIMNER_FAKE_ERROR_ID", "LIMNER_FAKE_FOOTPRINT_GIB", "LIMNER_TEST_GATES", "LIMNER_TEST_GATE_NAMES",
              "LIMNER_TEST_PREEXEC_SLEEP", "LIMNER_FAKE_ENV_ERROR", "LIMNER_MEMORY_LIMIT_GIB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("S2V_STATE_DIR", str(state))
    monkeypatch.setenv("LIMNER_FAKE_GENERATOR", "1")
    monkeypatch.setenv("S2V_TTS_PORT", "9")
    monkeypatch.setenv("S2V_ALIGNER_PORT", "9")
    (tmp_path / "models").mkdir()  # must exist: the store is the first candidate that exists
    monkeypatch.setenv("LIMNER_MODELS_DIR", str(tmp_path / "models"))
    return {"tmp": tmp_path, "root": root, "state": state}


def lmn(*argv, env=None, timeout=120, check=None):
    p = subprocess.run([sys.executable, "-m", "limner", *map(str, argv)], capture_output=True, text=True,
                       env=env or os.environ, timeout=timeout)
    lines = p.stdout.strip().splitlines()
    obj = json.loads(lines[-1]) if lines else None
    if check is not None:
        assert p.returncode == check, (p.returncode, obj, p.stderr)
    return obj, p.returncode
