"""The line-break table (spec §7.3): tests/data/linebreak.json is the reference, not the prose."""

import json
import os

import pytest

from conftest import DATA
from limner.compose.linebreak import em_width, wrap

CASES = json.load(open(os.path.join(DATA, "linebreak.json"), encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["text"] for c in CASES])
def test_table(case):
    t = case["text"]
    widths = [0.0 if c == "\n" else em_width(c) for c in t]
    lines, overflow = wrap(t, widths, case["width"])
    assert [t[s:e] for s, e in lines] == case["lines"]
    assert overflow == case["overflow"]
