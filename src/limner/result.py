"""The status of an attempt (spec §6.5) and the exit code it maps to (spec §10)."""

from __future__ import annotations

from . import util

FINAL = ("done", "failed", "busy", "cancelled", "crashed", "orphan_child")
LIVE = ("starting", "running")
LISTS = ("done", "skipped", "adopted", "regenerated", "modified", "untracked", "failed", "blocked",
         "accepted", "pending")


def new_status(run_id: str | None, attempt: str | None, state: str, generator: str, **kw) -> dict:
    st = {"contract": util.CONTRACT, "run_id": run_id, "attempt": attempt, "state": state,
          "stage": None, "stages": None, "current": None}
    for k in LISTS:
        st[k] = []
    st.update({"generator": generator, "free_gib_at_start": None, "warnings": [], "busy": None})
    st.update(kw)
    return st


def _root_cause(st: dict, jid: str, seen=None) -> str | None:
    seen = seen or set()
    if jid in seen:
        return None
    seen.add(jid)
    for b in st.get("blocked") or []:
        if b.get("id") == jid:
            if b.get("cause") == "blocked":
                return _root_cause(st, b.get("by"), seen)
            return b.get("cause")
    if any(f.get("id") == jid for f in st.get("failed") or []):
        return "failed"
    if jid in (st.get("modified") or []):
        return "modified"
    if jid in (st.get("untracked") or []):
        return "untracked"
    return None


def exit_code(st: dict) -> int:
    """70 > 2 > 73 > 11 > 5 > 4 > 0 (only the generation codes here)."""
    state = st.get("state")
    if state in LIVE:
        return util.RUNNING
    codes = set()
    failed = st.get("failed") or []
    if any(f.get("reason") == "environment" for f in failed):
        codes.add(util.BAD)
    if state == "busy":
        codes.add(util.BUSY)
    if state in ("crashed", "cancelled", "orphan_child"):
        codes.add(util.FAILED)
    if any(f.get("reason") != "environment" for f in failed):
        codes.add(util.FAILED)
    for b in st.get("blocked") or []:
        c = b.get("cause")
        if c == "blocked":
            c = _root_cause(st, b.get("by"))
        if c == "failed":
            codes.add(util.FAILED)
        elif c in ("modified", "untracked"):
            codes.add(util.HUMAN)
    if st.get("modified") or st.get("untracked"):
        codes.add(util.HUMAN)
    for c in (util.INTERNAL, util.BAD, util.BUSY, util.FAILED, util.HUMAN, util.PROBLEMS):
        if c in codes:
            return c
    return util.OK
