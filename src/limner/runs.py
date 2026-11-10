"""Where a run and its attempts live: `<state>/runs/<run_id>/` (spec §5, §6.1)."""

from __future__ import annotations

import os
import re

from . import util

RUN_ID = re.compile(r"^[0-9a-f]{16}$")
ATTEMPT = re.compile(r"^[0-9a-f]{16}$")


class Run:
    def __init__(self, state_dir: str, run_id: str):
        if not RUN_ID.match(run_id or ""):
            raise util.bad("--run-id must be 16 hex characters")
        self.state_dir = state_dir
        self.run_id = run_id
        self.dir = os.path.join(state_dir, "runs", run_id)
        self.lock = os.path.join(self.dir, "run.lock.d")
        self.current = os.path.join(self.dir, "current.json")

    def attempt_dir(self, attempt: str) -> str:
        if not ATTEMPT.match(attempt or ""):
            raise util.bad("--attempt must be 16 hex characters")
        return os.path.join(self.dir, "attempts", attempt)

    def current_attempt(self) -> str | None:
        d = util.read_json_or_none(self.current)
        a = d.get("attempt") if isinstance(d, dict) else None
        return a if isinstance(a, str) and ATTEMPT.match(a) else None


class Attempt:
    def __init__(self, run: Run, attempt: str):
        self.run = run
        self.id = attempt
        self.dir = run.attempt_dir(attempt)

    def p(self, name: str) -> str:
        return os.path.join(self.dir, name)

    @property
    def status_file(self) -> str:
        return self.p("status.json")

    def read_status(self) -> dict | None:
        d = util.read_json_or_none(self.status_file)
        return d if isinstance(d, dict) else None

    def write_status(self, st: dict) -> None:
        st["updated_at"] = util.utc_now()
        util.write_json_atomic(self.status_file, st)

    def cancel_requested(self) -> bool:
        return os.path.exists(self.p("cancel_requested"))

    def read(self, name: str) -> dict | None:
        d = util.read_json_or_none(self.p(name))
        return d if isinstance(d, dict) else None


def run_id_of(job_spec, force: list[str], accept: list[str]) -> str:
    body = util.canonical({"job": job_spec, "force": sorted(force), "accept": sorted(accept)})
    return util.sha256_bytes(body.encode("utf-8"))[:16]
