"""Resume rules (spec §6.4): what to do with each job, judged from the files on disk.

The worker judges under both locks at the start of every stage (and adopts orphan records); `lmn
run`'s preview applies the same rules read-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import paths, util
from .jobs import Job


@dataclass
class Decision:
    action: str  # generate | skipped | untracked | modified | accept
    reason: str | None = None  # generate: None (new) | job_changed | forced
    out_sha: str | None = None  # the out file's hash when judged (None = no file)
    record: dict | None = None
    adopted: bool = False
    adoptable: bool = False


def summary(rec: dict | None) -> dict | None:
    if not isinstance(rec, dict):
        return None
    return {k: rec.get(k) for k in ("kind", "generator", "out_sha256", "job_sha") if k in rec}


def orphan_exists(root: paths.Root, out: str) -> bool:
    """Read-only twin of paths.adopt_orphans (for the preview)."""
    tdir = root.sd("tmp")
    try:
        names = os.listdir(tdir)
    except OSError:
        return False
    if not os.path.isfile(out):
        return False
    cur = None
    for name in names:
        if not name.endswith(".record.json"):
            continue
        rec = util.read_json_or_none(os.path.join(tdir, name))
        if not isinstance(rec, dict) or not isinstance(rec.get("out"), str):
            continue
        if util.fold(rec["out"]) != util.fold(root.rel(out)):
            continue
        cur = cur or util.sha256_file(out)
        if rec.get("out_sha256") == cur and paths.tmp_owner_dead(name):
            return True
    return False


def decide(root: paths.Root, job: Job, jsha: str | None, force: set, accept: set,
           adopt: bool) -> Decision:
    adopted = adoptable = False
    if adopt:
        adopted = paths.adopt_orphans(root, job.out)
    else:
        adoptable = orphan_exists(root, job.out)
    if not os.path.isfile(job.out):
        return Decision("generate", adopted=adopted, adoptable=adoptable)
    cur = util.sha256_file(job.out)
    rec = util.read_json_or_none(root.record_path(job.out))
    if adoptable:
        # the preview reads what adoption would leave: the orphan's record
        rec = None
        for name in sorted(os.listdir(root.sd("tmp"))):
            r = util.read_json_or_none(os.path.join(root.sd("tmp"), name)) if name.endswith(".record.json") else None
            if isinstance(r, dict) and util.fold(str(r.get("out"))) == util.fold(job.out_rel) and \
                    r.get("out_sha256") == cur:
                rec = r
    if not isinstance(rec, dict):
        state = "untracked"
    elif rec.get("kind") != "generate":
        state = "modified"
    elif rec.get("out_sha256") != cur:
        state = "modified"
    elif jsha is not None and rec.get("job_sha") == jsha:
        return Decision("skipped", out_sha=cur, record=rec, adopted=adopted, adoptable=adoptable)
    else:
        return Decision("generate", "job_changed", out_sha=cur, record=rec, adopted=adopted,
                        adoptable=adoptable)
    if job.id in force:
        return Decision("generate", "forced", out_sha=cur, record=rec if isinstance(rec, dict) else None,
                        adopted=adopted, adoptable=adoptable)
    if job.id in accept:
        return Decision("accept", out_sha=cur, record=rec if isinstance(rec, dict) else None,
                        adopted=adopted, adoptable=adoptable)
    return Decision(state, out_sha=cur, record=rec if isinstance(rec, dict) else None, adopted=adopted,
                    adoptable=adoptable)


def human_record(job: Job, jsha: str, out_sha: str, old: dict | None) -> dict:
    """`--accept`: the person's file becomes the generated one for this job."""
    import limner

    return {"limner": limner.__version__, "contract": util.CONTRACT, "kind": "generate", "generator": "human",
            "out": job.out_rel, "out_sha256": out_sha, "job_sha": jsha, "accepted_from": summary(old),
            "accepted_at": util.utc_now()}


def write_record(root: paths.Root, out: str, rec: dict) -> str:
    """Replace the record of `out` (the old one goes to prev/, next to nothing: the file did not
    change)."""
    rpath = root.record_path(out)
    paths.safe_makedirs(root, os.path.dirname(rpath))
    tmp = os.path.join(root.tmp_dir(), paths.tmp_name(".accept.json"))
    paths.write_new_file(tmp, (util.dumps(rec) + "\n").encode("utf-8"))
    if os.path.exists(rpath):
        dst = root.prev_base(out, util.utc_stamp()) + ".json"
        paths.safe_makedirs(root, os.path.dirname(dst))
        paths.rename_into(rpath, dst)
    paths.rename_into(tmp, rpath)
    return rpath


def blocked_by(job: Job, outcome: dict) -> dict | None:
    """{"id", "by", "cause"} when a ref's producing job failed or is untracked/modified/blocked."""
    for r in job.refs:
        p = r["producer"]
        if p and outcome.get(p) in ("failed", "modified", "untracked", "blocked"):
            return {"id": job.id, "by": p, "cause": outcome[p]}
    return None
