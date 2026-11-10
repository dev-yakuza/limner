"""Job files (spec §6.2): validation, dependency stages, job_sha, reference letterboxing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import paths, util
from .util import bad

MAX_SIDE = 1024
MAX_SEED = 2**32 - 1
MAX_PROMPT = 1500
REF_MAX_BYTES = 20 * 1024 * 1024
REF_MAX_SIDE = 4096
REF_FORMATS = {"PNG", "JPEG", "WEBP"}
JOB_KEYS = {"id", "out", "width", "height", "prompt", "refs", "seed"}
TOP_KEYS = {"version", "model", "jobs"}


@dataclass
class Job:
    id: str
    out: str  # real absolute path
    out_rel: str
    width: int
    height: int
    prompt: str
    seed: int
    refs: list = field(default_factory=list)  # [{"path": real, "rel": rel, "producer": job id | None}]
    stage: int = 1


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_ref_file(path: str, what: str) -> None:
    from PIL import Image

    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise bad("%s: %s" % (what, e))
    if size > REF_MAX_BYTES:
        raise bad("%s: larger than 20 MB" % what)
    try:
        with Image.open(path) as im:
            fmt, (w, h) = im.format, im.size
    except Exception as e:  # noqa: BLE001 - any decoder error is a bad input
        raise bad("%s: not a readable image (%s)" % (what, e))
    if fmt not in REF_FORMATS:
        raise bad("%s: %s is not PNG, JPEG or WebP" % (what, fmt))
    if max(w, h) > REF_MAX_SIDE:
        raise bad("%s: the long side is over %d px" % (what, REF_MAX_SIDE))


def validate(spec, root: paths.Root, lock: dict) -> list[Job]:
    """Check a parsed job file; returns jobs with stages set. Exit 2 on any problem."""
    gen = lock["generation"]
    if not isinstance(spec, dict):
        raise bad("job file: expected an object")
    extra = set(spec) - TOP_KEYS
    if extra:
        raise bad("job file: unknown keys %s" % sorted(extra))
    if spec.get("version") != 1:
        raise bad("job file: version must be 1")
    if spec.get("model") != gen["model"]:
        raise bad("job file: model must be %r" % gen["model"])
    raw = spec.get("jobs")
    if not isinstance(raw, list) or not raw:
        raise bad("job file: jobs must be a non-empty list")
    jobs: list[Job] = []
    ids, outs = {}, {}
    for i, j in enumerate(raw):
        at = "jobs[%d]" % i
        if not isinstance(j, dict):
            raise bad("%s: expected an object" % at)
        extra = set(j) - JOB_KEYS
        if extra:
            raise bad("%s: unknown keys %s" % (at, sorted(extra)))
        jid = j.get("id")
        if not isinstance(jid, str) or not jid.strip() or len(jid) > 200 or "," in jid:
            raise bad("%s: id must be a non-empty string without commas" % at)
        at = "job %s" % jid
        for k in ("width", "height"):
            v = j.get(k)
            if not _is_int(v) or not 64 <= v <= MAX_SIDE or v % 16:
                raise bad("%s: %s must be a multiple of 16 from 64 to %d" % (at, k, MAX_SIDE))
        seed = j.get("seed")
        if not _is_int(seed) or not 0 <= seed <= MAX_SEED:
            raise bad("%s: seed must be an integer from 0 to %d" % (at, MAX_SEED))
        prompt = j.get("prompt")
        if not isinstance(prompt, str):
            raise bad("%s: prompt must be a string" % at)
        prompt = util.nfc(prompt)
        if not prompt.strip() or len(prompt) > MAX_PROMPT:
            raise bad("%s: prompt must have 1 to %d characters" % (at, MAX_PROMPT))
        refs = j.get("refs", [])
        if not isinstance(refs, list) or len(refs) > lock["generation"]["max_refs"] or \
                not all(isinstance(r, str) and r for r in refs):
            raise bad("%s: refs must be a list of 0 to %d paths" % (at, lock["generation"]["max_refs"]))
        out_raw = j.get("out")
        if not isinstance(out_raw, str):
            raise bad("%s: out must be a path" % at)
        out = root.writable(out_raw, "%s: out" % at)
        key = util.fold(jid)
        if key in ids:
            raise bad("%s: id appears twice" % at)
        ids[key] = jid
        if util.fold(out) in outs:
            raise bad("%s: out %s is also the out of job %s" % (at, out_raw, outs[util.fold(out)]))
        outs[util.fold(out)] = jid
        jobs.append(Job(id=jid, out=out, out_rel=root.rel(out), width=j["width"], height=j["height"],
                        prompt=prompt, seed=seed, refs=list(refs)))
    by_out = {util.fold(j.out): j for j in jobs}
    for j in jobs:
        resolved = []
        for r in j.refs:
            real = root.inside(r, "job %s: ref" % j.id)
            prod = by_out.get(util.fold(real))
            if prod is None:
                if not os.path.isfile(real):
                    raise bad("job %s: ref %s is not a file and no job makes it" % (j.id, r),
                              missing_input=r)
                check_ref_file(real, "job %s: ref %s" % (j.id, r))
            resolved.append({"path": real, "rel": root.rel(real), "producer": prod.id if prod else None})
        j.refs = resolved
    _stages(jobs)
    return jobs


def _stages(jobs: list[Job]) -> None:
    """stage = dependency depth (1 = no job refs); a cycle is exit 2."""
    by_id = {j.id: j for j in jobs}
    depth: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(jid: str) -> int:
        if jid in depth:
            return depth[jid]
        if jid in visiting:
            raise bad("jobs: dependency cycle through %s" % jid)
        visiting.add(jid)
        d = 1
        for r in by_id[jid].refs:
            if r["producer"]:
                d = max(d, visit(r["producer"]) + 1)
        visiting.discard(jid)
        depth[jid] = d
        return d

    for j in jobs:
        j.stage = visit(j.id)


def stages(jobs: list[Job]) -> list[list[Job]]:
    n = max(j.stage for j in jobs)
    return [[j for j in jobs if j.stage == s] for s in range(1, n + 1)]


def job_sha(job: Job, ref_shas: list[str], lock: dict) -> str:
    """sha256 of everything that decides the pixels (spec §6.4); not id, not out."""
    gen = lock["generation"]
    q = lock["quantized"]
    lo = lock["lora"]
    body = {
        "width": job.width,
        "height": job.height,
        "prompt": util.nfc(job.prompt),
        "seed": job.seed,
        "refs": list(ref_shas),
        "model_revision": lock["models"][q["from"]]["revision"],
        "bits": q["bits"],
        "lora_revision": lock["models"][lo["model"]]["revision"],
        "lora_file": lo["file"],
        "steps": gen["steps"],
        "guidance": gen["guidance"],
        "multi_ref": gen["multi_ref"],
        "skip_negative": gen["skip_negative"],
        "blank": gen["blank"],
        "letterbox": gen["letterbox"],
    }
    return util.sha256_bytes(util.canonical(body).encode("utf-8"))


def hex_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def letterbox(src: str, width: int, height: int, blank: str, dst: str) -> None:
    """A copy of `src` fitted inside width x height on the blank color (rule version 1): mflux
    stretches references to the output size, ignoring their aspect ratio."""
    from PIL import Image, ImageOps

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = min(width / im.width, height / im.height)
        w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
        im = im.resize((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), hex_rgb(blank))
        canvas.paste(im, ((width - w) // 2, (height - h) // 2))
    import io

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    paths.write_new_file(dst, buf.getvalue())


def blank_canvas(width: int, height: int, blank: str, dst: str) -> None:
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (width, height), hex_rgb(blank)).save(buf, format="PNG")
    paths.write_new_file(dst, buf.getvalue())
