"""Paths stay inside `--root` (spec §5), and the by-products live in the state dir.

Every path is resolved with symlinks followed and compared NFC + casefolded, so `Out/A.png`, a
symlink out of the root and an NFD spelling all land on the same verdict. Writes go through a
directory file descriptor with O_NOFOLLOW / rename (a rename replaces a symlink, never follows it).
"""

from __future__ import annotations

import os

from . import util
from .util import bad

FORBIDDEN = {".git", ".claude"}


class Root:
    def __init__(self, root: str, state_dir: str | None = None):
        if not root:
            raise bad("--root is required")
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise bad("--root %s is not a directory" % self.root)
        self.real = os.path.realpath(self.root)
        sd = os.path.abspath(state_dir) if state_dir else os.path.join(self.real, ".limner")
        self.state_dir = self.inside(sd, what="--state-dir")
        self.state_rel = self.rel(self.state_dir)

    # ------------------------------------------------------------------ checks

    def _within(self, real: str) -> bool:
        r, x = util.fold(self.real), util.fold(real)
        return x == r or x.startswith(r.rstrip("/") + "/")

    def inside(self, path: str, what: str = "path") -> str:
        """The real (symlinks resolved) absolute path of `path` (relative = to the root);
        exit 2 when it leaves the root."""
        if not isinstance(path, str) or not path or "\x00" in path:
            raise bad("%s: not a path: %r" % (what, path))
        p = path if os.path.isabs(path) else os.path.join(self.real, path)
        real = os.path.realpath(p)
        if not self._within(real):
            raise bad("%s %s is outside --root %s" % (what, path, self.root))
        return real

    def rel(self, real: str) -> str:
        if util.fold(real) == util.fold(self.real):
            return "."
        return real[len(self.real.rstrip("/")) + 1:]

    def writable(self, path: str, what: str = "out", png: bool = True) -> str:
        real = self.inside(path, what)
        parts = [util.fold(x) for x in self.rel(real).split("/")]
        if any(x in FORBIDDEN for x in parts):
            raise bad("%s %s: writing under .git/ or .claude/ is refused" % (what, path))
        if png and not real.casefold().endswith(".png"):
            raise bad("%s %s: only .png outputs" % (what, path))
        if os.path.isdir(real):
            raise bad("%s %s is a directory" % (what, path))
        return real

    def readable(self, path: str, what: str = "input") -> str:
        real = self.inside(path, what)
        if not os.path.isfile(real):
            raise bad("%s %s does not exist" % (what, path), missing_input=path)
        return real

    # ------------------------------------------------------------------ state dir

    def sd(self, *parts: str) -> str:
        return os.path.join(self.state_dir, *parts)

    def record_path(self, out_real: str) -> str:
        return self.sd("records", self.rel(out_real) + ".json")

    def prev_base(self, out_real: str, stamp: str) -> str:
        rel = self.rel(out_real)
        stem = rel[:-4] if rel.casefold().endswith(".png") else rel
        return self.sd("prev", "%s.%s" % (stem, stamp))

    def tmp_dir(self) -> str:
        d = self.sd("tmp")
        safe_makedirs(self, d)
        return d

    def state_dir_warnings(self, outs: list[str]) -> list[str]:
        sd = util.fold(self.state_dir)
        for o in outs:
            parent = util.fold(os.path.dirname(o))
            if sd == parent or sd.startswith(parent.rstrip("/") + "/"):
                return ["state_dir_in_output_dir"]
        return []


def safe_makedirs(root: Root, path: str) -> None:
    """mkdir -p, refusing to create anything outside the root (checked again after)."""
    real = root.inside(path, "directory")
    os.makedirs(real, exist_ok=True)
    if not root._within(os.path.realpath(real)):
        raise bad("directory %s left --root" % path)


def tmp_name(suffix: str) -> str:
    """`<pid>-<start ms>-<rand><suffix>`: the owner of a temporary file can be judged dead later."""
    me = util.me()
    ms = util.start_ms(me["pid_start"] or "") or 0
    return "%d-%d-%s%s" % (me["pid"], ms, util.rand_hex(6), suffix)


def tmp_owner_dead(name: str) -> bool:
    """True when the process named in a temporary file's name is gone."""
    try:
        pid_s, ms_s = name.split("-", 2)[:2]
        pid, ms = int(pid_s), int(ms_s)
    except ValueError:
        return True
    start = util.proc_start(pid)
    if start is None:
        return True
    got = util.start_ms(start)
    if got is None:
        return True
    import sys

    slack = 2000 if sys.platform.startswith("linux") else 0
    return abs(got - ms) > slack


def _dir_fd(path: str) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def write_new_file(path: str, data: bytes) -> None:
    """Create `path` (must not exist; no symlink followed)."""
    d, name = os.path.split(path)
    fd_dir = _dir_fd(d)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=fd_dir)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    finally:
        os.close(fd_dir)


def link_nofollow(src: str, dst: str) -> None:
    sd, sn = os.path.split(src)
    dd, dn = os.path.split(dst)
    a, b = _dir_fd(sd), _dir_fd(dd)
    try:
        os.link(sn, dn, src_dir_fd=a, dst_dir_fd=b, follow_symlinks=False)
    finally:
        os.close(a)
        os.close(b)


def rename_into(src: str, dst: str) -> None:
    sd, sn = os.path.split(src)
    dd, dn = os.path.split(dst)
    a, b = _dir_fd(sd), _dir_fd(dd)
    try:
        os.rename(sn, dn, src_dir_fd=a, dst_dir_fd=b)
    finally:
        os.close(a)
        os.close(b)


def is_symlink(path: str) -> bool:
    return os.path.islink(path)


def commit(root: Root, out: str, tmp_png: str, record: dict) -> dict:
    """The write order of spec §6.4 (also used by compose and rasterize):

    PNG in tmp/ (already there) -> record in tmp/ -> old out hard-linked into prev/ -> rename the
    PNG over out (atomic) -> old record next to its prev file -> rename the record into records/.
    `out` is never missing at any point; a death before the last rename leaves an orphan record
    that the next worker adopts. Returns {"record": path, "prev": path | None}.
    """
    safe_makedirs(root, os.path.dirname(out))
    if is_symlink(out):
        raise bad("out %s is a symlink" % out)
    stamp = util.utc_stamp()
    prev_png = None
    if os.path.exists(out):
        prev_png = root.prev_base(out, stamp) + ".png"
    rec = dict(record)
    rec["out"] = root.rel(out)
    rec["prev"] = root.rel(prev_png) if prev_png else None
    tmp_rec = os.path.join(root.tmp_dir(), tmp_name(".record.json"))
    write_new_file(tmp_rec, (util.dumps(rec) + "\n").encode("utf-8"))
    if prev_png:
        safe_makedirs(root, os.path.dirname(prev_png))
        link_nofollow(out, prev_png)
    rename_into(tmp_png, out)
    rpath = root.record_path(out)
    safe_makedirs(root, os.path.dirname(rpath))
    if prev_png and os.path.exists(rpath):
        rename_into(rpath, prev_png[:-4] + ".json")
    rename_into(tmp_rec, rpath)
    return {"record": rpath, "prev": prev_png}


def adopt_orphans(root: Root, out: str) -> bool:
    """Move an orphan record of `out` from tmp/ into records/ (spec §6.4 adoption): its `out`
    is this file, its `out_sha256` is the file's hash now, and its writer is dead."""
    tdir = root.sd("tmp")
    try:
        names = sorted(os.listdir(tdir))
    except OSError:
        return False
    if not os.path.isfile(out):
        return False
    cur = None
    adopted = False
    for name in names:
        if not name.endswith(".record.json"):
            continue
        p = os.path.join(tdir, name)
        rec = util.read_json_or_none(p)
        if not isinstance(rec, dict) or not isinstance(rec.get("out"), str):
            continue
        if util.fold(rec["out"]) != util.fold(root.rel(out)):
            continue
        if cur is None:
            cur = util.sha256_file(out)
        if rec.get("out_sha256") != cur or not tmp_owner_dead(name):
            continue
        rpath = root.record_path(out)
        safe_makedirs(root, os.path.dirname(rpath))
        if os.path.exists(rpath):
            prev = rec.get("prev")
            if isinstance(prev, str) and prev.casefold().endswith(".png"):
                dst = root.inside(prev[:-4] + ".json", "prev")
            else:
                dst = root.prev_base(out, util.utc_stamp()) + ".json"
            safe_makedirs(root, os.path.dirname(dst))
            rename_into(rpath, dst)
        rename_into(p, rpath)
        adopted = True
    return adopted


def stale_tmp(root: Root) -> list[str]:
    """Temporary files whose writer is gone (reported only; never removed)."""
    tdir = root.sd("tmp")
    try:
        names = sorted(os.listdir(tdir))
    except OSError:
        return []
    return [n for n in names if tmp_owner_dead(n)]
