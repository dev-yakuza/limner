"""Small shared helpers: errors with exit codes, JSON files, hashes, processes."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import secrets
import subprocess
import sys
import unicodedata

from . import dirlock

CONTRACT = 1

# exit codes (spec §10)
OK, BAD, PROBLEMS, HUMAN, RUNNING, FAILED, NOT_FOUND, INTERNAL, BUSY = 0, 2, 4, 5, 10, 11, 12, 70, 73


class Fail(Exception):
    """Ends a command with `code`; `extra` is merged into the JSON printed on stdout."""

    def __init__(self, code: int, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def bad(message: str, **extra) -> Fail:
    return Fail(BAD, message, **extra)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def fold(s: str) -> str:
    """The key two names are compared by (NFC + casefold)."""
    return nfc(s).casefold()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file_or_none(path: str) -> str | None:
    try:
        return sha256_file(path)
    except FileNotFoundError:
        return None


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    """A UTC time for file names (sortable, unique enough with the microseconds)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rand_hex(n: int = 8) -> str:
    return secrets.token_hex(n)


def reject_constant(name):
    raise ValueError("not JSON: %s" % name)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read(), parse_constant=reject_constant)


def read_json_or_none(path: str):
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def write_json_atomic(path: str, obj, mode: int = 0o644) -> None:
    """Write next to `path`, then rename over it (readers never see a half-written file)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".%s.%s.tmp" % (os.path.basename(path), rand_hex(4)))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
        os.rename(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text_atomic(path: str, text: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".%s.%s.tmp" % (os.path.basename(path), rand_hex(4)))
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.rename(tmp, path)


# --------------------------------------------------------------------------- processes


def proc_start(pid) -> str | None:
    """`LC_ALL=C TZ=UTC ps -o lstart=` of pid (None when gone)."""
    try:
        return dirlock.proc_start(pid)
    except dirlock.CannotCheck:
        return None


def proc_state(pid) -> tuple[str | None, str | None]:
    """(lstart, stat) of a process, or (None, None)."""
    if not dirlock.valid_pid(pid):
        return None, None
    env = dict(os.environ, LC_ALL="C", TZ="UTC")
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-o", "lstart=", "-p", str(int(pid))], env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as e:  # e.g. fork fails under memory pressure: we do not know
        raise dirlock.CannotCheck("cannot run ps: %s" % e)
    if out.returncode < 0:
        raise dirlock.CannotCheck("ps ended by signal %d" % -out.returncode)
    if out.returncode != 0:
        return None, None
    text = out.stdout.decode("utf-8", "replace").strip()
    if not text:
        return None, None
    stat, _, lstart = text.partition(" ")
    return lstart.strip() or None, stat


def alive(pid, pid_start) -> bool:
    """Alive = same pid and start time, and not a zombie (spec §6.1 `status`)."""
    if not pid or not pid_start:
        return False
    try:
        lstart, stat = proc_state(pid)
    except dirlock.CannotCheck:
        return True  # like dirlock: a process we cannot check is taken as alive
    if lstart is None:
        return False
    if not dirlock.same_start(lstart, pid_start):
        return False
    return not (stat or "").startswith("Z")


def start_ms(pid_start: str) -> int | None:
    return dirlock.parse_lstart(pid_start)


def me() -> dict:
    """{"pid", "pid_start"} of this process."""
    pid = os.getpid()
    return {"pid": pid, "pid_start": proc_start(pid)}


def is_apple_silicon() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() == "arm64"


def ram_gib() -> float:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout.strip()) / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError):
        return 0.0


def free_gib() -> float | None:
    """free + inactive + purgeable from `vm_stat`, GiB (a lower bound of what macOS can give)."""
    if sys.platform != "darwin":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return round(int(line.split()[1]) / 2**20, 1)
        except OSError:
            return None
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except OSError:
        return None
    page = 4096
    vals = {}
    for line in out.splitlines():
        if "page size of" in line:
            try:
                page = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        k, _, v = line.partition(":")
        try:
            vals[k.strip()] = int(v.strip().rstrip("."))
        except ValueError:
            pass
    pages = vals.get("Pages free", 0) + vals.get("Pages inactive", 0) + vals.get("Pages purgeable", 0)
    return round(pages * page / 2**30, 1)


def disk_free_gib(path: str) -> float | None:
    p = path
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    try:
        st = os.statvfs(p or "/")
    except OSError:
        return None
    return round(st.f_bavail * st.f_frsize / 2**30, 1)


def reset_signals() -> None:
    """Workers and their children: empty signal mask, default SIGTERM/SIGINT (spec §6.1)."""
    import signal

    signal.pthread_sigmask(signal.SIG_SETMASK, ())
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    d = os.environ.get("LIMNER_TEST_SIGNAL_REPORT")
    if d:  # test hook: what this process now has
        write_json_atomic(os.path.join(d, "%d.json" % os.getpid()), {
            "argv": sys.argv,
            "blocked": sorted(int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK, [])),
            "term_default": signal.getsignal(signal.SIGTERM) == signal.SIG_DFL,
            "int_default": signal.getsignal(signal.SIGINT) == signal.SIG_DFL})
