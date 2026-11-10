"""Fonts (spec §7.4): a fixed list downloaded by `lmn fonts install` (pinned URLs + sha256, OFL
texts next to them), never looked up on the system. Fallback is per character through a chain."""

from __future__ import annotations

import os
import urllib.request
from functools import lru_cache

from . import util
from .util import Fail, bad

_GF = "https://raw.githubusercontent.com/google/fonts/5bb28733cec313b63fbd415b574b5ae35c8aa8b8/ofl"
_PR = "https://raw.githubusercontent.com/orioncactus/pretendard/v1.3.9"

FONTS = {
    "pretendard": {
        "file": "PretendardVariable.ttf",
        "url": _PR + "/packages/pretendard/dist/public/variable/PretendardVariable.ttf",
        "sha256": "3090ccde0442bb347aa7685d9ba8b17436a60682df6e8f92a9a670de14056e22",
        "license": "Pretendard-OFL.txt",
        "license_url": _PR + "/LICENSE",
        "license_sha256": "d31ddd9f2bed32fd7e302a205cf2380ba0de6529152d239ef99cfb6f261bfc04",
    },
    "noto-sans-jp": {
        "file": "NotoSansJP[wght].ttf",
        "url": _GF + "/notosansjp/NotoSansJP%5Bwght%5D.ttf",
        "sha256": "c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f",
        "license": "NotoSansJP-OFL.txt",
        "license_url": _GF + "/notosansjp/OFL.txt",
        "license_sha256": "1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9",
    },
    "noto-sans-kr": {
        "file": "NotoSansKR[wght].ttf",
        "url": _GF + "/notosanskr/NotoSansKR%5Bwght%5D.ttf",
        "sha256": "194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252",
        "license": "NotoSansKR-OFL.txt",
        "license_url": _GF + "/notosanskr/OFL.txt",
        "license_sha256": "1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9",
    },
    "jetbrains-mono": {
        "file": "JetBrainsMono[wght].ttf",
        "url": _GF + "/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
        "sha256": "48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda",
        "license": "JetBrainsMono-OFL.txt",
        "license_url": _GF + "/jetbrainsmono/OFL.txt",
        "license_sha256": "b2fe5e8987594e9ffd1d2ca52a2f5d73eb8335243893c5d6254b5ad69269591d",
    },
}

CHAINS = {
    "ko": ["pretendard", "noto-sans-kr", "noto-sans-jp"],
    "ja": ["noto-sans-jp", "noto-sans-kr"],
    "en": ["pretendard", "noto-sans-jp"],
}
MONO = "jetbrains-mono"
# SVG text has no language: Pretendard (Hangul, Latin), then Japanese forms for kana/kanji
SVG_SANS = ["pretendard", "noto-sans-jp", "noto-sans-kr"]
LANGS = tuple(CHAINS)
ROLES = ("sans", "serif", "mono")


def fonts_dir() -> str:
    v = os.environ.get("LIMNER_FONTS_DIR")
    return v if v and os.path.isabs(v) else os.path.join(os.path.expanduser("~"), ".cache", "limner", "fonts")


def path_of(key: str) -> str:
    return os.path.join(fonts_dir(), FONTS[key]["file"])


@lru_cache(maxsize=None)
def _verified(path: str, size: int, mtime: float, sha: str) -> bool:
    return util.sha256_file(path) == sha


def installed(key: str) -> bool:
    p = path_of(key)
    try:
        st = os.stat(p)
    except OSError:
        return False
    return _verified(p, st.st_size, st.st_mtime, FONTS[key]["sha256"])


def require() -> None:
    missing = [k for k in FONTS if not installed(k)]
    if missing:
        raise bad("fonts missing or changed (%s): run `lmn fonts install`" % ", ".join(missing))


def chain(lang: str, role: str = "sans") -> list[str]:
    base = CHAINS.get(lang, CHAINS["en"])
    if role == "mono":
        return [MONO] + base
    return list(base)


@lru_cache(maxsize=None)
def cmap(path: str) -> frozenset:
    from fontTools.ttLib import TTFont

    with TTFont(path, lazy=True) as t:
        return frozenset(t.getBestCmap() or {})


@lru_cache(maxsize=None)
def family(path: str) -> str:
    """The family name the font answers to (name ID 16, else 1)."""
    from fontTools.ttLib import TTFont

    with TTFont(path, lazy=True) as t:
        n = t["name"]
        return n.getDebugName(16) or n.getDebugName(1) or os.path.basename(path)


@lru_cache(maxsize=None)
def axis_range(path: str) -> tuple[float, float] | None:
    from fontTools.ttLib import TTFont

    with TTFont(path, lazy=True) as t:
        if "fvar" not in t:
            return None
        for a in t["fvar"].axes:
            if a.axisTag == "wght":
                return a.minValue, a.maxValue
    return None


@lru_cache(maxsize=256)
def load(path: str, size: int, weight: int):
    from PIL import ImageFont

    f = ImageFont.truetype(path, max(1, int(size)), layout_engine=ImageFont.Layout.BASIC)
    rng = axis_range(path)
    if rng:
        try:
            f.set_variation_by_axes([min(max(weight, rng[0]), rng[1])])
        except Exception:  # noqa: BLE001 - not variable after all
            pass
    return f


def pick(paths_: list[str], ch: str) -> str | None:
    """The first font in the chain with a glyph for `ch`."""
    cp = ord(ch)
    for p in paths_:
        if cp in cmap(p):
            return p
    return None


def chain_paths(lang: str, role: str = "sans", file: str | None = None) -> list[str]:
    out = [file] if file else []
    return out + [path_of(k) for k in chain(lang, role)]


def all_paths() -> list[str]:
    return [path_of(k) for k in FONTS]


# --------------------------------------------------------------------------- install


def _download(url: str, sha: str, dst: str) -> None:
    tmp = "%s.%s.part" % (dst, util.rand_hex(4))
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        got = util.sha256_file(tmp)
        if got != sha:
            raise Fail(util.BAD, "%s: sha256 %s, expected %s" % (url, got, sha))
        os.rename(tmp, dst)
    except OSError as e:
        raise Fail(util.BAD, "download failed: %s (%s)" % (url, e))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def install() -> dict:
    d = fonts_dir()
    os.makedirs(d, exist_ok=True)
    got, kept = [], []
    for key, f in FONTS.items():
        for name, url, sha in ((f["file"], f["url"], f["sha256"]),
                               (f["license"], f["license_url"], f["license_sha256"])):
            p = os.path.join(d, name)
            if os.path.exists(p) and util.sha256_file(p) == sha:
                kept.append(name)
                continue
            _download(url, sha, p)
            got.append(name)
    _verified.cache_clear()
    return {"contract": util.CONTRACT, "fonts_dir": d, "downloaded": got, "kept": kept}


def cmd_install(args) -> tuple[dict, int]:
    return install(), util.OK
