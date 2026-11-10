"""`lmn rasterize` (spec §8): SVG -> PNG with resvg-py 0.5.0, after limner sanitizes the SVG.

resvg-py reads local files named by `href` (it has no switch to turn the resolver off) and drops,
without an error, text whose `font-family` is not one of the given fonts. So: no DTD/entities;
external references, scripts, event attributes and foreignObject removed; the fonts forced with a
`style_sheet` (`!important`; each text element gets exactly one of the classes `lmn-sans` /
`lmn-mono` from its effective font-family — two `!important` rules on one element cannot be
ordered, and resvg-py ignores `:not()`; the SVG's own `!important` on fonts is dropped); `svg_string` only, `resources_dir=None`, no system fonts.
"""

from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET

from . import fonts, paths, util
from .compose.command import check_out, parse_color, write_png
from .util import bad

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
MAX_SIDE, MAX_PX = 8192, 40_000_000
TEXT_TAGS = ("text", "tspan", "textPath")
DATA_IMAGE = re.compile(r"^\s*data:image/(png|jpeg|webp);base64,", re.I)
URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I)
IMPORT = re.compile(r"@import[^;]*;?", re.I)
FONT_IMPORTANT = re.compile(r"(font(?:-family)?\s*:[^;}]*?)\s*!\s*important", re.I)
GENERIC = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui", "ui-sans-serif",
           "ui-serif", "ui-monospace", "math", "emoji", "fangsong"}
MONO_NAMES = {"monospace", "ui-monospace", "jetbrains mono", "courier", "courier new", "menlo", "monaco",
              "consolas", "sf mono", "fira code", "fira mono", "source code pro", "dejavu sans mono",
              "liberation mono", "roboto mono", "ibm plex mono", "cascadia code", "noto sans mono"}
UNITS = {"": 1.0, "px": 1.0, "pt": 4 / 3, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


# --------------------------------------------------------------------------- CSS (just enough)


class Rule:
    def __init__(self, selector: str, families: list[str], order: int):
        self.selector = selector.strip()
        self.families = families
        self.order = order
        last = self.selector.split()[-1] if self.selector else ""
        self.tag = None
        self.ids, self.classes = [], []
        m = re.match(r"^([A-Za-z][\w-]*|\*)?", last)
        if m and m.group(1) and m.group(1) != "*":
            self.tag = m.group(1)
        rest = last[m.end():] if m else last
        for kind, name in re.findall(r"([.#])([\w-]+)", rest):
            (self.ids if kind == "#" else self.classes).append(name)
        self.supported = re.fullmatch(r"(?:[A-Za-z][\w-]*|\*)?(?:[.#][\w-]+)*", last) is not None
        self.specificity = (len(self.ids), len(self.classes), 1 if self.tag else 0)

    def matches(self, el) -> bool:
        if not self.supported:
            return False
        if self.tag and local(el.tag) != self.tag:
            return False
        if self.ids and el.get("id") not in self.ids:
            return False
        cls = set((el.get("class") or "").split())
        return all(c in cls for c in self.classes)


def families_of(decls: str) -> list[str] | None:
    """font-family (or the `font` shorthand) of a declaration block, the last one winning."""
    found = None
    for d in decls.split(";"):
        k, _, v = d.partition(":")
        k, v = k.strip().lower(), v.replace("!important", "").strip()
        if k == "font-family" and v:
            found = split_families(v)
        elif k == "font" and v:
            m = re.search(r"[\d.]+(?:px|pt|em|rem|%|pc|mm|cm|in)?(?:\s*/\s*\S+)?\s+(.+)$", v)
            if m:
                found = split_families(m.group(1))
    return found


def split_families(v: str) -> list[str]:
    return [f.strip().strip("'\"").strip() for f in v.split(",") if f.strip().strip("'\"").strip()]


def parse_css(css: str, start: int = 0) -> list[Rule]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = []
    for i, m in enumerate(re.finditer(r"([^{}]+)\{([^{}]*)\}", css)):
        fam = families_of(m.group(2))
        if fam is None:
            continue
        for sel in m.group(1).split(","):
            rules.append(Rule(sel, fam, start + i))
    return rules


def is_mono(families: list[str] | None, mono_names: set[str]) -> bool:
    return bool(families) and any(f.lower() in mono_names for f in families)


# --------------------------------------------------------------------------- sanitize


class Sanitized:
    def __init__(self):
        self.problems: list[dict] = []
        self.notes: list[dict] = []
        self.texts: list[tuple[str, bool]] = []  # (characters, mono)


def _clean_urls(value: str) -> tuple[str, bool]:
    changed = False

    def rep(m):
        nonlocal changed
        if m.group(2).strip().startswith("#"):
            return m.group(0)
        changed = True
        return "none"

    out = URL.sub(rep, value)
    return out, changed


def sanitize(svg_text: str, mono_names: set[str]) -> tuple[str, Sanitized]:
    s = Sanitized()
    if re.search(r"<!DOCTYPE|<!ENTITY", svg_text, re.I):
        raise bad("the SVG has a DTD or entities: refused")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        raise bad("not a well-formed SVG: %s" % e)
    if local(root.tag) != "svg":
        raise bad("the root element is not <svg>")
    removed_ext, removed_unsafe = [], []
    parent = {c: p for p in root.iter() for c in p}
    for el in list(root.iter()):
        t = local(el.tag)
        if t in ("script", "foreignObject"):
            p = parent.get(el)
            if p is not None:
                p.remove(el)
                removed_unsafe.append(t)
            continue
        for k in list(el.attrib):
            lk = local(k)
            if lk.lower().startswith("on"):
                del el.attrib[k]
                removed_unsafe.append("%s@%s" % (t, lk))
            elif lk == "href":
                v = el.attrib[k].strip()
                if not (v.startswith("#") or DATA_IMAGE.match(v)):
                    del el.attrib[k]
                    removed_ext.append("%s@href" % t)
            elif "url(" in el.attrib[k].lower():
                new, changed = _clean_urls(el.attrib[k])
                if changed:
                    el.attrib[k] = new
                    removed_ext.append("%s@%s" % (t, lk))
        if t == "style" and el.text:
            txt = el.text
            if IMPORT.search(txt):
                txt = IMPORT.sub("", txt)
                removed_ext.append("style@import")
            txt, changed = _clean_urls(txt)
            if changed:
                removed_ext.append("style url()")
            el.text = FONT_IMPORTANT.sub(r"\1", txt)
        if el.get("style"):
            el.set("style", FONT_IMPORTANT.sub(r"\1", el.get("style")))
    if removed_ext:
        s.problems.append({"code": "external_ref_removed", "layer": None, "detail": {"where": removed_ext}})
    if removed_unsafe:
        s.problems.append({"code": "unsafe_removed", "layer": None, "detail": {"where": removed_unsafe}})
    # the effective font-family of every text element, for the mono class and the report
    rules = []
    for el in root.iter():
        if local(el.tag) == "style" and el.text:
            rules += parse_css(el.text, len(rules))
    original: set[str] = set()

    def own_families(el):
        st = el.get("style")
        if st:
            f = families_of(st)
            if f is not None:
                return f
        best = None
        for r in rules:
            if r.matches(el):
                key = (r.specificity, r.order)
                if best is None or key >= best[0]:
                    best = (key, r.families)
        if best:
            return best[1]
        if el.get("font-family"):
            return split_families(el.get("font-family"))
        f = el.get("font")
        if f:
            return families_of("font:" + f)
        return None

    def walk(el, inherited):
        fam = own_families(el)
        eff = fam if fam is not None else inherited
        if fam:
            original.update(x for x in fam if x.lower() not in GENERIC)
        t = local(el.tag)
        if t in TEXT_TAGS:
            mono = is_mono(eff, mono_names)
            cls = [c for c in (el.get("class") or "").split() if c not in ("lmn-mono", "lmn-sans")]
            el.set("class", " ".join(cls + ["lmn-mono" if mono else "lmn-sans"]))
            chars = (el.text or "")
            s.texts.append((chars, mono))
        for c in el:
            walk(c, eff)
            if local(c.tag) in TEXT_TAGS or local(el.tag) in TEXT_TAGS:
                if c.tail and local(el.tag) in TEXT_TAGS:
                    s.texts.append((c.tail, is_mono(eff, mono_names)))

    walk(root, None)
    if original:
        s.notes.append({"code": "font_substituted", "detail": {"families": sorted(original)}})
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    return ET.tostring(root, encoding="unicode"), s


# --------------------------------------------------------------------------- size


def _length(v: str | None) -> float | None:
    if not v:
        return None
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*([a-z]*)\s*", v)
    if not m or m.group(2) not in UNITS:
        return None
    return float(m.group(1)) * UNITS[m.group(2)]


def intrinsic_size(svg_text: str) -> tuple[float, float]:
    root = ET.fromstring(svg_text)
    w, h = _length(root.get("width")), _length(root.get("height"))
    vb = root.get("viewBox")
    vw = vh = None
    if vb:
        parts = re.split(r"[\s,]+", vb.strip())
        if len(parts) == 4:
            try:
                vw, vh = float(parts[2]), float(parts[3])
            except ValueError:
                pass
    if w is None and h is None:
        w, h = vw or 100.0, vh or 100.0
    elif w is None:
        w = h * (vw / vh) if vw and vh else h
    elif h is None:
        h = w * (vh / vw) if vw and vh else w
    return w, h


def check_size(w: float, h: float, what: str) -> None:
    if max(w, h) > MAX_SIDE or w * h > MAX_PX:
        raise bad("%s %dx%d is over the limit (long side 8192, 40M px)" % (what, w, h))


# --------------------------------------------------------------------------- command


def style_sheet() -> tuple[str, list[str]]:
    sans = [fonts.family(fonts.path_of(k)) for k in fonts.SVG_SANS]
    mono = [fonts.family(fonts.path_of(fonts.MONO))] + sans
    q = lambda fs: ", ".join('"%s"' % f for f in fs)  # noqa: E731
    # resvg-py 0.5.0 ignores `:not(...)` selectors (measured), so every text element gets exactly
    # one of two classes from the sanitizer: lmn-sans or lmn-mono.
    css = (".lmn-sans { font-family: %s !important; }\n"
           ".lmn-mono { font-family: %s !important; }\n" % (q(sans), q(mono)))
    return css, sans + mono[:1]


def render(svg_text: str, width: int | None, background: str | None) -> bytes:
    import resvg_py

    css, _ = style_sheet()
    kw = {"svg_string": svg_text, "resources_dir": None, "skip_system_fonts": True, "font_files": fonts.all_paths(),
          "style_sheet": css, "log_information": False}
    if width:
        kw["width"] = int(width)
    if background:
        kw["background"] = background
    return bytes(resvg_py.svg_to_bytes(**kw))


def cmd_rasterize(args) -> tuple[dict, int]:
    root = paths.Root(args.root, args.state_dir)
    fonts.require()
    src = root.readable(args.svg, "--svg")
    out = check_out(root, args.out, "rasterize", [src])
    try:
        raw = open(src, "rb").read()
        svg_text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise bad("--svg %s: %s" % (args.svg, e))
    mono_names = MONO_NAMES | {fonts.family(fonts.path_of(fonts.MONO)).lower()}
    clean, s = sanitize(svg_text, mono_names)
    w, h = intrinsic_size(clean)
    check_size(w, h, "the SVG")
    if args.width is not None:
        if args.width <= 0:
            raise bad("--width must be positive")
        check_size(args.width, h * args.width / w, "the output")
    if args.background:
        parse_color(args.background, "--background")
    problems = list(s.problems)
    missing = set()
    for chars, mono in s.texts:
        chain = [fonts.path_of(k) for k in ([fonts.MONO] if mono else []) + fonts.SVG_SANS]
        for ch in chars:
            if ch.isspace():
                continue
            if fonts.pick(chain, ch) is None:
                missing.add(ch)
    if missing:
        problems.append({"code": "missing_glyph", "layer": None, "detail": {"chars": "".join(sorted(missing))}})
    png = render(clean, args.width, args.background)
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    im.load()
    san_path = root.sd("sanitized", root.rel(out)[:-4] + ".svg")
    paths.safe_makedirs(root, os.path.dirname(san_path))
    util.write_text_atomic(san_path, clean)
    import limner

    record = {"limner": limner.__version__, "contract": util.CONTRACT, "kind": "rasterize",
              "svg": root.rel(src), "svg_sha256": util.sha256_bytes(raw),
              "sanitized": root.rel(san_path), "sanitized_sha256": util.sha256_bytes(clean.encode("utf-8")),
              "width": im.width, "height": im.height, "background": args.background,
              "fonts": [{"file": os.path.basename(p), "sha256": util.sha256_file(p)} for p in fonts.all_paths()],
              "resvg_py": _version("resvg-py"), "created_at": util.utc_now()}
    res = write_png(root, out, im, record)
    obj = {"contract": util.CONTRACT, "out": root.rel(out), "out_sha256": res["out_sha256"], "record": res["record"],
           "problems": problems, "notes": s.notes, "layers": []}
    return obj, (util.PROBLEMS if problems else util.OK)


def _version(d):
    from importlib.metadata import version

    return version(d)
