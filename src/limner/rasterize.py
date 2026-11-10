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
COMMENT = re.compile(r"/\*.*?\*/", re.S)
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


MAX_COMPOUNDS = 32  # parts of one CSS selector
MAX_DEPTH = 256  # element nesting of the SVG
COMPOUND = re.compile(r"(?:[A-Za-z][\w-]*|\*)?(?:[.#][\w-]+)*")


class Rule:
    """A selector of compounds (`tag`, `*`, `.class`, `#id`) joined by descendant (space) or child
    (`>`) combinators. Anything else (pseudo-classes, attributes, `+`, `~`) is unsupported: such a
    rule is left out of the classification and reported in notes."""

    def __init__(self, selector: str, families: list[str], order: int):
        self.selector = selector.strip()
        self.families = families
        self.order = order
        toks = re.findall(r"\s*(>)\s*|\s+|([^\s>]+)", self.selector)
        self.chain: list[tuple[str | None, dict]] = []  # (combinator before, compound), left to right
        self.supported = bool(self.selector)
        comb = None
        for gt, comp in toks:
            if gt:
                comb = ">"
                continue
            if not comp:
                if comb is None and self.chain:
                    comb = " "
                continue
            if not COMPOUND.fullmatch(comp):
                self.supported = False
                break
            m = re.match(r"^([A-Za-z][\w-]*|\*)?", comp)
            tag = m.group(1) if m and m.group(1) and m.group(1) != "*" else None
            ids, classes = [], []
            for kind, name in re.findall(r"([.#])([\w-]+)", comp[m.end():]):
                (ids if kind == "#" else classes).append(name)
            self.chain.append((comb if self.chain else None, {"tag": tag, "ids": ids, "classes": classes}))
            comb = None
        if not self.chain:
            self.supported = False
        if len(self.chain) > MAX_COMPOUNDS:
            raise bad("a CSS selector has more than %d parts" % MAX_COMPOUNDS)
        self.specificity = (sum(len(c["ids"]) for _, c in self.chain), sum(len(c["classes"]) for _, c in self.chain),
                            sum(1 for _, c in self.chain if c["tag"]))

    @staticmethod
    def _one(c: dict, el) -> bool:
        if c["tag"] and local(el.tag) != c["tag"]:
            return False
        if c["ids"] and el.get("id") not in c["ids"]:
            return False
        cls = set((el.get("class") or "").split())
        return all(x in cls for x in c["classes"])

    def matches(self, el, parent: dict) -> bool:
        if not self.supported:
            return False

        memo: dict = {}  # (i, node) -> bool: without it a failing descendant chain backtracks exponentially

        def match_from(i, node) -> bool:
            key = (i, id(node))
            if key not in memo:
                memo[key] = _match_from(i, node)
            return memo[key]

        def _match_from(i, node) -> bool:
            comb, comp = self.chain[i]
            if not self._one(comp, node):
                return False
            if i == 0:
                return True
            nxt = self.chain[i][0]
            p = parent.get(node)
            if nxt == ">":
                return p is not None and match_from(i - 1, p)
            while p is not None:
                if match_from(i - 1, p):
                    return True
                p = parent.get(p)
            return False

        return match_from(len(self.chain) - 1, el)


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


DATA_MAX_SIDE, DATA_MAX_PX = 8192, 64_000_000  # compose's input limits


def check_data_image(uri: str) -> None:
    """A data: image is decoded lazily and its size checked before resvg decodes it (a 50 KB PNG
    can be 16000x16000)."""
    import base64
    import binascii

    from PIL import Image

    try:
        data = base64.b64decode(uri.split(",", 1)[1], validate=False)
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Image.DecompressionBombError:
        raise bad("an embedded data: image is over the limit (a decompression bomb)")
    except (binascii.Error, ValueError, OSError) as e:
        raise bad("an embedded data: image is not readable (%s)" % e)
    if max(w, h) > DATA_MAX_SIDE or w * h > DATA_MAX_PX:
        raise bad("an embedded data: image is %dx%d, over the limit (long side 8192, 64M px)" % (w, h))


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
    stack = [(root, 1)]
    while stack:
        el, d = stack.pop()
        if d > MAX_DEPTH:
            raise bad("the SVG nests elements deeper than %d" % MAX_DEPTH)
        stack.extend((c, d + 1) for c in el)
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
                if DATA_IMAGE.match(v):
                    check_data_image(v)
                elif not v.startswith("#"):
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
            txt = COMMENT.sub(" ", txt)  # `! /* x */ important` is still important to resvg
            el.text = FONT_IMPORTANT.sub(r"\1", txt)
        if el.get("style"):
            el.set("style", FONT_IMPORTANT.sub(r"\1", COMMENT.sub(" ", el.get("style"))))
    if removed_ext:
        s.problems.append({"code": "external_ref_removed", "layer": None, "detail": {"where": removed_ext}})
    if removed_unsafe:
        s.problems.append({"code": "unsafe_removed", "layer": None, "detail": {"where": removed_unsafe}})
    # the effective font-family of every text element, for the mono class and the report
    rules = []
    for el in root.iter():
        if local(el.tag) == "style" and el.text:
            rules += parse_css(el.text, len(rules))
    ignored = sorted({r.selector for r in rules if not r.supported})
    if ignored:
        s.notes.append({"code": "css_selector_ignored", "detail": {"selectors": ignored}})
    parent = {c: p for p in root.iter() for c in p}
    original: set[str] = set()

    def own_families(el):
        st = el.get("style")
        if st:
            f = families_of(st)
            if f is not None:
                return f
        best = None
        for r in rules:
            if r.matches(el, parent):
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

    def walk(el, inherited, in_text):
        fam = own_families(el)
        eff = fam if fam is not None else inherited
        if fam:
            original.update(x for x in fam if x.lower() not in GENERIC)
        t = local(el.tag)
        here = in_text or t in TEXT_TAGS
        if here:
            # every element that can carry text content (text, tspan, textPath, a, …) gets a class:
            # an element without one would override the forced family with its own
            mono = is_mono(eff, mono_names)
            cls = [c for c in (el.get("class") or "").split() if c not in ("lmn-mono", "lmn-sans")]
            el.set("class", " ".join(cls + ["lmn-mono" if mono else "lmn-sans"]))
            s.texts.append((el.text or "", mono))
        for c in el:
            walk(c, eff, here)
            if here and c.tail:
                s.texts.append((c.tail, is_mono(eff, mono_names)))

    walk(root, None, False)
    if original:
        s.notes.append({"code": "font_substituted", "detail": {"families": sorted(original)}})
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    return ET.tostring(root, encoding="unicode"), s


# --------------------------------------------------------------------------- size


FONT_SIZE = 16.0  # resvg-py's default font_size: what `em` means on the root element
REL_UNITS = {"em": FONT_SIZE, "ex": FONT_SIZE / 2}


def _length(v: str | None, what: str) -> float | None:
    """px of an absolute or font-relative length; None for missing or `%`; exit 2 for anything
    limner cannot size the way resvg will."""
    if v is None or not v.strip() or v.strip().endswith("%"):
        return None
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*", v)
    unit = m.group(2).lower() if m else None
    if not m or (unit not in UNITS and unit not in REL_UNITS):
        raise bad("the SVG's %s %r is not a length limner can size" % (what, v))
    return float(m.group(1)) * (UNITS.get(unit) or REL_UNITS[unit])


def fix_size(svg_text: str) -> tuple[str, float, float]:
    """The SVG's size as resvg will draw it, written back onto the root as px, so that limner's
    limit check and resvg's canvas are the same number. Without absolute sizes and a viewBox
    (resvg would use the content's bounds) the SVG is refused."""
    root = ET.fromstring(svg_text)
    w, h = _length(root.get("width"), "width"), _length(root.get("height"), "height")
    vb = root.get("viewBox")
    vw = vh = None
    if vb:
        parts = re.split(r"[\s,]+", vb.strip())
        try:
            vw, vh = float(parts[2]), float(parts[3])
        except (IndexError, ValueError):
            vw = vh = None
        if not (vw and vh and vw > 0 and vh > 0):
            raise bad("the SVG's viewBox %r is not usable" % vb)
    if w is None and h is None:
        if not vw:
            raise bad("the SVG needs absolute width/height or a viewBox (its size would come from its content)")
        w, h = vw, vh
    elif w is None:
        if not vw:
            raise bad("the SVG needs an absolute width or a viewBox")
        w = h * vw / vh
    elif h is None:
        if not vw:
            raise bad("the SVG needs an absolute height or a viewBox")
        h = w * vh / vw
    if not (w > 0 and h > 0):
        raise bad("the SVG's size %gx%g is empty" % (w, h))
    root.set("width", "%g" % w)
    root.set("height", "%g" % h)
    return ET.tostring(root, encoding="unicode"), w, h


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


NON_RENDERED = {"defs", "symbol", "clipPath", "mask", "pattern", "marker", "title", "desc", "metadata",
                "linearGradient", "radialGradient", "filter"}
VISIBLE = ("fill:#000;fill-opacity:1;stroke:#fff;stroke-width:2px;stroke-opacity:1;opacity:1;display:inline;"
           "visibility:visible;font-size:16px")


def _strip_text(root) -> None:
    for p in list(root.iter()):
        for c in list(p):
            if local(c.tag) == "text":
                p.remove(c)


def _force_visible(root) -> bool:
    """Move every rendered <text> to the end of the document (above everything), at the top left,
    plainly visible (black fill with a white outline, opaque, displayed, no transforms or
    positions). False when no text would ever be rendered."""
    parent = {c: p for p in root.iter() for c in p}
    moved = []
    for el in list(root.iter()):
        if local(el.tag) != "text":
            continue
        p, hidden = parent.get(el), False
        while p is not None:
            hidden = hidden or local(p.tag) in NON_RENDERED
            p = parent.get(p)
        if hidden:
            continue
        chars = "".join((n.text or "") for n in el.iter() if local(n.tag) not in NON_RENDERED) + \
            "".join((n.tail or "") for n in el.iter() if n is not el)
        if chars.strip():
            moved.append(el)
    for el in moved:
        parent[el].remove(el)
        for n in el.iter():
            for k in ("transform", "x", "y", "dx", "dy", "rotate"):
                n.attrib.pop(k, None)
            n.set("style", VISIBLE)
        el.set("x", "0")
        el.set("y", "0.9em")
        root.append(el)
    return bool(moved)


def _render_image(svg: str, width, background):
    from PIL import Image

    return Image.open(io.BytesIO(render(svg, width, background))).convert("RGBA")


def text_not_drawn(clean: str, im, width, background) -> bool:
    """The safety net for fonts that drop text silently: the picture equals the one without text,
    AND it still does when every text is forced visible on the canvas. Text that is invisible by
    design (white on white, display:none, off the canvas, unused in <defs>, only a <title>) is not
    reported: forcing it visible makes it differ."""
    from PIL import ImageChops

    root = ET.fromstring(clean)
    forced = ET.fromstring(clean)
    if not _force_visible(forced):
        return False
    _strip_text(root)
    bare = _render_image(ET.tostring(root, encoding="unicode"), width, background)
    if ImageChops.difference(im.convert("RGBA"), bare).getbbox(alpha_only=False) is not None:
        return False
    shown = _render_image(ET.tostring(forced, encoding="unicode"), width, background)
    return ImageChops.difference(shown, bare).getbbox(alpha_only=False) is None


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
    clean, w, h = fix_size(clean)
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
    if any(c.strip() for c, _ in s.texts) and text_not_drawn(clean, im, args.width, args.background):
        # the safety net: there is text, yet the picture is the same without it
        problems.append({"code": "text_not_drawn", "layer": None,
                         "detail": {"chars": "".join(c for c, _ in s.texts).strip()[:200]}})
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
