"""Text in a box: runs, per-character font fallback, fitting, drawing (spec §7.1, §7.3, §7.4).

Layout is Pillow's BASIC engine (fixed, spec §2): an advance per character, no kerning, no
shaping — so text is NFC-normalized first, and what BASIC cannot draw is reported.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from .. import fonts
from . import linebreak

LINE_HEIGHT = 1.32
DROP = {0x200D} | set(range(0xFE00, 0xFE10))  # ZWJ, variation selectors


@dataclass
class Style:
    weight: int
    color: tuple
    stroke_width: int = 0
    stroke_color: tuple = (255, 255, 255, 255)


@dataclass
class Prepared:
    text: str
    styles: list  # one Style per character
    font_paths: list  # one path per character (None = missing)
    missing: list = field(default_factory=list)
    combining: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def prepare(runs: list[dict], chain: list[str], base: Style, parse_color) -> Prepared:
    chars, styles = [], []
    dropped = 0
    for r in runs:
        t = unicodedata.normalize("NFC", r.get("text", ""))
        st = Style(int(r.get("weight", base.weight)), parse_color(r["color"]) if r.get("color") else base.color,
                   base.stroke_width, base.stroke_color)
        for ch in t:
            if ord(ch) in DROP:
                dropped += 1
                continue
            if ch == "\r":
                continue
            chars.append(ch)
            styles.append(st)
    text = "".join(chars)
    paths = []
    missing, combining = [], []
    for ch in text:
        if ch == "\n":
            paths.append(None)
            continue
        if unicodedata.combining(ch):
            combining.append(ch)
        p = fonts.pick(chain, ch)
        if p is None and ch not in linebreak.SPACES:
            missing.append(ch)
        paths.append(p or chain[0])
    notes = []
    if dropped:
        notes.append({"code": "removed_joiners", "count": dropped})
    return Prepared(text, styles, paths, sorted(set(missing)), sorted(set(combining)), notes)


def widths(p: Prepared, size: int) -> list[float]:
    out = []
    for ch, st, path in zip(p.text, p.styles, p.font_paths):
        if ch == "\n":
            out.append(0.0)
        else:
            out.append(fonts.load(path, size, st.weight).getlength(ch))
    return out


@dataclass
class Laid:
    size: int
    lines: list  # (start, end)
    widths: list
    overflow: bool
    line_h: float


def layout(p: Prepared, size: int, area_w: float) -> Laid:
    w = widths(p, size)
    lines, ov = linebreak.wrap(p.text, w, area_w)
    return Laid(size, lines, w, ov, size * LINE_HEIGHT)


def fits(l: Laid, area_w: float, area_h: float) -> bool:
    if l.overflow:
        return False
    if len(l.lines) * l.line_h > area_h + 0.5:
        return False
    return all(sum(l.widths[s:e]) <= area_w + 0.5 for s, e in l.lines)


def fit(p: Prepared, area_w: float, area_h: float, size, min_size: int, max_size: int) -> tuple[Laid, bool]:
    """(layout, too_small). A fixed `size` is used as it is."""
    if size:
        return layout(p, int(size), area_w), False
    lo, hi = max(1, int(min_size)), max(1, int(max_size))
    if hi < lo:
        hi = lo
    if not fits(layout(p, lo, area_w), area_w, area_h):
        return layout(p, lo, area_w), True
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(layout(p, mid, area_w), area_w, area_h):
            lo = mid
        else:
            hi = mid - 1
    return layout(p, lo, area_w), False


def draw(draw_rgba, p: Prepared, l: Laid, area: tuple, align: str, valign: str) -> list:
    """Draw onto an ImageDraw (RGBA); returns the line strings."""
    x0, y0, x1, y1 = area
    n = len(l.lines)
    total = n * l.line_h
    if valign == "middle":
        top = y0 + (y1 - y0 - total) / 2
    elif valign == "bottom":
        top = y1 - total
    else:
        top = y0
    first = next((pp for pp in p.font_paths if pp), None) or fonts.all_paths()[0]
    asc, desc = fonts.load(first, l.size, 400).getmetrics()
    out = []
    for i, (s, e) in enumerate(l.lines):
        lw = sum(l.widths[s:e])
        if align == "center":
            x = x0 + (x1 - x0 - lw) / 2
        elif align == "right":
            x = x1 - lw
        else:
            x = x0
        base = top + i * l.line_h + (l.line_h - (asc + desc)) / 2 + asc
        for k in range(s, e):
            ch = p.text[k]
            st = p.styles[k]
            if ch not in linebreak.SPACES and ch != "\n":
                f = fonts.load(p.font_paths[k], l.size, st.weight)
                kw = {}
                if st.stroke_width:
                    kw = {"stroke_width": st.stroke_width, "stroke_fill": st.stroke_color}
                draw_rgba.text((x, base), ch, font=f, fill=st.color, anchor="ls", **kw)
            x += l.widths[k]
        out.append(p.text[s:e])
    return out
