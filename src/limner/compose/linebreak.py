"""Line breaking (spec §7.3). Measured by a width per character (the caller's fonts), so the
rules are tested with a table (`tests/data/linebreak.json`) independent of fonts.

- explicit `\\n` is kept; inside a paragraph lines wrap automatically and are balanced
  (the greedy line count at the full width, then the narrowest width that keeps that count —
  the balancing of agent-comic-kit `letter.py`);
- unbreakable units: Latin letters/digits with `.'-_/` between them, a Korean eojeol, anything
  that is not CJK; Japanese (kana, kanji, CJK punctuation) may break between any two characters,
  except before a no-start character or after a no-end character (kinsoku);
- a unit wider than the line is not split: it gets its own line and the result reports overflow;
- spaces (U+0020, U+3000) at a line end are dropped.
"""

from __future__ import annotations

NO_START = set("、。，．・：；？！ー～〜…‥」』）］｝〉》】〕’”ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々ゝゞヽヾ〻゛゜"
               "!?.,:;)]}")
NO_END = set("「『（［｛〈《【〔‘“([{")
SPACES = {" ", "　"}


def cjk_breakable(ch: str) -> bool:
    """Characters a Japanese line may break around (Hangul is not: Korean breaks at spaces)."""
    cp = ord(ch)
    return (0x3000 <= cp <= 0x303F  # CJK symbols and punctuation
            or 0x3040 <= cp <= 0x30FF  # hiragana, katakana
            or 0x31F0 <= cp <= 0x31FF  # katakana phonetic extensions
            or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF  # kanji
            or 0x20000 <= cp <= 0x3FFFF
            or 0xFF00 <= cp <= 0xFFEF)  # full- and half-width forms


def can_break(prev: str, cur: str) -> bool:
    """A break opportunity between two adjacent characters of one paragraph."""
    if cur in SPACES:
        return False  # spaces stay with the unit before them
    if cur in NO_START or prev in NO_END:
        return False
    if prev in SPACES:
        return True
    return cjk_breakable(prev) or cjk_breakable(cur)


def units(text: str) -> list[tuple[int, int]]:
    """Unbreakable units (start, end), each with its trailing spaces."""
    out = []
    start = 0
    for i in range(1, len(text)):
        if can_break(text[i - 1], text[i]):
            out.append((start, i))
            start = i
    if text:
        out.append((start, len(text)))
    return out


def _trim(text: str, s: int, e: int) -> int:
    while e > s and text[e - 1] in SPACES:
        e -= 1
    return e


def _greedy(text, us, w, maxw) -> list[list[int]]:
    lines: list[list[int]] = []
    cur: list[int] = []
    cur_w = 0.0  # width including the trailing spaces of the last unit
    for k, (s, e) in enumerate(us):
        full = sum(w[s:e])
        bare = sum(w[s:_trim(text, s, e)])
        if cur and cur_w + bare > maxw + 1e-9:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur.append(k)
        cur_w += full
    if cur:
        lines.append(cur)
    return lines


def wrap_paragraph(text: str, w: list[float], maxw: float) -> tuple[list[tuple[int, int]], bool]:
    us = units(text)
    if not us:
        return [(0, 0)], False
    bare = [sum(w[s:_trim(text, s, e)]) for s, e in us]
    overflow = max(bare) > maxw + 1e-9
    best = _greedy(text, us, w, maxw)
    n = len(best)
    if n > 1:
        lo, hi = max(bare), float(maxw)
        if lo < hi:
            for _ in range(20):
                mid = (lo + hi) / 2
                if len(_greedy(text, us, w, mid)) <= n:
                    hi = mid
                else:
                    lo = mid
            cand = _greedy(text, us, w, hi)
            if len(cand) <= n:
                best = cand
    lines = []
    for ks in best:
        s = us[ks[0]][0]
        e = _trim(text, s, us[ks[-1]][1])
        lines.append((s, e))
    return lines, overflow


def wrap(text: str, widths: list[float], maxw: float) -> tuple[list[tuple[int, int]], bool]:
    """Lines as (start, end) offsets into `text` (end excludes dropped spaces), and whether a
    unit was wider than `maxw`. `widths[i]` is the advance of `text[i]` (0 for `\\n`)."""
    out: list[tuple[int, int]] = []
    overflow = False
    pos = 0
    for para in text.split("\n"):
        lines, ov = wrap_paragraph(para, widths[pos:pos + len(para)], maxw)
        overflow = overflow or ov
        out += [(s + pos, e + pos) for s, e in lines]
        pos += len(para) + 1
    return out, overflow


def em_width(ch: str) -> float:
    """The test table's measure: East Asian wide/fullwidth = 1, others = 0.5."""
    import unicodedata

    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5
