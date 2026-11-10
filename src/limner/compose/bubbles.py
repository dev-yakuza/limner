# Ported from agent-comic-kit `comic/imaging/letter.py`
# (https://github.com/Saeyeon-developer/agent-comic-kit, commit 5e0e8d6b1d5a1a6e05268cac44c4722696cea3c7),
# licensed under the Apache License 2.0 — see LICENSES/agent-comic-kit-APACHE-2.0.txt.
# Modified by dev-yakuza for limner: the body is sized from the caller's `box` (not from the text),
# the text area is a fixed fraction of the box per kind, the speech tail base is 0.28 of the short
# axis, the body and the tail are also drawn as separate masks for overlap checks, and the
# automatic placement and debug overlay were removed.
"""Speech bubbles: speech / shout / thought / narration (spec §7.2), drawn 3x supersampled."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

SS = 3  # supersampling factor
INK, PAPER = (20, 20, 20, 255), (255, 255, 255, 255)
WEIGHT = {"speech": 600, "shout": 900, "thought": 500, "narration": 500}
KINDS = tuple(WEIGHT)
TAIL_MAX = 90  # px at a 1080 px wide canvas, from the outer edge of the body
TAIL_MIN = {"speech": 24, "shout": 24, "thought": 40}
SPIKE = 1.22  # shout spikes reach 1.22 + 0.06 = 1.28 of the base ellipse
SPIKE_MAX = 1.28
TEXT_AREA = {"speech": (0.70, 0.64), "shout": (0.56, 0.50), "thought": (0.66, 0.58)}
NARRATION_PAD = 0.08
TAIL_BASE = 0.28  # of the short axis
TIP_EXCLUDED = 0.25  # the tail's last quarter (towards the mouth) is not an overlap


def _ellipse_point(cx, cy, a, b, ang):
    return cx + a * math.cos(ang), cy + b * math.sin(ang)


def _angle(cx, cy, a, b, target):
    return math.atan2((target[1] - cy) / b, (target[0] - cx) / a)


def _tail_room(cx, cy, ex, ey, target) -> float:
    """Distance from the body's outer edge (ellipse ex x ey) to target along the ray from the
    centre; negative when target lies inside."""
    dx, dy = target[0] - cx, target[1] - cy
    r = math.hypot(dx / ex, dy / ey)
    if r == 0:
        return -1.0
    d = math.hypot(dx, dy) * abs(1 - 1 / r)
    return d if r > 1 else -d


def _cap_tip(cx, cy, a, b, target, max_len):
    """Move the tail tip so it is at most max_len from the bubble edge; (tip, edge point, capped)."""
    ex, ey = _ellipse_point(cx, cy, a, b, _angle(cx, cy, a, b, target))
    dx, dy = target[0] - ex, target[1] - ey
    dist = math.hypot(dx, dy)
    if dist <= max_len or dist == 0:
        return tuple(target), (ex, ey), False
    return (ex + dx / dist * max_len, ey + dy / dist * max_len), (ex, ey), True


def body_axes(kind: str, box) -> tuple[float, float, float, float]:
    """(cx, cy, a, b): the base ellipse of a body whose bounding box (spikes, lobes) is `box`."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    if kind == "shout":
        return cx, cy, hx / SPIKE_MAX, hy / SPIKE_MAX
    if kind == "thought":
        # extents: a*0.92 + r, b*0.88 + r with r = 0.42 * min(a, b)
        b = hy / 1.30
        a = (hx - 0.42 * b) / 0.92
        if a < b:
            a = hx / 1.34
            b = (hy - 0.42 * a) / 0.88
        return cx, cy, a, b
    return cx, cy, hx, hy


def extents(kind: str, a: float, b: float) -> tuple[float, float]:
    if kind == "shout":
        return a * SPIKE_MAX, b * SPIKE_MAX
    if kind == "thought":
        r = min(a, b) * 0.42
        return a * 0.92 + r, b * 0.88 + r
    return a, b


def text_area(kind: str, box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if kind == "narration":
        return x0 + w * NARRATION_PAD, y0 + h * NARRATION_PAD, x1 - w * NARRATION_PAD, y1 - h * NARRATION_PAD
    fx, fy = TEXT_AREA[kind]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - w * fx / 2, cy - h * fy / 2, cx + w * fx / 2, cy + h * fy / 2


@dataclass
class Drawn:
    layer: Image.Image  # RGBA, 1x, the size of `region`
    body: Image.Image  # L mask, 1x
    tail: Image.Image  # L mask, 1x (without the excluded tip zone)
    tail_full: Image.Image  # L mask, 1x
    region: tuple  # (x0, y0, x1, y1) on the canvas
    notes: list = field(default_factory=list)


def draw(kind: str, box, tail, canvas_size: tuple[int, int]) -> Drawn:
    """Body and tail of one bubble, drawn in a local region of the canvas."""
    W, H = canvas_size
    scale = W / 1080
    notes = []
    cx, cy, a, b = body_axes(kind, box)
    ex, ey = extents(kind, a, b)
    tip = edge = None
    if tail is not None and kind != "narration":
        tip, edge, capped = _cap_tip(cx, cy, ex, ey, tuple(tail), TAIL_MAX * scale)
        if capped:
            notes.append({"code": "tail_capped", "max_px": round(TAIL_MAX * scale, 1)})
        if _tail_room(cx, cy, ex, ey, tip) < TAIL_MIN[kind] * scale:
            notes.append({"code": "tail_skipped"})
            tip = edge = None
    xs = [box[0], box[2]] + ([tip[0]] if tip else [])
    ys = [box[1], box[3]] + ([tip[1]] if tip else [])
    pad = 6 + 4 * scale
    rx0, ry0 = max(0, int(math.floor(min(xs) - pad))), max(0, int(math.floor(min(ys) - pad)))
    rx1, ry1 = min(W, int(math.ceil(max(xs) + pad))), min(H, int(math.ceil(max(ys) + pad)))
    rx1, ry1 = max(rx1, rx0 + 1), max(ry1, ry0 + 1)
    size = ((rx1 - rx0) * SS, (ry1 - ry0) * SS)

    def S(x, y):  # canvas px -> local supersampled px
        return (x - rx0) * SS, (y - ry0) * SS

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    body = Image.new("L", size, 0)
    tailm = Image.new("L", size, 0)
    d, db, dt = ImageDraw.Draw(layer), ImageDraw.Draw(body), ImageDraw.Draw(tailm)
    u = scale * SS
    stroke = max(1, round(3 * u))
    lcx, lcy = S(cx, cy)
    la, lb = a * SS, b * SS
    ltip = S(*tip) if tip else None
    if kind == "narration":
        rect = [*S(box[0], box[1]), *S(box[2], box[3])]
        d.rectangle(rect, fill=PAPER, outline=INK, width=stroke)
        db.rectangle(rect, fill=255)
    elif kind == "speech":
        d.ellipse([lcx - la - stroke, lcy - lb - stroke, lcx + la + stroke, lcy + lb + stroke], fill=INK)
        db.ellipse([lcx - la, lcy - lb, lcx + la, lcy + lb], fill=255)
        if ltip:
            ang = _angle(lcx, lcy, la, lb, ltip)
            px, py = _ellipse_point(lcx, lcy, la * 0.92, lb * 0.92, ang)
            tx, ty = -la * math.sin(ang), lb * math.cos(ang)
            tn = math.hypot(tx, ty) or 1.0
            half = TAIL_BASE * 2 * min(la, lb) / 2
            p1 = (px - tx / tn * half, py - ty / tn * half)
            p2 = (px + tx / tn * half, py + ty / tn * half)
            d.polygon([p1, ltip, p2], fill=PAPER, outline=INK, width=stroke)
            dt.polygon([p1, ltip, p2], fill=255)
        d.ellipse([lcx - la, lcy - lb, lcx + la, lcy + lb], fill=PAPER)
    elif kind == "shout":
        n, pts = 22, []
        tip_ang = _angle(lcx, lcy, la, lb, ltip) if ltip else None
        tip_i = None
        for i in range(n * 2):
            ang = 2 * math.pi * i / (n * 2)
            r = 1.0 if i % 2 else SPIKE + 0.06 * math.sin(i * 2.3)
            p = _ellipse_point(lcx, lcy, la * r, lb * r, ang)
            if ltip and i % 2 == 0 and abs((ang - tip_ang + math.pi) % (2 * math.pi) - math.pi) < math.pi / n:
                p = tuple(ltip)
                tip_i = i
            pts.append(p)
        if ltip and tip_i is not None:
            body_pts = list(pts)
            body_pts[tip_i] = _ellipse_point(lcx, lcy, la, lb, tip_ang)
            db.polygon(body_pts, fill=255)
            dt.polygon([pts[tip_i - 1], ltip, pts[(tip_i + 1) % len(pts)]], fill=255)
        else:
            db.polygon(pts, fill=255)
        d.polygon(pts, fill=PAPER, outline=INK, width=stroke)
    elif kind == "thought":
        lobes = 12
        r = min(la, lb) * 0.42
        centres = [_ellipse_point(lcx, lcy, la * 0.92, lb * 0.88, 2 * math.pi * i / lobes) for i in range(lobes)]
        for x, y in centres:
            d.ellipse([x - r - stroke, y - r - stroke, x + r + stroke, y + r + stroke], fill=INK)
            db.ellipse([x - r, y - r, x + r, y + r], fill=255)
        for x, y in centres:
            d.ellipse([x - r, y - r, x + r, y + r], fill=PAPER)
        d.ellipse([lcx - la * 0.95, lcy - lb * 0.95, lcx + la * 0.95, lcy + lb * 0.95], fill=PAPER)
        db.ellipse([lcx - la * 0.95, lcy - lb * 0.95, lcx + la * 0.95, lcy + lb * 0.95], fill=255)
        if ltip:
            oa, ob = extents("thought", la, lb)
            sx, sy = _ellipse_point(lcx, lcy, oa, ob, _angle(lcx, lcy, oa, ob, ltip))
            room = math.hypot(ltip[0] - sx, ltip[1] - sy)
            k_size = max(0.6, min(1.0, room / (70 * u)))
            for t, rr in ((0.30, 14), (0.62, 9), (0.90, 6)):
                x, y = sx + (ltip[0] - sx) * t, sy + (ltip[1] - sy) * t
                rr *= u * k_size
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=PAPER, outline=INK, width=stroke)
                dt.ellipse([x - rr, y - rr, x + rr, y + rr], fill=255)
    tail_full = tailm.copy()
    if ltip and edge:
        ledge = S(*edge)
        rr = TIP_EXCLUDED * math.hypot(ltip[0] - ledge[0], ltip[1] - ledge[1])
        ImageDraw.Draw(tailm).ellipse([ltip[0] - rr, ltip[1] - rr, ltip[0] + rr, ltip[1] + rr], fill=0)
    one = (rx1 - rx0, ry1 - ry0)

    def down(m):
        return m.resize(one, Image.LANCZOS).point(lambda v: 255 if v >= 128 else 0)

    return Drawn(layer.resize(one, Image.LANCZOS), down(body), down(tailm), down(tail_full),
                 (rx0, ry0, rx1, ry1), notes)
