"""`lmn compose` (spec §7): images, rectangles, text, captions and speech bubbles on a canvas,
then checks that are reported, never fixed (`problems`, exit 4). The file is always written."""

from __future__ import annotations

import io
import os
import re

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .. import fonts, paths, util
from ..util import bad
from . import bubbles, templates, text

PRESETS = {"instagram-4x5": (1080, 1350), "instagram-1x1": (1080, 1080), "og": (1200, 630), "x-16x9": (1600, 900)}
CANVAS_MAX_SIDE, CANVAS_MAX_PX = 8192, 40_000_000
INPUT_MAX_SIDE, INPUT_MAX_PX = 8192, 64_000_000
LAYER_TYPES = ("image", "text", "bubble", "caption", "rect")
COLOR = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
CONTRAST_MIN = 3.0
COORD_MIN, COORD_MAX = -CANVAS_MAX_SIDE, 2 * CANVAS_MAX_SIDE  # layer and avoid coordinates


def parse_color(s, where="color") -> tuple:
    if not isinstance(s, str) or not COLOR.match(s):
        raise bad("%s: %r is not #RGB, #RRGGBB or #RRGGBBAA" % (where, s))
    h = s[1:]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        h += "ff"
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4, 6))


def parse_box(b, where) -> tuple:
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                         for v in b)):
        raise bad("%s: box must be [x0, y0, x1, y1]" % where)
    if b[2] <= b[0] or b[3] <= b[1]:
        raise bad("%s: box must have x1 > x0 and y1 > y0" % where)
    return tuple(float(v) for v in b)


def open_checked(path: str, what: str) -> Image.Image:
    """Open lazily and check the size BEFORE decoding (Pillow's own bomb check fires at 2x)."""
    try:
        im = Image.open(path)
    except Exception as e:  # noqa: BLE001
        raise bad("%s: not a readable image (%s)" % (what, e))
    w, h = im.size
    if max(w, h) > INPUT_MAX_SIDE or w * h > INPUT_MAX_PX:
        im.close()
        raise bad("%s: %dx%d is over the input limit (long side 8192, 64M px)" % (what, w, h))
    return im


def _count(mask: Image.Image) -> int:
    return mask.histogram()[255] if mask.mode == "L" else 0


def _and(a: Image.Image, b: Image.Image) -> Image.Image:
    return ImageChops.multiply(a, b).point(lambda v: 255 if v >= 128 else 0)


def _rect_mask(size, box) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=255)
    return m


def _lum(c) -> float:
    def ch(v):
        v = v / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])


def low_contrast(bg: Image.Image, masks: list[tuple[Image.Image, tuple]]) -> float | None:
    """The 10th percentile of per-pixel WCAG contrast between each glyph color and the background
    under its glyph mask widened by 2 px. None when there is no glyph."""
    lums = {}
    vals = []
    rgb = bg.convert("RGB")
    for mask, color in masks:
        region = mask.filter(ImageFilter.MaxFilter(5))
        bbox = region.getbbox()
        if not bbox:
            continue
        fl = _lum(color)
        r = region.crop(bbox).get_flattened_data()
        px = rgb.crop(bbox).get_flattened_data()
        for m, p in zip(r, px):
            if m < 128:
                continue
            if p not in lums:
                lums[p] = _lum(p)
            a, b = fl, lums[p]
            hi, lo = max(a, b), min(a, b)
            vals.append((hi + 0.05) / (lo + 0.05))
    if not vals:
        return None
    vals.sort()
    return vals[int(len(vals) * 0.10)]


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_layer(layer: dict, i: int, W: int, H: int) -> None:
    """Types, ranges and enums of one layer: a wrong field is exit 2, never an internal error."""
    at = "layers[%d]" % i
    box = parse_box(layer.get("box"), at)
    if not all(COORD_MIN <= v <= COORD_MAX for v in box):
        raise bad("%s: box coordinates must be from %d to %d" % (at, COORD_MIN, COORD_MAX))
    bw, bh = box[2] - box[0], box[3] - box[1]
    if max(bw, bh) > CANVAS_MAX_SIDE or bw * bh > CANVAS_MAX_PX:
        raise bad("%s: box %dx%d is over the limit (long side 8192, 40M px)" % (at, bw, bh))
    t = layer["type"]

    def color(key, default_ok=True):
        if key in layer and layer[key] is not None:
            parse_color(layer[key], "%s.%s" % (at, key))

    def weight(v, where):
        if not (_int(v) and 1 <= v <= 1000):
            raise bad("%s: weight must be an integer from 1 to 1000" % where)

    if t in ("text", "caption", "bubble"):
        if "weight" in layer:
            weight(layer["weight"], at)
        color("color")
        runs = layer.get("runs")
        if not isinstance(runs, list) or not all(isinstance(r, dict) and isinstance(r.get("text"), str) for r in runs):
            raise bad("%s.runs must be a list of {\"text\": …}" % at)
        for k, r in enumerate(runs):
            if "weight" in r:
                weight(r["weight"], "%s.runs[%d]" % (at, k))
            if r.get("color") is not None:
                parse_color(r["color"], "%s.runs[%d].color" % (at, k))
        size = layer.get("size")
        if size is not None and not (_num(size) and 1 <= size <= CANVAS_MAX_SIDE):
            raise bad("%s.size must be null or a number >= 1" % at)
        for k in ("min_size", "max_size"):
            if k in layer and not (_num(layer[k]) and 1 <= layer[k] <= CANVAS_MAX_SIDE):
                raise bad("%s.%s must be a number >= 1" % (at, k))
        if layer.get("align", "left") not in ("left", "center", "right"):
            raise bad("%s.align must be left, center or right" % at)
        if layer.get("valign", "top") not in ("top", "middle", "bottom"):
            raise bad("%s.valign must be top, middle or bottom" % at)
        st = layer.get("stroke")
        if st is not None:
            if not isinstance(st, dict) or not (_int(st.get("width", 0)) and 0 <= st.get("width", 0) <= 100):
                raise bad("%s.stroke must be {\"width\": 0-100, \"color\": …}" % at)
            if st.get("color") is not None:
                parse_color(st["color"], "%s.stroke.color" % at)
        color("background")
    if t == "bubble":
        tail = layer.get("tail")
        if tail is not None and not (isinstance(tail, list) and len(tail) == 2 and
                                     all(_num(v) and COORD_MIN <= v <= COORD_MAX for v in tail)):
            raise bad("%s.tail must be [x, y] with coordinates from %d to %d" % (at, COORD_MIN, COORD_MAX))
    if t == "image":
        if layer.get("fit", "cover") not in ("cover", "contain", "stretch"):
            raise bad("%s.fit must be cover, contain or stretch" % at)
        b = layer.get("border")
        if b is not None:
            if not isinstance(b, dict) or not (_int(b.get("width", 0)) and 0 <= b.get("width", 0) <= 1000):
                raise bad("%s.border must be {\"width\": int, \"color\": …}" % at)
            if b.get("color") is not None:
                parse_color(b["color"], "%s.border.color" % at)
    if t == "rect":
        color("color")


class Composer:
    def __init__(self, root: paths.Root, layout: dict, where: str):
        self.root = root
        self.layout = layout
        self.where = where
        self.inputs: dict[str, str] = {}  # real path -> sha256
        self.fonts_used: set[str] = set()
        self.problems: list[dict] = []
        self.layers_out: list[dict] = []

    def problem(self, code, index, **detail):
        self.problems.append({"code": code, "layer": index, "detail": detail})

    # ------------------------------------------------------------------ canvas

    def canvas(self) -> Image.Image:
        lay = self.layout
        if not isinstance(lay, dict) or lay.get("version") != 1:
            raise bad("%s: version must be 1" % self.where)
        c = lay.get("canvas")
        if not isinstance(c, dict):
            raise bad("%s: canvas is required" % self.where)
        if "preset" in c:
            if c["preset"] not in PRESETS:
                raise bad("canvas.preset must be one of %s" % sorted(PRESETS))
            w, h = PRESETS[c["preset"]]
        else:
            w, h = c.get("width"), c.get("height")
            if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (w, h)):
                raise bad("canvas: give a preset or integer width and height")
        if max(w, h) > CANVAS_MAX_SIDE or w * h > CANVAS_MAX_PX:
            raise bad("canvas %dx%d is over the limit (long side 8192, 40M px)" % (w, h))
        bg = parse_color(c.get("background", "#ffffff"), "canvas.background")
        return Image.new("RGBA", (w, h), bg)

    # ------------------------------------------------------------------ layers

    def run(self) -> Image.Image:
        cv = self.canvas()
        W, H = cv.size
        layers = self.layout.get("layers", [])
        if not isinstance(layers, list):
            raise bad("layers must be a list")
        avoid = []
        for i, a in enumerate(self.layout.get("avoid", []) or []):
            if not isinstance(a, dict):
                raise bad("avoid[%d]: expected an object" % i)
            ab = parse_box(a.get("box"), "avoid[%d]" % i)
            if not all(COORD_MIN <= v <= COORD_MAX for v in ab):
                raise bad("avoid[%d]: box coordinates must be from %d to %d" % (i, COORD_MIN, COORD_MAX))
            avoid.append((str(a.get("name", "avoid%d" % i)), ab))
        bodies = {}  # layer index -> body mask (full canvas)
        tails = {}  # layer index -> tail polygon mask (full)
        occupied = {}  # layer index -> mask counted for overlap
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict) or layer.get("type") not in LAYER_TYPES:
                raise bad("layers[%d]: type must be one of %s" % (i, LAYER_TYPES))
            t = layer["type"]
            check_layer(layer, i, W, H)
            box = parse_box(layer.get("box"), "layers[%d]" % i)
            if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                self.problem("off_canvas", i, box=list(box))
            info = {"index": i, "type": t, "drawn_box": None, "size": None, "lines": None, "notes": []}
            if t == "image":
                info["drawn_box"] = self.image(cv, layer, box, i)
            elif t == "rect":
                col = parse_color(layer.get("color", "#000000"), "layers[%d].color" % i)
                over = Image.new("RGBA", cv.size, (0, 0, 0, 0))
                ImageDraw.Draw(over).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=col)
                cv.alpha_composite(over)
                info["drawn_box"] = [round(v) for v in box]
            elif t in ("text", "caption"):
                area = box
                if t == "caption":
                    bgc = parse_color(layer.get("background", "#ffffff"), "layers[%d].background" % i)
                    over = Image.new("RGBA", cv.size, (0, 0, 0, 0))
                    ImageDraw.Draw(over).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=bgc)
                    cv.alpha_composite(over)
                    pw, ph = (box[2] - box[0]) * 0.04, (box[3] - box[1]) * 0.12
                    area = (box[0] + pw, box[1] + ph, box[2] - pw, box[3] - ph)
                    occupied[i] = _rect_mask(cv.size, box)
                glyphs = self.text(cv, layer, area, i, info, default_weight=400, default_align="left",
                                   default_valign="middle" if t == "caption" else "top")
                if t == "text":
                    occupied[i] = glyphs
            elif t == "bubble":
                kind = layer.get("kind", "speech")
                if kind not in bubbles.KINDS:
                    raise bad("layers[%d].kind must be one of %s" % (i, bubbles.KINDS))
                tail = layer.get("tail")
                if tail is not None and not (isinstance(tail, list) and len(tail) == 2):
                    raise bad("layers[%d].tail must be [x, y]" % i)
                d = bubbles.draw(kind, box, tail, cv.size)
                cv.alpha_composite(d.layer, (d.region[0], d.region[1]))
                full = lambda m: _paste_full(cv.size, m, d.region)  # noqa: E731
                bodies[i] = full(d.body)
                tails[i] = full(d.tail_full)
                occupied[i] = ImageChops.lighter(bodies[i], full(d.tail))
                info["notes"] += d.notes
                self.text(cv, layer, bubbles.text_area(kind, box), i, info,
                          default_weight=bubbles.WEIGHT[kind], default_align="center", default_valign="middle",
                          default_color="#141414")
                info["drawn_box"] = list(d.region)
            self.layers_out.append(info)
        self.checks(cv.size, avoid, bodies, tails, occupied)
        return cv

    def image(self, cv, layer, box, i) -> list:
        src = layer.get("src")
        if not isinstance(src, str):
            raise bad("layers[%d].src must be a path" % i)
        real = self.root.readable(src, "layers[%d].src" % i)
        self.inputs[real] = util.sha256_file(real)
        im = open_checked(real, "layers[%d].src %s" % (i, src))
        im = ImageOps.exif_transpose(im).convert("RGBA")
        crop = layer.get("crop")
        if crop is not None:
            c = parse_box(crop, "layers[%d].crop" % i)
            if c[0] < 0 or c[1] < 0 or c[2] > im.width or c[3] > im.height:
                raise bad("layers[%d].crop %s is outside the %dx%d image" % (i, list(crop), im.width, im.height))
            im = im.crop(tuple(round(v) for v in c))
        x0, y0, x1, y1 = (round(v) for v in box)
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        fit = layer.get("fit", "cover")
        if fit == "stretch":
            im = im.resize((bw, bh), Image.LANCZOS)
            pos = (x0, y0)
        elif fit == "contain":
            s = min(bw / im.width, bh / im.height)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            pos = (x0 + (bw - im.width) // 2, y0 + (bh - im.height) // 2)
        elif fit == "cover":
            im = ImageOps.fit(im, (bw, bh), Image.LANCZOS)
            pos = (x0, y0)
        else:
            raise bad("layers[%d].fit must be cover, contain or stretch" % i)
        over = Image.new("RGBA", cv.size, (0, 0, 0, 0))
        over.paste(im, pos)
        cv.alpha_composite(over)
        border = layer.get("border")
        if border:
            bwid = int(border.get("width", 0))
            if bwid > 0:
                col = parse_color(border.get("color", "#000000"), "layers[%d].border.color" % i)
                o2 = Image.new("RGBA", cv.size, (0, 0, 0, 0))
                ImageDraw.Draw(o2).rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=bwid)
                cv.alpha_composite(o2)
        return [pos[0], pos[1], pos[0] + im.width, pos[1] + im.height]

    def text(self, cv, layer, area, i, info, default_weight, default_align, default_valign,
             default_color="#000000") -> Image.Image:
        lang = layer.get("lang", "en")
        if lang not in fonts.LANGS:
            raise bad("layers[%d].lang must be one of %s" % (i, fonts.LANGS))
        font = layer.get("font", "sans")
        file = None
        role = "sans"
        if isinstance(font, dict):
            f = font.get("file")
            if not isinstance(f, str):
                raise bad("layers[%d].font.file must be a path" % i)
            file = self.root.readable(f, "layers[%d].font.file" % i)
            self.inputs[file] = util.sha256_file(file)
        elif font in ("sans", "serif", "mono"):
            role = "mono" if font == "mono" else "sans"  # v0.1 ships sans only: serif falls back to sans
        else:
            raise bad("layers[%d].font must be sans, serif, mono or {\"file\": …}" % i)
        chain = fonts.chain_paths(lang, role, file)
        runs = layer.get("runs")
        if not isinstance(runs, list) or not all(isinstance(r, dict) and isinstance(r.get("text"), str) for r in runs):
            raise bad("layers[%d].runs must be a list of {\"text\": …}" % i)
        color = parse_color(layer.get("color", default_color), "layers[%d].color" % i)
        stroke = layer.get("stroke") or {}
        sw = int(stroke.get("width", 0) or 0)
        sc = parse_color(stroke.get("color", "#ffffff"), "layers[%d].stroke.color" % i)
        base = text.Style(int(layer.get("weight", default_weight)), color, sw, sc)
        p = text.prepare(runs, chain, base, lambda s: parse_color(s, "layers[%d].runs[].color" % i))
        info["notes"] += p.notes
        if p.missing:
            self.problem("missing_glyph", i, chars="".join(p.missing))
        if p.combining:
            self.problem("unsupported_combining", i, chars=" ".join("U+%04X" % ord(c) for c in p.combining))
        ax0, ay0, ax1, ay1 = area
        aw, ah = ax1 - ax0, ay1 - ay0
        min_size = int(layer.get("min_size", 12))
        max_size = int(layer.get("max_size", ah))
        laid, too_small = text.fit(p, aw, ah, layer.get("size"), min_size, max_size)
        if too_small:
            self.problem("too_small", i, min_size=min_size)
        over = Image.new("RGBA", cv.size, (0, 0, 0, 0))
        lines = text.draw(ImageDraw.Draw(over), p, laid, area, layer.get("align", default_align),
                          layer.get("valign", default_valign))
        for path in set(p.font_paths):
            if path:
                self.fonts_used.add(path)
        alpha = over.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
        bb = alpha.getbbox()
        if too_small or laid.overflow or (bb and (bb[0] < ax0 - 1 or bb[1] < ay0 - 1 or bb[2] > ax1 + 1
                                                    or bb[3] > ay1 + 1)):
            self.problem("overflow", i, ink=list(bb) if bb else None, area=[round(v, 1) for v in area])
        # contrast: each color's glyphs (or the stroke) against what is under them
        masks = []
        colors = {s.color for s in p.styles}
        if sw:
            masks.append((alpha, sc))
        elif len(colors) == 1:
            masks.append((alpha, colors.pop()))
        else:
            for col in {s.color for s in p.styles}:
                o = Image.new("RGBA", cv.size, (0, 0, 0, 0))
                sub = text.Prepared(p.text, [s if s.color == col else text.Style(s.weight, (0, 0, 0, 0))
                                             for s in p.styles], p.font_paths)
                text.draw(ImageDraw.Draw(o), sub, laid, area, layer.get("align", default_align),
                          layer.get("valign", default_valign))
                masks.append((o.getchannel("A").point(lambda v: 255 if v >= 128 else 0), col))
        c = low_contrast(cv, masks)
        if c is not None and c < CONTRAST_MIN:
            self.problem("low_contrast", i, p10=round(c, 2))
        cv.alpha_composite(over)
        info.update({"size": laid.size, "lines": lines})
        if info["drawn_box"] is None:
            info["drawn_box"] = list(bb) if bb else None
        return alpha

    # ------------------------------------------------------------------ checks

    def checks(self, size, avoid, bodies, tails, occupied) -> None:
        for i, m in occupied.items():
            for name, box in avoid:
                n = _count(_and(m, _rect_mask(size, box)))
                if n:
                    self.problem("overlap", i, avoid=name, pixels=n)
        keys = sorted(occupied)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                i, j = keys[x], keys[y]
                n = _count(_and(occupied[i], occupied[j]))
                if n:
                    self.problem("overlap", i, layer=j, pixels=n)
        for i, t in tails.items():
            for j, b in bodies.items():
                if i != j:
                    n = _count(_and(t, b))
                    if n:
                        self.problem("tail_crosses", i, bubble=j, pixels=n)


def _paste_full(size, m, region) -> Image.Image:
    full = Image.new("L", size, 0)
    full.paste(m, (region[0], region[1]))
    return full


# --------------------------------------------------------------------------- command


def check_out(root: paths.Root, out: str, kind: str, inputs: list[str]) -> str:
    real = root.writable(out, "--out")
    for p in inputs:
        if util.fold(p) == util.fold(real):
            raise bad("--out %s is also an input" % out)
    rec = util.read_json_or_none(root.record_path(real))
    if isinstance(rec, dict) and rec.get("kind") != kind:
        raise bad("--out %s was written by %s (its record): choose another path" % (out, rec.get("kind")))
    return real


def write_png(root: paths.Root, out: str, im: Image.Image, record: dict) -> dict:
    buf = io.BytesIO()
    if im.mode == "RGBA" and im.getextrema()[3][0] == 255:
        im = im.convert("RGB")
    im.save(buf, format="PNG")
    data = buf.getvalue()
    tmp = os.path.join(root.tmp_dir(), paths.tmp_name(".png"))
    paths.write_new_file(tmp, data)
    rec = dict(record)
    rec["out_sha256"] = util.sha256_bytes(data)
    res = paths.commit(root, out, tmp, rec)
    res["out_sha256"] = rec["out_sha256"]
    return res


def versions() -> dict:
    from PIL import features

    import PIL

    return {"pillow": PIL.__version__, "freetype": features.version("freetype2"), "layout_engine": "basic"}


def cmd_compose(args) -> tuple[dict, int]:
    root = paths.Root(args.root, args.state_dir)
    fonts.require()
    inputs = []
    if args.layout:
        lp = root.readable(args.layout, "--layout")
        inputs.append(lp)
        layout = templates._load(lp)
        source = {"layout": root.rel(lp), "layout_sha256": util.sha256_file(lp)}
    else:
        if not args.data:
            raise bad("--template needs --data")
        dp = root.readable(args.data, "--data")
        inputs.append(dp)
        data = templates._load(dp)
        tpl, schema, where = templates.find(args.template, root, args.templates)
        templates.validate(data, schema)
        layout = templates.apply(tpl, data)
        source = {"template": where, "data": root.rel(dp), "data_sha256": util.sha256_file(dp)}
    for layer in layout.get("layers", []) if isinstance(layout, dict) else []:
        if isinstance(layer, dict) and isinstance(layer.get("src"), str):
            inputs.append(root.inside(layer["src"], "layer src"))
    out = check_out(root, args.out, "compose", inputs)
    if getattr(args, "result", None) and any(util.fold(p) == util.fold(args.result) for p in inputs):
        raise bad("--result %s is also an input" % args.result)
    comp = Composer(root, layout, args.layout or args.template)
    im = comp.run()
    import limner

    record = {"limner": limner.__version__, "contract": util.CONTRACT, "kind": "compose", **source,
              "resolved_layout_sha256": util.sha256_bytes(util.canonical(layout).encode("utf-8")),
              "inputs": [{"path": root.rel(p), "sha256": s} for p, s in sorted(comp.inputs.items())],
              "fonts": [{"file": os.path.basename(p), "sha256": util.sha256_file(p)} for p in sorted(comp.fonts_used)],
              **versions(), "created_at": util.utc_now()}
    res = write_png(root, out, im, record)
    obj = {"contract": util.CONTRACT, "out": root.rel(out), "out_sha256": res["out_sha256"], "record": res["record"],
           "problems": comp.problems, "layers": comp.layers_out}
    return obj, (util.PROBLEMS if comp.problems else util.OK)
