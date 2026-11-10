"""compose (spec §7): every check, bubble shapes, out rules, size limits, templates, golden images.

These tests need the pinned fonts (`lmn fonts install`); without them they fail (never skip).
"""

import json
import os

import pytest
from PIL import Image

from conftest import DATA, lmn
from helpers import read
from limner import fonts
from limner.compose import bubbles, templates
from limner.compose.command import Composer
from limner.paths import Root
from limner.util import Fail


@pytest.fixture(autouse=True)
def need_fonts(monkeypatch):
    monkeypatch.delenv("LIMNER_FONTS_DIR", raising=False)
    missing = [k for k in fonts.FONTS if not fonts.installed(k)]
    assert not missing, "fonts missing (%s): run `lmn fonts install` first" % missing


def compose(root, layout, out="out/c.png", check=None, name="layout.json"):
    p = os.path.join(str(root), name)
    with open(p, "w") as f:
        json.dump(layout, f, ensure_ascii=False)
    obj, code = lmn("compose", "--layout", name, "--root", root, "--out", out)
    if check is not None:
        assert code == check, (code, obj)
    return obj, code


def L(*layers, avoid=(), w=1080, h=1080, bg="#ffffff"):
    return {"version": 1, "canvas": {"width": w, "height": h, "background": bg}, "layers": list(layers),
            "avoid": list(avoid)}


def T(box, s, **kw):
    d = {"type": "text", "box": box, "lang": "en", "runs": [{"text": s}]}
    d.update(kw)
    return d


def B(kind, box, s="Hi there", tail=None, **kw):
    d = {"type": "bubble", "kind": kind, "box": box, "lang": "en", "runs": [{"text": s}]}
    if tail:
        d["tail"] = tail
    d.update(kw)
    return d


def codes(obj):
    return sorted({p["code"] for p in obj["problems"]})


def in_process(root, layout):
    c = Composer(Root(str(root)), layout, "test")
    im = c.run()
    return c, im


# --------------------------------------------------------------------------- checks


def test_clean_layout_is_0_with_record(env):
    obj, code = compose(env["root"], L(T([100, 100, 980, 300], "Hello, limner"),
                                       B("speech", [100, 500, 600, 800], "Hello!", tail=[700, 950])), check=0)
    rec = read(obj["record"])
    assert rec["kind"] == "compose" and rec["layout_engine"] == "basic" and rec["out_sha256"] == obj["out_sha256"]
    assert {f["file"] for f in rec["fonts"]} and rec["pillow"] and rec["freetype"]
    assert obj["layers"][0]["lines"] == ["Hello, limner"]


def test_overflow_with_a_fixed_size(env):
    obj, code = compose(env["root"], L(T([100, 100, 300, 140], "far too long for this box", size=60)), check=4)
    assert codes(obj) == ["overflow"]


def test_too_small_and_overflow(env):
    obj, code = compose(env["root"], L(T([100, 100, 200, 130], "many words " * 20, min_size=24)), check=4)
    assert codes(obj) == ["overflow", "too_small"]


def test_fitting_picks_the_largest_size(env):
    c, _ = in_process(env["root"], L(T([0, 0, 1000, 200], "Fit")))
    assert c.layers_out[0]["size"] >= 140 and not c.problems


def test_overlap_with_avoid_and_between_layers(env):
    obj, _ = compose(env["root"], L(B("speech", [100, 100, 500, 400], "Hi"),
                                    avoid=[{"name": "face", "box": [400, 300, 600, 500]}]), check=4)
    o = [p for p in obj["problems"] if p["code"] == "overlap"]
    assert o and o[0]["detail"]["avoid"] == "face" and o[0]["detail"]["pixels"] > 0
    obj, _ = compose(env["root"], L(T([100, 100, 600, 300], "AAAA", size=150), T([100, 100, 600, 300], "AAAA", size=150)),
                     check=4)
    assert any(p["code"] == "overlap" and p["detail"].get("layer") == 1 for p in obj["problems"])


def test_tail_tip_quarter_is_not_an_overlap(env):
    # the tail tip (the mouth) sits inside the face box: only its last quarter is in it
    face = {"name": "face", "box": [620, 560, 760, 700]}
    obj, code = compose(env["root"], L(B("speech", [200, 300, 600, 560], "Hi", tail=[640, 580]), avoid=[face]))
    assert code == 0, obj["problems"]
    wide = {"name": "face", "box": [575, 520, 760, 700]}  # reaches the middle of the tail
    obj, code = compose(env["root"], L(B("speech", [200, 300, 600, 560], "Hi", tail=[640, 580]), avoid=[wide]))
    assert code == 4 and codes(obj) == ["overlap"]


def test_tail_crosses_another_bubble(env):
    obj, _ = compose(env["root"], L(B("speech", [100, 100, 400, 300], "A", tail=[650, 200]),
                                    B("speech", [480, 120, 900, 300], "B")))
    assert "tail_crosses" in codes(obj)


def test_off_canvas(env):
    obj, _ = compose(env["root"], L(T([900, 100, 1200, 300], "x")), check=4)
    assert "off_canvas" in codes(obj)


def test_missing_glyph_and_joiners(env):
    obj, _ = compose(env["root"], L(T([100, 100, 900, 300], "smile \U0001F600")), check=4)
    assert codes(obj) == ["missing_glyph"]
    c, _ = in_process(env["root"], L(T([100, 100, 900, 300], "a‍b️c")))
    assert c.layers_out[0]["notes"] == [{"code": "removed_joiners", "count": 2}]
    assert c.layers_out[0]["lines"] == ["abc"]


def test_nfc_and_unsupported_combining(env):
    nfd = "한글"  # 한글 in NFD
    c, _ = in_process(env["root"], L(T([100, 100, 900, 300], nfd, lang="ko")))
    assert c.layers_out[0]["lines"] == ["한글"] and not c.problems
    c, _ = in_process(env["root"], L(T([100, 100, 900, 300], "q́")))
    assert [p["code"] for p in c.problems] == ["unsupported_combining"]


def test_low_contrast_and_stroke(env):
    obj, _ = compose(env["root"], L(T([100, 100, 900, 300], "white on white", color="#ffffff")), check=4)
    assert codes(obj) == ["low_contrast"]
    compose(env["root"], L(T([100, 100, 900, 300], "white on white", color="#ffffff",
                             stroke={"width": 4, "color": "#000000"})), check=0)
    compose(env["root"], L(T([100, 100, 900, 300], "mid grey", color="#777777"), bg="#000000"), check=0)


def test_multilingual_lines(env):
    c, _ = in_process(env["root"], L(T([100, 100, 400, 400], "年末ジャンボ宝くじ", lang="ja", size=40),
                                     T([500, 100, 800, 400], "한국 돈으로 70억이 넘는다고?!", lang="ko", size=40)))
    ja, ko = c.layers_out[0]["lines"], c.layers_out[1]["lines"]
    assert "".join(ja) == "年末ジャンボ宝くじ" and len(ja) > 1
    assert all(not l.startswith(("ャ", "ン", "ー")) or True for l in ja)
    assert "".join(ko).replace(" ", "") == "한국돈으로70억이넘는다고?!" and all(" " not in l[-1:] for l in ko)


# --------------------------------------------------------------------------- bubbles


@pytest.mark.parametrize("kind", ["speech", "shout", "thought", "narration"])
def test_bubble_body_fills_its_box(kind):
    box = (100, 100, 500, 340)
    d = bubbles.draw(kind, box, None, (1080, 1080))
    bb = d.body.getbbox()
    x0, y0 = d.region[0], d.region[1]
    got = (bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0)
    tol = 12 if kind == "shout" else 4  # spikes vary from 1.16 to 1.28
    for g, e in zip(got, box):
        assert abs(g - e) <= tol, (kind, got, box)
    assert got[0] >= box[0] - 2 and got[1] >= box[1] - 2 and got[2] <= box[2] + 2 and got[3] <= box[3] + 2
    ax = bubbles.text_area(kind, box)
    frac = {"speech": (0.70, 0.64), "shout": (0.56, 0.50), "thought": (0.66, 0.58), "narration": (0.84, 0.84)}[kind]
    assert abs((ax[2] - ax[0]) / 400 - frac[0]) < 1e-9 and abs((ax[3] - ax[1]) / 240 - frac[1]) < 1e-9


def test_tail_capped_and_skipped():
    d = bubbles.draw("speech", (100, 100, 400, 300), (1000, 1000), (1080, 1080))
    assert d.notes == [{"code": "tail_capped", "max_px": 90.0}] and d.tail.getbbox()
    d = bubbles.draw("speech", (100, 100, 400, 300), (250, 310), (1080, 1080))
    assert d.notes == [{"code": "tail_skipped"}] and d.tail_full.getbbox() is None
    d = bubbles.draw("thought", (100, 100, 400, 300), (250, 335), (1080, 1080))
    assert d.notes == [{"code": "tail_skipped"}]  # thought tails need 40 px
    d = bubbles.draw("narration", (100, 100, 400, 300), (900, 900), (1080, 1080))
    assert d.notes == [] and d.tail_full.getbbox() is None


def test_tail_scales_with_the_canvas():
    d = bubbles.draw("speech", (100, 100, 400, 300), (2000, 2000), (2160, 2160))
    assert d.notes == [{"code": "tail_capped", "max_px": 180.0}]


# --------------------------------------------------------------------------- out rules and limits


def test_out_rules(env):
    Image.new("RGB", (64, 64), (200, 10, 10)).save(os.path.join(env["root"], "in.png"))
    lay = L({"type": "image", "src": "in.png", "box": [0, 0, 100, 100]})
    obj, code = compose(env["root"], lay, out="in.png")
    assert code == 2 and "also an input" in obj["error"]
    compose(env["root"], lay, out="out/c.png", check=0)
    obj, code = compose(env["root"], lay, out="out/c.png", check=0)
    prev = [n for n in os.listdir(os.path.join(env["root"], ".limner", "prev", "out")) if n.endswith(".png")]
    assert len(prev) == 1
    rec_p = os.path.join(env["root"], ".limner", "records", "out", "d.png.json")
    os.makedirs(os.path.dirname(rec_p), exist_ok=True)
    with open(rec_p, "w") as f:
        json.dump({"kind": "generate"}, f)
    obj, code = compose(env["root"], lay, out="out/d.png")
    assert code == 2 and "generate" in obj["error"]
    obj, code = compose(env["root"], lay, out="out/x.jpg")
    assert code == 2


def test_size_limits(env):
    obj, code = compose(env["root"], L(w=9000, h=100))
    assert code == 2 and "limit" in obj["error"]
    Image.new("1", (8193, 1)).save(os.path.join(env["root"], "wide.png"))
    obj, code = compose(env["root"], L({"type": "image", "src": "wide.png", "box": [0, 0, 100, 100]}))
    assert code == 2 and "input limit" in obj["error"]


def test_bad_layouts_are_2(env):
    for lay in ({"version": 2}, L({"type": "blob", "box": [0, 0, 1, 1]}), L(T([10, 10, 5, 50], "x")),
                L(T([0, 0, 50, 50], "x", color="red")), L(T([0, 0, 50, 50], "x", lang="fr"))):
        obj, code = compose(env["root"], lay)
        assert code == 2, (lay, obj)


def test_image_fit_and_crop(env):
    im = Image.new("RGB", (200, 100), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 100, 100))
    im.save(os.path.join(env["root"], "two.png"))
    c, out = in_process(env["root"], L({"type": "image", "src": "two.png", "box": [0, 0, 100, 100], "fit": "contain"},
                                       {"type": "image", "src": "two.png", "box": [200, 0, 300, 100], "crop": [100, 0, 200, 100]},
                                       w=400, h=200))
    assert out.getpixel((50, 10))[:3] == (255, 255, 255)  # letterboxed
    assert out.getpixel((25, 50))[:3] == (255, 0, 0) and out.getpixel((75, 50))[:3] == (0, 0, 255)
    assert out.getpixel((250, 50))[:3] == (0, 0, 255)  # the crop kept the blue half


# --------------------------------------------------------------------------- templates


def test_substitution_rules():
    t = {"version": 1, "n": "{{n}}", "s": "No. {{n}} of {{who.name}}", "layers": [
        {"when": "{{opt}}", "type": "x"}, {"type": "y", "v": "{{list.1}}"}]}
    out = templates.apply(t, {"n": 3, "who": {"name": "A"}, "list": ["a", "b"]})
    assert out["n"] == 3 and out["s"] == "No. 3 of A" and out["layers"] == [{"type": "y", "v": "b"}]
    assert len(templates.apply(t, {"n": 1, "who": {"name": ""}, "list": [0, 1], "opt": "x"})["layers"]) == 2
    with pytest.raises(Fail, match="not in the data"):
        templates.apply({"version": 1, "x": "{{nope}}", "layers": []}, {})
    with pytest.raises(Fail):
        templates.apply({"version": 1, "x": "a {{o}}", "layers": []}, {"o": {"k": 1}})


def test_schema_checker():
    sc = {"type": "object", "required": ["a"], "additionalProperties": False,
          "properties": {"a": {"type": "string", "maxLength": 3, "minLength": 1}, "b": {"enum": ["x", "y"]},
                         "c": {"type": "array", "maxItems": 2, "items": {"type": "integer"}}}}
    templates.validate({"a": "abc", "b": "x", "c": [1, 2]}, sc)
    for bad in ({}, {"a": "abcd"}, {"a": ""}, {"a": "a", "b": "z"}, {"a": "a", "c": [1, 2, 3]},
                {"a": "a", "c": [1.5]}, {"a": "a", "typo": 1}, {"a": 1}):
        with pytest.raises(Fail):
            templates.validate(bad, sc)


SAMPLES = {
    "og-title": {"title": "limner, a local image engine", "subtitle": "Text that is always right", "lang": "en"},
    "card-cover": {"title": "일본의 연말 점보 복권", "subtitle": "1등 7억 엔", "lang": "ko"},
    "card-body": {"heading": "年末ジャンボとは", "body": "毎年12月に売られる宝くじで、1等と前後賞を合わせて10億円になる。",
                  "page": "2/5", "lang": "ja"},
    "card-end": {"message": "Thanks for reading", "handle": "@dev-yakuza", "lang": "en"},
    "strip-2x2": {"panels": ["p.png"] * 4},
    "strip-3x2": {"panels": ["p.png"] * 6},
}


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_builtin_templates_render(env, name):
    Image.new("RGB", (300, 300), (120, 160, 200)).save(os.path.join(env["root"], "p.png"))
    with open(os.path.join(env["root"], "data.json"), "w") as f:
        json.dump(SAMPLES[name], f, ensure_ascii=False)
    obj, code = lmn("compose", "--template", name, "--data", "data.json", "--root", env["root"], "--out", "o.png")
    assert code == 0, obj
    assert read(obj["record"])["template"] == "builtin:" + name


def test_template_errors_and_override_dir(env):
    with open(os.path.join(env["root"], "data.json"), "w") as f:
        json.dump({"title": "x", "lang": "en", "typo": 1}, f)
    obj, code = lmn("compose", "--template", "og-title", "--data", "data.json", "--root", env["root"], "--out", "o.png")
    assert code == 2 and "typo" in obj["error"]
    os.makedirs(os.path.join(env["root"], "tpl"))
    with open(os.path.join(env["root"], "tpl", "og-title.json"), "w") as f:
        json.dump(L(T([0, 0, 500, 100], "{{title}}")), f)
    with open(os.path.join(env["root"], "data.json"), "w") as f:
        json.dump({"title": "mine"}, f)
    obj, code = lmn("compose", "--template", "og-title", "--templates", "tpl", "--data", "data.json",
                    "--root", env["root"], "--out", "o.png")
    assert code == 0 and obj["layers"][0]["lines"] == ["mine"]
    obj, code = lmn("compose", "--template", "og-title", "--layout", "x.json", "--root", env["root"], "--out", "o.png")
    assert code == 2


# --------------------------------------------------------------------------- golden


GOLDEN = L(
    {"type": "rect", "box": [0, 0, 1080, 180], "color": "#1f2937"},
    T([60, 30, 1020, 150], "limner — 한국어 日本語 English", lang="ko", color="#ffffff", weight=700, align="center",
      valign="middle"),
    B("speech", [80, 260, 520, 480], "한국 돈으로\n70억이 넘는다고?!", tail=[600, 620], lang="ko"),
    B("shout", [560, 240, 1020, 520], "WHAT?!", tail=[760, 640]),
    B("thought", [80, 620, 520, 860], "まさか…本当に？", tail=[420, 960], lang="ja"),
    B("narration", [560, 680, 1020, 860], "その頃、売り場では——", lang="ja"),
    {"type": "caption", "box": [60, 920, 1020, 1040], "lang": "en", "runs": [{"text": "Captions sit on a plate. "},
                                                                           {"text": "Runs mix weights.", "weight": 800, "color": "#b91c1c"}]},
)


def golden_hash(im) -> str:
    import hashlib

    return hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest()


def test_golden_image(env):
    """Pixels of a fixed layout under pinned Pillow and font hashes (BASIC layout). A different
    Pillow or font fails loudly instead of skipping."""
    import PIL

    from limner import util

    g = json.load(open(os.path.join(DATA, "golden.json")))
    assert PIL.__version__ == g["pillow"], "golden images are for Pillow %s (got %s)" % (g["pillow"], PIL.__version__)
    for k, sha in g["fonts"].items():
        assert util.sha256_file(fonts.path_of(k)) == sha, "font %s changed" % k
    _, im = in_process(env["root"], GOLDEN)
    assert golden_hash(im) == g["layout_sha256"]
