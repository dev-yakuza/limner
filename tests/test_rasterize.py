"""rasterize (spec §8): sanitizing, forced fonts, mono, limits. Needs the pinned fonts."""

import io
import os

import pytest
from PIL import Image

from conftest import lmn
from helpers import read
from limner import fonts, rasterize


@pytest.fixture(autouse=True)
def need_fonts(monkeypatch):
    monkeypatch.delenv("LIMNER_FONTS_DIR", raising=False)
    assert all(fonts.installed(k) for k in fonts.FONTS), "run `lmn fonts install` first"


def svg(body, w=400, h=80, head=""):
    return ('%s<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="%d" height="%d"><rect width="%d" height="%d" fill="white"/>%s</svg>' % (head, w, h, w, h, body))


def run(root, text, out="o.png", *extra):
    with open(os.path.join(str(root), "in.svg"), "w") as f:
        f.write(text)
    return lmn("rasterize", "--svg", "in.svg", "--root", root, "--out", out, *extra)


def dark(root, out="o.png", box=None):
    im = Image.open(os.path.join(str(root), out)).convert("L")
    if box:
        im = im.crop(box)
    return sum(1 for p in im.getdata() if p < 128)


def ink_width(root, out="o.png"):
    im = Image.open(os.path.join(str(root), out)).convert("L").point(lambda p: 255 if p < 128 else 0)
    bb = im.getbbox()
    return bb[2] - bb[0]


def test_dtd_and_entities_are_refused(env):
    for head in ('<!DOCTYPE svg [<!ENTITY x "y">]>', '<!DOCTYPE svg SYSTEM "http://x/y.dtd">'):
        obj, code = run(env["root"], svg('<text x="5" y="40">&amp;</text>', head=head))
        assert code == 2 and "DTD" in obj["error"]


def test_local_files_are_never_read(env):
    secret = os.path.join(env["root"], "secret.png")
    Image.new("RGB", (400, 80), (255, 0, 0)).save(secret)
    body = ('<image width="400" height="80" href="%s"/><image width="400" height="80" xlink:href="secret.png"/>'
            '<filter id="f"><feImage href="%s"/></filter><rect width="10" height="10" filter="url(#f)"/>'
            '<rect width="400" height="80" fill="url(%s#p)"/>' % (secret, secret, secret))
    obj, code = run(env["root"], svg(body))
    assert code == 4 and [p["code"] for p in obj["problems"]] == ["external_ref_removed"]
    im = Image.open(os.path.join(env["root"], "o.png")).convert("RGB")
    assert not any(p[0] > 200 and p[1] < 60 for p in im.getdata())
    clean = open(os.path.join(env["root"], ".limner", read(obj["record"])["sanitized"][len(".limner/"):])).read()
    assert secret not in clean and "secret.png" not in clean


def test_data_images_and_internal_refs_stay(env):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 0, 255)).save(buf, format="PNG")
    import base64

    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = ('<defs><rect id="r" width="20" height="20" fill="black"/></defs><use href="#r"/>'
            '<image x="100" width="40" height="40" href="%s"/>' % uri)
    obj, code = run(env["root"], svg(body))
    assert code == 0, obj
    im = Image.open(os.path.join(env["root"], "o.png")).convert("RGB")
    assert im.getpixel((5, 5)) == (0, 0, 0) and im.getpixel((120, 20))[2] > 200


def test_style_imports_and_urls_removed(env):
    body = ('<style>@import url("/etc/x.css"); rect.a { fill: url(/abs/p.svg#g); } </style>'
            '<rect class="a" width="10" height="10"/>')
    obj, code = run(env["root"], svg(body))
    assert code == 4 and obj["problems"][0]["code"] == "external_ref_removed"
    where = obj["problems"][0]["detail"]["where"]
    assert "style@import" in where and "style url()" in where


def test_scripts_events_foreign_objects_removed(env):
    body = ('<script>alert(1)</script><rect width="10" height="10" onclick="x()"/>'
            '<foreignObject width="10" height="10"><div xmlns="http://www.w3.org/1999/xhtml">x</div></foreignObject>')
    obj, code = run(env["root"], svg(body))
    assert code == 4 and [p["code"] for p in obj["problems"]] == ["unsafe_removed"]


@pytest.mark.parametrize("body", [
    '<text x="5" y="50" font-size="36" font-family="Nope">가나Ab字</text>',  # attribute
    '<text x="5" y="50" font-size="36" style="font-family: Nope">가나Ab字</text>',  # style=""
    '<style>text { font-family: Nope }</style><text x="5" y="50" font-size="36">가나Ab字</text>',  # <style>
    '<text x="5" y="50" font-size="36">가나Ab字</text>',  # no family
    '<text x="5" y="50" font-size="36" font-family="Inter, sans-serif">가나Ab字</text>',  # generic
    '<text x="5" y="50" font-size="36" font-family="Noto Sans KR Regular">가나Ab字</text>',  # name mismatch
    '<style>text { font-family: Nope !important }</style><text x="5" y="50" font-size="36">가나Ab字</text>',
])
def test_text_is_always_drawn(env, body):
    obj, code = run(env["root"], svg(body))
    assert code == 0, obj
    assert dark(env["root"]) > 300


def test_font_substituted_is_a_note(env):
    obj, code = run(env["root"], svg('<text x="5" y="50" font-size="36" font-family="Inter, sans-serif">Ab</text>'))
    assert code == 0 and obj["notes"] == [{"code": "font_substituted", "detail": {"families": ["Inter"]}}]


MONO_TEXT = "iiiiiiii"


@pytest.mark.parametrize("body", [
    '<text x="5" y="50" font-size="36" font-family="monospace">%s</text>',
    '<style>.code { font-family: Menlo, monospace }</style><text class="code" x="5" y="50" font-size="36">%s</text>',
    '<g font-family="monospace"><text x="5" y="50" font-size="36">%s</text></g>',
    '<text x="5" y="50" font-size="36" font-family="monospace"><tspan>%s</tspan></text>',
])
def test_mono_is_drawn_with_the_mono_font(env, body):
    run(env["root"], svg('<text x="5" y="50" font-size="36">%s</text>' % MONO_TEXT), out="sans.png")
    sans = ink_width(env["root"], "sans.png")
    obj, code = run(env["root"], svg(body % MONO_TEXT), out="mono.png")
    assert code == 0, obj
    assert ink_width(env["root"], "mono.png") > sans * 1.5  # i is narrow in sans, a full cell in mono


def test_missing_glyph(env):
    obj, code = run(env["root"], svg('<text x="5" y="50" font-size="36">ok \U0001F600</text>'))
    assert code == 4 and obj["problems"][0]["code"] == "missing_glyph"


def test_size_limits_and_width(env):
    obj, code = run(env["root"], '<svg xmlns="http://www.w3.org/2000/svg" width="20000" height="20000"/>')
    assert code == 2 and "limit" in obj["error"]
    obj, code = run(env["root"], svg(""), "o.png", "--width", "9000")
    assert code == 2
    obj, code = run(env["root"], svg('<rect width="10" height="10"/>', w=200, h=100), "o.png", "--width", "800")
    assert code == 0 and Image.open(os.path.join(env["root"], "o.png")).size == (800, 400)


def test_background_and_record(env):
    obj, code = run(env["root"], '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"/>', "o.png",
                    "--background", "#ff0000")
    assert code == 0
    assert Image.open(os.path.join(env["root"], "o.png")).convert("RGB").getpixel((5, 5)) == (255, 0, 0)
    rec = read(obj["record"])
    assert rec["kind"] == "rasterize" and rec["svg_sha256"] and rec["sanitized_sha256"]


def test_effective_family_rules():
    mono = rasterize.MONO_NAMES
    text = ('<svg xmlns="http://www.w3.org/2000/svg"><style>#a { font-family: monospace } .b { font-family: Arial }'
            ' text.c { font-family: Courier }</style>'
            '<text id="a" class="b">1</text><text class="b" style="font-family: monospace">2</text>'
            '<text class="c">3</text><text font-family="monospace" class="b">4</text></svg>')
    clean, s = rasterize.sanitize(text, mono)
    assert [m for _, m in s.texts] == [True, True, True, False]  # id > class; style > rules; rules > attribute


# --------------------------------------------------------------------------- review round 1


T1 = "가나Ab字"


@pytest.mark.parametrize("body", [
    '<text x="5" y="50" font-size="36"><a style="font-family:Nope">%s</a></text>' % T1,
    '<text x="5" y="50" font-size="36"><a font-family="Nope">%s</a></text>' % T1,
    '<style>a { font-family: Nope }</style><text x="5" y="50" font-size="36"><a>%s</a></text>' % T1,
    '<style>text { font-family: Nope !/**/important }</style><text x="5" y="50" font-size="36">%s</text>' % T1,
    '<style>text { font-family: Nope ! /* x */ important }</style><text x="5" y="50" font-size="36">%s</text>' % T1,
    '<text x="5" y="50" font-size="36" style="font-family:Nope !/**/important">%s</text>' % T1,
])
def test_text_inside_links_and_commented_important_is_drawn(env, body):
    """R1-3."""
    obj, code = run(env["root"], svg(body))
    assert code == 0, obj
    assert dark(env["root"]) > 300


def test_text_not_drawn_is_reported(env):
    """R1-3 safety net: text whose glyphs leave no ink is a problem, not a silent success."""
    obj, code = run(env["root"], svg('<text x="5" y="50" font-size="36" fill="none">abc</text>'))
    assert code == 4 and [p["code"] for p in obj["problems"]] == ["text_not_drawn"]


@pytest.mark.parametrize("root_attrs", [
    'width="700em" height="700em" viewBox="0 0 10 10"',
    'width="2000ex" height="2000ex" viewBox="0 0 10 10"',
    'width="100%" height="100%"',
    '',
])
def test_size_limit_cannot_be_bypassed_by_units(env, root_attrs):
    """R1-4: limner's size is the size resvg renders (or the SVG is refused)."""
    s = '<svg xmlns="http://www.w3.org/2000/svg" %s><rect width="9000" height="9000" fill="red"/></svg>' % root_attrs
    obj, code = run(env["root"], s)
    assert code == 2, obj


def test_em_size_is_what_resvg_draws(env):
    s = '<svg xmlns="http://www.w3.org/2000/svg" width="10em" height="5em" viewBox="0 0 10 5"><rect width="10" height="5"/></svg>'
    obj, code = run(env["root"], s)
    assert code == 0 and Image.open(os.path.join(env["root"], "o.png")).size == (160, 80)


def test_data_image_bomb_is_refused(env):
    """R1-4: a small data: PNG that decodes to a huge image."""
    import base64

    buf = io.BytesIO()
    Image.new("1", (9000, 9000), 1).save(buf, format="PNG", optimize=True)  # 81M px, about 10 KB
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    obj, code = run(env["root"], svg('<image width="100" height="80" href="%s"/>' % uri))
    assert code == 2 and "data:" in obj["error"], obj


def test_descendant_selectors_are_matched_whole():
    """R1-10."""
    text = ('<svg xmlns="http://www.w3.org/2000/svg"><style>g.code text { font-family: monospace }'
            ' text:first-child { font-family: monospace }</style>'
            '<g class="code"><text>1</text></g><g><text>2</text></g></svg>')
    clean, s = rasterize.sanitize(text, rasterize.MONO_NAMES)
    assert [m for _, m in s.texts] == [True, False]
    assert any(n["code"] == "css_selector_ignored" for n in s.notes)
