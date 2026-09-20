"""2D contact-sheet renderer.

Draws a spec's `plan()` output as a plan view plus a side elevation.  Pure
shapely + PIL, no OCC, so a board costs a fraction of one 3D build.

Rendered on a light green ground so through-holes are unmistakable: a hole shows
the ground through it.  Supersampled before downscaling so edges stay smooth.
See the legend for raised / flush / recessed / through / accent.

Shapes are painted through a MASK rather than as filled polygons.  PIL has no
notion of a polygon with holes, and getting that wrong bites twice: punching
interiors to the ground colour paints fake holes through solid material (a bore
collar is an annulus whose middle is still part), while ignoring interiors fills
an accent ring in as a solid disc.  A mask handles both correctly.

What it can judge: silhouette, where features sit, how dense, colour split.
What it CANNOT judge: chamfer quality, how light breaks over drafted faces.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from shapely.geometry import Polygon as ShPoly

SS = 2                                   # supersample factor
PAGE   = (0xDA, 0xE8, 0xD4)              # sheet background
GROUND = (0xBF, 0xDA, 0xB6)              # cell background = what a THROUGH hole shows
EDGE   = (0x6F, 0x86, 0x69)
INK    = (0x1A, 0x1D, 0x21)
DIM    = (0x4E, 0x5A, 0x4A)
SHADOW = (0x8E, 0x95, 0x9E)


def _font(sz, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try: return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", n), sz)
        except Exception: pass
    return ImageFont.load_default()


def _polys(g):
    if g is None or (hasattr(g, "is_empty") and g.is_empty): return []
    if g.geom_type == "Polygon": return [g]
    if g.geom_type == "MultiPolygon": return list(g.geoms)
    return []


def _paint(img, g, xf, fill, outline=None, width=1):
    """Fill a (Multi)Polygon honouring interior rings, without repainting what
    lies underneath.  Interiors are excluded from the mask, never painted over."""
    ps = _polys(g)
    if not ps: return
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for p in ps:
        md.polygon([xf(x, y) for x, y in p.exterior.coords], fill=255)
        for r in p.interiors:
            md.polygon([xf(x, y) for x, y in r.coords], fill=0)
    img.paste(fill, mask=mask)
    if outline:
        d = ImageDraw.Draw(img)
        for p in ps:
            d.line([xf(x, y) for x, y in p.exterior.coords], fill=outline, width=width)
            for r in p.interiors:
                d.line([xf(x, y) for x, y in r.coords], fill=outline, width=width)


def _mask(img, ps, xf):
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for p in ps:
        md.polygon([xf(x, y) for x, y in p.exterior.coords], fill=255)
        for r in p.interiors:
            md.polygon([xf(x, y) for x, y in r.coords], fill=0)
    return mask


def _shade(img, g, xf, factor=0.74, outline=None, width=1):
    """Darken what is underneath: a RECESS, which is not the same as a hole.
    Side-wall pockets are only a few mm deep -- painting them in the ground
    colour would claim they go through, which they do not."""
    ps = _polys(g)
    if not ps: return
    img.paste(ImageEnhance.Brightness(img).enhance(factor), mask=_mask(img, ps, xf))
    if outline:
        d = ImageDraw.Draw(img)
        for p in ps:
            d.line([xf(x, y) for x, y in p.exterior.coords], fill=outline, width=width)


def cell(P, size=(1760, 730), title="", sub="", note=""):
    """One board cell: plan view over side elevation, supersampled."""
    sp = P["spec"]
    pal = __import__("spec").PALETTES[sp["palette"]]
    WH, DK, AC = pal["white"], pal["dark"], pal["accent"]
    Wc, Hc = size[0] * SS, size[1] * SS
    img = Image.new("RGB", (Wc, Hc), GROUND)
    ImageDraw.Draw(img).rectangle([0, 0, Wc - 1, Hc - 1], outline=EDGE, width=SS)

    gx0, gy0, gx1, gy1 = P["grown"].bounds
    ZB, ZT, ZTOP = P["ZB"], P["ZT"], P["ZTOP"]
    m, top = 22 * SS, 52 * SS
    s = (Wc - 2 * m) / (gx1 - gx0)
    ph = (gy1 - gy0) * s

    # ---- plan view -------------------------------------------------------
    oy = top
    xf = lambda x, y: (m + (x - gx0) * s, oy + (gy1 - y) * s)
    _paint(img, P["grown"], xf, WH, outline=EDGE, width=SS)
    # raised elements get a drop shadow: a white pad on a white face is invisible
    for k, h in (("frame", sp["frame_h"]), ("rail", sp["rail_h"]), ("pads", sp["pad_h"])):
        if h <= 0.3: continue
        off = max(1.0, min(h * 0.55, 5.0)) * SS
        sxf = lambda x, y, o=off: (m + (x - gx0) * s + o, oy + (gy1 - y) * s + o)
        _paint(img, P[k], sxf, SHADOW)
        _paint(img, P[k], xf, WH, outline=(0xFF, 0xFF, 0xFF), width=SS)
    for k in ("collars", "pock", "wins"):
        _paint(img, P[k], xf, DK)
    for k in ("strip", "ring", "blocks"):
        _paint(img, P.get(k), xf, AC)
    for k in ("KEEP", "cutouts"):
        _paint(img, P.get(k), xf, GROUND, outline=(0x53, 0x63, 0x4F), width=max(1, SS // 2))

    # ---- side elevation --------------------------------------------------
    oy2 = top + ph + 18 * SS
    zf = lambda x, z: (m + (x - gx0) * s, oy2 + (ZTOP - z) * s)
    CH = max(sp["chamfer"], 0.01)
    body = ShPoly([(gx0 + CH, ZB), (gx1 - CH, ZB), (gx1, ZB + CH), (gx1, ZT - CH),
                   (gx1 - CH, ZT), (gx0 + CH, ZT), (gx0, ZT - CH), (gx0, ZB + CH)])
    SZ = sp["split_z"]
    _paint(img, body, zf, DK)
    _paint(img, body.intersection(ShPoly([(gx0 - 9, SZ), (gx1 + 9, SZ),
                                          (gx1 + 9, ZT + 9), (gx0 - 9, ZT + 9)])), zf, WH)
    for key, h in (("frame", sp["frame_h"]), ("rail", sp["rail_h"]), ("pads", sp["pad_h"])):
        if h <= 0.3: continue
        for p in _polys(P[key]):
            a, _, b, _ = p.bounds
            _paint(img, ShPoly([(a, ZT), (b, ZT), (b - 1.5, ZT + h), (a + 1.5, ZT + h)]),
                   zf, WH, outline=EDGE, width=max(1, SS // 2))
    for pr in P["lo_pr"] + P["hi_pr"]:
        _shade(img, ShPoly(pr).intersection(body), zf,
               outline=(0x44, 0x4B, 0x52), width=max(1, SS // 2))

    # ---- labels ----------------------------------------------------------
    dr = ImageDraw.Draw(img)
    dr.text((m, 11 * SS), title, font=_font(23 * SS, True), fill=INK)
    if sub: dr.text((m, 35 * SS), sub, font=_font(15 * SS), fill=DIM)
    if note: dr.text((m, Hc - 24 * SS), note, font=_font(15 * SS), fill=DIM)
    return img.resize(size, Image.LANCZOS)


def sheet(cells, cols, out, header="", sub="", size=(1760, 730)):
    """Composite cells into ONE png -- never N separate files."""
    rows = (len(cells) + cols - 1) // cols
    hh = 132 if header else 0
    W, H = cols * size[0], hh + rows * size[1]
    img = Image.new("RGB", (W, H), PAGE)
    dr = ImageDraw.Draw(img)
    if header:
        dr.text((30, 16), header, font=_font(38, True), fill=INK)
        if sub: dr.text((30, 60), sub, font=_font(22), fill=DIM)
        pal = __import__("spec").PALETTES["arctic"]
        key = [("RAISED", pal["white"], True), ("FLUSH", pal["white"], False),
               ("GRAPHITE PANEL", pal["dark"], False),
               ("RECESS (blind, not through)", (0xBC, 0xBE, 0xC0), False),
               ("THROUGH", GROUND, False), ("ACCENT", pal["accent"], False)]
        x, f = 30, _font(20)
        for name, col, shadow in key:
            if shadow: dr.rectangle([x + 5, 101, x + 35, 127], fill=SHADOW)
            dr.rectangle([x, 96, x + 30, 122], fill=col, outline=EDGE)
            dr.text((x + 40, 99), name, font=f, fill=INK)
            x += 40 + int(dr.textlength(name, font=f)) + 42
    for i, c in enumerate(cells):
        img.paste(c, ((i % cols) * size[0], hh + (i // cols) * size[1]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print(f"wrote {out}  ({W}x{H})")
    return out
