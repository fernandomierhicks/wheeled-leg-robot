"""Colour-aware 3D renderer: all filament bodies in one z-buffer, tiled to ONE png.

`render3d.render()` shades everything the same grey, which cannot show a colour
split -- and the whole point of this layer is the colour split.  This renders the
white / graphite / accent bodies together, each in its own palette colour.

ONE TRAP, PAID FOR ONCE: a filament body is usually SEVERAL solids (the accent
came out as five).  `import_step(f).solids()[0]` silently renders one fragment --
it once produced a render of a part with almost no accent on it while the build
was perfectly fine.  Always tessellate the whole compound, and cross-check the
volume printed here against the one `build()` reported.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (0x17, 0x19, 0x1C)
ORDER = ("graphite", "white", "accent")
VIEWS = {"iso": (24, -58), "top": (89.9, -90), "side": (0.5, -90),
         "low": (8, -128), "rear": (20, -122)}


def _font(sz, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", n), sz)
        except Exception:
            pass
    return ImageFont.load_default()


def tess_bodies(res, pal, dev=0.13):
    """[(verts, tris, rgb)] for the three filament bodies of a build() result."""
    from render3d import tessellate
    out, allV = [], []
    for name in ORDER:
        b = res.get(name if name != "graphite" else "graphite")
        if b is None or b.volume < 1:
            continue
        V, T, _ = tessellate(b, dev)
        if len(T) == 0:
            continue
        col = pal["accent"] if name == "accent" else (
            pal["white"] if name == "white" else pal["dark"])
        out.append((V, T, col))
        allV.append(V)
    return out, (np.vstack(allV) if allV else np.zeros((1, 3)))


def view(bodies, allV, size, elev, azim, bg=BG, light=None):
    """`size` is an int (square) or (W, H).  These parts are ~7:1, so a square
    frame spends most of its pixels on background -- pass a wide one.

    `light="camera"` puts the light at the viewer.  The fixed default points
    roughly +Z, so a face-on view of a -Z face renders every triangle at the
    0.24 ambient floor: the Femur's SHOW face comes out near-black and its BACK
    comes out bright, whatever colour either of them actually is.  Judging a
    colour split from such a view reads it exactly backwards.  The default is
    unchanged so every existing board reproduces.
    """
    WD, HT = (size, size) if isinstance(size, (int, float)) else size
    e, a = np.radians(elev), np.radians(azim)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    Pall = np.stack([allV @ right, allV @ up, allV @ fwd], 1)
    mn, mx = Pall[:, :2].min(0), Pall[:, :2].max(0)
    ctr = (mn + mx) / 2
    ext = np.maximum(mx - mn, 1e-6)
    s = min(WD / (ext[0] * 1.06), HT / (ext[1] * 1.10))
    img = np.full((HT, WD, 3), bg, np.uint8)
    zbuf = np.full((HT, WD), -1e18)
    if light == "camera":
        light = fwd / np.linalg.norm(fwd)
    elif light is None:
        light = np.array([0.35, 0.55, 0.75]); light /= np.linalg.norm(light)
    else:
        light = np.asarray(light, float); light /= np.linalg.norm(light)
    for V, T, base in bodies:
        P = np.stack([V @ right, V @ up, V @ fwd], 1)
        xy = (P[:, :2] - ctr) * s + np.array([WD / 2, HT / 2])
        xy[:, 1] = HT - xy[:, 1]
        base = np.asarray(base, float)
        for ti in np.argsort(P[T][:, :, 2].mean(1)):
            i0, i1, i2 = T[ti]
            n = np.cross(V[i1] - V[i0], V[i2] - V[i0])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n /= ln
            if n @ fwd <= 0:
                continue
            lam = max(0.0, n @ light)
            col = np.clip(base * (0.24 + 0.78 * lam ** 0.85) + 34 * lam ** 8, 0, 255)
            q = xy[[i0, i1, i2]]
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, WD - 1), min(y1, HT - 1)
            if x1 < x0 or y1 < y0:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            d = (q[1, 1] - q[2, 1]) * (q[0, 0] - q[2, 0]) + (q[2, 0] - q[1, 0]) * (q[0, 1] - q[2, 1])
            if abs(d) < 1e-9:
                continue
            w0 = ((q[1, 1] - q[2, 1]) * (xx - q[2, 0]) + (q[2, 0] - q[1, 0]) * (yy - q[2, 1])) / d
            w1 = ((q[2, 1] - q[0, 1]) * (xx - q[2, 0]) + (q[0, 0] - q[2, 0]) * (yy - q[2, 1])) / d
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not m.any():
                continue
            z = w0 * P[i0, 2] + w1 * P[i1, 2] + w2 * P[i2, 2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            m &= z > sub
            sub[m] = z[m]
            img[y0:y1 + 1, x0:x1 + 1][m] = col[None, None, :]
    return Image.fromarray(img)


def sheet(tiles, cols, out, header="", sub="", size=760, hh=104):
    """tiles = [(title, subtitle, PIL image)] -> one png."""
    TW, TH = (size, size) if isinstance(size, (int, float)) else size
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * TW, hh + rows * (TH + 46) + 18   # 18: the last row's label
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if header:
        d.text((26, 18), header, font=_font(38, True), fill=(0xF2, 0xF4, 0xF7))
        if sub:
            d.text((26, 63), sub, font=_font(20), fill=(0x93, 0x9C, 0xA8))
    for i, (title, st, im) in enumerate(tiles):
        x, y = (i % cols) * TW, hh + (i // cols) * (TH + 46)
        img.paste(im, (x, y))
        d.text((x + 16, y + TH + 4), title, font=_font(23, True), fill=(0xE6, 0xEA, 0xEF))
        if st:
            d.text((x + 16, y + TH + 26), st, font=_font(16), fill=(0x8B, 0x94, 0xA1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print(f"wrote {out}  ({W}x{H})")
    return out
