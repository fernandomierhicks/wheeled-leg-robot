r"""Phase 2 markup sheets -- the sheets Fernando draws on, and the transform
that makes his drawing mean something.

    C:/Users/ferna/cadenv/Scripts/python.exe tools/marksheet.py            # both sets
    C:/Users/ferna/cadenv/Scripts/python.exe tools/marksheet.py --asm      # assembly only
    C:/Users/ferna/cadenv/Scripts/python.exe tools/marksheet.py --parts    # per-part only
    C:/Users/ferna/cadenv/Scripts/python.exe tools/marksheet.py --quick    # small + coarse, for a smoke test

THE LEGEND IS HIS, AND IT IS THE INVERSE OF THE ONE HANDOFF.md ORIGINALLY SPECIFIED
(decision 24).  RED = material MAY BE REMOVED.  GREEN = material MAY BE ADDED.
Anything unmarked is left alone and the derived keep-out still applies there, so
silence is the conservative answer rather than the permissive one.  Reading a
sheet under the old legend would cut metal exactly where it must never be touched,
which is why the legend is printed on every sheet in words, not just in colour.

WHY THIS IS NOT parts/render_part.py.  A styled render is a picture; a markup
sheet is a measuring instrument.  It must be:

  - ORTHOGRAPHIC and AXIS-ALIGNED, so pixel -> mm is exact and constant.  A
    perspective view makes a mark unmappable and there is no way to tell from
    the png that it happened.
  - UNSTYLED, and in colours nowhere near red or green, so his pen is always
    the most saturated thing on the sheet.
  - accompanied by a SIDECAR, which is the whole technical enabler:
      <sheet>.json   camera basis, px_per_mm, and every part's 4x4 transform
      <sheet>.npz    per-pixel part id + per-pixel depth
    With those, a marked pixel resolves to (part, exact point in that part's OWN
    local frame) -- the frame the recipes build in.  Without them a mark is
    worth about as much as a gesture at a screen.

OCCLUSION, handled rather than suffered.  In the outboard view the wheel sits
proud of everything at Z 169..203 and would hide the whole tibia end.  So the
in-scope parts are rasterised in their own pass and always win the pixel;
context that sits IN FRONT of them is drawn back over as a 50% checkerboard.
The clearance stays visible, the markable surface is never hidden, and the
dither says plainly "something is in front of you here".
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import paths
from build123d import import_step
from render3d import tessellate
from collide import leaves

POSE = "Middle single part"          # decision 24: he chose Middle only
MARKS = os.path.join(paths.OUT, "marks")

BG      = (250, 250, 251)            # near-white: a red or green pen has to win
CONTEXT = (172, 176, 182)            # motors, bearings, fasteners, wheel
INK     = (24, 26, 30)
GRID    = (214, 219, 226)
GRID50  = (178, 186, 196)

# Tints for the parts he may mark.  Every one is a blue / violet / teal / amber:
# NOTHING in the red or green families, so his pen never competes with the part.
SCOPE = {
    "Femur":                  ((92, 122, 172),  "Femur"),
    "Femur_inside_InsideBox": ((124, 154, 198), "Femur (mirrored, left)"),
    "Tibia":                  ((132, 112, 178), "Tibia"),
    "Coupler":                ((68, 146, 162),  "Coupler"),
    "Side panel":             ((118, 132, 154), "Side panel"),
    "RobotMount":             ((96, 106, 176),  "RobotMount"),
    "EncoderCarrier":         ((168, 142, 96),  "EncoderCarrier (unstyled)"),
    "EncoderCableClamp":      ((186, 162, 118), "EncoderCableClamp (unstyled)"),
    "Switch_Mount":           ((150, 126, 84),  "Switch_Mount (unstyled)"),
}

# fwd is the direction from the object TOWARDS the camera.
ASM_VIEWS = [
    ("outboard", (0, 0, 1), (0, 1, 0),
     "OUTBOARD  -  looking inboard down -Z.  THE SHOW FACE of every part."),
    ("inboard", (0, 0, -1), (0, 1, 0),
     "INBOARD  -  looking outboard up +Z.  THE BACK of every part, the side you have never been shown."),
    ("edge_+X", (1, 0, 0), (0, 0, 1),
     "EDGE-ON from +X.  Lateral stack-up: outboard is UP the page."),
    ("edge_-X", (-1, 0, 0), (0, 0, 1),
     "EDGE-ON from -X.  Lateral stack-up: outboard is UP the page."),
    ("edge_+Y", (0, 1, 0), (0, 0, 1),
     "EDGE-ON from +Y, looking from the hip towards the wheel.  Outboard is UP the page."),
    ("edge_-Y", (0, -1, 0), (0, 0, 1),
     "EDGE-ON from -Y, looking from the wheel towards the hip.  Outboard is UP the page."),
]

PART_VIEWS = [
    ("show",     "SHOW FACE  -  the side seen on the finished robot"),
    ("back",     "BACK FACE  -  hidden in the assembly, must carry the same aesthetic"),
    ("end_X",    "END-ON down X  -  added material must read as structure, not as a fin"),
    ("end_Y",    "END-ON down Y  -  added material must read as structure, not as a fin"),
]


def _font(sz, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", n), sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _axname(v):
    """'+X' / '-Y' ... for an axis-aligned basis vector.  Derived, not typed in,
    so the label on the sheet cannot drift from the camera that made it."""
    i = int(np.argmax(np.abs(v)))
    return f"{'+' if v[i] > 0 else '-'}{'XYZ'[i]}"


MAX_PX = 1800        # longest side of the drawing area
MIN_PX = 560
MAX_SCALE = 12.0     # px/mm; past this the sheet is just a huge file
PAD_PX = 64


def fit_canvas(fwd, up, pts):
    """Size the sheet to the SUBJECT, not the other way round.

    The femur is 232 x 65 mm.  Forcing it into a fixed 1680 x 1300 frame spends
    two thirds of the sheet on empty background and costs a third of the
    resolution he has to mark at, so the canvas takes the subject's aspect and
    the scale is capped so a small end-on section does not blow up to 30 px/mm.
    """
    fwd, up = _unit(fwd), _unit(up)
    right = _unit(np.cross(up, fwd))
    B = np.stack([right, np.cross(fwd, right), fwd], 1)
    P = np.asarray(pts, float) @ B
    ext = np.maximum(P[:, :2].max(0) - P[:, :2].min(0), 1e-6)
    s = min(MAX_PX / ext[0], MAX_PX / ext[1], MAX_SCALE)
    return (max(MIN_PX, int(ext[0] * s)) + 2 * PAD_PX,
            max(MIN_PX, int(ext[1] * s)) + 2 * PAD_PX)


class Ortho:
    """An orthographic camera whose pixel<->mm map is exact and recorded.

    Image x runs along `right`, image y runs DOWN so it matches PIL and numpy.
    Everything needed to invert the map lives in `meta()`; nothing is implicit.
    """

    def __init__(self, fwd, up, W, H, pts, pad_px=PAD_PX):
        self.fwd = _unit(fwd)
        self.right = _unit(np.cross(_unit(up), self.fwd))
        self.up = np.cross(self.fwd, self.right)
        self.W, self.H = W, H
        self.B = np.stack([self.right, self.up, self.fwd], 1)   # world -> view
        P = np.asarray(pts, float) @ self.B
        mn, mx = P[:, :2].min(0), P[:, :2].max(0)
        self.ctr = (mn + mx) / 2
        ext = np.maximum(mx - mn, 1e-6)
        self.s = min((W - 2 * pad_px) / ext[0], (H - 2 * pad_px) / ext[1])   # px per mm

    def view(self, V):
        return np.asarray(V, float) @ self.B

    def to_px(self, P):
        x = (P[:, 0] - self.ctr[0]) * self.s + self.W / 2
        y = self.H / 2 - (P[:, 1] - self.ctr[1]) * self.s
        return np.stack([x, y], 1)

    def px_to_world(self, px, py, depth):
        """A pixel plus its recorded depth -> the exact world point on the surface."""
        r = (px - self.W / 2) / self.s + self.ctr[0]
        u = (self.H / 2 - py) / self.s + self.ctr[1]
        return self.right * r + self.up * u + self.fwd * depth

    def mm_x(self, px):
        return (px - self.W / 2) / self.s + self.ctr[0]

    def mm_y(self, py):
        return (self.H / 2 - py) / self.s + self.ctr[1]

    def x_to_px(self, mm):
        return (mm - self.ctr[0]) * self.s + self.W / 2

    def y_to_px(self, mm):
        return self.H / 2 - (mm - self.ctr[1]) * self.s

    def meta(self):
        return {
            "W": self.W, "H": self.H,
            "px_per_mm": self.s, "mm_per_px": 1.0 / self.s,
            "right": list(self.right), "up": list(self.up), "fwd": list(self.fwd),
            "centre_mm": list(self.ctr),
            "note": "world = right*((px-W/2)/s + ctr[0]) + up*((H/2-py)/s + ctr[1]) "
                    "+ fwd*depth[py,px]; fwd points from the object to the camera",
        }


def raster(items, cam):
    """items = [(V, T, rgb, pid)] -> (float image, depth, id).  pid 0 = nothing."""
    H, W = cam.H, cam.W
    img = np.zeros((H, W, 3), np.float32)
    z = np.full((H, W), -1e18, np.float32)
    ids = np.zeros((H, W), np.uint16)
    light = _unit(cam.fwd * 0.72 + cam.up * 0.46 + cam.right * 0.34)
    for V, T, col, pid in items:
        if not len(T):
            continue
        P = cam.view(V)
        xy = cam.to_px(P)
        base = np.asarray(col, float)
        n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        ln = np.linalg.norm(n, axis=1)
        ok = (ln > 1e-12)
        n[ok] /= ln[ok, None]
        facing = (n @ cam.fwd) > 0
        for ti in np.nonzero(ok & facing)[0]:
            i0, i1, i2 = T[ti]
            q = xy[[i0, i1, i2]]
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, W - 1), min(y1, H - 1)
            if x1 < x0 or y1 < y0:
                continue
            d = ((q[1, 1] - q[2, 1]) * (q[0, 0] - q[2, 0]) +
                 (q[2, 0] - q[1, 0]) * (q[0, 1] - q[2, 1]))
            if abs(d) < 1e-9:
                continue
            lam = max(0.0, float(n[ti] @ light))
            c = np.clip(base * (0.42 + 0.62 * lam ** 0.8) + 30 * lam ** 10, 0, 255)
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            w0 = ((q[1, 1] - q[2, 1]) * (xx - q[2, 0]) + (q[2, 0] - q[1, 0]) * (yy - q[2, 1])) / d
            w1 = ((q[2, 1] - q[0, 1]) * (xx - q[2, 0]) + (q[0, 0] - q[2, 0]) * (yy - q[2, 1])) / d
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not m.any():
                continue
            zz = w0 * P[i0, 2] + w1 * P[i1, 2] + w2 * P[i2, 2]
            sub = z[y0:y1 + 1, x0:x1 + 1]
            m &= zz > sub
            if not m.any():
                continue
            sub[m] = zz[m]
            img[y0:y1 + 1, x0:x1 + 1][m] = c
            ids[y0:y1 + 1, x0:x1 + 1][m] = pid
    return img, z, ids


def compose(scope, context, cam):
    """In-scope parts ALWAYS win the pixel.  Context in front of them comes back
    as a checkerboard, so the crowding is visible but never hides a surface he
    has to mark."""
    H, W = cam.H, cam.W
    s_img, s_z, s_id = scope
    c_img, c_z, c_id = context
    out = np.full((H, W, 3), BG, np.float32)
    cm = c_id > 0
    sm = s_id > 0
    out[cm] = c_img[cm]
    out[sm] = s_img[sm]
    front = sm & cm & (c_z > s_z)
    yy, xx = np.mgrid[0:H, 0:W]
    checker = ((xx >> 1) + (yy >> 1)) % 2 == 0
    dith = front & checker
    out[dith] = out[dith] * 0.55 + c_img[dith] * 0.45
    return out, s_z, s_id


def grid_and_frame(img, ids, cam, xlabel, ylabel, step=10.0, major=50.0):
    """A millimetre grid on the view's own axes, so a mark can be sized by eye.

    Drawn only on BACKGROUND pixels, with the 50 mm lines ghosted faintly across
    the metal.  A full grid over the part reads as texture, and texture is what
    his pen has to beat -- the first draft of this sheet lost the parts under it.
    """
    H, W = cam.H, cam.W
    bg = ids == 0
    g, g50 = np.array(GRID, np.float32), np.array(GRID50, np.float32)
    lines_x, lines_y = [], []
    x0, x1 = cam.mm_x(0), cam.mm_x(W)
    y1, y0 = cam.mm_y(H), cam.mm_y(0)
    for mm in np.arange(np.ceil(min(x0, x1) / step) * step, max(x0, x1) + 1e-9, step):
        px = int(round(cam.x_to_px(mm)))
        if 0 <= px < W:
            maj = abs((mm + major / 2) % major - major / 2) < 1e-6
            col = bg[:, px]
            img[col, px] = g50 if maj else g
            if maj:
                img[~col, px] = img[~col, px] * 0.74 + g50 * 0.26
                lines_x.append((px, mm))
    for mm in np.arange(np.ceil(min(y0, y1) / step) * step, max(y0, y1) + 1e-9, step):
        py = int(round(cam.y_to_px(mm)))
        if 0 <= py < H:
            maj = abs((mm + major / 2) % major - major / 2) < 1e-6
            row = bg[py, :]
            img[py, row] = g50 if maj else g
            if maj:
                img[py, ~row] = img[py, ~row] * 0.74 + g50 * 0.26
                lines_y.append((py, mm))

    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = _font(15)
    for px, mm in lines_x:
        d.text((px + 3, H - 19), f"{mm:.0f}", font=f, fill=(128, 136, 148))
    for py, mm in lines_y:
        d.text((4, py + 2), f"{mm:.0f}", font=f, fill=(128, 136, 148))
    if xlabel:
        d.text((W - 110, H - 42), xlabel, font=_font(17, True), fill=(110, 118, 132))
    if ylabel:
        d.text((W - 110, H - 64), ylabel, font=_font(17, True), fill=(110, 118, 132))
    return im


HDR = 124


def overlay(im, cam, title, sub, legend, scalebar=50.0):
    """Header band with the legend in WORDS, and a footer band with the part key
    and the scale bar.  Both are OUTSIDE the drawing: nothing the pipeline writes
    may sit on a surface he has to mark."""
    W = cam.W
    ncol = 3
    rows = (len(legend) + 1 + ncol - 1) // ncol if legend else 0
    ftr = 52 + rows * 26
    out = Image.new("RGB", (W, cam.H + HDR + ftr), (255, 255, 255))
    out.paste(im, (0, HDR))
    d = ImageDraw.Draw(out)

    d.rectangle([0, 0, W, HDR - 1], fill=(28, 31, 36))
    d.text((22, 16), title, font=_font(31, True), fill=(242, 244, 247))
    d.text((22, 58), sub, font=_font(17), fill=(150, 158, 170))

    # the legend, stated in words as well as colour -- see module docstring
    lx = max(int(W * 0.52), W - 500)
    d.rectangle([lx - 16, 10, W - 12, HDR - 12], outline=(78, 84, 94))
    d.rectangle([lx, 22, lx + 26, 44], fill=(214, 38, 38))
    d.text((lx + 36, 23), "RED  = material MAY BE REMOVED here",
           font=_font(18, True), fill=(240, 242, 245))
    d.rectangle([lx, 54, lx + 26, 76], fill=(28, 168, 72))
    d.text((lx + 36, 55), 'GREEN = material MAY BE ADDED  (say how far, "+6")',
           font=_font(18, True), fill=(240, 242, 245))
    d.text((lx, 88), "UNMARKED = leave it alone.   Free text welcome.",
           font=_font(16), fill=(158, 166, 178))

    # part key, below the drawing
    y0 = HDR + cam.H + 10
    if legend:
        cw = (W - 30) // ncol
        for i, (col, name) in enumerate(list(legend) + [
                (CONTEXT, "motors / bearings / fasteners / wheel   "
                          "(checkered = sits in front of the part)")]):
            x = 16 + (i % ncol) * cw
            y = y0 + (i // ncol) * 26
            d.rectangle([x, y, x + 24, y + 17], fill=col, outline=(90, 96, 106))
            d.text((x + 32, y - 1), name, font=_font(16), fill=(52, 58, 68))

    # scale bar
    L = scalebar * cam.s
    bx, by = 16, out.size[1] - 28
    for k in range(5):
        d.rectangle([bx + L * k / 5, by, bx + L * (k + 1) / 5, by + 10],
                    fill=INK if k % 2 == 0 else (255, 255, 255), outline=INK)
    d.text((bx + L + 12, by - 4), f"{scalebar:.0f} mm    ({cam.s:.3f} px/mm, "
           f"{1000 / cam.s:.1f} um/px)", font=_font(17, True), fill=INK)
    return out


def write_sheet(png, cam, img, z, ids, key, title, sub, legend, xlabel, ylabel, extra):
    im = grid_and_frame(img, ids, cam, xlabel, ylabel)
    im = overlay(im, cam, title, sub, legend)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    im.save(png)
    meta = cam.meta()
    meta.update(extra)
    # THE SHEET IS OFFSET BY THE HEADER; the sidecar buffers are NOT.  A mark at
    # sheet pixel (x, y) is buffer pixel (x, y - header_px).  Getting this wrong
    # slides every mark 124 mm/px worth of header up the part.
    meta["header_px"] = HDR
    meta["ids"] = key
    meta["legend"] = {"red": "may be REMOVED", "green": "may be ADDED",
                      "unmarked": "leave alone"}
    json.dump(meta, open(png[:-4] + ".json", "w"), indent=1)
    np.savez_compressed(png[:-4] + ".npz", depth=z.astype(np.float32), ids=ids)
    print(f"  wrote {os.path.basename(png)}  {im.size[0]}x{im.size[1]}  "
          f"{cam.s:.3f} px/mm", flush=True)
    return im


# ----------------------------------------------------------------- assembly

def assembly_sheets(W, H, dev_scope, dev_ctx):
    step = os.path.join(paths.EXPORTS, POSE + ".STEP")
    print(f"tessellating {POSE} ...", flush=True)
    scope, ctx, key, xf = [], [], {}, {}
    allV = []
    nxt = 1
    for name, shp, loc in leaves(step):
        moved = shp.moved(loc)
        if name in SCOPE:
            col, label = SCOPE[name]
            V, T, _ = tessellate(moved, dev_scope)
            if not len(T):
                continue
            pid = nxt; nxt += 1
            scope.append((V, T, col, pid))
            key[str(pid)] = name
            t = loc.wrapped.Transformation()
            xf[str(pid)] = [[t.Value(r, c) for c in range(1, 5)] for r in range(1, 4)] \
                           + [[0, 0, 0, 1]]
            allV.append(V)
        else:
            V, T, _ = tessellate(moved, dev_ctx)
            if len(T):
                ctx.append((V, T, CONTEXT, 1))
                allV.append(V)
    allV = np.vstack(allV)
    print(f"  {len(scope)} markable, {len(ctx)} context, "
          f"{sum(len(t) for _, t, _, _ in scope + ctx)} tris", flush=True)

    legend = []
    seen = set()
    for _, _, col, pid in scope:
        nm = key[str(pid)]
        if nm not in seen:
            seen.add(nm)
            legend.append((col, SCOPE[nm][1]))

    out = os.path.join(MARKS, "assembly")
    sheets = []
    for vname, fwd, up, blurb in ASM_VIEWS:
        print(f"{vname} ...", flush=True)
        vw, vh = fit_canvas(fwd, up, allV) if W is None else (W, H)
        cam = Ortho(fwd, up, vw, vh, allV)
        s = raster(scope, cam)
        c = raster(ctx, cam)
        img, z, ids = compose(s, c, cam)
        im = write_sheet(
            os.path.join(out, f"asm_{vname}.png"), cam, img, z, ids, key,
            f"ASSEMBLY  -  {vname}", blurb, legend,
            f"across: {_axname(cam.right)} -->", f"up: {_axname(cam.up)}",
            {"kind": "assembly", "pose": POSE, "view": vname,
             "frame": "assembly global (hip at origin, +Z outboard)",
             "axes": {"across": _axname(cam.right), "up": _axname(cam.up),
                      "towards_camera": _axname(cam.fwd)},
             "part_transform_world_from_local": xf})
        sheets.append((vname, im))
    return sheets


# ----------------------------------------------------------------- per part

def part_sheets(W, H, dev):
    out = os.path.join(MARKS, "parts")
    sheets = []
    for f in sorted(os.listdir(paths.SPECS)):
        if not f.endswith(".json"):
            continue
        part = f[:-5]
        part = {"side panel": "Side panel", "robotmount": "RobotMount"}.get(
            part, part.capitalize())
        sp = json.load(open(os.path.join(paths.SPECS, f)))
        show = sp.get("show_face", "+Z")
        try:
            src = import_step(paths.part_step(part))
        except FileNotFoundError as e:
            print(f"  {part}: {e}"); continue
        print(f"{part}  (show face {show}) ...", flush=True)
        V, T, _ = tessellate(src, dev)
        zf = 1.0 if show == "+Z" else -1.0
        views = {
            "show":  ((0, 0, zf), (0, 1, 0)),
            "back":  ((0, 0, -zf), (0, 1, 0)),
            "end_X": ((1, 0, 0), (0, 0, 1)),
            "end_Y": ((0, 1, 0), (0, 0, 1)),
        }
        for vname, blurb in PART_VIEWS:
            fwd, up = views[vname]
            vw, vh = fit_canvas(fwd, up, V) if W is None else (W, H)
            cam = Ortho(fwd, up, vw, vh, V)
            img, z, ids = raster([(V, T, (150, 155, 163), 1)], cam)
            img = np.where((ids > 0)[..., None], img, np.array(BG, np.float32))
            im = write_sheet(
                os.path.join(out, f"{part.replace(' ', '_')}_{vname}.png"),
                cam, img, z, ids, {"1": part},
                f"{part}  -  {vname.replace('_', ' ')}", blurb, [],
                f"across: {_axname(cam.right)} -->", f"up: {_axname(cam.up)}",
                {"kind": "part", "part": part, "view": vname, "show_face": show,
                 "frame": "part local (the frame the recipe builds in)",
                 "axes": {"across": _axname(cam.right), "up": _axname(cam.up),
                          "towards_camera": _axname(cam.fwd)},
                 "camera_fwd_local": list(fwd)})
            sheets.append((f"{part} {vname}", im))
    return sheets


def index(sheets, png, header):
    cols = 3
    tw = 620
    tiles = [(nm, im.resize((tw, int(im.size[1] * tw / im.size[0])))) for nm, im in sheets]
    th = max(im.size[1] for _, im in tiles)
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * tw, 72 + rows * (th + 30)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 18), header, font=_font(30, True), fill=(24, 26, 30))
    for i, (nm, im) in enumerate(tiles):
        x, y = (i % cols) * tw, 72 + (i // cols) * (th + 30)
        img.paste(im, (x, y))
        d.text((x + 10, y + th + 4), nm, font=_font(18, True), fill=(60, 66, 76))
    img.save(png)
    print(f"wrote {png}  ({W}x{H})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", action="store_true")
    ap.add_argument("--parts", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    both = not (a.asm or a.parts)
    W, H = (620, 500) if a.quick else (None, None)   # None -> fit_canvas
    dev_s, dev_c = (0.8, 2.0) if a.quick else (0.22, 0.7)
    os.makedirs(MARKS, exist_ok=True)
    if a.asm or both:
        s = assembly_sheets(W, H, dev_s, dev_c)
        index(s, os.path.join(MARKS, "INDEX_assembly.png"),
              "Assembly markup sheets  -  RED = may remove, GREEN = may add")
    if a.parts or both:
        s = part_sheets(W, H, dev_s)
        index(s, os.path.join(MARKS, "INDEX_parts.png"),
              "Per-part markup sheets  -  RED = may remove, GREEN = may add")
