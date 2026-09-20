r"""Phase 2, step 3 -- read Fernando's marks back and ECHO WHAT I UNDERSTOOD.

    C:/Users/ferna/cadenv/Scripts/python.exe tools/readmarks.py <marked.png> [...]
    C:/Users/ferna/cadenv/Scripts/python.exe tools/readmarks.py --all
    C:/Users/ferna/cadenv/Scripts/python.exe tools/readmarks.py --selftest

A marked sheet goes in; two things come out:

    out/marks/read/<sheet>_ECHO.png   what I think he said, drawn back over it
    input/marks/<Part>.json           the same thing in PART-LOCAL millimetres

NOTHING IS USED UNTIL HE CONFIRMS THE ECHO.  Reading a mark 10 mm off and then
silently styling to it is the exact class of error this phase exists to kill,
and it would be invisible -- the part would just come out wrong, months later,
in plastic.  So every zone is reported with its part, its local bounding box and
its extent in mm, and he gets to say "no, that one is the other rib".

HOW A PIXEL BECOMES A MILLIMETRE.  tools/marksheet.py wrote, beside every png:

    .json   camera basis (right/up/fwd), px_per_mm, centre, header offset, and
            world_from_local for every part instance in the view
    .npz    per-pixel part id + per-pixel depth along the view direction

so   sheet pixel -> buffer pixel (minus the header)
                -> part id           (who he marked)
                -> depth             (where the surface actually is)
                -> world point       (exact, not inferred)
                -> local point       (via inverse world_from_local)

The depth buffer is what makes this exact rather than a guess: without it a
pixel is a ray and the answer is "somewhere along here".

THRESHOLDS.  The sheets are deliberately drawn in blues, violets, teals, ambers
and greys -- no red, no green anywhere in the palette -- so a pen is separated
by hue alone and no tuning is needed per sheet.  The header band is excluded by
geometry, which is what keeps the printed legend swatches from being read as
two enormous marks.
"""
import os, re, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import paths

MARKS = os.path.join(paths.OUT, "marks")
ECHO = os.path.join(MARKS, "read")
OUTJSON = os.path.join(paths.INPUT, "marks")

MIN_PX = 60          # a blob smaller than this is a slip of the pen, not a zone
SNAP_PX = 14         # how far a mark may miss the metal and still be snapped on


def _font(sz, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", n), sz)
        except Exception:
            pass
    return ImageFont.load_default()


def pen_masks(rgb):
    """Strongly red and strongly green pixels, by hue separation from the sheet."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    red = (r > 105) & (r - np.maximum(g, b) > 42)
    grn = (g > 85) & (g - np.maximum(r, b) > 36)
    return red, grn


def _inv(T):
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    Ri = np.linalg.inv(R)
    return Ri, -Ri @ t


def zones(mask, ids, depth, meta, cam_basis, colour):
    """Connected blobs -> one dict per zone, in the marked part's own frame."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    right, up, fwd = cam_basis
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < MIN_PX:
            continue
        pid = ids[ys, xs]
        hit = pid > 0
        snapped = 0
        if hit.sum() < 0.25 * len(ys):
            # the mark mostly missed the metal -- dilate the blob and retry, so
            # a ring drawn AROUND a feature still resolves to that feature
            d = ndimage.binary_dilation(lab == k, iterations=SNAP_PX)
            ys, xs = np.nonzero(d & (ids > 0))
            if len(ys) < MIN_PX // 2:
                out.append({"colour": colour, "part": None, "px": len(ys),
                            "note": "MARK DOES NOT LAND ON ANY PART -- ambiguous"})
                continue
            pid, hit, snapped = ids[ys, xs], np.ones(len(ys), bool), 1
        ys, xs, pid = ys[hit], xs[hit], pid[hit]
        vals, cnt = np.unique(pid, return_counts=True)
        # ONE ZONE PER PART, not a majority vote.  A stroke drawn along the edge
        # two parts share means BOTH of them -- and "grow this edge" is exactly
        # the mark most likely to straddle.  Voting would silently throw half of
        # it away and report a confident answer about the wrong part.
        for winner, n_px in zip(vals.tolist(), cnt.tolist()):
            if n_px < MIN_PX // 2:
                continue
            share = float(n_px / cnt.sum())
            m = pid == winner
            yy, xx = ys[m], xs[m]

            name = meta["ids"][str(winner)]
            s, W, H = meta["px_per_mm"], meta["W"], meta["H"]
            cx, cy = np.asarray(meta["centre_mm"], float)
            r = (xx - W / 2) / s + cx
            u = (H / 2 - yy) / s + cy
            f = depth[yy, xx]
            world = (np.asarray(right) * r[:, None] + np.asarray(up) * u[:, None]
                     + np.asarray(fwd) * f[:, None])
            xf = meta.get("part_transform_world_from_local", {}).get(str(winner))
            if xf is None:
                local = world                  # a per-part sheet is already local
            else:
                Ri, ti = _inv(xf)
                local = world @ Ri.T + ti
            out.append({
                "colour": colour, "part": name, "view": meta["view"],
                "blob": k, "px": int(len(xx)),
                "area_mm2": round(len(xx) / (s * s), 1),
                "share_of_stroke": round(share, 2), "snapped": bool(snapped),
                "local_bbox_mm": [[round(float(v), 2) for v in local.min(0)],
                                  [round(float(v), 2) for v in local.max(0)]],
                "local_centroid_mm": [round(float(v), 2) for v in local.mean(0)],
                "sheet_px_bbox": [int(xx.min()), int(yy.min()),
                                  int(xx.max()), int(yy.max())],
            })
    return out


def sidecar(png):
    """Find the sheet this marked png came from.

    He will not name the file back exactly -- it will be `asm_outboard_marked`,
    or `asm_outboard (1)`, or `asm_outboard copy`, depending on what he marked it
    in. So trailing junk is peeled off one segment at a time until the sidecars
    turn up, rather than demanding a filename.
    """
    stem = png[:-4]
    tries, s = [stem], stem
    for _ in range(4):
        # peel one trailing segment: "_marked", " (1)", " copy", anything.  The
        # FULL stem is tried first, so this can only ever fall back, never shadow.
        s2 = re.sub(r"[ _\-]+[^ _\-]+$|\s*\(\d+\)$", "", s)
        if s2 == s or not s2:
            break
        s, _ = s2, tries.append(s2)
    for base in tries:
        if os.path.exists(base + ".json") and os.path.exists(base + ".npz"):
            return base
    return match_sheet(png)


def match_sheet(png, report=True):
    """Identify the sheet BY CONTENT when the filename cannot say.

    He named his files `add.png` and `remove.png`, one colour per file, which no
    amount of name-peeling will resolve.  But a marked sheet is still the sheet
    everywhere he did not draw, so: among the sheets of exactly matching
    dimensions, take the one that agrees on the most non-pen pixels.  The
    agreement fraction is printed, because a confident match onto the WRONG
    sheet would put every mark on the wrong part with no outward sign.
    """
    im = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    red, grn = pen_masks(im)
    pen = red | grn
    best = []
    for p in sorted(glob.glob(os.path.join(MARKS, "**", "*.png"), recursive=True)):
        if "INDEX" in p or not (os.path.exists(p[:-4] + ".json")
                                and os.path.exists(p[:-4] + ".npz")):
            continue
        s = np.asarray(Image.open(p).convert("RGB")).astype(np.int16)
        if s.shape != im.shape:
            continue
        sred, sgrn = pen_masks(s)
        free = ~(pen | sred | sgrn)
        if free.sum() < 0.2 * free.size:
            continue
        agree = float((np.abs(im - s).max(2)[free] < 12).mean())
        best.append((agree, p))
    if not best:
        raise SystemExit(
            f"cannot identify which sheet {os.path.basename(png)} came from: no "
            f"sheet in out/marks has its exact dimensions {im.shape[1]}x{im.shape[0]}. "
            f"Was it resized, cropped or re-exported?")
    best.sort(reverse=True)
    agree, p = best[0]
    if agree < 0.80:
        raise SystemExit(
            f"cannot identify {os.path.basename(png)}: closest sheet is "
            f"{os.path.basename(p)} at only {agree:.1%} agreement on unmarked "
            f"pixels. Refusing to guess -- every mark would land on the wrong part.")
    if report:
        runner = f"   (next best {os.path.basename(best[1][1])} {best[1][0]:.1%})" \
                 if len(best) > 1 else ""
        print(f"  {os.path.basename(png)} -> {os.path.basename(p)} "
              f"by content, {agree:.1%} agreement{runner}")
    return p[:-4]


def read_sheet(png):
    base = sidecar(png)
    meta = json.load(open(base + ".json"))
    buf = np.load(base + ".npz")
    ids, depth = buf["ids"], buf["depth"]
    hdr, H, W = meta["header_px"], meta["H"], meta["W"]

    im = Image.open(png).convert("RGB")
    a = np.asarray(im)
    if a.shape[1] != W:
        raise SystemExit(f"{png} is {a.shape[1]} px wide, the sheet was {W}. "
                         f"It has been resized or cropped, so no mark can be located. "
                         f"Mark the png as written, at 100%.")
    body = a[hdr:hdr + H]
    red, grn = pen_masks(body)
    basis = (meta["right"], meta["up"], meta["fwd"])
    zs = (zones(red, ids, depth, meta, basis, "red") +
          zones(grn, ids, depth, meta, basis, "green"))
    return meta, im, hdr, zs


def echo(png, meta, im, hdr, zs):
    """Draw my reading back over his sheet, so a misread is obvious at a glance."""
    out = im.copy()
    d = ImageDraw.Draw(out)
    f, fb = _font(16), _font(17, True)
    for i, z in enumerate(zs, 1):
        if z.get("part") is None:
            continue
        x0, y0, x1, y1 = z["sheet_px_bbox"]
        col = (214, 38, 38) if z["colour"] == "red" else (28, 168, 72)
        d.rectangle([x0 - 3, y0 + hdr - 3, x1 + 3, y1 + hdr + 3], outline=col, width=3)
        act = "REMOVE" if z["colour"] == "red" else "ADD"
        lo, hi = z["local_bbox_mm"]
        d.rectangle([x0 - 3, y0 + hdr - 25, x0 + 250, y0 + hdr - 4], fill=col)
        d.text((x0 + 2, y0 + hdr - 24), f"{i}. {act}  {z['part']}", font=fb,
               fill=(255, 255, 255))
        d.text((x0 + 2, y0 + hdr + (y1 - y0) + 6),
               f"local x {lo[0]:.0f}..{hi[0]:.0f}  y {lo[1]:.0f}..{hi[1]:.0f}  "
               f"z {lo[2]:.0f}..{hi[2]:.0f} mm", font=f, fill=(20, 22, 26))
    os.makedirs(ECHO, exist_ok=True)
    p = os.path.join(ECHO, os.path.basename(png)[:-4] + "_ECHO.png")
    out.save(p)
    return p


def persist(all_zones):
    """input/marks/<Part>.json -- his map, in the frame the recipes build in.
    plan() consumes this; where he said nothing, the derived keep-out still rules."""
    os.makedirs(OUTJSON, exist_ok=True)
    by = {}
    for z in all_zones:
        if z.get("part"):
            by.setdefault(z["part"], []).append(z)
    for part, zs in by.items():
        p = os.path.join(OUTJSON, f"{part}.json")
        json.dump({"part": part, "confirmed": False,
                   "legend": {"red": "may be REMOVED", "green": "may be ADDED"},
                   "zones": zs}, open(p, "w"), indent=1)
        print(f"  wrote {p}  ({len(zs)} zones, confirmed=false)")
    return by


def selftest():
    """Round-trip the geometry with marks I draw myself, so the mapping is proven
    BEFORE he spends an evening marking sheets.  Paints a blob at a known pixel,
    reads it back, and checks the reported local point against the same pixel
    resolved directly through the buffers."""
    R = 9
    sheets = sorted(glob.glob(os.path.join(MARKS, "assembly", "*.png")))
    if not sheets:
        raise SystemExit("no sheets yet -- run tools/marksheet.py first")
    ok = True
    for src in sheets:
        meta = json.load(open(src[:-4] + ".json"))
        buf = np.load(src[:-4] + ".npz")
        ids, depth = buf["ids"], buf["depth"]
        hdr = meta["header_px"]

        # Paint each blob WHOLLY INSIDE one part, by eroding that part's mask by
        # the blob radius.  A blob straddling two parts is legitimately reported
        # as two zones now, and then "which zone is this mark" has no single
        # answer -- the first version of this test paired zones by sort order and
        # failed itself, not the tool.
        im = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(im)
        picks = []
        for pid, col in zip(sorted(np.unique(ids[ids > 0]).tolist()),
                            [(220, 20, 20), (20, 180, 60)] * 9):
            inner = ndimage.binary_erosion(ids == pid, iterations=R + 3)
            yy, xx = np.nonzero(inner)
            if len(yy) < 1:
                continue                      # part too thin in this view to test
            k = len(yy) // 2
            cx, cy = int(xx[k]), int(yy[k])
            d.ellipse([cx - R, cy + hdr - R, cx + R, cy + hdr + R], fill=col)
            picks.append((cx, cy, "red" if col[0] > 100 else "green", int(pid)))
        if not picks:
            continue
        tmp = src[:-4] + "_selftest_marked.png"
        im.save(tmp)
        _, _, _, zs = read_sheet(tmp)
        os.remove(tmp)

        right, up, fwd = (np.asarray(meta[k], float) for k in ("right", "up", "fwd"))
        s, W, H = meta["px_per_mm"], meta["W"], meta["H"]
        cx0, cy0 = meta["centre_mm"]
        print(f"\n{os.path.basename(src)}: {len(picks)} marks painted, "
              f"{len(zs)} zones read back")
        for px, py, colour, pid in picks:
            want_part = meta["ids"][str(pid)]
            world = (right * ((px - W / 2) / s + cx0) + up * ((H / 2 - py) / s + cy0)
                     + fwd * depth[py, px])
            Ri, ti = _inv(meta["part_transform_world_from_local"][str(pid)])
            want = world @ Ri.T + ti
            # the zone for THIS mark: same colour, same part, and its box contains
            # the pixel the blob was centred on
            cand = [z for z in zs if z["colour"] == colour and z["part"] == want_part
                    and z["sheet_px_bbox"][0] <= px <= z["sheet_px_bbox"][2]
                    and z["sheet_px_bbox"][1] <= py <= z["sheet_px_bbox"][3]]
            if not cand:
                print(f"  {colour:<5} {want_part:<24} NO ZONE READ BACK for the mark "
                      f"at ({px},{py})"); ok = False
                continue
            z = cand[0]
            got = np.asarray(z["local_centroid_mm"], float)
            err = float(np.linalg.norm(want - got))
            lo = np.asarray(z["local_bbox_mm"][0], float)
            hi = np.asarray(z["local_bbox_mm"][1], float)
            # THE INVARIANT IS CONTAINMENT, NOT PROXIMITY TO THE CENTROID.  A
            # 9 px disc laid across a step reads surfaces millimetres apart in
            # depth, so its 3D centroid is legitimately nowhere near its centre
            # pixel -- the EncoderCarrier does exactly this edge-on, 11 mm out.
            # What must hold is that the pixel I marked maps INSIDE the zone that
            # came back from it.  Centroid offset stays as information.
            inside = bool(np.all(want >= lo - 0.5) and np.all(want <= hi + 0.5))
            print(f"  {colour:<5} {want_part:<24} expected "
                  f"({want[0]:8.2f},{want[1]:8.2f},{want[2]:8.2f})  centroid "
                  f"({got[0]:8.2f},{got[1]:8.2f},{got[2]:8.2f})  d {err:5.2f} mm  "
                  f"{'in zone' if inside else 'OUTSIDE ZONE'}")
            if not inside:
                print(f"    marked pixel does NOT map inside the zone read back "
                      f"from it: zone spans x {lo[0]:.1f}..{hi[0]:.1f} "
                      f"y {lo[1]:.1f}..{hi[1]:.1f} z {lo[2]:.1f}..{hi[2]:.1f}")
                ok = False
    print("\nselftest PASSED" if ok else "\nselftest FAILED")
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        sys.exit(0 if selftest() else 1)
    if "--all" in args:
        args = sorted(glob.glob(os.path.join(MARKS, "**", "*_marked.png"), recursive=True))
        if not args:
            sys.exit("no *_marked.png anywhere under out/marks")
    if not args:
        sys.exit(__doc__)
    allz = []
    for png in args:
        meta, im, hdr, zs = read_sheet(png)
        p = echo(png, meta, im, hdr, zs)
        print(f"\n{os.path.basename(png)}  ({meta['view']})  -> {os.path.basename(p)}")
        for i, z in enumerate(zs, 1):
            if z.get("part") is None:
                print(f"  {i}. {z['colour'].upper()}: {z['note']}"); continue
            lo, hi = z["local_bbox_mm"]
            print(f"  {i}. {z['colour'].upper():<5} {z['part']:<24} "
                  f"{z['area_mm2']:7.1f} mm2  local "
                  f"x {lo[0]:7.1f}..{hi[0]:7.1f}  y {lo[1]:7.1f}..{hi[1]:7.1f}  "
                  f"z {lo[2]:6.1f}..{hi[2]:6.1f}"
                  + ("   [snapped]" if z["snapped"] else "")
                  + ("" if z["share_of_stroke"] > 0.85
                     else f"   [{z['share_of_stroke']:.0%} of that stroke]"))
        allz += zs
    print("\n--- persisted ---")
    persist(allz)
    print("\nNOTHING IS USED UNTIL YOU CONFIRM THE ECHO SHEETS.")
