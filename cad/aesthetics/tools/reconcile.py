r"""His suggestion against the measurement.  Decision 26, closing the loop.

    C:/Users/ferna/cadenv/Scripts/python.exe tools/reconcile.py [Part ...]

He said the marks are ideas and the collision check is the judge.  So this puts
the two on the same grid and reports, per link:

    TAKE      he marked it AND it is free          -> use as drawn
    PULL BACK he marked it but something is there  -> his idea, geometry says no
    OFFER     free but he did not mark it          -> room he could not see

`out/marks/read/<Part>_RECONCILE.png` shows all three, and the numbers say how
much of his intent survives contact with the assembly.

Neither input is authoritative on its own.  The marks are a hand drawing on a
screenshot, so their edges are worth a millimetre or two at best.  The free map
is a voxel measurement whose clearance and grow-window are parameters.  What the
pair is good for is deciding WHERE to grow and ROUGHLY HOW FAR -- the real gate
is still lib/collide.py on the finished solids.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
import shapely
from shapely import wkt as shwkt
import paths
import freemap as FM
import marksheet as MS

LINKS = ["Femur", "Tibia", "Coupler"]


def raster(geom, x0, y0, res, ny, nx):
    """Boolean mask of a shapely geometry on the free-map grid."""
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    return shapely.contains_xy(geom, X.ravel(), Y.ravel()).reshape(ny, nx)


def run(part):
    fz = os.path.join(FM.OUTDIR, f"{part}.npz")
    mk = os.path.join(paths.INPUT, "marks", f"{part}.json")
    if not os.path.exists(fz):
        print(f"{part}: no free map -- run tools/freemap.py first"); return None
    d = np.load(fz)
    free, reach, sect = d["free"], d["reach"], d["sect"]
    x0, y0, _ = d["origin"]
    res = float(d["res"][0])
    ny, nx = free.shape
    cell = res * res

    add = None
    if os.path.exists(mk):
        doc = json.load(open(mk))
        if doc.get("add_wkt"):
            add = raster(shwkt.loads(doc["add_wkt"]), x0, y0, res, ny, nx)
    if add is None:
        print(f"{part}: no ADD marks"); return None

    add_out = add & (~sect)                 # his mark outside the existing metal
    take = add_out & free
    pull = add_out & (~free)
    offer = free & (~add)

    r = reach[take]
    out = {
        "part": part,
        "marked_outside_silhouette_mm2": round(float(add_out.sum()) * cell, 1),
        "take_mm2": round(float(take.sum()) * cell, 1),
        "pull_back_mm2": round(float(pull.sum()) * cell, 1),
        "offer_mm2": round(float(offer.sum()) * cell, 1),
        "take_fraction": round(float(take.sum()) / max(add_out.sum(), 1), 3),
        "reach_under_his_mark_p50_mm": round(float(np.percentile(r, 50)), 2) if r.size else 0.0,
        "reach_under_his_mark_p90_mm": round(float(np.percentile(r, 90)), 2) if r.size else 0.0,
    }

    up = max(1, int(round(4.0 * res)))
    img = np.full((ny, nx, 3), 250, np.float32)
    img[sect] = (150, 155, 163)
    img[offer] = (170, 186, 214)            # available, unclaimed
    img[take] = (34, 170, 78)               # his idea, and it fits
    img[pull] = (214, 44, 44)               # his idea, and it does not
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    im = im.transpose(Image.FLIP_TOP_BOTTOM).resize((nx * up, ny * up), Image.NEAREST)

    hdr, ftr = 150, 44
    o = Image.new("RGB", (im.size[0], im.size[1] + hdr + ftr), (255, 255, 255))
    o.paste(im, (0, hdr))
    dr = ImageDraw.Draw(o)
    dr.rectangle([0, 0, o.size[0], hdr - 1], fill=(28, 31, 36))
    dr.text((20, 14), f"{part}  -  YOUR MARKS vs THE MEASUREMENT",
            font=MS._font(30, True), fill=(242, 244, 247))
    dr.text((20, 56), f"GREEN  take: you marked it and it is free          "
            f"{out['take_mm2']:.0f} mm2  ({out['take_fraction']:.0%} of your mark)",
            font=MS._font(18), fill=(120, 226, 160))
    dr.text((20, 82), f"RED    pull back: you marked it, something is there  "
            f"{out['pull_back_mm2']:.0f} mm2", font=MS._font(18), fill=(255, 150, 150))
    dr.text((20, 108), f"PALE   offer: free, but you did not mark it        "
            f"{out['offer_mm2']:.0f} mm2", font=MS._font(18), fill=(180, 198, 226))
    bar = 50.0 / res * up
    by = o.size[1] - 30
    for k in range(5):
        dr.rectangle([20 + bar * k / 5, by, 20 + bar * (k + 1) / 5, by + 10],
                     fill=(24, 26, 30) if k % 2 == 0 else (255, 255, 255),
                     outline=(24, 26, 30))
    dr.text((28 + bar, by - 4), "50 mm", font=MS._font(17, True), fill=(24, 26, 30))
    p = os.path.join(paths.OUT, "marks", "read", f"{part}_RECONCILE.png")
    o.save(p)
    print(f"  wrote {os.path.basename(p)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=LINKS)
    a = ap.parse_args()
    rows = [r for r in (run(p) for p in (a.parts or LINKS)) if r]
    print(f"\n{'part':<9} {'his mark':>10} {'TAKE':>10} {'PULL BACK':>10} "
          f"{'OFFER':>10}   reach under his mark")
    for r in rows:
        print(f"{r['part']:<9} {r['marked_outside_silhouette_mm2']:9.0f}  "
              f"{r['take_mm2']:9.0f}  {r['pull_back_mm2']:9.0f}  {r['offer_mm2']:9.0f}   "
              f"p50 {r['reach_under_his_mark_p50_mm']:5.2f}  "
              f"p90 {r['reach_under_his_mark_p90_mm']:5.2f} mm   "
              f"({r['take_fraction']:.0%} of his mark usable)")
    out = os.path.join(paths.INPUT, "marks", "reconcile.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
