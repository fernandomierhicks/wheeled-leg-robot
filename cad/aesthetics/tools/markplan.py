r"""Phase 2, step 4 -- his marks as PART-LOCAL POLYGONS that plan() can consume.

    C:/Users/ferna/cadenv/Scripts/python.exe tools/markplan.py

`readmarks.py` answers "which part, roughly where" and is the confirmation
artifact.  This answers the question the recipes actually ask: *what region*.

WHY A BOUNDING BOX WILL NOT DO.  He drew a BAND FOLLOWING THE PERIMETER of each
link.  Its bounding box is the whole part -- the Tibia's halo wraps it
completely -- so handing plan() a bbox would authorise growth everywhere, the
exact inverse of what he drew.  Marks have to survive as regions.

WHY THE DEPTH BUFFER IS NOT USED HERE.  For an ADD mark most of the meaningful
area lies OUTSIDE the silhouette, in the background, where there is no depth
because there is no surface -- and that is precisely where the new material
goes.  `readmarks.zones()` keeps only on-part pixels and would throw that half
away.  But an axis-aligned PLAN view needs no depth for its in-plane
coordinates: the camera's `right` and `up` are the global X and Y axes, so a
pixel gives global XY directly, off the part as readily as on it.  Off-part pen
is credited to the NEAREST part within `CAP_MM`, so a band drawn just outside
the tibia belongs to the tibia and not to whatever else is in frame.

THE Z-ROTATION ASSUMPTION IS ASSERTED, NOT ASSUMED.  Dropping depth is only
valid if the part's placement maps local Z onto global Z, i.e. the rotation is
about Z alone.  That holds for all three links, and `plan_basis` raises if it
ever stops holding rather than quietly returning wrong millimetres.

NO buffer(0) ANYWHERE IN THIS FILE.  That call is what made a self-touching ring
"valid" by splitting it into two lobes touching at a point, which extruded into
the non-manifold solid that shattered in SolidWorks and cost Phase 1.  Validity
here is fixed with shapely's make_valid().
"""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon, MultiLineString
from shapely.ops import polygonize, unary_union
from shapely import make_valid
import paths
import readmarks as RM

LINKS = ["Femur", "Tibia", "Coupler"]        # decision 25: links only
ALIAS = {"Femur_inside_InsideBox": "Femur"}  # the mirrored left femur inherits
CAP_MM = 12.0        # how far off the metal a mark may sit and still be its mark
DOWN = 2             # px block before polygonising; ~0.4 mm at these scales
SIMPLIFY_MM = 0.4
MIN_AREA_MM2 = 4.0

OUT = os.path.join(paths.INPUT, "marks")


def plan_basis(meta, pid):
    """Sheet (col,row) -> that part's LOCAL (x, y), for an axis-aligned plan view."""
    right = np.asarray(meta["right"], float)
    up = np.asarray(meta["up"], float)
    fwd = np.asarray(meta["fwd"], float)
    if abs(abs(fwd[2]) - 1) > 1e-9:
        raise SystemExit(f"{meta['view']} is not a plan view down Z; "
                         f"markplan only handles outboard/inboard sheets")
    T = np.asarray(meta["part_transform_world_from_local"][str(pid)], float)
    R, t = T[:3, :3], T[:3, 3]
    off = max(abs(R[0, 2]), abs(R[1, 2]), abs(R[2, 0]), abs(R[2, 1]))
    if off > 1e-6 or abs(abs(R[2, 2]) - 1) > 1e-6:
        raise SystemExit(
            f"part id {pid} is NOT placed by a rotation about Z "
            f"(off-axis term {off:.2e}, R22 {R[2,2]:.6f}).  Local XY would then "
            f"depend on depth, and this file drops depth on purpose -- see the "
            f"module docstring.  Refusing to emit millimetres that are wrong.")
    R2, t2 = R[:2, :2], t[:2]
    R2i = np.linalg.inv(R2)
    s, W, H = meta["px_per_mm"], meta["W"], meta["H"]
    cx, cy = meta["centre_mm"]

    def to_local(px, py):
        r = (np.asarray(px, float) - W / 2) / s + cx
        u = (H / 2 - np.asarray(py, float)) / s + cy
        world = np.stack([r * right[0] + u * up[0], r * right[1] + u * up[1]], 1)
        return (world - t2) @ R2i.T

    return to_local


def mask_polygons(mask, to_local):
    """A pixel mask -> shapely polygons in local mm.

    Built from BOUNDARY EDGES: every cell edge with mask on one side and not the
    other, assembled by shapely.polygonize.  Edges interior to the mask cancel,
    so the result is the region's true outline including any holes -- which a
    convex hull or a bbox would both destroy, and a perimeter band is nothing
    but outline.
    """
    m = ndimage.binary_closing(mask, np.ones((3, 3), bool), iterations=2)
    m = ndimage.binary_fill_holes(m)
    if DOWN > 1:
        h, w = m.shape
        h, w = h // DOWN * DOWN, w // DOWN * DOWN
        m = m[:h, :w].reshape(h // DOWN, DOWN, w // DOWN, DOWN).max((1, 3))
    p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
    p[1:-1, 1:-1] = m
    segs = []
    ys, xs = np.nonzero(p[:, 1:] != p[:, :-1])          # vertical edges at x=xs+1
    segs += [((x + 1, y), (x + 1, y + 1)) for y, x in zip(ys.tolist(), xs.tolist())]
    ys, xs = np.nonzero(p[1:, :] != p[:-1, :])          # horizontal edges at y=ys+1
    segs += [((x, y + 1), (x + 1, y + 1)) for y, x in zip(ys.tolist(), xs.tolist())]
    if not segs:
        return []
    out = []
    for poly in polygonize(MultiLineString(segs)):
        if not poly.is_valid:
            poly = make_valid(poly)                     # never buffer(0)
        for g in (poly.geoms if hasattr(poly, "geoms") else [poly]):
            if not isinstance(g, Polygon) or g.is_empty:
                continue
            # grid -> padded-sheet px -> sheet px -> local mm
            def conv(seq):
                a = np.asarray(seq, float)
                px = (a[:, 0] - 1) * DOWN
                py = (a[:, 1] - 1) * DOWN
                return to_local(px, py)
            shell = conv(g.exterior.coords)
            holes = [conv(r.coords) for r in g.interiors]
            q = Polygon(shell, [h for h in holes if len(h) >= 4])
            if not q.is_valid:
                q = make_valid(q)
            for r in (q.geoms if hasattr(q, "geoms") else [q]):
                if isinstance(r, Polygon) and r.area >= MIN_AREA_MM2:
                    out.append(r.simplify(SIMPLIFY_MM))
    return out


def gather(marked_png, colour):
    """[(part, [polygons])] for one marked sheet and one pen colour."""
    base = RM.sidecar(marked_png)
    meta = json.load(open(base + ".json"))
    buf = np.load(base + ".npz")
    ids = buf["ids"]
    s, hdr = meta["px_per_mm"], meta["header_px"]
    a = np.asarray(Image.open(marked_png).convert("RGB"))[hdr:hdr + meta["H"]]
    red, grn = RM.pen_masks(a)
    pen = red if colour == "remove" else grn
    if pen.sum() < 200:
        return {}, meta
    # nearest part for every background pixel, capped
    dist, (iy, ix) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    near = np.where(dist / s <= CAP_MM, ids[iy, ix], 0)
    owner = np.where(ids > 0, ids, near)

    name2id = {v: int(k) for k, v in meta["ids"].items()}
    got = {}
    for name, pid in name2id.items():
        part = ALIAS.get(name, name)
        if part not in LINKS:
            continue
        m = pen & (owner == pid)
        if m.sum() < 200:
            continue
        polys = mask_polygons(m, plan_basis(meta, pid))
        if polys:
            got.setdefault(part, []).extend(polys)
    return got, meta


def echo_sheets():
    """Draw the extracted polygons back onto each link's own show sheet.

    This is the artifact he confirms.  The bbox echo from readmarks.py cannot
    show a perimeter band -- it draws a box round the whole part -- so this
    renders the REGION, filled, with its area, on the part alone and at known
    scale.
    """
    from shapely import wkt as shwkt
    made = []
    for part in LINKS:
        j = os.path.join(OUT, f"{part}.json")
        sheet = os.path.join(paths.OUT, "marks", "parts",
                             f"{part.replace(' ', '_')}_show")
        if not (os.path.exists(j) and os.path.exists(sheet + ".png")):
            continue
        doc = json.load(open(j))
        meta = json.load(open(sheet + ".json"))
        right = np.asarray(meta["right"], float)
        up = np.asarray(meta["up"], float)
        s, W, H = meta["px_per_mm"], meta["W"], meta["H"]
        cx, cy = meta["centre_mm"]
        hdr = meta["header_px"]

        def to_px(xy):
            a = np.asarray(xy, float)
            r = a[:, 0] * right[0] + a[:, 1] * right[1]
            u = a[:, 0] * up[0] + a[:, 1] * up[1]
            return [( (rr - cx) * s + W / 2, H / 2 - (uu - cy) * s + hdr )
                    for rr, uu in zip(r, u)]

        im = Image.open(sheet + ".png").convert("RGB")
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for key, fill, line in (("add_wkt", (28, 168, 72, 90), (18, 140, 58)),
                                ("remove_wkt", (214, 38, 38, 90), (180, 26, 26))):
            if not doc.get(key):
                continue
            g = shwkt.loads(doc[key])
            for poly in (g.geoms if hasattr(g, "geoms") else [g]):
                d.polygon(to_px(poly.exterior.coords), fill=fill, outline=line, width=3)
                for r in poly.interiors:
                    d.polygon(to_px(r.coords), fill=(250, 250, 251, 235), outline=line, width=2)
        im = Image.alpha_composite(im.convert("RGBA"), lay).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([12, hdr + 8, 470, hdr + 62], fill=(255, 255, 255), outline=(60, 66, 76))
        d.text((22, hdr + 14), f"GREEN add {doc['add_area_mm2']:.0f} mm2   "
               f"(grow through full local section)", fill=(18, 120, 52), font=RM._font(17, True))
        d.text((22, hdr + 38), f"RED remove {doc['remove_area_mm2']:.0f} mm2   "
               f"(pocket; through only in small areas)", fill=(170, 26, 26), font=RM._font(17, True))
        p = os.path.join(paths.OUT, "marks", "read", f"{part}_PLAN_ECHO.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        im.save(p)
        made.append(p)
        print(f"  wrote {os.path.basename(p)}")
    return made


if __name__ == "__main__":
    hf = os.path.join(paths.INPUT, "human feedback")
    jobs = []
    for f in sorted(glob.glob(os.path.join(hf, "*.png"))):
        stem = os.path.basename(f).lower()
        if "remove" in stem:
            jobs.append((f, "remove"))
        elif "add" in stem:
            jobs.append((f, "add"))
    if not jobs:
        sys.exit(f"nothing to read in {hf}")

    acc = {p: {"add": [], "remove": []} for p in LINKS}
    used = {p: {"add": [], "remove": []} for p in LINKS}
    for png, colour in jobs:
        try:
            got, meta = gather(png, colour)
        except SystemExit as e:
            print(f"  {os.path.basename(png)} [{colour}]: skipped -- {e}")
            continue
        if not got:
            print(f"  {os.path.basename(png)} [{colour}]: no plan-view marks "
                  f"on any link (view {meta['view']})")
            continue
        for part, polys in got.items():
            acc[part][colour] += polys
            used[part][colour].append(f"{os.path.basename(png)} -> {meta['view']}")
            print(f"  {os.path.basename(png)} [{colour}] {part}: {len(polys)} polygon(s), "
                  f"{sum(p.area for p in polys):8.1f} mm2")

    os.makedirs(OUT, exist_ok=True)
    print("\n--- written ---")
    for part in LINKS:
        add = unary_union(acc[part]["add"]) if acc[part]["add"] else None
        rem = unary_union(acc[part]["remove"]) if acc[part]["remove"] else None
        if add is None and rem is None:
            continue
        doc = {
            "part": part,
            "confirmed": False,
            "frame": "part local XY in mm; local Z is the lateral axis",
            "legend": {"add": "material MAY BE ADDED", "remove": "material MAY BE REMOVED"},
            "grow_depth": "full_local_section",   # decision 25
            "remove_depth": "pocket by default; through-cut only in small areas",
            "min_wall_mm": 3.0,
            "sources": used[part],
            "add_area_mm2": round(add.area, 1) if add else 0.0,
            "remove_area_mm2": round(rem.area, 1) if rem else 0.0,
            "add_wkt": add.wkt if add else None,
            "remove_wkt": rem.wkt if rem else None,
        }
        p = os.path.join(OUT, f"{part}.json")
        json.dump(doc, open(p, "w"), indent=1)
        print(f"  {part:<9} add {doc['add_area_mm2']:8.1f} mm2   "
              f"remove {doc['remove_area_mm2']:8.1f} mm2   -> {os.path.basename(p)}")
    print("\n--- echo sheets ---")
    echo_sheets()
    print("\nconfirmed=false on every one.  NOTHING IS USED UNTIL HE CONFIRMS.")
