"""Step 26: the encoder carrier + cable clamp (Links/) in GLACIER -- flush colour inlays.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/26_style_encoder.py preview [A|B ...]

Both parts sit on the INBOARD face of the tibia, round the wheel axis (the encoder
reads the magnet on the motor shaft through the tibia): the carrier's encoder
housing, the arm the cable runs up, and the clamp over the cable.  What is seen of
them is their inboard faces (the parts' local -Y for the carrier, +Y for the
clamp; all planar), so the styling is COLOUR ONLY: flush 0.6 mm inlays on those
faces (the 07/15/22 recipe: cut, the same sketch extruded back as a new body).
Nothing is added, nothing that mates or fits changes -- no collision check needed.

Faces (part-local mm, measured from the v5 parts' STEP, 25_fasteners `scan`):
  carrier  y -5.0   the encoder housing's face, 13.1 x 14.6
           y  7.8   the arm's two strips beside the clamp + the lug rings
           y  6.0   the two side tabs of the housing
  clamp    y 10.0   its whole top (537 mm2)
Keep-outs: the screw heads 25_fasteners put there (r 2.725 + 0.5), 0.4 mm inside
every face edge and hole (gotcha 34: never over a fillet), and on the carrier
arm nothing under the clamp (its silhouette + 0.3: hidden, and its feet sit there).

The look -- the links' circuit vocabulary at this scale:
  CLAMP    a frame band round its top, a chevron field pointing down the cable,
           a three-line bus up the neck between the four screws (the cable path)
  CARRIER  the housing face as the ENCODER: a ring of code-wheel ticks round a
           pad (the magnet), an index tick; traces with doglegs and pads up the
           arm strips to the top screws; a chevron on each side tab
Palettes (preview both): A = dark carrier (the wheel hub's graphite, it sits in
front of it) + white clamp;  B = both white, like the links.
"""
import os
import sys
import json
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__))
FAST = os.path.join(HERE, "out", "fasteners")
RENDERS = os.path.join(HERE, "out", "renders")
CARRIER, CLAMP = "Tibia-1/EncoderCarrier-1", "Tibia-1/EncoderCableClamp-1"
HEAD_KEEP = 2.725 + 0.5        # M3 roundhead + 0.5 (the washers' / wheel's flush-inlay rule)
EDGE = 0.4                     # inlays stay this far inside a flat
INLAY = 0.6                    # depth
W = 0.8                        # trace width
COLOURS = {"white": (0.93, 0.94, 0.95), "graphite": (0.25, 0.27, 0.30), "blue": (0.13, 0.45, 0.95)}
PALETTES = {"A": {"carrier": "graphite", "clamp": "white"},
            "B": {"carrier": "white", "clamp": "white"}}


# ---- what is there ------------------------------------------------------------------
def faces():
    """{part: {y: shapely face outline}} + heads {part: [(x, z)]} + the clamp's
    silhouette in the carrier's frame, from 25_fasteners' scan/plan."""
    d = json.load(open(os.path.join(FAST, "encoder_faces.json")))
    out = {p: {float(y): shape(g[0]) for y, g in d[p]["faces"].items()} for p in d}
    heads = {p: [(x, z) for x, z, y in d[p]["heads"]] for p in d}
    return out, heads


def clamp_shadow():
    """The clamp's silhouette on the carrier arm (carrier x, z) + 0.3."""
    from build123d import import_step
    import importlib
    sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "lib"))
    from render3d import tessellate
    sc = json.load(open(os.path.join(FAST, "scan.json")))
    comps = {c["name"]: c for c in sc["comps"]}
    Mc, Mk = np.array(comps[CARRIER]["M"]), np.array(comps[CLAMP]["M"])
    V, T, _ = tessellate(import_step(os.path.join(FAST, "src", comps[CLAMP]["step"] + ".step")), 0.1)
    V = np.array(V) @ Mk[:, :3].T + Mk[:, 3]
    V = (V - Mc[:, 3]) @ Mc[:, :3]                      # -> carrier frame
    tris = [Polygon(V[list(t)][:, [0, 2]]) for t in T]
    return unary_union([t.buffer(1e-3) for t in tris if t.area > 1e-6]).buffer(0.3)


# ---- primitives ------------------------------------------------------------------
def trace(pts, w=W):
    return LineString(pts).buffer(w / 2, cap_style=2, join_style=2, mitre_limit=2.0)


def pad(x, z, r=0.75):
    return Point(x, z).buffer(r, quad_segs=16)


def tick(x, z, ang, length=1.6, w=0.6):
    a = math.radians(ang)
    dx, dz = math.cos(a) * length / 2, math.sin(a) * length / 2
    return LineString([(x - dx, z - dz), (x + dx, z + dz)]).buffer(w / 2, cap_style=2)


def chevron(cx, cz, w, h, t, pointing=-1):
    """A chevron pointing along x (pointing = -1: towards -x), arms `t` thick."""
    s = pointing
    tip, back = cx + s * h / 2, cx - s * h / 2
    return LineString([(back, cz + w / 2), (tip, cz), (back, cz - w / 2)]).buffer(
        t / 2, cap_style=2, join_style=2, mitre_limit=3.0)


def room(face, heads, extra=None):
    g = face.buffer(-EDGE, join_style=2)
    keep = unary_union([Point(h).buffer(HEAD_KEEP, quad_segs=24) for h in heads])
    g = g.difference(keep)
    if extra is not None:
        g = g.difference(extra)
    return g


# ---- the design --------------------------------------------------------------------
def design():
    F, heads = faces()
    out = {}
    # CLAMP top, y 10.0 (local x -48 .. -10.5 along the cable, z +-13.6)
    top = F[CLAMP][10.0]
    rm = room(top, heads[CLAMP])
    frame = top.buffer(-EDGE, join_style=2).difference(top.buffer(-EDGE - 1.2, join_style=2))
    chev = unary_union([chevron(-21.0 + k * 3.2, 0.0, 9.0, 4.0, 1.2, pointing=-1) for k in range(3)])
    bus = unary_union([trace([(-31.0, z), (-47.0, z)], 0.5) for z in (-0.85, 0.0, 0.85)])
    blue_k = unary_union([bus, trace([(-12.4, -6.2), (-12.4, 6.2)], 0.6)])
    dark_k = unary_union([frame, chev, pad(-29.4, 0.0, 1.05)])
    out[CLAMP] = {10.0: {"dark": dark_k.difference(blue_k).intersection(rm), "blue": blue_k.intersection(rm)}}

    # CARRIER housing face, y -5.0 (x +-6.55, z -7.57 .. 7.03): the encoder
    hf = F[CARRIER][-5.0]
    hr = room(hf, [])
    c0 = (0.0, -0.3)
    ring = unary_union([tick(c0[0] + 4.6 * math.cos(math.radians(a)), c0[1] + 4.6 * math.sin(math.radians(a)),
                             a, 1.5, 0.55) for a in range(0, 360, 30)])
    index = tick(c0[0], c0[1] + 6.1, 90, 1.3, 0.9)
    magnet = Point(c0).buffer(2.0, quad_segs=32)
    out[CARRIER] = {-5.0: {"light": unary_union([ring, magnet.difference(Point(c0).buffer(0.9))]).intersection(hr),
                           "blue": unary_union([index, Point(c0).buffer(0.9)]).intersection(hr)}}

    # CARRIER arm strips, y 7.8: traces up each visible margin to the top screw
    arm = F[CARRIER][7.8]
    shadow = clamp_shadow()
    ar = room(arm, heads[CARRIER], shadow)
    blue_a, light_a = [], []
    for s in (+1, -1):
        x0 = s * 12.9
        blue_a.append(trace([(x0, 14.0), (x0, 30.0), (s * 13.4, 31.0), (s * 13.4, 36.0)]))
        light_a += [pad(x0, 14.0, 0.8), pad(s * 13.4, 36.0, 0.8)]
        light_a += [tick(x0, z, 0, 1.6, 0.6) for z in (19.0, 20.6, 22.2)]
    out[CARRIER][7.8] = {"blue": unary_union(blue_a).difference(unary_union(light_a)).intersection(ar),
                         "light": unary_union(light_a).intersection(ar)}
    # CARRIER side tabs, y 6.0: a chevron on each, pointing at the housing
    tabs = F[CARRIER][6.0]
    tr = room(tabs, [])
    tc = unary_union([chevron(s * 10.6, 0.0, 3.4, 2.2, 0.7, pointing=-s) for s in (+1, -1)])
    out[CARRIER][6.0] = {"blue": tc.intersection(tr)}
    return out, F, heads


def colour_of(part, role, pal):
    """Inlay roles -> filament colours, given the part's base colour in palette `pal`."""
    base = PALETTES[pal]["carrier" if part == CARRIER else "clamp"]
    if role == "blue":
        return "blue"
    if role == "dark":
        return "graphite" if base == "white" else "white"
    if role == "light":
        return "white" if base == "graphite" else "graphite"
    raise ValueError(role)


# ---- preview (no SolidWorks) ---------------------------------------------------------
def _polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for q in getattr(g, "geoms", []) for p in _polys(q)]


def preview2d(D, F, heads, pal, path):
    from PIL import Image, ImageDraw
    S = 14.0
    img = Image.new("RGB", (1500, 1150), (24, 26, 30))
    d = ImageDraw.Draw(img)
    rgb = {k: tuple(int(255 * x) for x in v) for k, v in COLOURS.items()}
    # carrier seen from inboard (-Y): x mirrored so the picture reads as on the robot
    for part, ox, oz, flip in ((CARRIER, 330, 330, -1), (CLAMP, 1150, 560, +1)):
        base = rgb[PALETTES[pal]["carrier" if part == CARRIER else "clamp"]]

        def px(x, z):
            return (ox + flip * x * S, oz + z * S) if part == CARRIER else (ox + z * S, oz - (x + 29) * S)
        for y, face in F[part].items():
            for p in _polys(face):
                d.polygon([px(*q) for q in p.exterior.coords], fill=base)
                for h in p.interiors:
                    d.polygon([px(*q) for q in h.coords], fill=(24, 26, 30))
        for y, roles in D[part].items():
            for role, g in roles.items():
                for p in _polys(g):
                    d.polygon([px(*q) for q in p.exterior.coords], fill=rgb[colour_of(part, role, pal)])
                    for h in p.interiors:
                        d.polygon([px(*q) for q in h.coords], fill=base)
        for x, z in heads[part]:
            cx, cy = px(x, z)
            r = 2.725 * S
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 154, 160), outline=(90, 94, 100), width=2)
    d.text((20, 15), f"palette {pal}: carrier {PALETTES[pal]['carrier']}, clamp {PALETTES[pal]['clamp']}  "
                     f"(inboard faces, flush 0.6 mm inlays; grey discs = the new screw heads)", fill=(220, 224, 230))
    img.save(path)
    print("wrote", path)


def preview3d(D, pal, path):
    """The two parts at their place on the tibia, inlays as thin coloured skins."""
    from build123d import import_step
    sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "lib"))
    import render_color as RC
    from render3d import tessellate
    sc = json.load(open(os.path.join(FAST, "scan.json")))
    bodies, enc = [], []
    rgb = {k: tuple(int(255 * x) for x in v) for k, v in COLOURS.items()}

    def to_r(V, M):
        W_ = np.asarray(V) @ M[:, :3].T + M[:, 3]
        return np.stack([W_[:, 0], -W_[:, 2], W_[:, 1]], 1)
    for c in sc["comps"]:
        if not c["name"].startswith("Tibia-1/") or not c["visible"]:
            continue
        M = np.array(c["M"])
        if c.get("step"):
            V, T, _ = tessellate(import_step(os.path.join(FAST, "src", c["step"] + ".step")), 0.08)
            col = (150, 155, 162)
            if c["name"] in (CARRIER, CLAMP):
                col = rgb[PALETTES[pal]["carrier" if c["name"] == CARRIER else "clamp"]]
            bodies.append((to_r(V, M), np.array(T), col))
            if c["name"] in (CARRIER, CLAMP):
                enc.append(to_r(V, M))
        if c["name"] in (CARRIER, CLAMP):
            outward = -1 if c["name"] == CARRIER else +1          # local y of the inboard side
            for y, roles in D[c["name"]].items():
                for role, g in roles.items():
                    for p in _polys(g):
                        tri = _triangulate(p)
                        if not len(tri):
                            continue
                        P3 = np.array([[x, y + outward * 0.03, z] for x, z in tri.reshape(-1, 2)])
                        T3 = np.arange(len(P3)).reshape(-1, 3)
                        if outward > 0:                 # (x, z) CCW faces local -Y: turn it to +Y
                            T3 = T3[:, ::-1]
                        bodies.append((to_r(P3, M), T3, rgb[colour_of(c["name"], role, pal)]))
        if os.path.basename(c["file"]).lower().startswith(("m3 roundhead", "m2 roundhead")) and False:
            pass
    enc = np.vstack(enc)
    tiles = [(t, "", RC.view(bodies, enc, (900, 700), e, a, light="camera"))
             for t, e, a in (("inboard face-on", 0, 90), ("from front-below", -25, 40),
                             ("from back-below", -25, 140), ("3/4 from above", 35, 60))]
    RC.sheet(tiles, 2, path, header=f"Encoder carrier + clamp, palette {pal}",
             sub=f"carrier {PALETTES[pal]['carrier']}, clamp {PALETTES[pal]['clamp']}; flush 0.6 mm inlays "
                 "(screws not drawn)", size=(900, 700))


def _triangulate(p):
    from shapely.ops import triangulate
    tris = [t for t in triangulate(p) if t.representative_point().within(p)]
    return np.array([list(t.exterior.coords)[:3] for t in tris]) if tris else np.zeros((0, 3, 2))


def preview(pals):
    D, F, heads = design()
    for k, roles in D.items():
        for y, rr in roles.items():
            print(f"  {k.split('/')[-1]:18s} y {y:5.1f}: " +
                  ", ".join(f"{r} {g.area:6.1f} mm2" for r, g in rr.items()))
    os.makedirs(RENDERS, exist_ok=True)
    for pal in pals:
        preview2d(D, F, heads, pal, os.path.join(RENDERS, f"encoder_style_{pal}_faces.png"))
        preview3d(D, pal, os.path.join(RENDERS, f"encoder_style_{pal}.png"))


# ---- build (SolidWorks): his pick 2026-10-08 = palette A ---------------------------------
PICK = "A"
PARTS = {CARRIER: r"Links\EncoderCarrier.SLDPRT", CLAMP: r"Links\EncoderCableClamp.SLDPRT"}
INTO = {CARRIER: +1, CLAMP: -1}          # local +Y / -Y: the way INTO the part from its inboard face
FILAMENT = {"white": 1, "graphite": 2, "blue": 3}
PRINT_UP = {CARRIER: lambda V: V[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0]),   # local -Y up, back on the bed
            CLAMP: lambda V: V[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])}     # local +Y up, feet on the bed
OUT = os.path.join(HERE, "out")


def _expect(what, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"    {what:38s} {got:11.4f}  expected {want:11.4f}  {'ok' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"{what}: {got:.4f} vs {want:.4f} -- NOT saved")


def _open(sw, rel, kind=None):
    import swlib
    from swlib import c, wrap, sld
    d, err, warn = sw.OpenDoc6(os.path.join(swlib.V5, rel), kind or c.swDocPART, c.swOpenDocOptions_Silent,
                               "", 0, 0)
    if d is None:
        raise SystemExit(f"cannot open {rel} (err {err})")
    m = wrap(d, sld.IModelDoc2)
    sw.ActivateDoc3(m.GetTitle(), False, 0, 0)
    return m


def _layers(g):
    """Split a region into sketches with no island inside another loop's hole
    (README gotcha 31: SolidWorks reads such a sketch and refuses every cut with
    it): layer k = the polygons nested inside k others' holes."""
    ps = _polys(g)
    depth = [sum(1 for q in ps if q is not p and any(Polygon(h).contains(p) for h in q.interiors)) for p in ps]
    return [unary_union([p for p, d in zip(ps, depth) if d == k]) for k in range(max(depth, default=-1) + 1)]


def build_part(sw, part, D, restyle):
    """Base colour on the original body, then every inlay: Top Plane sketch (part
    (X, -Z)), cut 0.6 into the face (from 0.5 outside it), the same sketch extruded
    back as new bodies, coloured.  Partition checked by volume."""
    import swstyle as S
    from swlib import wrap, sld
    rel = PARTS[part]
    model = _open(sw, rel)
    if any(f.Name.startswith("GL_") for f in S._iter_features(model)):
        if not restyle:
            raise SystemExit(f"{rel} already has GL_ features -- pass --restyle to rebuild")
        bs = S.bodies(model)
        print(f"  --restyle: deleted {S.strip_styling(model, sum(S.volume(b) for b in bs))} GL_ features")
    bs = S.bodies(model)
    if len(bs) != 1:
        raise SystemExit(f"{rel}: expected the original single body, got {len(bs)}")
    body = bs[0]
    v0 = S.total_volume(model)
    box0 = [v * 1000 for v in body.GetBodyBox()]
    base = PALETTES[PICK]["carrier" if part == CARRIER else "clamp"]
    S.colour(body, COLOURS[base])
    made, n = 0, 0
    for y, roles in D[part].items():
        for role, g0 in roles.items():
          for li, g in enumerate(_layers(g0)):                          # gotcha 31: no island in a hole
            if g.is_empty:
                continue
            col = colour_of(part, role, PICK)
            sk_g = affinity.scale(g, 1.0, -1.0, origin=(0, 0))          # Top Plane sketch = (X, -Z)
            key = f"y{y:+.1f}_{role}{li or ''}".replace(".", "p").replace("+", "").replace("-", "m")
            sk = S.sketch(model, sk_g, name=f"GL_{key}", plane="Top Plane")
            a, b = (y - 0.5, y + INLAY) if INTO[part] > 0 else (y - INLAY, y + 0.5)
            S.cut(model, sk, a, b, name=f"GL_{key}_cut", scope=[body])
            t0, t1 = (y, y + INLAY) if INTO[part] > 0 else (y - INLAY, y)
            new = S.tool(model, sk, t0, t1, name=f"GL_{key}_inlay")
            for nb in new:
                S.colour(nb, COLOURS[col])
            made += len(new)
            n += len(S.polys(g, 0))
            print(f"    y {y:5.1f} {role:6s} -> {col:8s} {g.area:7.2f} mm2, {len(new)} bodies")
            body = max(S.bodies(model), key=S.volume)
    for f in S._iter_features(model):
        if f.Name.startswith("GL_") and f.GetTypeName2() == "ProfileFeature":
            model.ClearSelection2(True)
            f.Select2(False, 0)
            model.BlankSketch()
    model.ClearSelection2(True)
    model.EditRebuild3()
    S.name_by_colour(model)
    bs = S.bodies(model)
    lo = [min(b.GetBodyBox()[i] for b in bs) * 1000 for i in range(3)]
    hi = [max(b.GetBodyBox()[i + 3] for b in bs) * 1000 for i in range(3)]
    _expect("volume unchanged (mm3)", S.total_volume(model), v0, 1e-3)
    _expect("bodies = 1 + inlays", len(bs), 1 + made, 0)
    _expect("box moved (mm)", max(abs(p - q) for p, q in zip(lo + hi, box0)), 0.0, 0.01)
    inl = sum(S.volume(b) for b in bs if S.group_of(b) != base)
    want = sum(g.area for rr in D[part].values() for g in rr.values()) * INLAY
    _expect("inlay volume = area x 0.6 (mm3)", inl, want, 0.05 * want + 0.05)
    return model


def build(restyle=False):
    import swlib
    from importlib import import_module
    from swlib import c
    D, F, heads = design()
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    models = {}
    for part in (CARRIER, CLAMP):
        print(f"\n== {PARTS[part]}  ({PALETTES[PICK]['carrier' if part == CARRIER else 'clamp']} base)")
        models[part] = build_part(sw, part, D, restyle)
    mate_errors = import_module("10_verify_styled").mate_errors
    bad = []
    for rel in (r"Links\Tibia.SLDASM", r"Links\MirrorTibia.SLDASM", os.path.relpath(swlib.ASM, swlib.V5)):
        a = _open(sw, rel, c.swDocASSEMBLY)
        a.ForceRebuild3(False)
        bad += [(rel,) + e for e in mate_errors(a)]
    print(f"  mates in error (Tibia, MirrorTibia, ROBOT): {len(bad)}" + "".join(f"\n    {b}" for b in bad))
    if bad:
        raise SystemExit("mate errors -- NOT saved")
    import_module("16_check_and_save").chain(sw, list(PARTS.values()))
    for part, m in models.items():
        export(sw, part, m)


def export(sw, part, model):
    """STEP (one solid per body) + Bambu 3MF in print orientation: the styled face UP
    (inlays = the last layers), the flat back on the bed.  Filament 1 white / 2
    graphite / 3 blue."""
    import swstyle as S
    from importlib import import_module
    from swlib import c
    sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "lib"))
    import export3mf
    body_mesh = import_module("11_check_and_export").body_mesh
    name = os.path.splitext(os.path.basename(PARTS[part]))[0]
    os.makedirs(os.path.join(OUT, "styled"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "print"), exist_ok=True)
    stp = os.path.join(OUT, "styled", f"{name}_glacier_native.step")
    ok, err, warn = model.Extension.SaveAs3(stp, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                            None, None, 0, 0)
    print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(stp, HERE)}")
    parts, by = [], {}
    for b in S.bodies(model):
        V, T = body_mesh(b)
        if V is None:
            continue
        fil = FILAMENT[S.group_of(b)]
        parts.append((b.Name, PRINT_UP[part](np.asarray(V)), T, fil))
        by[fil] = by.get(fil, 0) + 1
    out = os.path.join(OUT, "print", f"{name}_glacier_native.3mf")
    export3mf.write_3mf(parts, out, name=f"{name} GLACIER (native SolidWorks), styled face up")
    print(f"  3MF: {len(parts)} bodies (white x{by.get(1, 0)}, graphite x{by.get(2, 0)}, "
          f"blue x{by.get(3, 0)}) -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "preview":
        preview(args[1:] or ["A", "B"])
    elif args and args[0] == "build":
        build("--restyle" in args)
    else:
        print(__doc__)
