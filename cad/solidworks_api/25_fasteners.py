"""Step 25: fasteners everywhere the robot has a hole for one (README "Fasteners").

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/25_fasteners.py <command>

  scan [--fresh] [--all]   SolidWorks, read-only: placements + one STEP per part file
                     (--all: the left side too -> scan_all.json, for audit)
  audit              offline: every screw of scan_all.json -- head buried / not seated,
                     tip sticking out, axis through solid, doubled
  holes              offline: every bore of every part, robot frame
  lines              offline: coaxial bores of different parts = screw lines; which hold a fastener
  profile <id> ... [--rho 1.45,2.4]   offline: material along a line (grip, blind ends, seats)
  plan               offline: GROUPS / STACKS -> every fastener's exact placement -> plan.json
  models             SolidWorks: the new fastener parts in Common/ (fix-lengths: their configurations)
  build [--dry] [--left] [asm ...]    insert + mate (idempotent); --left: plan_left.json
  standins [--dry]   CornerBracket configurations dropping the flatheads that double a longer screw
  left [--report]    the left leg live (mirror instances unsuppressed, left brackets shown)
  plan-left          the left twins of the fasteners the mirror features did not carry over
  mirror-colours     the left opposite-hand parts take their source bodies' colours
  fix-left-body [--dry]   MirrorSIDE PANEL's drifted shaft / RobotMount / screw back to the mirror, locked
  mirror-fasteners   (DO NOT RUN: ModifyDefinition on BodyMirror deleted MirrorBODY-2 -- kept as a record)
  bom                SolidWorks, read-only: every fastener in ROBOT -> out/fasteners/BOM.md

Nothing here saves: after a SolidWorks step, 16_check_and_save.py --chain <what changed>.

scan (READ-ONLY, SolidWorks): every leaf part of ROBOT.SLDASM that is resolved
(the suppressed left leg and the suppressed v3 box parts drop out, as do
AK_SIM / WheelHanger): its placement, visibility and configuration ->
out/fasteners/scan.json, and one STEP per part file in its own frame ->
out/fasteners/src/ (kept if already there; --fresh re-exports).  Fasteners are
not exported: their axis is a known local axis of the model (FASTENERS).

Why STEP and not the API: against this session (77 documents) every COM call
costs ~10 ms; a face-by-face walk of the styled parts ran > 10 min without
finishing a part.  One SaveAs3 per part and OCC offline is minutes.

holes (offline, OCC): every hole-like cylinder (material outside it) r 0.7..5
mm of every part, in the assembly frame -> out/fasteners/holes.json.
"""
import os
import re
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out", "fasteners")
SRC = os.path.join(OUT, "src")
SCAN = os.path.join(OUT, "scan.json")

SKIP_TOP = ("AK_SIM-1", "WheelHanger-1")       # not-for-real helpers (swlib.NOT_FOR_COLLISION);
                                               # Mirror* (the left leg) is the right one mirrored
NO_HOLES = re.compile(r"^Bearing 6804", re.I)  # bought parts with no fastener holes: not exported
# fastener models: local axis (unit), from the probe of each model 2026-10-08
FASTENERS = {"m3 roundhead.sldprt": (0, 0, 1), "m3 flathead.sldprt": (0, 0, 1),
             "m3x8 flathead.sldprt": (0, 0, 1), "m4 roundhead.sldprt": (0, 0, 1),
             "m3 nut.sldprt": (0, 0, 1), "m3 washer.sldprt": (0, 1, 0),
             "m3 standoff.sldprt": (0, 0, 1), "m2 standoff.sldprt": (0, 1, 0),
             "m2.5 roundhead.sldprt": (0, 0, 1), "m2 roundhead.sldprt": (0, 0, 1),
             "m3 standoff mf.sldprt": (0, 0, 1)}
SCAN_ALL = os.path.join(OUT, "scan_all.json")   # scan --all: the left side too (audit); scan.json
                                                # stays the right-only scan the line ids come from
R_MIN, R_MAX = 0.7, 5.0                         # mm: M1.6 tap .. M4 head counterbore


def _key(path, cfg):
    return f"{os.path.splitext(os.path.basename(path))[0]}__{cfg}"


# ---- scan (SolidWorks, read-only) ----------------------------------------------
def scan(fresh=False, everything=False):
    """everything: the left side (Mirror*) too, -> scan_all.json (for `audit`)."""
    import time
    import swlib
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    comps = swlib.components(model)
    docs = {os.path.normcase(wrap(d, sld.IModelDoc2).GetPathName()): wrap(d, sld.IModelDoc2)
            for d in sw.GetDocuments() or []}
    os.makedirs(SRC, exist_ok=True)
    out, exported = [], {}
    t0 = time.time()
    for n in sorted(comps):
        cp = comps[n]
        top = n.split("/")[0]
        if top in SKIP_TOP or (top.startswith("Mirror") and not everything) or cp.IsSuppressed() \
                or cp.GetChildren():
            continue
        path = cp.GetPathName() or ""
        if not path.lower().endswith(".sldprt"):
            continue
        cfg = cp.ReferencedConfiguration
        vis, par = bool(cp.Visible), cp.GetParent()
        while par is not None:                      # hidden if any parent is hidden
            par = wrap(par, sld.IComponent2)
            vis = vis and bool(par.Visible)
            par = par.GetParent()
        base = os.path.basename(path).lower()
        rec = dict(name=n, file=os.path.relpath(path, swlib.V5), config=cfg, visible=vis,
                   M=np.round(swlib.placement(cp), 6).tolist(), fastener=base in FASTENERS)
        out.append(rec)
        # --all exports the nuts / washers / standoffs too: the audit counts them as material
        if (rec["fastener"] and not (everything and base not in SCREWS)) or NO_HOLES.search(os.path.basename(path)):
            continue
        key = _key(path, cfg)
        rec["step"] = key
        if key in exported:
            continue
        step = os.path.join(SRC, key + ".step")
        doc = docs.get(os.path.normcase(path))
        active = wrap(doc.ConfigurationManager.ActiveConfiguration, sld.IConfiguration).Name if doc else None
        if os.path.exists(step) and not fresh:
            exported[key] = "cached"
        elif doc is None:
            exported[key] = "NOT LOADED"
        elif active != cfg:
            exported[key] = f"SKIPPED: active config {active!r} != {cfg!r} (gotcha 32: not switched)"
        else:
            ok, err, warn = doc.Extension.SaveAs3(step, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                                  None, None, 0, 0)
            exported[key] = "ok" if ok else f"FAILED err {err}"
        print(f"  {time.time() - t0:5.0f}s  {key:60s} {exported[key]}", flush=True)
    with open(SCAN_ALL if everything else SCAN, "w") as f:
        json.dump(dict(comps=out, exported=exported), f, indent=0)
    print(f"{len(out)} parts ({sum(r['fastener'] for r in out)} fasteners), {len(exported)} part files "
          f"-> {SCAN_ALL if everything else SCAN}")


# ---- holes (offline, OCC) --------------------------------------------------------
_SHAPES = {}


def shape(key):
    """The part's STEP (its own frame, mm) as one OCC shape, cached."""
    if key not in _SHAPES:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        rd = STEPControl_Reader()
        if rd.ReadFile(os.path.join(SRC, key + ".step")) != IFSelect_RetDone:
            raise RuntimeError(f"cannot read {key}.step")
        rd.TransferRoots()
        _SHAPES[key] = rd.OneShape()
    return _SHAPES[key]


def local_holes(key):
    """Bores of one part file in its own frame: cylinders r R_MIN..R_MAX whose
    material is OUTSIDE (the face normal points at the axis), split faces merged.
    Each: r, point p on the axis at the low end, unit axis a, length L, the
    angle the merged faces cover (deg) -- a slot end or a fillet covers < 180."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.BRepTools import BRepTools
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.gp import gp_Pnt, gp_Vec
    raw = []
    ex = TopExp_Explorer(shape(key), TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        ex.Next()
        ad = BRepAdaptor_Surface(f)
        if ad.GetType() != GeomAbs_Cylinder:
            continue
        cy = ad.Cylinder()
        r = cy.Radius()
        if not R_MIN <= r <= R_MAX:
            continue
        ax = cy.Axis()
        o = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        a = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
        pnt, vec = gp_Pnt(), gp_Vec()
        BRepGProp_Face(f).Normal((u0 + u1) / 2, (v0 + v1) / 2, pnt, vec)
        q = np.array([pnt.X(), pnt.Y(), pnt.Z()]) - o
        rad = q - (q @ a) * a
        if np.array([vec.X(), vec.Y(), vec.Z()]) @ rad >= 0:      # normal away from the axis: a boss
            continue
        raw.append(dict(r=r, o=o, a=a, v0=v0, v1=v1, span=np.degrees(u1 - u0)))
    out = []
    for h in raw:
        if h["a"] @ np.array([1.0, 1.0, 1.0]) < 0:                 # one canonical direction per line
            h["a"], h["v0"], h["v1"] = -h["a"], -h["v1"], -h["v0"]
        p0 = h["o"] + h["v0"] * h["a"]
        for m in out:
            d = p0 - m["p"]
            if abs(m["r"] - h["r"]) < 1e-3 and m["a"] @ h["a"] > 1 - 1e-9 \
                    and np.linalg.norm(d - (d @ m["a"]) * m["a"]) < 1e-3:
                s0 = float(d @ m["a"])
                s1 = s0 + h["v1"] - h["v0"]
                if s0 < m["L"] + 1e-3 and s1 > -1e-3:             # overlapping extents: one bore
                    if abs(s0) < 1e-3 and abs(s1 - m["L"]) < 1e-3:
                        m["span"] += h["span"]                    # the other half of the same band
                    else:                                         # a band above/below: extend
                        lo, hi = min(0.0, s0), max(m["L"], s1)
                        m["p"] = m["p"] + lo * m["a"]
                        m["L"] = hi - lo
                        m["span"] = max(m["span"], h["span"])
                    break
        else:
            out.append(dict(r=round(h["r"], 4), p=p0, a=h["a"], L=h["v1"] - h["v0"], span=h["span"]))
    return out


def holes():
    """Every bore of every part instance in the assembly frame -> holes.json."""
    with open(SCAN) as f:
        sc = json.load(f)
    cache, out = {}, []
    for rec in sc["comps"]:
        key = rec.get("step")
        if not key or sc["exported"].get(key) not in ("ok", "cached"):
            continue
        if key not in cache:
            cache[key] = local_holes(key)
            print(f"  {key:60s} {len(cache[key]):4d} bores", flush=True)
        M = np.array(rec["M"])
        for h in cache[key]:
            out.append(dict(comp=rec["name"], file=rec["file"], visible=rec["visible"], r=h["r"],
                            p=(M[:, :3] @ h["p"] + M[:, 3]).round(4).tolist(),
                            a=(M[:, :3] @ h["a"]).round(6).tolist(), L=round(h["L"], 4),
                            span=round(h["span"], 1)))
    with open(os.path.join(OUT, "holes.json"), "w") as f:
        json.dump(out, f, indent=0)
    print(f"{len(out)} bores in {len(cache)} part files -> holes.json")


# ---- lines (offline) ---------------------------------------------------------------
SHANK = (0.75, 2.4)     # mm radius: a bore a screw shank goes into (M2 tap .. M4 clearance)
FULL = 300.0            # deg: a bore, not a slot end or a fillet
ON_LINE = 0.2           # mm: two bores on one screw line


def fastener_axes(sc):
    out = []
    for rec in sc["comps"]:
        if rec["fastener"]:
            M = np.array(rec["M"])
            a = M[:, :3] @ np.array(FASTENERS[os.path.basename(rec["file"]).lower()], float)
            out.append(dict(comp=rec["name"], file=rec["file"], config=rec["config"], p=M[:, 3], a=a))
    return out


def lines():
    """Coaxial bores of all parts -> screw lines; which already have a fastener."""
    with open(SCAN) as f:
        sc = json.load(f)
    with open(os.path.join(OUT, "holes.json")) as f:
        H = json.load(f)
    for h in H:
        h["p"], h["a"] = np.array(h["p"]), np.array(h["a"])
    shank = [h for h in H if SHANK[0] <= h["r"] <= SHANK[1] and h["span"] >= FULL]
    L = []                                   # each: a, p (a point), members
    for h in shank:
        for ln in L:
            d = h["p"] - ln["p"]
            if abs(ln["a"] @ h["a"]) > 1 - 1e-6 and np.linalg.norm(d - (d @ ln["a"]) * ln["a"]) < ON_LINE:
                ln["m"].append(h)
                break
        else:
            L.append(dict(a=h["a"], p=h["p"], m=[h]))
    big = [h for h in H if h["r"] > SHANK[1] and h["span"] >= FULL]
    fast = fastener_axes(sc)
    for ln in L:
        a, p = ln["a"], ln["p"]

        def on(q, b, tol=ON_LINE):
            d = q - p
            return abs(a @ b) > 1 - 1e-6 and np.linalg.norm(d - (d @ a) * a) < tol

        def span(h):
            s = sorted([float((h["p"] - p) @ a), float((h["p"] + h["L"] * h["a"] - p) @ a)])
            return [round(s[0], 3), round(s[1], 3)]
        for h in ln["m"]:
            h["t"] = span(h)
        ln["cb"] = [dict(h, t=span(h)) for h in big if on(h["p"], h["a"])]
        lo = min(h["t"][0] for h in ln["m"] + ln["cb"])
        hi = max(h["t"][1] for h in ln["m"] + ln["cb"])
        ln["f"] = [dict(x, t=round(float((x["p"] - p) @ a), 3)) for x in fast
                   if on(x["p"], x["a"], 0.3) and lo - 25 <= (x["p"] - p) @ a <= hi + 25]
        ln["parts"] = sorted({h["comp"] for h in ln["m"]})
    L.sort(key=lambda ln: (bool(ln["f"]), len(ln["parts"]), ln["parts"]))
    rows = []
    for i, ln in enumerate(L):
        rows.append(dict(id=i, a=ln["a"].round(6).tolist(), p=ln["p"].round(4).tolist(),
                         parts=ln["parts"],
                         bores=[dict(comp=h["comp"], r=h["r"], t=h["t"], vis=h["visible"]) for h in ln["m"]],
                         cbores=[dict(comp=h["comp"], r=h["r"], t=h["t"]) for h in ln["cb"]],
                         fasteners=[dict(comp=x["comp"], config=x["config"], t=x["t"]) for x in ln["f"]]))
    with open(os.path.join(OUT, "lines.json"), "w") as f:
        json.dump(rows, f, indent=1)
    n_f = sum(1 for r in rows if r["fasteners"])
    print(f"{len(rows)} lines: {n_f} with a fastener, {len(rows) - n_f} without "
          f"({sum(1 for r in rows if not r['fasteners'] and len(r['parts']) > 1)} through 2+ parts)")
    return rows


# ---- profile (offline): material along a screw line ---------------------------------
_BOX = {}


def bbox(key):
    """The part's bounding box in its own frame (mm), cached."""
    if key not in _BOX:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        b = Bnd_Box()
        BRepBndLib.Add_s(shape(key), b)
        x0, y0, z0, x1, y1, z1 = b.Get()
        _BOX[key] = (np.array([x0, y0, z0]), np.array([x1, y1, z1]))
    return _BOX[key]


def material(key, M, p, a, t0, t1, rho=0.0, n=8):
    """Intervals [ta, tb] of the line p + t a (assembly frame) that lie in the
    part's material, t0..t1.  rho > 0: n lines parallel to the axis at radius rho,
    an interval counted only where ALL of them are in material (a shank of radius
    rho bites there) -- returned per offset line otherwise."""
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.IntCurveSurface import IntCurveSurface_In, IntCurveSurface_Out
    from OCP.gp import gp_Pnt, gp_Lin, gp_Dir
    R, t = M[:, :3], M[:, 3]
    pl = R.T @ (p - t)
    al = R.T @ a
    isec = IntCurvesFace_ShapeIntersector()
    isec.Load(shape(key), 1e-6)
    u = np.cross(al, [1.0, 0, 0] if abs(al[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(al, u)
    offs = [np.zeros(3)] if rho == 0 else [rho * (np.cos(k * 2 * np.pi / n) * u + np.sin(k * 2 * np.pi / n) * v)
                                            for k in range(n)]
    res = []
    for o in offs:
        q = pl + o + (t0 - 1.0) * al
        isec.Perform(gp_Lin(gp_Pnt(*q), gp_Dir(*al)), 0.0, (t1 - t0) + 2.0)
        hits = sorted(((isec.WParameter(i) + t0 - 1.0, isec.Transition(i)) for i in range(1, isec.NbPnt() + 1)),
                      key=lambda h: h[0])
        iv, start = [], None
        for k, (w, tr) in enumerate(hits):
            if tr == IntCurveSurface_In and start is None:
                start = w
            elif tr == IntCurveSurface_Out and start is not None:
                iv.append([round(start, 3), round(w, 3)])
                start = None
            elif tr == IntCurveSurface_Out and k == 0:          # the line STARTS inside material
                iv.append([round(t0 - 1.0, 3), round(w, 3)])
        res.append(iv)
    if rho == 0:
        return res[0]
    # intersection of all offset lines' intervals
    common = res[0]
    for iv in res[1:]:
        common = [[max(a0, b0), min(a1, b1)] for a0, a1 in common for b0, b1 in iv if min(a1, b1) > max(a0, b0)]
    return common


def profile(ids, rhos=(1.45,)):
    with open(SCAN) as f:
        sc = json.load(f)
    with open(os.path.join(OUT, "lines.json")) as f:
        rows = {r["id"]: r for r in json.load(f)}
    comps = {c["name"]: c for c in sc["comps"]}
    for i in ids:
        r = rows[i]
        p, a = np.array(r["p"]), np.array(r["a"])
        ts = [b["t"] for b in r["bores"] + r["cbores"]] + [[x["t"], x["t"]] for x in r["fasteners"]]
        t0, t1 = min(x[0] for x in ts) - 12, max(x[1] for x in ts) + 12
        print(f"\nL{i}  a {np.round(a, 3).tolist()}  p {np.round(p, 2).tolist()}  t {t0:.1f}..{t1:.1f}")
        for b in sorted(r["bores"] + r["cbores"], key=lambda b: b["t"][0]):
            print(f"   bore  {b['comp'].split('/')[-1]:28s} r{b['r']:.3f}  t {b['t'][0]:7.2f} .. {b['t'][1]:7.2f}")
        for x in r["fasteners"]:
            print(f"   FAST  {x['comp']:40s} {x['config']}  origin t {x['t']:.2f}")
        # every part whose material the line (or a 1.5 mm shank round it) meets
        seg = p + np.linspace(t0, t1, int(t1 - t0) + 2)[:, None] * a
        for c in sc["comps"]:
            key = c.get("step")
            if not key or c["fastener"]:
                continue
            M = np.array(c["M"])
            lo, hi = bbox(key)
            ql = (seg - M[:, 3]) @ M[:, :3]                     # the segment in the part's frame
            if not np.any(np.all((ql > lo - 3) & (ql < hi + 3), axis=1)):
                continue
            ax = material(key, M, p, a, t0, t1)
            sh = [material(key, M, p, a, t0, t1, rho=r) for r in rhos]
            if ax or any(sh):
                print(f"   {c['name'].split('/')[-1]:28s} axis {ax}   "
                      + "   ".join(f"r{r} {s}" for r, s in zip(rhos, sh)))


# ---- plan (offline): what goes where ----------------------------------------------------
# One row per joint type, from the line profiles (2026-10-08) and his answers (M2.5 on
# the AK45; standoffs + screws under the electronics; bumper screws in CAD; wheel hub
# left out).  seat ~ t of the head's bearing face (a flathead: its top, flush) along the
# line; head = +1 if the head is on the +t side.  `seg` picks one end of a line that
# runs through the whole box (both skirts / both bumpers): '+' t > -50, '-' t < -50.
# The seat is refined from the seat part's material at radius `probe` (within 0.6 mm).
#   asm      the lowest assembly holding every part the screw joins (his hierarchy rule)
#   bore     the part whose bore the screw is concentric with; seat: the part it bears on
R3 = r"Common\M3 Roundhead.SLDPRT"
F3 = r"Common\M3 Flathead.SLDPRT"
R25 = r"Common\M2.5 Roundhead.SLDPRT"
R2 = r"Common\M2 Roundhead.SLDPRT"
SO3 = r"Common\M3 standoff MF.SLDPRT"
TIB, SP, FEM = r"Links\Tibia.SLDASM", r"Body\SIDE PANEL.SLDASM", r"Links\Femur.SLDASM"
BPWA, BOX, SWM = r"Box\BottomPanelWithAvionics.SLDASM", r"Box\Box.SLDASM", r"Body\LimitSwitch\SwicthMount.SLDASM"
GROUPS = [
    # name            lines                       model cfg     seat    head asm   bore                 seat part          probe seg
    ("carrier-tibia", [161, 162, 163, 164, 166], R3, "10mm", -0.5, -1, TIB, "EncoderCarrier-1", "EncoderCarrier-1", 2.4, None),
    ("clamp-carrier", [155, 156], R3, "14mm", 10.0, +1, TIB, "EncoderCableClamp-1", "EncoderCableClamp-1", 2.8, None),
    # these four sit on DOMED bosses (top 9.9 at r ~2.2, nothing flat; the carrier face
    # under the clamp is chamfered there too): the head height is held by a distance mate
    # (0.1) to the clamp's own flat top, t 10.0, picked where screw L155 bears on it
    ("clamp-carrier-dome", [157, 158, 159, 160], R3, "14mm", 9.9, +1, TIB, "EncoderCableClamp-1",
     "EncoderCableClamp-1", 2.2, None,
     dict(fixed=True, ref_line=155, ref_t=10.0, ref_part="EncoderCableClamp-1", ref_r=3.3)),   # 3.3: clear of
                                                                                                # L155's head
    ("encoder-pcb", [169, 170], R2, "6mm", 6.6, +1, TIB, "Encoder-1", "Encoder-1", 1.5, None),
    ("stator-sidepanel", [107, 108, 109, 110, 111], R25, "12mm", 20.5, +1, SP, "Side panel-1", "Side panel-1", 2.0, None),
    ("femur-rotor", [152, 153, 154], R25, "16mm", -9.72, -1, FEM, "Femur-1", "Femur-1", 2.0, None),
    ("tailstrut-floor", [125, 126, 127, 128], R3, "14mm", -10.0, -1, BPWA, "TailStrut-1", "TailStrut-1", 2.4, None),
    ("caster-axle", [140], R3, "30mm", 20.0, +1, BPWA, "TailStrut-1", "TailStrut-1", 2.4, None),
    ("battery-cap", [114, 115], R3, "30mm", 41.8, +1, BPWA, "BatteryCap-1", "BatteryCap-1", 2.8, None),
    ("battery-holder", [116, 117], F3, "6mm", 2.0, +1, BPWA, "BottomPanel-1", "Battery holder-1", 3.6, None),
    ("limit-switch", [112, 113], R2, "8mm", 5.7, +1, SWM, "LimitSwitch-2", "LimitSwitch-2", 1.3, None),
    ("hood-ring+", [147, 148, 149, 150, 151], F3, "10mm", 2.5, +1, BOX, "FacetRing-1", "FacetHood-1", 3.6, "+"),
    ("hood-ring-", [147, 148, 149, 150, 151], F3, "10mm", None, -1, BOX, "FacetRing-1", "FacetHood-1", 3.6, "-"),
    ("bumper-front-ctr", [129], R3, "20mm", -15.03, -1, BOX, "FacetFront-1", "BumperFront-1", 2.4, None),
    ("bumper-back-ctr", [214], R3, "25mm", -15.87, -1, BOX, "FacetBack-1", "BumperBack-1", 2.4, None),
    ("bumper-back", [243, 244, 245, 255, 256, 257], R3, "20mm", 0.0, -1, BOX, "FacetBack-1", "BumperBack-1", 2.4, "+"),
    ("bumper-front", [246, 247, 248], R3, "20mm", 3.71, +1, BOX, "FacetFront-1", "BumperFront-1", 2.4, None),
    ("bumper-front2", [255, 256, 257], R3, "20mm", 223.10, +1, BOX, "FacetFront-1", "BumperFront-1", 2.4, "far"),
    ("oled", [143, 144, 145, 146], R2, "4mm", -1.2, -1, BOX, "OLED-1", "OLED-1", 1.5, None),
]
# standoff stacks under the electronics (BottomPanelWithAvionics): floor top at t 5
STACKS = [  # name, lines, [(standoff cfg, base t)], screw cfg, seat t, bore part, seat part
    ("odrive", [120, 121, 122, 123, 124], [("12mm", 5.0)], "6mm", 18.71, "Odrive-1", "Odrive-1"),
    ("odrive-resistor", [167, 168], [("12mm", 5.0), ("5mm", 18.71)], "6mm", 26.16, "Resistor-1", "Resistor-1"),
    ("imu", [118, 119], [("8mm", 5.0)], "6mm", 14.13, "IMU BNO085-1", "IMU BNO085-1"),
]
# the CornerBracket's own M3x8 flatheads that stand in for a longer screw (two models of
# one screw in one hole): under the bumpers (the real one: M3x20/25 from the bumper) and
# in the 5 side-panel holes (the real one: the side panel's M3x35).  Mirror instances
# (CornerBracket-9..12/17/18/30/31, BPWA -3/-4) take their seed's configuration.
STAND_IN = {  # component (ROBOT path) -> flatheads to drop
    "Box-1/CornerBracket-1": [1], "Box-1/CornerBracket-2": [1], "Box-1/CornerBracket-13": [2],
    "Box-1/CornerBracket-27": [1], "Box-1/CornerBracket-3": [1, 2], "Box-1/CornerBracket-4": [1, 2],
    "Box-1/CornerBracket-14": [1, 2], "Box-1/CornerBracket-28": [1],
    "Box-1/BottomPanelWithAvionics-1/CornerBracket-2": [1],
}
LEFT_OUT = {   # found, deliberately not modelled -- in the BOM as open items
    "wheel hub -> motor can (4 per wheel, r17 on the hub face)":
        "the hub bosses pass through 7 mm holes in the can; nothing behind them is modelled to thread into "
        "(his call 2026-10-08: leave out, flag)",
    "EncoderCarrier r1.6 hole over the tibia (line 165)":
        "the carrier is SOLID on the head side (2.6 mm wall) -- no screw can go in; a 6th carrier->tibia "
        "screw would need that wall opened",
}
TO_CHECK = {   # in the model and in the count, but probably not real -- his call
    "4 short screws at the femur <-> Femur_inside joint, per leg (2 x M3x20 in Femur, 2 x M3x18 in FEMUR_INSIDE)":
        "each enters the far end of a self-tap bore the M3x50 from the other side is already in "
        "(lines 226-229): it clamps nothing.  Remove them and the BOM drops 4 x M3x20 + 4 x M3x18",
}
CONFIG_LEN = {R3: None, F3: None, R25: None, R2: None}   # length = the configuration name


def _len(cfg):
    return float(cfg.replace("mm", ""))


def plan():
    """GROUPS/STACKS -> every fastener with its exact placement (ROBOT frame) -> plan.json."""
    with open(SCAN) as f:
        sc = json.load(f)
    with open(os.path.join(OUT, "lines.json")) as f:
        rows = {r["id"]: r for r in json.load(f)}
    comps = {c_["name"]: c_ for c_ in sc["comps"]}

    def comp_of(row, leaf):
        for b in row["bores"] + row["cbores"]:
            if b["comp"].split("/")[-1] == leaf:
                return b["comp"]
        for n in comps:                         # a part with no bore on the line (a seat only)
            if n.split("/")[-1] == leaf:
                return n
        raise SystemExit(f"{leaf} not on line {row['id']}")

    def refine(row, seat_leaf, guess, head, probe):
        """The seat part's surface facing the head side, at radius `probe`, nearest `guess`."""
        cn = comp_of(row, seat_leaf)
        c_ = comps[cn]
        p, a = np.array(row["p"]), np.array(row["a"])
        iv = material(c_["step"], np.array(c_["M"]), p, a, guess - 15, guess + 15, rho=probe)
        ends = [x[1] if head > 0 else x[0] for x in iv]       # the face looking at the head
        if not ends:
            return None, cn
        t = min(ends, key=lambda e: abs(e - guess))
        return (t if abs(t - guess) < 0.6 else None), cn

    with open(os.path.join(OUT, "holes.json")) as f:
        H = json.load(f)

    def own_bore(row, comp, t_near):
        """The bore of `comp` on this line nearest t_near, exactly: a screw goes on THAT
        axis (bores of one line differ by up to ON_LINE; a concentric mate onto an axis
        0.05 mm off moves the screw, and the mate is refused)."""
        p, a = np.array(row["p"]), np.array(row["a"])
        best = None
        for h in H:
            if h["comp"] != comp or not SHANK[0] <= h["r"] <= SHANK[1]:
                continue
            ha, hp = np.array(h["a"]), np.array(h["p"])
            d = hp - p
            if abs(ha @ a) < 1 - 1e-6 or np.linalg.norm(d - (d @ a) * a) > ON_LINE:
                continue
            mid = float((hp + h["L"] / 2 * ha - p) @ a)
            if best is None or abs(mid - t_near) < abs(best[0] - t_near):
                best = (mid, h)
        if best is None:
            raise SystemExit(f"no bore of {comp} on line {row['id']}")
        h = best[1]
        A = np.array(h["a"]) * np.sign(np.array(h["a"]) @ a)
        P0 = np.array(h["p"]) + h["L"] / 2 * np.array(h["a"])       # the bore's middle, on its axis
        return P0, A, h["r"]

    def item(row, group, model, cfg, asm, t, head, bore_comp, seat_comp):
        p, a = np.array(row["p"]), np.array(row["a"])
        P0, A, br = own_bore(row, bore_comp, t)
        S_ = P0 + float((p + t * a - P0) @ A) * A                     # the seat, on the bore's axis
        return dict(group=group, line=row["id"], model=model, config=cfg, asm=asm,
                    seat=S_.round(5).tolist(), dir=(head * A).round(8).tolist(),
                    bore=bore_comp, bore_mid=P0.round(5).tolist(), bore_r=br, seat_part=seat_comp,
                    t=round(t, 3))

    out, bad = [], []
    for g in GROUPS:
        name, ids, model, cfg, seat, head, asm, bore, seat_part, probe, seg = g[:11]
        opt = g[11] if len(g) > 11 else {}
        for i in ids:
            r = rows[i]
            guess = seat
            if seg == "-":       # the far (low-t) skirt: its outer face is the mirror of the '+' one
                lo = min(b["t"][0] for b in r["bores"] if b["comp"].endswith(seat_part))
                guess = lo - 1.5
            if opt.get("fixed"):
                t, cn = guess, comp_of(r, seat_part)
            else:
                t, cn = refine(r, seat_part, guess, head, probe)
            if t is None:
                bad.append(f"{name} L{i}: no {seat_part} face near t {guess:.2f} at r {probe}")
                continue
            it = item(r, name, model, cfg, asm, t, head, comp_of(r, bore), cn)
            it["probe"] = probe
            if "ref_t" in opt:                      # the plane the distance mate measures from
                rr_ = rows[opt.get("ref_line", i)]
                A = np.array(it["dir"]) * head
                if abs(abs(A @ np.array(rr_["a"])) - 1) > 1e-6:
                    raise SystemExit(f"{name} L{i}: reference line {rr_['id']} is not parallel")
                P0 = np.array(own_bore(rr_, comp_of(rr_, opt["ref_part"]), opt["ref_t"])[0]) \
                    if opt.get("ref_line") else np.array(it["bore_mid"])
                q = np.array(rr_["p"]) + opt["ref_t"] * np.array(rr_["a"])
                it["ref"] = dict(point=(P0 + float((q - P0) @ A) * A).round(5).tolist(),
                                 part=comp_of(rr_, opt["ref_part"]), r=opt["ref_r"],
                                 dist=round(abs(t - opt["ref_t"]), 4))
            out.append(it)
    for name, ids, stack, scfg, seat, bore, seat_part in STACKS:
        for i in ids:
            r = rows[i]
            for k, (so, base) in enumerate(stack):
                host = "BottomPanel-1" if k == 0 else "Odrive-1"
                t, cn = refine(r, host, base, +1, 3.6)           # the face the standoff stands on
                if t is None:
                    bad.append(f"{name} L{i}: no {host} top near t {base}")
                    continue
                it = item(r, f"{name}-standoff{k + 1}", SO3, so, BPWA, t, +1, comp_of(r, host), cn)
                it["probe"] = 3.6
                out.append(it)
            t, cn = refine(r, seat_part, seat, +1, 2.4)
            if t is None:
                bad.append(f"{name} L{i}: no {seat_part} top near t {seat}")
                continue
            it = item(r, f"{name}-screw", R3, scfg, BPWA, t, +1, comp_of(r, bore), cn)
            it["probe"] = 2.4
            out.append(it)
    for x in out:
        print(f"  {x['group']:26s} L{x['line']:3d}  {os.path.basename(x['model'])[:-7]:16s} {x['config']:5s} "
              f"seat t {x['t']:8.3f}  on {x['seat_part'].split('/')[-1]:22s} -> {os.path.basename(x['asm'])}")
    with open(os.path.join(OUT, "plan.json"), "w") as f:
        json.dump(dict(add=out, stand_in=STAND_IN, left_out=LEFT_OUT), f, indent=1)
    print(f"{len(out)} fasteners planned" + ("".join("\n  REFUSED " + b for b in bad) if bad else ""))
    return out, bad


# ---- build (SolidWorks): insert + mate every planned fastener -------------------------
SHANK_R = {R3: 1.5, F3: 1.5, R25: 1.25, R2: 1.0, SO3: 1.5}     # the cylinder the concentric uses
MID_ORIGIN = (R3, F3)          # v5's own screws: origin at mid-length, head face at +L/2
FLAT = (F3,)                   # bears with its TOP flush with the seat face
MPFX = "FS_"                   # mate names


def _frame(D):
    z = np.asarray(D, float) / np.linalg.norm(D)
    x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def fastener_M(it, S, D):
    """3x4 placement (mm, the target assembly's frame) of the fastener model."""
    R = _frame(D)
    o = np.asarray(S, float)
    if it["model"] in MID_ORIGIN:
        o = o - R[:, 2] * _len(it["config"]) / 2
    return np.hstack([R, o[:, None]])


class Asm:
    """One target assembly, open in its own window, with the cheap helpers the
    build needs (Mater.faces / placements walk every component -- minutes in Box)."""

    def __init__(self, sw, rel, T):
        import swlib
        from swlib import c, wrap, sld
        self.sw, self.rel = sw, rel
        path = os.path.join(swlib.V5, rel)
        doc, err, warn = sw.OpenDoc6(path, c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)
        if doc is None:
            raise SystemExit(f"cannot open {rel} ({err})")
        self.doc = wrap(doc, sld.IModelDoc2)
        self.asm = wrap(self.doc, sld.IAssemblyDoc)
        self.T = T                                  # this assembly in ROBOT (3x4 mm)
        self.activate()

    def activate(self):
        self.sw.ActivateDoc3(self.doc.GetTitle(), False, 0, 0)

    def local(self, p, d=None):
        R, t = self.T[:, :3], self.T[:, 3]
        return R.T @ (np.asarray(p) - t) if d is None else R.T @ np.asarray(d)

    def comp(self, name):
        from swlib import wrap, sld
        for x in self.asm.GetComponents(False) or []:
            cp = wrap(x, sld.IComponent2)
            if cp.Name2 == name:
                return cp
        return None

    def ray(self, o, d, want_leaf):
        """First face hit from o along d (mm, this assembly's frame) -> (face, component)."""
        from swlib import c, wrap, sld
        m = self.doc
        m.ClearSelection2(True)
        ok = m.Extension.SelectByRay(*(np.asarray(o) / 1000.0), *np.asarray(d, float), 0.0001,
                                     c.swSelFACES, False, 0, 0)
        if not ok:
            return None, None, "no hit"
        sm = wrap(m.SelectionManager, sld.ISelectionMgr)
        f = wrap(sm.GetSelectedObject6(1, -1), sld.IFace2)
        cp = wrap(sm.GetSelectedObjectsComponent4(1, -1), sld.IComponent2)
        m.ClearSelection2(True)
        nm = cp.Name2 if cp is not None else "?"
        if nm.split("/")[-1] != want_leaf:
            return None, None, f"hit {nm}, not {want_leaf}"
        return f, cp, "ok"

    def mates(self):
        from swlib import wrap, sld
        import swstyle as S
        out = []
        for f in S._iter_features(self.doc):
            if f.GetTypeName2() == "MateGroup":
                s = wrap(f.GetFirstSubFeature(), sld.IFeature)
                while s is not None:
                    out.append(s)
                    s = wrap(s.GetNextSubFeature(), sld.IFeature)
        return out

    def mate(self, name, kind, fa, fb, watch, dist=0.0, spin_ok=None):
        """Concentric / coincident / distance between two IFace2 (component context);
        kept only if nothing in `watch` moved and the mate has no error.  Else deleted
        + put back.  spin_ok: the fastener, whose turning about its own axis (local z, a
        free DOF a concentric may take) is not a move (27's ring screws: 0.585 every time)."""
        import swlib
        import swmate
        from swlib import c, wrap, sld
        T = {"coincident": c.swMateCOINCIDENT, "concentric": c.swMateCONCENTRIC,
             "distance": c.swMateDISTANCE}[kind]
        ref = {n: swlib.placement(cp) for n, cp in watch.items()}
        sm = wrap(self.doc.SelectionManager, sld.ISelectionMgr)
        last = "?"
        tries = [(al, False) for al in (c.swMateAlignCLOSEST, c.swMateAlignALIGNED, c.swMateAlignANTI_ALIGNED)]
        if kind == "distance":
            tries += [(al, True) for al, _ in tries]
        d = dist / 1000.0
        for align, flip in tries:
            before = {s.Name for s in self.mates()}
            self.doc.ClearSelection2(True)
            for i, f in enumerate((fa, fb)):
                sd = wrap(sm.CreateSelectData(), sld.ISelectData)
                sd.Mark = 1
                if not wrap(f, sld.IEntity).Select4(i > 0, sd):
                    raise RuntimeError(f"{name}: cannot select face {i}")
            res = self.asm.AddMate5(T, align, flip, d, d, d, 0, 0, 0, 0, 0, False, False, 0)
            mate, err = res if isinstance(res, tuple) else (res, None)
            self.doc.ClearSelection2(True)
            self.doc.EditRebuild3()
            new = [s for s in self.mates() if s.Name not in before]

            def dev(n, cp):
                P = swlib.placement(cp)
                if n == spin_ok:                         # axis + origin only
                    return np.abs(P[:, 2:] - ref[n][:, 2:]).max()
                return np.abs(P - ref[n]).max()
            moved = max((dev(n, cp) for n, cp in watch.items()), default=0)
            if mate is not None and err in (None, 1) and len(new) == 1 and new[0].GetErrorCode2()[0] == 0 \
                    and moved < 1e-3:
                new[0].Name = name
                return True, "ok"
            last = f"err {err}, {len(new)} new, moved {moved:.4f}"
            for s in new:                                      # gotcha 43: a refused one can linger
                s.Select2(False, 0)
                self.doc.Extension.DeleteSelection2(0)
            self.doc.EditRebuild3()
            for n, cp in watch.items():                        # gotcha 44: put back by hand
                if np.abs(swlib.placement(cp) - ref[n]).max() > 1e-4:
                    swmate.set_placement(self.sw, cp, ref[n])
            self.doc.EditRebuild3()
        return False, last


def _faces_of(cp, M):
    """(kind, face, normal / axis (this assembly), offset or point, r, area) of a small part."""
    from swlib import c, wrap, sld
    out = []
    r = cp.GetBodies3(c.swSolidBody)
    for b in (r[0] if isinstance(r, tuple) else r) or []:
        for f in wrap(b, sld.IBody2).GetFaces() or []:
            f = wrap(f, sld.IFace2)
            s = wrap(f.GetSurface(), sld.ISurface)
            if s.IsPlane():
                p = s.PlaneParams
                n = np.array(p[:3]) * (-1 if f.FaceInSurfaceSense() else 1)
                n = M[:, :3] @ n
                out.append(("plane", f, n, float(n @ (M[:, :3] @ (np.array(p[3:6]) * 1e3) + M[:, 3])), None,
                            f.GetArea() * 1e6))
            elif s.IsCylinder():
                p = s.CylinderParams
                out.append(("cyl", f, M[:, :3] @ np.array(p[3:6]), M[:, :3] @ (np.array(p[:3]) * 1e3) + M[:, 3],
                            p[6] * 1e3, f.GetArea() * 1e6))
    return out


def build(only=None, dry=False, plan_file="plan.json"):
    """Insert every planned fastener (plan.json) of the target assemblies `only`
    (v5-relative, default all), mate it, check.  Saves nothing (16 --chain)."""
    import swlib
    import swmate
    from swlib import c, wrap, sld
    with open(os.path.join(OUT, plan_file)) as f:
        P = json.load(f)
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rc = swlib.components(robot)
    by = {}
    for it in P["add"]:
        if only is None or it["asm"] in only:
            by.setdefault(it["asm"], []).append(it)
    log = []
    for rel, items in by.items():
        path = os.path.normcase(os.path.join(swlib.V5, rel))
        inst = [items[0]["inst"]] if "inst" in items[0] else \
            sorted(n for n, cp in rc.items() if os.path.normcase(cp.GetPathName() or "") == path
                   and not n.split("/")[0].startswith("Mirror") and not cp.IsSuppressed())
        if not inst:
            raise SystemExit(f"{rel}: no live instance in ROBOT")
        T = swlib.placement(rc[inst[0]])
        prefix = inst[0] + "/"
        print(f"\n== {rel}  (instance {inst[0]}), {len(items)} fasteners", flush=True)
        for mrel in sorted({it["model"] for it in items}):          # AddComponent5 wants it loaded
            sw.OpenDoc6(os.path.join(swlib.V5, mrel), c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        A = Asm(sw, rel, T)
        existing = {s.Name for s in A.mates()}
        placed = {}                                  # fasteners already in this assembly, by model
        for x in A.asm.GetComponents(True) or []:
            cp = wrap(x, sld.IComponent2)
            placed.setdefault(os.path.normcase(cp.GetPathName() or ""), []).append(cp)
        for it in items:
            tag = f"{MPFX}{it['group']}_L{it['line']}"
            S_ = A.local(it["seat"])
            D = A.local(None, it["dir"])
            M = fastener_M(it, S_, D)
            mpath = os.path.normcase(os.path.join(swlib.V5, it["model"]))
            cp = next((x for x in placed.get(mpath, [])
                       if np.abs(swlib.placement(x)[:, 3] - M[:, 3]).max() < 0.01
                       and swlib.placement(x)[:, 2] @ M[:, 2] > 0.99999), None)
            if cp is not None:                       # placed by an earlier run: fix its configuration
                if cp.ReferencedConfiguration != it["config"]:
                    old = cp.ReferencedConfiguration
                    cp.ReferencedConfiguration = it["config"]
                    A.doc.EditRebuild3()
                    print(f"  ~ {tag}: {cp.Name2} configuration {old} -> {cp.ReferencedConfiguration}")
                if f"{tag}_conc" in existing and f"{tag}_seat" in existing:
                    print(f"  = {tag} ({cp.Name2} {cp.ReferencedConfiguration}, mated already)")
                    log.append((tag, "ok", f"{cp.Name2}; already in"))
                    continue
            elif f"{tag}_conc" in existing or f"{tag}_seat" in existing:
                raise SystemExit(f"{tag}: mates exist but no fastener sits at the planned place")
            if cp is not None:                       # placed, not mated: out of the rays' way
                cp.Visible = 0
            B = A.local(it["bore_mid"])
            u = _frame(D)[:, 0]
            bore_leaf = it["bore"].split("/")[-1]
            seat_leaf = it["seat_part"].split("/")[-1]
            fb, cb, why = A.ray(B, u, bore_leaf)                     # bore wall, from its axis outwards
            if fb is not None:
                s = wrap(fb.GetSurface(), sld.ISurface)
                if not s.IsCylinder() or abs(s.CylinderParams[6] * 1e3 - it["bore_r"]) > 0.02:
                    fb, why = None, "bore ray hit a face that is not the bore"
            # seat, from just above it: the plane facing the head AT the seat.  Round the
            # axis and in towards the bore until one hits (a clamp's seat is a narrow flat
            # between fillets: one direction at one radius can land on a fillet).
            v = _frame(D)[:, 1]
            fs, why2 = None, "seat ray hit no plane facing the head at the seat"
            # a domed seat: the plane is the reference part's face at ref (distance mate)
            Q, leaf, radii = (S_, seat_leaf, (it["probe"], (it["probe"] + it["bore_r"]) / 2, it["bore_r"] + 0.25)) \
                if "ref" not in it else (A.local(it["ref"]["point"]), it["ref"]["part"].split("/")[-1],
                                         (it["ref"]["r"],))
            for rr in radii:
                for k in range(8):
                    w = np.cos(k * np.pi / 4) * u + np.sin(k * np.pi / 4) * v
                    f_, c_, _ = A.ray(Q + 0.3 * D + w * rr, -D, leaf)
                    if f_ is None:
                        continue
                    s = wrap(f_.GetSurface(), sld.ISurface)
                    if not s.IsPlane():
                        continue
                    Ms = swlib.placement(c_)
                    pp = s.PlaneParams
                    n = (Ms[:, :3] @ np.array(pp[:3])) * (-1 if f_.FaceInSurfaceSense() else 1)
                    off = n @ (Ms[:, :3] @ (np.array(pp[3:6]) * 1e3) + Ms[:, 3])
                    if n @ D > 0.9999 and abs(off - Q @ n) < 0.02:
                        fs, cs = f_, c_
                        break
                if fs is not None:
                    break
            if fb is not None and fs is not None and "thread" in it:   # it must screw INTO something
                th = it["thread"]
                f_, c_, why = A.ray(A.local(th["mid"]), u, th["leaf"])
                s = wrap(f_.GetSurface(), sld.ISurface) if f_ is not None else None
                if s is None or not s.IsCylinder() or abs(s.CylinderParams[6] * 1e3 - th["r"]) > 0.05:
                    fb, why = None, f"no {th['leaf']} tap bore r{th['r']} on this axis ({why})"
            if cp is not None:
                cp.Visible = 1
            if fb is None or fs is None:
                log.append((tag, "REFUSED", why if fb is None else why2))
                print(f"  ! {tag}: {why if fb is None else why2}", flush=True)
                continue
            if dry:
                print(f"  ok {tag}: bore on {cb.Name2}, seat on {cs.Name2}")
                continue
            if cp is None:
                p0 = M[:, 3] / 1000.0
                cp = wrap(A.asm.AddComponent5(os.path.join(swlib.V5, it["model"]),
                                              c.swAddComponentConfigOptions_CurrentSelectedConfig, "", False,
                                              it["config"], *p0), sld.IComponent2)
                if cp is None:
                    log.append((tag, "REFUSED", "AddComponent5 failed"))
                    print(f"  ! {tag}: AddComponent5 failed", flush=True)
                    continue
                # AddComponent5 IGNORES ExistingConfigName: the component comes in with the
                # model's active configuration (measured: every M3 Roundhead came in 6mm)
                cp.ReferencedConfiguration = it["config"]
                swmate.set_placement(sw, cp, M)
                A.doc.EditRebuild3()
            if cp.ReferencedConfiguration != it["config"]:
                log.append((tag, "WRONG CONFIG", f"{cp.Name2} is {cp.ReferencedConfiguration}"))
                print(f"  ! {tag}: {cp.Name2} is {cp.ReferencedConfiguration}, wanted {it['config']}", flush=True)
                continue
            err = np.abs(swlib.placement(cp) - M).max()
            F = _faces_of(cp, swlib.placement(cp))
            want_n = D if it["model"] in FLAT else -D
            cyl = [x for x in F if x[0] == "cyl" and abs(x[4] - SHANK_R[it["model"]]) < 0.01
                   and abs(abs(x[2] @ D) - 1) < 1e-6]
            pl = [x for x in F if x[0] == "plane" and x[2] @ want_n > 0.9999 and abs(x[3] - S_ @ want_n) < 0.01]
            if not cyl or not pl or err > 1e-3:
                log.append((tag, "PLACED, NOT MATED", f"placement err {err:.4f}, shank {len(cyl)}, face {len(pl)}"))
                print(f"  ! {tag}: placed, no mate (err {err:.4f}, shank {len(cyl)}, face {len(pl)})", flush=True)
                continue
            watch = {cp.Name2: cp, cb.Name2: cb, cs.Name2: cs}
            # a partial earlier run (killed between the two mates) leaves one: add only the other
            ok1, w1 = (True, "exists") if f"{tag}_conc" in existing else \
                A.mate(f"{tag}_conc", "concentric", max(cyl, key=lambda x: x[5])[1], fb, watch, spin_ok=cp.Name2)
            if f"{tag}_seat" in existing:
                ok2, w2 = True, "exists"
            elif "ref" in it:
                ok2, w2 = A.mate(f"{tag}_seat", "distance", max(pl, key=lambda x: x[5])[1], fs, watch,
                                 dist=it["ref"]["dist"])
            else:
                ok2, w2 = A.mate(f"{tag}_seat", "coincident", max(pl, key=lambda x: x[5])[1], fs, watch)
            st = "ok" if ok1 and ok2 else "MATE FAILED"
            log.append((tag, st, f"{cp.Name2}; conc {w1}; seat {w2}"))
            print(f"  {'+' if st == 'ok' else '!'} {tag:40s} {cp.Name2:28s} {it['config']:5s} "
                  f"conc {w1}, seat {w2}", flush=True)
        A.doc.EditRebuild3()
        errs = swmate.mate_errors(A.doc)
        print(f"  {rel}: mate errors {errs if errs else 0}", flush=True)
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.EditRebuild3()
    bad = [x for x in log if x[1] != "ok"]
    print(f"\n{len(log) - len(bad)} fasteners in and mated; {len(bad)} not:" +
          "".join(f"\n  {x}" for x in bad))
    with open(os.path.join(OUT, "build_log.json"), "a") as f:
        json.dump(log, f)
        f.write("\n")


# ---- stand-ins (SolidWorks): the bracket flatheads that double a longer screw -----------
CB = r"Box\CornerBracket.SLDASM"
CB_CONFIGS = {(1,): "NoFH1", (2,): "NoFH2", (1, 2): "NoFH12"}


def feature_errors(doc):
    """Non-mate features in error (mirror components, patterns): (name, type, code, warning)."""
    import swstyle as S
    out = []
    for f in S._iter_features(doc):
        try:
            code, warn = f.GetErrorCode2()
        except Exception:
            continue
        if code and not f.IsSuppressed2(1, None)[0]:
            out.append((f.Name, f.GetTypeName2(), code, bool(warn)))
    return out


def remate_on_bracket(sw, d, s, cn, dc):
    """Mate `s` (a concentric on the stand-in flathead `cn`) -> the same concentric
    on the bracket part's coaxial bore.  The other face is taken from the mate
    itself.  Refused (and the old mate restored by reload) if anything moves."""
    import swlib
    import swmate
    from swlib import c, wrap, sld
    m = wrap(s.GetSpecificFeature2(), sld.IMate2)
    if m.Type != c.swMateCONCENTRIC:
        raise SystemExit(f"{s.Name}: not a concentric (type {m.Type}) -- re-make by hand")
    other = None
    for i in range(m.GetMateEntityCount()):
        me = m.MateEntity(i)
        rc_ = wrap(me.ReferenceComponent, sld.IComponent2)
        if rc_ is not None and rc_.Name2 != cn:
            other = (wrap(me.Reference, sld.IFace2), rc_)
    fh = dc[cn]
    Mf = swlib.placement(fh)
    ax, p0 = Mf[:, 2], Mf[:, 3]
    br = dc[cn.rsplit("/", 1)[0] + "/CornerBracket-1"]
    bore = None
    for kind, f, a, p, r, area in _faces_of(br, swlib.placement(br)):
        if kind == "cyl" and 1.5 < r < 1.8 and abs(abs(a @ ax) - 1) < 1e-6:
            q = p - p0
            if np.linalg.norm(q - (q @ ax) * ax) < 0.01:
                bore = f
    if other is None or bore is None:
        raise SystemExit(f"{s.Name}: cannot find the faces (other {other is not None}, bracket bore {bore is not None})")
    name = s.Name
    top = cn.split("/")[0]
    watch = {top: dc[top], br.Name2: br, other[1].Name2: other[1]}
    ref = {n: swlib.placement(cp) for n, cp in watch.items()}
    d.ClearSelection2(True)
    s.Select2(False, 0)
    d.Extension.DeleteSelection2(0)
    d.ClearSelection2(True)
    A = Asm.__new__(Asm)                      # the open document, no reopen
    A.sw, A.doc, A.asm = sw, d, wrap(d, sld.IAssemblyDoc)
    ok, why = A.mate(name, "concentric", bore, other[0], watch)
    moved = max(np.abs(swlib.placement(cp) - ref[n]).max() for n, cp in watch.items())
    print(f"  {name}: re-made on {br.Name2}'s bore -> {why}, moved {moved:.5f} mm")
    if not ok or moved > 1e-3:
        raise SystemExit(f"{name}: re-mate failed -- reload {d.GetTitle()} from disk (nothing saved)")


def standins(dry=False):
    """CornerBracket.SLDASM gets one configuration per set of flatheads to drop
    (the flathead suppressed in it); each STAND_IN bracket instance is switched to
    it.  Refuses first if any mate in an open v5 assembly references one of the
    flatheads that would go (it would break)."""
    import swlib
    import swmate
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rc = swlib.components(robot)
    drop = {}                                  # ROBOT path of each flathead that goes
    for inst, fhs in STAND_IN.items():
        if inst not in rc:
            raise SystemExit(f"no component {inst}")
        for k in fhs:
            drop[f"{inst}/M3x8 flathead-{k}"] = inst
    # 1. no mate may hold one of them
    held = []
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if d.GetType() != c.swDocASSEMBLY or not os.path.normcase(d.GetPathName()).startswith(
                os.path.normcase(swlib.V5)):
            continue
        pre = {os.path.normcase(swlib.ASM): "", os.path.normcase(os.path.join(swlib.V5, BOX)): "Box-1/",
               os.path.normcase(os.path.join(swlib.V5, BPWA)): "Box-1/BottomPanelWithAvionics-1/"}.get(
            os.path.normcase(d.GetPathName()))
        if pre is None:
            continue
        for s in swmate.mates_of(d):
            for cn in swmate.mate_components(s):
                if pre + cn in drop:
                    held.append((d.GetTitle(), s.Name, cn))
    # a live mate on a stand-in is re-made on the BRACKET's own bore (coaxial with the
    # screw: the same constraint); one whose other side is suppressed (the v3 panels)
    # is dormant and stays as it is
    live = []
    for doc_title, mname, cn in held:
        d = next(wrap(x, sld.IModelDoc2) for x in sw.GetDocuments() if wrap(x, sld.IModelDoc2).GetTitle() == doc_title)
        s = wrap(wrap(d, sld.IAssemblyDoc).FeatureByName(mname), sld.IFeature)
        others = [o for o in swmate.mate_components(s) if o != cn]
        dc = {wrap(x, sld.IComponent2).Name2: wrap(x, sld.IComponent2)
              for x in wrap(d, sld.IAssemblyDoc).GetComponents(False) or []}
        if any(dc[o].IsSuppressed() for o in others if o in dc):
            print(f"  {mname}: dormant (its other side {others} is suppressed) -- left as is")
        else:
            live.append((d, s, cn, dc))
    print(f"  {len(drop)} stand-in flatheads; {len(live)} live mate(s) to re-make on the bracket bore")
    if dry:
        return
    for d, s, cn, dc in live:
        remate_on_bracket(sw, d, s, cn, dc)
    # 2. the configurations
    cbd = wrap(sw.OpenDoc6(os.path.join(swlib.V5, CB), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)[0],
               sld.IModelDoc2)
    sw.ActivateDoc3(cbd.GetTitle(), False, 0, 0)
    cm = cbd.ConfigurationManager
    base = wrap(cm.ActiveConfiguration, sld.IConfiguration).Name
    names = list(cbd.GetConfigurationNames() or [])
    print(f"  CornerBracket configurations {names}, active {base}")
    for fhs, nm in CB_CONFIGS.items():
        if nm not in names:
            if cm.AddConfiguration2(nm, "flathead(s) %s suppressed: a longer screw from outside "
                                        "uses that hole (25_fasteners)" % (fhs,), "", 0, "", "", True) is None:
                raise SystemExit(f"AddConfiguration2 {nm} failed")
        cbd.ShowConfiguration2(nm)
        for k in (1, 2):
            cp = None
            for x in wrap(cbd, sld.IAssemblyDoc).GetComponents(True) or []:
                x = wrap(x, sld.IComponent2)
                if x.Name2 == f"M3x8 flathead-{k}":
                    cp = x
            cp.SetSuppression2(c.swComponentSuppressed if k in fhs else c.swComponentFullyResolved)
        cbd.EditRebuild3()
        print(f"    {nm}: " + ", ".join(f"{x.Name2} {'suppressed' if x.IsSuppressed() else 'in'}"
                                       for x in (wrap(y, sld.IComponent2) for y in
                                                 wrap(cbd, sld.IAssemblyDoc).GetComponents(True))
                                       if "flathead" in x.Name2))
    cbd.ShowConfiguration2(base)
    cbd.EditRebuild3()
    # 3. the instances, in the assembly that holds each
    for rel, pre in ((BOX, "Box-1/"), (BPWA, "Box-1/BottomPanelWithAvionics-1/")):
        d = wrap(sw.OpenDoc6(os.path.join(swlib.V5, rel), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)[0],
                 sld.IModelDoc2)
        sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
        top = {wrap(x, sld.IComponent2).Name2: wrap(x, sld.IComponent2)
               for x in wrap(d, sld.IAssemblyDoc).GetComponents(True) or []}
        for inst, fhs in STAND_IN.items():
            if not inst.startswith(pre) or "/" in inst[len(pre):]:
                continue
            cp = top[inst[len(pre):]]
            cp.ReferencedConfiguration = CB_CONFIGS[tuple(sorted(fhs))]
            print(f"    {rel}: {cp.Name2} -> {cp.ReferencedConfiguration}")
        d.EditRebuild3()
        print(f"    {rel}: mate errors {swmate.mate_errors(d)}, feature errors {feature_errors(d)}")
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.EditRebuild3()
    rc = swlib.components(robot)
    gone = [n for n in rc if n.endswith(("M3x8 flathead-1", "M3x8 flathead-2")) and rc[n].IsSuppressed()]
    print(f"  ROBOT: {len(gone)} bracket flatheads now suppressed (mirror instances included):")
    for n in sorted(gone):
        print(f"     {n}")
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}")


# ---- the left side (SolidWorks): the robot's mirror features, all of them live ----------
LEFT = ["MirrorFemur-4", "MirrorBODY-2", "MirrorCOUPLER-2", "MirrorFEMUR_INSIDE-2", "MirrorTibia-2"]
LEFT_BRACKETS = ["CornerBracket-9", "CornerBracket-10", "CornerBracket-11", "CornerBracket-12",
                 "CornerBracket-17", "CornerBracket-18"]       # Box: the left mirror instances, hidden


def left(report_only=False):
    """Unsuppress the left leg (the ROBOT mirror features' instances) and show the
    Box's hidden left brackets; report what each mirrored sub-assembly holds against
    its right-hand source, so missing fasteners show up."""
    import swlib
    import swmate
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    asm = wrap(robot, sld.IAssemblyDoc)
    top = {wrap(x, sld.IComponent2).Name2: wrap(x, sld.IComponent2) for x in asm.GetComponents(True) or []}
    if not report_only:
        for n in LEFT:
            cp = top[n]
            if cp.IsSuppressed():
                r = cp.SetSuppression2(c.swComponentFullyResolved)
                print(f"  {n}: unsuppressed ({r})")
        box = wrap(sw.OpenDoc6(os.path.join(swlib.V5, BOX), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)[0],
                   sld.IModelDoc2)
        bt = {wrap(x, sld.IComponent2).Name2: wrap(x, sld.IComponent2)
              for x in wrap(box, sld.IAssemblyDoc).GetComponents(True) or []}
        for n in LEFT_BRACKETS:
            if not bt[n].Visible:
                bt[n].Visible = 1
                print(f"  Box {n}: shown")
        sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
        robot.EditRebuild3()
    rc = swlib.components(robot)
    for src, mir in (("Femur-1", "MirrorFemur-4"), ("BODY-1", "MirrorBODY-2"), ("COUPLER-1", "MirrorCOUPLER-2"),
                     ("FEMUR_INSIDE-1", "MirrorFEMUR_INSIDE-2"), ("Tibia-1", "MirrorTibia-2")):
        def kinds(pre):
            k = {}
            for n, cp in rc.items():
                if n.startswith(pre + "/") and not cp.IsSuppressed() and not cp.GetChildren():
                    f = os.path.splitext(os.path.basename(cp.GetPathName() or "?"))[0]
                    k[f] = k.get(f, 0) + 1
            return k
        a, b = kinds(src), kinds(mir)
        diff = {f: (a.get(f, 0), b.get(f, 0)) for f in set(a) | set(b)
                if a.get(f, 0) != b.get(f, 0) and not f.startswith("Mirror")}
        print(f"  {src:15s} {sum(a.values()):3d} parts | {mir:22s} {sum(b.values()):3d} parts"
              + (f"  differ: {diff}" if diff else ""))
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}, feature errors {feature_errors(robot)}")


# The left leg's fasteners the mirror features did not carry over, inserted directly into
# the left sub-assemblies.  (Appending them to BodyMirror with ModifyDefinition DELETED
# MirrorBODY-2 from ROBOT, 2026-10-08 -- reverted by reloading ROBOT; do not retry.)
# Each left screw = its right twin carried by the part its holes are in: a rigid copy
# where the left part is an INSTANCE of the right one (rotated), the robot's mirror
# (z -> -z) where it is an opposite-hand part.  `thread`: the part it screws into --
# the build refuses the screw unless a tap bore of that part is on its axis.
# Measured 2026-10-08 (the thread check): on the left the AK45 ROTOR instance is turned
# about its axis so none of its taps is behind the femur's holes, and the LimitSwitch
# instance sits ~2.45 mm along its screw axis off the mirrored mount's taps (an instance
# is placed by its origin, which is not its mid-plane).  Both are the existing mirror
# set-up, not the screws (physically the rotor's angle is set when the femur is bolted on,
# and the switch is one bought part).  Those screws follow the part that is SEEN --
# the femur, the switch -- with no thread check (`thread_check: False`), and are reported.
LEFT_GROUPS = {
    "femur-rotor": dict(R="Femur-1/Femur-1", L="MirrorFemur-4/Femur-1", asm=r"Links\MirrorFemur.SLDASM",
                        inst="MirrorFemur-4", thread=("AK45-10 Rotor-1", "AK45-10 Rotor-1"), thread_check=False),
    "stator-sidepanel": dict(R="BODY-1/SIDE PANEL-1/Side panel-1",
                             L="MirrorBODY-2/MirrorSIDE PANEL-1/MirrorSide panel-1",
                             asm=r"Body\MirrorSIDE PANEL.SLDASM", inst="MirrorBODY-2/MirrorSIDE PANEL-1",
                             thread=("AK45-10 Stator-1", "AK45-10 Stator-1")),
    "limit-switch": dict(R="BODY-1/SIDE PANEL-1/SwicthMount-1/LimitSwitch-2",
                         L="MirrorBODY-2/MirrorSIDE PANEL-1/MirrorSwicthMount-1/LimitSwitch-1",
                         asm=r"Body\LimitSwitch\MirrorSwicthMount.SLDASM",
                         inst="MirrorBODY-2/MirrorSIDE PANEL-1/MirrorSwicthMount-1",
                         thread=("Switch_Mount-1", "MirrorSwitch_Mount-1"), thread_check=False),
}


def plan_left():
    """plan.json's right-leg items of LEFT_GROUPS -> their left twins -> plan_left.json."""
    import swlib
    with open(os.path.join(OUT, "plan.json")) as f:
        P = json.load(f)
    with open(os.path.join(OUT, "lines.json")) as f:
        rows = {r["id"]: r for r in json.load(f)}
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rc = swlib.components(robot)
    S_ = np.diag([1.0, 1.0, -1.0])                            # the robot's mirror plane: Front Plane, z = 0
    out = []
    for it in P["add"]:
        g = LEFT_GROUPS.get(it["group"])
        if g is None:
            continue
        cr, cl = rc[g["R"]], rc[g["L"]]
        same = os.path.normcase(cr.GetPathName()) == os.path.normcase(cl.GetPathName())
        if same:                                              # an instance: rigid copy
            PR, PL = swlib.placement(cr), swlib.placement(cl)
            Rm = PL[:, :3] @ PR[:, :3].T
            X = lambda q: Rm @ (np.asarray(q) - PR[:, 3]) + PL[:, 3]
            V = lambda d: Rm @ np.asarray(d)
        else:                                                 # opposite hand: the robot's mirror
            X = lambda q: S_ @ np.asarray(q)
            V = lambda d: S_ @ np.asarray(d)
        r = rows[it["line"]]
        p, a = np.array(r["p"]), np.array(r["a"])
        tb = [b for b in r["bores"] if b["comp"].split("/")[-1] == g["thread"][0]]
        if not tb:
            raise SystemExit(f"{it['group']} L{it['line']}: no {g['thread'][0]} bore on the right")
        b = max(tb, key=lambda b: b["t"][1] - b["t"][0])
        q = dict(it)
        q.update(group=it["group"] + "-left", asm=g["asm"], inst=g["inst"],
                 seat=X(it["seat"]).round(5).tolist(), dir=V(it["dir"]).round(8).tolist(),
                 bore_mid=X(it["bore_mid"]).round(5).tolist(),
                 bore=g["L"].split("/")[-1], seat_part=g["L"].split("/")[-1],
                 how="rigid copy (instance)" if same else "mirrored (opposite hand)")
        if g.get("thread_check", True):
            q["thread"] = dict(leaf=g["thread"][1], r=b["r"],
                               mid=X(p + (b["t"][0] + b["t"][1]) / 2 * a).round(5).tolist())
        out.append(q)
        print(f"  {q['group']:24s} L{q['line']:3d} {q['how']:26s} seat {np.round(q['seat'], 2).tolist()}")
    with open(os.path.join(OUT, "plan_left.json"), "w") as f:
        json.dump(dict(add=out), f, indent=1)
    print(f"{len(out)} left fasteners -> plan_left.json")


# MirrorSIDE PANEL (the left body, made by BodyMirror) lost the coincident that holds
# InsideFemurShaft-1 along its axis (right: Coincident6 to the side panel).  Measured
# 2026-10-08 with the left leg live: the shaft sat 20 mm along the hip axis and the
# wrong way round, the RobotMount (coincident to it) and its M3 Roundhead-16 followed --
# the left box wall 10 mm INTO the box (137 interferences with the box), the left
# Femur_inside bearing off its shaft.  Fix: those three at the exact mirror of their
# right twins (S @ P_right @ m, m a reflection the part is symmetric under: the shaft
# about its local z (OCC: full volume), the mirror part about its own z, the screw about
# its x), their old mates deleted, each LOCKED to the left side panel.
LEFT_BODY_FIX = {   # MirrorSIDE PANEL component: (its right twin in SIDE PANEL, m)
    "InsideFemurShaft-1": ("InsideFemurShaft-1", (1, 1, -1)),
    "MirrorRobotMount-1": ("RobotMount-1", (1, 1, -1)),
    "M3 Roundhead-16": ("M3 Roundhead-16", (-1, 1, 1)),
}


def fix_left_body(dry=False):
    import swlib
    import swmate
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)

    def adoc(rel):
        d = wrap(sw.OpenDoc6(os.path.join(swlib.V5, rel), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)[0],
                 sld.IModelDoc2)
        return d, {wrap(x, sld.IComponent2).Name2: wrap(x, sld.IComponent2)
                   for x in wrap(d, sld.IAssemblyDoc).GetComponents(True) or []}
    R_, rcomp = adoc(r"Body\SIDE PANEL.SLDASM")
    L_, lcomp = adoc(r"Body\MirrorSIDE PANEL.SLDASM")
    S_ = np.diag([1.0, 1.0, -1.0])
    target = {}
    for ln, (rn, m) in LEFT_BODY_FIX.items():
        P = swlib.placement(rcomp[rn])
        R = S_ @ P[:, :3] @ np.diag(m)
        if abs(np.linalg.det(R) - 1) > 1e-6:
            raise SystemExit(f"{ln}: not a rotation")
        target[ln] = np.hstack([R, (S_ @ P[:, 3])[:, None]])
        now = swlib.placement(lcomp[ln])
        print(f"  {ln:22s} now t {now[:, 3].round(3)} -> {target[ln][:, 3].round(3)}; "
              f"rotation change {np.abs(now[:, :3] - R).max():.3f}")
    if dry:
        return
    sw.ActivateDoc3(L_.GetTitle(), False, 0, 0)
    gone = [s for s in swmate.mates_of(L_) if any(x in LEFT_BODY_FIX for x in swmate.mate_components(s))]
    print(f"  deleting {[s.Name for s in gone]}")
    for s in gone:
        L_.ClearSelection2(True)
        s.Select2(False, 0)
        L_.Extension.DeleteSelection2(0)
    L_.ClearSelection2(True)
    L_.EditRebuild3()
    others = {n: swlib.placement(cp) for n, cp in lcomp.items() if n not in LEFT_BODY_FIX}
    for ln, M in target.items():
        swmate.set_placement(sw, lcomp[ln], M)
    L_.EditRebuild3()
    for ln, M in target.items():
        err = np.abs(swlib.placement(lcomp[ln]) - M).max()
        print(f"  {ln:22s} placed, error {err:.5f}")
        if err > 1e-3:
            raise SystemExit(f"{ln} did not stay where it was put -- reload MirrorSIDE PANEL (nothing saved)")
    moved = [n for n, P in others.items() if np.abs(swlib.placement(lcomp[n]) - P).max() > 1e-3]
    print(f"  other components moved: {moved}")
    m_ = swmate.Mater(sw, L_, prefix="LF_")
    for ln in LEFT_BODY_FIX:
        m_.lock(f"{ln.replace(' ', '')}_lock_MirrorSidePanel", ln, "MirrorSide panel-1")
    L_.EditRebuild3()
    st = swmate.status(L_)
    print(f"  status: " + ", ".join(f"{n} {st.get(n)}" for n in LEFT_BODY_FIX) +
          f"; mate errors {swmate.mate_errors(L_)}")
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.EditRebuild3()
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}")


# The left leg's opposite-hand parts are DERIVED mirror parts of the styled ones: they carry
# the bodies but not the bodies' colours, so they showed their v4 part appearance (orange
# tibia, magenta side panel ...).  Colour only: each mirrored body takes its source body's
# colour (gotcha 18: a body colour beats the part's appearance).  Re-run after a restyle.
MIRROR_PARTS = {r"Links\MirrorFemur.SLDPRT": r"Links\Femur.SLDPRT",      # the true left femur since 27
                r"Links\MirrorTibia.SLDPRT": r"Links\Tibia.SLDPRT",
                r"Links\MirrorCoupler.SLDPRT": r"Links\Coupler.SLDPRT",
                r"Links\MirrorFemur_inside.SLDPRT": r"Links\Femur_inside.SLDPRT",
                r"Body\MirrorSide panel.SLDPRT": r"Body\Side panel.SLDPRT",
                r"Body\OldRobotBodyMount\MirrorRobotMount.SLDPRT": r"Body\OldRobotBodyMount\RobotMount.SLDPRT",
                r"Body\LimitSwitch\MirrorSwitch_Mount.SLDPRT": r"Body\LimitSwitch\Switch_Mount.SLDPRT"}


def trench_colours(sw, dry=False):
    """RobotMount: 24_pipes' trench Combine (PT_trench1/2) made new bodies WITHOUT a body
    colour -- the white plate and the graphite inlay the trench crosses (README
    'Corrugated pipes': 'the white plate and one graphite inlay') -- so both showed
    the part appearance (pale blue-white: the graphite inlay did not show).  Colour
    them back: the bigger one white, the other graphite."""
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    d = wrap(sw.OpenDoc6(os.path.join(swlib.V5, r"Body\OldRobotBodyMount\RobotMount.SLDPRT"), c.swDocPART,
                         c.swOpenDocOptions_Silent, "", 0, 0)[0], sld.IModelDoc2)
    bare = sorted((b for b in S.bodies(d) if b.MaterialPropertyValues2 is None), key=S.volume, reverse=True)
    if len(bare) != 2:
        print(f"  RobotMount: {len(bare)} uncoloured bodies (expected the 2 trench bodies) -- left as is")
        return
    for b, g in zip(bare, ("white", "graphite")):
        print(f"  RobotMount {b.Name}: {S.volume(b):.2f} mm3 -> {g}")
        if not dry:
            S.colour(b, S.GROUPS[g])
    d.EditRebuild3()


def mirror_colours(dry=False):
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    swlib.open_v5(sw)
    trench_colours(sw, dry)

    def doc(rel):
        return wrap(sw.OpenDoc6(os.path.join(swlib.V5, rel), c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)[0],
                    sld.IModelDoc2)

    def sig(b):
        """Volume (mm3) and surface area (mm2): unchanged by any mirror (the parts are
        mirrored about planes that need not pass through their origin)."""
        m = b.GetMassProperties(1.0)
        return m[3] * 1e9, np.array([m[4] * 1e6])

    done = []
    for mrel, srel in MIRROR_PARTS.items():
        src, mir = doc(srel), doc(mrel)
        sb = [(b, *sig(b)) for b in S.bodies(src)]
        mb = [(b, *sig(b)) for b in S.bodies(mir)]
        if len(sb) != len(mb):
            print(f"  ! {mrel}: {len(mb)} bodies, its source has {len(sb)} -- not coloured")
            continue
        used, pairs, worst, margin = set(), [], 0.0, np.inf
        rel = lambda i, v, a: abs(sb[i][1] - v) / max(v, 1e-9) + float(np.abs(sb[i][2] - a).sum()) / max(a.sum(), 1e-9)
        col = lambda i: tuple(np.round((sb[i][0].MaterialPropertyValues2 or [None] * 3)[:3], 3)) \
            if sb[i][0].MaterialPropertyValues2 is not None else None
        for b, v, a in sorted(mb, key=lambda x: -x[1]):            # big bodies first
            cand = sorted((rel(i, v, a), i) for i in range(len(sb)) if i not in used)
            k = cand[0][1]
            used.add(k)
            worst = max(worst, cand[0][0])                         # RELATIVE: mass properties of a
            # a near-twin only matters if it carries another colour  # 228 cm3 body differ by ~0.4 mm3
            rival = next((r for r, i in cand[1:] if col(i) != col(k)), None)
            if rival is not None:
                margin = min(margin, rival / max(cand[0][0], 1e-12))
            pairs.append((b, sb[k][0]))
        # the corrugated pipe bodies' many faces put ~2e-4 of noise on area (MirrorFemur): up to
        # 1e-3 is accepted when no pairing could have taken another colour (that rival 10x worse)
        if worst > 1e-3 or (worst > 1e-4 and margin < 10):
            print(f"  ! {mrel}: bodies do not match their source (worst {worst:.4f}, margin {margin:.1f}) "
                  f"-- not coloured")
            continue
        n = 0
        for b, s in pairs:
            rgb = s.MaterialPropertyValues2 or src.MaterialPropertyValues   # unstyled: the part's colour
            if rgb is not None:
                if not dry:
                    S.colour(b, list(rgb[:3]))
                n += 1
        mir.EditRebuild3()
        groups = {}
        for b in S.bodies(mir):
            if b.MaterialPropertyValues2 is not None:
                g = S.group_of(b)
                groups[g] = groups.get(g, 0) + 1
        print(f"  {mrel:48s} {len(mb):3d} bodies matched (worst relative {worst:.1e}), {n} coloured: {groups}")
        done.append(mrel)
    return done


# the ROBOT mirror feature that carries each right-leg sub-assembly over to the left
MIRROR_OF = {"Femur-1": "FemurMirror", "Tibia-1": "TibiaMirror", "BODY-1": "BodyMirror",
             "COUPLER-1": "CouplerMirror", "FEMUR_INSIDE-1": "MirrorComponent5"}


def mirror_fasteners(dry=False):
    """Every fastener of the right leg that its ROBOT mirror feature does not list
    yet is appended to that feature's INSTANCE list (as every other screw already
    is: a screw is its own mirror image), aligned to the component origin with the
    orientation the feature already uses.  Then: mate / feature errors, and each
    mirrored sub-assembly's part count against its source (`left --report`)."""
    import pythoncom
    from win32com.client import VARIANT
    import swlib
    import swmate
    from swlib import wrap, sld
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    asm = wrap(robot, sld.IAssemblyDoc)
    rc = swlib.components(robot)
    models = {os.path.basename(m).lower() for m in (R3, F3, R25, R2, SO3)}
    fast = {n for n, cp in rc.items() if os.path.basename(cp.GetPathName() or "").lower() in models
            and not n.startswith(("Mirror", "Box-1"))}
    for top, fname in MIRROR_OF.items():
        feat = wrap(asm.FeatureByName(fname), sld.IFeature)
        d = wrap(feat.GetDefinition(), sld.IMirrorComponentFeatureData)
        d.AccessSelections(robot, None)
        modified = False
        try:
            cur = [wrap(x, sld.IComponent2) for x in (d.ComponentsToInstanceAlignToComponentOrigin or [])]
            ori = list(d.ComponentOrientationsAlignToComponentOrigin or [])
            have = {x.Name2 for x in cur}
            add = sorted(n for n in fast if n.startswith(top + "/") and n not in have)
            print(f"  {fname:16s} lists {len(cur)} instances (orientations {sorted(set(ori))}); "
                  f"to add {len(add)}: {[a.split('/')[-1] for a in add]}")
            if dry or not add:
                continue
            o = max(set(ori), key=ori.count) if ori else 0
            comps = cur + [rc[n] for n in add]
            d.ComponentsToInstanceAlignToComponentOrigin = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH,
                                                                   [x._oleobj_ for x in comps])
            d.ComponentOrientationsAlignToComponentOrigin = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_I4,
                                                                    ori + [o] * len(add))
            modified = bool(feat.ModifyDefinition(d, robot, None))
            print(f"    ModifyDefinition -> {modified}")
        finally:
            if not modified:            # ModifyDefinition releases the access itself when it succeeds
                d.ReleaseSelectionAccess()
        robot.EditRebuild3()
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}, feature errors {feature_errors(robot)}")
    left(report_only=True)


# ---- audit (offline, OCC): is every screw of the robot, both sides, really in its hole? ----
# Model frame of each screw (probed 2026-10-08, IComponent2 bodies): axis local +Z towards
# the head.  hf = the head's bearing face (a flathead: its flush top), tip = the shank end,
# both as z(L); hr / hh head radius / height; sr shank radius.
SCREWS = {
    "m3 roundhead.sldprt": dict(hf=lambda L: L / 2, tip=lambda L: -L / 2, hr=2.85, hh=1.6, sr=1.5),
    "m4 roundhead.sldprt": dict(hf=lambda L: L / 2, tip=lambda L: -L / 2, hr=3.8, hh=2.2, sr=2.0),
    "m2.5 roundhead.sldprt": dict(hf=lambda L: 0.0, tip=lambda L: -L, hr=2.35, hh=1.3, sr=1.25),
    "m2 roundhead.sldprt": dict(hf=lambda L: 0.0, tip=lambda L: -L, hr=1.75, hh=1.1, sr=1.0),
    "m3 flathead.sldprt": dict(hf=lambda L: L / 2, tip=lambda L: -L / 2, hr=3.0, hh=1.7, sr=1.5, flat=True),
    "m3x8 flathead.sldprt": dict(hf=lambda L: 6.115, tip=lambda L: -6.2, hr=3.0, hh=1.7, sr=1.5, flat=True),
}
_ISEC = {}


def hits(key, M, p, a, t0, t1):
    """Material intervals of the line p + t a (assembly frame), t0..t1, of one part
    instance.  Depth-counted In/Out transitions, so the touching bodies of a styled
    part (an inlay on its plate) join into one interval."""
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.IntCurveSurface import IntCurveSurface_In, IntCurveSurface_Out
    from OCP.gp import gp_Pnt, gp_Lin, gp_Dir
    if key not in _ISEC:
        isec = IntCurvesFace_ShapeIntersector()
        isec.Load(shape(key), 1e-6)
        _ISEC[key] = isec
    isec = _ISEC[key]
    R, t = M[:, :3], M[:, 3]
    q = R.T @ (p + t0 * a - t)
    isec.Perform(gp_Lin(gp_Pnt(*q), gp_Dir(*(R.T @ a))), 0.0, t1 - t0)
    ev = sorted((isec.WParameter(i) + t0, +1 if isec.Transition(i) == IntCurveSurface_In else -1)
                for i in range(1, isec.NbPnt() + 1)
                if isec.Transition(i) in (IntCurveSurface_In, IntCurveSurface_Out))
    depth, low, cum = 0, 0, 0
    for _, s in ev:
        cum += s
        low = min(low, cum)
    depth = -low                                        # the line starts inside that many solids
    out, start = [], (t0 if depth > 0 else None)
    for w, s in ev:
        depth += s
        if depth > 0 and start is None:
            start = w
        elif depth <= 0 and start is not None:
            out.append((start, w))
            start = None
    if start is not None:
        out.append((start, t1))
    return out


def _union(ivs):
    out = []
    for a0, a1 in sorted(ivs):
        if out and a0 <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], a1))
        else:
            out.append((a0, a1))
    return out


def _len_in(ivs, lo, hi):
    return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in ivs)


def _inside(ivs, t):
    return any(a <= t <= b for a, b in ivs)


def audit(scan_file=SCAN_ALL):
    """Every screw in `scan_file` against every visible non-fastener part near it:
      buried     the head's volume is in material (a reversed screw: head in the hole)
      unseated   nothing under the head's bearing face (flathead: round its rim)
      sticks out mm of shank from the tip with no material round it (in air)
      in solid   mm of the screw's axis inside material (no hole there)
      doubled    another screw on the same axis, overlapping
    -> out/fasteners/audit.json + a table of the bad ones."""
    with open(scan_file) as f:
        sc = json.load(f)
    # hidden parts count too: hiding is a view state (he hides the hood to look inside)
    parts = [x for x in sc["comps"] if os.path.basename(x["file"]).lower() not in SCREWS and x.get("step")
             and sc["exported"].get(x["step"]) in ("ok", "cached")]
    for x in parts:
        x["Ma"] = np.array(x["M"])
        lo, hi = bbox(x["step"])
        x["box"] = (lo - 1, hi + 1)
    rows = []
    screws = [x for x in sc["comps"] if os.path.basename(x["file"]).lower() in SCREWS]
    for n_, x in enumerate(screws):
        s = SCREWS[os.path.basename(x["file"]).lower()]
        m = re.match(r"([\d.]+)mm", x["config"])
        L = float(m.group(1)) if m else 0.0
        M = np.array(x["M"])
        a = M[:, 2] / np.linalg.norm(M[:, 2])
        hf = M[:, :3] @ np.array([0, 0, s["hf"](L)]) + M[:, 3]          # t = 0: the head's bearing face
        Ls = s["hf"](L) - s["tip"](L)                                    # tip at t = -Ls
        t0, t1 = -Ls - 4.0, s["hh"] + 3.0
        seg = hf + np.linspace(t0, t1, 12)[:, None] * a
        near = []
        for y in parts:
            ql = (seg - y["Ma"][:, 3]) @ y["Ma"][:, :3]
            lo, hi = y["box"]
            if np.any(np.all((ql > lo - 4) & (ql < hi + 4), axis=1)):
                near.append(y)
        u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)

        def line_by(r, k, n=8):
            o = hf + r * (np.cos(2 * np.pi * k / n) * u + np.sin(2 * np.pi * k / n) * v)
            return {y["name"]: hits(y["step"], y["Ma"], o, a, t0, t1) for y in near}

        def line(r, k, n=8):
            return _union([iv for ivs in line_by(r, k, n).values() for iv in ivs])
        axb = line_by(0.0, 0)
        ax = _union([iv for ivs in axb.values() for iv in ivs])
        flat = s.get("flat", False)
        # head volume (a flathead's head is IN its countersink: only above its top counts)
        hz = (0.25, 1.2) if flat else (0.2, s["hh"] - 0.2)
        head_b = [line_by(0.65 * s["hr"], k) for k in range(8)] + [axb]
        buried = np.mean([_len_in(_union([iv for ivs in b.values() for iv in ivs]), *hz) / (hz[1] - hz[0])
                          for b in head_b])
        by_head = sorted({nm for b in head_b for nm, ivs in b.items() if _len_in(ivs, *hz) > 0.1})
        # seat: material just under the bearing face, between the shank and the head's rim;
        # a flathead: the countersink wall beside its cone, 0.8 under the top (cone r 2.2 there)
        seat = np.mean([_inside(line(s["hr"] - 0.3 if flat else
                                     min(s["hr"] - 0.25, (s["sr"] + s["hr"]) / 2 + 0.3), k),
                                -0.8 if flat else -0.3) for k in range(8)])
        # enclosure of the shank: lines just outside it
        enc = [line(s["sr"] + 0.7, k) for k in range(8)]
        ts = np.arange(-Ls, 0.0, 0.25)
        enclosed = np.array([sum(_inside(iv, t) for iv in enc) >= 3 for t in ts])
        out_mm = float(ts[np.argmax(enclosed)] + Ls) if enclosed.any() else Ls
        free_mm = float(np.sum(~enclosed) * 0.25)
        solid = _len_in(ax, -Ls + 0.3, 0.0 - 0.3)
        by_solid = sorted(nm for nm, ivs in axb.items() if _len_in(ivs, -Ls + 0.3, -0.3) > 0.3)
        rows.append(dict(name=x["name"], file=os.path.basename(x["file"]), config=x["config"],
                         hf=hf.round(3).tolist(), a=a.round(6).tolist(), Ls=round(Ls, 3),
                         buried=round(float(buried), 2), by_head=by_head, seated=round(float(seat), 2),
                         sticks_out=round(out_mm, 2), free=round(free_mm, 2), in_solid=round(solid, 2),
                         by_solid=by_solid, near=[y["name"] for y in near]))
        if n_ % 50 == 0:
            print(f"  {n_}/{len(screws)}", flush=True)
    # doubled: two screws on one axis whose shanks overlap
    for i, r in enumerate(rows):
        r["doubled"] = []
        for j, q in enumerate(rows):
            if i == j:
                continue
            a, b = np.array(r["a"]), np.array(q["a"])
            d = np.array(q["hf"]) - np.array(r["hf"])
            if abs(a @ b) > 0.9999 and np.linalg.norm(d - (d @ a) * a) < 0.5:
                tq = sorted([float(d @ a), float(d @ a) - q["Ls"] * float(a @ b)])
                if min(0.0, tq[1]) - max(-r["Ls"], tq[0]) > 0.5:
                    r["doubled"].append(q["name"])
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def bad(r):
        why = []
        if r["buried"] > 0.3:
            why.append(f"HEAD BURIED {r['buried']:.0%} in {','.join(n.split('/')[-1] for n in r['by_head'])}")
        if r["seated"] < 0.5:
            why.append(f"head not seated ({r['seated']:.0%})")
        if r["sticks_out"] > 1.0:
            why.append(f"tip sticks out {r['sticks_out']:.1f}")
        if r["in_solid"] > 0.5:
            why.append(f"axis in solid {r['in_solid']:.1f} ({','.join(n.split('/')[-1] for n in r['by_solid'])})")
        if r["doubled"]:
            why.append(f"doubled with {', '.join(d.split('/')[-1] for d in r['doubled'])}")
        return why
    nb = 0
    for r in sorted(rows, key=lambda r: r["name"]):
        why = bad(r)
        if why:
            nb += 1
            print(f"  {r['name']:62s} {r['file'][:-7]:15s} {r['config']:10s} " + "; ".join(why))
    print(f"{len(rows)} screws audited, {nb} with a problem -> audit.json")


# ---- BOM (SolidWorks, read-only): every fastener in the robot as it now is ---------------
BOM_NAMES = {   # file (lower) -> (what, how the config names a length)
    "m3 roundhead.sldprt": ("M3 button head (ISO 7380)", "cfg"),
    "m3 flathead.sldprt": ("M3 countersunk (ISO 10642)", "cfg"),
    "m3x8 flathead.sldprt": ("M3 countersunk (ISO 10642) -- file says x8, the model is 12 long", "12mm"),
    "m4 roundhead.sldprt": ("M4 button head (ISO 7380)", "cfg"),
    "m2.5 roundhead.sldprt": ("M2.5 button head (ISO 7380)", "cfg"),
    "m2 roundhead.sldprt": ("M2 button head", "cfg"),
    "m3 nut.sldprt": ("M3 hex nut", "-"),
    "m3 washer.sldprt": ("M3 washer 7 x 0.5", "-"),
    "m3 standoff mf.sldprt": ("M3 standoff, male-female (6 mm male)", "cfg"),
    "m3 standoff.sldprt": ("M3 standoff, male-female", "cfg"),
    "m2 standoff.sldprt": ("M2 standoff", "cfg"),
}


def bom():
    import collections
    import swlib
    from swlib import wrap, sld
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rows = collections.Counter()
    where = collections.defaultdict(collections.Counter)
    rc = swlib.components(robot)
    for n, cp in rc.items():
        parts = n.split("/")
        if parts[0] in SKIP_TOP or any(rc["/".join(parts[:i])].IsSuppressed() for i in range(1, len(parts) + 1)
                                       if "/".join(parts[:i]) in rc):
            continue                                  # it, or a parent of it, is suppressed
        f = os.path.basename(cp.GetPathName() or "").lower()
        if f not in BOM_NAMES:
            continue
        what, how = BOM_NAMES[f]
        cfg = cp.ReferencedConfiguration
        L = {"cfg": cfg.replace("mm", " mm") if cfg.endswith("mm") else cfg, "-": ""}.get(how, how)
        if f == "m2 standoff.sldprt" and cfg == "Default":
            L = "15 mm"                               # its Default configuration is 15 long
        rows[(what, L)] += 1
        where[(what, L)][parts[0]] += 1
    lines = ["| fastener | length | count | where (top-level assembly: count) |", "|---|---|---:|---|"]
    num = lambda L: float(re.sub(r"[^0-9.]", "", L) or 0)
    for (what, L), k in sorted(rows.items(), key=lambda kv: (kv[0][0], num(kv[0][1]))):
        w = ", ".join(f"{a}: {b}" for a, b in sorted(where[(what, L)].items()))
        lines.append(f"| {what} | {L} | {k} | {w} |")
    lines.append(f"| **total** | | **{sum(rows.values())}** | |")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "BOM.md"), "w", encoding="utf-8") as fh:
        fh.write("# Fastener BOM -- counted in ROBOT.SLDASM (25_fasteners.py bom)\n\n" + txt + "\n\n"
                 "Not in the model (open):\n" + "".join(f"* {k}: {v}\n" for k, v in LEFT_OUT.items()) +
                 "\nIn the model, to check:\n" + "".join(f"* {k}: {v}\n" for k, v in TO_CHECK.items()))
    print(txt)


# ---- models (SolidWorks): the fasteners the robot needs that v5 had no model for ----
# New parts in Common/, plain (no modelled thread).  ORIGIN AT THE HEAD'S BEARING FACE,
# +Z towards the head (a standoff: at its male-side face, +Z towards the female end),
# so a placement is "origin on the seat, +Z out of the hole".  One configuration per
# length; only the shank (hex) extrusion depth differs between them.
STEEL = (0.62, 0.64, 0.66)
BRASS = (0.80, 0.66, 0.32)
MODELS = {   # file: (kind, d, head dia / hex across flats, head height, socket AF, lengths)
    r"Common\M2.5 Roundhead.SLDPRT": ("screw", 2.5, 4.7, 1.3, 1.5, (12, 16)),    # ISO 7380 M2.5
    r"Common\M2 Roundhead.SLDPRT": ("screw", 2.0, 3.5, 1.1, 1.3, (4, 6, 8)),     # DIN 7380 style
    r"Common\M3 standoff MF.SLDPRT": ("standoff", 3.0, 5.5, 6.0, None, (5, 8, 12)),  # male 6, female through
}


def _hexagon(af, rot=0.0):
    import math
    from shapely.geometry import Polygon
    r = af / 2 / math.cos(math.pi / 6)
    return Polygon([(r * math.cos(rot + k * math.pi / 3), r * math.sin(rot + k * math.pi / 3)) for k in range(6)])


def _circle(m, r, name):
    """A TRUE circle on the Front Plane (swstyle.sketch would make a polygon, and a
    faceted shank has no cylinder to mate a concentric to)."""
    import swstyle as S
    m.ClearSelection2(True)
    if not m.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, False, 0, None, 0):
        raise SystemExit("cannot select Front Plane")
    sm = m.SketchManager
    sm.InsertSketch(True)
    sm.AddToDB = True
    try:
        if sm.CreateCircleByRadius(0, 0, 0, r / 1000.0) is None:
            raise SystemExit("CreateCircleByRadius failed")
    finally:
        sm.AddToDB = False
        sm.InsertSketch(True)
    sk = S.last_feature(m)
    sk.Name = name
    return sk


def build_model(sw, rel):
    import math
    from shapely.geometry import Point
    import swlib
    import swstyle as S
    from swlib import c, wrap, sld
    path = os.path.join(swlib.V5, rel)
    if os.path.exists(path):
        raise SystemExit(f"{rel} exists -- not overwritten")
    kind, d, big, h, sock, lengths = MODELS[rel]
    L0 = lengths[0]
    doc = sw.NewDocument(sw.GetUserPreferenceStringValue(c.swDefaultTemplatePart), 0, 0, 0)
    m = wrap(doc, sld.IModelDoc2)
    fm = m.FeatureManager
    if kind == "screw":
        sk = _circle(m, d / 2, "SK_shank")
        S.raised(m, sk, 0.0, L0, up=False, draft=0.0, name="Shank")
        sk = _circle(m, big / 2, "SK_head")
        dr = math.degrees(math.atan((big / 2 - 0.36 * big) / h))      # domed-looking button head
        S.raised(m, sk, 0.0, h, up=True, draft=dr, name="Head")
        sk = S.sketch(m, _hexagon(sock), "SK_socket")
        S.cut(m, sk, h - 0.6 * h, h + 0.5, name="Socket")
        drive = "Shank"
    else:
        sk = S.sketch(m, _hexagon(big, math.pi / 6), "SK_hex", grow=0)
        S.raised(m, sk, 0.0, L0, up=True, draft=0.0, name="Hex")
        sk = _circle(m, d / 2, "SK_male")
        S.raised(m, sk, 0.0, h, up=False, draft=0.0, name="Male")
        sk = _circle(m, 1.25, "SK_female")                                      # M3 tap drill
        S._select(m, sk)
        f = fm.FeatureCut4(True, False, True, c.swEndCondThroughAll, 0, 0, 0,
                           False, False, False, False, 0, 0, False, False, False, False,
                           False, True, True, False, False, False, c.swStartSketchPlane, 0, False, False)
        m.ClearSelection2(True)
        if f is None:
            raise SystemExit("female bore cut failed")
        wrap(f, sld.IFeature).Name = "Female"
        drive = "Hex"
    for b in S.bodies(m):
        S.colour(b, STEEL if kind == "screw" else BRASS)
    # configurations: one per length, only the drive feature's depth differs
    cm = m.ConfigurationManager
    wrap(cm.ActiveConfiguration, sld.IConfiguration).Name = f"{L0}mm"
    for L in lengths[1:]:
        if cm.AddConfiguration2(f"{L}mm", "", "", 0, "", "", True) is None:
            raise SystemExit(f"AddConfiguration2 {L}mm failed")
    set_lengths(m, rel)
    ok, err, warn = m.Extension.SaveAs3(path, c.swSaveAsCurrentVersion, c.swSaveAsOptions_Silent, None, None, 0, 0)
    if not ok:
        raise SystemExit(f"save {rel} failed ({err}, {warn})")
    print(f"  saved {rel}")


def set_lengths(m, rel):
    """Each configuration's drive depth = its length, set while that configuration
    is ACTIVE (swSetValue_InThisConfiguration).  The InSpecificConfigurations form
    with a Python list of names returned without changing anything (measured).
    Checked: one body per configuration, its extent along Z as designed."""
    import swstyle as S
    from swlib import c, wrap, sld
    kind, d, big, h, sock, lengths = MODELS[rel]
    for L in lengths:
        nm = f"{L}mm"
        if nm not in (m.GetConfigurationNames() or []):
            raise SystemExit(f"{rel}: no configuration {nm}")
        m.ShowConfiguration2(nm)                 # returns False when nm is already active
        # fetched AFTER the switch: a dimension object stays bound to the configuration
        # that was active when it was fetched (measured: the value went to the wrong one)
        dim = wrap(m.Parameter(f"D1@{'Shank' if kind == 'screw' else 'Hex'}"), sld.IDimension)
        dim.SetSystemValue3(L / 1000.0, c.swSetValue_InThisConfiguration, None)
        m.EditRebuild3()                         # ForceRebuild3 alone left the old geometry (measured)
        bs = S.bodies(m)
        box = [v * 1000 for v in bs[0].GetBodyBox()]
        lo, hi = (-L, h) if kind == "screw" else (-h, L)
        print(f"    {nm:6s} bodies {len(bs)}  {S.total_volume(m):8.2f} mm3  z {box[2]:.2f} .. {box[5]:.2f}")
        if len(bs) != 1 or abs(box[2] - lo) > 0.01 or abs(box[5] - hi) > 0.01:
            raise SystemExit(f"{rel} [{nm}]: wrong geometry (want z {lo} .. {hi})")
    m.ShowConfiguration2(f"{lengths[0]}mm")


def fix_lengths():
    """Re-apply set_lengths to the already-saved new models (open in SolidWorks) and save."""
    import swlib
    from swlib import c, wrap, sld
    sw, _ = swlib.connect()
    for rel in MODELS:
        p = os.path.join(swlib.V5, rel)
        doc, err, warn = sw.OpenDoc6(p, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        m = wrap(doc, sld.IModelDoc2)
        print(rel)
        set_lengths(m, rel)
        ok, err, warn = m.Save3(c.swSaveAsOptions_Silent, 0, 0)
        print(f"  {'saved' if ok else 'SAVE FAILED'}")


def models():
    import swlib
    sw, _ = swlib.connect()
    for rel in MODELS:
        print(rel)
        build_model(sw, rel)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    if cmd == "models":
        models()
    elif cmd == "fix-lengths":
        fix_lengths()
    elif cmd == "plan":
        plan()
    elif cmd == "standins":       # standins [--dry]
        standins("--dry" in args)
    elif cmd == "left":           # left [--report]
        left("--report" in args)
    elif cmd == "bom":
        bom()
    elif cmd == "mirror-fasteners":   # mirror-fasteners --dry   (the real run deleted MirrorBODY-2)
        if "--dry" not in args:
            raise SystemExit("refused: ModifyDefinition on BodyMirror deleted MirrorBODY-2 (2026-10-08). "
                             "Use plan-left + build --left.  --dry only lists what the features lack.")
        mirror_fasteners(True)
    elif cmd == "build":          # build [--dry] [--left] [v5-relative assembly ...]
        rest = [x for x in args[1:] if x not in ("--dry", "--left")]
        build(rest or None, "--dry" in args, "plan_left.json" if "--left" in args else "plan.json")
    elif cmd == "plan-left":
        plan_left()
    elif cmd == "mirror-colours":     # mirror-colours [--dry]
        mirror_colours("--dry" in args)
    elif cmd == "fix-left-body":      # fix-left-body [--dry]
        fix_left_body("--dry" in args)
    elif cmd == "scan":           # scan [--fresh] [--all]
        scan("--fresh" in args, "--all" in args)
    elif cmd == "audit":
        audit()
    elif cmd == "holes":
        holes()
    elif cmd == "lines":
        lines()
    elif cmd == "profile":         # profile <line id> ... [--rho 1.45,3.2]
        rh = [float(x) for x in args[args.index("--rho") + 1].split(",")] if "--rho" in args else [1.45]
        profile([int(x) for x in args[1:] if x.isdigit()], rh)
    else:
        print(__doc__)
