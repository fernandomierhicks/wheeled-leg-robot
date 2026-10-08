"""Step 21: the D FACET print box replaces the v3 box panels/top in the v5 robot.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/21_box_facet.py import
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/21_box_facet.py distance
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/21_box_facet.py install
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/21_box_facet.py check

  import    v5/Box/Facet/<Part>.step (from cad/aesthetics/parts/box_facet_print.py, built
            in the Distance1 = 75 robot frame) -> v5/Box/Facet/<Part>.SLDPRT
  reimport <Part>   a changed STEP into the EXISTING, open part, in place (lock-mated parts
            only; nothing closed, nothing saved -- `16 --chain` after); colours every body
            from the STEP (3D Interconnect drops STEP colour)
  colour <Part>     just that colouring, on the open part
  distance  ROBOT.SLDASM Distance1 80 -> 75: RobotMount inner faces at z +-75, the
            v3 box (150 wide) now meets BOTH side walls (his call, 2026-10-04: keep
            the floor, accept the 330 mm track)
  install   in Box.SLDASM, in memory (nothing is saved here):
              1. PLANE1 (the left side's mirror plane) re-anchored: it was the mid-
                 plane of FrontPanel-1's side faces, it becomes Front Plane offset
                 75 mm -- same place, no part under it
              2. the new parts inserted exactly where they were built
              3. every mate that touches an old part is recorded (geometry in the Box
                 frame), the old parts are suppressed (kept, his call), and each mate
                 is re-made on the new part's identical face: the new plates ARE the
                 v3 panels, the ring keeps the v3 Cap's screw holes
              4. bumpers -> panels, hood -> ring planned by swmate; strip locked to hood
  save-box  saves Box.SLDASM, refused unless every component is fully defined, 0 mate errors
  save-robot  saves ROBOT.SLDASM alone, refused unless styled parts are styled, no
            AI_HipDrive, Distance1 = 75, Box-1 fully defined, 0 mate errors
  check     constraint status + mate errors in Box and ROBOT
Order: import, install, fixup, save-box, (close all) distance, save-robot, check.  `install`
opens Box.SLDASM ALONE: with ROBOT open as well the rebuilds ran the GPU out of
memory and SolidWorks died (2026-10-06).
Every new mate is named FX_*, and is refused (rolled back) if it moves anything.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swmate as M
from swlib import c, wrap, sld

BOXD = os.path.join(swlib.V5, "Box")
BOXP = os.path.join(BOXD, "Box.SLDASM")
FACET = os.path.join(BOXD, "Facet")
PARTS = ["FacetFront", "FacetBack", "BumperFront", "BumperBack", "FacetRing", "FacetHood", "NeopixelStrip"]
OLD = {"BackPanel-1": "FacetBack-1", "FrontPanel-1": "FacetFront-1", "Cap-1": "FacetRing-1",
       "NeopixelSupport-1": None, "NeoPixelCage-1": None, "Neopixel-1": None, "Lid-1": None,
       "TPU_protector-1": None, "Cushion_support-1": None, "Cushion_support-2": None}
LID_SCREWS = [f"M3x8 flathead-{i}" for i in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
D75 = 75.0


def find_doc(sw, path):
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    return None


# Box-1's place in ROBOT once Distance1 = 75 (read back by `distance`: t = -150, 15, 75, no rotation)
P_BOX_D75 = np.array([[1.0, 0, 0, -150.0], [0, 1.0, 0, 15.0], [0, 0, 1.0, 75.0]])


def box_only(sw):
    """Box.SLDASM on its own -- NOT through ROBOT.  2026-10-06: with ROBOT and Box
    both open, the install's rebuilds ran the NVIDIA OpenGL driver out of memory
    and SolidWorks died mid-run (nothing saved, nothing lost).  Refuses if any
    component resolved outside v5 (gotcha 4)."""
    d = find_doc(sw, BOXP)
    if d is None:
        doc, err, warn = sw.OpenDoc6(BOXP, c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)
        d = find_doc(sw, BOXP)
        if d is None:
            raise SystemExit(f"could not open {BOXP} (err {err})")
    sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
    bad = [(n, cp.GetPathName()) for n, cp in M.comps(d).items()
           if cp.GetPathName() and not os.path.normcase(cp.GetPathName()).startswith(os.path.normcase(swlib.V5))]
    if bad:
        raise SystemExit(f"REFUSING: components outside v5: {bad[:5]}")
    return d


class quiet:
    """No graphics / feature-tree updates while a batch of API edits runs."""
    def __init__(self, sw, doc):
        self.sw, self.doc = sw, doc

    def __enter__(self):
        self.sw.CommandInProgress = True
        self.view = wrap(self.doc.ActiveView, sld.IModelView)
        if self.view is not None:
            self.view.EnableGraphicsUpdate = False
        self.doc.FeatureManager.EnableFeatureTree = False
        return self

    def __exit__(self, *exc):
        self.doc.FeatureManager.EnableFeatureTree = True
        if self.view is not None:
            self.view.EnableGraphicsUpdate = True
        self.sw.CommandInProgress = False
        return False


def box_doc(sw):
    swlib.open_v5(sw)
    d = find_doc(sw, BOXP)
    doc, err, warn = sw.OpenDoc6(BOXP, c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)   # its own window
    d = find_doc(sw, BOXP)
    sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
    return d


def features(doc):
    out = {}
    f = wrap(doc.FirstFeature(), sld.IFeature)
    while f is not None:
        out[f.Name.strip()] = f
        f = wrap(f.GetNextFeature(), sld.IFeature)
    return out


# ------------------------------------------------------------------ import
def do_import(sw):
    for n in PARTS:
        step = os.path.join(FACET, n + ".step")
        prt = os.path.join(FACET, n + ".SLDPRT")
        if os.path.exists(prt):
            print("  exists", os.path.relpath(prt, swlib.V5))
            continue
        res = sw.LoadFile4(step, "r", None, 0)
        d = wrap(res[0] if isinstance(res, tuple) else res, sld.IModelDoc2)
        if d is None or d.GetType() != c.swDocPART:
            raise SystemExit(f"{n}: STEP did not import as a part ({None if d is None else d.GetType()})")
        ok, err, warn = d.Extension.SaveAs3(prt, 0, c.swSaveAsOptions_Silent, None, None, 0, 0)
        bodies = wrap(d, sld.IPartDoc).GetBodies2(c.swSolidBody, False) or []
        vol = sum(wrap(b, sld.IBody2).GetMassProperties(1.0)[3] for b in bodies) * 1e9
        print(f"  {n}: {len(bodies)} bodies, {vol / 1000:.2f} cm3, saved {ok}")
        sw.CloseDoc(d.GetTitle())


def do_reimport(sw, name):
    """A Facet part's new geometry IN PLACE from its STEP (v5/Box/Facet/<name>.step), in the
    open part document -- for when `import` would need the file deleted, i.e. everything
    closed (2026-10-07: 72 documents open, dirty).  The file, the component and its mates
    stay; refused unless every mate on the component is a Lock (a lock has no face
    references, so new faces cannot break it -- the hood's two are).  Deletes the old
    import feature and any PP_/PT_ pipe features (`24 build` re-adds them), inserts the
    STEP (IPartDoc.InsertImportedFeature, 3D Interconnect), then checks the bodies
    against the STEP's solids by volume.  Saves nothing: run `16 --chain` after."""
    import swstyle as S
    from build123d import import_step
    prt, step = os.path.join(FACET, name + ".SLDPRT"), os.path.join(FACET, name + ".step")
    box = find_doc(sw, BOXP)
    if box is not None:
        for s in M.mates_of(box):
            if any(cn.split("/")[0] == name + "-1" for cn in M.mate_components(s)) and s.GetTypeName2() != "MateLock":
                raise SystemExit(f"REFUSING: {s.Name} ({s.GetTypeName2()}) references {name}-1's faces")
    doc = find_doc(sw, prt)
    if doc is None:
        sw.OpenDoc6(prt, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        doc = find_doc(sw, prt)
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    old = [f for f in S._iter_features(doc) if f.GetTypeName2() in ("MBimport", "BaseBody")
           or f.Name.startswith(("PP_", "PT_"))]
    v_old = sum(S.volume(b) for b in S.bodies(doc))
    doc.ClearSelection2(True)
    for k, f in enumerate(old):
        f.Select2(k > 0, 0)
    doc.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
    doc.ClearSelection2(True)
    doc.EditRebuild3()
    left = S.bodies(doc)
    if left:
        raise SystemExit(f"{name}: {len(left)} bodies left after deleting {[f.Name for f in old]} -- stopped")
    res = wrap(doc, sld.IPartDoc).InsertImportedFeature(step, 0)
    feat = wrap(res[0] if isinstance(res, tuple) else res, sld.IFeature)
    err = res[1] if isinstance(res, tuple) else None
    doc.EditRebuild3()
    got = sorted(S.volume(b) for b in S.bodies(doc))
    want = sorted(s.volume for s in import_step(step).solids())
    worst = max((abs(a - b) / max(1.0, 2e-6 * b / 0.05) for a, b in zip(got, want)), default=float("inf")) \
        if len(got) == len(want) else float("inf")
    print(f"  {name}: deleted {[f.Name for f in old]} ({v_old / 1000:.2f} cm3); inserted "
          f"{None if feat is None else feat.Name} (err {err}): {len(got)} bodies {sum(got) / 1000:.3f} cm3 vs the "
          f"STEP's {len(want)} {sum(want) / 1000:.3f} cm3, worst {worst:.4f} mm3  {'OK' if worst < 0.05 else 'FAIL'}")
    if worst >= 0.05:
        raise SystemExit(f"{name}: bodies do not match the STEP -- nothing saved; reload the part from disk to undo")
    colour_from_step(doc, step)
    if box is not None:
        sw.ActivateDoc3(box.GetTitle(), False, 0, 0)
        box.EditRebuild3()
        report(box)


def colour_from_step(doc, step):
    """Every body of the part coloured as its solid in the STEP: 3D Interconnect
    (InsertImportedFeature) brings the geometry but NOT the STEP's colours (2026-10-07:
    `24 export` read every body's colour from the part), so they are read with
    stepcolor.read and set per body, matched by centre of mass + volume."""
    import swstyle as S
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "aesthetics", "lib")))
    import stepcolor
    from build123d import Solid
    ref = []
    for s, rgb in stepcolor.read(step):
        so = Solid(s)
        c = so.center()
        ref.append((np.array([c.X, c.Y, c.Z]), so.volume, rgb))
    used, worst, n = set(), 0.0, {}
    for b in S.bodies(doc):
        mp = b.GetMassProperties(1.0)
        ctr, vol = np.array(mp[:3]) * 1000.0, mp[3] * 1e9
        k = min((i for i in range(len(ref)) if i not in used),
                key=lambda i: np.linalg.norm(ref[i][0] - ctr) + abs(ref[i][1] - vol) / max(vol, 1.0))
        d = np.linalg.norm(ref[k][0] - ctr)
        if d > 0.05 or abs(ref[k][1] - vol) > max(0.05, 2e-6 * vol) or ref[k][2] is None:
            raise SystemExit(f"colour: body {b.Name} ({vol:.3f} mm3) has no match in the STEP (nearest {d:.3f} mm)")
        used.add(k)
        worst = max(worst, d)
        S.colour(b, ref[k][2])
        key = tuple(round(x * 255) for x in ref[k][2])
        n[key] = n.get(key, 0) + 1
    doc.EditRebuild3()
    print(f"  coloured {sum(n.values())} bodies from the STEP {n}, worst centre {worst:.4f} mm")


def do_colour(sw, name):
    prt = os.path.join(FACET, name + ".SLDPRT")
    doc = find_doc(sw, prt)
    if doc is None:
        raise SystemExit(f"{name} is not open")
    sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
    colour_from_step(doc, os.path.join(FACET, name + ".step"))


# ------------------------------------------------------------------ Distance1
def do_distance(sw):
    model = swlib.open_v5(sw)
    asm = wrap(model, sld.IAssemblyDoc)
    feat = wrap(asm.FeatureByName("Distance1"), sld.IFeature)
    mate = wrap(feat.GetSpecificFeature2(), sld.IMate2)
    dim = wrap(wrap(mate.DisplayDimension, sld.IDisplayDimension).GetDimension2(0), sld.IDimension)
    rm = "BODY-1/SIDE PANEL-1/RobotMount-1"
    before = swlib.placement(M.comps(model)[rm])[2, 3]
    print(f"  Distance1 {dim.GetSystemValue3(1, None)[0] * 1000:.3f} mm, RobotMount z {before:.3f}")
    dim.SetSystemValue3(D75 / 1000.0, c.swSetValue_InThisConfiguration, None)
    model.EditRebuild3()
    after = swlib.placement(M.comps(model)[rm])[2, 3]
    box = swlib.placement(M.comps(model)["Box-1"])
    print(f"  Distance1 {dim.GetSystemValue3(1, None)[0] * 1000:.3f} mm, RobotMount z {after:.3f}, "
          f"Box-1 t {np.round(box[:, 3], 3)}, mate errors {M.mate_errors(model)}")
    if abs(after - D75) > 1e-6:
        raise SystemExit("RobotMount did not land at z = 75 -- read back, not trusted (gotcha 6)")


# ------------------------------------------------------------------ install
def geom(m, ent_face, comp_name):
    """Plane / cylinder of a mate entity face, in the Box frame (swmate's dicts)."""
    for f in m.faces(comp_name):
        if f["face"].IsSame(ent_face):
            return f
    return None


def datum_name(me):
    """The Box plane a mate entity refers to: by name if SolidWorks gives it, else
    by its geometry (normal + offset 0)."""
    try:
        nm = wrap(me.Reference, sld.IFeature).Name
        if nm in M.DATUMS:
            return nm
    except Exception:
        pass
    p = np.array(me.EntityParams, float)
    pt, n = p[:3] * 1000, p[3:6]
    for nm, (dn, off) in M.DATUMS.items():
        if abs(abs(n @ dn) - 1) < 1e-6 and abs(pt @ dn - off) < 1e-3:
            return nm
    return "?"


def record_mates(m, doc):
    """Every mate touching an OLD part: kind, lock, distance, both entities as
    geometry (or a datum plane name)."""
    rec = []
    for s in M.mates_of(doc):
        comps = M.mate_components(s)
        if not any(cn.split("/")[0] in OLD for cn in comps) or s.IsSuppressed():
            continue
        mt = wrap(s.GetSpecificFeature2(), sld.IMate2)
        tname = s.GetTypeName2()
        kind = {"MateCoincident": "coincident", "MateConcentric": "concentric", "MateParallel": "parallel",
                "MateDistanceDim": "distance"}.get(tname)
        ents = []
        for i in range(mt.GetMateEntityCount()):
            me = mt.MateEntity(i)
            rc = me.ReferenceComponent
            if rc is None:
                ents.append(("datum", datum_name(me)))
                continue
            cn = wrap(rc, sld.IComponent2).Name2
            try:
                face = wrap(me.Reference, sld.IFace2)
                g = geom(m, face, cn)
            except Exception:
                g = None
            ents.append((cn, g))
        dist = 0.0
        if kind == "distance":
            dd = wrap(wrap(mt.DisplayDimension, sld.IDisplayDimension).GetDimension2(0), sld.IDimension)
            dist = dd.GetSystemValue3(1, None)[0] * 1000
        lock = False
        if kind == "concentric":
            try:
                lock = bool(wrap(s.GetDefinition(), sld.IConcentricMateFeatureData).LockRotation)
            except Exception:
                lock = "conceL" in s.Name
        rec.append(dict(name=s.Name, kind=kind, ents=ents, dist=dist, lock=lock))
    return rec


def match(m, newname, g):
    """The face of the new part with the same geometry as g (a face of an old part)."""
    if g is None:
        return None
    try:
        if g["kind"] == "plane":
            return m.plane(newname, g["n"], g["off"])
        return m.cyl(newname, g["axis"], g["pt"], r=g["r"])
    except LookupError:
        return None


def transfer(m, rec):
    made, skipped = [], []
    for r in rec:
        if r["kind"] is None:
            skipped.append((r["name"], "type")); continue
        sides, ok = [], True
        for cn, g in r["ents"]:
            if cn == "datum":
                sides.append(g); continue
            top = cn.split("/")[0]
            if top in OLD:
                new = OLD[top]
                if new is None:
                    ok = False; break
                f = match(m, new, g)
                if f is None:
                    ok = False; break
                sides.append(f)
            else:
                if g is None:
                    ok = False; break
                sides.append(g)
        if not ok:
            skipped.append((r["name"], "no counterpart")); continue
        nm = "FX_" + r["name"].replace("BX_", "").replace("BackPanel-1", "FacetBack").replace(
            "FrontPanel-1", "FacetFront").replace("Cap-1", "FacetRing").replace("BackPanel", "FacetBack").replace(
            "FrontPanel", "FacetFront").replace("Cap", "FacetRing")
        try:
            m.mate(nm, r["kind"], sides[0], sides[1], dist=r["dist"], lock=r["lock"])
            made.append(nm)
        except (RuntimeError, SystemExit) as e:
            skipped.append((r["name"], str(e)[:90]))
    return made, skipped


def install(sw):
    doc = box_only(sw)
    with quiet(sw, doc):
        _install(sw, doc)
    doc.EditRebuild3()
    report(doc)


def _install(sw, doc):
    P_box = P_BOX_D75
    asm = wrap(doc, sld.IAssemblyDoc)
    m = M.Mater(sw, doc, prefix="")
    comps = M.comps(doc, top=True)

    print("1. insert the new parts where they were built (part frame = ROBOT d75 frame)")
    Minv = np.hstack([P_box[:, :3].T, -(P_box[:, :3].T @ P_box[:, 3])[:, None]])
    for n in PARTS:
        if n + "-1" in comps:
            print(f"   {n}-1 already in"); continue
        prt = os.path.join(FACET, n + ".SLDPRT")
        if find_doc(sw, prt) is None:
            sw.OpenDoc6(prt, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
            sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
        cp = wrap(asm.AddComponent5(prt, 0, "", False, "", 0.0, 0.0, 0.0), sld.IComponent2)
        M.set_placement(sw, cp, Minv)
        print(f"   + {cp.Name2}")
    doc.EditRebuild3()
    m = M.Mater(sw, doc, prefix="")

    print("2. PLANE1")
    reanchor_plane1_impl(sw, doc, m)
    m = M.Mater(sw, doc, prefix="")

    print("3. record the old parts' mates, suppress the old parts, re-make each mate on the new part")
    rec = record_mates(m, doc)
    print(f"   {len(rec)} mates touch an old part")
    ref = M.placements(doc)
    comps = M.comps(doc, top=True)
    for n in list(OLD) + LID_SCREWS:
        if n in comps and not comps[n].IsSuppressed():
            comps[n].SetSuppression2(c.swComponentSuppressed)
    doc.EditRebuild3()
    moved = M.compare({k: v for k, v in ref.items() if k.split("/")[0] not in OLD and k.split("/")[0] not in LID_SCREWS},
                      M.placements(doc))[0]
    print(f"   suppressed {len(OLD) + len(LID_SCREWS)} old components; moved by it: {moved[:5]}")
    m = M.Mater(sw, doc, prefix="")
    made, skipped = transfer(m, rec)
    print(f"   re-made {len(made)}, not re-made {len(skipped)}:")
    for nm, why in skipped:
        print(f"      - {nm}: {why}")

    print("4. new parts with no v3 mates of their own")
    for cname, hosts, tag in (("BumperFront-1", ["FacetFront-1"], "FX_BumperFront"),
                              ("BumperBack-1", ["FacetBack-1"], "FX_BumperBack"),
                              ("FacetHood-1", ["FacetRing-1"], "FX_Hood")):
        st = M.CS.get(M.comps(doc)[cname].GetConstrainedStatus())
        if st == "fully":
            print(f"   {cname} already fully defined"); continue
        chosen, rows, cands = M.plan(m, cname, hosts, max_n=3)
        print(f"   {cname}: {len(cands)} candidates, plan {[x[5] for x in chosen]}")
        M.apply(m, cname, chosen, tag)
    if M.CS.get(M.comps(doc)["NeopixelStrip-1"].GetConstrainedStatus()) != "fully":
        m.lock("FX_Strip_lock_Hood", "NeopixelStrip-1", "FacetHood-1")


def reanchor_plane1_impl(sw, doc, m):
    """PLANE1, the mirror plane of the left side, was the mid-plane of FrontPanel-1's
    two side faces.  FacetFront's plate IS that panel, so the same mid-plane is
    re-pointed at FacetFront-1's side faces (Box frame z = +5 and -155): same
    place, and suppressing the old panel no longer takes the left brackets with it.
    (ModifyDefinition refuses a switch to a one-reference offset plane.)"""
    feats = features(doc)
    f = feats["PLANE1"]

    def origin():
        return np.array(wrap(features(doc)["PLANE1"].GetSpecificFeature2(), sld.IRefPlane).Transform.ArrayData)[9:12] * 1000

    def parents():
        return [wrap(p, sld.IFeature).Name for p in (features(doc)["PLANE1"].GetParents() or [])]

    o0 = origin()
    if "FrontPanel-1" not in parents():
        print(f"   PLANE1 already free of FrontPanel-1 (parents {parents()}), origin {np.round(o0, 3)}")
        return
    zr = -o0[2]                                      # PLANE1 sits at Box z = o0[2]; faces are +-80 from it
    fa = m.plane("FacetFront-1", (0, 0, 1), o0[2] + 80.0)
    fb = m.plane("FacetFront-1", (0, 0, -1), -(o0[2] - 80.0))
    before = M.placements(doc)
    rp = wrap(f.GetDefinition(), sld.IRefPlaneFeatureData)
    rp.AccessSelections(doc, None)
    rp.SetReference(0, fa["face"])
    rp.SetReference(1, fb["face"])
    rp.SetConstraint(0, c.swRefPlaneReferenceConstraint_MidPlane)
    rp.SetConstraint(1, c.swRefPlaneReferenceConstraint_MidPlane)
    ok = f.ModifyDefinition(rp, doc, None)
    doc.EditRebuild3()
    o1 = origin()
    moved = M.compare(before, M.placements(doc))[0]
    print(f"   ModifyDefinition {ok}: origin {np.round(o1, 3)} (was {np.round(o0, 3)}), parents {parents()}, "
          f"moved {len(moved)}")
    if not (ok and np.allclose(o1, o0, atol=1e-6) and "FrontPanel-1" not in parents() and not moved):
        raise SystemExit("PLANE1 could not be re-pointed at FacetFront-1 without moving anything")


def report(doc):
    st = M.status(doc)
    under = {n: s for n, s in st.items() if s != "fully"}
    print(f"   Box.SLDASM: {len(st)} active top-level components, not fully defined: {under}")
    print(f"   mate errors: {M.mate_errors(doc)}")


def fixup(sw):
    """What the transfer cannot re-make, done explicitly (2026-10-06 run: the three
    datum mates of each v3 panel came back 'no counterpart', and swmate.plan finds
    no valid set for the hood -- its only real contact is the skirt on the flange,
    every screw concentric repeats that face's translation, gotcha 45):
      FacetBack   inner face on Right Plane, bottom on Top Plane, side face 5 mm
                  from Front Plane -- exactly the v3 BackPanel's BX_BackPanel_1..3
      FacetFront  bottom on Top Plane, side 5 mm from Front Plane (its floor
                  coincidence was transferred)
      FacetRing   back-centre screw hole concentric (rotation locked) with CornerBracket-26's
      FacetHood   locked to FacetRing (as placed)"""
    doc = box_only(sw)
    with quiet(sw, doc):
        m = M.Mater(sw, doc, prefix="")
        # Box frame = ROBOT d75 frame - (-150, 15, 75): plate inner face x 0, bottom y 0, side z +5
        m.mate("FX_FacetBack_1_coinc_RightPlane", "coincident", m.plane("FacetBack-1", (1, 0, 0), 0.0), "Right Plane")
        for part in ("FacetBack", "FacetFront"):
            m.mate(f"FX_{part}_2_coinc_TopPlane", "coincident", m.plane(part + "-1", (0, -1, 0), 0.0), "Top Plane")
            m.mate(f"FX_{part}_3_overhang_side_5mm", "distance", m.plane(part + "-1", (0, 0, 1), 5.0), "Front Plane",
                   dist=5.0)
        # the v3 Cap was held sideways by two lock-concentrics on the BackPanel's corner
        # radii, which the ring does not have: its back-centre screw hole (ROBOT x -140,
        # z 0 -> Box x 10, z -75) goes concentric with CornerBracket-26's vertical hole
        # (rotation locked: the transferred FX_FacetRing_3 parallel is between horizontal faces)
        m.delete_mates(["FX_FacetRing_4_conce_CB26"])
        m.mate("FX_FacetRing_4_conceL_CB26", "concentric", m.cyl("FacetRing-1", (0, 1, 0), (10.0, 0.0, -75.0), r=1.7),
               m.cyl("CornerBracket-26", (0, 1, 0), (10.0, 0.0, -75.0)), lock=True)
        if not any(sm.Name.startswith("FX_Hood") for sm in M.mates_of(doc)):
            m.lock("FX_Hood_lock_Ring", "FacetHood-1", "FacetRing-1")
    doc.EditRebuild3()
    report(doc)


def save_box(sw):
    """Save Box.SLDASM only if every top-level component is fully defined and no
    mate is in error.  (The Facet parts were saved by `import` and are unchanged.)"""
    doc = box_only(sw)
    doc.ForceRebuild3(False)
    st = M.status(doc)
    under = {n: s for n, s in st.items() if s != "fully"}
    errs = M.mate_errors(doc)
    if under or errs:
        raise SystemExit(f"REFUSING TO SAVE Box: not fully defined {under}, mate errors {errs}")
    ok, err, warn = doc.Save3(c.swSaveAsOptions_Silent, 0, 0)
    print(f"  saved Box.SLDASM: {ok} (err {err}, warn {warn}); {len(st)} components fully defined, 0 mate errors")


def save_robot(sw):
    """Save ROBOT.SLDASM ALONE (not Save All: dozens of parts come up dirty from a
    rebuild on open, and saving them is git noise).  Refused unless: every styled
    part is styled in memory (gotcha 37), no AI_HipDrive helper mate, Distance1 is
    75, Box-1 fully defined, 0 mate errors in ROBOT."""
    from importlib import import_module
    import swstyle as S
    model = swlib.open_v5(sw)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    model.ForceRebuild3(False)
    bad = []
    styled = import_module("10_verify_styled").STYLED
    for rel in styled:
        d = find_doc(sw, os.path.join(swlib.V5, rel))
        gl = [f for f in S._iter_features(d) if f.Name.startswith("GL_")] if d is not None else []
        if d is None or not gl or any(f.IsSuppressed2(c.swThisConfiguration, None)[0] for f in gl):
            bad.append(rel)
    asm = wrap(model, sld.IAssemblyDoc)
    if wrap(asm.FeatureByName(swlib.HipDriver.NAME), sld.IFeature) is not None:
        bad.append("AI_HipDrive present")
    mate = wrap(wrap(asm.FeatureByName("Distance1"), sld.IFeature).GetSpecificFeature2(), sld.IMate2)
    dist = wrap(wrap(mate.DisplayDimension, sld.IDisplayDimension).GetDimension2(0), sld.IDimension).GetSystemValue3(1, None)[0] * 1000
    if abs(dist - D75) > 1e-6:
        bad.append(f"Distance1 = {dist}")
    st = M.CS.get(M.comps(model)["Box-1"].GetConstrainedStatus())
    errs = M.mate_errors(model)
    if st != "fully" or errs:
        bad.append(f"Box-1 {st}, mate errors {errs}")
    if bad:
        raise SystemExit("REFUSING TO SAVE ROBOT: " + "; ".join(bad))
    ok, err, warn = model.Save3(c.swSaveAsOptions_Silent, 0, 0)
    print(f"  saved ROBOT.SLDASM: {ok} (err {err}, warn {warn}); Distance1 {dist:.3f}, Box-1 fully defined, "
          f"0 mate errors, {len(styled)} styled parts styled")


def do_check(sw):
    model = swlib.open_v5(sw)
    doc = box_doc(sw)
    report(doc)
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    model.ForceRebuild3(False)
    print(f"   ROBOT: Box-1 {M.CS.get(M.comps(model)['Box-1'].GetConstrainedStatus())}, "
          f"mate errors {M.mate_errors(model)}")


if __name__ == "__main__":
    sw, _ = swlib.connect()
    if sys.argv[1] in ("reimport", "colour"):
        {"reimport": do_reimport, "colour": do_colour}[sys.argv[1]](sw, sys.argv[2])
        raise SystemExit
    {"import": do_import, "distance": do_distance, "install": install, "fixup": fixup, "check": do_check,
     "save-box": save_box, "save-robot": save_robot}[sys.argv[1]](sw)
