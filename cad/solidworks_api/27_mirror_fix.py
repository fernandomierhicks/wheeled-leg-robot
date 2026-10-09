"""Step 27: the left side as the exact mirror of the right; the box's ring screws (README "Mirror fix").

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/27_mirror_fix.py <command> [--dry]

  brackets   Box + BottomPanelWithAvionics: the bracket mirror features dissolved; every left
             bracket at the EXACT mirror of its right twin, the front top corners flipped (nut
             trap down), the stand-ins the ring screws replace dropped, each bracket locked
  ring       the FacetRing's 9 screws + CornerBracket-26's FacetBack screw (M3 countersunk,
             flush in their countersinks) + an M3 nut in each top-corner bracket's trap
  features <name> ... --dry   what the leg mirror features' instance orientations should be.
             REFUSES for real: ModifyDefinition on TibiaMirror deleted MirrorTibia-2 (2026-10-08)
  leftleg    THE LEFT LEG FIX: Links/LeftFemur.SLDASM (the true left femur MirrorFemur.SLDPRT,
             rotor, screws) + Links/LeftTibia.SLDASM (clamp + 13 encoder screws corrected), every
             part at the exact mirror and locked, in ROBOT in place of the feature outputs
             MirrorFemur-4 / MirrorTibia-2 (suppressed); LeftFemur locked to the left inner plate,
             LeftTibia pinned at the knee and at E as the right one
  left, femur   (superseded by leftleg: the same fixes INSIDE MirrorTibia / MirrorFemur.SLDASM,
             which the mirror features regenerate on every hip move -- they do not last)
  switch     the limit-switch M2s raised onto the switch face (right); the left switch and its
             M2s at the exact mirror of the right ones; locked

Nothing here saves: 16_check_and_save.py --chain afterwards.

Why the left brackets were off: the v3 mirror features made INSTANCES of the bracket placed
by an orientation choice, and no orientation of an instance is the mirror image of this
bracket -- it is symmetric only under swapping its own x and y (OCC: x<->y common volume
1.00000, z -> 10-z 0.955).  So the mirror is S . P . swap: the instance origin stays the
mirror of its twin's, flathead-1 and -2 trade places (configurations NoFH1 <-> NoFH2).
Measured before: 6 left brackets 0.229 mm off the mirror, CornerBracket-30/31 upside down.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importlib import import_module
import swlib
import swmate
import swstyle as S
from swlib import c, wrap, sld

sys.stdout.reconfigure(line_buffering=True)       # runs are long: let a log file follow them
F = import_module("25_fasteners")
BOX, BPWA = r"Box\Box.SLDASM", r"Box\BottomPanelWithAvionics.SLDASM"
SZ = np.diag([1.0, 1.0, -1.0])                                     # the robot mirror, z -> -z
SWAP = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])           # the bracket's own symmetry
CFG_SWAP = {"Default": "Default", "NoFH1": "NoFH2", "NoFH2": "NoFH1", "NoFH12": "NoFH12"}

# Box: left bracket <- its right twin; the front top corner flipped about its diagonal (his
# call 2026-10-08): the corner hole's hex trap faced UP under the ring, so a ring screw's nut
# would pull on the ring and clamp nothing.  Flipped, both side holes stay on their panels.
BOX_TWINS = {"CornerBracket-9": "CornerBracket-1", "CornerBracket-10": "CornerBracket-2",
             "CornerBracket-11": "CornerBracket-3", "CornerBracket-12": "CornerBracket-4",
             "CornerBracket-17": "CornerBracket-14", "CornerBracket-18": "CornerBracket-13",
             "CornerBracket-30": "CornerBracket-28", "CornerBracket-31": "CornerBracket-21"}
FLIP = ["CornerBracket-4"]
FLIP_M = np.array([[0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, -1.0, 10.0]])   # (x,y,z) -> (y,x,10-z)
# the stand-in flatheads a ring / FacetBack screw replaces (the bracket's own 12 mm flathead
# sits for a 5.1 mm wall; in the 3 mm ring or FacetBack plate its head stood 2.1 mm proud)
RING_CFG = {"CornerBracket-21": "NoFH2", "CornerBracket-28": "NoFH12", "CornerBracket-26": "NoFH12"}
LOCK_TO = {"CornerBracket-9": "FacetBack-1", "CornerBracket-10": "FacetBack-1", "CornerBracket-18": "FacetBack-1",
           "CornerBracket-11": "FacetFront-1", "CornerBracket-12": "FacetFront-1",
           "CornerBracket-17": "FacetFront-1", "CornerBracket-4": "FacetFront-1",
           "CornerBracket-30": "FacetRing-1", "CornerBracket-31": "FacetRing-1"}
BOX_MIRRORS = ["Mirror components", "Mirror side panel"]          # stripped names (one has a trailing blank)
# BottomPanelWithAvionics: its two left floor brackets
BPWA_TWINS = {"CornerBracket-3": "CornerBracket-2", "CornerBracket-4": "CornerBracket-1"}
BPWA_MIRRORS = ["MirrorComponent1"]
MPFX = "MF_"


def h(P):
    return np.vstack([np.asarray(P, float), [0, 0, 0, 1]])


def mirrored(P, m=SWAP):
    """Robot-frame placement of the exact mirror (z -> -z) of a component at P."""
    return np.hstack([SZ @ P[:, :3] @ m, (SZ @ P[:, 3])[:, None]])


def open_asm(sw, rel):
    d = wrap(sw.OpenDoc6(os.path.join(swlib.V5, rel), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)[0],
             sld.IModelDoc2)
    sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
    return d


def dissolve(d, names, dry):
    for f in [x for x in S._iter_features(d) if x.GetTypeName2() == "MirrorCompFeat"]:
        if f.Name.strip() not in names:
            continue
        if dry:
            print(f"  would dissolve {f.Name!r}")
            continue
        before = swmate.placements(d)
        d.ClearSelection2(True)
        f.Select2(False, 0)
        ok = wrap(d, sld.IAssemblyDoc).DissolveComponentPattern()
        d.ClearSelection2(True)
        d.EditRebuild3()
        moved, gone, new = swmate.compare(before, swmate.placements(d))
        print(f"  dissolved {f.Name.strip()!r}: {ok}; moved {len(moved)}, gone {gone}, new {new}")
        if not ok or moved or gone:
            raise SystemExit("dissolve changed the assembly -- reload it (nothing saved)")


def unmate(d, comp, live_only=False):
    """Delete every mate of this document that references `comp` (or a child of it);
    live_only: leave suppressed ones (an opposite-hand sub-assembly keeps its source's mates
    as suppressed copies)."""
    gone = []
    for s in swmate.mates_of(d):
        if live_only and s.IsSuppressed():
            continue
        if any(x == comp or x.startswith(comp + "/") for x in swmate.mate_components(s)):
            gone.append(s.Name)
            d.ClearSelection2(True)
            s.Select2(False, 0)
            d.Extension.DeleteSelection2(0)
    d.ClearSelection2(True)
    d.EditRebuild3()
    return gone


def place(sw, d, comp, M, cfg=None):
    if cfg is not None and comp.ReferencedConfiguration != cfg:
        comp.ReferencedConfiguration = cfg
        d.EditRebuild3()
    swmate.set_placement(sw, comp, M)
    d.EditRebuild3()
    err = np.abs(swlib.placement(comp) - M).max()
    if err > 1e-3:
        raise SystemExit(f"{comp.Name2} did not stay where it was put ({err:.4f}) -- reload (nothing saved)")
    return err


def dropped(sw, d):
    """A mate still on a bracket flathead the configuration now suppresses holds nothing (it
    reports no error) -- FX_CB26_1_conce_FacetBack held CornerBracket-26, and the FacetRing is
    concentric-locked to that bracket: both went UNDER.  Re-made on the bracket's own coaxial
    bore (25.remate_on_bracket: the same constraint, refused if anything moves)."""
    dc = swmate.comps(d)
    for s in swmate.mates_of(d):
        for cn in swmate.mate_components(s):
            if "/M3x8 flathead-" in cn and cn in dc and dc[cn].IsSuppressed() and not s.IsSuppressed():
                others = [o for o in swmate.mate_components(s) if o != cn]
                if any(o in dc and dc[o].IsSuppressed() for o in others):
                    continue                              # dormant: its other side is suppressed too
                print(f"  {s.Name}: on the dropped {cn} -> re-made on the bracket's bore")
                F.remate_on_bracket(sw, d, s, cn, dc)
                dc = swmate.comps(d)


def brackets(dry=False):
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    # a ROBOT-level mate on a bracket that moves would break
    moving = set(FLIP) | set(BOX_TWINS)
    held = [(s.Name, x) for s in swmate.mates_of(robot) for x in swmate.mate_components(s)
            if x.split("/")[0] == "Box-1" and len(x.split("/")) > 1 and x.split("/")[1] in moving]
    if held:
        raise SystemExit(f"ROBOT mates hold a bracket that moves: {held}")
    T_box = swlib.placement(swlib.components(robot)["Box-1"])

    # ---- Box ----
    d = open_asm(sw, BOX)
    dissolve(d, BOX_MIRRORS, dry)
    cp = swmate.comps(d, True)
    for n, cfg in RING_CFG.items():
        print(f"  {n}: {cp[n].ReferencedConfiguration} -> {cfg} (ring / FacetBack screw replaces the stand-in)")
        if not dry:
            cp[n].ReferencedConfiguration = cfg
    if not dry:
        d.EditRebuild3()
        dropped(sw, d)
    for n in FLIP:
        P = swlib.placement(cp[n])
        if P[:, 2] @ [0, 1.0, 0] > 0.99:              # local z up: the trap (z 0..2.5) is at the bottom
            print(f"  {n}: trap already down")
            continue
        Mn = (h(P) @ h(FLIP_M))[:3]
        print(f"  {n}: flip about its diagonal (trap down), mates deleted: "
              f"{[] if dry else unmate(d, n)}")
        if not dry:
            place(sw, d, cp[n], Mn)
    for ln, rn in BOX_TWINS.items():
        # Box frame: the robot mirror is z -> -150 - z (Box-1 sits at robot z +75, R = I)
        Pr = h(T_box) @ h(swlib.placement(cp[rn]))
        target = (np.linalg.inv(h(T_box)) @ h(mirrored(Pr[:3])))[:3]
        now = swlib.placement(cp[ln])
        cfg = CFG_SWAP[cp[rn].ReferencedConfiguration]
        print(f"  {ln:18s} <- {rn:17s} moves {np.abs(now[:, 3] - target[:, 3]).max():6.3f} mm, "
              f"rotation {np.abs(now[:, :3] - target[:, :3]).max():.3f}; {cp[ln].ReferencedConfiguration} -> {cfg}")
        if not dry and (np.abs(now - target).max() > 1e-3 or cp[ln].ReferencedConfiguration != cfg):
            gone = unmate(d, ln)
            if gone:
                print(f"     mates deleted: {gone}")
            place(sw, d, cp[ln], target, cfg)
    if not dry:
        dropped(sw, d)
        m = swmate.Mater(sw, d, prefix=MPFX)
        for n, panel in LOCK_TO.items():
            m.lock(f"{n.replace('CornerBracket-', 'CB')}_lock_{panel.split('-')[0]}", n, panel)
        d.EditRebuild3()
        print(f"  Box: mate errors {swmate.mate_errors(d)}, feature errors {F.feature_errors(d)}")

    # ---- BottomPanelWithAvionics ----
    rc = swlib.components(robot)
    T_b = swlib.placement(rc["Box-1/BottomPanelWithAvionics-1"])
    d = open_asm(sw, BPWA)
    dissolve(d, BPWA_MIRRORS, dry)
    cp = swmate.comps(d, True)
    for ln, rn in BPWA_TWINS.items():
        Pr = h(T_b) @ h(swlib.placement(cp[rn]))
        target = (np.linalg.inv(h(T_b)) @ h(mirrored(Pr[:3])))[:3]
        now = swlib.placement(cp[ln])
        cfg = CFG_SWAP[cp[rn].ReferencedConfiguration]
        print(f"  BPWA {ln:18s} <- {rn:17s} moves {np.abs(now[:, 3] - target[:, 3]).max():6.3f} mm; "
              f"{cp[ln].ReferencedConfiguration} -> {cfg}")
        if not dry and (np.abs(now - target).max() > 1e-3 or cp[ln].ReferencedConfiguration != cfg):
            gone = unmate(d, ln)
            if gone:
                print(f"     mates deleted: {gone}")
            place(sw, d, cp[ln], target, cfg)
    if not dry:
        dropped(sw, d)
        m = swmate.Mater(sw, d, prefix=MPFX)
        for ln in BPWA_TWINS:
            m.lock(f"{ln.replace('CornerBracket-', 'CB')}_lock_BottomPanel", ln, "BottomPanel-1")
        d.EditRebuild3()
        print(f"  BPWA: mate errors {swmate.mate_errors(d)}, feature errors {F.feature_errors(d)}")
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.EditRebuild3()
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}")


# ---- ring: the FacetRing's screws ----------------------------------------------------------
F3, NUT = r"Common\M3 Flathead.SLDPRT", r"Box\M3 nut.SLDPRT"
NUT_HALF = 1.352                  # the M3 nut model: z +-1.352, flats normal to its local x
TRAP = (5.0, 5.0, 2.5)            # CornerBracket.SLDPRT: the corner bore's hex trap, z 0..2.5
CORNER = {"CornerBracket-1", "CornerBracket-4", "CornerBracket-9", "CornerBracket-12"}


def trap_angle():
    """Angle (rad, about local z) of a flat normal of the bracket's corner hex trap."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    ex = TopExp_Explorer(F.shape("CornerBracket__Default"), TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        ex.Next()
        ad = BRepAdaptor_Surface(f)
        if ad.GetType() != GeomAbs_Plane:
            continue
        n = ad.Plane().Axis().Direction()
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, g)
        q = g.CentreOfMass()
        if abs(n.Z()) < 1e-6 and 0 < q.Z() < TRAP[2] and np.hypot(q.X() - TRAP[0], q.Y() - TRAP[1]) < 3.5:
            return float(np.arctan2(n.Y(), n.X()))
    raise SystemExit("no hex trap flat found on CornerBracket")


def ring(dry=False):
    """The ring's 9 holes + CB-26's FacetBack hole -> plan_ring.json -> 25's build (insert +
    concentric + flush coincident, checked); then a nut in each top-corner bracket's trap."""
    with open(F.SCAN_ALL) as f:
        sc = json.load(f)
    by = {x["name"]: x for x in sc["comps"]}
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rc = swlib.components(robot)
    items = []

    def seat_on(part, p, a, rho=3.6):
        """t (along a, from p) of the part's face looking along +a, at radius rho."""
        x = by[part]
        u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        iv = F.hits(x["step"], np.array(x["M"]), p + rho * u, a, -8, 8)
        return max(w for _, w in iv)

    def bracket_on(p, a):
        """The bracket bore on this axis NEAREST along it: brackets lower down the same
        vertical line (CB-18 under CB-9, CB-27 under CB-26) share the axis."""
        cand = []
        for n, cp in rc.items():
            if n.count("/") == 1 and n.startswith("Box-1/CornerBracket-"):
                Mb = swlib.placement(cp)
                q = Mb[:, :3] @ np.array(TRAP) + Mb[:, 3] - p
                if abs(abs(Mb[:, 2] @ a) - 1) < 1e-6 and np.linalg.norm(q - (q @ a) * a) < 0.05:
                    cand.append((abs(q @ a), n.split("/")[1], "corner"))
                for o, ax in (((10, 0.5, 5), (0, 1, 0)), ((0.5, 10, 5), (1, 0, 0))):
                    q = Mb[:, :3] @ np.array(o, float) + Mb[:, 3] - p
                    if abs(abs((Mb[:, :3] @ np.array(ax, float)) @ a) - 1) < 1e-6 \
                            and np.linalg.norm(q - (q @ a) * a) < 0.05:
                        cand.append((abs(q @ a), n.split("/")[1], "side"))
        if not cand:
            return None, None
        return min(cand)[1:]
    ringc = by["Box-1/FacetRing-1"]
    Mr = np.array(ringc["M"])
    k = 0
    for hh in F.local_holes(ringc["step"]):
        a = Mr[:, :3] @ hh["a"]
        if not (1.6 <= hh["r"] <= 1.8 and hh["span"] >= 300 and abs(a[1]) > 0.99):
            continue
        a = a * np.sign(a[1])                                   # up: the head side
        p0 = Mr[:, :3] @ hh["p"] + Mr[:, 3]
        mid = p0 + a * ((Mr[:, :3] @ hh["a"]) @ a) * hh["L"] / 2
        top = seat_on("Box-1/FacetRing-1", mid, a)
        br, kind = bracket_on(mid, a)
        if br is None:
            raise SystemExit(f"ring hole at {mid.round(2)}: no bracket bore on its axis")
        cfg = "14mm" if kind == "corner" else "10mm"
        k += 1
        items.append(dict(group="ring", line=900 + k, model=F3, config=cfg, asm=BOX,
                          seat=(mid + top * a).round(5).tolist(), dir=a.round(8).tolist(), bore="Box-1/FacetRing-1",
                          bore_mid=mid.round(5).tolist(), bore_r=hh["r"], seat_part="Box-1/FacetRing-1",
                          t=round(float(top), 3), probe=3.6, bracket=br, kind=kind))
    # CB-26's FacetBack screw: on its flathead-1's axis (that stand-in is dropped)
    Mb = swlib.placement(rc["Box-1/CornerBracket-26"])
    o = Mb[:, :3] @ np.array([0.5, 10.0, 5.0]) + Mb[:, 3]
    a = -(Mb[:, :3] @ np.array([1.0, 0, 0]))                       # out of the bracket, towards the head
    fb = by["Box-1/FacetBack-1"]
    Mf = np.array(fb["M"])
    best = None
    for hh in F.local_holes(fb["step"]):
        ha, hp = Mf[:, :3] @ hh["a"], Mf[:, :3] @ hh["p"] + Mf[:, 3]
        q = hp - o
        if 1.6 <= hh["r"] <= 1.8 and abs(abs(ha @ a) - 1) < 1e-6 and np.linalg.norm(q - (q @ a) * a) < 0.05:
            mid = hp + ha * hh["L"] / 2
            if best is None or abs((mid - o) @ a) < abs((best[0] - o) @ a):
                best = (mid, hh["r"])
    mid, r_ = best
    top = seat_on("Box-1/FacetBack-1", mid, a)
    items.append(dict(group="facetback", line=911, model=F3, config="10mm", asm=BOX,
                      seat=(mid + top * a).round(5).tolist(), dir=a.round(8).tolist(), bore="Box-1/FacetBack-1",
                      bore_mid=mid.round(5).tolist(), bore_r=r_, seat_part="Box-1/FacetBack-1",
                      t=round(float(top), 3), probe=3.6, bracket="CornerBracket-26", kind="side"))
    for it in items:
        print(f"  {it['group']:9s} L{it['line']} {it['config']:5s} seat {np.round(it['seat'], 2).tolist()} "
              f"into {it['bracket']} ({it['kind']})")
    with open(os.path.join(F.OUT, "plan_ring.json"), "w") as f:
        json.dump(dict(add=items), f, indent=1)
    F.build([BOX], dry, "plan_ring.json")
    # the nuts: in each top-corner bracket's hex trap, against its ceiling, flats on the trap's
    th = trap_angle()
    Rn = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    d = open_asm(sw, BOX)
    T_box = swlib.placement(rc["Box-1"])
    # a nut an earlier run locked into the wrong bracket (one lower down the same axis) goes
    want = {f"MF_nut_{it['bracket'].replace('CornerBracket-', 'CB')}_lock" for it in items if it["kind"] == "corner"}
    for s in swmate.mates_of(d):
        if s.Name.startswith("MF_nut_") and s.Name not in want:
            nut = next(x for x in swmate.mate_components(s) if x.startswith("M3 nut"))
            print(f"  {s.Name}: wrong bracket -- deleting {nut}{'' if not dry else ' (dry)'}")
            if not dry:
                d.ClearSelection2(True)
                swmate.comps(d, True)[nut].Select4(False, None, False)
                d.Extension.DeleteSelection2(0)
                d.ClearSelection2(True)
                d.EditRebuild3()
    cp = swmate.comps(d, True)
    nut_path = os.path.normcase(os.path.join(swlib.V5, NUT))
    placed = [x for x in cp.values() if os.path.normcase(x.GetPathName() or "") == nut_path and "/" not in x.Name2]
    m = swmate.Mater(sw, d, prefix=MPFX)
    sw.OpenDoc6(os.path.join(swlib.V5, NUT), c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
    d = open_asm(sw, BOX)
    for it in items:
        if it["kind"] != "corner":
            continue
        Mb = swlib.placement(cp[it["bracket"]])
        Mn = np.hstack([Mb[:, :3] @ Rn, (Mb[:, :3] @ np.array([TRAP[0], TRAP[1], TRAP[2] - NUT_HALF]) + Mb[:, 3])[:, None]])
        have = next((x for x in placed if np.abs(swlib.placement(x)[:, 3] - Mn[:, 3]).max() < 0.01), None)
        print(f"  nut in {it['bracket']}: {'there already' if have else 'insert'} at {Mn[:, 3].round(2).tolist()}")
        if dry or have is not None:
            continue
        x = wrap(wrap(d, sld.IAssemblyDoc).AddComponent5(os.path.join(swlib.V5, NUT),
                                                         c.swAddComponentConfigOptions_CurrentSelectedConfig, "",
                                                         False, "", *(Mn[:, 3] / 1000.0)), sld.IComponent2)
        if x is None:
            raise SystemExit("AddComponent5 (nut) failed")
        place(sw, d, x, Mn)
        m._comps = swmate.comps(d)
        m.lock(f"nut_{it['bracket'].replace('CornerBracket-', 'CB')}_lock", x.Name2, it["bracket"])
    d.EditRebuild3()
    print(f"  Box: mate errors {swmate.mate_errors(d)}")
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.EditRebuild3()
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}")


# ---- the leg: the ROBOT mirror features' instance orientations --------------------------
# An instance mirrored about the robot mid-plane is placed with its origin at the mirror of
# its twin's and its axes from one of four options, R' = S R D_k:
#   0 MirroredX_MirroredY  D = diag(1,1,-1)    1 MirroredAndFlippedX_MirroredY  D = diag(-1,1,1)
#   2 MirroredX_MirroredAndFlippedY  D = diag(1,-1,1)    3 both flipped  D = -I
# It is the exact mirror image only if the part is symmetric under D_k about its own origin.
# Measured (OCC, 2026-10-08): a screw under 1 and 2 (any plane through its axis) -- 0 turns it
# head-for-tip, which is what every left encoder screw was; EncoderCarrier under 1 (x = 0);
# EncoderCableClamp under 0 (z = 0); AK45-10 Rotor under 1 (x = 0).
ORIENT = {
    "TibiaMirror": {"Tibia-1/M3 Roundhead-*": 1, "Tibia-1/M2 Roundhead-*": 1, "Tibia-1/EncoderCableClamp-1": 0},
    "FemurMirror": {"Femur-1/M2.5 Roundhead-*": 1, "Femur-1/AK45-10 Rotor-1": 1},
}
LEFT_TOPS = ["MirrorTibia-2", "MirrorFemur-4", "MirrorBODY-2", "MirrorCOUPLER-2", "MirrorFEMUR_INSIDE-2"]


def _want(rules, name):
    for pat, k in rules.items():
        if name == pat or (pat.endswith("*") and name.startswith(pat[:-1])):
            return k
    return None


def features(names, dry=False):
    """ModifyDefinition of the named ROBOT mirror features: ONLY the orientation of listed
    instances changes.  DO NOT RUN for real: on 2026-10-08 the orientation-only edit of
    TibiaMirror returned True and DELETED MirrorTibia-2 from ROBOT (recovered by
    ReloadOrReplace; everything had been saved first) -- the same as appending to BodyMirror
    did.  `--dry` only lists what the orientations should be; `left` does the fix."""
    if not dry:
        raise SystemExit("refused: ModifyDefinition on a leg mirror feature deletes its left sub-assembly "
                         "(2026-10-08).  Use `left`; --dry lists the orientations.")
    import pythoncom
    from win32com.client import VARIANT
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    asm = wrap(robot, sld.IAssemblyDoc)
    for fname in names:
        feat = next((f for f in S._iter_features(robot) if f.GetTypeName2() == "MirrorCompFeat"
                     and f.Name.strip() == fname), None)
        if feat is None:
            raise SystemExit(f"no mirror feature {fname}")
        fd = wrap(feat.GetDefinition(), sld.IMirrorComponentFeatureData)
        fd.AccessSelections(robot, None)
        done = False
        try:
            cur = [wrap(x, sld.IComponent2) for x in (fd.ComponentsToInstanceAlignToComponentOrigin or [])]
            ori = list(fd.ComponentOrientationsAlignToComponentOrigin or [])
            new = [(_want(ORIENT[fname], x.Name2) if _want(ORIENT[fname], x.Name2) is not None else o)
                   for x, o in zip(cur, ori)]
            for x, o, n in zip(cur, ori, new):
                if o != n:
                    print(f"  {fname}: {x.Name2:44s} {o} -> {n}")
            if dry or new == ori:
                continue
            fd.ComponentOrientationsAlignToComponentOrigin = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_I4, new)
            done = bool(feat.ModifyDefinition(fd, robot, None))
            print(f"  {fname}: ModifyDefinition -> {done}")
        finally:
            if not done:
                fd.ReleaseSelectionAccess()
        robot.EditRebuild3()
        top = {wrap(x, sld.IComponent2).Name2 for x in asm.GetComponents(True) or []}
        lost = [n for n in LEFT_TOPS if n not in top]
        if lost:
            raise SystemExit(f"LOST {lost} -- reload ROBOT from disk now (ReloadOrReplace), nothing is saved")
        print(f"  {fname}: left sub-assemblies all there; ROBOT mate errors {swmate.mate_errors(robot)}, "
              f"feature errors {F.feature_errors(robot)}")


# ---- the leg: exact mirror by placement + lock, inside the left sub-assemblies ----------------
# The features can't be edited (above), but a mate inside an opposite-hand sub-assembly wins over
# the feature's placement (measured: MirrorFemur.SLDASM's femur sat where its own mates put it,
# not at the feature's orientation).  So each wrong left component goes to the exact mirror of
# its right twin, S . P_right . D (D: the reflection its part is symmetric under -- ORIENT's
# measurements), and is LOCKED to the part it is bolted to.
D = {0: np.diag([1.0, 1, -1]), 1: np.diag([-1.0, 1, 1])}
LEFT_FIX = {   # sub-assembly file: (left top instance, right top instance, [(right leaf pattern, D, lock to)])
    r"Links\MirrorTibia.SLDASM": ("MirrorTibia-2", "Tibia-1", [
        ("EncoderCableClamp-1", 0, "EncoderCarrier-1"),
        ("M3 Roundhead-1", 1, "MirrorTibia-1"), ("M3 Roundhead-2", 1, "MirrorTibia-1"),   # carrier -> tibia
        ("M3 Roundhead-3", 1, "MirrorTibia-1"), ("M3 Roundhead-4", 1, "MirrorTibia-1"),
        ("M3 Roundhead-5", 1, "MirrorTibia-1"),
        ("M3 Roundhead-6", 1, "EncoderCableClamp-1"), ("M3 Roundhead-7", 1, "EncoderCableClamp-1"),  # clamp -> carrier
        ("M3 Roundhead-8", 1, "EncoderCableClamp-1"), ("M3 Roundhead-9", 1, "EncoderCableClamp-1"),
        ("M3 Roundhead-10", 1, "EncoderCableClamp-1"), ("M3 Roundhead-11", 1, "EncoderCableClamp-1"),
        ("M2 Roundhead-1", 1, "EncoderCarrier-1"), ("M2 Roundhead-2", 1, "EncoderCarrier-1")]),   # PCB -> carrier
}


def left_fix(rel, dry=False, extra=()):
    """Every listed left component of `rel` at the exact mirror of its right twin, locked.
    The left twin of a right component is the left component nearest that mirror position
    (instance numbers differ: the left femur's M2.5s are -4..-6)."""
    sub, rsub, items = LEFT_FIX[rel] if rel in LEFT_FIX else extra
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rc = swlib.components(robot)
    T = swlib.placement(rc[sub])
    d = open_asm(sw, rel)
    cl = swmate.comps(d, True)
    done = []
    for rleaf, k, to in items:
        PR = swlib.placement(rc[f"{rsub}/{rleaf}"])
        Mt = (np.linalg.inv(h(T)) @ h(np.hstack([SZ @ PR[:, :3] @ D[k], (SZ @ PR[:, 3])[:, None]])))[:3]
        stem = rleaf.rsplit("-", 1)[0]
        if rleaf in cl and rleaf not in done:            # the feature keeps the right's instance names
            ln = rleaf
        else:                                            # renumbered (the left femur's M2.5s): nearest
            ln = min((n for n in cl if n.rsplit("-", 1)[0] == stem and n not in done),
                     key=lambda n: np.linalg.norm(swlib.placement(cl[n])[:, 3] - Mt[:, 3]))
        done.append(ln)
        now = swlib.placement(cl[ln])
        print(f"  {ln:22s} <- {rleaf:22s} D{k}: moves {np.abs(now[:, 3] - Mt[:, 3]).max():7.3f} mm, "
              f"rotation {np.abs(now[:, :3] - Mt[:, :3]).max():.3f}; lock to {to}")
        if dry:
            continue
        gone = unmate(d, ln, live_only=True)
        if gone:
            print(f"     mates deleted: {gone}")
        place(sw, d, cl[ln], Mt)
        swmate.Mater(sw, d, prefix=MPFX).lock(f"{ln.replace(' ', '')}_lock_{to.replace(' ', '')}", ln, to)
    if dry:
        return
    d.EditRebuild3()
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.ForceRebuild3(False)
    # held? (the feature must not have put anything back)
    rc = swlib.components(robot)
    worst = 0.0
    for rleaf, k, to in items:
        PR = swlib.placement(rc[f"{rsub}/{rleaf}"])
        tgt = np.hstack([SZ @ PR[:, :3] @ D[k], (SZ @ PR[:, 3])[:, None]])
        worst = max(worst, min(np.abs(swlib.placement(cp) - tgt).max() for n, cp in rc.items()
                               if n.startswith(sub + "/")))
    print(f"  after a ROBOT rebuild: worst distance to the mirror {worst:.5f}; "
          f"{rel} mate errors {swmate.mate_errors(d)}; ROBOT {swmate.mate_errors(robot)}")
    if worst > 1e-3:
        raise SystemExit("the mirror feature moved something back -- reload ROBOT + the sub-assembly (nothing saved)")


# ---- the left femur: a true left-hand part --------------------------------------------------
# FemurMirror makes the left femur an INSTANCE of Femur.SLDPRT (orientation 2: the right femur
# rotated 180 deg about its long axis) -- no rotation of an asymmetric part is its mirror image,
# and that rotated femur collides with the left tibia (208 mm3 at hip -21, README "The robot
# assembly").  His call 2026-10-08: the true mirror, Links/MirrorFemur.SLDPRT (the opposite-hand
# part this feature once derived from the styled v5 Femur), in MirrorFemur.SLDASM at the exact
# mirror of the right femur, locked to the left rotor; the feature's Femur.SLDPRT instance
# there suppressed (kept, as his rule for replaced parts).
MFEMUR, FEMUR = r"Links\MirrorFemur.SLDPRT", r"Links\Femur.SLDPRT"
MFEMUR_ASM = r"Links\MirrorFemur.SLDASM"


def _part(sw, rel):
    d = wrap(sw.OpenDoc6(os.path.join(swlib.V5, rel), c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)[0],
             sld.IModelDoc2)
    if d is None:
        raise SystemExit(f"cannot open {rel}")
    return d


def _props(doc):
    """Volume, centre and inertia (about the centre, part axes) of all solid bodies."""
    V, C, I = 0.0, np.zeros(3), np.zeros((3, 3))
    rows = []
    for b in S.bodies(doc):
        m = b.GetMassProperties(1.0)
        v, cc = m[3] * 1e9, np.array(m[:3]) * 1e3
        Ib = np.array([[m[6], m[9], m[10]], [m[9], m[7], m[11]], [m[10], m[11], m[8]]]) * 1e15
        rows.append((v, cc, Ib))
        V += v
        C += v * cc
    C /= V
    for v, cc, Ib in rows:                       # parallel axis to the common centre
        d = cc - C
        I += Ib + v * (d @ d * np.eye(3) - np.outer(d, d))
    return V, C, I


def femur(dry=False):
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    fdoc, mdoc = _part(sw, FEMUR), _part(sw, MFEMUR)
    mdoc.ForceRebuild3(False)
    (Vf, Cf, If), (Vm, Cm, Im) = _props(fdoc), _props(mdoc)
    print(f"  Femur {len(S.bodies(fdoc))} bodies {Vf:.2f} mm3 | MirrorFemur {len(S.bodies(mdoc))} bodies {Vm:.2f} mm3")
    if abs(Vf - Vm) > 1e-3 * Vf:
        raise SystemExit("MirrorFemur.SLDPRT is not the current Femur's mirror (volume) -- rebuild it first")
    best = None
    for name, m in (("x", np.diag([-1.0, 1, 1])), ("y", np.diag([1.0, -1, 1])), ("z", np.diag([1.0, 1, -1]))):
        # a mirror about a plane x_k = c: centre maps by m plus an offset along k only
        off = Cm - m @ Cf
        err = np.abs(Im - m @ If @ m).max() / np.abs(If).max() + np.linalg.norm(off - (off * (np.diag(m) < 0)))
        if best is None or err < best[0]:
            best = (err, name, m, off)
    err, name, m, off = best
    print(f"  MirrorFemur.SLDPRT = Femur mirrored about its {name} plane, offset {off.round(4).tolist()} (fit {err:.1e})")
    if err > 1e-3:
        raise SystemExit("cannot identify the mirror transform")
    Mpart = np.hstack([m, off[:, None]])                   # MirrorFemur geometry = Mpart . Femur geometry
    rc = swlib.components(robot)
    PR = swlib.placement(rc["Femur-1/Femur-1"])
    T = swlib.placement(rc["MirrorFemur-4"])
    # left placement L with L . Mpart = S . PR  ->  L = S . PR . Mpart^-1
    Lr = h(np.hstack([SZ @ PR[:, :3], (SZ @ PR[:, 3])[:, None]])) @ np.linalg.inv(h(Mpart))
    target = (np.linalg.inv(h(T)) @ Lr)[:3]
    d = open_asm(sw, MFEMUR_ASM)
    cp = swmate.comps(d, True)
    mpath = os.path.normcase(os.path.join(swlib.V5, MFEMUR))
    have = next((x for x in cp.values() if os.path.normcase(x.GetPathName() or "") == mpath), None)
    print(f"  MirrorFemur.SLDASM: {'has' if have else 'insert'} MirrorFemur.SLDPRT at {target[:, 3].round(3).tolist()}; "
          f"suppress {[n for n, x in cp.items() if os.path.basename(x.GetPathName() or '').lower() == 'femur.sldprt']}")
    if dry:
        return
    if have is None:
        have = wrap(wrap(d, sld.IAssemblyDoc).AddComponent5(os.path.join(swlib.V5, MFEMUR),
                                                            c.swAddComponentConfigOptions_CurrentSelectedConfig, "",
                                                            False, "", *(target[:, 3] / 1000.0)), sld.IComponent2)
        if have is None:
            raise SystemExit("AddComponent5 MirrorFemur.SLDPRT failed")
    place(sw, d, have, target)
    for n, x in swmate.comps(d, True).items():
        if os.path.basename(x.GetPathName() or "").lower() == "femur.sldprt" and not x.IsSuppressed():
            x.SetSuppression2(c.swComponentSuppressed)
            print(f"  {n} (Femur.SLDPRT, rotated) suppressed")
    d.EditRebuild3()
    if not have.IsFixed():                 # as the right femur is coordinate-mated to its sub-assembly
        d.ClearSelection2(True)
        have.Select4(False, None, False)
        wrap(d, sld.IAssemblyDoc).FixComponent()
        d.ClearSelection2(True)
        d.EditRebuild3()
    print(f"  {have.Name2}: placed + fixed, error {np.abs(swlib.placement(have) - target).max():.5f}; "
          f"mate errors {swmate.mate_errors(d)}")
    # its rotor and the 3 M2.5s at the exact mirror, locked to it (the M3s are right already)
    left_fix(MFEMUR_ASM, extra=("MirrorFemur-4", "Femur-1", [
        ("AK45-10 Rotor-1", 1, have.Name2), ("M2.5 Roundhead-1", 1, have.Name2),
        ("M2.5 Roundhead-2", 1, have.Name2), ("M2.5 Roundhead-3", 1, have.Name2)]))


# ---- the left femur + tibia OUT of the mirror features (his call, 2026-10-08) -------------------
# MirrorFemur.SLDASM / MirrorTibia.SLDASM are owned by ROBOT's FemurMirror / TibiaMirror: every hip
# move regenerates them from the features' instance orientations (measured: the left encoder screws
# snapped back head-for-tip, a directly inserted part was deleted, mates inside do not hold), and
# ModifyDefinition on those features deletes the left sub-assembly (3 times, any arrays).  So the
# left femur and tibia become ordinary assemblies, copies with every part at the exact mirror and
# locked, inserted where the feature outputs were; those (MirrorFemur-4, MirrorTibia-2) suppressed.
# Motion: LeftFemur locked to the left inner plate it is bolted to; LeftTibia pinned at the knee
# (to LeftFemur) and at E (to the left coupler) exactly as the right tibia -- the left coupler and
# inner plate are feature outputs that ARE exact mirrors and follow the right hip.
LT_ASM, LF_ASM = r"Links\LeftTibia.SLDASM", r"Links\LeftFemur.SLDASM"
TIBIA_D = {"EncoderCableClamp-1": 0}                 # everything else that moves: a screw, D1
TIBIA_FIX = ["EncoderCableClamp-1"] + [f"M3 Roundhead-{k}" for k in range(1, 12)] + ["M2 Roundhead-1", "M2 Roundhead-2"]
FEMUR_FIX = ["AK45-10 Rotor-1", "M2.5 Roundhead-1", "M2.5 Roundhead-2", "M2.5 Roundhead-3",
             "M3 Roundhead-1", "M3 Roundhead-2", "M3 Roundhead-3", "M3 Roundhead-4"]


def _copy_asm(sw, src_rel, dst_rel):
    dst = os.path.join(swlib.V5, dst_rel)
    if not os.path.exists(dst):
        d = open_asm(sw, src_rel)
        ok = d.Extension.SaveAs3(dst, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
        print(f"  {src_rel} -> {dst_rel}: copy {ok}")
    return open_asm(sw, dst_rel)


def _local_mirror(PR, k):
    """In the left sub-assembly's frame (same rotation as the right one, origin mirrored)."""
    return np.hstack([SZ @ PR[:, :3] @ D[k], (SZ @ PR[:, 3])[:, None]])


def _lock_all(sw, d, anchor):
    m = swmate.Mater(sw, d, prefix=MPFX)
    for n, cp in swmate.comps(d, True).items():
        if n != anchor and not cp.IsSuppressed():
            m.lock(f"{n.replace(' ', '')}_lock", n, anchor)


def left_tibia(sw):
    right = swmate.comps(open_asm(sw, r"Links\Tibia.SLDASM"), True)
    d = _copy_asm(sw, r"Links\MirrorTibia.SLDASM", LT_ASM)
    cl = swmate.comps(d, True)
    for n in TIBIA_FIX:
        Mt = _local_mirror(swlib.placement(right[n]), TIBIA_D.get(n, 1))
        print(f"  LeftTibia {n:22s} moves {np.abs(swlib.placement(cl[n])[:, 3] - Mt[:, 3]).max():7.3f} mm")
        place(sw, d, cl[n], Mt)
    _lock_all(sw, d, "MirrorTibia-1")
    d.EditRebuild3()
    print(f"  LeftTibia: mate errors {swmate.mate_errors(d)}, status {set(swmate.status(d).values())}")


def left_femur(sw):
    right = swmate.comps(open_asm(sw, r"Links\Femur.SLDASM"), True)
    d = _copy_asm(sw, r"Links\MirrorFemur.SLDASM", LF_ASM)
    for s in swmate.mates_of(d):                       # the copy's own (stale) mates
        d.ClearSelection2(True); s.Select2(False, 0); d.Extension.DeleteSelection2(0)
    d.ClearSelection2(True)
    cl = swmate.comps(d, True)
    if "Femur-1" in cl:                                # the rotated Femur.SLDPRT
        cl["Femur-1"].Select4(False, None, False); d.Extension.DeleteSelection2(0); d.ClearSelection2(True)
    cl = swmate.comps(d, True)
    mf = next((x for x in cl.values() if os.path.basename(x.GetPathName()).lower() == "mirrorfemur.sldprt"), None)
    if mf is None:
        mf = wrap(wrap(d, sld.IAssemblyDoc).AddComponent5(os.path.join(swlib.V5, MFEMUR),
                                                          c.swAddComponentConfigOptions_CurrentSelectedConfig, "",
                                                          False, "", 0, 0, 0), sld.IComponent2)
    place(sw, d, mf, np.hstack([np.eye(3), np.zeros((3, 1))]))     # = S . identity . (z mirror of the part)
    if not mf.IsFixed():
        d.ClearSelection2(True); mf.Select4(False, None, False); wrap(d, sld.IAssemblyDoc).FixComponent()
        d.ClearSelection2(True)
    cl = swmate.comps(d, True)
    used = []
    for n in FEMUR_FIX:
        Mt = _local_mirror(swlib.placement(right[n]), 1)
        stem = n.rsplit("-", 1)[0]
        ln = n if n in cl and n not in used and not n.startswith("M2.5") else \
            min((x for x in cl if x.rsplit("-", 1)[0] == stem and x not in used),
                key=lambda x: np.linalg.norm(swlib.placement(cl[x])[:, 3] - Mt[:, 3]))
        used.append(ln)
        print(f"  LeftFemur {ln:22s} <- {n:20s} moves {np.abs(swlib.placement(cl[ln])[:, 3] - Mt[:, 3]).max():7.3f} mm")
        place(sw, d, cl[ln], Mt)
    _lock_all(sw, d, mf.Name2)
    d.EditRebuild3()
    print(f"  LeftFemur: mate errors {swmate.mate_errors(d)}, status {set(swmate.status(d).values())}")


def left_leg(dry=False):
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    if dry:
        print(__doc__)
        return
    left_femur(sw)
    left_tibia(sw)
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    asm = wrap(robot, sld.IAssemblyDoc)
    top = swmate.comps(robot, True)
    new = {}
    for rel, old in ((LF_ASM, "MirrorFemur-4"), (LT_ASM, "MirrorTibia-2")):
        path = os.path.normcase(os.path.join(swlib.V5, rel))
        have = next((x for x in top.values() if os.path.normcase(x.GetPathName() or "") == path), None)
        T = swlib.placement(top[old])
        if have is None:
            have = wrap(asm.AddComponent5(os.path.join(swlib.V5, rel), c.swAddComponentConfigOptions_CurrentSelectedConfig,
                                          "", False, "", *(T[:, 3] / 1000.0)), sld.IComponent2)
        place(sw, robot, have, T)
        new[old] = have.Name2
        print(f"  ROBOT: {have.Name2} at {old}'s placement")
    for old in ("MirrorFemur-4", "MirrorTibia-2"):
        if not top[old].IsSuppressed():
            top[old].SetSuppression2(c.swComponentSuppressed)
            print(f"  ROBOT: {old} (the feature's output) suppressed")
    robot.EditRebuild3()
    m = swmate.Mater(sw, robot, prefix=MPFX)
    LF, LT = new["MirrorFemur-4"], new["MirrorTibia-2"]
    m.lock("LeftFemur_lock_FemurInside", LF, "MirrorFEMUR_INSIDE-2")
    # the right tibia's joints, mirrored: (right mate, left a, left b)
    joints = (("Coincident1", f"{LT}/Bearing 6804-2RS (20x32x7)-2", f"{LF}/MirrorFemur-1", "Knee_coinc"),
              ("Concentric2", f"{LT}/Bearing 6804-2RS (20x32x7)-2", f"{LF}/MirrorFemur-1", "Knee_conc"),
              ("Concentric4", f"{LT}/Bearing 6804-2RS (20x32x7)-4", "MirrorCOUPLER-2/MirrorCoupler-1", "E_conc"),
              ("Coincident11", f"{LT}/Bearing 6804-2RS (20x32x7)-4", "MirrorCOUPLER-2/MirrorCoupler-1", "E_coinc"))
    for rname, a, b, tag in joints:
        mt = wrap(wrap(asm.FeatureByName(rname), sld.IFeature).GetSpecificFeature2(), sld.IMate2)
        faces = []
        for i, leaf in enumerate((a, b)):
            pr = np.array(mt.MateEntity(i).EntityParams[:7], float)
            pt, ax = SZ @ (pr[:3] * 1000), SZ @ pr[3:6]
            if mt.Type == c.swMateCONCENTRIC:
                faces.append(m.cyl(leaf, ax, pt, r=pr[6] * 1000))
            else:
                faces.append(m.plane(leaf, ax, float(ax @ pt)))
        m.mate(tag, "concentric" if mt.Type == c.swMateCONCENTRIC else "coincident", faces[0], faces[1])
    robot.EditRebuild3()
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}; {LF} {swmate.status(robot).get(LF)}, "
          f"{LT} {swmate.status(robot).get(LT)}")


# ---- the limit switches ------------------------------------------------------------------------
# Right: the two M2x8 heads sat 0.5 mm INSIDE the switch body (25's seat ray landed on a shallow
# recess round the hole, smaller than the head): raised onto the switch face, locked to it.
# Left: BodyMirror places the switch as an origin-aligned instance, but the switch's symmetry
# plane does not pass through its origin, so it sat 2.45 mm off its mount along the screw
# axis -- and the left M2s, which follow the switch, came out head-for-tip.  Both placed at the
# exact mirror of the right ones (the switch about its own symmetry plane SW_SYM), locked.
SWM, MSWM = r"Body\LimitSwitch\SwicthMount.SLDASM", r"Body\LimitSwitch\MirrorSwicthMount.SLDASM"
R_SWM, L_SWM = "BODY-1/SIDE PANEL-1/SwicthMount-1", "MirrorBODY-2/MirrorSIDE PANEL-1/MirrorSwicthMount-1"
R_SW, L_SW = "LimitSwitch-2", "LimitSwitch-1"
R_MOUNT, L_MOUNT = "Switch_Mount-1", "MirrorSwitch_Mount-1"
# The switch is a bought part (lever, terminals): no mirror image of it exists, so the left one
# is the same switch turned over onto the mirrored mount -- reflected about the mid-plane of its
# body AT THE HOLES: local x 0 (the mount face, its origin) .. 5.7 (the head face), measured
# along both screw axes (OCC, 2026-10-08) -> x = 2.85.  BodyMirror had reflected it about its
# bounding-box middle (x 3.975, the box -0.1 .. 8.05 includes the lever): 2.25 mm off its mount
# along the screws (the "~2.45 mm" of the fasteners README), heads then buried in the switch.
SW_SYM = (0, 2.85)


def _reflect(k, c0):
    """4x4 reflection about the local plane x_k = c0."""
    m = np.eye(4)
    m[k, k] = -1.0
    m[k, 3] = 2.0 * c0
    return m


def switch(dry=False):
    with open(F.SCAN_ALL) as f:
        sc = json.load(f)
    by = {x["name"]: x for x in sc["comps"]}
    sw, _ = swlib.connect()
    robot = swlib.open_v5(sw)
    rc = swlib.components(robot)
    # 1. right: each M2 head onto the switch face (the highest switch material within the head)
    d = open_asm(sw, SWM)
    T = swlib.placement(rc[R_SWM])
    cp = swmate.comps(d, True)
    sw_rec = by[f"{R_SWM}/{R_SW}"]
    for n in ("M2 Roundhead-1", "M2 Roundhead-2"):
        P = h(T) @ h(swlib.placement(cp[n]))
        a = P[:3, 2]
        u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        face = max(w for k in range(8) for r in (1.2, 1.6)
                   for _, w in F.hits(sw_rec["step"], np.array(sw_rec["M"]),
                                      P[:3, 3] + r * (np.cos(k * np.pi / 4) * u + np.sin(k * np.pi / 4) * v), a, -3, 3))
        new = P.copy()
        new[:3, 3] += face * a
        Ml = (np.linalg.inv(h(T)) @ new)[:3]
        print(f"  right {n}: head face {face:+.3f} mm off the switch face -> raised")
        if dry or abs(face) < 1e-3:
            continue
        print(f"     mates deleted: {unmate(d, n)}")
        place(sw, d, cp[n], Ml)
        swmate.Mater(sw, d, prefix=MPFX).lock(f"{n.replace(' ', '').replace('Roundhead', '')}_lock_switch", n, R_SW)
    # 2. left: the switch and its screws at the exact mirror of the right ones
    if SW_SYM is None:
        raise SystemExit("SW_SYM (the switch's symmetry plane) not set")
    rc = swlib.components(robot)
    TL = swlib.placement(rc[L_SWM])
    dl = open_asm(sw, MSWM)
    cl = swmate.comps(dl, True)
    todo = [(L_SW, f"{R_SWM}/{R_SW}", _reflect(*SW_SYM))] + \
           [(n, f"{R_SWM}/{n}", h(np.hstack([np.diag([-1.0, 1, 1]), np.zeros((3, 1))]))) for n in
            ("M2 Roundhead-1", "M2 Roundhead-2")]
    for ln, rn, m in todo:
        PR = h(swlib.placement(swlib.components(robot)[rn]))
        target = (np.linalg.inv(h(TL)) @ h(np.hstack([SZ @ PR[:3, :3], (SZ @ PR[:3, 3])[:, None]])) @ m)[:3]
        now = swlib.placement(cl[ln])
        print(f"  left {ln}: moves {np.abs(now[:, 3] - target[:, 3]).max():.3f} mm, "
              f"rotation {np.abs(now[:, :3] - target[:, :3]).max():.3f}")
        if dry:
            continue
        print(f"     mates deleted: {unmate(dl, ln)}")
        place(sw, dl, cl[ln], target)
    if not dry:
        m_ = swmate.Mater(sw, dl, prefix=MPFX)
        m_.lock("LimitSwitch_lock_mount", L_SW, L_MOUNT)
        for n in ("M2 Roundhead-1", "M2 Roundhead-2"):
            m_.lock(f"{n.replace(' ', '').replace('Roundhead', '')}_lock_switch", n, L_SW)
        print(f"  {MSWM}: mate errors {swmate.mate_errors(dl)}")
    sw.ActivateDoc3(robot.GetTitle(), False, 0, 0)
    robot.EditRebuild3()
    print(f"  ROBOT mate errors {swmate.mate_errors(robot)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    dry = "--dry" in args
    if cmd == "leftleg":
        left_leg(dry)
    elif cmd == "femur":
        femur(dry)
    elif cmd == "switch":
        switch(dry)
    elif cmd == "brackets":
        brackets(dry)
    elif cmd == "ring":
        ring(dry)
    elif cmd == "features":       # features TibiaMirror FemurMirror --dry   (refuses for real)
        features([a for a in args[1:] if not a.startswith("--")], dry)
    elif cmd == "left":           # left [--dry]: the left tibia's clamp + encoder screws
        for rel in LEFT_FIX:
            left_fix(rel, dry)
    else:
        print(__doc__)
