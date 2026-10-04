"""Step 18: lock the TPU tyre to the rim -- axial ribs on the rim, grooves in the tyre.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/18_tire_lock.py [--redo]

The tyre (Motor/Wheel Motor/TPU wheel) is a 10 mm TPU ring pressed onto the
rim's outer band (Wheel), 0.7 mm radial interference, and nothing else holds
it in rotation: friction alone, and it slips (his report, 2026-10-03).

  rim    36 axial RIBS on the band, one on each spoke root (the band is 0.85 mm
         thick and the 36 S-spokes meet it at 2.38 + k*10 deg, so torque goes
         spoke -> rib -> tyre and never bends the band between spokes).
         RIB_W wide, RIB_H tall, from the bottom lip (z 0) to RIB_Z1; the top
         end is a RAMP (revolve cut) so the tyre, pushed on from the top over
         the wedge lip, rides up and drops onto the ribs.
  tyre   36 axial GROOVES through its bore, same angles, RIB_W wide (line to
         line), floor level with the rib tops (TOP_FIT 0): the press fit stays
         on the band between the grooves, where it grips.  The floor cannot go
         much deeper: the tyre's end faces are flat only to r 48.0 before the
         shoulder, so a floor past it notches the corner at both ends.
  asm    TL_TyreLock: Right Plane of the tyre coincident with the rim's.  The
         tyre's rotation was free (only Concentric4 + Coincident8), so without
         it nothing puts the grooves on the ribs.  The 36-fold pattern makes
         the aligned and the flipped solution equally right.

Every feature is named TL_*; --redo deletes them first (the mate is kept).
Backs the three files up to _originals/ once.  Saves NOTHING: run
16_check_and_save.py afterwards.  All lengths mm, part-local: the axis is
local Z through the origin in both parts, and z is the same in both.
"""
import os
import sys
import math
import shutil
from shapely.geometry import Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swstyle as S
from swlib import c, wrap, sld

RIM = r"Motor\Wheel Motor\Wheel.SLDPRT"
TYRE = r"Motor\Wheel Motor\TPU wheel.SLDPRT"
ASM = r"Motor\Wheel Motor\WheelMotorASM.SLDASM"

N = 36              # ribs = spokes
A0 = 2.38           # deg, first spoke root in the rim's frame (measured)
BAND_OD = 46.70     # rim band outer radius
BAND_ID = 45.85     # rim band inner radius (the spokes start inside this)
BORE = 46.00        # tyre bore radius
FIT = BAND_OD - BORE    # 0.70 radial press fit of the tyre on the band (as designed)
TOP_FIT = 0.0       # radial interference at the rib tops: 0 = groove floor level with them.
                    # 0.70 (= FIT) was the first try: in CAD the rib then stood 0.7 mm into
                    # the TPU (his look, 2026-10-03); stretched on, ~line to line.  At 0 the
                    # stretched tyre leaves ~0.7 mm over the rib tops: torque goes through
                    # the flanks either way, and it presses on more easily.
RIB_W = 3.0         # rib width = groove width (line to line)
RIB_H = 1.2         # rib height above the band
RIB_CH = 0.3        # chamfer on the rib's two top corners
RIB_Z1 = 26.0       # rib top end (the wedge lip starts at z 26.9)
RAMP = 2.6          # axial length of the lead-in ramp at the top end


def rect(u0, u1, w, a_deg, ch=0.0):
    """A rectangle u0..u1 along the radius at angle a, w wide, top corners chamfered."""
    h = w / 2
    pts = [(u0, -h), (u1 - ch, -h), (u1, -h + ch), (u1, h - ch), (u1 - ch, h), (u0, h)]
    a = math.radians(a_deg)
    ca, sa = math.cos(a), math.sin(a)
    return Polygon([(u * ca - v * sa, u * sa + v * ca) for u, v in pts])


def ribs():
    # inner end inside the band wall (its corners at r 46.22 > BAND_ID): merges,
    # and nothing pokes into the spoke gaps
    return unary_union([rect(BAND_OD - 0.5, BAND_OD + RIB_H, RIB_W, A0 + k * 360 / N, RIB_CH)
                        for k in range(N)])


def grooves():
    return unary_union([rect(BORE - 0.5, BAND_OD + RIB_H - TOP_FIT, RIB_W, A0 + k * 360 / N)
                        for k in range(N)])


def _doc(sw, rel, kind=c.swDocPART):
    path = os.path.join(swlib.V5, rel)
    d, err, warn = sw.OpenDoc6(path, kind, c.swOpenDocOptions_Silent, "", 0, 0)
    if d is None:
        raise SystemExit(f"cannot open {path} (err {err})")
    m = wrap(d, sld.IModelDoc2)
    sw.ActivateDoc3(m.GetTitle(), False, 0, 0)
    return m


def _strip(model):
    n = 0
    for _ in range(3):
        model.ClearSelection2(True)
        k = 0
        for f in list(S._iter_features(model)):
            if f.Name.startswith("TL_") and f.Select2(k > 0, 0):
                k += 1
        if not k:
            break
        model.Extension.DeleteSelection2(c.swDelete_Absorbed | c.swDelete_Children)
        n += k
    model.ClearSelection2(True)
    model.EditRebuild3()
    return n


def ramp(model):
    """Revolve cut over the rib tops: above the line (BAND_OD, RIB_Z1) ->
    (BAND_OD + RIB_H, RIB_Z1 - RAMP).  Top Plane sketch coordinates are
    (X, -Z) (README gotcha 17); the centreline is the part Z axis."""
    r0, r1, ro = BAND_OD + 0.005, BAND_OD + RIB_H + 0.05, BAND_OD + RIB_H + 0.8
    z0, z1 = RIB_Z1 + 0.05, RIB_Z1 - RAMP * (r1 - r0) / RIB_H
    prof = [(r0, z0), (r1, z1), (ro, z1), (ro, z0)]
    model.ClearSelection2(True)
    if not model.Extension.SelectByID2("Top Plane", "PLANE", 0, 0, 0, False, 0, None, 0):
        raise S.FeatureFailed("cannot select Top Plane")
    sm = model.SketchManager
    sm.InsertSketch(True)
    sm.AddToDB = True
    try:
        for (ra, za), (rb, zb) in zip(prof, prof[1:] + prof[:1]):
            sm.CreateLine(ra * S.MM, -za * S.MM, 0, rb * S.MM, -zb * S.MM, 0)
        sm.CreateCenterLine(0, 0, 0, 0, -40 * S.MM, 0)
    finally:
        sm.AddToDB = False
        sm.InsertSketch(True)
    sk = S.last_feature(model)
    sk.Name = "TL_RampSk"
    model.ClearSelection2(True)
    sk.Select2(False, 0)
    f = model.FeatureManager.FeatureRevolve2(True, True, False, True, False, False,
                                             c.swEndCondBlind, 0, 2 * math.pi, 0, False, False,
                                             0, 0, 0, 0, 0, True, False, True)
    model.ClearSelection2(True)
    if f is None:
        raise S.FeatureFailed("ramp revolve cut refused")
    wrap(f, sld.IFeature).Name = "TL_RibRamp"


def _mates(a):
    f = wrap(a.FirstFeature(), sld.IFeature)
    while f is not None:
        if f.GetTypeName2() == "MateGroup":
            s = wrap(f.GetFirstSubFeature(), sld.IFeature)
            while s is not None:
                yield s
                s = wrap(s.GetNextSubFeature(), sld.IFeature)
        f = wrap(f.GetNextFeature(), sld.IFeature)


def _plane_mate(a, plane, name):
    """Coincident mate between the same datum plane of tyre and rim."""
    asm = wrap(a, sld.IAssemblyDoc)
    if asm.FeatureByName(name) is not None:
        print(f"  {name} already there")
        return
    title = os.path.splitext(os.path.basename(ASM))[0]
    a.ClearSelection2(True)
    ok = (a.Extension.SelectByID2(f"{plane}@TPU wheel-1@{title}", "PLANE", 0, 0, 0, False, 1, None, 0) and
          a.Extension.SelectByID2(f"{plane}@Wheel-1@{title}", "PLANE", 0, 0, 0, True, 1, None, 0))
    if not ok:
        raise SystemExit(f"could not select the two {plane}s")
    res = asm.AddMate5(c.swMateCOINCIDENT, c.swMateAlignCLOSEST, False, 0, 0, 0, 0, 0, 0, 0, 0,
                       False, False, 0)
    mate, err = res if isinstance(res, tuple) else (res, None)
    a.ClearSelection2(True)
    if mate is None or err not in (None, c.swAddMateError_NoError):
        raise SystemExit(f"AddMate5 ({name}) failed, error {err}")
    *_, last = _mates(a)
    last.Name = name
    a.EditRebuild3()
    print(f"  added {name} (tyre {plane} coincident with the rim's)")


def lock_mate(sw):
    """TL_TyreLock: Right Planes (rotation).  TL_TyreSeat: Front Planes (both
    parts' seat is local z 0) -- it REPLACES his Coincident8, tyre bottom face
    to a rim edge, which the grooves break (error 51: they cut through that
    face's inner boundary).  A datum mate does not depend on tyre topology."""
    a = _doc(sw, ASM, c.swDocASSEMBLY)
    _plane_mate(a, "Right Plane", "TL_TyreLock")
    _plane_mate(a, "Front Plane", "TL_TyreSeat")
    a.ForceRebuild3(False)
    for m in list(_mates(a)):
        code, _ = m.GetErrorCode2()
        if m.Name == "Coincident8" and code != 0:
            a.ClearSelection2(True)
            m.Select2(False, 0)
            a.Extension.DeleteSelection2(0)
            a.ClearSelection2(True)
            print(f"  deleted Coincident8 (error {code}; replaced by TL_TyreSeat)")
    a.ForceRebuild3(False)
    bad = [(m.Name, m.GetErrorCode2()[0]) for m in _mates(a)
           if m.GetErrorCode2()[0] != 0 and not m.IsSuppressed()]
    if bad:
        raise SystemExit(f"mates in error in {ASM}: {bad}")
    return a


def main(redo):
    for rel in (RIM, TYRE, ASM):
        bak = os.path.join(swlib.V5, "_originals", rel)
        if not os.path.exists(bak):
            os.makedirs(os.path.dirname(bak), exist_ok=True)
            shutil.copy2(os.path.join(swlib.V5, rel), bak)
            print(f"  backed up {rel}")
    sw, _ = swlib.connect()
    for rel, what in ((RIM, "ribs"), (TYRE, "grooves")):
        m = _doc(sw, rel)
        if any(f.Name.startswith("TL_") for f in S._iter_features(m)):
            if not redo:
                raise SystemExit(f"{rel} already has TL_ features -- use --redo")
            print(f"  --redo: deleted {_strip(m)} TL_ features from {rel}")
        v0 = S.total_volume(m)
        if what == "ribs":
            sk = S.sketch(m, ribs(), name="TL_RibsSk", grow=0.0)
            S.boss(m, sk, 0.0, RIB_Z1, name="TL_Ribs")
            ramp(m)
        else:
            sk = S.sketch(m, grooves(), name="TL_GroovesSk")
            S.cut(m, sk, name="TL_Grooves")
        m.EditRebuild3()
        bs = S.bodies(m)
        print(f"  {rel}: {what}, {sk.__dict__.get('n_lines', 0)} sketch lines, "
              f"{v0:.1f} -> {S.total_volume(m):.1f} mm3 ({S.total_volume(m) - v0:+.1f}), {len(bs)} body")
        if len(bs) != 1:
            raise SystemExit(f"{rel}: {len(bs)} bodies -- a rib or groove came out detached")
    a = lock_mate(sw)
    model = swlib.open_v5(sw)
    model.ForceRebuild3(False)
    comps = swlib.components(model)
    rim = [n for n in comps if n.endswith("WheelMotorASM-1/Wheel-1")][0]
    tyre = [n for n in comps if n.endswith("WheelMotorASM-1/TPU wheel-1")][0]
    R1, R2 = swlib.placement(comps[rim])[:, :3], swlib.placement(comps[tyre])[:, :3]
    rel_ = R1.T @ R2
    ang = math.degrees(math.atan2(rel_[1, 0], rel_[0, 0]))
    d = ang % (360 / N)
    on = rel_[2, 2] > 0.999 and min(d, 360 / N - d) < 1e-3
    print(f"  tyre frame vs rim frame: {ang:+.3f} deg about the axis -> grooves "
          f"{'ON the ribs' if on else 'OFF THE RIBS'}")
    if not on:
        raise SystemExit("the tyre's grooves do not sit on the rim's ribs")


if __name__ == "__main__":
    main("--redo" in sys.argv[1:])
