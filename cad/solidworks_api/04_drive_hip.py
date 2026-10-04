"""Step 4 of the SolidWorks automation ladder: move the leg, and prove it moved RIGHT.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/04_drive_hip.py

Works on cad/v5 Ai designed/ROBOT.SLDASM (his sandbox).  Drives the hip with
swlib.HipDriver (a helper angle mate -- the hip mate's own value cannot cross
0, see README gotcha 6), then checks:

  1. the femur, coupler and tibia land where the three v4 STEP exports put them
     (read by tools/kinematics.py) -- in millimetres, whole placement;
  2. what else follows the femur: the inner femur plate, and the mirrored leg.

Puts the hip back where it was.  Never saves.
"""
import os
import sys
import math
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "lib"))
sys.path.insert(0, os.path.join(HERE, "..", "aesthetics", "tools"))
import swlib

LINKS = {"femur": "Femur-1/Femur-1", "coupler": "COUPLER-1/Coupler-1",
         "tibia": "Tibia-1/Tibia-1"}
FOLLOWERS = ("FEMUR_INSIDE-1", "MirrorFemur-4", "MirrorFEMUR_INSIDE-2",
             "MirrorCOUPLER-2", "MirrorTibia-2", "COUPLER-1", "Tibia-1")


def angle(M):
    return math.degrees(math.atan2(M[1, 0], M[0, 0]))


def err_mm(A, B):
    """Placement difference as a distance: translation + rotation over 250 mm,
    the same measure tools/kinematics.py validates with."""
    return float(np.abs(A[:, 3] - B[:, 3]).max() + np.abs(A[:3, :3] - B[:3, :3]).max() * 250.0)


sw, sld = swlib.connect()
t0 = time.time()
model = swlib.open_v5(sw)
print(f"v5 ROBOT.SLDASM open, every reference inside v5 ({time.time() - t0:.1f} s)")
comps = swlib.components(model)
hip = swlib.HipDriver(model)
hip0 = hip.hip()
print(f"hip driver ready; leg is at hip {hip0:+.3f} deg")

# --- 1. land on the exported poses -----------------------------------------
import kinematics as K
print("\n1. reading the three v4 STEP exports (about a minute) ...")
leg = K.build(verbose=False)
keys = {"femur": leg.fk, "coupler": leg.ck, "tibia": leg.tk}
worst = 0.0
for pose in leg.poses:
    want = swlib._wrapd(math.degrees(leg._theta(pose, leg.fk)) + 180.0)
    t = time.time()
    got = hip.set(want)
    dt = time.time() - t
    row = []
    for k, name in LINKS.items():
        e = err_mm(swlib.placement(comps[name]), leg.P[pose][keys[k]])
        worst = max(worst, e)
        row.append(f"{k} {e:7.4f}")
    print(f"   {pose:<20s} hip {want:+7.2f} (got {got:+7.2f})  ->  " + "  ".join(row)
          + f"  mm   ({dt:.2f} s)")
print(f"   worst {worst:.4f} mm -> "
      + ("SOLIDWORKS REPRODUCES THE STEP EXPORTS" if worst < 0.05 else "MISMATCH"))

# --- 2. what follows the femur ---------------------------------------------
print("\n2. what moves when the hip goes -28 -> +57?")
hip.set(-28.0)
a = {n: swlib.placement(comps[n]) for n in FOLLOWERS}
hip.set(57.0)
b = {n: swlib.placement(comps[n]) for n in FOLLOWERS}
for n in FOLLOWERS:
    turn = swlib._wrapd(angle(b[n]) - angle(a[n]))
    shift = float(np.linalg.norm(b[n][:, 3] - a[n][:, 3]))
    print(f"   {n:22s} turned {turn:+8.3f} deg, origin moved {shift:7.2f} mm")

hip.set(hip0)
print(f"\nhip restored to {hip.hip():+.3f} deg; nothing saved")
