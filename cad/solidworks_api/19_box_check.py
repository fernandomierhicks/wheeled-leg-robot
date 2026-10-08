"""Step 19: check the body box in the v5 robot.  Changes nothing, never saves.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/19_box_check.py

The box (cad/v5 Ai designed/Box, imported from v3 on 2026-10-04, README "The
body box") sits between the two RobotMounts, which ARE its side walls.  This
reports, on what is loaded from disk:

  1. constraint status and mate errors in Box.SLDASM and every sub-assembly
     (expected: all fully defined, except SupportWheel's spin on its axle)
  2. live in-context references in any box part (expected: none -- they were
     broken on import; broken ones are still listed by SolidWorks, status 0)
  3. Box-1 in ROBOT.SLDASM: fully defined, 0 mate errors
  4. the side-panel screw pattern: each of the 10 RobotMount box holes against
     the nearest box bracket screw, right side measured, left side against the
     right RobotMount mirrored about the robot's mirror plane (the left leg is
     suppressed), plus the gap between each box side face and its RobotMount
"""
import os
import sys
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swmate as M
from swlib import c, wrap, sld

BOXD = os.path.join(swlib.V5, "Box")
RM = "BODY-1/SIDE PANEL-1/RobotMount-1"
SUBS = ["Box.SLDASM", "BottomPanelWithAvionics.SLDASM", "CornerBracket.SLDASM", r"Electronics\CustomBoard.SLDASM"]
ST = {0: "broken", 1: "locked", 3: "in-context", 4: "out-of-context", 5: "dangling"}


def doc_of(sw, path):
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    return None


def main():
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    if "Box-1" not in M.comps(model, top=True):
        raise SystemExit("no Box-1 in ROBOT.SLDASM")

    print("1. box assemblies")
    for rel in SUBS:
        d = doc_of(sw, os.path.join(BOXD, rel))
        if d is None:
            print(f"   {rel}: not loaded")
            continue
        st = M.status(d)
        under = [n for n in st if st[n] != "fully"]
        print(f"   {rel:34s} {dict(collections.Counter(st.values()))}  errors {M.mate_errors(d)}"
              + (f"  not fully: {under}" if under else ""))

    print("2. live in-context references in box documents (broken ones are listed by SolidWorks too)")
    live = 0
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        p = d.GetPathName()
        if not os.path.normcase(p).startswith(os.path.normcase(BOXD)) or "Electronics" in p:
            continue
        if d.Extension.ListExternalFileReferencesCount():
            r = d.Extension.ListExternalFileReferences()
            n = sum(1 for s in r[4] if s in (3, 4))               # in-context / out-of-context
            other = collections.Counter(ST.get(s, s) for s in r[4])
            live += n
            print(f"   {os.path.basename(p):34s} live {n}  ({dict(other)})")
    print(f"   {live} live  (the 'dangling' ones are the IMU / resistor vendor STEP-import links)")

    print("3. Box-1 in ROBOT")
    box_mates = [s.Name for s in M.mates_of(model) if any(x.startswith("Box-1") for x in M.mate_components(s))]
    print(f"   {M.CS.get(M.comps(model)['Box-1'].GetConstrainedStatus())}, mates {box_mates}, "
          f"ROBOT mate errors {M.mate_errors(model)}")

    print("4. side-panel screw pattern")
    m = M.Mater(sw, model)
    z_in = swlib.placement(M.comps(model)[RM])[2, 3]
    holes = []
    for f in m.faces(RM):
        if f["kind"] == "cyl" and abs(f["r"] - 1.70) < 0.01 and abs(abs(f["axis"][2]) - 1) < 1e-6:
            if not any(np.allclose(f["pt"][:2], h, atol=0.01) for h in holes):
                holes.append(f["pt"][:2])
    # every CornerBracket sub-assembly (box level and inside BottomPanelWithAvionics)
    brackets = [n for n in m._comps if n.startswith("Box-1/") and n.split("/")[-1].startswith("CornerBracket-")
                and m._comps[n].GetChildren()]
    screws = []
    for n in brackets:
        for f in m.faces(n):
            if f["kind"] == "cyl" and f["r"] <= 1.8 and abs(abs(f["axis"][2]) - 1) < 1e-6:
                screws.append((n.replace("Box-1/", ""), f["pt"][:2], f["aabb"]))
    floor = [f for f in m.faces("Box-1/BottomPanelWithAvionics-1/BottomPanel-1")
             if f["kind"] == "plane" and abs(abs(f["n"][2]) - 1) < 1e-6]
    zr = max(f["off"] * f["n"][2] for f in floor)
    zl = min(f["off"] * f["n"][2] for f in floor)
    worst = {}
    for tag, sel in (("right", lambda s: s[2][1][2] > zr - 25), ("left", lambda s: s[2][0][2] < zl + 25)):
        cand = [s for s in screws if sel(s)]
        worst[tag] = 0.0
        print(f"   {tag}:")
        for h in holes:
            n, pt, _ = min(cand, key=lambda s: np.linalg.norm(s[1] - h))
            d = float(np.linalg.norm(pt - h))
            worst[tag] = max(worst[tag], d)
            print(f"     hole ({h[0]:8.2f},{h[1]:7.2f})  {n:42s} {d:7.4f} mm")
    print(f"   box side faces z = {zr:.3f} / {zl:.3f} (mid-plane {(zr + zl) / 2:.3f}); "
          f"RobotMount inner faces z = {z_in:.3f} / {-z_in:.3f}")
    print(f"   gaps: right {z_in - zr:+.3f} mm, left {zl + z_in:+.3f} mm;  "
          f"worst hole offset: right {worst['right']:.4f}, left {worst['left']:.4f} mm")


if __name__ == "__main__":
    main()
