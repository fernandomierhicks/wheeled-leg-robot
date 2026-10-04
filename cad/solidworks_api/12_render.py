"""Step 12: pictures of the styled assembly, straight out of SolidWorks.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/12_render.py

The hip at the three exported poses (Retracted -28, Middle +19.98, Extended
+57), each from outboard (*Front: the leg's show side is global +Z) and from
the trimetric; then the five styled parts on their own.  PNGs go to
out/renders/.  Puts the hip back.  Never saves.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
from swlib import c, wrap, sld

OUT = os.path.join(HERE, "out", "renders")
POSES = {"retracted": -28.0, "middle": 19.98, "extended": 57.0}
PARTS = {"Femur": r"Links\Femur.SLDPRT", "Coupler": r"Links\Coupler.SLDPRT",
         "Tibia": r"Links\Tibia.SLDPRT", "Side panel": r"Body\Side panel.SLDPRT",
         "RobotMount": r"Body\OldRobotBodyMount\RobotMount.SLDPRT"}


LEAF = {"Femur-1/Femur-1": r"Links\Femur.SLDPRT", "COUPLER-1/Coupler-1": r"Links\Coupler.SLDPRT",
        "Tibia-1/Tibia-1": r"Links\Tibia.SLDPRT"}


def zoom_to_parts(sw, model):
    """Zoom to the leg.  Zoom-to-fit -- and IComponent2.GetBox even with
    sketches excluded -- count the Tibia's +-1 m colour-split sketches (its box
    came back 2.7 m wide), so the styled links are measured from their BODIES
    in the part frame, moved by their placement; other components keep GetBox."""
    import numpy as np
    import swstyle as S
    comps = swlib.components(model)
    pts = []
    for name, rel in LEAF.items():
        d = next((wrap(x, sld.IModelDoc2) for x in sw.GetDocuments() or []
                  if os.path.normcase(wrap(x, sld.IModelDoc2).GetPathName())
                  == os.path.normcase(os.path.join(swlib.V5, rel))), None)
        if d is None or name not in comps:
            continue
        M = swlib.placement(comps[name])
        for b in S.bodies(d):
            x0, y0, z0, x1, y1, z1 = [v * 1000 for v in b.GetBodyBox()]
            for c3 in [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]:
                pts.append(M[:3, :3] @ np.array(c3) + M[:3, 3])
    asm = wrap(model, sld.IAssemblyDoc)
    for x in asm.GetComponents(True) or []:
        cp = wrap(x, sld.IComponent2)
        if cp.IsSuppressed() or cp.IsHidden(True) or cp.Name2 in swlib.NOT_FOR_COLLISION:
            continue
        boxes = []
        if cp.Name2 == "Tibia-1":         # its own box is inflated by the Tibia part:
            for ch in cp.GetChildren() or []:      # take the wheel, motor... instead
                ch = wrap(ch, sld.IComponent2)
                if not ch.Name2.endswith("/Tibia-1") and not ch.IsSuppressed():
                    boxes.append(ch.GetBox(False, False))
        else:
            boxes.append(cp.GetBox(False, False))
        for b in boxes:
            if b:
                pts += [np.array(b[:3]) * 1000, np.array(b[3:6]) * 1000]
    P = np.array(pts)
    lo, hi = P.min(0) / 1000, P.max(0) / 1000
    model.ViewZoomTo2(*lo, *hi)


def shot(sw, model, view, path):
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    # hide sketches, planes, points: the styling's 2 m "everything" box sketches
    # otherwise win zoom-to-fit and the leg renders as a speck
    model.Extension.SetUserPreferenceToggle(c.swViewDisplayHideAllTypes, 0, True)
    model.Extension.SetUserPreferenceToggle(c.swDisplaySketches, 0, False)
    model.Extension.SetUserPreferenceToggle(c.swDisplayPlanes, 0, False)
    model.GraphicsRedraw2()
    model.ShowNamedView2(view, -1)
    if model.GetType() == c.swDocASSEMBLY:
        zoom_to_parts(sw, model)
    else:
        model.ViewZoomtofit2()
    ok, err, warn = model.Extension.SaveAs3(path, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy,
                                            None, None, 0, 0)
    print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(path, HERE)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    hip = swlib.HipDriver(model)
    hip0 = hip.hip()
    for name, a in POSES.items():
        got = hip.set(a)
        for view in ("*Front", "*Trimetric"):
            shot(sw, model, view, os.path.join(OUT, f"assembly_{name}_{view.strip('*').lower()}.png"))
    hip.set(hip0)
    for part, rel in PARTS.items():
        path = os.path.join(swlib.V5, rel)
        d = next((wrap(x, sld.IModelDoc2) for x in sw.GetDocuments() or []
                  if os.path.normcase(wrap(x, sld.IModelDoc2).GetPathName()) == os.path.normcase(path)), None)
        if d is None:
            continue
        for view in ("*Front", "*Back", "*Trimetric"):
            shot(sw, d, view, os.path.join(OUT, f"part_{part.replace(' ', '_')}_{view.strip('*').lower()}.png"))
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)


if __name__ == "__main__":
    main()
