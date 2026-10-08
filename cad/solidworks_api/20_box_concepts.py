"""Step 20: the body-box concepts on the real robot.  Never edits or saves ROBOT or Box.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/20_box_concepts.py import <step_dir>
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/20_box_concepts.py assemble
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/20_box_concepts.py render
    (any of them with --names A_helm,D_facet to pick concepts; default A_helm,B_carapace,C_shells)

  import    BoxConcept_<n>.step (from cad/aesthetics/parts/box_concepts.py) ->
            v5/Box/Concepts/BoxConcept_<n>.SLDPRT (colours come through the STEP)
  assemble  v5/Box/Concepts/Concept_<n>.SLDASM = ROBOT.SLDASM + the shell, both
            fixed at the origin (the shells are built in ROBOT coordinates).  The
            box panels a shell replaces are hidden in the CONCEPT assembly's
            display state, so ROBOT and Box are not touched.
  render    PNGs of the current robot and every concept, whole robot and body
            close-ups, into out/renders/box_concepts/

Framing (both paid for, 2026-10-04):
  * zoom-to-fit counts the styled Tibia's +-1 m colour-split sketches: hide that
    part while fitting, show it again before the capture
  * for close-ups, zoom to a SELECTION of sketch-free parts that bound the body
    (the shell, the v3 cap/floor/panels, the hip stator).  Selecting BODY-1 pulls
    in the styled plates' 2 m styling sketches; ViewZoomTo2 frames the right spot
    only in *Front (gotcha 33) -- back-projecting the box did not fix it.
Left side: the mirrored leg is suppressed in v5, so the views are from the right.
"""
import os
import sys
import shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swmate as M
from swlib import c, wrap, sld

BOXD = os.path.join(swlib.V5, "Box")
BOXP = os.path.join(BOXD, "Box.SLDASM")
CON = os.path.join(BOXD, "Concepts")
NAMES = ["A_helm", "B_carapace", "C_shells"]
HIDE = ["BackPanel-1", "FrontPanel-1", "Cap-1", "NeoPixelCage-1", "NeopixelSupport-1", "Neopixel-1",
        "TPU_protector-1", "Cushion_support-1", "Cushion_support-2", "Lid-1"]
OUT = os.path.join(HERE, "out", "renders", "box_concepts")
VIEWS = {"iso": "*Isometric", "side": "*Front", "face": "*Right", "back": "*Left", "top": "*Top"}
LEGS = ("Femur-1", "FEMUR_INSIDE-1", "COUPLER-1", "Tibia-1", "AK_SIM-1", "WheelHanger-1")
BOUND = ("BoxConcept", "Box-1/Cap-1", "BottomPanelWithAvionics-1/BottomPanel-1", "Box-1/FrontPanel-1",
         "Box-1/BackPanel-1", "AK45-10 Stator-1")


def find_doc(sw, path):
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    return None


def dirty_guard(sw):
    return [os.path.basename(p) for p in (swlib.ASM, BOXP)
            if find_doc(sw, p) is not None and find_doc(sw, p).GetSaveFlag()]


def do_import(sw, src):
    os.makedirs(CON, exist_ok=True)
    for n in NAMES:
        step = os.path.join(CON, f"BoxConcept_{n}.step")
        shutil.copy2(os.path.join(src, f"BoxConcept_{n}.step"), step)
        prt = os.path.join(CON, f"BoxConcept_{n}.SLDPRT")
        if os.path.exists(prt):
            print("  exists", os.path.relpath(prt, swlib.V5))
            continue
        res = sw.LoadFile4(step, "r", None, 0)
        d = wrap(res[0] if isinstance(res, tuple) else res, sld.IModelDoc2)
        ok, err, warn = d.Extension.SaveAs3(prt, 0, c.swSaveAsOptions_Silent, None, None, 0, 0)
        n_b = len(wrap(d, sld.IPartDoc).GetBodies2(c.swSolidBody, False) or [])
        print(f"  {n}: {n_b} bodies, saved {ok}")
        sw.CloseDoc(d.GetTitle())


def do_assemble(sw):
    swlib.open_v5(sw)                                     # AddComponent5 needs ROBOT loaded
    tmpl = sw.GetUserPreferenceStringValue(c.swDefaultTemplateAssembly)
    for n in NAMES:
        path = os.path.join(CON, f"Concept_{n}.SLDASM")
        if os.path.exists(path):
            print("  exists", os.path.relpath(path, swlib.V5))
            continue
        prt = os.path.join(CON, f"BoxConcept_{n}.SLDPRT")
        if find_doc(sw, prt) is None:                     # AddComponent5 needs it loaded
            sw.OpenDoc6(prt, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
        d = wrap(sw.NewDocument(tmpl, 0, 0, 0), sld.IModelDoc2)
        sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
        asm = wrap(d, sld.IAssemblyDoc)
        for p in (swlib.ASM, prt):
            cp = wrap(asm.AddComponent5(p, 0, "", False, "", 0.0, 0.0, 0.0), sld.IComponent2)
            M.set_placement(sw, cp, np.hstack([np.eye(3), np.zeros((3, 1))]))
            d.ClearSelection2(True)
            cp.Select4(False, None, False)
            asm.FixComponent()
        d.ClearSelection2(True)
        d.EditRebuild3()
        hidden = 0
        for name, cp in M.comps(d).items():
            parts = name.split("/")
            if len(parts) == 3 and parts[1] == "Box-1" and (parts[2] in HIDE or parts[2].startswith("M3x8 flathead")):
                cp.Visible = c.swComponentHidden
                hidden += 1
        ok, err, warn = d.Extension.SaveAs3(path, 0, c.swSaveAsOptions_Silent, None, None, 0, 0)
        print(f"  {n}: hid {hidden} box parts, saved {ok}")


def _prep(sw, d, view):
    sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
    d.Extension.SetUserPreferenceToggle(c.swViewDisplayHideAllTypes, 0, True)
    d.ViewDisplayShaded()                    # no edges: the sampled shells read as hatching
    d.ShowNamedView2(view, -1)


def _save(d, path):
    d.GraphicsRedraw2()
    return d.Extension.SaveAs3(path, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)[0]


def whole(sw, d, view, path):
    _prep(sw, d, view)
    tib = [cp for n, cp in M.comps(d).items() if n.endswith("Tibia-1/Tibia-1")]
    for cp in tib:
        cp.Visible = c.swComponentHidden
    d.ViewZoomtofit2()
    for cp in tib:
        cp.Visible = c.swComponentVisible
    return _save(d, path)


def body(sw, d, view, path):
    _prep(sw, d, view)
    d.ClearSelection2(True)
    first = True
    for n, cp in M.comps(d).items():
        if any(k in n for k in BOUND) and cp.Visible and not cp.IsSuppressed() and not cp.GetChildren():
            cp.Select4(not first, None, False)
            first = False
    d.ViewZoomToSelection()
    d.ClearSelection2(True)
    return _save(d, path)


def do_render(sw):
    os.makedirs(OUT, exist_ok=True)
    for tag, p in [("current", swlib.ASM)] + [(n, os.path.join(CON, f"Concept_{n}.SLDASM")) for n in NAMES]:
        d = find_doc(sw, p) or wrap(sw.OpenDoc6(p, c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)[0],
                                    sld.IModelDoc2)
        for k, v in VIEWS.items():
            whole(sw, d, v, os.path.join(OUT, f"{tag}_{k}.png"))
        legs = [cp for n, cp in M.comps(d).items()
                if (n.count("/") == 1 and n.split("/")[1] in LEGS) or (n.count("/") == 0 and n in LEGS)]
        vis0 = [cp.Visible for cp in legs]
        for cp in legs:
            cp.Visible = c.swComponentHidden
        for k, v in VIEWS.items():
            body(sw, d, v, os.path.join(OUT, f"{tag}_body_{k}.png"))
        for cp, v0 in zip(legs, vis0):
            cp.Visible = v0
        print(f"  {tag}: rendered")
    print("  ROBOT / Box dirty in memory (do NOT save them):", dirty_guard(sw))


if __name__ == "__main__":
    for a in sys.argv:
        if a.startswith("--names="):
            NAMES[:] = a.split("=", 1)[1].split(",")
    sys.argv = [a for a in sys.argv if not a.startswith("--names=")]
    sw, _ = swlib.connect()
    cmd = sys.argv[1]
    if cmd == "import":
        do_import(sw, sys.argv[2])
    else:
        {"assemble": do_assemble, "render": do_render}[cmd](sw)
