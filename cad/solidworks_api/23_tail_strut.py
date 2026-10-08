"""Step 23: the TAIL STRUT replaces the v3 BackWheelSupport under the box.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/aesthetics/parts/tail_strut.py "cad/v5 Ai designed/Box/TailStrut" --3mf cad/solidworks_api/out/print
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/23_tail_strut.py import
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/23_tail_strut.py install
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/23_tail_strut.py check
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/23_tail_strut.py save
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/23_tail_strut.py render

  import   v5/Box/TailStrut/TailStrut.step (built by cad/aesthetics/parts/tail_strut.py in
           the ROBOT d75 frame) -> TailStrut.SLDPRT
  install  in BottomPanelWithAvionics.SLDASM, in memory: TailStrut-1 inserted exactly where
           it was built; every mate touching BackWheelSupport-1 recorded (geometry in the
           sub-assembly frame); BackWheelSupport-1 suppressed (kept, his rule); each mate
           re-made on TailStrut-1's identical face (the strut keeps the v3 top face, the
           4 mount holes, the axle holes and the fork's inner faces).  TS_* names, every
           mate refused if it moves anything (swmate).
  check    mates (BottomPanelWithAvionics, Box, ROBOT), constraint status, the caster
           unmoved, interference of TailStrut-1 with everything in Box
  save     TailStrut.SLDPRT AND every assembly above it (BottomPanelWithAvionics, Box,
           ROBOT) through 16_check_and_save.chain; refused on any mate error / an
           under-defined strut (not Save All: ~30 parts come up dirty on load)
  render   rear, side, below, on the robot
"""
import os
import sys
import numpy as np
from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swmate as M
from swlib import c, wrap, sld

F = import_module("21_box_facet")                 # quiet, find_doc, record_mates, match
DIR = os.path.join(swlib.V5, "Box", "TailStrut")
STEP = os.path.join(DIR, "TailStrut.step")
PRT = os.path.join(DIR, "TailStrut.SLDPRT")
SUB = os.path.join(swlib.V5, "Box", "BottomPanelWithAvionics.SLDASM")
SUB_IN_ROBOT = "Box-1/BottomPanelWithAvionics-1"
OLD, NEW = "BackWheelSupport-1", "TailStrut-1"
CASTER = "SupportWheel-1"
OUT = os.path.join(HERE, "out")


def do_import(sw):
    if os.path.exists(PRT):
        print("  exists", os.path.relpath(PRT, swlib.V5))
        return
    res = sw.LoadFile4(STEP, "r", None, 0)
    d = wrap(res[0] if isinstance(res, tuple) else res, sld.IModelDoc2)
    if d is None or d.GetType() != c.swDocPART:
        raise SystemExit(f"STEP did not import as a part ({None if d is None else d.GetType()})")
    ok, err, warn = d.Extension.SaveAs3(PRT, 0, c.swSaveAsOptions_Silent, None, None, 0, 0)
    bodies = wrap(d, sld.IPartDoc).GetBodies2(c.swSolidBody, False) or []
    vol = sum(wrap(b, sld.IBody2).GetMassProperties(1.0)[3] for b in bodies) * 1e9
    print(f"  TailStrut: {len(bodies)} bodies, {vol / 1000:.2f} cm3, saved {ok}")


def sub_doc(sw):
    """The sub-assembly in its own window; its placement in ROBOT read first."""
    robot = swlib.open_v5(sw)
    P = swlib.placement(swlib.components(robot)[SUB_IN_ROBOT])
    sw.OpenDoc6(SUB, c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)
    d = F.find_doc(sw, SUB)
    if d is None:
        raise SystemExit(f"could not open {SUB}")
    sw.ActivateDoc3(d.GetTitle(), False, 0, 0)
    return d, P


def transfer(m, rec):
    made, skipped = [], []
    for r in rec:
        sides, ok = [], True
        for cn, g in r["ents"]:
            if cn == "datum":
                sides.append(g)
            elif cn.split("/")[0] == OLD:
                f = F.match(m, NEW, g)
                ok = ok and f is not None
                sides.append(f)
            else:
                ok = ok and g is not None
                sides.append(g)
        if not ok or r["kind"] is None:
            skipped.append((r["name"], "no counterpart" if r["kind"] else "type"))
            continue
        nm = "TS_" + r["name"].replace("BX_", "").replace("WheelSupport", "Strut")
        try:
            m.mate(nm, r["kind"], sides[0], sides[1], dist=r["dist"], lock=r["lock"])
            made.append(nm)
        except (RuntimeError, SystemExit) as e:
            skipped.append((r["name"], str(e)[:90]))
    return made, skipped


def install(sw):
    doc, P_sub = sub_doc(sw)
    with F.quiet(sw, doc):
        _install(sw, doc, P_sub)
    doc.EditRebuild3()
    report(doc)


def _install(sw, doc, P_sub):
    asm = wrap(doc, sld.IAssemblyDoc)
    comps = M.comps(doc, top=True)
    caster0 = swlib.placement(comps[CASTER])
    print("1. insert the strut where it was built (part frame = ROBOT d75 frame)")
    if NEW in comps:
        print(f"   {NEW} already in")
    else:
        if F.find_doc(sw, PRT) is None:
            sw.OpenDoc6(PRT, c.swDocPART, c.swOpenDocOptions_Silent, "", 0, 0)
            sw.ActivateDoc3(doc.GetTitle(), False, 0, 0)
        cp = wrap(asm.AddComponent5(PRT, 0, "", False, "", 0.0, 0.0, 0.0), sld.IComponent2)
        R, t = P_sub[:, :3], P_sub[:, 3]
        M.set_placement(sw, cp, np.hstack([R.T, -(R.T @ t)[:, None]]))
        print(f"   + {cp.Name2}")
    doc.EditRebuild3()
    m = M.Mater(sw, doc, prefix="")
    print("2. record the v3 support's mates, suppress it, re-make each mate on the strut")
    F.OLD = {OLD: NEW}
    rec = F.record_mates(m, doc)
    for r in rec:
        print(f"   {r['name']:40s} {r['kind']:10s} lock={r['lock']} dist={r['dist']:.3f}")
    ref = M.placements(doc)
    comps = M.comps(doc, top=True)
    if not comps[OLD].IsSuppressed():
        comps[OLD].SetSuppression2(c.swComponentSuppressed)
    doc.EditRebuild3()
    moved = M.compare({k: v for k, v in ref.items() if not k.startswith(OLD)}, M.placements(doc))[0]
    print(f"   suppressed {OLD}; moved by it: {moved[:5]}")
    m = M.Mater(sw, doc, prefix="")
    made, skipped = transfer(m, rec)
    print(f"   re-made {len(made)}, not re-made {len(skipped)}" + "".join(f"\n      - {a}: {b}" for a, b in skipped))
    d = np.abs(swlib.placement(M.comps(doc, top=True)[CASTER]) - caster0).max()
    print(f"   caster moved {d:.6f} mm")


def report(doc):
    st = M.status(doc)
    errs = M.mate_errors(doc)
    print(f"  {os.path.basename(SUB)}: {NEW} {st.get(NEW)}, {CASTER} {st.get(CASTER)}, "
          f"{OLD} {'suppressed' if M.comps(doc, top=True)[OLD].IsSuppressed() else 'ACTIVE'}, "
          f"mate errors {len(errs)}" + "".join(f"\n    {e}" for e in errs))
    return st, errs


def check(sw):
    doc, _ = sub_doc(sw)
    doc.ForceRebuild3(False)
    st, errs = report(doc)
    bad = list(errs)
    for path in (os.path.join(swlib.V5, "Box", "Box.SLDASM"), swlib.ASM):
        d = F.find_doc(sw, path)
        if d is not None:
            d.ForceRebuild3(False)
            e = M.mate_errors(d)
            bad += e
            print(f"  {os.path.basename(path)}: mate errors {len(e)}" + "".join(f"\n    {x}" for x in e))
    # interference of the strut with everything in the box, as placed
    interferences = import_module("05_interference").interferences
    box = F.find_doc(sw, os.path.join(swlib.V5, "Box", "Box.SLDASM"))
    sw.OpenDoc6(box.GetPathName(), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)
    sw.ActivateDoc3(box.GetTitle(), False, 0, 0)
    hits, _ = interferences(box, subassemblies=False)
    mine = [(v, n) for v, n, *_ in hits if any("TailStrut" in x for x in n)]
    print(f"  interference in Box involving the strut: {len(mine)}" +
          "".join(f"\n    {v:9.3f} mm3  {'  x  '.join(n)}" for v, n in mine))
    return not bad and not mine and st.get(NEW) == "fully"


def save(sw):
    doc, _ = sub_doc(sw)
    doc.ForceRebuild3(False)
    st, errs = report(doc)
    if errs or st.get(NEW) != "fully" or not M.comps(doc, top=True)[OLD].IsSuppressed():
        raise SystemExit("REFUSED: mate errors, strut not fully defined, or the v3 support still active")
    # the strut AND every assembly above it (BottomPanelWithAvionics, Box, ROBOT)
    import_module("16_check_and_save").chain(sw, [PRT])


def render(sw):
    out = os.path.join(OUT, "renders")
    box = F.find_doc(sw, os.path.join(swlib.V5, "Box", "Box.SLDASM"))
    sw.OpenDoc6(box.GetPathName(), c.swDocASSEMBLY, c.swOpenDocOptions_Silent, "", 0, 0)
    sw.ActivateDoc3(box.GetTitle(), False, 0, 0)
    for view, tag, rot in (("*Left", "rear", None), ("*Front", "side", None), ("*Bottom", "below", None),
                           ("*Left", "rear_low", (-0.35, 0.6))):
        box.Extension.SetUserPreferenceToggle(c.swViewDisplayHideAllTypes, 0, True)
        box.ShowNamedView2(view, -1)
        if rot:
            box.ActiveView.RotateAboutCenter(*rot)
        box.ViewZoomtofit2()
        p = os.path.join(out, f"tail_strut_box_{tag}.png")
        ok, err, warn = box.Extension.SaveAs3(p, 0, c.swSaveAsOptions_Silent | c.swSaveAsOptions_Copy, None, None, 0, 0)
        print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(p, HERE)}")
    robot = swlib.open_v5(sw)
    import_module("12_render").shot(sw, robot, "*Front", os.path.join(out, "tail_strut_robot_side.png"))


if __name__ == "__main__":
    sw, _ = swlib.connect()
    {"import": do_import, "install": install, "check": check, "save": save, "render": render}[sys.argv[1]](sw)
