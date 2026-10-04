"""Step 16: check the styled state, then save every v5 document.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/16_check_and_save.py [--dry]

Use this instead of Save All in SolidWorks after any automation run (README
gotcha 37: a Save All once wrote four links to disk UNSTYLED).  In order:

  1. every styled part is loaded and STYLED: refuses if any part has all of
     its GL_ features suppressed (the OFF baseline 10/13/14 use); a few
     suppressed by hand are reported, not refused
  2. the AI_HipDrive helper mate is deleted (saved, it locks the leg)
  3. mates: 0 in error in every open v5 assembly, or nothing is saved
  4. every dirty v5 document is saved: parts, then sub-assemblies, then
     ROBOT.SLDASM.  Nothing outside v5 is touched.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importlib import import_module
import swlib
import swstyle as S
from swlib import c, wrap, sld

mate_errors = import_module("10_verify_styled").mate_errors
STYLED = import_module("10_verify_styled").STYLED + [
    r"Links\BearingWasher.SLDPRT", r"Body\SmallBearingWahser.SLDPRT", r"Body\InsideFemurShaft.SLDPRT"]


def v5_docs(sw):
    out = []
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()).startswith(os.path.normcase(swlib.V5)):
            out.append(d)
    return out


def main(dry):
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    docs = {os.path.normcase(d.GetPathName()): d for d in v5_docs(sw)}

    # 1. styled?
    bad = []
    for rel in STYLED:
        d = docs.get(os.path.normcase(os.path.join(swlib.V5, rel)))
        if d is None:
            bad.append(f"{rel}: not loaded")
            continue
        gl = [f for f in S._iter_features(d) if f.Name.startswith("GL_")]
        sup = [f.Name for f in gl if f.IsSuppressed2(c.swThisConfiguration, None)[0]]
        n = len(S.bodies(d))
        print(f"  {rel:42s} GL {len(gl):3d}, suppressed {len(sup):3d}, bodies {n:3d}"
              + (f"   suppressed: {', '.join(sup)}" if 0 < len(sup) <= 6 else ""))
        if not gl or len(sup) == len(gl) or n < 2:
            bad.append(f"{rel}: NOT STYLED in memory ({len(sup)}/{len(gl)} GL_ suppressed, {n} bodies)")
    if bad:
        raise SystemExit("REFUSING TO SAVE:\n  " + "\n  ".join(bad))

    # 2. helper mate
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    if swlib.HipDriver.remove(model):
        print("  deleted the AI_HipDrive helper mate")

    # 3. mates
    errs = []
    for d in v5_docs(sw):
        if d.GetType() == c.swDocASSEMBLY:
            d.ForceRebuild3(False)
            errs += [(d.GetTitle(),) + e for e in mate_errors(d)]
    print(f"  mates in error: {len(errs)}" + "".join(f"\n    {e}" for e in errs))
    if errs:
        raise SystemExit("REFUSING TO SAVE: mate errors")

    # 4. save
    asm = os.path.normcase(swlib.ASM)
    order = sorted(v5_docs(sw), key=lambda d: (os.path.normcase(d.GetPathName()) == asm,
                                               d.GetType() == c.swDocASSEMBLY))
    dirty = [d for d in order if d.GetSaveFlag()]
    print(f"  {len(dirty)} of {len(order)} v5 documents have unsaved changes")
    for d in dirty:
        rel = os.path.relpath(d.GetPathName(), swlib.V5)
        if dry:
            print(f"    would save {rel}")
            continue
        ok, err, warn = d.Save3(c.swSaveAsOptions_Silent, 0, 0)
        print(f"    {'saved ' if ok else 'FAILED'} {rel}" + ("" if ok else f" (err {err}, warn {warn})"))
    left = [os.path.relpath(d.GetPathName(), swlib.V5) for d in v5_docs(sw) if d.GetSaveFlag()]
    print(f"  still unsaved: {len(left)}" + "".join(f"\n    {r}" for r in left))


if __name__ == "__main__":
    main("--dry" in sys.argv[1:])
