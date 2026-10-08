"""Step 16: check the styled state, then save every v5 document.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/16_check_and_save.py [--dry]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/16_check_and_save.py --chain <v5-relative part> ... [--dry]

--chain (his rule, 2026-10-06: an edited part never leaves its assemblies
fragile): saves the named parts AND every open v5 assembly that contains them
at any depth, sub-assemblies first, ROBOT.SLDASM last -- after the same
guards (1-3 below, mates checked in exactly those assemblies).  Nothing else
is saved, so the ~30 parts that come up dirty on every load stay out of git.
Every script that edits a part ends with chain() (22, 23).

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


def styled_ok(sw, docs):
    """[] if every styled part is loaded and styled in memory (gotcha 37)."""
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
    return bad


def _save(d):
    """Save3 one document, and prove it: the document must come back clean
    (gotcha 37: a call on a doc without a window can return fine and do nothing)."""
    rel = os.path.relpath(d.GetPathName(), swlib.V5)
    ok, err, warn = d.Save3(c.swSaveAsOptions_Silent, 0, 0)
    ok = bool(ok) and not d.GetSaveFlag()
    print(f"    {'saved ' if ok else 'FAILED'} {rel}" + ("" if ok else f" (err {err}, warn {warn})"))
    return ok


def chain(sw, parts, dry=False):
    """Save `parts` (v5-relative or absolute paths) and every open v5 assembly
    containing any of them, bottom-up, ROBOT last.  Refuses (saves nothing) on
    an unstyled styled part, or a mate error in any of those assemblies."""
    model = swlib.open_v5(sw)
    docs = {os.path.normcase(d.GetPathName()): d for d in v5_docs(sw)}
    want = [os.path.normcase(p if os.path.isabs(p) else os.path.join(swlib.V5, p)) for p in parts]
    missing = [w for w in want if w not in docs]
    if missing:
        raise SystemExit(f"REFUSING TO SAVE: not loaded {missing}")
    # every open v5 assembly holding one of the parts at any depth, with what it holds
    holds = {}
    for k, d in docs.items():
        if d.GetType() == c.swDocASSEMBLY:
            paths = {os.path.normcase(wrap(x, sld.IComponent2).GetPathName() or "")
                     for x in wrap(d, sld.IAssemblyDoc).GetComponents(False) or []}
            if paths & set(want):
                holds[k] = paths
    # sub-assemblies first: an assembly goes after every assembly it contains
    order = []
    left = dict(holds)
    while left:
        ready = [k for k, ps in left.items() if not (ps & (set(left) - {k}))]
        if not ready:
            raise SystemExit(f"assembly cycle? {list(left)}")
        for k in sorted(ready, key=lambda k: k == os.path.normcase(swlib.ASM)):
            order.append(k)
            del left[k]
    bad = styled_ok(sw, docs)
    if os.path.normcase(swlib.ASM) in holds:
        sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
        if swlib.HipDriver.remove(model):
            print("  deleted the AI_HipDrive helper mate")
    errs = []
    for k in order:
        docs[k].ForceRebuild3(False)
        errs += [(docs[k].GetTitle(),) + e for e in mate_errors(docs[k])]
    if bad or errs:
        raise SystemExit("REFUSING TO SAVE:" + "".join("\n  " + str(x) for x in bad + errs))
    print(f"  chain: {len(want)} part(s) + {len(order)} assemblies, 0 mate errors")
    for k in want + order:
        if dry:
            print(f"    would save {os.path.relpath(docs[k].GetPathName(), swlib.V5)}")
        elif not _save(docs[k]):
            raise SystemExit("save failed -- stopping (later assemblies not saved)")


def main(dry):
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    docs = {os.path.normcase(d.GetPathName()): d for d in v5_docs(sw)}

    # 1. styled?
    bad = styled_ok(sw, docs)
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
    args = sys.argv[1:]
    if "--chain" in args:
        sw, _ = swlib.connect()
        chain(sw, [a for a in args if a not in ("--chain", "--dry")], "--dry" in args)
    else:
        main("--dry" in args)
