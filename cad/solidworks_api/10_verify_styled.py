"""Step 10: does the styled robot still assemble, move and touch where it should?

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/10_verify_styled.py [step_deg]

Every styling feature is named GL_*, so SUPPRESSING them puts each part back to
its original geometry inside the very same assembly.  That gives a clean A/B,
same mates, same session, same poses:

  1. MATES     every mate solves with the styling on
  2. CONTACT   which pairs TOUCH (coincident faces) at the Middle pose, off vs
               on.  A pair that stops touching means a removal ate a surface
               something rests on -- the "cutout that breaks the mechanical
               assembly" case.  Interference cannot see it: a gap is not an
               overlap.
  3. INSIDE    full-detail interference at the Middle pose, off vs on: catches
               styling that runs into a part of its OWN sub-assembly (a femur
               flange into its own motor rotor), which never changes with pose.
  4. SWEEP     sub-assembly-block interference at every pose -28..+57, off vs
               on.  A pair that is new, or grows by more than 1 mm3, is the
               styling's fault.

Writes out/verify_<step>deg.txt.  Leaves the styling ON.  Toggling rebuilds the
bodies, resets their names and can drop colours, so the five styled parts are
RELOADED from disk at the end.  Nothing is saved.
"""
import os
import sys
import time
from importlib import import_module

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
import swstyle as S
from swlib import c, wrap, sld

interferences = import_module("05_interference").interferences
STYLED = [r"Links\Femur.SLDPRT", r"Links\Coupler.SLDPRT", r"Links\Tibia.SLDPRT",
          r"Body\Side panel.SLDPRT", r"Body\OldRobotBodyMount\RobotMount.SLDPRT"]
LO, HI = -28.0, 57.0
TOL = 1.0      # mm3


def _doc(sw, rel):
    path = os.path.join(swlib.V5, rel)
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if os.path.normcase(d.GetPathName()) == os.path.normcase(path):
            return d
    return None


def set_styling(sw, model, on):
    """Suppress / unsuppress every GL_* feature in every styled part."""
    n = 0
    for rel in STYLED:
        d = _doc(sw, rel)
        if d is None:
            continue
        f = wrap(d.FirstFeature(), sld.IFeature)
        while f is not None:
            if f.Name.startswith("GL_"):
                f.SetSuppression2(c.swUnSuppressFeature if on else c.swSuppressFeature,
                                  c.swThisConfiguration, None)
                n += 1
            f = wrap(f.GetNextFeature(), sld.IFeature)
        d.EditRebuild3()
    model.ForceRebuild3(False)
    return n


def mate_errors(model):
    bad = []
    f = wrap(model.FirstFeature(), sld.IFeature)
    while f is not None:
        if f.GetTypeName2() == "MateGroup":
            s = wrap(f.GetFirstSubFeature(), sld.IFeature)
            while s is not None:
                code, warn = s.GetErrorCode2()
                if code != 0 and not s.IsSuppressed():
                    bad.append((s.Name, code, warn))
                s = wrap(s.GetNextSubFeature(), sld.IFeature)
        f = wrap(f.GetNextFeature(), sld.IFeature)
    return bad


def touching(model):
    """Pairs that touch or overlap, with coincident faces counted."""
    asm = wrap(model, sld.IAssemblyDoc)
    mgr = wrap(asm.InterferenceDetectionManager, sld.IInterferenceDetectionMgr)
    mgr.TreatCoincidenceAsInterference = True
    mgr.TreatSubAssembliesAsComponents = False
    mgr.IncludeMultibodyPartInterferences = False
    mgr.IgnoreHiddenBodies = True
    out = set()
    for x in mgr.GetInterferences() or []:
        it = wrap(x, sld.IInterference)
        names = tuple(sorted(wrap(cp, sld.IComponent2).Name2 for cp in (it.Components or [])))
        if not any(n.split("/")[0] in swlib.NOT_FOR_COLLISION for n in names):
            out.add(names)
    mgr.Done()
    return out


def volumes(model, blocks):
    hits, _ = interferences(model, subassemblies=blocks)
    v = {}
    for vol, names, *_ in hits:
        k = tuple(sorted(names))
        v[k] = v.get(k, 0.0) + vol
    return v


def main(step):
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    hip = swlib.HipDriver(model)
    hip0 = hip.hip()
    styled = [r for r in STYLED if _doc(sw, r) is not None]
    say(f"styled parts loaded: {len(styled)}/{len(STYLED)}")

    # NEVER measure "styling ON" after un-suppressing: the colour features
    # hold references to specific bodies, and after an off/on cycle the Coupler
    # came back with 42 bodies and 571552 mm3 (copies and slab tools left
    # standing) where the saved part has 36 and 77094 -- every "collision" it
    # then reported was that debris.  ON is always measured from the files on
    # disk; OFF by suppressing; then reload.
    def pristine():
        for rel in STYLED:
            if _doc(sw, rel) is not None:
                swlib.reload_from_disk(sw, os.path.join(swlib.V5, rel))
        sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
        model.ForceRebuild3(False)

    pristine()
    # 1. mates -- the top level AND every open v5 sub-assembly
    bad = []
    for d in sw.GetDocuments() or []:
        d = wrap(d, sld.IModelDoc2)
        if d.GetType() == c.swDocASSEMBLY and os.path.normcase(d.GetPathName()).startswith(
                os.path.normcase(swlib.V5)):
            d.ForceRebuild3(False)
            bad += [(d.GetTitle(),) + b for b in mate_errors(d)]
    say(f"\n1. MATES with styling ON: {len(bad)} in error" + "".join(f"\n   {b}" for b in bad))

    # 2 + 3 at the Middle pose: ON (as saved) first, then OFF
    hip.set(19.98)
    res = {}
    for on in (True, False):
        if not on:
            set_styling(sw, model, False)
            hip.set(19.98)
        t = time.time()
        res[on] = (touching(model), volumes(model, blocks=False))
        say(f"   styling {'ON ' if on else 'OFF'}: {len(res[on][0])} touching pairs, "
            f"{len(res[on][1])} interfering ({time.time() - t:.0f} s)")
    lost = res[False][0] - res[True][0]
    say(f"\n2. CONTACT lost by the styling: {len(lost)}" + "".join(f"\n   {' x '.join(p)}" for p in sorted(lost)))
    new = {k: (res[False][1].get(k, 0.0), v) for k, v in res[True][1].items()
           if v - res[False][1].get(k, 0.0) > TOL}
    say(f"\n3. INSIDE (full detail, Middle pose) new or grown: {len(new)}" +
        "".join(f"\n   {a:9.2f} -> {b:9.2f} mm3  {' x '.join(k)}" for k, (a, b) in sorted(new.items(), key=lambda t: -t[1][1])))

    # 4. sweep: OFF (still suppressed), then reload -> ON as saved
    n = int(round((HI - LO) / step)) + 1
    angles = [LO + i * (HI - LO) / (n - 1) for i in range(n)]
    sweep = {}
    for on in (False, True):
        if on:
            pristine()
        t = time.time()
        sweep[on] = []
        for a in angles:
            hip.set(a)
            sweep[on].append(volumes(model, blocks=True))
        say(f"   sweep styling {'ON ' if on else 'OFF'}: {n} poses in {time.time() - t:.0f} s")
    worst = {}
    for i, a in enumerate(angles):
        for k, v in sweep[True][i].items():
            d = v - sweep[False][i].get(k, 0.0)
            if d > TOL and (k not in worst or d > worst[k][1]):
                worst[k] = (a, d, sweep[False][i].get(k, 0.0), v)
    say(f"\n4. SWEEP {n} poses, new or grown by the styling: {len(worst)}" +
        "".join(f"\n   +{d:8.2f} mm3 at hip {a:+6.1f} ({b:.2f} -> {v:.2f})  {' x '.join(k)}"
                for k, (a, d, b, v) in sorted(worst.items(), key=lambda t: -t[1][1])))

    hip.set(hip0)
    pristine()
    out = os.path.join(HERE, "out", f"verify_{step:g}deg.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(log) + "\n")
    say(f"\nwrote {out}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
