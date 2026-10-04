"""Step 5 of the SolidWorks automation ladder: one Interference Detection, timed.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/05_interference.py [hip_deg] [--subassemblies]

Puts the hip at `hip_deg` (default +19.98, the Middle export), runs SolidWorks'
Interference Detection over the whole assembly, and lists every interference
with its volume and the two components.  Same options as the Evaluate ->
Interference Detection dialog with: coincidence NOT treated as interference,
parts not sub-assemblies, hidden bodies ignored, fasteners in their own folder.

AK_SIM-1 and WheelHanger-1 are hidden and do not count (his call); anything
touching them is dropped even if SolidWorks reports it.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
from swlib import wrap, sld


def interferences(model, subassemblies=False):
    """[(volume_mm3, [component names], is_fastener, is_possible)], seconds.

    subassemblies=True treats each sub-assembly as one block: interferences
    INSIDE a sub-assembly (screws in their own part, etc.) are not reported.
    """
    asm = wrap(model, sld.IAssemblyDoc)
    mgr = wrap(asm.InterferenceDetectionManager, sld.IInterferenceDetectionMgr)
    mgr.TreatCoincidenceAsInterference = False
    mgr.TreatSubAssembliesAsComponents = bool(subassemblies)
    mgr.IncludeMultibodyPartInterferences = False
    mgr.IgnoreHiddenBodies = True
    mgr.MakeInterferingPartsTransparent = False
    mgr.CreateFastenersFolder = True
    t = time.time()
    raw = mgr.GetInterferences() or []
    dt = time.time() - t
    out = []
    for x in raw:
        it = wrap(x, sld.IInterference)
        names = [wrap(cp, sld.IComponent2).Name2 for cp in (it.Components or [])]
        if any(n.split("/")[0] in swlib.NOT_FOR_COLLISION for n in names):
            continue
        out.append((it.Volume * 1e9, names, bool(it.IsFastener),
                    bool(it.IsPossibleInterference)))
    mgr.Done()
    return out, dt


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want = float(args[0]) if args else 19.98
    blocks = "--subassemblies" in sys.argv
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    hip = swlib.HipDriver(model)
    hip0 = hip.hip()
    got = hip.set(want)
    print(f"hip {want:+.2f} (got {got:+.2f})")
    hits, dt = interferences(model, blocks)
    print(f"Interference Detection: {dt:.1f} s, {len(hits)} interference(s)\n")
    for vol, names, fast, poss in sorted(hits, key=lambda h: -h[0]):
        tag = " [fastener]" if fast else ""
        tag += " [possible]" if poss else ""
        print(f"  {vol:10.3f} mm3  {'  x  '.join(names)}{tag}")
    hip.set(hip0)
