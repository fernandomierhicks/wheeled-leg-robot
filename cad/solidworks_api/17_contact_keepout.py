"""Step 17: the contacts the robot is DESIGNED to make, as a keep-out for the styling.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/17_contact_keepout.py

Some pairs touch on purpose over part of the stroke.  In the baseline sweep of
the ORIGINAL parts (out/sweep_off.json) two do, both on the inner femur plate:
the -Y edge pressing the retract limit switch over the last 2 deg, and the
coupler landing on the knee tube -- the retracted hard stop.  A removal over
either deletes the contact without a trace: interference cannot report a
contact that is no longer there, and 14's compare only looks for pairs that
GREW.

A pair whose volume changes with the hip is a contact, not a fit (a bearing in
its seat or a screw in its hole is the same at every pose).  For each such pair
touching a styled part, at every pose where it touches, this reads the
interference body with the styling OFF -- the original contact, not what the
styling left of it -- moves it into the styled part's frame, and writes its
plan outline + CLEAR to out/contact_keepout.json.  08 keeps every removal AND
addition off it.  Reloads the styled parts from disk at the end.
"""
import os
import sys
import json
from importlib import import_module
from shapely.geometry import MultiPoint, mapping
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
from swlib import wrap, sld

V = import_module("10_verify_styled")
K = import_module("13_collision_keepout")
W = import_module("14_sweep_compare")
OUT = os.path.join(HERE, "out", "contact_keepout.json")
CLEAR = 2.0      # mm round the contact, in plan
VARY = 0.01      # mm3; a pair that changes by more than this over the stroke is a contact


def contacts():
    """{pair: [hip, ...]}: every pair of the OFF baseline that touches a styled
    part and changes with the hip, with the poses where it is non-zero."""
    base = json.load(open(W.path("off")))
    out = {}
    for k in sorted({k for pose in base for k in pose["pairs"]}):
        if not any(n in K.STYLED_COMPONENTS for n in k.split(" x ")):
            continue
        v = [pose["pairs"].get(k, 0.0) for pose in base]
        if max(v) - min(v) > VARY:
            out[k] = [pose["hip"] for pose in base if pose["pairs"].get(k, 0.0) > 0]
    return out


def main():
    pairs = contacts()
    for k, hips in pairs.items():
        print(f"  contact at {len(hips)} pose(s) {hips[0]:+.1f}..{hips[-1]:+.1f}: {k}")
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    comps = swlib.components(model)
    asm = wrap(model, sld.IAssemblyDoc)
    W.reload_all(sw, model)
    V.set_styling(sw, model, False)
    hip = swlib.HipDriver(model)
    hip0 = hip.hip()
    keep = {}
    try:
        for a in sorted({h for hs in pairs.values() for h in hs}):
            got = hip.set(a)
            want = {k for k, hs in pairs.items() if a in hs}
            mgr = wrap(asm.InterferenceDetectionManager, sld.IInterferenceDetectionMgr)
            mgr.TreatCoincidenceAsInterference = False
            mgr.TreatSubAssembliesAsComponents = True       # as the baseline sweep
            mgr.IncludeMultibodyPartInterferences = False
            mgr.IgnoreHiddenBodies = True
            for x in mgr.GetInterferences() or []:
                it = wrap(x, sld.IInterference)
                cps = [wrap(cp, sld.IComponent2) for cp in (it.Components or [])]
                if " x ".join(sorted(cp.Name2 for cp in cps)) not in want:
                    continue
                pts = K._points(it)
                if pts is None:
                    continue
                for cp in cps:
                    part = K.STYLED_COMPONENTS.get(cp.Name2)
                    if part is None:
                        continue
                    M = swlib.placement(comps[cp.Name2])
                    local = (M[:3, :3].T @ (pts - M[:3, 3]).T).T
                    hull = MultiPoint([tuple(p[:2]) for p in local]).convex_hull
                    keep.setdefault(part, []).append(hull.buffer(CLEAR))
                    x0, y0, x1, y1 = hull.bounds
                    print(f"  hip {got:+6.1f} {it.Volume * 1e9:7.3f} mm3 on {part}: x {x0:.1f}..{x1:.1f} "
                          f"y {y0:.1f}..{y1:.1f} z {local[:, 2].min():.1f}..{local[:, 2].max():.1f}")
            mgr.Done()
    finally:
        hip.set(hip0)
        W.reload_all(sw, model)          # styling back ON, from disk (gotcha 37)
    out = {}
    for part, gs in keep.items():
        u = unary_union(gs)
        out[part] = [mapping(g) for g in (u.geoms if hasattr(u, "geoms") else [u])]
        print(f"  {part}: contact keep-out {u.area:.1f} mm2")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
