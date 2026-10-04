"""Step 13: turn every collision the styling causes into a keep-out for the styling.

    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/13_collision_keepout.py sweep  [step_deg]
    C:/Users/ferna/cadenv/Scripts/python.exe cad/solidworks_api/13_collision_keepout.py bodies [max_poses]

Removals can never cause a collision, so every new interference is ADDED
material (flange or raised frame) running into a neighbour.  This finds each
one and records where it is, in the styled part's own frame.

  sweep   styling OFF vs ON, sub-assembly blocks, every pose -28..+57, plus full
          detail at the Middle pose (pairs inside a sub-assembly).  Saves the
          (pose, pair) list the styling made worse to out/collision_todo.json.
  bodies  for each saved pose not yet done: Interference Detection,
          IInterference.GetInterferenceBody() -> box corners + vertices, moved
          into the styled part's local frame (inverse of its placement) -> plan
          outline + CLEAR, merged per part into out/collision_keepout.json.
          Resumable: does at most max_poses per call (default 30) and records
          what it has done, so no single run is long enough to be killed --
          the first one-shot version was terminated (exit 143) after ~35 min.

08/09 subtract out/collision_keepout.json from each part's ADDITIONS on the next
--restyle.  Re-run 10 to prove the sweep is clean.  Both phases reload the five
styled parts from disk at the end (toggling resets names and drops colours).
"""
import os
import sys
import json
import time
from importlib import import_module
import numpy as np
from shapely.geometry import MultiPoint, shape, mapping
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swlib
from swlib import c, wrap, sld

V = import_module("10_verify_styled")
TODO = os.path.join(HERE, "out", "collision_todo.json")
OUT = os.path.join(HERE, "out", "collision_keepout.json")
CLEAR = 1.0          # mm around the overlap, in plan
TOL = 1.0            # mm3
MIDDLE = 19.98
STYLED_COMPONENTS = {"Femur-1/Femur-1": "Femur", "COUPLER-1/Coupler-1": "Coupler",
                     "Tibia-1/Tibia-1": "Tibia", "BODY-1/SIDE PANEL-1/Side panel-1": "Side panel",
                     "BODY-1/SIDE PANEL-1/RobotMount-1": "RobotMount"}


def _reload_all(sw, model):
    for rel in V.STYLED:
        if V._doc(sw, rel) is not None:
            swlib.reload_from_disk(sw, os.path.join(swlib.V5, rel))
    sw.ActivateDoc3(model.GetTitle(), False, 0, 0)
    model.ForceRebuild3(False)


def sweep(step):
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    hip = swlib.HipDriver(model)
    n = int(round((V.HI - V.LO) / step)) + 1
    angles = [V.LO + i * (V.HI - V.LO) / (n - 1) for i in range(n)]
    base, inside = {}, {}
    # ON from the files on disk FIRST, then OFF by suppressing -- never ON after
    # un-suppressing: the colour features do not survive that cycle cleanly and
    # leave copies and slab tools standing (README gotcha 32)
    _reload_all(sw, model)
    for on in (True, False):
        if not on:
            V.set_styling(sw, model, False)
        hip.set(MIDDLE)
        inside[on] = V.volumes(model, blocks=False)
        base[on] = []
        t = time.time()
        for a in angles:
            hip.set(a)
            base[on].append(V.volumes(model, blocks=True))
        print(f"  styling {'ON ' if on else 'OFF'}: {n} poses in {time.time() - t:.0f} s", flush=True)
    todo = []
    for i, a in enumerate(angles):
        pairs = [list(k) for k, v in base[True][i].items() if v - base[False][i].get(k, 0.0) > TOL]
        if pairs:
            todo.append({"hip": a, "blocks": True, "pairs": pairs, "done": False})
    static = [list(k) for k, v in inside[True].items() if v - inside[False].get(k, 0.0) > TOL]
    if static:
        todo.append({"hip": MIDDLE, "blocks": False, "pairs": static, "done": False})
    json.dump(todo, open(TODO, "w"), indent=1)
    print(f"  {len(todo)} pose(s) to read: {sum(len(t['pairs']) for t in todo)} (pose, pair) collisions -> {TODO}")
    _reload_all(sw, model)


def _points(it):
    b = wrap(it.GetInterferenceBody(), sld.IBody2)
    if b is None:
        return None
    x0, y0, z0, x1, y1, z1 = [v * 1000 for v in b.GetBodyBox()]
    pts = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    try:
        for v in b.GetVertices() or []:
            pts.append(tuple(q * 1000 for q in wrap(v, sld.IVertex).GetPoint()))
    except Exception:
        pass
    return np.array(pts)


def bodies(max_poses):
    todo = json.load(open(TODO))
    left = [t for t in todo if not t["done"]]
    if not left:
        print("  nothing left to read")
        return
    sw, _ = swlib.connect()
    model = swlib.open_v5(sw)
    comps = swlib.components(model)
    hip = swlib.HipDriver(model)
    hip0 = hip.hip()
    keep = json.load(open(OUT)) if os.path.exists(OUT) else {}
    asm = wrap(model, sld.IAssemblyDoc)
    t0 = time.time()
    for t in left[:max_poses]:
        hip.set(t["hip"])
        want = {tuple(sorted(p)) for p in t["pairs"]}
        mgr = wrap(asm.InterferenceDetectionManager, sld.IInterferenceDetectionMgr)
        mgr.TreatCoincidenceAsInterference = False
        mgr.TreatSubAssembliesAsComponents = bool(t["blocks"])
        mgr.IncludeMultibodyPartInterferences = False
        mgr.IgnoreHiddenBodies = True
        got = 0
        for x in mgr.GetInterferences() or []:
            it = wrap(x, sld.IInterference)
            cps = [wrap(cp, sld.IComponent2) for cp in (it.Components or [])]
            if tuple(sorted(cp.Name2 for cp in cps)) not in want or it.Volume * 1e9 < TOL:
                continue
            pts = _points(it)
            if pts is None:
                continue
            for cp in cps:
                part = STYLED_COMPONENTS.get(cp.Name2)
                if part is None:
                    continue
                M = swlib.placement(comps[cp.Name2])
                local = (M[:3, :3].T @ (pts - M[:3, 3]).T).T
                hull = MultiPoint([tuple(p[:2]) for p in local]).convex_hull.buffer(CLEAR)
                prev = [shape(g) for g in keep.get(part, [])]
                u = unary_union(prev + [hull])
                keep[part] = [mapping(g) for g in (u.geoms if hasattr(u, "geoms") else [u])]
                got += 1
        mgr.Done()
        t["done"] = True
        json.dump(keep, open(OUT, "w"), indent=1)          # progress survives a kill
        json.dump(todo, open(TODO, "w"), indent=1)
        print(f"  hip {t['hip']:+6.1f}: {got} footprint(s) from {len(want)} pair(s)", flush=True)
    rest = sum(1 for t in todo if not t["done"])
    for part, gs in keep.items():
        print(f"  {part:11s} keep-out {unary_union([shape(g) for g in gs]).area:8.1f} mm2")
    print(f"  {time.time() - t0:.0f} s; {rest} pose(s) still to read")
    hip.set(hip0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "sweep":
        sweep(arg or 1.0)
    else:
        bodies(int(arg or 30))
