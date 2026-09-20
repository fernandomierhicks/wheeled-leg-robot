r"""Does the styling collide with anything, ANYWHERE IN THE STROKE?

    C:/Users/ferna/cadenv/Scripts/python.exe lib/collide.py --sweep 21
    C:/Users/ferna/cadenv/Scripts/python.exe lib/collide.py --sweep 21 --parts Femur Tibia Coupler
    C:/Users/ferna/cadenv/Scripts/python.exe lib/collide.py            # the old 3 poses

Styling only ever ADDS material outside the source silhouette, so it cannot
break a hole -- `verify.py` proves that -- but it can absolutely run a new boss
into a neighbouring part.

THE SWEEP.  The v4 export ships three poses (Retracted, Middle, Extended), and
for a long time this file checked only those.  Decision 17 argued they bracket
the travel because the worst interference landed at the Retracted hard stop --
but that shows one PAIR peaks at an endpoint and says nothing about another pair
peaking in between.  Fernando's constraint is "collision free through the ENTIRE
stroke", so `--sweep N` places every instance at N hip angles across the whole
85 deg of travel using the 4-bar solved in tools/kinematics.py, and REFUSES to
run if that model does not reconstruct the three exported poses -- an
unvalidated model reporting "no collisions" is worse than three honest samples.

COST.  Every pair is a boolean at roughly a second, and that is the whole
runtime.  Three things keep it usable: source STEPs are imported once rather
than per pair, the styled boolean is evaluated first and the source boolean is
skipped whenever it cannot change the verdict, and `--parts` narrows the check
to the parts actually being rebuilt.

The number that matters is the DELTA, not the raw overlap.  Real CAD is full of
nominally-touching faces -- a bearing in its seat, a bolt in its counterbore --
and a boolean across two coincident faces returns a small non-zero volume.  So
every pair is measured TWICE, once with the source part and once with the styled
part, and only the increase is attributed to the styling.

Frames: a styled part is built in its source part's own local frame and is
mirrored back before export, so it drops into the assembly at exactly the
component transform the source used.  That is asserted rather than assumed --
if an export ever lands in a different frame, this stops instead of quietly
reporting no collisions.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from build123d import Compound, Location, import_step
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TCollection import TCollection_ExtendedString
from OCP.IFSelect import IFSelect_RetDone

POSES = ["Retracted", "Middle single part", "Extended"]
TOL = 1.0          # mm3; below this a "collision" is boolean noise on a shared face
PAD = 0.5          # mm; bbox prefilter slack

_SRC = {}          # part -> source solid, imported ONCE


def source(name):
    """The source STEP for a part, cached.

    This used to be `import_step(paths.part_step(oname))` INSIDE the inner pair
    loop -- re-reading the same file from disk for every candidate pair in every
    pose.  On three poses that is hundreds of redundant STEP parses; on a
    21-angle sweep it would have dominated the run completely.
    """
    if name not in _SRC:
        _SRC[name] = import_step(paths.part_step(name))
    return _SRC[name]


def loc_from_matrix(M):
    """build123d Location from a 3x4 affine (what tools/kinematics.py returns)."""
    from OCP.gp import gp_Trsf
    t = gp_Trsf()
    t.SetValues(float(M[0][0]), float(M[0][1]), float(M[0][2]), float(M[0][3]),
                float(M[1][0]), float(M[1][1]), float(M[1][2]), float(M[1][3]),
                float(M[2][0]), float(M[2][1]), float(M[2][2]), float(M[2][3]))
    return Location(t)


def configurations(sweep=0, poses=None):
    """[(label, [(name, local_shape, Location)])] to check against.

    With `sweep=N`, N hip angles across the whole 85 deg travel, placed by the
    validated 4-bar in tools/kinematics.py.  Without it, the three exported
    poses -- which is all this file could ever do before, and is NOT what
    "collision free through the entire stroke" means.
    """
    poses = list(poses or POSES)
    if not sweep:
        for pose in poses:
            step = os.path.join(paths.EXPORTS, pose + ".STEP")
            if not os.path.exists(step):
                print(f"  (no STEP for pose {pose})")
                continue
            yield pose, leaves(step)
        return

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "tools"))
    import numpy as np
    import kinematics as K
    leg = K.build(verbose=False)
    ok, worst = K.validate(leg)
    if not ok:
        raise SystemExit(f"the kinematic model reconstructs the exported poses to "
                         f"only {worst:.3f} mm -- refusing to call anything "
                         f"collision free using it")
    ref = leg.poses[0]
    base = leaves(os.path.join(paths.EXPORTS, ref + ".STEP"))
    seen, keys = {}, []
    for name, shp, loc in base:
        k = seen.get(name, 0); seen[name] = k + 1
        keys.append(f"{name}#{k}")

    def comp(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        return np.hstack([A[:3, :3] @ B[:3, :3],
                          (A[:3, :3] @ B[:3, 3] + A[:3, 3])[:, None]])

    def inv(M):
        M = np.asarray(M, float)
        R, t = M[:3, :3], M[:3, 3]
        Ri = R.T if abs(np.linalg.det(R) - 1) < 1e-9 else np.linalg.inv(R)
        return np.hstack([Ri, (-Ri @ t)[:, None]])

    reprs = {"femur": leg.fk, "coupler": leg.ck, "tibia": leg.tk}
    for q in leg.sweep(sweep):
        got = leg.at(q)
        if got is None:
            raise SystemExit(f"mechanism does not close at q={np.degrees(q):.2f} deg")
        D = {g: comp(np.vstack([got[g], [0, 0, 0, 1]])[:3], inv(leg.P[ref][key]))
             for g, key in reprs.items()}
        out = []
        for (name, shp, loc), key in zip(base, keys):
            g = next((g for g, mem in leg.groups.items() if key in mem), None)
            M = leg.P[ref][key] if g is None else comp(D[g], leg.P[ref][key])
            out.append((name, shp, loc_from_matrix(M)))
        yield f"q={np.degrees(q):+.2f}deg", out


def _name(lab):
    a = TDataStd_Name()
    return a.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), a) else "?"


def leaves(step):
    """[(name, local_shape, global_location)] for every leaf instance."""
    doc = TDocStd_Document(TCollection_ExtendedString("d"))
    rd = STEPCAFControl_Reader(); rd.SetNameMode(True)
    if rd.ReadFile(step) != IFSelect_RetDone:
        raise SystemExit(f"cannot read {step}")
    rd.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    out = []

    def walk(lab, loc):
        if st.IsAssembly_s(lab):
            comps = TDF_LabelSequence(); st.GetComponents_s(lab, comps)
            for j in range(1, comps.Length() + 1):
                rl = TDF_Label()
                if st.GetReferredShape_s(comps.Value(j), rl):
                    walk(rl, loc * Location(st.GetLocation_s(comps.Value(j)).Transformation()))
        else:
            out.append((_name(lab), Compound(st.GetShape_s(lab)), loc))

    roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), Location())
    return out


def styled_parts(only=None):
    """{Part: styled solid} for every part that has a current styled export.

    `only` narrows it to the parts being rebuilt.  Every pair costs two booleans
    at ~1 s each, so checking the two out-of-scope plates -- whose geometry is
    not changing -- is 40% of the run for no information.
    """
    out = {}
    for f in sorted(os.listdir(paths.SPECS)):
        if not f.endswith(".json"):
            continue
        part = f[:-5].capitalize()
        if only and part not in only:
            continue
        tag = json.load(open(os.path.join(paths.SPECS, f))).get("tag", "arctic")
        p = paths.styled_step(part, tag)
        if os.path.exists(p):
            out[part] = import_step(p)
    return out


def overlap(a, b):
    try:
        x = a & b
    except Exception:
        return None
    return 0.0 if x is None else x.volume


def check(pose, styled, inst=None, verbose=True):
    if inst is None:
        step = os.path.join(paths.EXPORTS, pose + ".STEP")
        if not os.path.exists(step):
            print(f"  (no STEP for pose {pose})"); return []
        inst = leaves(step)
    print(f"\n=== {pose} ===  {len(inst)} leaf instances")
    hits = []
    for i, (name, shp, loc) in enumerate(inst):
        if name not in styled:
            continue
        sty = styled[name]
        # frame guard -- see module docstring
        bs, bl = sty.bounding_box(), shp.bounding_box()
        src = source(name)
        bsrc = src.bounding_box()
        for ax in "XYZ":
            if abs(getattr(bsrc.min, ax) - getattr(bl.min, ax)) > 0.01 or \
               abs(getattr(bsrc.max, ax) - getattr(bl.max, ax)) > 0.01:
                raise SystemExit(
                    f"FRAME MISMATCH on {name}: the assembly's copy is not in the same "
                    f"local frame as {paths.part_step(name)}.\n"
                    f"  assembly {ax} {getattr(bl.min,ax):.3f}..{getattr(bl.max,ax):.3f}\n"
                    f"  export   {ax} {getattr(bsrc.min,ax):.3f}..{getattr(bsrc.max,ax):.3f}\n"
                    f"Placing the styled part on the assembly transform would be wrong.")
        p_sty, p_src = sty.moved(loc), src.moved(loc)
        bb_sty = p_sty.bounding_box()
        if verbose:
            print(f"  {name}: styled grew "
                  f"{bs.size.X-bsrc.size.X:+.1f} X, {bs.size.Y-bsrc.size.Y:+.1f} Y, "
                  f"{bs.size.Z-bsrc.size.Z:+.1f} Z")
        for j, (oname, oshp, oloc) in enumerate(inst):
            if i == j:
                continue
            other = (styled[oname].moved(oloc) if oname in styled else oshp.moved(oloc))
            ob = other.bounding_box()
            if (bb_sty.min.X > ob.max.X + PAD or bb_sty.max.X < ob.min.X - PAD or
                bb_sty.min.Y > ob.max.Y + PAD or bb_sty.max.Y < ob.min.Y - PAD or
                bb_sty.min.Z > ob.max.Z + PAD or bb_sty.max.Z < ob.min.Z - PAD):
                continue
            # THE STYLED BOOLEAN FIRST, AND OFTEN THE ONLY ONE.  The number that
            # matters is d = v_new - v_old, and v_old >= 0, so v_new <= TOL means
            # d <= TOL and the pair cannot be a new collision no matter what the
            # source did.  The overwhelming majority of pairs that clear the bbox
            # prefilter return v_new = 0, so computing v_old up front doubled the
            # boolean count for nothing -- and booleans are ~1 s each on these
            # solids, which is the entire cost of this file.
            v_new = overlap(p_sty, other)
            if v_new is None:
                print(f"     ?? {oname}: boolean failed"); continue
            if v_new <= TOL:
                continue
            base_other = (source(oname).moved(oloc) if oname in styled else other)
            v_old = overlap(p_src, base_other)
            if v_old is None:
                print(f"     ?? {oname}: source boolean failed"); continue
            d = v_new - v_old
            if d > TOL:
                hits.append((pose, name, oname, v_old, v_new, d))
                print(f"     COLLIDES with {oname}: {v_old:8.1f} -> {v_new:8.1f} mm3  "
                      f"(+{d:.1f} from styling)")
            elif v_new > TOL:
                print(f"     touches {oname}: {v_new:7.1f} mm3, pre-existing "
                      f"(delta {d:+.1f})")
    return hits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("poses", nargs="*")
    ap.add_argument("--sweep", type=int, default=0,
                    help="check N hip angles across the WHOLE 85 deg travel, "
                         "placed by the validated 4-bar, instead of the three "
                         "exported poses.  This is what 'collision free through "
                         "the entire stroke' actually requires.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="only check these styled parts (e.g. Femur Tibia Coupler)")
    a = ap.parse_args()
    sty = styled_parts(a.parts)
    print(f"styled parts checked: {', '.join(sorted(sty)) or '(none)'}")
    if a.sweep:
        print(f"sweeping {a.sweep} hip angles across the travel")
    all_hits, n_cfg = [], 0
    for label, inst in configurations(a.sweep, a.poses or None):
        n_cfg += 1
        all_hits += check(label, sty, inst, verbose=not a.quiet)
    print("\n" + "=" * 62)
    if all_hits:
        # One line per PAIR at its worst configuration.  21 angles would
        # otherwise print 21 copies of the same interference and bury the
        # question that matters, which is "which pair, and how bad at worst".
        worst = {}
        for pose, x, y, o, n, d in all_hits:
            k = (x, y)
            if k not in worst or d > worst[k][1]:
                worst[k] = (pose, d)
        print(f"{len(all_hits)} collision report(s) over {n_cfg} configuration(s), "
              f"{len(worst)} distinct pair(s):")
        for (x, y), (pose, d) in sorted(worst.items(), key=lambda t: -t[1][1]):
            print(f"  {x} x {y}: worst +{d:.1f} mm3 at {pose}")
        sys.exit(1)
    print(f"no new collisions in any of the {n_cfg} configuration(s) checked")
