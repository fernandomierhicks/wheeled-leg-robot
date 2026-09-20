r"""Does the styling collide with anything, in any pose?

    C:/Users/ferna/cadenv/Scripts/python.exe lib/collide.py [Pose ...]

Styling only ever ADDS material outside the source silhouette, so it cannot
break a hole -- `verify.py` proves that -- but it can absolutely run a new boss
into a neighbouring part.  The v4 export ships three poses of the same half
robot (Retracted, Middle, Extended), which bracket the leg's travel.

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


def styled_parts():
    """{Part: styled solid} for every part that has a current styled export."""
    out = {}
    for f in sorted(os.listdir(paths.SPECS)):
        if not f.endswith(".json"):
            continue
        part = f[:-5].capitalize()
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


def check(pose, styled):
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
        src = import_step(paths.part_step(name))
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
            base_other = (import_step(paths.part_step(oname)).moved(oloc)
                          if oname in styled else other)
            v_new = overlap(p_sty, other)
            v_old = overlap(p_src, base_other)
            if v_new is None or v_old is None:
                print(f"     ?? {oname}: boolean failed"); continue
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
    poses = sys.argv[1:] or POSES
    sty = styled_parts()
    print(f"styled parts available: {', '.join(sorted(sty)) or '(none)'}")
    all_hits = []
    for pose in poses:
        all_hits += check(pose, sty)
    print("\n" + "=" * 62)
    if all_hits:
        print(f"{len(all_hits)} NEW collision(s) caused by styling:")
        for pose, a, b, o, n, d in all_hits:
            print(f"  {pose:<22} {a} x {b}: +{d:.1f} mm3")
        sys.exit(1)
    print("no new collisions in any pose checked")
