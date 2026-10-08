"""Geometry-driven mating that never moves anything (built for the box import, 19).

    import swmate as M
    m = M.Mater(sw, doc)                       # doc = the assembly the mates go in
    chosen, rows, cands = M.plan(m, "CornerBracket-13", ["BackPanel-1"], ["Front Plane"])
    M.apply(m, "CornerBracket-13", chosen, "CB13")

Faces are found by geometry in the ASSEMBLY frame (mm): planes by outward normal
and offset n.x, cylinders by axis, a point and radius.  `plan` picks a minimal
set of mates from real contacts -- touching faces, coaxial holes/shanks, flush
faces -- that removes all 6 degrees of freedom; `Mater.mate` adds each one and
REJECTS it if any component in the assembly moved (rolls it back and tries the
next alignment).  So a re-mate can only ever restate where things already are.

Rules measured on SolidWorks 2023, each one a failed run (README gotchas 43-47):
  * a translation may be fixed only ONCE.  Two parallel concentrics, or a face
    coincidence plus a concentric whose axis lies in that face, are refused at
    AddMate5 (OverDefinedAssembly) even when the geometry is consistent.
    Rotational overlap (face + perpendicular hole) is fine.
  * a refused AddMate5 can still leave a broken mate behind; delete it.
  * EditUndo2 does not undo an API AddMate5.  Roll back by deleting the mate
    and putting moved components back with Transform2 (free DOF only).
  * on a free component a concentric can slide it along a free direction:
    mate a face first.
"""
import math
import numpy as np
import pythoncom
from win32com.client import VARIANT

import swlib
from swlib import c, wrap, sld

TOL = 1e-3   # mm
CS = {1: "unknown", 2: "UNDER", 3: "fully", 4: "OVER", 5: "NOSOL", 6: "INVALID", 7: "autosolve-off"}
DATUMS = {"Front Plane": (np.array([0, 0, 1.0]), 0.0),
          "Top Plane": (np.array([0, 1.0, 0]), 0.0),
          "Right Plane": (np.array([1.0, 0, 0]), 0.0)}


# ---- assembly bookkeeping --------------------------------------------------
def comps(doc, top=False):
    return {wrap(x, sld.IComponent2).Name2: wrap(x, sld.IComponent2)
            for x in wrap(doc, sld.IAssemblyDoc).GetComponents(top) or []}


def placements(doc):
    out = {}
    for n, cp in comps(doc).items():
        if not cp.IsSuppressed():
            try:
                out[n] = swlib.placement(cp).tolist()
            except Exception:
                pass
    return out


def compare(a, b, tol=TOL):
    """(moved, gone, new) between two placements() snapshots."""
    moved = []
    for n in a:
        if n in b:
            A, B = np.array(a[n]), np.array(b[n])
            dt = np.abs(A[:, 3] - B[:, 3]).max()
            dr = np.abs(A[:, :3] - B[:, :3]).max()
            if dt > tol or dr > 1e-5:
                moved.append((n, round(float(dt), 4), round(float(dr), 6)))
    return moved, [n for n in a if n not in b], [n for n in b if n not in a]


def mates_of(doc):
    out = []
    f = wrap(doc.FirstFeature(), sld.IFeature)
    while f is not None:
        if f.GetTypeName2() == "MateGroup":
            s = wrap(f.GetFirstSubFeature(), sld.IFeature)
            while s is not None:
                out.append(s)
                s = wrap(s.GetNextSubFeature(), sld.IFeature)
        f = wrap(f.GetNextFeature(), sld.IFeature)
    return out


def mate_components(s):
    try:
        m = wrap(s.GetSpecificFeature2(), sld.IMate2)
        out = []
        for i in range(m.GetMateEntityCount()):
            rc = m.MateEntity(i).ReferenceComponent
            out.append(wrap(rc, sld.IComponent2).Name2 if rc is not None else "(asm)")
        return out
    except Exception:
        return []


def mate_errors(doc):
    bad = []
    for s in mates_of(doc):
        code, warn = s.GetErrorCode2()
        if code != 0 and not s.IsSuppressed():
            bad.append((s.Name, code, warn))
    return bad


def status(doc, top=True):
    return {n: CS.get(cp.GetConstrainedStatus(), "?") for n, cp in comps(doc, top).items()
            if not cp.IsSuppressed()}


def set_placement(sw, cp, M):
    """3x4 [R|t] (column vectors, mm) -> IComponent2.Transform2 (row vectors, m).
    Sticks only along free degrees of freedom."""
    M = np.array(M, float)
    arr = list(M[:, :3].T.reshape(-1)) + list(M[:, 3] / 1000.0) + [1.0, 0, 0, 0]
    mu = wrap(sw.GetMathUtility(), sld.IMathUtility)
    cp.Transform2 = wrap(mu.CreateTransform(VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, arr)),
                         sld.IMathTransform)


def _xf(M, p):
    return M[:, :3] @ np.asarray(p) + M[:, 3]


def overlap(a, b, tol=0.05):
    return bool(np.all(a[0] <= b[1] + tol) and np.all(b[0] <= a[1] + tol))


# ---- faces + mates ---------------------------------------------------------
class Mater:
    def __init__(self, sw, doc, prefix="BX_"):
        self.sw, self.doc = sw, doc
        self.asm = wrap(doc, sld.IAssemblyDoc)
        self.prefix = prefix
        self._comps = comps(doc)
        self._cache = {}

    def comp(self, name):
        if name not in self._comps:
            self._comps = comps(self.doc)
        return self._comps[name]

    def faces(self, name):
        """Faces of a part component, or of every leaf part under a sub-assembly,
        in the assembly frame.  Cached: nothing is allowed to move."""
        if name not in self._cache:
            self.comp(name)
            kids = [k for k in self._comps if k.startswith(name + "/") and not self._comps[k].IsSuppressed()]
            leaves = [k for k in kids if not any(o.startswith(k + "/") for o in kids)] or [name]
            self._cache[name] = [f for k in leaves for f in self._part_faces(k)]
        return self._cache[name]

    def _part_faces(self, name):
        cp = self.comp(name)
        M = swlib.placement(cp)
        out = []
        r = cp.GetBodies3(c.swSolidBody)
        for b in (r[0] if isinstance(r, tuple) else r) or []:
            for f in wrap(b, sld.IBody2).GetFaces() or []:
                f = wrap(f, sld.IFace2)
                s = wrap(f.GetSurface(), sld.ISurface)
                bx = np.array(f.GetBox()) * 1e3
                P = np.array([_xf(M, (x, y, z)) for x in (bx[0], bx[3]) for y in (bx[1], bx[4])
                              for z in (bx[2], bx[5])])
                d = dict(face=f, comp=name, area=f.GetArea() * 1e6, ctr=_xf(M, (bx[:3] + bx[3:]) / 2),
                         aabb=(P.min(0), P.max(0)))
                if s.IsPlane():
                    p = s.PlaneParams
                    n = np.array(p[:3])
                    if f.FaceInSurfaceSense():        # measured: negate to get the OUTWARD normal
                        n = -n
                    n = M[:, :3] @ n
                    d.update(kind="plane", n=n, off=float(n @ _xf(M, np.array(p[3:6]) * 1e3)))
                elif s.IsCylinder():
                    p = s.CylinderParams
                    d.update(kind="cyl", pt=_xf(M, np.array(p[:3]) * 1e3), axis=M[:, :3] @ np.array(p[3:6]),
                             r=p[6] * 1e3)
                else:
                    continue
                out.append(d)
        return out

    def plane(self, name, n, off):
        n = np.asarray(n, float) / np.linalg.norm(n)
        cand = [f for f in self.faces(name) if f["kind"] == "plane" and f["n"] @ n > 0.99999
                and abs(f["off"] - off) < 10 * TOL]
        if not cand:
            raise LookupError(f"{name}: no plane n={n} off={off}")
        return max(cand, key=lambda f: f["area"])

    def cyl(self, name, axis, through, r=None):
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        out = []
        for f in self.faces(name):
            if f["kind"] != "cyl" or abs(abs(f["axis"] @ a) - 1) > 1e-6:
                continue
            d = f["pt"] - np.asarray(through, float)
            if np.linalg.norm(d - (d @ a) * a) <= 10 * TOL and (r is None or abs(f["r"] - r) <= 0.01):
                out.append(f)
        if not out:
            raise LookupError(f"{name}: no cylinder axis {a} through {through} r={r}")
        return max(out, key=lambda f: f["area"])

    def _select(self, items):
        self.doc.ClearSelection2(True)
        smgr = wrap(self.doc.SelectionManager, sld.ISelectionMgr)
        for i, it in enumerate(items):
            if isinstance(it, str):          # "Front Plane" or a full SelectByID2 plane name
                ok = self.doc.Extension.SelectByID2(it, "PLANE", 0, 0, 0, i > 0, 1, None, 0)
            else:
                sd = wrap(smgr.CreateSelectData(), sld.ISelectData)
                sd.Mark = 1
                ok = wrap(it["face"], sld.IEntity).Select4(i > 0, sd)
            if not ok:
                raise RuntimeError(f"could not select {it if isinstance(it, str) else it['comp']}")

    def _restore(self, before):
        moved, _, _ = compare(before, placements(self.doc))
        for n in sorted(set(x[0].split("/")[0] for x in moved)):
            set_placement(self.sw, self.comp(n), before[n])
        if moved:
            self.doc.EditRebuild3()
        return compare(before, placements(self.doc))[0]

    def mate(self, name, kind, a, b, dist=0.0, lock=False, ref=None):
        """kind: coincident | concentric | distance | parallel.  a/b: face dicts or plane
        names.  Kept only if no component moved (against `ref`, default: now)."""
        full = self.prefix + name
        if wrap(self.asm.FeatureByName(full), sld.IFeature) is not None:
            print(f"    = {full:46s} (exists)")
            return None
        T = {"coincident": c.swMateCOINCIDENT, "concentric": c.swMateCONCENTRIC,
             "distance": c.swMateDISTANCE, "parallel": c.swMatePARALLEL}[kind]
        before = placements(self.doc)
        tries = [(c.swMateAlignCLOSEST, False), (c.swMateAlignALIGNED, False), (c.swMateAlignANTI_ALIGNED, False)]
        if kind == "distance":
            tries += [(al, True) for al, _ in tries]
        last = None
        for align, flip in tries:
            names0 = set(s.Name for s in mates_of(self.doc))
            self._select([a, b])
            res = self.asm.AddMate5(T, align, flip, dist / 1000.0, 0, 0, 0, 0, 0, 0, 0, False, lock, 0)
            mate, err = res if isinstance(res, tuple) else (res, None)
            self.doc.ClearSelection2(True)
            self.doc.EditRebuild3()
            new = [s for s in mates_of(self.doc) if s.Name not in names0]
            ok = mate is not None and err in (None, 1) and len(new) == 1 and new[0].GetErrorCode2()[0] == 0 \
                and not compare(ref if ref is not None else before, placements(self.doc))[0]
            if ok:
                new[0].Name = full
                print(f"    + {full:46s} {kind}{' (rotation locked)' if lock else ''}")
                return new[0]
            last = f"AddMate5 err {err}" if err not in (None, 1) else "moved something or errored"
            for s in new:                     # a refused AddMate5 can still leave a mate behind
                s.Select2(False, 0)
                self.doc.Extension.DeleteSelection2(0)
            self.doc.EditRebuild3()
            left = self._restore(before)
            if left:
                raise SystemExit(f"could not roll back {full}: still moved {left[:3]}")
        raise RuntimeError(f"mate {full} ({kind}) failed: {last}")

    def lock(self, name, a, b):
        """Lock mate between two components: a rigid pair, exactly as placed."""
        full = self.prefix + name
        if wrap(self.asm.FeatureByName(full), sld.IFeature) is not None:
            return None
        before = placements(self.doc)
        names0 = set(s.Name for s in mates_of(self.doc))
        self.doc.ClearSelection2(True)
        smgr = wrap(self.doc.SelectionManager, sld.ISelectionMgr)
        for i, cn in enumerate((a, b)):
            sd = wrap(smgr.CreateSelectData(), sld.ISelectData)
            sd.Mark = 1
            self.comp(cn).Select4(i > 0, sd, False)
        res = self.asm.AddMate5(c.swMateLOCK, c.swMateAlignCLOSEST, False, 0, 0, 0, 0, 0, 0, 0, 0, False, False, 0)
        self.doc.ClearSelection2(True)
        self.doc.EditRebuild3()
        new = [s for s in mates_of(self.doc) if s.Name not in names0]
        if len(new) != 1 or new[0].GetErrorCode2()[0] or compare(before, placements(self.doc))[0]:
            raise SystemExit(f"lock {full} failed")
        new[0].Name = full
        print(f"    + {full:46s} lock")
        return new[0]

    def delete_mates(self, names):
        for n in names:
            f = wrap(self.asm.FeatureByName(n), sld.IFeature)
            if f is not None:
                self.doc.ClearSelection2(True)
                f.Select2(False, 0)
                self.doc.Extension.DeleteSelection2(0)
        self.doc.ClearSelection2(True)
        self.doc.EditRebuild3()


# ---- planning ----------------------------------------------------------------
def _perp(a):
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u1 = np.cross(a, t)
    u1 /= np.linalg.norm(u1)
    return u1, np.cross(a, u1)


def rows_plane(n, p):
    """Twist constraints [v, w] of a face coincidence: one translation, two rotations."""
    t1, t2 = _perp(n)
    return [np.r_[n, np.cross(p, n)], np.r_[0, 0, 0, t1], np.r_[0, 0, 0, t2]]


def rows_axis(a, p):
    u1, u2 = _perp(a)
    return [np.r_[u1, np.cross(p, u1)], np.r_[u2, np.cross(p, u2)], np.r_[0, 0, 0, u1], np.r_[0, 0, 0, u2]]


def rank(rows):
    return int(np.linalg.matrix_rank(np.array(rows), tol=1e-6)) if rows else 0


def contacts(m, cname, hosts, datums=(), max_r=3.0, flush=False):
    """Candidate mates of `cname` against placed hosts: (kind, a, b, rows, score, label).
    Touching faces (opposite normals, same plane, overlapping), coaxial cylinders
    (radii within 0.5 mm: a hole, or a shank in a hole), faces lying in a datum
    plane, and with flush=True coplanar same-facing faces (score -1)."""
    fc = m.faces(cname)
    out = []
    for dname in datums:
        n, off = DATUMS[dname]
        for f in fc:
            if f["kind"] == "plane" and abs(abs(f["n"] @ n) - 1) < 1e-6 and abs((f["n"] @ n) * f["off"] - off) < TOL:
                p = f["ctr"] - ((f["ctr"] @ f["n"]) - f["off"]) * f["n"]
                out.append(("coincident", f, dname, rows_plane(f["n"], p), f["area"] + 1e6, f"face -> {dname}"))
    for h in hosts:
        fh = m.faces(h)
        for fa in fc:
            for fb in fh:
                if fa["kind"] == "plane" == fb["kind"]:
                    d = fa["n"] @ fb["n"]
                    if d < -0.999999 and abs(fa["off"] + fb["off"]) < TOL and overlap(fa["aabb"], fb["aabb"]):
                        lo = np.maximum(fa["aabb"][0], fb["aabb"][0])
                        hi = np.minimum(fa["aabb"][1], fb["aabb"][1])
                        p = (lo + hi) / 2
                        p = p - ((p @ fa["n"]) - fa["off"]) * fa["n"]
                        out.append(("coincident", fa, fb, rows_plane(fa["n"], p),
                                    float(np.prod(np.sort(np.maximum(hi - lo, 0))[1:])), f"face -> {h} face"))
                    elif flush and d > 0.999999 and abs(fa["off"] - fb["off"]) < TOL and min(fa["area"], fb["area"]) >= 5 \
                            and overlap(fa["aabb"], fb["aabb"], 25.0):
                        p = fa["ctr"] - ((fa["ctr"] @ fa["n"]) - fa["off"]) * fa["n"]
                        out.append(("coincident", fa, fb, rows_plane(fa["n"], p), -1.0, f"face flush with {h} face"))
                elif fa["kind"] == "cyl" == fb["kind"] and fa["r"] <= max_r and fb["r"] <= max_r:
                    a = fa["axis"] / np.linalg.norm(fa["axis"])
                    if abs(abs(fb["axis"] @ a) - 1) > 1e-6 or abs(fa["r"] - fb["r"]) > 0.5:
                        continue
                    dd = fb["pt"] - fa["pt"]
                    if np.linalg.norm(dd - (dd @ a) * a) <= TOL and overlap(fa["aabb"], fb["aabb"], 1.0):
                        out.append(("concentric", fa, fb, rows_axis(a, fa["pt"]), fa["r"],
                                    f"r{fa['r']:.2f} -> {h} r{fb['r']:.2f}"))
    return out


def valid(combo):
    """SolidWorks' rule as observed: every translation fixed once (the v-parts of all
    translational rows independent); rotational overlap allowed; rank 6."""
    rows = [r for x in combo for r in x[3]]
    tr = [r[:3] for r in rows if np.linalg.norm(r[:3]) > 1e-9]
    if tr and np.linalg.matrix_rank(np.array(tr), tol=1e-6) < len(tr):
        return False
    return rank(rows) == 6


def plan(m, cname, hosts, datums=(), max_n=4, flush=False):
    """Best minimal valid set of 2..max_n mates.  Preference: datum face (100) >
    touching face (50) > concentric (40) > lock-rotation concentric (35) > flush
    face (30) > parallel (10).  Returns (chosen, rows, candidates)."""
    import itertools
    cands = contacts(m, cname, hosts, datums, flush=flush)
    pool = []
    for x in cands:
        if x[0] == "coincident":
            w = (100 if isinstance(x[2], str) else 30 if x[4] < 0 else 50) + max(min(x[4], 1e5), 0) / 1e5
            pool.append((x[0], x[1], x[2], x[3], w, x[5]))
            t1, t2 = _perp(x[1]["n"])
            pool.append(("parallel", x[1], x[2], [np.r_[0, 0, 0, t1], np.r_[0, 0, 0, t2]], 10, x[5] + " (parallel)"))
        else:
            a = x[1]["axis"] / np.linalg.norm(x[1]["axis"])
            pool.append((x[0], x[1], x[2], x[3], 40, x[5]))
            pool.append((x[0], x[1], x[2], x[3] + [np.r_[0, 0, 0, a]], 35, x[5] + " (rotation locked)", "lock"))
    best = None
    for n_ in range(2, max_n + 1):
        for combo in itertools.combinations(pool, n_):
            keys = [(id(x[1]["face"]), x[2] if isinstance(x[2], str) else id(x[2]["face"])) for x in combo]
            if len(set(keys)) < len(keys) or not valid(combo):
                continue
            if any(valid(combo[:i] + combo[i + 1:]) for i in range(n_)):      # minimal only
                continue
            score = sum(x[4] for x in combo) - 0.5 * n_
            if best is None or score > best[0]:
                best = (score, combo)
    if best is None:
        return [], [], cands
    chosen = list(best[1])
    return chosen, [r for x in chosen for r in x[3]], cands


def apply(m, cname, chosen, tag):
    for i, x in enumerate(chosen):
        lock = len(x) > 6 and x[6] == "lock"
        hn = x[2] if isinstance(x[2], str) else x[2]["comp"].split("/")[-1]
        m.mate(f"{tag}_{i + 1}_{x[0][:5]}{'L' if lock else ''}_{hn}".replace(" ", ""), x[0], x[1], x[2], lock=lock)
