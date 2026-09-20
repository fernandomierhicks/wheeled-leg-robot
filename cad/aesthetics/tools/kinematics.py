r"""Solve the leg's 4-bar from the exported poses, so the sweep can be sampled
at any hip angle instead of only the three angles SolidWorks happened to write.

    C:/Users/ferna/cadenv/Scripts/python.exe tools/kinematics.py --validate
    C:/Users/ferna/cadenv/Scripts/python.exe tools/kinematics.py --poses 13

WHY.  `collide.py` samples THREE poses.  Decision 17 reasoned that because the
worst interference landed at Retracted -- the Q_RET hard stop -- the poses must
bracket the travel.  That is evidence, not proof: it shows one PAIR peaks at an
endpoint, and says nothing about another pair peaking in between.  Decision 26
makes the collision check the final arbiter of the whole rebuild, so the judge
should not rest on an assumption nobody has tested.

WHAT IT DOES.  Nothing is taken from documentation; everything is recovered
from the three exported assemblies, which are ground truth:

  1. Group the leaf instances by how they move.  Anything whose transform is
     identical in all three poses is FIXED (body, hip stator, the mount plates).
     The rest fall into rigid groups that share a transform -- the femur group,
     the coupler group, the tibia group (which carries the wheel and encoder).
  2. A fixed-pivot group rotates about one point: solving
     (R_i - R_j) p = -(t_i - t_j) recovers it.  That gives A for the femur and
     F for the coupler.
  3. A joint shared by two groups is a point fixed in BOTH local frames, so
     R_a(i) c_a + t_a(i) = R_b(i) c_b + t_b(i) holds at every pose.  That is
     linear in (c_a, c_b): two equations per pose, four unknowns, three poses --
     over-determined, so the residual is a real check rather than a fit.
  4. With A, F, |AC|, |FE|, |CE| known the mechanism closes analytically at any
     hip angle q, and every moving part's transform follows from its group.

THE VALIDATION IS THE POINT.  `--validate` reconstructs the three exported poses
from the solved model and reports the worst vertex error in millimetres.  If
that is not small, the model is wrong and must not be used to claim a part is
collision free -- so it prints a verdict rather than leaving it to judgement.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import paths
from collide import leaves, POSES

TOL_FIXED = 1e-6        # mm; below this two transforms are the same placement
TOL_GROUP = 1e-6


def mat(loc):
    t = loc.wrapped.Transformation()
    return np.array([[t.Value(r + 1, c + 1) for c in range(4)] for r in range(3)])


def planar(M):
    """(theta, tx, ty) for a placement that is a rotation about Z."""
    R = M[:3, :3]
    off = max(abs(R[0, 2]), abs(R[1, 2]), abs(R[2, 0]), abs(R[2, 1]))
    if off > 1e-6:
        return None
    return np.arctan2(R[1, 0], R[0, 0]), M[0, 3], M[1, 3], R[2, 2], M[2, 3]


def read_poses(poses):
    """{pose: {name#i: 3x4 matrix}} plus the instance order, from the exports."""
    out, order = {}, None
    for pose in poses:
        step = os.path.join(paths.EXPORTS, pose + ".STEP")
        if not os.path.exists(step):
            raise SystemExit(f"missing pose export: {step}")
        inst = leaves(step)
        seen, d, names = {}, {}, []
        for name, shp, loc in inst:
            k = seen.get(name, 0)
            seen[name] = k + 1
            key = f"{name}#{k}"
            d[key] = mat(loc)
            names.append(key)
        out[pose] = d
        if order is None:
            order = names
        elif set(names) != set(order):
            raise SystemExit(f"{pose} has a different instance set from the first "
                             f"pose; the exports are not the same assembly")
    return out, order


def delta(M0, M1):
    """The world-frame motion carrying placement M0 to M1, as a 3x4 affine."""
    R0, t0 = M0[:3, :3], M0[:3, 3]
    R1, t1 = M1[:3, :3], M1[:3, 3]
    R = R1 @ R0.T
    return np.hstack([R, (t1 - R @ t0)[:, None]])


def group_motion(P, order, poses):
    """Split instances into FIXED and rigid moving bodies.

    A rigid body is defined by RELATIVE motion, not by placement: every part
    bolted to the tibia undergoes the SAME world motion from pose to pose even
    though each sits somewhere different.  Keying on absolute placement instead
    puts all 24 screws in 24 singleton groups, which is what the first version
    of this did.
    """
    base = poses[0]
    fixed, moving = [], {}
    for key in order:
        sig = []
        for pose in poses[1:]:
            sig.append(np.round(delta(P[base][key], P[pose][key]), 4).ravel())
        s = np.concatenate(sig)
        ident = np.hstack([np.eye(3), np.zeros((3, 1))]).ravel()
        if np.abs(s - np.tile(ident, len(poses) - 1)).max() <= 1e-4:
            fixed.append(key)
        else:
            moving.setdefault(tuple(s.tolist()), []).append(key)
    return fixed, list(moving.values())


def pivot(P, poses, key):
    """The fixed point of a group that rotates about one centre, least squares.

    Solves for the pivot in the part's LOCAL frame -- (R_i - R_j) u = t_j - t_i
    is the condition that material point u lands in the same world spot in every
    pose -- then returns it in WORLD coordinates.  Returning `u` itself and using
    it as a world point is a silent 70 mm error: it made |AC| come out 113.6 mm
    against the 187.58 mm in CLAUDE.md, and the reconstruction missed by 585 mm.
    """
    A, b = [], []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            Mi, Mj = P[poses[i]][key], P[poses[j]][key]
            A.append(Mi[:2, :2] - Mj[:2, :2])
            b.append(-(Mi[:2, 3] - Mj[:2, 3]))
    A = np.vstack(A); b = np.concatenate(b)
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = float(np.abs(A @ u - b).max())
    M0 = P[poses[0]][key]
    return M0[:2, :2] @ u + M0[:2, 3], res


def joint(P, poses, ka, kb):
    """The point shared by two moving groups, in each group's LOCAL frame."""
    rows, rhs = [], []
    for p in poses:
        Ma, Mb = P[p][ka], P[p][kb]
        R = np.zeros((2, 4))
        R[:, :2] = Ma[:2, :2]
        R[:, 2:] = -Mb[:2, :2]
        rows.append(R)
        rhs.append(Mb[:2, 3] - Ma[:2, 3])
    A = np.vstack(rows); b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = float(np.abs(A @ x - b).max())
    return x[:2], x[2:], res


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


class Leg:
    """The solved mechanism.  `at(q)` gives every moving group's planar placement."""

    def __init__(self, P, poses, femur_key, coupler_key, tibia_key):
        self.P, self.poses = P, poses
        self.fk, self.ck, self.tk = femur_key, coupler_key, tibia_key
        self.A, rA = pivot(P, poses, femur_key)
        self.F, rF = pivot(P, poses, coupler_key)
        self.c_f, self.c_t, rC = joint(P, poses, femur_key, tibia_key)
        self.e_c, self.e_t, rE = joint(P, poses, coupler_key, tibia_key)
        self.res = {"pivot_A": rA, "pivot_F": rF, "joint_C": rC, "joint_E": rE}

        self.q0 = {p: self._theta(p, femur_key) for p in poses}
        self.th_f0 = self._theta(poses[0], femur_key)
        # link lengths, from the solved points at the reference pose
        C0 = self._world(poses[0], femur_key, self.c_f)
        E0 = self._world(poses[0], coupler_key, self.e_c)
        self.AC = np.linalg.norm(C0 - self.A)
        self.FE = np.linalg.norm(E0 - self.F)
        self.CE = np.linalg.norm(E0 - C0)
        # Which of the two circle-circle solutions the real mechanism uses.
        # Derive it the same way `at()` consumes it -- as the sign of E0's
        # component along the normal n -- rather than from a cross product whose
        # argument order is easy to flip.  Getting it backwards reconstructs the
        # mirror-image linkage, which is silent: link lengths stay perfect and
        # the femur still lands exactly, while the coupler and tibia go 400 mm out.
        u0 = (self.F - C0) / np.linalg.norm(self.F - C0)
        n0 = np.array([-u0[1], u0[0]])
        self.branch = float(np.sign(np.dot(E0 - C0, n0))) or 1.0

    def _theta(self, pose, key):
        M = self.P[pose][key]
        return np.arctan2(M[1, 0], M[0, 0])

    def travel(self):
        """The travel, as offsets from pose[0], measured the SHORT way round.

        The exported hip angles are 152.00, -160.02 and -123.00 degrees: the
        travel straddles +-180, so `linspace(min, max)` walks the long way round
        through 0 and leaves the mechanism unable to close over most of it.
        Offsets from the first pose, wrapped to (-180, 180], put them back in
        order -- and they come out 0, 47.98 and 85.00 degrees, i.e. exactly the
        85 degrees of stop-to-stop travel CLAUDE.md records, recovered here from
        the transforms alone.
        """
        q0 = self._theta(self.poses[0], self.fk)
        d = [(self._theta(p, self.fk) - q0 + np.pi) % (2 * np.pi) - np.pi
             for p in self.poses]
        return min(d), max(d)

    def sweep(self, n):
        """`n` hip angles spanning the exported travel, endpoints included."""
        q0 = self._theta(self.poses[0], self.fk)
        d0, d1 = self.travel()
        return [(q0 + d + np.pi) % (2 * np.pi) - np.pi
                for d in np.linspace(d0, d1, n)]

    def _world(self, pose, key, local_xy):
        M = self.P[pose][key]
        return M[:2, :2] @ local_xy + M[:2, 3]

    def _place(self, pose, key, th, origin_world, local_anchor):
        """Rebuild a 3x4 placement: same Z behaviour as the export, new planar part."""
        M = self.P[pose][key].copy()
        R2 = rot(th)
        M[:2, :2] = R2 @ (rot(-self._theta(pose, key)) @ self.P[pose][key][:2, :2])
        M[:2, 3] = origin_world - M[:2, :2] @ local_anchor
        return M

    def at(self, q, ref=None):
        """Planar placements for the three groups at hip angle `q` (radians,
        absolute, in the same convention as the exported femur angle)."""
        ref = ref or self.poses[0]
        dth = q - self._theta(ref, self.fk)
        # femur: pure rotation about A
        Mf = self.P[ref][self.fk].copy()
        R = rot(dth)
        Mf[:2, :2] = R @ Mf[:2, :2]
        Mf[:2, 3] = self.A + R @ (Mf[:2, 3] - self.A)
        C = Mf[:2, :2] @ self.c_f + Mf[:2, 3]

        # coupler end E: circle(F, FE) x circle(C, CE)
        d = self.F - C
        L = np.linalg.norm(d)
        if L > self.CE + self.FE or L < abs(self.CE - self.FE):
            return None                      # mechanism cannot close here
        a = (self.CE ** 2 - self.FE ** 2 + L ** 2) / (2 * L)
        h2 = self.CE ** 2 - a ** 2
        if h2 < 0:
            return None
        h = np.sqrt(h2)
        u = d / L
        n = np.array([-u[1], u[0]])
        E = C + a * u + self.branch * h * n

        # coupler: rotation about F carrying its local e_c onto E
        th_c0 = self._theta(ref, self.ck)
        v0 = self._world(ref, self.ck, self.e_c) - self.F
        v1 = E - self.F
        dthc = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        Mc = self.P[ref][self.ck].copy()
        Rc = rot(dthc)
        Mc[:2, :2] = Rc @ Mc[:2, :2]
        Mc[:2, 3] = self.F + Rc @ (Mc[:2, 3] - self.F)

        # tibia: the rigid motion carrying (C0,E0) onto (C,E)
        C0 = self._world(ref, self.tk, self.c_t)
        E0 = self._world(ref, self.tk, self.e_t)
        a0, a1 = E0 - C0, E - C
        dtht = np.arctan2(a1[1], a1[0]) - np.arctan2(a0[1], a0[0])
        Mt = self.P[ref][self.tk].copy()
        Rt = rot(dtht)
        Mt[:2, :2] = Rt @ Mt[:2, :2]
        Mt[:2, 3] = C + Rt @ (Mt[:2, 3] - C0)
        return {"femur": Mf, "coupler": Mc, "tibia": Mt}


def build(poses=None, verbose=True):
    poses = list(poses or POSES)
    P, order = read_poses(poses)
    fixed, groups = group_motion(P, order, poses)
    if verbose:
        print(f"{len(order)} instances: {len(fixed)} fixed, {len(groups)} moving group(s)")
        for gi, g in enumerate(groups):
            print(f"  group {gi}: {len(g):2d}  {', '.join(sorted(x.split('#')[0] for x in g)[:6])}"
                  + (" ..." if len(g) > 6 else ""))
    def find(sub):
        for g in groups:
            if any(x.split("#")[0] == sub for x in g):
                return g, next(x for x in g if x.split("#")[0] == sub)
        raise SystemExit(f"{sub} is not in any moving group -- cannot solve the leg")
    gF, kF = find("Femur")
    gC, kC = find("Coupler")
    gT, kT = find("Tibia")
    leg = Leg(P, poses, kF, kC, kT)
    leg.groups = {"femur": gF, "coupler": gC, "tibia": gT}
    leg.fixed = fixed
    return leg


def validate(leg, verbose=True):
    """Reconstruct every exported pose and report the worst placement error."""
    worst = 0.0
    print("\nvalidation -- rebuilding each exported pose from the solved model:")
    for pose in leg.poses:
        q = leg._theta(pose, leg.fk)
        got = leg.at(q)
        if got is None:
            print(f"  {pose:<22} MECHANISM FAILED TO CLOSE"); return False, 1e9
        row = []
        for nm, key in (("femur", leg.fk), ("coupler", leg.ck), ("tibia", leg.tk)):
            want = leg.P[pose][key]
            M = got[nm]
            dR = float(np.abs(M[:2, :2] - want[:2, :2]).max())
            dt = float(np.abs(M[:2, 3] - want[:2, 3]).max())
            # rotation error over a 250 mm link, as a distance
            err = dt + dR * 250.0
            worst = max(worst, err)
            row.append(f"{nm} {err:6.3f}")
        print(f"  {pose:<22} " + "   ".join(row) + "  mm")
    print(f"\n  solve residuals: " + ", ".join(f"{k} {v:.2e}" for k, v in leg.res.items()))
    print(f"  link lengths: |AC| {leg.AC:.3f}   |FE| {leg.FE:.3f}   |CE| {leg.CE:.3f} mm")
    ok = worst < 0.05
    print(f"  worst reconstruction error {worst:.4f} mm  -> "
          + ("MODEL VALIDATED" if ok else "MODEL IS WRONG, DO NOT USE IT"))
    return ok, worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--poses", type=int, default=0,
                    help="if set, print N hip angles spanning the exported range")
    a = ap.parse_args()
    leg = build()
    ok, worst = validate(leg)
    if a.poses and ok:
        print(f"\nexported hip angles: " +
              ", ".join(f"{p}={np.degrees(leg._theta(p, leg.fk)):.2f}deg" for p in leg.poses))
        qs = leg.sweep(a.poses)
        d0, d1 = leg.travel()
        print(f"travel spans {np.degrees(d1 - d0):.2f} deg measured from "
              f"{leg.poses[0]}; sampling {a.poses}:")
        n_ok = 0
        for q in qs:
            got = leg.at(q)
            n_ok += got is not None
            print(f"  q={np.degrees(q):8.2f} deg  " +
                  ("closes" if got is not None else "DOES NOT CLOSE"))
        print(f"{n_ok}/{a.poses} angles close")
    sys.exit(0 if ok else 1)
