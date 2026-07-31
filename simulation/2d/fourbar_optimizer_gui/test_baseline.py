"""test_baseline.py — verification of the core before any optimization work.

Run:  python simulation/2d/fourbar_optimizer_gui/test_baseline.py

The load-bearing test is #1: our A-at-origin IK must reproduce the archive's
`baseline1_leg_analysis/physics.py:solve_ik` exactly, once the A_Z frame shift
is applied.  Everything downstream rests on that.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_ARCHIVE = os.path.normpath(os.path.join(
    _HERE, "..", "..", "mujoco", "archive", "baseline1_leg_analysis"))

from model import (LinkageSpec, baseline1, archive_baseline1, ARCHIVE_A_Z,
                   ARCHIVE_Q_RET, ARCHIVE_Q_EXT, ARCHIVE_STROKE_DEG, FEMUR,
                   COUPLER, MOTOR, TIBIA, pair_key)
from kinematics import solve_pose, sweep, branch_is_continuous, trace_world
from geometry import (ShapeSet, disc_hull, sat_overlap, circle_poly_overlap,
                      collisions, transform, RES_FAST, RES_DRAW)
from evaluate import find_range, evaluate

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
def test_ik_vs_archive():
    print("\n1. IK cross-check vs archive physics.py:solve_ik")
    sys.path.insert(0, _ARCHIVE)
    try:
        import importlib
        arch = importlib.import_module("physics")
    except Exception as e:                                    # pragma: no cover
        check("archive import", False, f"{type(e).__name__}: {e}")
        return
    finally:
        pass

    spec = archive_baseline1()      # exact optimized values, not the rounded build
    p_arch = dict(
        L_femur=spec.L_femur, L_stub=spec.L_stub, L_tibia=spec.L_tibia,
        Lc=spec.Lc, F_X=spec.F_X,
        F_Z=spec.to_body_frame_fz(),      # back to body-centre frame
        A_Z=ARCHIVE_A_Z,
    )
    check("F_Z frame shift is +5.29 mm", abs(spec.F_Z - 0.00529) < 1e-9,
          f"F_Z={spec.F_Z*1000:.3f} mm (archive body-frame {p_arch['F_Z']*1000:.2f} mm)")

    worst = 0.0
    n = 0
    for q in np.linspace(-1.60, -0.20, 141):
        a = arch.solve_ik(float(q), p_arch)
        m = solve_pose(spec, float(q))              # seed branch rule, like archive
        if a is None or m is None:
            continue
        n += 1
        for key in ("C", "E", "W"):
            ax, az = a[key]
            mx, mz = m.nodes[key]
            worst = max(worst, abs(ax - mx), abs(az - (mz + ARCHIVE_A_Z)))
    check("C/E/W match archive across sweep", n > 100 and worst < 1e-9,
          f"{n} poses, max abs error {worst:.2e} m")


def test_loop_closure():
    print("\n2. Loop closure  |E-F| == Lc")
    spec = baseline1()
    worst = 0.0
    for q in np.linspace(-1.6, -0.2, 200):
        p = solve_pose(spec, float(q))
        if p:
            worst = max(worst, abs(p.closure_error() - spec.Lc))
    check("coupler length preserved", worst < 1e-12, f"max error {worst:.2e} m")


def test_branch_continuity():
    print("\n3. Assembly-branch continuity")
    spec = baseline1()
    qs = np.linspace(-1.6, -0.2, 400)
    poses = sweep(spec, qs)
    solved = sum(p is not None for p in poses)
    check("alpha continuous across sweep", branch_is_continuous(poses),
          f"{solved}/{len(qs)} poses solved")

    # The archive's per-angle |q_knee| rule, for comparison.
    ind = [solve_pose(spec, float(q)) for q in qs]
    jumps = 0
    prev = None
    for p in ind:
        if p is None:
            prev = None
            continue
        if prev is not None and abs(p.alpha - prev) > math.radians(15):
            jumps += 1
        prev = p.alpha
    print(f"       (archive's independent branch rule shows {jumps} jump(s) "
          f"on this geometry)")


def test_collision_primitives():
    print("\n4. Collision primitives")
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    check("SAT: overlapping squares", sat_overlap(sq, sq + 0.5))
    check("SAT: separated squares", not sat_overlap(sq, sq + 2.0))
    check("SAT: touching-but-clear", not sat_overlap(sq, sq + np.array([1.01, 0.0])))
    check("SAT: rotation causes overlap",
          not sat_overlap(sq, sq + np.array([1.2, 1.2])) and
          sat_overlap(sq, transform(sq - 0.5, math.radians(45), (1.35, 1.35))))
    check("circle inside polygon", circle_poly_overlap((0.5, 0.5), 0.1, sq))
    check("circle touching edge", circle_poly_overlap((1.05, 0.5), 0.1, sq))
    check("circle clear of polygon", not circle_poly_overlap((1.5, 0.5), 0.1, sq))
    check("circle near vertex", circle_poly_overlap((1.05, 1.05), 0.09, sq))
    check("circle clear of vertex", not circle_poly_overlap((1.2, 1.2), 0.2, sq))

    # Disc hull of two different-radius discs = tapered slot.
    h = disc_hull([("a", 0.0, 0.0, 0.02), ("b", 0.10, 0.0, 0.01)], 32)
    ok = (abs(h[:, 0].min() - (-0.02)) < 1e-9 and abs(h[:, 0].max() - 0.11) < 1e-9
          and abs(h[:, 1].max() - 0.02) < 1e-3)
    check("tapered hull spans both discs", ok,
          f"x [{h[:,0].min()*1000:.1f}, {h[:,0].max()*1000:.1f}] mm, "
          f"z max {h[:,1].max()*1000:.2f} mm")


def test_range_no_collisions():
    print("\n5. Archive-baseline range with collisions DISABLED (vs archive stroke)")
    spec = archive_baseline1()
    r = find_range(spec, ShapeSet(spec, RES_FAST), check_collisions=False)
    if not check("range solved", r.valid, r.reason):
        return
    print(f"       q_lo={math.degrees(r.q_lo):8.3f}deg  ({r.stop_lo})")
    print(f"       q_hi={math.degrees(r.q_hi):8.3f}deg  ({r.stop_hi})")
    print(f"       archive: Q_EXT={math.degrees(ARCHIVE_Q_EXT):.3f}deg  "
          f"Q_RET={math.degrees(ARCHIVE_Q_RET):.3f}deg  stroke={ARCHIVE_STROKE_DEG}deg")
    # The archive trimmed 3 deg of margin off each singular end and clamped
    # Q_ret at -0.35 rad, so our raw singular-to-singular span is wider.
    check("range brackets the archive stroke",
          r.q_lo <= ARCHIVE_Q_EXT and r.q_hi >= ARCHIVE_Q_RET,
          f"ours {r.stroke_deg:.2f}deg vs archive {ARCHIVE_STROKE_DEG}deg")


def test_range_with_collisions():
    print("\n6. Baseline range with collisions ENABLED  <-- the new information")
    spec = baseline1()
    shapes = ShapeSet(spec, RES_FAST)
    m_free = evaluate(spec, shapes, check_collisions=False)
    m_coll = evaluate(spec, shapes, check_collisions=True)

    if not check("evaluates with collisions on", m_coll.valid, m_coll.reason):
        return
    print(f"       collisions OFF : {m_free.summary()}")
    print(f"       collisions ON  : {m_coll.summary()}")
    print(f"       blocked by     : lo={m_coll.stop_lo}  hi={m_coll.stop_hi}")
    print(f"       >>> real link widths COST "
          f"{m_free.stroke_deg - m_coll.stroke_deg:.2f}deg of stroke and "
          f"{m_free.travel_mm - m_coll.travel_mm:.1f} mm of travel")
    check("collision range is a subset of the free range",
          m_coll.stroke_deg <= m_free.stroke_deg + 1e-6)


def test_metrics():
    print("\n7. Traced-path metrics")
    spec = baseline1()
    m = evaluate(spec, ShapeSet(spec, RES_FAST))
    if not check("metrics computed", m.valid, m.reason):
        return
    print(f"       traced point : {spec.primary_trace().name}")
    print(f"       travel       : {m.travel_mm:.2f} mm")
    print(f"       max |dev|    : {m.max_dev_mm:.2f} mm   (rms {m.rms_dev_mm:.2f})")
    print(f"       mean x       : {m.mean_x_mm:.2f} mm")
    check("path sampled", m.path is not None and len(m.path) > 100,
          f"{0 if m.path is None else len(m.path)} points")
    check("travel is positive", m.travel_mm > 1.0)


def test_wheel_and_dogleg():
    print("\n8. Optional wheel + dogleg tibia")
    spec = baseline1()
    spec.wheel_enabled = True
    m = evaluate(spec, ShapeSet(spec, RES_FAST))
    check("wheel enabled still evaluates", m.valid, m.reason or m.summary())

    spec2 = baseline1()
    spec2.w_perp = 0.020
    spec2.sync_primary_to_W()
    p0 = solve_pose(spec2, spec2.q_seed)
    base = solve_pose(baseline1(), spec2.q_seed)
    d = math.dist(p0.nodes["W"], base.nodes["W"])
    check("dogleg moves W by w_perp", abs(d - 0.020) < 1e-9, f"moved {d*1000:.3f} mm")
    check("dogleg does not break closure",
          abs(p0.closure_error() - spec2.Lc) < 1e-12)


def test_headless_layering():
    print("\n9. Core imports without any GUI toolkit")
    import importlib
    blocked = {"PyQt6": None, "matplotlib": None}
    saved = {k: sys.modules.get(k) for k in blocked}
    for k in blocked:
        sys.modules[k] = None
    try:
        for mod in ("model", "kinematics", "geometry", "evaluate"):
            importlib.reload(importlib.import_module(mod))
        ok = True
        detail = ""
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    check("model/kinematics/geometry/evaluate are GUI-free", ok, detail)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("fourbar_optimizer_gui — baseline verification")
    print("=" * 72)

    test_ik_vs_archive()
    test_loop_closure()
    test_branch_continuity()
    test_collision_primitives()
    test_range_no_collisions()
    test_range_with_collisions()
    test_metrics()
    test_wheel_and_dogleg()
    test_headless_layering()

    n_pass = sum(1 for _, ok in _results if ok)
    print("\n" + "=" * 72)
    print(f"{n_pass}/{len(_results)} checks passed")
    print("=" * 72)
    sys.exit(0 if n_pass == len(_results) else 1)
