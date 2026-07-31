"""api.py — headless facade + CLI.

Everything the GUI can do is reachable without the GUI, so runs can be
scripted, batched or driven from a notebook.  This module must never import
PyQt6 or matplotlib.

    python -m fourbar_optimizer_gui eval  [--preset p.json]
    python -m fourbar_optimizer_gui sweep [--preset p.json] [--csv out.csv]
    python -m fourbar_optimizer_gui gui
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import LinkageSpec, baseline1
from kinematics import solve_pose, sweep as sweep_poses, trace_world
from geometry import ShapeSet, collisions, RES_FAST, RES_DRAW
from evaluate import find_range, evaluate, Metrics, RangeResult

__all__ = [
    "LinkageSpec", "baseline1", "solve_pose", "find_range", "evaluate",
    "ShapeSet", "collisions", "load_spec", "save_spec", "metrics_dict",
    "sweep_table",
]


def load_spec(path: str | None) -> LinkageSpec:
    return baseline1() if not path else LinkageSpec.load(path)


def save_spec(spec: LinkageSpec, path: str):
    spec.save(path)


def metrics_dict(m: Metrics) -> dict:
    return {
        "valid": m.valid, "reason": m.reason,
        "travel_mm": round(m.travel_mm, 4),
        "max_dev_mm": round(m.max_dev_mm, 4),
        "rms_dev_mm": round(m.rms_dev_mm, 4),
        "stroke_deg": round(m.stroke_deg, 4),
        "mean_x_mm": round(m.mean_x_mm, 4),
        "q_lo_deg": round(math.degrees(m.q_lo), 4),
        "q_hi_deg": round(math.degrees(m.q_hi), 4),
        "stop_lo": m.stop_lo, "stop_hi": m.stop_hi,
    }


def sweep_table(spec: LinkageSpec, n: int = 200, check_collisions: bool = True):
    """Per-angle trace of the primary point across the reachable range."""
    shapes = ShapeSet(spec, RES_FAST)
    rng = find_range(spec, shapes, check_collisions)
    if not rng.valid:
        return rng, []
    tp = spec.primary_trace()
    qs = np.linspace(rng.q_lo, rng.q_hi, n)
    rows = []
    prev = solve_pose(spec, spec.q_seed).alpha
    for q in qs:
        p = solve_pose(spec, float(q), prev)
        if p is None:
            continue
        prev = p.alpha
        wx, wz = trace_world(p, tp)
        hits = collisions(spec, p, shapes)
        rows.append(dict(
            q_deg=math.degrees(q), alpha_deg=math.degrees(p.alpha),
            knee_deg=math.degrees(p.q_knee), kr=p.kr,
            C_x_mm=p.nodes["C"][0] * 1000, C_z_mm=p.nodes["C"][1] * 1000,
            E_x_mm=p.nodes["E"][0] * 1000, E_z_mm=p.nodes["E"][1] * 1000,
            trace_x_mm=wx * 1000, trace_z_mm=wz * 1000,
            collisions="|".join(f"{a}/{b}" for a, b in hits),
        ))
    return rng, rows


# ---------------------------------------------------------------------------
def _cmd_eval(args):
    spec = load_spec(args.preset)
    shapes = ShapeSet(spec, RES_FAST)
    out = {
        "with_collisions": metrics_dict(evaluate(spec, shapes, True)),
        "without_collisions": metrics_dict(evaluate(spec, shapes, False)),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_sweep(args):
    spec = load_spec(args.preset)
    rng, rows = sweep_table(spec, args.n, not args.no_collisions)
    if not rng.valid:
        print(json.dumps({"valid": False, "reason": rng.reason}))
        return 1
    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")
    else:
        print(json.dumps(rows[:: max(1, len(rows) // 20)], indent=2))
    return 0


def _cmd_opt(args):
    from optimize import OptConfig, optimize, DEFAULT_BOUNDS, VARS
    spec = load_spec(args.preset)
    free = tuple(args.free.split(",")) if args.free else VARS
    cfg = OptConfig(free=free, mode=args.mode, tol_mm=args.tol,
                    w_travel=args.w_travel, w_vert=args.w_vert,
                    algo=args.algo, budget=args.budget, seed=args.seed,
                    max_link_mm=args.max_link, csv_path=args.csv)

    last = [0.0]

    def cb(n, f, s, m):
        if args.progress and n - last[0] >= 250:
            last[0] = n
            print(f"[{n:6d}] best {f:9.2f}  travel {m.travel_mm:7.2f} mm  "
                  f"dev {m.max_dev_mm:6.2f} mm", flush=True)

    if args.restarts > 1:
        from optimize import optimize_multi
        multi = optimize_multi(spec, cfg, restarts=args.restarts, callback=cb)
        res = multi.best
        conv = {
            "restarts": len(multi.runs),
            "fitness_per_run": [round(f, 3) for f in multi.fitnesses()],
            "spread_pct": round(multi.spread_pct(), 3),
            "verdict": multi.verdict(),
        }
    else:
        res = optimize(spec, cfg, callback=cb)
        conv = None

    if args.out:
        res.best_spec.save(args.out)

    m = res.best_metrics
    out = {
        "n_evals": res.n_evals, "elapsed_s": round(res.elapsed_s, 2),
        "best_fitness": round(res.best_fitness, 4),
        "stall_evals": res.stall_evals,
        "convergence": conv or res.convergence_note(),
        "metrics": metrics_dict(m),
        "geometry_mm": {k: round(getattr(res.best_spec, k) * 1000, 3)
                        for k in free},
        "link_spans_mm": {k: round(v * 1000, 2)
                          for k, v in res.best_spec.link_spans().items()},
        "out": args.out, "csv": args.csv,
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_gui(args):
    import gui
    return gui.main(args.preset)


def build_parser():
    p = argparse.ArgumentParser(prog="fourbar_optimizer_gui",
                                description="2D 4-bar linkage design lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="print metrics as JSON")
    pe.add_argument("--preset")
    pe.set_defaults(func=_cmd_eval)

    ps = sub.add_parser("sweep", help="per-angle trace table")
    ps.add_argument("--preset")
    ps.add_argument("--csv")
    ps.add_argument("-n", type=int, default=200)
    ps.add_argument("--no-collisions", action="store_true")
    ps.set_defaults(func=_cmd_sweep)

    po = sub.add_parser("opt", help="run a geometry search")
    po.add_argument("--preset")
    po.add_argument("--mode", choices=["constrained", "weighted"],
                    default="constrained")
    po.add_argument("--tol", type=float, default=5.0,
                    help="constrained: max |deviation| in mm")
    po.add_argument("--w-travel", type=float, default=1.0)
    po.add_argument("--w-vert", type=float, default=2.0)
    po.add_argument("--algo", choices=["de", "es"], default="de")
    po.add_argument("--budget", type=int, default=6000)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--max-link", type=float, default=200.0,
                    help="cap on any link span, radius-centre to radius-centre [mm]")
    po.add_argument("--restarts", type=int, default=1,
                    help="independent runs from different seeds; the spread "
                         "across them is the convergence evidence")
    po.add_argument("--free", help="comma-separated subset of the free vars")
    po.add_argument("--out", help="write the winning preset here")
    po.add_argument("--csv", help="log every evaluation")
    po.add_argument("--progress", action="store_true")
    po.set_defaults(func=_cmd_opt)

    pg = sub.add_parser("gui", help="launch the PyQt6 app")
    pg.add_argument("--preset")
    pg.set_defaults(func=_cmd_gui)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
