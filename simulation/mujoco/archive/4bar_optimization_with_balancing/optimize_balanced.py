"""optimize_balanced.py  —  4-bar geometry optimizer with active balancing.

Phase 0  --mode oat   One-at-a-time sensitivity scan
Phase 1  --mode grid  2-D grid on two named parameters
Phase 2  --mode evo   (1+lambda)-ES evolutionary optimizer

Run:
    python simulation/mujoco/4bar_optimization_with_balancing/optimize_balanced.py --mode evo --hours 8
"""

import argparse
import copy
import csv as _csv
import math
import multiprocessing
import os
import sys
import time
from collections import deque

import numpy as np

# ── locate eval_jump_balanced in same directory ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from eval_jump_balanced import (
    DEFAULT, CSV_COLS, CSV_PATH, MOTOR_MASS,
    _log_csv,
)

# ── Search bounds for the 6 free geometry parameters ────────────────────────
PARAM_RANGES = {
    'L_femur': (0.050, 0.180),
    'L_stub':  (0.022, 0.080),
    'L_tibia': (0.060, 0.180),
    'Lc':      (0.050, 0.180),
    'F_X':     (-0.060, 0.000),
    'F_Z':     (-0.0185, 0.0565),
}

# ── ES hyper-parameters ───────────────────────────────────────────────────────
LAMBDA          = 8
SIGMA_FRAC_INIT = 0.15
SIGMA_FRAC_MIN  = 1e-4
SIGMA_FRAC_MAX  = 0.50
SUCCESS_TARGET  = 1.0 / 5.0
ADAPT_WINDOW    = 10
ADAPT_UP        = 1.22
ADAPT_DOWN      = 1.0 / 1.22


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _params_str(p: dict) -> str:
    return (f"L_f={p['L_femur']*1000:.0f}mm  L_s={p['L_stub']*1000:.0f}mm  "
            f"L_t={p['L_tibia']*1000:.0f}mm  Lc={p['Lc']*1000:.0f}mm  "
            f"F_X={p['F_X']*1000:.0f}mm  F_Z={p['F_Z']*1000:.1f}mm")

def _row_to_params(row: dict) -> dict:
    p = copy.deepcopy(DEFAULT)
    for k in PARAM_RANGES:
        col = k + '_mm'
        try:
            val = row.get(col, '')
            if val not in ('', None):
                p[k] = float(val) / 1000.0
        except (ValueError, TypeError):
            pass
    return p

def _load_best_from_csv() -> tuple:
    if not os.path.exists(CSV_PATH):
        return copy.deepcopy(DEFAULT), -1.0
    best_h, best_row = -1.0, None

    def _read_and_find_best(reader):
        local_best_h, local_best_row = -1.0, None
        for row in reader:
            if row.get('status') == 'PASS':
                try:
                    h = float(row.get('jump_height_mm') or -1)
                    if h > local_best_h:
                        local_best_h, local_best_row = h, row
                except (ValueError, TypeError): pass
        return local_best_h, local_best_row

    try:
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            best_h, best_row = _read_and_find_best(_csv.DictReader(f))
    except (FileNotFoundError, UnicodeDecodeError):
        try:
            with open(CSV_PATH, newline='', encoding='latin-1') as f:
                best_h, best_row = _read_and_find_best(_csv.DictReader(f))
        except FileNotFoundError: return copy.deepcopy(DEFAULT), -1.0

    if best_row is None: return copy.deepcopy(DEFAULT), -1.0
    return _row_to_params(best_row), best_h

def _write_worker_result(row: dict) -> int:
    from eval_jump_balanced import _next_run_id
    row['run_id'] = _next_run_id()
    _log_csv(row)
    h = row.get('jump_height_mm')
    h_str = f"{h:6.1f} mm" if isinstance(h, (int, float)) else "  FAIL  "
    warn = row.get('bearing_warnings') or ''
    print(f"[{row['run_id']:4d}] {row['label']:<30s}  {h_str}" + (f"  WARN:{warn}" if warn else ""))
    return row['run_id']


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------
def _eval_worker(args):
    p, label = args
    import datetime
    from eval_jump_balanced import check_feasibility, find_stroke, validate_stroke, run_balanced_sim, MOTOR_MASS

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    AF_mm = math.sqrt(p['F_X']**2 + (p['F_Z'] - p['A_Z'])**2) * 1000.0
    af_clearance = AF_mm - 26.5
    total_mass_g = (p['m_box'] + p['m_femur'] + p['m_tibia'] + p['m_coupler'] + p['m_wheel'] + MOTOR_MASS) * 1000.0

    row = dict(
        run_id=0, label=label, timestamp=ts,
        L_femur_mm=round(p['L_femur']*1000, 2), L_stub_mm=round(p['L_stub']*1000, 2),
        L_tibia_mm=round(p['L_tibia']*1000, 2), Lc_mm=round(p['Lc']*1000, 2),
        F_X_mm=round(p['F_X']*1000, 2), F_Z_mm=round(p['F_Z']*1000, 2),
        A_Z_mm=round(p['A_Z']*1000, 2), AF_mm=round(AF_mm, 2),
        AF_motor_clearance_mm=round(af_clearance, 2),
        m_box_g=round(p['m_box']*1000, 1), total_mass_g=round(total_mass_g, 1),
    )

    ok, reason, computed, bearing_warnings = check_feasibility(p)
    row.update(computed); row['bearing_warnings'] = bearing_warnings
    if not ok:
        row.update(status='FAIL', fail_reason=reason); return row

    stroke = find_stroke(p)
    if stroke is None:
        row.update(status='FAIL', fail_reason='stroke search failed'); return row
    Q_ret, Q_ext = stroke

    sv_ok, sv_reason = validate_stroke(p, Q_ret, Q_ext)
    if not sv_ok:
        row.update(status='FAIL', fail_reason=sv_reason); return row

    row.update(
        Q_retracted_rad=round(Q_ret, 5), Q_extended_rad=round(Q_ext, 5),
        stroke_deg=round(abs(math.degrees(Q_ret - Q_ext)), 2)
    )

    h_mm, sim_err = run_balanced_sim(p, Q_ret, Q_ext)
    if sim_err:
        row.update(status='FAIL', fail_reason=sim_err); return row

    row.update(jump_height_mm=round(h_mm, 2), status='PASS', fail_reason=''); return row


# ---------------------------------------------------------------------------
# Phase 0: One-at-a-time sensitivity scan
# ---------------------------------------------------------------------------
def run_oat(n_levels: int = 10, n_workers: int = None):
    n_params, n_runs = len(PARAM_RANGES), n_levels * len(PARAM_RANGES)
    if n_workers is None: n_workers = multiprocessing.cpu_count()

    print("="*68 + f"\nPHASE 0: One-at-a-time sensitivity scan\n"
          f"  {n_levels} levels x {n_params} params = {n_runs} runs ({n_workers} workers)\n"
          f"  CSV: {CSV_PATH}\n" + "="*68)

    base, param_results, t0 = copy.deepcopy(DEFAULT), {}, time.perf_counter()

    for pname, (lo, hi) in PARAM_RANGES.items():
        levels = np.linspace(lo, hi, n_levels)
        def _make(pn, v): p = copy.deepcopy(base); p[pn] = float(v); return p
        args = [(_make(pname, v), f"oat_{pname}_{v*1000:.0f}mm") for v in levels]

        print(f"\n--- {pname}  ({lo*1000:.0f}–{hi*1000:.0f} mm) ---")
        with multiprocessing.Pool(processes=min(n_workers, n_levels)) as pool:
            rows = pool.map(_eval_worker, args)

        heights = []
        for (p_dict, _), row in zip(args, rows):
            _write_worker_result(row)
            h = row.get('jump_height_mm')
            heights.append((p_dict[pname] * 1000, h if isinstance(h, (int, float)) else None))
        param_results[pname] = heights

    elapsed = time.perf_counter() - t0
    print("\n" + "="*68 + "\nOAT SENSITIVITY SUMMARY\n"
          f"{'Parameter':<12} {'min_h':>7} {'max_h':>7} {'range':>7} {'best_val':>10} {'valid':>8}\n" + "-"*68)
    ranked = []
    for pname, vh_list in param_results.items():
        valid = [(v, h) for v, h in vh_list if h is not None]
        if not valid: print(f"{pname:<12}  all FAIL"); continue
        hs, best_v = [h for _, h in valid], valid[int(np.argmax([h for _, h in valid]))][0]
        rng = max(hs) - min(hs)
        ranked.append((pname, min(hs), max(hs), rng, best_v, len(valid)))
        print(f"{pname:<12} {min(hs):>7.1f} {max(hs):>7.1f} {rng:>7.1f} {best_v:>10.1f} {len(valid):>4d}/{len(vh_list)}")

    print("="*68)
    ranked.sort(key=lambda x: -x[3])
    print(f"Most sensitive: {', '.join(r[0] for r in ranked[:3])}")
    print(f"Total: {elapsed:.1f} s ({n_runs} runs @ ~{elapsed/n_runs*1000:.0f} ms each)")
    if len(ranked) >= 2:
        print(f"\nSuggested next step:\n  python {os.path.basename(__file__)} --mode grid "
              f"--p1 {ranked[0][0]} --p2 {ranked[1][0]} --levels 7")


# ---------------------------------------------------------------------------
# Phase 1: 2-D grid sweep
# ---------------------------------------------------------------------------
def run_grid(p1: str, p2: str, levels: int = 5, n_workers: int = None):
    if p1 not in PARAM_RANGES or p2 not in PARAM_RANGES:
        sys.exit(f"Unknown param. Choose from: {list(PARAM_RANGES.keys())}")
    if n_workers is None: n_workers = multiprocessing.cpu_count()

    n_runs = levels * levels
    print("="*68 + f"\nPHASE 1: 2-D grid  {p1} x {p2}  ({levels}x{levels} = {n_runs} runs) ({n_workers} workers)\n"
          f"  CSV: {CSV_PATH}\n" + "="*68)

    p1_vals, p2_vals, t0 = np.linspace(*PARAM_RANGES[p1], levels), np.linspace(*PARAM_RANGES[p2], levels), time.perf_counter()
    args = []
    for v1 in p1_vals:
        for v2 in p2_vals:
            p = copy.deepcopy(DEFAULT); p[p1], p[p2] = float(v1), float(v2)
            args.append((p, f"grid_{p1}_{v1*1000:.0f}_{p2}_{v2*1000:.0f}"))

    with multiprocessing.Pool(processes=min(n_workers, n_runs)) as pool:
        rows = pool.map(_eval_worker, args)

    best_h, best_p = -1.0, None
    for (p_dict, _), row in zip(args, rows):
        _write_worker_result(row)
        h = row.get('jump_height_mm')
        if isinstance(h, (int, float)) and h > best_h: best_h, best_p = h, p_dict

    elapsed = time.perf_counter() - t0
    print(f"\nGrid complete in {elapsed:.1f} s")
    if best_p: print(f"Best: {best_h:.1f} mm  —  {p1}={best_p[p1]*1000:.1f} mm  {p2}={best_p[p2]*1000:.1f} mm")
    print(f"\nSuggested next step:\n  python {os.path.basename(__file__)} --mode evo --hours 8")


# ---------------------------------------------------------------------------
# Phase 2: (1+lambda)-ES evolutionary optimizer
# ---------------------------------------------------------------------------
def run_evo(hours: float = None, max_iters: int = None, seed: int = None, n_workers: int = None):
    if hours is None and max_iters is None: hours = 8.0
    if n_workers is None: n_workers = multiprocessing.cpu_count()
    if seed is not None: np.random.seed(seed)

    parent, parent_h = _load_best_from_csv()
    if parent_h < 0:
        print("No prior PASS results in CSV. Evaluating baseline first...")
        from eval_jump_balanced import evaluate
        row = evaluate(copy.deepcopy(DEFAULT), label="evo_seed")
        parent_h = row.get('jump_height_mm', -1.0)
        if not isinstance(parent_h, float): parent_h = -1.0
        parent = copy.deepcopy(DEFAULT)
    else:
        print(f"Seeding from best CSV result: {parent_h:.1f} mm")

    sigmas = {k: (hi - lo) * SIGMA_FRAC_INIT for k, (lo, hi) in PARAM_RANGES.items()}
    t_start, deadline = time.perf_counter(), (time.perf_counter() + hours * 3600.0) if hours else None
    gen, n_evals, n_invalid, best_h, best_gen = 0, 0, 0, parent_h, 0
    success_window = deque(maxlen=ADAPT_WINDOW)

    print("="*68 + f"\nPHASE 2: (1+{LAMBDA})-ES  workers={n_workers}\n"
          f"  Starting from: {parent_h:.1f} mm\n  {_params_str(parent)}")
    if hours: print(f"  Duration: {hours:.1f} hours ({hours*60:.0f} min)")
    else: print(f"  Max generations: {max_iters}")
    print("="*68)

    with multiprocessing.Pool(processes=n_workers) as pool:
        while True:
            if (deadline and time.perf_counter() >= deadline) or (max_iters is not None and gen >= max_iters): break

            children = []
            for _ in range(LAMBDA):
                child = copy.deepcopy(parent)
                for k, (lo, hi) in PARAM_RANGES.items():
                    child[k] = _clamp(child[k] + np.random.normal(0.0, sigmas[k]), lo, hi)
                children.append(child)

            args = [(c, f"evo_g{gen:06d}_c{i}") for i, c in enumerate(children)]
            rows = pool.map(_eval_worker, args)

            gen_best_h, gen_best_p = -1.0, None
            for child_p, row in zip(children, rows):
                _write_worker_result(row); n_evals += 1
                h = row.get('jump_height_mm')
                if row.get('status') == 'PASS' and isinstance(h, (int, float)):
                    if h > gen_best_h: gen_best_h, gen_best_p = h, child_p
                else: n_invalid += 1

            improved = (gen_best_p is not None and gen_best_h > parent_h)
            success_window.append(1 if improved else 0)
            if improved:
                parent, parent_h = gen_best_p, gen_best_h
                if gen_best_h > best_h: best_h, best_gen = gen_best_h, gen

            if len(success_window) >= ADAPT_WINDOW:
                sr = sum(success_window) / len(success_window)
                for k, (lo, hi) in PARAM_RANGES.items():
                    rng = hi - lo
                    if sr > SUCCESS_TARGET: sigmas[k] = min(sigmas[k] * ADAPT_UP, rng * SIGMA_FRAC_MAX)
                    else: sigmas[k] = max(sigmas[k] * ADAPT_DOWN, rng * SIGMA_FRAC_MIN)

            gen += 1
            if gen % 10 == 0:
                elapsed_min = (time.perf_counter() - t_start) / 60.0
                invalid_pct = 100.0 * n_invalid / max(1, n_evals)
                sigma_str = "  ".join(f"{k}={v*1000:.2f}mm" for k, v in sigmas.items())
                print(f"\n[gen {gen:5d}] best={best_h:5.1f}mm  parent={parent_h:5.1f}mm  evals={n_evals:6d}  "
                      f"invalid={invalid_pct:.0f}%  elapsed={elapsed_min:.1f}min\n"
                      f"  sigma: {sigma_str}\n  parent: {_params_str(parent)}")

    elapsed_min = (time.perf_counter() - t_start) / 60.0
    print("\n" + "="*68 + f"\nEVO complete: {gen} generations, {n_evals} evals, {elapsed_min:.1f} min\n"
          f"All-time best: {best_h:.1f} mm (found at gen {best_gen})\n"
          f"Best params:   {_params_str(parent)}\n\nTo view top results:\n"
          f"  python -c \"import pandas as pd; df=pd.read_csv(r'{CSV_PATH}'); "
          f"print(df.nlargest(20,'jump_height_mm'))\"\n"
          f"To replay best:  python {os.path.join(_HERE, 'replay_balanced.py')} --top 1")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="4-bar geometry optimizer with active balancing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_balanced.py --mode oat
  python optimize_balanced.py --mode grid --p1 L_femur --p2 Lc --levels 7
  python optimize_balanced.py --mode evo --hours 8
""")
    ap.add_argument('--mode', choices=['oat', 'grid', 'evo'], default='oat')
    ap.add_argument('--levels', type=int, default=10)
    ap.add_argument('--p1', default='L_femur')
    ap.add_argument('--p2', default='Lc')
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    if args.mode == 'oat':
        run_oat(n_levels=args.levels, n_workers=args.workers)
    elif args.mode == 'grid':
        run_grid(args.p1, args.p2, levels=args.levels, n_workers=args.workers)
    elif args.mode == 'evo':
        run_evo(hours=args.hours, max_iters=args.iters, seed=args.seed, n_workers=args.workers)

if __name__ == '__main__':
    # This is required for multiprocessing on Windows/macOS
    multiprocessing.freeze_support()
    main()