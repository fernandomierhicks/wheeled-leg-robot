"""pipeline.py — S1→S8 automated tuning pipeline.

Runs the full 8-step gain chain sequentially:
  S1: LQR on s01_lqr_pitch_step            (seed: defaults)
  S2: LQR on s02_leg_height_gain_sched     (seed: S1 best)  → baseline
  S3: VelPI on s03_vel_pi_disturbance       (seed: defaults)
  S4: VelPI on s04_vel_pi_staircase         (seed: S3 best)
  S5: VelPI on s05_vel_pi_leg_cycling       (seed: S4 best)  → baseline
  S6: YawPI on s06_yaw_pi_turn              (seed: defaults)
  S7: YawPI on s07_drive_turn               (seed: S6 best)  → baseline
  S8: Susp  on s08_terrain_compliance       (seed: defaults)  → baseline

After each baseline step, gains are written to logs/baseline_gains.json.

Usage:
    python -m master_sim.optimizer.pipeline
    python -m master_sim.optimizer.pipeline --hours 2
    python -m master_sim.optimizer.pipeline --start S3 --end S5
    python -m master_sim.optimizer.pipeline --hours 0.02  # smoke test
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

_PACKAGE = pathlib.Path(__file__).parent.parent.resolve()
LOGS_DIR = _PACKAGE / "logs"
BASELINE_JSON = LOGS_DIR / "baseline_gains.json"


# ── Pipeline step definitions ────────────────────────────────────────────────
# (step_id, optimizer_module, scenario, param_keys, seed_from_step, is_baseline)

STEPS = [
    ("S1", "master_sim.optimizer.optimize_lqr",
     "s01_lqr_pitch_step",
     ["Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R"],
     None, False),

    ("S2", "master_sim.optimizer.optimize_lqr",
     "s02_leg_height_gain_sched",
     ["Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R"],
     "S1", True),

    ("S3", "master_sim.optimizer.optimize_vel_pi",
     "s03_vel_pi_disturbance",
     ["KP_V", "KI_V"],
     None, False),

    ("S4", "master_sim.optimizer.optimize_vel_pi",
     "s04_vel_pi_staircase",
     ["KP_V", "KI_V"],
     "S3", False),

    ("S5", "master_sim.optimizer.optimize_vel_pi",
     "s05_vel_pi_leg_cycling",
     ["KP_V", "KI_V"],
     "S4", True),

    ("S6", "master_sim.optimizer.optimize_yaw_pi",
     "s06_yaw_pi_turn",
     ["KP_YAW", "KI_YAW"],
     None, False),

    ("S7", "master_sim.optimizer.optimize_yaw_pi",
     "s07_drive_turn",
     ["KP_YAW", "KI_YAW"],
     "S6", True),

    ("S8", "master_sim.optimizer.optimize_suspension",
     "s08_terrain_compliance",
     ["LEG_K_S", "LEG_B_S", "LEG_K_ROLL", "LEG_D_ROLL"],
     None, True),
]

# Map from search-space param key to defaults.py dataclass field path
# Used to seed chain-starting steps from current defaults
_DEFAULTS_MAP = {
    "Q_PITCH":      ("gains", "lqr", "Q_pitch"),
    "Q_PITCH_RATE": ("gains", "lqr", "Q_pitch_rate"),
    "Q_VEL":        ("gains", "lqr", "Q_vel"),
    "R":            ("gains", "lqr", "R"),
    "KP_V":         ("gains", "velocity_pi", "Kp"),
    "KI_V":         ("gains", "velocity_pi", "Ki"),
    "KP_YAW":       ("gains", "yaw_pi", "Kp"),
    "KI_YAW":       ("gains", "yaw_pi", "Ki"),
    "LEG_K_S":      ("gains", "suspension", "K_s"),
    "LEG_B_S":      ("gains", "suspension", "B_s"),
    "LEG_K_ROLL":   ("gains", "suspension", "K_roll"),
    "LEG_D_ROLL":   ("gains", "suspension", "D_roll"),
}


# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_FILE: pathlib.Path | None = None


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _banner(msg: str):
    bar = "=" * 72
    _log(bar)
    _log(f"  {msg}")
    _log(bar)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_defaults_seed(param_keys: list[str]) -> dict:
    """Read current default gains from DEFAULT_PARAMS."""
    from master_sim.defaults import DEFAULT_PARAMS
    p = DEFAULT_PARAMS
    seed = {}
    for k in param_keys:
        path = _DEFAULTS_MAP[k]
        obj = p
        for attr in path:
            obj = getattr(obj, attr)
        seed[k] = obj
    return seed


def _read_best_from_csv(scenario: str, param_keys: list[str]) -> dict | None:
    """Read best gains from a scenario's CSV."""
    from master_sim.optimizer.run_log import get_scenario_csv_path, load_best_params
    csv_path = get_scenario_csv_path(scenario)
    return load_best_params(scenario, csv_path, param_keys)


def _format_seed(gains: dict) -> str:
    return ",".join(f"{k}={v:.8g}" for k, v in gains.items())


def _save_baseline(step_results: dict):
    """Write all baseline gains to logs/baseline_gains.json."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with BASELINE_JSON.open("w", encoding="utf-8") as f:
        json.dump(step_results, f, indent=2)
    _log(f"  Wrote baseline gains to {BASELINE_JSON.name}")


# ── Step runner ──────────────────────────────────────────────────────────────

def _run_step(step_id: str, optimizer_module: str, scenario: str,
              param_keys: list[str], seed_gains: dict | None,
              hours: float, workers: int | None,
              patience: int, tol: float, fresh: bool) -> dict | None:
    """Run one optimizer step as a subprocess. Returns best gains or None."""
    from master_sim.optimizer.run_log import get_scenario_csv_path

    csv_path = pathlib.Path(get_scenario_csv_path(scenario))

    # --fresh: delete existing CSV
    if fresh and csv_path.exists():
        csv_path.unlink()
        _log(f"  Deleted {csv_path.name} (--fresh)")

    # Skip if existing PASS result (resume behaviour)
    if not fresh:
        existing = _read_best_from_csv(scenario, param_keys)
        if existing is not None:
            _log(f"  Skipping {step_id}: existing PASS in {csv_path.name}")
            _log(f"  Best: {_format_seed(existing)}")
            return existing

    # Build subprocess command
    cmd = [
        sys.executable, "-m", optimizer_module,
        "--scenario", scenario,
        "--hours", str(hours),
        "--patience", str(patience),
        "--tol", str(tol),
    ]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    if seed_gains is not None:
        cmd += ["--seed-gains", _format_seed(seed_gains)]

    _log(f"  CMD: {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(_PACKAGE / ".."))
    elapsed = (time.perf_counter() - t0) / 60.0

    if result.returncode != 0:
        _log(f"  ERROR: optimizer exited with code {result.returncode}")
        return None

    best = _read_best_from_csv(scenario, param_keys)
    if best is None:
        _log(f"  ERROR: no PASS result after optimization")
        return None

    _log(f"  {step_id} done in {elapsed:.1f} min — best: {_format_seed(best)}")
    return best


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    global _LOG_FILE

    ap = argparse.ArgumentParser(
        description="S1→S8 automated tuning pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m master_sim.optimizer.pipeline                  # all 8, 1h each
  python -m master_sim.optimizer.pipeline --hours 2        # 2h per step
  python -m master_sim.optimizer.pipeline --hours 0.02     # smoke test
  python -m master_sim.optimizer.pipeline --start S3 --end S5
  python -m master_sim.optimizer.pipeline --fresh
""")
    ap.add_argument("--hours",    type=float, default=1.0)
    ap.add_argument("--workers",  type=int,   default=None)
    ap.add_argument("--patience", type=int,   default=200)
    ap.add_argument("--tol",      type=float, default=1e-4)
    ap.add_argument("--start",    type=str,   default=None)
    ap.add_argument("--end",      type=str,   default=None)
    ap.add_argument("--fresh",    action="store_true")
    args = ap.parse_args()

    # Set up log file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = LOGS_DIR / f"pipeline_{ts_str}.log"

    # Filter steps
    step_ids = [s[0] for s in STEPS]
    start_idx = step_ids.index(args.start) if args.start else 0
    end_idx = step_ids.index(args.end) + 1 if args.end else len(STEPS)
    active_steps = STEPS[start_idx:end_idx]

    _banner(f"S1->S8 Pipeline  |  {len(active_steps)} steps  |  {args.hours:.2f}h/step  |  {ts_str}")
    _log(f"  Log: {_LOG_FILE}")
    _log(f"  Steps: {[s[0] for s in active_steps]}")

    step_results: dict[str, dict] = {}

    # Pre-load existing results for steps before start (needed for seeding)
    for step_id, _, scenario, param_keys, _, _ in STEPS[:start_idx]:
        existing = _read_best_from_csv(scenario, param_keys)
        if existing:
            step_results[step_id] = existing

    pipeline_t0 = time.perf_counter()
    failed_steps = []

    for i, (step_id, opt_module, scenario, param_keys, seed_from, is_baseline) in enumerate(active_steps):
        _banner(f"[{i + 1}/{len(active_steps)}] {step_id} — {scenario}")

        # Resolve seed gains
        seed_gains = None
        if seed_from is not None:
            seed_gains = step_results.get(seed_from)
            if seed_gains:
                _log(f"  Seeding from {seed_from}: {_format_seed(seed_gains)}")
            else:
                _log(f"  WARNING: {seed_from} has no result — falling back to defaults")

        if seed_gains is None:
            seed_gains = _read_defaults_seed(param_keys)
            _log(f"  Seeding from defaults: {_format_seed(seed_gains)}")

        best = _run_step(
            step_id=step_id,
            optimizer_module=opt_module,
            scenario=scenario,
            param_keys=param_keys,
            seed_gains=seed_gains,
            hours=args.hours,
            workers=args.workers,
            patience=args.patience,
            tol=args.tol,
            fresh=args.fresh,
        )

        if best is None:
            _log(f"  FAILED: {step_id}")
            failed_steps.append(step_id)
            continue

        step_results[step_id] = best

        if is_baseline:
            _log(f"  [BASELINE] {step_id}: {_format_seed(best)}")
            _save_baseline(step_results)

    # Final summary
    total_min = (time.perf_counter() - pipeline_t0) / 60.0
    _banner(f"Pipeline complete — {total_min:.0f} min total")

    for step_id, _, scenario, param_keys, _, is_baseline in STEPS:
        if step_id not in step_results:
            continue
        tag = " [BASELINE]" if is_baseline else ""
        _log(f"  {step_id}{tag}: {_format_seed(step_results[step_id])}")

    if failed_steps:
        _log(f"\n  FAILED: {failed_steps}")
    else:
        _log("\n  All steps completed successfully.")


if __name__ == "__main__":
    main()
