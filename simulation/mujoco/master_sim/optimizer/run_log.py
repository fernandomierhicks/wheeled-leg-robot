"""run_log.py — CSV run logging for the master_sim optimizer.

Flexible schema: any dict can be logged; first write creates the header,
subsequent writes append (extra keys are ignored, missing keys get "").

Usage:
    from master_sim.optimizer.run_log import log_run, get_best_run, load_all_runs
"""
from __future__ import annotations

import csv
import os
import pathlib

# ── Default logs directory ───────────────────────────────────────────────────
_PACKAGE = pathlib.Path(__file__).parent.parent.resolve()
LOGS_DIR = _PACKAGE / "logs"


def _ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Column schema (superset of all optimizers) ──────────────────────────────
CSV_COLS = [
    # Identity
    "run_id", "scenario", "label", "gen", "timestamp",
    # LQR cost weights
    "Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R",
    # Velocity PI gains
    "KP_V", "KI_V",
    # Yaw PI gains
    "KP_YAW", "KI_YAW",
    # Suspension / roll gains
    "LEG_K_S", "LEG_B_S", "LEG_K_ROLL", "LEG_D_ROLL",
    # Performance metrics
    "rms_pitch_deg", "max_pitch_deg", "wheel_travel_m", "wheel_liftoff_s",
    "vel_track_rms_ms", "settle_time_s", "survived_s",
    # Fitness breakdown
    "fitness_balance", "fitness_disturbance",
    "fitness_drive_slow", "fitness_drive_med", "fitness_obstacle",
    "fitness",
    # Run status
    "status", "fail_reason",
]


def get_scenario_csv_path(scenario: str) -> str:
    """Return scenario-specific CSV path.

    Example: scenario='s02_leg_height_gain_sched' -> 'logs/S_s02_leg_height_gain_sched.csv'
    """
    _ensure_logs_dir()
    safe_name = scenario.replace(" ", "_").replace("/", "_")
    return str(LOGS_DIR / f"S_{safe_name}.csv")


# ── ID management ────────────────────────────────────────────────────────────

def next_run_id(csv_path: str) -> int:
    """Return the next available run_id (1-based)."""
    if not os.path.exists(csv_path):
        return 1
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 1
        ids = []
        for r in rows:
            try:
                ids.append(int(r["run_id"]))
            except (ValueError, KeyError):
                pass
        return max(ids) + 1 if ids else 1
    except Exception:
        return 1


# ── Write ────────────────────────────────────────────────────────────────────

def log_run(row: dict, csv_path: str) -> int:
    """Append a run row to the CSV. Assigns run_id if not set. Returns run_id."""
    _ensure_logs_dir()
    if not row.get("run_id"):
        row["run_id"] = next_run_id(csv_path)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})
    return int(row["run_id"])


# ── Read ─────────────────────────────────────────────────────────────────────

def _read_all(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(csv_path, newline="", encoding="latin-1") as f:
            return list(csv.DictReader(f))


def load_all_runs(csv_path: str) -> list[dict]:
    """Return all rows as dicts."""
    return _read_all(csv_path)


def get_best_run(scenario: str, csv_path: str) -> dict | None:
    """Return the PASS row with lowest fitness for the given scenario."""
    rows = _read_all(csv_path)
    valid = [r for r in rows
             if r.get("scenario") == scenario and r.get("status") == "PASS"]
    if not valid:
        # Also try matching without scenario filter (single-scenario CSVs)
        valid = [r for r in rows if r.get("status") == "PASS"]
    if not valid:
        return None

    def _fit(r):
        try:
            return float(r["fitness"])
        except (ValueError, KeyError):
            return float("inf")
    return min(valid, key=_fit)


def load_best_params(scenario: str, csv_path: str, param_keys: list[str]) -> dict | None:
    """Extract best gains from CSV for seeding. Returns dict or None."""
    row = get_best_run(scenario, csv_path)
    if row is None:
        return None
    params = {}
    for k in param_keys:
        raw = row.get(k, "")
        try:
            params[k] = float(raw)
        except (ValueError, TypeError):
            return None
    return params


# ── Legacy CSV reader (for importing old logs) ──────────────────────────────

def read_legacy_csv(csv_path: str) -> list[dict]:
    """Read CSV from old LQR_Control_optimization or latency_sensitivity folders."""
    return _read_all(csv_path)


# ── CLI helper ───────────────────────────────────────────────────────────────

def list_runs(csv_path: str, scenario: str | None = None) -> None:
    """Print formatted run table to stdout."""
    rows = _read_all(csv_path)
    if scenario:
        rows = [r for r in rows if r.get("scenario") == scenario]

    if not rows:
        print("No runs found.")
        return

    header = (f"{'ID':>5}  {'Label':<28}  {'Status':<6}  "
              f"{'Fitness':>8}  {'RMS pitch':>9}  {'Surv':>5}")
    print(header)
    print("-" * len(header))
    for r in rows:
        fit = r.get("fitness", "")
        rms = r.get("rms_pitch_deg", "")
        sur = r.get("survived_s", "")
        try:
            fit = f"{float(fit):8.4f}"
        except (ValueError, TypeError):
            fit = f"{'':>8}"
        try:
            rms = f"{float(rms):9.3f}"
        except (ValueError, TypeError):
            rms = f"{'':>9}"
        try:
            sur = f"{float(sur):5.1f}"
        except (ValueError, TypeError):
            sur = f"{'':>5}"
        print(f"{r.get('run_id', ''):>5}  {r.get('label', ''):<28}  "
              f"{r.get('status', ''):<6}  {fit}  {rms}  {sur}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m master_sim.optimizer.run_log <csv_path> [scenario]")
        sys.exit(1)
    p = sys.argv[1]
    s = sys.argv[2] if len(sys.argv) > 2 else None
    list_runs(p, s)
    best = get_best_run(s or "", p)
    if best:
        print(f"\nBest: id={best.get('run_id', '?')}  "
              f"fitness={best.get('fitness', '?')}  "
              f"label={best.get('label', '?')}")
