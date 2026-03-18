"""run_log.py — CSV run logging for LQR_Control_optimization.

Schema: PD balance controller gains + performance metrics.
Supports: log, load, list, overwrite, and query by scenario / fitness.

Usage:
    from run_log import log_run, load_run, list_runs, overwrite_run, get_best_run
"""
import csv
import os

# ── File location ───────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "results.csv")

# ── Schema ──────────────────────────────────────────────────────────────────
CSV_COLS = [
    # Identity
    "run_id", "scenario", "label", "timestamp",
    # LQR cost weights (optimizer search space)
    "Q_PITCH", "Q_PITCH_RATE", "Q_VEL", "R",
    # Velocity PI gains (optimizer search space — Phase 2+)
    "KP_V", "KI_V",
    # Performance metrics (lower is better except survived_s)
    "rms_pitch_deg", "max_pitch_deg", "wheel_travel_m", "wheel_liftoff_s",
    "vel_track_rms_ms",   # RMS velocity tracking error [m/s] (drive scenarios)
    "settle_time_s", "survived_s",
    # Per-scenario fitness breakdown
    "fitness_balance", "fitness_disturbance",
    "fitness_drive_slow", "fitness_drive_med", "fitness_obstacle",
    "fitness",
    # Run status
    "status", "fail_reason",
]


# ---------------------------------------------------------------------------
# ID management
# ---------------------------------------------------------------------------
def next_run_id(csv_path: str = CSV_PATH) -> int:
    """Return the next available run_id (1-based, never reuses)."""
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


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def log_run(row: dict, csv_path: str = CSV_PATH) -> int:
    """Append a run row to the CSV.  Assigns run_id if not already set.

    Returns the run_id that was written.
    """
    if not row.get("run_id"):
        row["run_id"] = next_run_id(csv_path)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})
    return int(row["run_id"])


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def _read_all(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(csv_path, newline="", encoding="latin-1") as f:
            return list(csv.DictReader(f))


def load_all_runs(csv_path: str = CSV_PATH) -> list[dict]:
    """Return all rows as a list of dicts."""
    return _read_all(csv_path)


def load_run(run_id: int, csv_path: str = CSV_PATH) -> dict:
    """Load a single row by run_id.  Raises ValueError if not found."""
    rows = _read_all(csv_path)
    matches = [r for r in rows if str(r.get("run_id")) == str(run_id)]
    if not matches:
        raise ValueError(f"run_id={run_id} not found in {csv_path}")
    return matches[0]


def get_best_run(scenario: str = "balance", csv_path: str = CSV_PATH) -> dict | None:
    """Return the PASS row with the lowest fitness for the given scenario."""
    rows = _read_all(csv_path)
    valid = [r for r in rows
             if r.get("scenario") == scenario and r.get("status") == "PASS"]
    if not valid:
        return None
    def _fit(r):
        try:
            return float(r["fitness"])
        except (ValueError, KeyError):
            return float("inf")
    return min(valid, key=_fit)


def get_runs_by_scenario(scenario: str, csv_path: str = CSV_PATH) -> list[dict]:
    """Return all rows for a given scenario, ordered by run_id."""
    rows = _read_all(csv_path)
    return [r for r in rows if r.get("scenario") == scenario]


# ---------------------------------------------------------------------------
# Overwrite (resimulate a run)
# ---------------------------------------------------------------------------
def overwrite_run(run_id: int, new_row: dict, csv_path: str = CSV_PATH) -> None:
    """Replace the row with matching run_id in-place.

    Preserves all other rows.  new_row['run_id'] is forced to run_id.
    Raises ValueError if run_id not found.
    """
    rows = _read_all(csv_path)
    found = False
    for i, r in enumerate(rows):
        if str(r.get("run_id")) == str(run_id):
            new_row["run_id"] = run_id
            rows[i] = {k: new_row.get(k, "") for k in CSV_COLS}
            found = True
            break
    if not found:
        raise ValueError(f"run_id={run_id} not found — cannot overwrite")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------
def list_runs(scenario: str = None, csv_path: str = CSV_PATH) -> None:
    """Print a formatted table of runs to stdout."""
    rows = _read_all(csv_path)
    if scenario:
        rows = [r for r in rows if r.get("scenario") == scenario]

    if not rows:
        print("No runs found.")
        return

    header = (f"{'ID':>5}  {'Scenario':<10}  {'Label':<22}  {'Status':<6}  "
              f"{'Fitness':>8}  {'RMS°':>6}  {'Surv s':>6}  Timestamp")
    print(header)
    print("-" * len(header))
    for r in rows:
        fit = r.get("fitness", "")
        rms = r.get("rms_pitch_deg", "")
        sur = r.get("survived_s", "")
        try:
            fit = f"{float(fit):8.3f}"
        except (ValueError, TypeError):
            fit = f"{'':>8}"
        try:
            rms = f"{float(rms):6.2f}"
        except (ValueError, TypeError):
            rms = f"{'':>6}"
        try:
            sur = f"{float(sur):6.2f}"
        except (ValueError, TypeError):
            sur = f"{'':>6}"
        print(f"{r.get('run_id',''):>5}  {r.get('scenario',''):<10}  "
              f"{r.get('label',''):<22}  {r.get('status',''):<6}  "
              f"{fit}  {rms}  {sur}  {r.get('timestamp','')}")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    scenario_filter = sys.argv[1] if len(sys.argv) > 1 else None
    list_runs(scenario=scenario_filter)
    best = get_best_run(scenario=scenario_filter or "balance")
    if best:
        print(f"\nBest run: id={best['run_id']}  fitness={best.get('fitness','')}  "
              f"rms={best.get('rms_pitch_deg','')}°  label={best.get('label','')}")
