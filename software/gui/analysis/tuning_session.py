"""tuning_session.py — Phase-1 hardware tuning loop session state (tuning.md §5).

Pure Python, no Qt/hardware dependency. Claude calls this directly (Python,
not a CLI) during a live tuning session — there's no §2-3 remote-control loop
yet to chain a CLI step onto, so this is a library, not a script.

Owns two things per tuning-session directory (normally under `data/logs/runs/`):
  - `session.json` — resumable state: current stage, current best gains +
    fitness, current step size, trial count since last improvement.
  - `trials.csv`   — append-only, one row per hardware run (mirrors the sim's
    `logs/S1_*.csv` convention via a fixed column schema + DictWriter, see
    simulation/mujoco/master_sim/optimizer/run_log.py).

And the adaptive-step search itself: a serialized (one hardware trial at a
time — physical trials can't be parallelized) single-parent variant of the
sim's (1+lambda)-ES 1/5-success-rule (optimizer/es_engine.py), exactly as
tuning.md §5 specifies: perturb the stage's gain pair together as a scaled
pair from the current best (±15% initial step); trial passes safety AND
improves fitness -> adopt as new best, grow step x1.2; trial fails safety OR
is worse -> discard, shrink step x0.8.

Note: "adopt as new best" here only updates this module's bookkeeping. Actually
persisting a stage's tuned gains to the firmware (PARAM_FLAG_PERSISTENT
param_set) is §2-3's job, not built yet — see tuning.md Status.
"""

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("lqr", "vel_pi", "yaw_pi")
_NEXT_STAGE = {"lqr": "vel_pi", "vel_pi": "yaw_pi", "yaw_pi": None}

# Gain pair + firmware bounds per stage (firmware/robot_teensy/teensy/lib/
# ParamRegistry/param_registry.cpp) — seed = current firmware default, so an
# unstarted session begins exactly where the firmware already sits.
STAGE_GAINS = {
    "lqr":    {"names": ("lqr_k_pitch_ret", "lqr_k_rate_ret"),
               "seed": (-0.3, -0.1), "bounds": ((-20.0, 0.0), (-5.0, 0.0))},
    "vel_pi": {"names": ("vel_pi_kp", "vel_pi_ki"),
               "seed": (0.2, 0.1), "bounds": ((0.0, 5.0), (0.0, 5.0))},
    "yaw_pi": {"names": ("yaw_pi_kp", "yaw_pi_ki"),
               "seed": (0.2, 0.1), "bounds": ((0.0, 5.0), (0.0, 5.0))},
}

STEP_INIT = 0.15     # +/-15% starting perturbation
STEP_GROW = 1.2
STEP_SHRINK = 0.8
STEP_MIN = 0.01
STEP_MAX = 1.0
TRIAL_BUDGET = 10           # stop the stage after this many trials...
STAGNATION_LIMIT = 4        # ...or this many consecutive non-improving trials, whichever first


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── session.json ───────────────────────────────────────────────────────────

@dataclass
class SessionState:
    stage: str | None                    # "lqr"/"vel_pi"/"yaw_pi", or None once all 3 converge
    best_gains: dict                     # {gain_name: value}
    best_fitness: float
    step_size: float
    trial_in_stage: int
    trials_since_improvement: int
    stage_complete: dict                 # {"lqr": bool, "vel_pi": bool, "yaw_pi": bool}
    updated_at: str = field(default_factory=_now_iso)


def default_session(stage: str = "lqr") -> SessionState:
    names = STAGE_GAINS[stage]["names"]
    seed = STAGE_GAINS[stage]["seed"]
    return SessionState(
        stage=stage,
        best_gains=dict(zip(names, seed)),
        best_fitness=float("inf"),
        step_size=STEP_INIT,
        trial_in_stage=0,
        trials_since_improvement=0,
        stage_complete={s: False for s in STAGES},
    )


def load_session(path) -> SessionState:
    """Read session.json, or seed a fresh Phase-1 session (stage="lqr") if
    the file doesn't exist yet — this is the "pick up where things left off"
    entry point tuning.md §5 calls for at the start of each conversation."""
    path = Path(path)
    if not path.exists():
        return default_session()
    data = json.loads(path.read_text())
    return SessionState(**data)


def save_session(path, state: SessionState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = _now_iso()
    path.write_text(json.dumps(asdict(state), indent=2))


# ── Adaptive-step search ─────────────────────────────────────────────────────

def propose_next(state: SessionState, rng: random.Random | None = None) -> dict:
    """One candidate: the current best gain pair scaled together by a single
    random factor in [-step_size, +step_size] (a *coupled* perturbation, not
    two independent draws — "perturb the stage's gain pair together as a
    scaled pair", tuning.md §5), then clamped to each gain's firmware bounds."""
    if state.stage is None:
        raise ValueError("session has no active stage — all 3 stages already converged")
    rng = rng or random.Random()
    names = STAGE_GAINS[state.stage]["names"]
    bounds = STAGE_GAINS[state.stage]["bounds"]
    delta = rng.uniform(-state.step_size, state.step_size)
    candidate = {}
    for name, (lo, hi) in zip(names, bounds):
        val = state.best_gains[name] * (1.0 + delta)
        candidate[name] = min(max(val, lo), hi)
    return candidate


def record_trial(state: SessionState, gains: dict, fitness: float, safety_ok: bool) -> bool:
    """Update state in place per the 1/5-success-rule: accept (new best, grow
    step) if safety_ok and fitness improves on the current best; otherwise
    discard and shrink step. Returns whether this trial was accepted."""
    accepted = safety_ok and fitness < state.best_fitness
    if accepted:
        state.best_gains = dict(gains)
        state.best_fitness = fitness
        state.step_size = min(state.step_size * STEP_GROW, STEP_MAX)
        state.trials_since_improvement = 0
    else:
        state.step_size = max(state.step_size * STEP_SHRINK, STEP_MIN)
        state.trials_since_improvement += 1
    state.trial_in_stage += 1
    return accepted


def should_advance_stage(state: SessionState) -> bool:
    return (state.trial_in_stage >= TRIAL_BUDGET
            or state.trials_since_improvement >= STAGNATION_LIMIT)


def advance_stage(state: SessionState) -> SessionState:
    """Move to the next stage, seeded from *this* stage's tuned best (tuning.md
    §5: "Each stage's tuned result immediately becomes the new persisted
    default for that stage's params so the next stage's test starts from an
    already-improved base"). Actually writing that default to the firmware
    (param_set + PARAM_FLAG_PERSISTENT) is §2-3's job, not this module's."""
    if state.stage is None:
        return state
    state.stage_complete[state.stage] = True
    next_stage = _NEXT_STAGE[state.stage]
    state.stage = next_stage
    if next_stage is not None:
        names = STAGE_GAINS[next_stage]["names"]
        seed = STAGE_GAINS[next_stage]["seed"]
        state.best_gains = dict(zip(names, seed))
        state.best_fitness = float("inf")
    state.step_size = STEP_INIT
    state.trial_in_stage = 0
    state.trials_since_improvement = 0
    return state


# ── trials.csv ────────────────────────────────────────────────────────────

# Fixed, stage-agnostic schema (extra keys ignored, missing keys -> "" on
# write) — same flexible-but-stable-header pattern as run_log.py's CSV_COLS.
TRIALS_CSV_FIELDS = [
    "timestamp", "stage", "trial_in_stage", "trials_since_improvement", "step_size",
    "gain1_name", "gain1_value", "gain2_name", "gain2_value",
    "wlog_path", "accepted", "fitness", "fitness_breakdown",
    "safety_ok", "safety_reasons",
    "duration_s", "n_samples",
    "rms_pitch_deg", "ise_pitch", "ise_pitch_rate", "rms_pitch_rate_dps", "max_pitch_deg",
    "settle_time_s", "zero_cross_hz", "dominant_freq_hz",
    "vel_track_rms_ms", "yaw_track_rms_rads",
    "health_vel_pi_sat_frac", "health_yaw_pi_sat_frac",
    "health_wm_l_vel_limited_frac", "health_wm_r_vel_limited_frac", "health_loop_overrun_frac",
    "max_hip_l_torque_nm", "max_hip_r_torque_nm", "max_wm_l_vel_turns_s", "max_wm_r_vel_turns_s",
    "fault_fired", "faults_seen",
    "gain_sched_alpha_min", "gain_sched_alpha_max", "gain_sched_alpha_mean",
]


def trial_row(stage: str, gains: dict, eval_result: dict, accepted: bool, step_size: float,
              trial_in_stage: int, trials_since_improvement: int, wlog_path: str = "",
              timestamp: str | None = None) -> dict:
    """Flatten one analysis.wlog_metrics.evaluate() result (see tuning.md §4)
    plus this trial's search-state bookkeeping into one trials.csv row."""
    names = list(gains.keys())
    m = eval_result.get("metrics", {})
    alpha = m.get("gain_sched_alpha_range") or {}
    health = m.get("health_fractions", {})
    return {
        "timestamp": timestamp or _now_iso(),
        "stage": stage,
        "trial_in_stage": trial_in_stage,
        "trials_since_improvement": trials_since_improvement,
        "step_size": round(step_size, 5),
        "gain1_name": names[0], "gain1_value": gains[names[0]],
        "gain2_name": names[1], "gain2_value": gains[names[1]],
        "wlog_path": wlog_path,
        "accepted": accepted,
        "fitness": eval_result.get("fitness"),
        "fitness_breakdown": json.dumps(eval_result.get("fitness_breakdown", {})),
        "safety_ok": eval_result.get("safety_ok"),
        "safety_reasons": json.dumps(eval_result.get("safety_reasons", [])),
        "duration_s": m.get("duration_s"),
        "n_samples": m.get("n_samples"),
        "rms_pitch_deg": m.get("rms_pitch_deg"),
        "ise_pitch": m.get("ise_pitch"),
        "ise_pitch_rate": m.get("ise_pitch_rate"),
        "rms_pitch_rate_dps": m.get("rms_pitch_rate_dps"),
        "max_pitch_deg": m.get("max_pitch_deg"),
        "settle_time_s": m.get("settle_time_s"),
        "zero_cross_hz": m.get("zero_cross_hz"),
        "dominant_freq_hz": m.get("dominant_freq_hz"),
        "vel_track_rms_ms": m.get("vel_track_rms_ms"),
        "yaw_track_rms_rads": m.get("yaw_track_rms_rads"),
        "health_vel_pi_sat_frac": health.get("vel_pi_sat_frac"),
        "health_yaw_pi_sat_frac": health.get("yaw_pi_sat_frac"),
        "health_wm_l_vel_limited_frac": health.get("wm_l_vel_limited_frac"),
        "health_wm_r_vel_limited_frac": health.get("wm_r_vel_limited_frac"),
        "health_loop_overrun_frac": health.get("loop_overrun_frac"),
        "max_hip_l_torque_nm": m.get("max_hip_l_torque_nm"),
        "max_hip_r_torque_nm": m.get("max_hip_r_torque_nm"),
        "max_wm_l_vel_turns_s": m.get("max_wm_l_vel_turns_s"),
        "max_wm_r_vel_turns_s": m.get("max_wm_r_vel_turns_s"),
        "fault_fired": m.get("fault_fired"),
        "faults_seen": json.dumps(m.get("faults_seen", [])),
        "gain_sched_alpha_min": alpha.get("min"),
        "gain_sched_alpha_max": alpha.get("max"),
        "gain_sched_alpha_mean": alpha.get("mean"),
    }


def append_trial(csv_path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIALS_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRIALS_CSV_FIELDS})
