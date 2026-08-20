"""Reproduce the latest hardware jumps and tune the firmware jump sequence.

This is deliberately a simulation-only candidate generator.  It never writes
firmware parameters or communicates with a robot.  The reference pass uses the
launch parameters present in LOG0015; the search pass exercises the new nudge,
landing detector, and LQR handoff across both recorded starting conditions.

Run from ``simulation/mujoco``::

    python -m v4_twin_279mm_baseline.twin.tools.jump_study --samples 128
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Callable, Iterable

import mujoco
import numpy as np

from ...defaults import DEFAULT_PARAMS
from ...models.motor import motor_taper
from ...physics import alpha_to_hip_q, sim_q_to_firmware_hip
from ...scenarios.base import ScenarioConfig
from ...sim_loop import build_model_and_data, get_pitch_and_rate, init_sim, run
from ..params_control import PARAMS_BY_NAME


PACKAGE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
GUI_DIR = REPO_ROOT / "software" / "gui"
DEFAULT_LOG = (
    REPO_ROOT / "data" / "logs" / "runs"
    / "20260812T053231_282183Z_SD_LOG0015" / "LOG0015.WLOG"
)
DEFAULT_REPORT = PACKAGE_DIR / "jump_optimization_report.json"
DEFAULT_CANDIDATE = PACKAGE_DIR / "jump_optimization_candidate.json"
DEFAULT_PLOT = PACKAGE_DIR / "jump_optimization_comparison.png"

TRIGGER_S = 0.0
SIM_DURATION_S = 3.25

LAUNCH_NAMES = (
    "jump_effort", "jump_crouch_angle", "jump_crouch_speed",
    "jump_extend_angle", "jump_retract_speed", "jump_retract_angle",
    "jump_nudge_fwd_vel", "jump_nudge_fwd_dur",
)
HANDOFF_NAMES = (
    "jmp_handoff_kp_mul", "jmp_handoff_kr_mul", "jmp_handoff_kv_mul",
    "jmp_handoff_torque",
)
SEARCH_NAMES = LAUNCH_NAMES + HANDOFF_NAMES

# Bounds are intentionally no wider than the firmware schema and are centred
# on the two successful hardware launches.  The objective favours a modest,
# recoverable hop rather than maximum height.
SEARCH_BOUNDS = {
    "jump_effort": (0.70, 1.00),
    "jump_crouch_angle": (0.24, 0.42),
    "jump_crouch_speed": (0.70, 2.20),
    "jump_extend_angle": (1.10, 1.34),
    "jump_retract_speed": (0.90, 5.00),
    "jump_retract_angle": (0.55, 0.95),
    "jump_nudge_fwd_vel": (0.04, 0.30),
    "jump_nudge_fwd_dur": (0.04, 0.16),
    "jmp_handoff_kp_mul": (1.00, 2.75),
    "jmp_handoff_kr_mul": (1.00, 3.00),
    "jmp_handoff_kv_mul": (0.50, 1.75),
    # The matched ODrive plant is already current-limited by a 0.4 N.m input
    # command. Larger firmware limits are therefore unidentifiable/inert here.
    "jmp_handoff_torque": (0.40, 0.40),
}


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    log_number: int
    pitch_rad: float
    pitch_rate_rads: float
    roll_rad: float
    wheel_turns_s: float
    velocity_command_ms: float
    hip_firmware_rad: float
    phase_s: dict[str, float]
    inferred_landing_s: float


REFERENCE_CASES = (
    ReferenceCase(
        "recorded_jump_1", 1, -0.1017040312, -0.1796875,
        -0.0184779856, 0.4429370463, 0.1649999917, -0.4087507725,
        {"CROUCH": 0.0, "EXTEND": 0.146059, "RETRACT": 0.335824,
         "LEGACY_HOLD": 1.227823},
        0.560164,
    ),
    ReferenceCase(
        "recorded_jump_2", 2, -0.1404269487, -0.2109375,
        -0.0158663169, 0.1422767490, 0.0015000000, -0.4186689854,
        {"CROUCH": 0.0, "EXTEND": 0.170224, "RETRACT": 0.359984,
         "LEGACY_HOLD": 1.254147},
        0.570143,
    ),
)


class TraceRecorder:
    """Retain only the channels needed for jump scoring and plots."""

    FIELDS = (
        "t", "pitch", "pitch_rate", "roll", "wheel_vel", "v_target",
        "hip_q_avg", "tau_hip_L", "tau_hip_R", "tau_sym",
        "wheel_z_L", "wheel_z_R", "az_linear", "gx_imu", "gy_imu",
        "gz_imu", "mode", "landing_source", "airborne_seen", "jump_fault",
    )

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def __call__(self, tick: dict) -> None:
        self.rows.append({name: tick.get(name) for name in self.FIELDS})


def _absolute_attitude_init(case: ReferenceCase) -> Callable:
    """Create a MuJoCo initializer for a recorded jump-start state."""
    def initialise(model, data, params) -> None:
        s_root = int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")])
        d_root = int(model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")])
        cr, sr = math.cos(case.roll_rad / 2.0), math.sin(case.roll_rad / 2.0)
        cp, sp = math.cos(case.pitch_rad / 2.0), math.sin(case.pitch_rad / 2.0)
        # yaw = 0: quaternion in w, x, y, z order.
        data.qpos[s_root + 3:s_root + 7] = (cr * cp, sr * cp, cr * sp, -sr * sp)
        wheel_rad_s = case.wheel_turns_s * 2.0 * math.pi
        data.qvel[d_root] = (wheel_rad_s + case.pitch_rate_rads) * params.robot.wheel_r
        data.qvel[d_root + 4] = case.pitch_rate_rads
        for name in ("wheel_L", "wheel_R"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qvel[int(model.jnt_dofadr[joint_id])] = wheel_rad_s
        mujoco.mj_forward(model, data)
        wheel_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_L"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_R"),
        ]
        lowest = min(float(data.xpos[body_id][2]) for body_id in wheel_ids)
        data.qpos[s_root + 2] += params.robot.wheel_r - lowest
        mujoco.mj_forward(model, data)
    return initialise


def _scenario(case: ReferenceCase, params) -> ScenarioConfig:
    q_start = case.hip_firmware_rad + params.robot.Q_RET
    return ScenarioConfig(
        name=f"jump_study_{case.name}",
        display_name=case.name,
        duration=SIM_DURATION_S,
        active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
        hip_mode="jump",
        v_profile=lambda _t, value=case.velocity_command_ms: value,
        hip_profile=lambda _t, value=q_start: value,
        init_fn=_absolute_attitude_init(case),
        initial_hip_q=q_start,
        jump_time=TRIGGER_S,
    )


def _phase_times(rows: list[dict]) -> tuple[dict[str, float], float | None]:
    if not rows:
        return {}, None
    first_crouch = next((float(row["t"]) for row in rows
                         if row.get("mode") == "CROUCH"), None)
    if first_crouch is None:
        return {}, None
    entries: dict[str, float] = {}
    previous = None
    for row in rows:
        mode = str(row.get("mode", ""))
        if mode != previous and float(row["t"]) >= first_crouch:
            entries.setdefault(mode, float(row["t"]) - first_crouch)
            previous = mode
    return entries, first_crouch


def _true_contact(rows: list[dict], wheel_radius: float,
                  jump_start: float) -> tuple[float | None, float | None, float]:
    margin = 0.0025
    airborne = False
    liftoff_t = None
    touchdown_t = None
    peak_clearance = 0.0
    for row in rows:
        now = float(row["t"])
        if now < jump_start:
            continue
        left = float(row["wheel_z_L"])
        right = float(row["wheel_z_R"])
        both_clearance = min(left, right) - wheel_radius
        peak_clearance = max(peak_clearance, both_clearance)
        if not airborne and both_clearance > margin:
            airborne = True
            liftoff_t = now - jump_start
        elif airborne and min(left, right) <= wheel_radius + 0.001:
            touchdown_t = now - jump_start
            break
    return liftoff_t, touchdown_t, peak_clearance


def score_run(case: ReferenceCase, params, *, seed: int = 11,
              keep_trace: bool = False) -> dict:
    recorder = TraceRecorder()
    metrics = run(params, _scenario(case, params), callbacks=[recorder], rng_seed=seed)
    phase, start = _phase_times(recorder.rows)
    if start is None:
        return {"case": case.name, "objective": 1e6, "fell": True,
                "fault": "jump did not start", "phase_s": {}, "trace": []}
    liftoff, touchdown, clearance = _true_contact(
        recorder.rows, params.robot.wheel_r, start)
    landing = phase.get("LANDING")
    running = phase.get("BALANCE")
    handoff = phase.get("HANDOFF")
    window = [row for row in recorder.rows
              if start <= float(row["t"]) <= start + 2.25]
    pitch_deg = [math.degrees(float(row["pitch"])) for row in window]
    wheel_tps = [abs(float(row["wheel_vel"])) / (2.0 * math.pi) for row in window]
    hip_torque = [max(abs(float(row["tau_hip_L"])),
                      abs(float(row["tau_hip_R"]))) for row in window]
    fault = next((str(row.get("jump_fault")) for row in recorder.rows
                  if row.get("jump_fault")), "")
    landing_source = next((str(row.get("landing_source")) for row in recorder.rows
                           if row.get("landing_source")), "")
    result = {
        "case": case.name,
        "rng_seed": int(seed),
        "phase_s": {name: round(value, 6) for name, value in phase.items()},
        "liftoff_s": None if liftoff is None else round(liftoff, 6),
        "true_touchdown_s": None if touchdown is None else round(touchdown, 6),
        "landing_detected_s": None if landing is None else round(landing, 6),
        "landing_detection_error_s": (
            None if landing is None or touchdown is None else round(landing - touchdown, 6)),
        "landing_source": landing_source,
        "peak_wheel_clearance_m": round(clearance, 6),
        "peak_abs_pitch_deg": round(max((abs(value) for value in pitch_deg), default=0.0), 5),
        "min_pitch_deg": round(min(pitch_deg, default=0.0), 5),
        "max_pitch_deg": round(max(pitch_deg, default=0.0), 5),
        "peak_wheel_speed_turns_s": round(max(wheel_tps, default=0.0), 5),
        "peak_hip_torque_nm": round(max(hip_torque, default=0.0), 5),
        "handoff_started_s": None if handoff is None else round(handoff, 6),
        "running_s": None if running is None or running < 0.01 else round(running, 6),
        "fell": bool(metrics["fell"]),
        "fault": fault,
        "sim_metrics": metrics,
    }
    result["objective"] = round(_objective(result), 8)
    if keep_trace:
        result["trace"] = recorder.rows
    return result


def _objective(result: dict) -> float:
    """Risk-weighted cost: false landing/fall dominate height and speed."""
    cost = 0.0
    clearance = float(result["peak_wheel_clearance_m"])
    target_clearance = 0.025
    if result["fell"]:
        cost += 1200.0
    if result["fault"]:
        cost += 900.0
    if result["liftoff_s"] is None or result["true_touchdown_s"] is None:
        cost += 500.0 + 12000.0 * max(0.0, target_clearance - clearance)
    else:
        cost += 3500.0 * max(0.0, target_clearance - clearance) ** 2
        cost += 1500.0 * max(0.0, clearance - 0.065) ** 2
    error = result["landing_detection_error_s"]
    if error is None:
        cost += 450.0
    else:
        # A detector that precedes physical contact is unsafe; a slightly late
        # detector merely delays recovery authority.
        cost += 9000.0 * max(0.0, -float(error) - 0.004) ** 2
        cost += 1500.0 * max(0.0, float(error) - 0.025) ** 2
    if result["running_s"] is None:
        cost += 350.0
    else:
        running_s = float(result["running_s"])
        cost += 0.35 * running_s
        cost += 8.0 * max(0.0, running_s - 1.10) ** 2
    cost += 0.22 * max(0.0, float(result["peak_abs_pitch_deg"]) - 14.0) ** 2
    cost += 0.25 * max(0.0, float(result["peak_wheel_speed_turns_s"]) - 10.0) ** 2
    cost += 0.08 * float(result["peak_abs_pitch_deg"])
    return cost


def _aggregate(results: Iterable[dict]) -> float:
    values = [float(result["objective"]) for result in results]
    return float(np.mean(values) + 0.65 * max(values))


def _with_values(base, values: dict[str, float]):
    firmware = dict(base.firmware_params)
    firmware.update({name: float(value) for name, value in values.items()})
    return replace(base, firmware_params=tuple(sorted(firmware.items())))


def _actuator_scaled(params, scale: float):
    """Scale both fitted actuator conversions for plant-sensitivity checks."""
    wheel = replace(
        params.motors.wheel,
        odrive_torque_constant=params.motors.wheel.odrive_torque_constant / scale,
    )
    hip = replace(
        params.motors.hip,
        torque_scale_ret=params.motors.hip.torque_scale_ret * scale,
        torque_scale_ext=params.motors.hip.torque_scale_ext * scale,
    )
    return replace(params, motors=replace(params.motors, wheel=wheel, hip=hip))


def _reference_params() -> tuple[object, dict[str, float]]:
    """LOG0015 launch settings, with pre-feature nudge disabled."""
    values = {
        "jump_enable": 1.0,
        "jump_torque_max": 8.0,
        "jump_ramp_down": 0.174533,
        "jump_omega_max": 17.4533,
        "jump_hs_margin": 0.0872665,
        "jump_kp": 120.0,
        "jump_kd": 1.0,
        "jump_ext_kd": 0.1,
        "jump_ext_timeout": 1.0,
        "jump_effort": 1.0,
        "jump_crouch_angle": 0.349066,
        "jump_crouch_speed": 0.785398,
        "jump_extend_angle": 1.22173,
        "jump_retract_speed": 0.872665,
        "jump_retract_angle": 0.872665,
        "jump_torque_rate": 300.0,
        "jump_nudge_fwd_vel": 0.0,
        "jump_nudge_fwd_dur": 0.0,
        # Keep the detector/handoff defaults introduced after this log.
        "jump_land_timeout": 1.0,
        "jmp_handoff_timeout": 1.5,
    }
    return _with_values(DEFAULT_PARAMS, values), values


def _search_base(reference_values: dict[str, float]):
    values = dict(reference_values)
    values.update({
        "jump_ext_timeout": 0.45,
        "jump_land_timeout": 0.90,
        "jump_air_accel_z": -3.0,
        "jump_land_accel_z": 1.5,
        "jump_land_gyro_imp": 2.5,
        "jump_land_min_air": 0.16,
        "jmp_handoff_vel_lim": 10.0,
        "jmp_handoff_pitch": 0.065,
        "jmp_handoff_rate": 1.25,
        "jmp_handoff_hold_s": 0.12,
        "jmp_handoff_timeout": 1.35,
    })
    return _with_values(DEFAULT_PARAMS, values), values


def _sobol_points(count: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc
        exponent = int(math.ceil(math.log2(max(1, count))))
        return qmc.Sobol(len(SEARCH_NAMES), scramble=True, seed=seed).random_base2(exponent)[:count]
    except ImportError:
        return np.random.default_rng(seed).random((count, len(SEARCH_NAMES)))


def optimise(base, samples: int, seed: int) -> tuple[dict[str, float], list[dict], list[dict]]:
    points = _sobol_points(samples, seed)
    ranked: list[tuple[float, dict[str, float], list[dict]]] = []
    for index, point in enumerate(points):
        candidate = {
            name: SEARCH_BOUNDS[name][0]
                  + float(point[axis]) * (SEARCH_BOUNDS[name][1] - SEARCH_BOUNDS[name][0])
            for axis, name in enumerate(SEARCH_NAMES)
        }
        params = _with_values(base, candidate)
        results = [score_run(case, params, seed=seed + case.log_number)
                   for case in REFERENCE_CASES]
        ranked.append((_aggregate(results), candidate, results))
        if (index + 1) % 16 == 0 or index + 1 == samples:
            best = min(item[0] for item in ranked)
            print(f"search {index + 1:4d}/{samples}: best objective {best:.4f}", flush=True)
    ranked.sort(key=lambda item: item[0])

    # Re-test the best diverse candidates with four independent noise seeds.
    finalists: list[tuple[float, dict[str, float], list[dict]]] = []
    for _score, candidate, _results in ranked[:min(8, len(ranked))]:
        params = _with_values(base, candidate)
        results = [
            score_run(case, params, seed=noise_seed)
            for case in REFERENCE_CASES
            for noise_seed in (31 + case.log_number, 71 + case.log_number)
        ]
        for actuator_scale in (0.85, 1.15):
            variant = _actuator_scaled(params, actuator_scale)
            for case in REFERENCE_CASES:
                result = score_run(case, variant, seed=111 + case.log_number)
                result["plant_variant"] = f"actuator_scale_{actuator_scale:.2f}"
                results.append(result)
        finalists.append((_aggregate(results), candidate, results))
    finalists.sort(key=lambda item: item[0])
    _score, best_values, validation = finalists[0]
    best_values = refine_recovery_controls(base, best_values, seed + 1000)
    params = _with_values(base, best_values)
    validation = _validation_runs(params)
    selected = [score_run(case, _with_values(base, best_values),
                          seed=seed + case.log_number, keep_trace=True)
                for case in REFERENCE_CASES]
    return best_values, validation, selected


def _validation_runs(params) -> list[dict]:
    results = [
        score_run(case, params, seed=noise_seed)
        for case in REFERENCE_CASES
        for noise_seed in (31 + case.log_number, 71 + case.log_number)
    ]
    for actuator_scale in (0.85, 1.15):
        variant = _actuator_scaled(params, actuator_scale)
        for case in REFERENCE_CASES:
            result = score_run(case, variant, seed=111 + case.log_number)
            result["plant_variant"] = f"actuator_scale_{actuator_scale:.2f}"
            results.append(result)
    return results


def refine_recovery_controls(base, initial: dict[str, float], seed: int) -> dict[str, float]:
    """Focused low-dimensional search after the coarse launch search."""
    best = dict(initial)

    nudge_candidates = [
        dict(best, jump_nudge_fwd_vel=SEARCH_BOUNDS["jump_nudge_fwd_vel"][0]
             + float(point[0]) * (SEARCH_BOUNDS["jump_nudge_fwd_vel"][1]
                                  - SEARCH_BOUNDS["jump_nudge_fwd_vel"][0]),
             jump_nudge_fwd_dur=SEARCH_BOUNDS["jump_nudge_fwd_dur"][0]
             + float(point[1]) * (SEARCH_BOUNDS["jump_nudge_fwd_dur"][1]
                                  - SEARCH_BOUNDS["jump_nudge_fwd_dur"][0]))
        for point in _sobol_points(32, seed)[:, :2]
    ]
    nudge_candidates.append(dict(best))
    best = min(
        nudge_candidates,
        key=lambda values: _aggregate(
            score_run(case, _with_values(base, values), seed=seed + case.log_number)
            for case in REFERENCE_CASES),
    )

    handoff_names = (
        "jmp_handoff_kp_mul", "jmp_handoff_kr_mul", "jmp_handoff_kv_mul")
    handoff_candidates = []
    for point in _sobol_points(64, seed + 1)[:, :3]:
        values = dict(best)
        for axis, name in enumerate(handoff_names):
            lower, upper = SEARCH_BOUNDS[name]
            values[name] = lower + float(point[axis]) * (upper - lower)
        values["jmp_handoff_torque"] = 0.4
        handoff_candidates.append(values)
    handoff_candidates.extend((
        dict(best),
        dict(best, jmp_handoff_kp_mul=1.0, jmp_handoff_kr_mul=1.0,
             jmp_handoff_kv_mul=1.0, jmp_handoff_torque=0.4),
    ))
    best = min(
        handoff_candidates,
        key=lambda values: _aggregate(
            score_run(case, _with_values(base, values), seed=seed + 10 + case.log_number)
            for case in REFERENCE_CASES),
    )
    print("focused nudge/handoff refinement complete", flush=True)
    return best


def reaction_wheel_authority(params, pulse_s: float = 0.10) -> dict:
    """Measure airborne pitch authority from equal wheel-torque pulses.

    The result subtracts an otherwise identical zero-wheel-torque free-flight
    run, isolating the internal reaction-wheel motion from leg and gravity
    transients.  Torque inputs use the fitted ODrive command conversion.
    """
    def pulse(reported_torque_nm: float, speed_limit_tps: float) -> dict:
        model, data = build_model_and_data(params)
        init_sim(model, data, params, q_hip_init=params.robot.Q_NOM)
        joint = lambda name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = lambda name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
        root_joint = joint("root_free")
        root_qpos = int(model.jnt_qposadr[root_joint])
        root_dof = int(model.jnt_dofadr[root_joint])
        data.qpos[root_qpos + 2] += 0.30
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pitch_0, rate_0 = get_pitch_and_rate(data, body, root_dof + 4)

        wheel_pairs = [
            (int(model.jnt_dofadr[joint("wheel_L")]), actuator("wheel_act_L")),
            (int(model.jnt_dofadr[joint("wheel_R")]), actuator("wheel_act_R")),
        ]
        hip_pairs = [
            (int(model.jnt_qposadr[joint("hip_L")]),
             int(model.jnt_dofadr[joint("hip_L")]), actuator("hip_act_L")),
            (int(model.jnt_qposadr[joint("hip_R")]),
             int(model.jnt_dofadr[joint("hip_R")]), actuator("hip_act_R")),
        ]
        peak_wheel_tps = 0.0
        limit_reached_s = None
        while data.time < pulse_s:
            for qpos, dof, act in hip_pairs:
                reported_hip = float(np.clip(
                    120.0 * (params.robot.Q_NOM - data.qpos[qpos])
                    - data.qvel[dof], -8.0, 8.0))
                data.ctrl[act] = reported_hip * params.motors.hip.torque_scale(0.5)
            for dof, act in wheel_pairs:
                turns_s = abs(float(data.qvel[dof])) / (2.0 * math.pi)
                if turns_s < speed_limit_tps:
                    data.ctrl[act] = motor_taper(
                        reported_torque_nm * params.motors.wheel.command_torque_scale,
                        data.qvel[dof], params.battery.V_nom,
                        params.motors, params.battery)
                else:
                    data.ctrl[act] = 0.0
                    if limit_reached_s is None:
                        limit_reached_s = float(data.time)
                peak_wheel_tps = max(peak_wheel_tps, turns_s)
            mujoco.mj_step(model, data)
        pitch_1, rate_1 = get_pitch_and_rate(data, body, root_dof + 4)
        return {
            "pitch_change_deg": math.degrees(pitch_1 - pitch_0),
            "pitch_rate_change_deg_s": math.degrees(rate_1 - rate_0),
            "peak_wheel_speed_turns_s": peak_wheel_tps,
            "limit_reached_s": limit_reached_s,
        }

    baseline = pulse(0.0, math.inf)
    higher_torque = float(dict(params.firmware_params)["jmp_handoff_torque"])
    command_levels = ((0.4, 6.0), (0.4, 10.0), (higher_torque, 10.0))
    results = []
    for torque, speed_limit in command_levels:
        measured = pulse(torque, speed_limit)
        results.append({
            "reported_torque_per_wheel_nm": round(torque, 6),
            "physical_torque_per_wheel_nm_at_zero_speed": round(
                motor_taper(
                    torque * params.motors.wheel.command_torque_scale, 0.0,
                    params.battery.V_nom, params.motors, params.battery), 6),
            "wheel_speed_limit_turns_s": speed_limit,
            "pulse_s": pulse_s,
            "time_to_speed_limit_s": (
                None if measured["limit_reached_s"] is None
                else round(measured["limit_reached_s"], 5)),
            "body_pitch_change_deg_vs_zero": round(
                measured["pitch_change_deg"] - baseline["pitch_change_deg"], 5),
            "body_pitch_rate_change_deg_s_vs_zero": round(
                measured["pitch_rate_change_deg_s"]
                - baseline["pitch_rate_change_deg_s"], 5),
            "peak_wheel_speed_turns_s": round(
                measured["peak_wheel_speed_turns_s"], 5),
        })
    return {
        "method": "100 ms equal-torque pulse in free flight; zero-torque run subtracted",
        "plant_caveat": "uses the provisional fitted ODrive torque conversion",
        "results": results,
    }


def ablation_study(base, values: dict[str, float], seed: int) -> dict:
    variants = {
        "optimized": dict(values),
        "nudge_disabled": dict(values, jump_nudge_fwd_vel=0.0,
                               jump_nudge_fwd_dur=0.0),
        "normal_running_handoff": dict(
            values, jmp_handoff_kp_mul=1.0, jmp_handoff_kr_mul=1.0,
            jmp_handoff_kv_mul=1.0, jmp_handoff_torque=0.0),
        "old_100ms_landing_blanking": dict(values, jump_land_min_air=0.10),
    }
    return {
        name: [_jsonable(score_run(
            case, _with_values(base, variant), seed=seed + case.log_number))
               for case in REFERENCE_CASES]
        for name, variant in variants.items()
    }


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "trace"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_log_episodes(path: Path):
    if str(GUI_DIR) not in sys.path:
        sys.path.insert(0, str(GUI_DIR))
    from analysis.jump_analysis import analyze_jumps
    from analysis.wlog_metrics import decode_wlog
    decoded = decode_wlog(path)
    return decoded, analyze_jumps(decoded.t_s, decoded.fields)


def _plot(path: Path, log_path: Path, reference_results: list[dict],
          selected_results: list[dict]) -> None:
    import matplotlib.pyplot as plt

    decoded, episodes = _load_log_episodes(log_path)
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 10.0), sharex="col")
    fields = decoded.fields
    for column, (case, episode, reference, selected) in enumerate(
            zip(REFERENCE_CASES, episodes, reference_results, selected_results)):
        log_index = np.arange(episode.i0, episode.i1 + 1)
        log_t = decoded.t_s[log_index] - decoded.t_s[episode.i0]
        logged = (
            np.degrees(fields["pitch_rad"][log_index]),
            np.degrees(fields["pitch_rate_rads"][log_index]),
            fields["wheel_vel_avg"][log_index],
            fields["hip_l_pos_rad"][log_index],
        )
        labels = ("pitch [deg]", "pitch rate [deg/s]", "wheel [turn/s]", "hip [rad]")
        for axis, actual, label in zip(axes[:, column], logged, labels):
            axis.plot(log_t, actual, color="black", lw=1.0, label="hardware log")
            axis.set_ylabel(label)
            axis.grid(alpha=0.22)
        for result, style, name in (
                (reference, "--", "sim: log launch"),
                (selected, "-", "sim: optimized")):
            trace = result["trace"]
            start = next(float(row["t"]) for row in trace if row["mode"] == "CROUCH")
            sim_t = np.asarray([float(row["t"]) - start for row in trace])
            series = (
                np.degrees([float(row["pitch"]) for row in trace]),
                np.degrees([float(row["pitch_rate"]) for row in trace]),
                np.asarray([float(row["wheel_vel"]) for row in trace]) / (2.0 * math.pi),
                np.asarray([sim_q_to_firmware_hip(float(row["hip_q_avg"]), DEFAULT_PARAMS.robot)
                            for row in trace]),
            )
            for axis, values in zip(axes[:, column], series):
                axis.plot(sim_t, values, style, lw=1.15, label=name)
        axes[0, column].set_title(case.name.replace("_", " ").title())
        axes[-1, column].set_xlim(-0.05, 1.65)
        axes[-1, column].set_xlabel("time from jump command [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, 0.970))
    figure.suptitle("Hardware jump reproduction and optimized firmware-style simulation", y=0.997)
    figure.tight_layout(rect=(0, 0, 1, 0.935))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _candidate_payload(values: dict[str, float], validation: list[dict]) -> dict:
    return {
        "status": "SIMULATION_ONLY_DO_NOT_PUSH_UNREVIEWED",
        "source_log": str(DEFAULT_LOG.relative_to(REPO_ROOT)).replace("\\", "/"),
        "parameters": {name: round(float(values[name]), 7) for name in SEARCH_NAMES},
        "fixed_safety_parameters": {
            "jump_air_accel_z": -3.0,
            "jump_land_accel_z": 1.5,
            "jump_land_gyro_imp": 2.5,
            "jump_land_min_air": 0.16,
            "jump_land_timeout": 0.90,
            "jmp_handoff_vel_lim": 10.0,
            "jmp_handoff_pitch": 0.065,
            "jmp_handoff_rate": 1.25,
            "jmp_handoff_hold_s": 0.12,
            "jmp_handoff_timeout": 1.35,
        },
        "validation": _jsonable(validation),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args(argv)
    if args.samples < 8:
        parser.error("--samples must be at least 8")
    if not args.log.exists():
        parser.error(f"reference log not found: {args.log}")
    unknown = [name for name in SEARCH_NAMES if name not in PARAMS_BY_NAME]
    if unknown:
        raise RuntimeError(f"generated firmware parameter view is stale: {unknown}")

    reference_params, reference_values = _reference_params()
    reference_results = [score_run(case, reference_params,
                                   seed=args.seed + case.log_number,
                                   keep_trace=True)
                         for case in REFERENCE_CASES]
    print("reference reproduction:")
    for result in reference_results:
        print(json.dumps(_jsonable(result), indent=2), flush=True)

    search_base, fixed_values = _search_base(reference_values)
    best, validation, selected = optimise(search_base, args.samples, args.seed)
    candidate = _candidate_payload(best, validation)
    optimized_params = _with_values(search_base, best)
    report = {
        "method": {
            "sampler": "scrambled Sobol",
            "samples": args.samples,
            "seed": args.seed,
            "cases": [case.name for case in REFERENCE_CASES],
            "finalist_noise_seeds_per_case": 2,
            "objective": "risk-weighted modest-hop recovery; false landing and falls dominate",
        },
        "reference_log": str(args.log),
        "hardware_reference": [_jsonable(case.__dict__) for case in REFERENCE_CASES],
        "reference_reproduction": _jsonable(reference_results),
        "optimized_candidate": candidate,
        "selected_runs": _jsonable(selected),
        "ablations": ablation_study(search_base, best, args.seed + 300),
        "airborne_reaction_wheel_authority": reaction_wheel_authority(optimized_params),
        "fixed_search_values": fixed_values,
        "search_bounds": SEARCH_BOUNDS,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.candidate.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    _plot(args.plot, args.log, reference_results, selected)
    print(f"report: {args.report}")
    print(f"candidate: {args.candidate}")
    print(f"plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
