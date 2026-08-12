"""Robust staged tuning of the real balance controllers in matched MuJoCo.

This optimizer deliberately does not touch plant parameters, trim, watchdogs,
torque limits, or command limits.  It starts from the latest GUI export, runs
the firmware-equivalent controller, and searches controller gains against a
deterministic ensemble around the provisional matched plant.

The output is a complete GUI-compatible parameter snapshot, but it is never
written over ``Default gains.json`` and is never sent to the robot.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import time

import mujoco

from .es_engine import ESConfig, ESOptimizer
from .search_space import ParamSpec, SearchSpace
from ..defaults import DEFAULT_PARAMS
from ..physics import alpha_to_hip_q
from ..robot_match import CONTROLLER_SNAPSHOT_ENV, control_snapshot_sha256
from ..scenarios.base import ScenarioConfig
from ..sim_loop import run
from ..twin.params_control import PARAMS_BY_NAME, validate_values


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT / "software" / "gui" / "parameter_exports"
    / "Robust_balance_candidate_2026-08-11.json"
)
DEFAULT_REPORT = PACKAGE_DIR / "gain_optimization_report.json"
DEFAULT_BASELINE_SNAPSHOT = PACKAGE_DIR / "gain_optimization_baseline.json"


@dataclass(frozen=True)
class PlantVariation:
    name: str
    wheel_torque_factor: float = 1.0
    hip_torque_factor: float = 1.0
    sensor_delay_s: float = 0.002
    actuator_delay_s: float = 0.001
    cg_x_shift_m: float = 0.0
    pitch_inertia_factor: float = 1.0
    contact_friction_factor: float = 1.0
    joint_damping_factor: float = 1.0
    noise_factor: float = 1.0
    battery_voltage_factor: float = 1.0


# The nominal matched model is always first.  The remaining cases are stress
# tests, not alternative fits: no plant value is selected by this optimizer.
PLANT_ENSEMBLE = (
    PlantVariation("nominal"),
    PlantVariation(
        "low_authority_high_delay", wheel_torque_factor=0.85,
        hip_torque_factor=0.90, sensor_delay_s=0.006,
        actuator_delay_s=0.004, pitch_inertia_factor=1.20,
    ),
    PlantVariation(
        "high_authority_low_inertia", wheel_torque_factor=1.15,
        hip_torque_factor=1.10, sensor_delay_s=0.004,
        actuator_delay_s=0.002, pitch_inertia_factor=0.80,
    ),
    PlantVariation(
        "cg_forward_low_friction", wheel_torque_factor=0.90,
        cg_x_shift_m=0.006, contact_friction_factor=0.65,
        joint_damping_factor=0.70,
    ),
    PlantVariation(
        "cg_backward_high_friction", wheel_torque_factor=1.10,
        cg_x_shift_m=-0.006, contact_friction_factor=1.20,
        joint_damping_factor=1.30,
    ),
    PlantVariation(
        "noisy_low_battery", noise_factor=2.5,
        battery_voltage_factor=0.85, sensor_delay_s=0.004,
        actuator_delay_s=0.003,
    ),
    PlantVariation(
        "combined_conservative", wheel_torque_factor=0.85,
        hip_torque_factor=0.85, sensor_delay_s=0.006,
        actuator_delay_s=0.004, cg_x_shift_m=0.005,
        pitch_inertia_factor=1.20, contact_friction_factor=0.70,
        joint_damping_factor=0.75, noise_factor=2.0,
        battery_voltage_factor=0.90,
    ),
)


def _spec(name: str, low: float, high: float) -> ParamSpec:
    definition = PARAMS_BY_NAME[name]
    if low < definition.min or high > definition.max:
        raise ValueError(f"optimizer bounds for {name} exceed firmware schema")
    return ParamSpec(low, high, zero_ok=low == 0.0, scale="linear")


# Conservative engineering bounds are intentionally narrower than schema
# storage bounds.  The latter include bench-test headroom that is not a safe
# autonomous search region.
LQR_SPACE = SearchSpace(params={
    "lqr_k_pitch_ret": _spec("lqr_k_pitch_ret", -2.0, -0.15),
    "lqr_k_pitch_ext": _spec("lqr_k_pitch_ext", -2.0, -0.15),
    "lqr_k_rate_ret": _spec("lqr_k_rate_ret", -0.80, -0.03),
    "lqr_k_rate_ext": _spec("lqr_k_rate_ext", -0.80, -0.03),
    "lqr_k_vel": _spec("lqr_k_vel", -0.20, -0.005),
    "lqr_barrier_k": _spec("lqr_barrier_k", 0.0, 3.0),
})
VELOCITY_SPACE = SearchSpace(params={
    "vel_pi_kp": _spec("vel_pi_kp", 0.05, 0.45),
    "vel_pi_ki": _spec("vel_pi_ki", 0.0, 0.20),
    "vel_pi_kff": _spec("vel_pi_kff", 0.0, 0.12),
    "vel_pi_rate_lim": _spec("vel_pi_rate_lim", 0.15, 0.75),
    "vel_pi_int_max": _spec("vel_pi_int_max", 0.10, 1.20),
})
YAW_SPACE = SearchSpace(params={
    "yaw_pi_kp": _spec("yaw_pi_kp", 0.01, 0.20),
    "yaw_pi_ki": _spec("yaw_pi_ki", 0.0, 0.10),
    "yaw_pi_torque_max": _spec("yaw_pi_torque_max", 0.02, 0.15),
    "yaw_pi_int_max": _spec("yaw_pi_int_max", 0.01, 0.20),
})
HIP_ROLL_SPACE = SearchSpace(params={
    "hip_running_kp": _spec("hip_running_kp", 12.0, 45.0),
    "hip_running_kd": _spec("hip_running_kd", 0.20, 1.50),
    "hip_running_tff_ret": _spec("hip_running_tff_ret", -4.0, -0.5),
    "hip_running_tff_ext": _spec("hip_running_tff_ext", -4.0, -0.5),
    "hip_roll_kp": _spec("hip_roll_kp", 10.0, 45.0),
    "hip_roll_kd": _spec("hip_roll_kd", 0.20, 1.50),
    "roll_kp": _spec("roll_kp", 0.20, 3.0),
    "roll_kd": _spec("roll_kd", 0.005, 0.20),
    "roll_ki": _spec("roll_ki", 0.0, 0.20),
    "roll_int_max": _spec("roll_int_max", 0.02, 0.20),
    "ff1_alpha": _spec("ff1_alpha", 0.0, 0.40),
    "ff2_alpha": _spec("ff2_alpha", 0.0, 0.25),
})
INTEGRATED_SPACE = SearchSpace(params={
    **LQR_SPACE.params,
    **VELOCITY_SPACE.params,
    **YAW_SPACE.params,
    **HIP_ROLL_SPACE.params,
})
SPACE_BY_STAGE = {
    "lqr": LQR_SPACE,
    "velocity": VELOCITY_SPACE,
    "yaw": YAW_SPACE,
    "hip_roll": HIP_ROLL_SPACE,
    "integrated": INTEGRATED_SPACE,
}


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _transition(t: float, start: float, duration: float,
                before: float, after: float) -> float:
    if t <= start:
        return before
    if t >= start + duration:
        return after
    return before + (after - before) * _smoothstep((t - start) / duration)


def _velocity_profile(t: float) -> float:
    # All commands stay inside the current profile-3 limit of 0.75 m/s.
    if t < 1.0:
        return 0.0
    if t < 2.0:
        return _transition(t, 1.0, 1.0, 0.0, 0.45)
    if t < 3.0:
        return 0.45
    if t < 4.0:
        return _transition(t, 3.0, 1.0, 0.45, -0.35)
    if t < 5.0:
        return -0.35
    if t < 6.0:
        return _transition(t, 5.0, 1.0, -0.35, 0.65)
    if t < 7.0:
        return 0.65
    if t < 8.0:
        return _transition(t, 7.0, 1.0, 0.65, 0.0)
    return 0.0


def _cruise_profile(t: float) -> float:
    return _transition(t, 0.5, 1.0, 0.0, 0.40)


def _yaw_profile(t: float) -> float:
    # Inside the current 1.5 rad/s profile-3 limit.
    if t < 2.0:
        return 0.0
    if t < 3.0:
        return _transition(t, 2.0, 1.0, 0.0, 1.20)
    if t < 5.0:
        return 1.20
    if t < 6.0:
        return _transition(t, 5.0, 1.0, 1.20, -1.20)
    if t < 8.0:
        return -1.20
    if t < 9.0:
        return _transition(t, 8.0, 1.0, -1.20, 0.0)
    return 0.0


def _roll_profile(t: float) -> float:
    if t < 3.0:
        return 0.0
    if t < 4.0:
        return _transition(t, 3.0, 1.0, 0.0, 0.07)
    if t < 6.0:
        return 0.07
    if t < 7.0:
        return _transition(t, 6.0, 1.0, 0.07, -0.07)
    if t < 9.0:
        return -0.07
    if t < 10.0:
        return _transition(t, 9.0, 1.0, -0.07, 0.0)
    return 0.0


def _body_disturbance(t: float) -> float:
    if 1.5 <= t < 1.65:
        return 5.0
    if 3.0 <= t < 3.15:
        return -5.0
    return 0.0


def _integrated_disturbance(t: float) -> float:
    for start, force in ((2.5, 4.0), (5.5, -4.0), (8.5, 4.0)):
        if start <= t < start + 0.15:
            return force
    return 0.0


def _roll_disturbance(t: float) -> float:
    return 8.0 if 4.5 <= t < 4.7 or 7.5 <= t < 7.7 else 0.0


def _hip_cycle(t: float) -> float:
    robot = DEFAULT_PARAMS.robot
    alpha = 0.5 - 0.45 * math.cos(2.0 * math.pi * t / 6.0)
    return alpha_to_hip_q(alpha, robot)


def _hip_cycle_velocity(t: float) -> float:
    robot = DEFAULT_PARAMS.robot
    dalpha = 0.45 * 2.0 * math.pi / 6.0 * math.sin(2.0 * math.pi * t / 6.0)
    return dalpha * (robot.Q_EXT - robot.Q_ALPHA_RET)


def _zero(_: float) -> float:
    return 0.0


def _pitch_perturb(model, data, degrees: float) -> None:
    root = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")]
    current = 2.0 * math.atan2(float(data.qpos[root + 5]),
                               float(data.qpos[root + 3]))
    target = current + math.radians(degrees)
    data.qpos[root + 3:root + 7] = (
        math.cos(target / 2.0), 0.0, math.sin(target / 2.0), 0.0)
    mujoco.mj_forward(model, data)


def _pitch_plus_7(model, data, _params) -> None:
    _pitch_perturb(model, data, 7.0)


def _pitch_minus_7(model, data, _params) -> None:
    _pitch_perturb(model, data, -7.0)


def _scenario_suite(stage: str) -> tuple[ScenarioConfig, ...]:
    robot = DEFAULT_PARAMS.robot
    static_ret = ScenarioConfig(
        name="robust_static_ret_plus7", display_name="Retracted +7 deg",
        duration=4.5, active_controllers=frozenset({"lqr"}),
        hip_mode="position", initial_hip_q=robot.Q_ALPHA_RET,
        hip_profile=lambda _t: robot.Q_ALPHA_RET,
        init_fn=_pitch_plus_7, dist_fn=_body_disturbance,
    )
    static_ext = ScenarioConfig(
        name="robust_static_ext_minus7", display_name="Extended -7 deg",
        duration=4.5, active_controllers=frozenset({"lqr"}),
        hip_mode="position", initial_hip_q=robot.Q_EXT,
        hip_profile=lambda _t: robot.Q_EXT,
        init_fn=_pitch_minus_7, dist_fn=_body_disturbance,
    )
    height = ScenarioConfig(
        name="robust_height_sweep", display_name="Height sweep + pushes",
        duration=8.0, active_controllers=frozenset({"lqr"}),
        hip_mode="position", hip_profile=_hip_cycle,
        hip_vel_profile=_hip_cycle_velocity, dist_fn=_body_disturbance,
    )
    drive = ScenarioConfig(
        name="robust_drive", display_name="Rate-shaped reversible drive",
        duration=9.0, active_controllers=frozenset({"lqr", "velocity_pi"}),
        hip_mode="position", v_profile=_velocity_profile,
        use_theta_ref_correction=True,
    )
    drive_height = ScenarioConfig(
        name="robust_drive_height", display_name="Drive + height + pushes",
        duration=9.0, active_controllers=frozenset({"lqr", "velocity_pi"}),
        hip_mode="position", v_profile=_velocity_profile,
        hip_profile=_hip_cycle, hip_vel_profile=_hip_cycle_velocity,
        dist_fn=_integrated_disturbance, use_theta_ref_correction=True,
    )
    yaw = ScenarioConfig(
        name="robust_yaw_drive", display_name="Drive + reversible yaw",
        duration=10.0,
        active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
        hip_mode="position", v_profile=_cruise_profile,
        omega_profile=_yaw_profile, use_theta_ref_correction=True,
    )
    integrated = ScenarioConfig(
        name="robust_integrated", display_name="Drive/yaw/roll/height stress",
        duration=11.0,
        active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
        hip_mode="impedance", v_profile=_velocity_profile,
        omega_profile=_yaw_profile, roll_profile=_roll_profile,
        hip_profile=_hip_cycle, hip_vel_profile=_hip_cycle_velocity,
        dist_fn=_integrated_disturbance, roll_dist_fn=_roll_disturbance,
        use_theta_ref_correction=True,
    )
    if stage == "lqr":
        return static_ret, static_ext, height
    if stage == "velocity":
        return drive, drive_height
    if stage == "yaw":
        return drive, yaw
    if stage == "hip_roll":
        return height, integrated
    if stage == "integrated":
        return static_ret, static_ext, height, drive, drive_height, yaw, integrated
    raise ValueError(f"unknown optimizer stage {stage!r}")


def _with_control_overrides(overrides: dict[str, float]):
    values = dict(DEFAULT_PARAMS.firmware_params)
    values.update({name: float(value) for name, value in overrides.items()})
    validate_values(values)
    return replace(DEFAULT_PARAMS, firmware_params=tuple(sorted(values.items())))


def _with_plant_variation(params, variation: PlantVariation):
    wheel = params.motors.wheel
    hip = params.motors.hip
    motors = replace(
        params.motors,
        wheel=replace(
            wheel,
            odrive_torque_constant=(
                wheel.odrive_torque_constant / variation.wheel_torque_factor)),
        hip=replace(
            hip,
            torque_scale_ret=hip.torque_scale_ret * variation.hip_torque_factor,
            torque_scale_ext=hip.torque_scale_ext * variation.hip_torque_factor,
        ),
    )
    robot = replace(
        params.robot,
        box_cg_x=params.robot.box_cg_x + variation.cg_x_shift_m,
        battery_cg_x=params.robot.battery_cg_x + variation.cg_x_shift_m,
    )
    latency = replace(
        params.latency, sensor_delay_s=variation.sensor_delay_s,
        actuator_delay_s=variation.actuator_delay_s,
    )
    noise = replace(
        params.noise,
        pitch_std_rad=params.noise.pitch_std_rad * variation.noise_factor,
        pitch_rate_std_rad_s=(
            params.noise.pitch_rate_std_rad_s * variation.noise_factor),
        accel_std=params.noise.accel_std * variation.noise_factor,
        roll_std_rad=params.noise.roll_std_rad * variation.noise_factor,
    )
    battery = replace(
        params.battery,
        V_nom=params.battery.V_nom * variation.battery_voltage_factor,
    )
    return replace(
        params, robot=robot, motors=motors, latency=latency,
        noise=noise, battery=battery,
    )


def _mutate_model(model, data, variation: PlantVariation) -> None:
    # MuJoCo stores principal inertias in each body's local frame. The robot's
    # pitch axis is local Y in its symmetric nominal pose.
    model.body_inertia[1:, 1] *= variation.pitch_inertia_factor
    model.geom_friction[:, 0] *= variation.contact_friction_factor
    model.dof_damping[:] *= variation.joint_damping_factor
    mujoco.mj_forward(model, data)


def _case_score(metrics: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    duration = max(1e-6, float(metrics.get("requested_duration_s", 0.0)
                               or metrics.get("survived_s", 1.0)))
    survival = float(metrics["survived_s"]) / duration
    if metrics["fell"]:
        reasons.append(metrics.get("fail_reason") or "fell")
    if metrics["max_pitch_deg"] > 20.0:
        reasons.append("trim-relative pitch exceeded 20 deg")
    if metrics["rms_pitch_rate_dps"] > 35.0:
        reasons.append("pitch-rate RMS exceeded 35 deg/s")
    if metrics["max_roll_deg"] > 12.0:
        reasons.append("roll exceeded 12 deg")
    if metrics["peak_hip_torque_nm"] > 7.5:
        reasons.append("hip torque exceeded 7.5 N.m")
    if metrics["wheel_liftoff_s"] > 0.10:
        reasons.append("wheel liftoff exceeded 0.10 s")
    if metrics["wheel_torque_saturation_frac"] > 0.50:
        reasons.append("wheel torque saturated over 50% of the case")

    soft = (
        0.30 * metrics["rms_pitch_deg"] / 2.0
        + 0.16 * metrics["rms_pitch_rate_dps"] / 8.0
        + 0.08 * metrics["max_pitch_deg"] / 10.0
        + 0.18 * metrics["vel_track_rms_ms"] / 0.20
        + 0.10 * metrics["yaw_track_rms_rads"] / 0.20
        + 0.07 * metrics["roll_track_rms_deg"] / 3.0
        + 0.05 * metrics["hip_track_rms_rad"] / 0.08
        + 0.04 * metrics["wheel_torque_saturation_frac"] / 0.10
        + 0.02 * metrics["rms_tau_sym_nm"] / 0.20
    )
    if reasons:
        soft += 1000.0 + 100.0 * max(0.0, 1.0 - survival)
    return float(soft), reasons


def evaluate_candidate(candidate: dict[str, float], *, stage: str,
                       fixed_overrides: dict[str, float],
                       ensemble_size: int, seed: int) -> dict:
    overrides = dict(fixed_overrides)
    overrides.update(candidate)
    base = _with_control_overrides(overrides)
    variations = PLANT_ENSEMBLE[:max(1, ensemble_size)]
    scores: list[float] = []
    details: list[dict] = []
    failures: list[str] = []
    for variation_index, variation in enumerate(variations):
        params = _with_plant_variation(base, variation)
        for scenario_index, scenario in enumerate(_scenario_suite(stage)):
            metrics = run(
                params, scenario,
                rng_seed=seed + 100 * variation_index + scenario_index,
                model_mutator=partial(_mutate_model, variation=variation),
            )
            metrics["requested_duration_s"] = scenario.duration
            score, reasons = _case_score(metrics)
            scores.append(score)
            details.append({
                "variation": variation.name, "scenario": scenario.name,
                "score": score, "metrics": metrics, "reasons": reasons,
            })
            failures.extend(
                f"{variation.name}/{scenario.name}: {reason}" for reason in reasons)
    worst = max(scores)
    median = statistics.median(scores)
    mean = statistics.fmean(scores)

    # A small regularizer prevents unearned large gain changes when two
    # candidates are otherwise indistinguishable under the stress suite.
    defaults = dict(DEFAULT_PARAMS.firmware_params)
    normalized_delta = []
    for name, value in candidate.items():
        spec = SPACE_BY_STAGE[stage].params[name]
        normalized_delta.append(
            ((float(value) - defaults[name]) / max(1e-9, spec.hi - spec.lo)) ** 2)
    regularization = 0.02 * math.sqrt(
        statistics.fmean(normalized_delta)) if normalized_delta else 0.0
    return {
        "fitness": worst + 0.20 * median + 0.05 * mean + regularization,
        "status": "PASS" if not failures else "FAIL",
        "worst_case_score": worst,
        "median_case_score": median,
        "mean_case_score": mean,
        "regularization": regularization,
        "failures": failures,
        "details": details,
    }


def _complete_snapshot(values: dict[str, float]) -> dict:
    return {
        f"0x{definition.id:04X}": {"name": name, "value": float(values[name])}
        for name, definition in sorted(
            PARAMS_BY_NAME.items(), key=lambda item: item[1].id)
        if name in values
    }


def _parameter_diff(before: dict[str, float], after: dict[str, float]) -> list[dict]:
    return [
        {
            "name": name, "before": before[name], "after": after[name],
            "delta": after[name] - before[name],
        }
        for name in sorted(after)
        if name in before and not math.isclose(
            before[name], after[name], rel_tol=0.0, abs_tol=1e-12)
    ]


def _gain_scale_sweep(values: dict[str, float], ensemble_size: int,
                      seed: int) -> list[dict]:
    names = (
        "lqr_k_pitch_ret", "lqr_k_pitch_ext", "lqr_k_rate_ret",
        "lqr_k_rate_ext", "lqr_k_vel",
    )
    output = []
    for factor in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        scaled = dict(values)
        for name in names:
            scaled[name] = values[name] * factor
        result = evaluate_candidate(
            {name: scaled[name] for name in LQR_SPACE.names}, stage="lqr",
            fixed_overrides=scaled, ensemble_size=ensemble_size, seed=seed,
        )
        output.append({
            "factor": factor, "status": result["status"],
            "fitness": result["fitness"],
            "worst_case_score": result["worst_case_score"],
            "failures": result["failures"],
        })
    return output


def _write_checkpoint(output_path: Path, report_path: Path,
                      values: dict[str, float], report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_complete_snapshot(values), indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=30,
                        help="generations per staged controller group")
    parser.add_argument("--final-generations", type=int, default=12,
                        help="joint-polish generations")
    parser.add_argument("--lambda", dest="lambda_", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--validation-ensemble-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline-snapshot", type=Path,
                        default=DEFAULT_BASELINE_SNAPSHOT)
    parser.add_argument(
        "--stages", default="lqr,velocity,yaw,hip_roll,integrated",
        help="comma-separated staged search order")
    args = parser.parse_args()

    if args.generations < 0 or args.final_generations < 0:
        parser.error("generation counts must be nonnegative")
    baseline = dict(DEFAULT_PARAMS.firmware_params)
    args.baseline_snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_snapshot.write_text(
        json.dumps(_complete_snapshot(baseline), indent=2) + "\n",
        encoding="utf-8")
    # Windows multiprocessing starts fresh interpreters. Make every worker use
    # this exact, lock-validated snapshot even if the GUI export changes while
    # optimization is running.
    os.environ[CONTROLLER_SNAPSHOT_ENV] = str(args.baseline_snapshot.resolve())
    optimized = dict(baseline)
    stages = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = [name for name in stages if name not in SPACE_BY_STAGE]
    if unknown:
        parser.error(f"unknown stages: {', '.join(unknown)}")

    started = time.time()
    report = {
        "schema_version": 1,
        "created_unix_s": started,
        "purpose": "offline robust balance-controller gain optimization",
        "plant_parameters_optimized": False,
        "robot_was_commanded_or_written": False,
        "jump_controller_optimized": False,
        "jump_exclusion_reason": (
            "jump plant/contact fidelity is not identified and jump is outside "
            "the requested reliable balancing objective"),
        "fixed_parameters": [
            "all physical parameters", "pitch trims", "watchdog thresholds",
            "LQR torque limit", "profile command limits", "roll offset limit",
        ],
        "baseline_controller_sha256": control_snapshot_sha256(baseline),
        "baseline_controller_snapshot": str(args.baseline_snapshot),
        "plant_ensemble": [asdict(item) for item in PLANT_ENSEMBLE],
        "search_bounds": {
            stage: {name: asdict(spec) for name, spec in space.params.items()}
            for stage, space in SPACE_BY_STAGE.items()
        },
        "stage_results": [],
        "status": "running",
    }

    for stage_index, stage in enumerate(stages):
        space = SPACE_BY_STAGE[stage]
        generations = (args.final_generations if stage == "integrated"
                       else args.generations)
        seed_params = {name: optimized[name] for name in space.names}
        evaluator = partial(
            evaluate_candidate, stage=stage, fixed_overrides=dict(optimized),
            ensemble_size=args.ensemble_size, seed=args.seed + 1000 * stage_index,
        )
        baseline_result = evaluator(seed_params)
        print(
            f"\n[{stage}] seed fitness={baseline_result['fitness']:.6f} "
            f"status={baseline_result['status']} generations={generations}")
        if generations > 0:
            optimizer = ESOptimizer(
                space, evaluator,
                config=ESConfig(
                    lambda_=args.lambda_,
                    n_workers=args.workers,
                    rng_seed=args.seed + stage_index,
                    sigma_init=0.06 if stage != "integrated" else 0.025,
                    sigma_min=0.003,
                    sigma_max=0.20,
                    patience=max(10, generations),
                    restart_patience=max(10, generations),
                    max_restarts=0,
                    tol=1e-5,
                    use_threads=False,
                ),
            )
            result = optimizer.run(max_iters=generations, seed_params=seed_params)
            optimized.update(result["best_params"])
        else:
            result = {
                "best_params": seed_params,
                "best_fitness": baseline_result["fitness"],
                "n_evals": 1, "n_gens": 0, "elapsed_s": 0.0,
                "stopped_reason": "zero_generations",
            }
        candidate_result = evaluate_candidate(
            {name: optimized[name] for name in space.names}, stage=stage,
            fixed_overrides=optimized, ensemble_size=args.ensemble_size,
            seed=args.seed + 1000 * stage_index,
        )
        report["stage_results"].append({
            "stage": stage,
            "seed_evaluation": baseline_result,
            "optimizer": result,
            "candidate_evaluation": candidate_result,
        })
        current_values = dict(baseline)
        current_values.update(optimized)
        report["optimized_parameter_diff"] = _parameter_diff(baseline, current_values)
        _write_checkpoint(args.output, args.report, current_values, report)

    final_values = dict(baseline)
    final_values.update(optimized)
    report["baseline_validation"] = evaluate_candidate(
        {}, stage="integrated", fixed_overrides=baseline,
        ensemble_size=args.validation_ensemble_size, seed=args.seed + 9000,
    )
    report["candidate_validation"] = evaluate_candidate(
        {}, stage="integrated", fixed_overrides=final_values,
        ensemble_size=args.validation_ensemble_size, seed=args.seed + 9000,
    )
    report["baseline_lqr_gain_scale_sweep"] = _gain_scale_sweep(
        baseline, args.validation_ensemble_size, args.seed + 10000)
    report["candidate_lqr_gain_scale_sweep"] = _gain_scale_sweep(
        final_values, args.validation_ensemble_size, args.seed + 11000)
    report["optimized_parameter_diff"] = _parameter_diff(baseline, final_values)
    report["candidate_controller_sha256"] = control_snapshot_sha256(final_values)
    report["elapsed_s"] = time.time() - started
    report["status"] = (
        "PASS" if report["candidate_validation"]["status"] == "PASS" else "FAIL")
    report["output_snapshot"] = str(args.output)
    _write_checkpoint(args.output, args.report, final_values, report)
    print(json.dumps({
        "status": report["status"],
        "baseline_fitness": report["baseline_validation"]["fitness"],
        "candidate_fitness": report["candidate_validation"]["fitness"],
        "changed_parameters": report["optimized_parameter_diff"],
        "output": str(args.output), "report": str(args.report),
        "elapsed_s": report["elapsed_s"],
    }, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
