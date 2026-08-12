"""Fit the robot-match report from the latest WLOG and saved configuration.

This deliberately reuses the GUI's shared WLOG decoder and plateau analysis;
there is no second binary log parser in the simulator.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize

from ...params import MotorParams, RobotGeometry, SimParams, WheelMotorParams
from ...physics import alpha_to_hip_q, get_equilibrium_pitch
from ...robot_match import (
    MATCH_REPORT, REPO_ROOT, control_snapshot_sha256,
)
from ...sim_loop import SimController, build_model_and_data, init_sim
from .param_snapshot import load_snapshot


GUI_DIR = REPO_ROOT / "software" / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from analysis.leg_height_sweep import band_split, plateau_report  # noqa: E402
from analysis.param_sidecar import load_matching_sidecar  # noqa: E402
from analysis.wlog_metrics import DecodedRun, compute_metrics, decode_run  # noqa: E402


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _latest_wlog() -> Path:
    logs = list((REPO_ROOT / "data" / "logs" / "runs").glob("*/*.WLOG"))
    if not logs:
        raise FileNotFoundError("no WLOG files under data/logs/runs")
    return max(logs, key=lambda p: (p.parent.name, p.name))


def _running_only(run: DecodedRun) -> DecodedRun:
    mask = run.fields["robot_state"].astype(np.int64) == 3
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"{run.path}: no RUNNING records")
    return DecodedRun(
        path=run.path,
        telem_version=run.telem_version,
        sample_rate_hz=run.sample_rate_hz,
        count=int(indices.size),
        t_micros=run.t_micros[indices],
        t_s=run.t_s[indices] - run.t_s[indices[0]],
        fields={name: values[indices] for name, values in run.fields.items()},
        has_gain_sched_alpha=run.has_gain_sched_alpha,
        source_kind=run.source_kind,
    )


def _state_durations(run: DecodedRun) -> dict[str, float]:
    names = {
        0: "STARTUP", 1: "CALIBRATION", 2: "STANDBY", 3: "RUNNING",
        4: "ESTOP", 5: "MANUAL", 6: "CMD_REJECT", 7: "JUMPING",
        8: "STANDING_UP", 9: "DISARMING",
    }
    dt = np.diff(run.t_s, append=run.t_s[-1] + 1.0 / run.sample_rate_hz)
    states = run.fields["robot_state"].astype(np.int64)
    return {
        names.get(int(state), str(int(state))): round(float(dt[states == state].sum()), 3)
        for state in np.unique(states)
    }


def _fit_geometry(control: dict[str, float], plateaus) -> tuple[dict, RobotGeometry]:
    robot = replace(RobotGeometry(), calib_backoff_rad=control["calib_backoff_rad"])
    alphas = np.linspace(0.0, 1.0, 5)
    trim_ret = control["lqr_pitch_trim_ret"]
    trim_ext = control["lqr_pitch_trim_ext"]
    trim_curve = control["lqr_trim_curve"]
    targets = list(trim_ret + alphas * (trim_ext - trim_ret)
                   + trim_curve * alphas * (1.0 - alphas))
    fit_alphas = list(alphas)
    weights = [1.0] * len(fit_alphas)

    equilibrium = [row for row in plateaus if row.equilibrium]
    for row in equilibrium:
        fit_alphas.append(row.alpha)
        targets.append(row.balance_rad)
        weights.append(2.0)

    weights_arr = np.sqrt(np.asarray(weights, dtype=np.float64))
    targets_arr = np.asarray(targets, dtype=np.float64)

    def candidate_from_vector(vector) -> RobotGeometry:
        battery_x, battery_z, extra_femur, extra_tibia, extra_coupler = vector
        return replace(
            robot,
            battery_cg_x=float(battery_x),
            battery_cg_z=float(battery_z),
            m_femur=robot.measured_femur_mass + float(extra_femur) / 2.0,
            m_tibia=robot.measured_tibia_mass + float(extra_tibia) / 2.0,
            m_coupler=robot.measured_coupler_mass + float(extra_coupler) / 2.0,
        )

    def residual(vector):
        candidate = candidate_from_vector(vector)
        predicted = np.asarray([
            get_equilibrium_pitch(candidate, alpha_to_hip_q(alpha, candidate))
            for alpha in fit_alphas
        ])
        return weights_arr * (predicted - targets_arr)

    measured_link_sum = (robot.measured_femur_mass + robot.measured_tibia_mass
                         + robot.measured_coupler_mass)
    initial_extras = [
        robot.unassigned_mass * mass / measured_link_sum
        for mass in (robot.measured_femur_mass, robot.measured_tibia_mass,
                     robot.measured_coupler_mass)
    ]
    # Chassis half-size is x=70,z=52 mm; battery half-size is x=60,z=18 mm.
    # "Towards the front" therefore permits x=[0,+10] mm and z=+/-34 mm.
    bounds = [
        (0.0, 0.010), (-0.034, 0.034),
        (0.0, robot.unassigned_mass), (0.0, robot.unassigned_mass),
        (0.0, robot.unassigned_mass),
    ]
    solution = minimize(
        lambda vector: float(np.mean(np.square(residual(vector)))),
        x0=np.asarray([0.010, -0.030, *initial_extras]),
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "eq",
            "fun": lambda vector: float(np.sum(vector[2:]) - robot.unassigned_mass),
        },
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    if not solution.success:
        raise RuntimeError(f"mass/CG fit failed: {solution.message}")
    matched = candidate_from_vector(solution.x)
    residual_deg = np.degrees(residual(solution.x) / weights_arr)
    geometry = {
        "box_cg_x_m": matched.box_cg_x,
        "box_cg_z_m": matched.box_cg_z,
        "battery_cg_x_m": matched.battery_cg_x,
        "battery_cg_z_m": matched.battery_cg_z,
        "effective_link_mass_kg_each": {
            "femur": matched.m_femur,
            "tibia": matched.m_tibia,
            "coupler": matched.m_coupler,
        },
        "unassigned_mass_allocation_kg_total": {
            "femur_bodies": float(solution.x[2]),
            "tibia_bodies": float(solution.x[3]),
            "coupler_bodies": float(solution.x[4]),
        },
        "fit_method": (
            "constrained weighted fit of battery position and unweighed mass "
            "allocation against the exported trim schedule plus stationary "
            "RUNNING plateaus"),
        "battery_fit_bounds_m": {"x": [0.0, 0.010], "z": [-0.034, 0.034]},
        "weighted_rms_residual_deg": float(np.sqrt(np.mean(
            np.square(residual(solution.x)))) * 180.0 / math.pi),
        "unweighted_residual_deg": [float(value) for value in residual_deg],
        "equilibrium_plateaus": [
            {
                "gain_sched_alpha": row.alpha,
                "duration_s": row.duration_s,
                "balance_pitch_deg": math.degrees(row.balance_rad),
                "mean_tau_sym_nm": row.mean_tau_nm,
                "wheel_drift_turns_s": row.drift_turns_s,
            }
            for row in equilibrium
        ],
        "provisional": True,
        "caveat": (
            "The fit places the battery at its forward/lower packaging limits and "
            "assigns the unresolved whole-robot mass to the femur bodies as an "
            "equivalent lump. These placements are identification variables, not "
            "component measurements; confirm with T0.2/T0.3 CG measurements."),
    }
    return geometry, matched


def _static_hip_response(params: SimParams, alpha: float,
                         duration_s: float = 2.0) -> dict[str, float]:
    """Settle one MuJoCo stance and return its plant-side hip response.

    Controller values are taken unchanged from ``params.firmware_params``.
    Only the command-to-joint torque conversion in ``HipMotorParams`` is an
    identification variable.
    """
    q_hip = alpha_to_hip_q(alpha, params.robot)
    model, data = build_model_and_data(params)
    init_sim(model, data, params, q_hip_init=q_hip)
    controller = SimController(model, data, params, rng_seed=0)
    rows: list[tuple[float, float, float, float]] = []
    step = 0
    average_after = max(0.0, duration_s - 0.5)
    while data.time < duration_s:
        if step % params.timing.ctrl_steps == 0:
            tick = controller.tick(
                model, data, q_hip_target=q_hip,
                v_target_ms=0.0, omega_target=0.0,
                use_lqr=True, use_velocity_pi=True, use_yaw_pi=True,
                use_impedance=False, use_roll_leveling=False,
                use_suspension=True, use_ff1=True, use_ff2=True,
            )
            if data.time >= average_after:
                rows.append((
                    tick["q_nom_L"] - tick["hip_q_L"],
                    tick["q_nom_R"] - tick["hip_q_R"],
                    tick["tau_hip_L"], tick["tau_hip_R"],
                ))
        mujoco.mj_step(model, data)
        step += 1
    values = np.asarray(rows, dtype=np.float64)
    return {
        "sag_l_rad": float(np.mean(values[:, 0])),
        "sag_r_rad": float(np.mean(values[:, 1])),
        "reported_torque_l_nm": float(np.mean(values[:, 2])),
        "reported_torque_r_nm": float(np.mean(values[:, 3])),
    }


def _fit_hip_drive(control: dict[str, float], plateaus,
                   robot: RobotGeometry, wheel: WheelMotorParams) -> dict:
    """Fit plant-side hip torque transmission from trustworthy static sag.

    This intentionally does not tune ``hip_running_kp/kd/tff``.  Those remain
    byte-for-byte values from the controller snapshot.  A scheduled effective
    torque scale is used because a single scale cannot reproduce the two
    measured linkage load points; that schedule remains provisional until the
    unloaded hip and holding-torque tests separate gearbox loss from geometry.
    """
    fit_rows = [row for row in plateaus if row.equilibrium]
    all_rows = [row for row in plateaus if row.equilibrium or row.alpha < 0.1]
    if len(fit_rows) < 2:
        return {
            "torque_scale_ret": 1.0,
            "torque_scale_ext": 1.0,
            "fit_method": "not fitted: fewer than two equilibrium hip-load anchors",
            "anchors": [],
            "provisional": True,
        }

    base_motors = MotorParams(wheel=wheel)

    def make_params(scales) -> SimParams:
        hip = replace(
            base_motors.hip,
            torque_scale_ret=float(scales[0]),
            torque_scale_ext=float(scales[1]),
        )
        return SimParams(
            robot=robot,
            motors=replace(base_motors, hip=hip),
            firmware_params=tuple(sorted(control.items())),
        )

    cache: dict[tuple[float, float, float], dict[str, float]] = {}

    def response(scales, alpha: float) -> dict[str, float]:
        key = (round(float(scales[0]), 7), round(float(scales[1]), 7),
               round(float(alpha), 9))
        if key not in cache:
            cache[key] = _static_hip_response(make_params(scales), alpha)
        return cache[key]

    def residual(scales) -> np.ndarray:
        errors = []
        for row in fit_rows:
            predicted = response(scales, row.alpha)
            measured_sag = 0.5 * (row.hip_sag_l_rad + row.hip_sag_r_rad)
            measured_torque = row.mean_hip_torque_nm
            predicted_sag = 0.5 * (
                predicted["sag_l_rad"] + predicted["sag_r_rad"])
            predicted_torque = 0.5 * (
                predicted["reported_torque_l_nm"]
                + predicted["reported_torque_r_nm"])
            errors.extend((
                (predicted_sag - measured_sag) / 0.005,
                (predicted_torque - measured_torque) / 0.10,
            ))
        return np.asarray(errors, dtype=np.float64)

    solution = least_squares(
        residual, x0=np.asarray([0.85, 0.65]),
        bounds=(np.asarray([0.25, 0.25]), np.asarray([1.25, 1.25])),
        diff_step=1e-3, xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=80,
    )
    if not solution.success:
        raise RuntimeError(f"hip transmission fit failed: {solution.message}")

    scales = solution.x
    anchors = []
    for row in all_rows:
        predicted = response(scales, row.alpha)
        measured_sag = 0.5 * (row.hip_sag_l_rad + row.hip_sag_r_rad)
        predicted_sag = 0.5 * (
            predicted["sag_l_rad"] + predicted["sag_r_rad"])
        predicted_torque = 0.5 * (
            predicted["reported_torque_l_nm"]
            + predicted["reported_torque_r_nm"])
        anchors.append({
            "gain_sched_alpha": row.alpha,
            "used_for_fit": bool(row.equilibrium),
            "measured_sag_l_rad": row.hip_sag_l_rad,
            "measured_sag_r_rad": row.hip_sag_r_rad,
            "measured_mean_sag_rad": measured_sag,
            "simulated_mean_sag_rad": predicted_sag,
            "sag_residual_rad": predicted_sag - measured_sag,
            "measured_mean_reported_torque_nm": row.mean_hip_torque_nm,
            "simulated_mean_reported_torque_nm": predicted_torque,
            "torque_residual_nm": predicted_torque - row.mean_hip_torque_nm,
        })
    return {
        "torque_scale_ret": float(scales[0]),
        "torque_scale_ext": float(scales[1]),
        "fit_method": (
            "MuJoCo static-settle least squares against equilibrium hip sag and "
            "reported load torque; controller kp/kd/tff held fixed"),
        "fitted_equilibrium_anchor_count": len(fit_rows),
        "normalized_rms_residual": float(np.sqrt(np.mean(residual(scales) ** 2))),
        "anchors": anchors,
        "provisional": True,
        "caveat": (
            "This is an effective command-to-joint torque schedule inferred from "
            "two loaded poses. It may absorb linkage geometry, mass placement, "
            "gearbox loss, and torque-telemetry scale; T2.3/T2.4 must separate them."),
    }


def build_report(wlog_path: Path | None = None) -> dict:
    wlog_path = Path(wlog_path) if wlog_path else _latest_wlog()
    export_path = REPO_ROOT / "software" / "gui" / "parameter_exports" / "Default gains.json"
    odrive_path = (REPO_ROOT / "components" / "characterization" / "odrive" /
                   "USB GUI" / "savedPresets" / "Both axis working CS 3 and 4 .json")
    control = load_snapshot(export_path)
    run = decode_run(wlog_path)
    running = _running_only(run)
    sidecar = load_matching_sidecar(wlog_path)

    def initial(name: str, fallback: float) -> float:
        if sidecar is None:
            return fallback
        value = sidecar.initial_value(name)
        return fallback if value is None else value

    plateaus = plateau_report(
        run,
        torque_limit_nm=initial("lqr_torque_limit", control["lqr_torque_limit"]),
        rate_lim=initial("vel_pi_rate_lim", control["vel_pi_rate_lim"]),
        theta_max_fwd=initial("theta_max_fwd_ret", control["theta_max_fwd_ret"]),
        theta_max_bwd=initial("theta_max_bwd_ret", control["theta_max_bwd_ret"]),
    )
    geometry, matched_robot = _fit_geometry(control, plateaus)

    odrive = json.loads(odrive_path.read_text(encoding="utf-8"))
    axis_drives = [odrive[f"axis{axis}.motor.config"] for axis in (0, 1)]
    configured_kt = float(np.mean([item["torque_constant"] for item in axis_drives]))
    current_limit = float(min(item["current_lim"] for item in axis_drives))
    wheel = WheelMotorParams(KV=70.0, current_limit=current_limit,
                             odrive_torque_constant=configured_kt)
    hip_drive = _fit_hip_drive(control, plateaus, matched_robot, wheel)

    metrics = compute_metrics(running)
    f = running.fields
    alpha = f["gain_sched_alpha"]
    pitch_error = f["pitch_rad"] - f["theta_ref"] - f["pitch_trim_rad"]
    torque_limit = initial("lqr_torque_limit", control["lqr_torque_limit"])
    saturation = float(np.mean(np.abs(f["tau_sym"]) >= 0.98 * torque_limit))
    sidecar_differences = []
    if sidecar is not None:
        for name in sorted(set(control) & sidecar.names):
            logged_value = sidecar.initial_value(name)
            if abs(control[name] - logged_value) > 1e-5:
                sidecar_differences.append({
                    "name": name,
                    "export_value": control[name],
                    "log_initial_value": logged_value,
                })
    hip_fits = {}
    for side in ("l", "r"):
        slope, intercept = np.polyfit(alpha, f[f"hip_{side}_pos_rad"], 1)
        residual = f[f"hip_{side}_pos_rad"] - (intercept + slope * alpha)
        hip_fits[side] = {
            "intercept_rad": float(intercept), "span_rad": float(slope),
            "rms_residual_rad": float(np.sqrt(np.mean(residual ** 2))),
        }

    return {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "control_export": _relative(export_path),
        "control_snapshot": {
            "parameter_count": len(control),
            "sha256": control_snapshot_sha256(control),
            "fit_policy": (
                "immutable during identification; only plant-side mass, CG, "
                "inertia, friction, delay, contact, and actuator conversions may vary"),
        },
        "latest_wlog": _relative(wlog_path),
        "odrive_snapshot": _relative(odrive_path),
        "parameter_reconciliation": {
            "export_parameter_count": len(control),
            "log_sidecar_parameter_count": len(sidecar.names) if sidecar else 0,
            "common_parameter_count": len(set(control) & sidecar.names) if sidecar else 0,
            "meaningful_initial_differences": sidecar_differences,
            "running_active_profiles": sorted(
                int(value) for value in np.unique(f["active_profile"])),
            "running_torque_limit_nm": torque_limit,
            "note": (
                "Historical replay uses the sidecar values active during the run. "
                "Current scenarios use the latest export. Differences are listed "
                "explicitly above and are not plant-fit variables."),
        },
        "mass_inventory": {
            "measured_robot_without_battery_kg": 3.242,
            "measured_battery_kg": matched_robot.m_battery,
            "model_robot_with_battery_kg": matched_robot.total_mass,
            "model_robot_without_battery_kg": matched_robot.total_mass_without_battery,
            "electronics_box_without_battery_kg": matched_robot.m_box,
            "hip_motor_catalog_kg_each": matched_robot.motor_mass,
            "wheel_motor_kg_each": matched_robot.wheel_motor_mass,
            "wheel_tpu_kg_each": matched_robot.wheel_tpu_mass,
            "wheel_pla_rim_kg_each": matched_robot.wheel_rim_mass,
            "pla_coupler_kg_each": matched_robot.measured_coupler_mass,
            "pla_femur_kg_each": matched_robot.measured_femur_mass,
            "pla_tibia_kg_each": matched_robot.measured_tibia_mass,
            "bearing_6804_kg_each": matched_robot.m_bearing,
            "bearing_count": 2 * matched_robot.bearings_per_leg,
            "known_plus_catalog_without_battery_kg": (
                matched_robot.total_mass_without_battery
                - matched_robot.unassigned_mass),
            "unassigned_mass_distributed_to_links_kg": matched_robot.unassigned_mass,
            "note": (
                "AK45 masses remain the existing 260 g catalog values because no "
                "replacement measurement was supplied. The total-mass residual "
                "absorbs fasteners, shafts, mounts, wiring, and any catalog error."),
        },
        "geometry": geometry,
        "wheel_drive": {
            "motor_kv_rpm_per_v": wheel.KV,
            "physical_kt_nm_per_a": wheel.Kt,
            "odrive_configured_torque_constant_nm_per_a": configured_kt,
            "odrive_current_limit_a": current_limit,
            "command_to_physical_torque_scale": wheel.command_torque_scale,
            "physical_current_limited_torque_nm": wheel.torque_limit,
            "provisional": True,
            "caveat": (
                "Derived from the latest saved ODrive preset, not a live "
                "configuration read or blocked-wheel torque measurement. It assumes "
                "the installed wheel motor is the documented 70 KV variant."),
        },
        "hip_drive": hip_drive,
        "log_evidence": {
            "telem_version": run.telem_version,
            "sample_rate_hz": run.sample_rate_hz,
            "samples": run.count,
            "duration_s": float(run.t_s[-1]),
            "state_durations_s": _state_durations(run),
            "running_samples": running.count,
            "running_duration_s": float(running.t_s[-1]),
            "running_faults": int(np.count_nonzero(f["fault_code"])),
            "running_rms_trim_relative_pitch_error_deg": metrics["rms_pitch_deg"],
            "running_rms_pitch_rate_deg_s": metrics["rms_pitch_rate_dps"],
            "running_lqr_band_1p5_to_4_hz_rms_deg": math.degrees(
                band_split(pitch_error, running.sample_rate_hz)["lqr"]),
            "running_torque_saturation_fraction": saturation,
            "max_wheel_speed_turns_s": {
                "left": metrics["max_wm_l_vel_turns_s"],
                "right": metrics["max_wm_r_vel_turns_s"],
            },
            "hip_position_fit": hip_fits,
            "hip_static_load_anchors": [
                {
                    "gain_sched_alpha": row.alpha,
                    "mean_hip_torque_nm": row.mean_hip_torque_nm,
                    "hip_sag_l_rad": row.hip_sag_l_rad,
                    "hip_sag_r_rad": row.hip_sag_r_rad,
                }
                for row in plateaus
                if row.equilibrium or row.alpha < 0.1
            ],
            "wheel_current_observed": False,
        },
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wlog", type=Path)
    parser.add_argument("--output", type=Path, default=MATCH_REPORT)
    args = parser.parse_args()
    report = build_report(args.wlog)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
