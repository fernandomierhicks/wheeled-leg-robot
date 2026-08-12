"""Headless MuJoCo stability matrix for the robot-matched default profile."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ...defaults import DEFAULT_PARAMS
from ...physics import alpha_to_hip_q, get_equilibrium_pitch
from ...robot_match import MATCH_REPORT, control_snapshot_sha256
from ...sim_loop import SimController, build_model_and_data, init_sim


def simulate_case(alpha: float, perturb_deg: float, duration_s: float = 10.0,
                  seed: int = 1, *, params=DEFAULT_PARAMS,
                  profile: str = "robot_matched") -> dict:
    robot = params.robot
    q_hip = alpha_to_hip_q(alpha, robot)
    model, data = build_model_and_data(params)
    init_sim(model, data, params, q_hip_init=q_hip)

    root = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")]
    theta_eq = get_equilibrium_pitch(robot, q_hip)
    theta_initial = theta_eq + math.radians(perturb_deg)
    data.qpos[root + 3:root + 7] = (
        math.cos(theta_initial / 2.0), 0.0,
        math.sin(theta_initial / 2.0), 0.0)
    mujoco.mj_forward(model, data)

    controller = SimController(model, data, params, rng_seed=seed)
    firmware = dict(params.firmware_params)
    trim = (firmware["lqr_pitch_trim_ret"]
            + alpha * (firmware["lqr_pitch_trim_ext"]
                       - firmware["lqr_pitch_trim_ret"])
            + firmware["lqr_trim_curve"] * alpha * (1.0 - alpha))
    errors = []
    peak_command = 0.0
    fault = None
    tick = None
    step = 0
    while data.time < duration_s:
        if step % params.timing.ctrl_steps == 0:
            tick = controller.tick(
                model, data, q_hip_target=q_hip,
                v_target_ms=0.0, omega_target=0.0,
                use_lqr=True, use_velocity_pi=True, use_yaw_pi=True,
                use_impedance=False, use_roll_leveling=False,
                use_suspension=True, use_ff1=True, use_ff2=True,
            )
            errors.append(tick["pitch"] - tick["theta_ref"] - trim)
            peak_command = max(peak_command, abs(tick["tau_cmd_L"]),
                               abs(tick["tau_cmd_R"]))
            fault = tick["firmware_fault"]
            if tick["fell"]:
                break
        mujoco.mj_step(model, data)
        step += 1

    errors_arr = np.asarray(errors)
    return {
        "profile": profile,
        "gain_sched_alpha": alpha,
        "initial_perturbation_deg": perturb_deg,
        "requested_duration_s": duration_s,
        "survived_s": float(data.time),
        "stable": bool(data.time >= duration_s and fault is None),
        "firmware_fault": fault,
        "physical_equilibrium_deg": math.degrees(theta_eq),
        "firmware_trim_deg": math.degrees(trim),
        "rms_trim_relative_error_deg": float(np.degrees(
            np.sqrt(np.mean(errors_arr ** 2)))) if errors_arr.size else None,
        "max_trim_relative_error_deg": float(np.degrees(
            np.max(np.abs(errors_arr)))) if errors_arr.size else None,
        "peak_firmware_torque_command_nm": peak_command,
        "final_pitch_deg": math.degrees(tick["pitch"]) if tick else None,
        "final_velocity_ms": tick["v_measured"] if tick else None,
    }


def build_validation(duration_s: float = 10.0) -> dict:
    alphas = (0.0, 0.5, 0.72947, 0.920414, 1.0)
    cases = [
        simulate_case(alpha, perturb, duration_s, seed=index + 1)
        for index, (alpha, perturb) in enumerate(
            (pair for alpha in alphas for pair in ((alpha, 0.0), (alpha, 5.0))))
    ]
    wheel = DEFAULT_PARAMS.motors.wheel
    unit_scale_motors = replace(
        DEFAULT_PARAMS.motors,
        wheel=replace(wheel, odrive_torque_constant=wheel.Kt),
    )
    robot = DEFAULT_PARAMS.robot
    measured_link_sum = (robot.measured_femur_mass + robot.measured_tibia_mass
                         + robot.measured_coupler_mass)
    neutral_robot = replace(
        robot,
        battery_cg_x=0.0,
        battery_cg_z=0.0,
        m_femur=(robot.measured_femur_mass
                 + robot.unassigned_mass * robot.measured_femur_mass
                 / measured_link_sum / 2.0),
        m_tibia=(robot.measured_tibia_mass
                + robot.unassigned_mass * robot.measured_tibia_mass
                / measured_link_sum / 2.0),
        m_coupler=(robot.measured_coupler_mass
                  + robot.unassigned_mass * robot.measured_coupler_mass
                  / measured_link_sum / 2.0),
    )
    ablation_params = {
        "neither_mass_placement_nor_odrive_scale": replace(
            DEFAULT_PARAMS, robot=neutral_robot, motors=unit_scale_motors),
        "mass_placement_only": replace(DEFAULT_PARAMS, motors=unit_scale_motors),
        "odrive_scale_only": replace(DEFAULT_PARAMS, robot=neutral_robot),
        "mass_placement_and_odrive_scale": DEFAULT_PARAMS,
    }
    ablations = [
        simulate_case(0.72947, 0.0, duration_s, seed=100 + index,
                      params=params, profile=name)
        for index, (name, params) in enumerate(ablation_params.items())
    ]
    return {
        "robot_match_report": MATCH_REPORT.name,
        "same_firmware_parameter_count": len(DEFAULT_PARAMS.firmware_params),
        "controller_snapshot_sha256": control_snapshot_sha256(
            dict(DEFAULT_PARAMS.firmware_params)),
        "controller_fit_allowed": False,
        "wheel_command_to_physical_torque_scale": wheel.command_torque_scale,
        "wheel_physical_torque_limit_nm": wheel.torque_limit,
        "hip_reported_to_physical_torque_scale": {
            "retracted": DEFAULT_PARAMS.motors.hip.torque_scale_ret,
            "extended": DEFAULT_PARAMS.motors.hip.torque_scale_ext,
        },
        "all_stable": all(case["stable"] for case in cases),
        "cases": cases,
        "representative_ablation_alpha_0p72947": ablations,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_validation(args.duration)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["all_stable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
