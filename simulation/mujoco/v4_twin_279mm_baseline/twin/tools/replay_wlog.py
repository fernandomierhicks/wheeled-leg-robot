"""Replay a hardware WLOG through MuJoCo without changing controller gains.

The default is a windowed multiple-shooting replay.  Every contiguous RUNNING
sample is consumed, but MuJoCo is reset from recorded telemetry at short window
boundaries so the score measures local plant response instead of unbounded
long-horizon drift.  Closed-loop mode runs the firmware-equivalent controller;
open-loop mode applies recorded wheel torque commands and hip setpoints.
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from ...defaults import DEFAULT_PARAMS
from ...models.motor import motor_taper
from ...physics import (
    firmware_hip_to_sim_q, hip_q_to_alpha, sim_q_to_firmware_hip, solve_ik,
)
from ...robot_match import control_snapshot_sha256
from ...sim_loop import (
    SimController, build_model_and_data, get_pitch_and_rate, get_roll_and_rate,
    init_sim,
)
from ..params_control import PARAMS_BY_NAME


ROOT = Path(__file__).resolve().parents[5]
GUI_ROOT = ROOT / "software" / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))
from analysis.param_sidecar import load_matching_sidecar  # noqa: E402
from analysis.wlog_metrics import decode_wlog  # noqa: E402


STATE_RUNNING = 3
_STATE_CHANNELS = (
    "pitch_rad", "pitch_rate_rads", "wheel_vel_avg", "yaw_rate_rads",
    "hip_l_pos_rad", "hip_r_pos_rad",
)
_CONTROL_CHANNELS = (
    "tau_sym", "tau_yaw", "hip_l_torque_nm", "hip_r_torque_nm",
)


def _nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = float(np.sqrt(np.mean(np.square(estimate - reference))))
    scale = float(np.percentile(reference, 95) - np.percentile(reference, 5))
    if scale < 1e-9:
        scale = max(float(np.sqrt(np.mean(np.square(reference)))), 1.0)
    return error / scale


def _rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(estimate - reference))))


def _wire_value(name: str, value: float) -> float:
    """Accept only float32 endpoint roundoff from a robot sidecar."""
    definition = PARAMS_BY_NAME[name]
    tolerance = 1e-6 * max(1.0, abs(definition.min), abs(definition.max))
    if value < definition.min and math.isclose(
            value, definition.min, rel_tol=0.0, abs_tol=tolerance):
        return definition.min
    if value > definition.max and math.isclose(
            value, definition.max, rel_tol=0.0, abs_tol=tolerance):
        return definition.max
    return float(value)


def _controller_timeline(run, sidecar) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Merge the immutable export with the log's recorded parameter history."""
    values = dict(DEFAULT_PARAMS.firmware_params)
    series: dict[str, np.ndarray] = {}
    if sidecar is None:
        return values, series
    for name in sidecar.names:
        if name not in PARAMS_BY_NAME:
            continue
        initial = sidecar.initial_value(name)
        if initial is not None:
            values[name] = _wire_value(name, initial)
        if "command" not in PARAMS_BY_NAME[name].flags:
            timeline = sidecar.series(name, run.t_micros)
            if timeline is not None:
                series[name] = np.asarray(timeline, dtype=np.float64)
    return values, series


def _running_windows(states: np.ndarray, sample_rate_hz: int,
                     window_s: float, max_windows: int | None) -> list[np.ndarray]:
    running = np.flatnonzero(np.asarray(states, dtype=np.int64) == STATE_RUNNING)
    if running.size == 0:
        return []
    split_at = np.flatnonzero(np.diff(running) != 1) + 1
    segments = np.split(running, split_at)
    width = max(2, int(round(window_s * sample_rate_hz)))
    windows: list[np.ndarray] = []
    for segment in segments:
        for offset in range(0, segment.size, width):
            window = segment[offset:offset + width]
            if window.size >= 2:
                windows.append(window)
                if max_windows is not None and len(windows) >= max_windows:
                    return windows
    return windows


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(np.asarray(values, dtype=np.float64),
                       np.asarray(times, dtype=np.float64), edge_order=1)


def _set_leg_configuration(model, data, robot, side: str, q_hip: float) -> None:
    ik = solve_ik(q_hip, robot.as_dict())
    if ik is None:
        raise RuntimeError(f"IK failed while replaying {side} hip at q={q_hip:.4f}")
    for joint_name, value in (
        (f"hinge_F_{side}", ik["q_coupler_F"]),
        (f"hip_{side}", ik["q_hip"]),
        (f"knee_joint_{side}", ik["q_knee"]),
    ):
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint]] = value


def _seed_controller(controller: SimController, fields: dict[str, np.ndarray],
                     index: int, q_cmd: float) -> None:
    firmware = controller.firmware_ctrl
    p = firmware.params
    v_ref = float(fields["v_ref"][index])
    vel = float(fields["wheel_vel_avg"][index])
    theta_ref = float(fields["theta_ref"][index])
    firmware.prev_v_desired = v_ref
    firmware.theta_ref_rlt = theta_ref
    firmware.hip_cmd_rlt = hip_q_to_alpha(q_cmd, controller.params.robot)
    if abs(p["vel_pi_ki"]) > 1e-12:
        estimate = (theta_ref - p["vel_pi_kp"] * (v_ref - vel)) / p["vel_pi_ki"]
        firmware.vel_integral = float(np.clip(
            estimate, -p["vel_pi_int_max"], p["vel_pi_int_max"]))
    yaw_error = (float(fields["omega_cmd_rds"][index])
                 - float(fields["yaw_rate_rads"][index]))
    if abs(p["yaw_pi_ki"]) > 1e-12:
        estimate = ((float(fields["tau_yaw"][index])
                     - p["yaw_pi_kp"] * yaw_error) / p["yaw_pi_ki"])
        firmware.yaw_integral = float(np.clip(
            estimate, -p["yaw_pi_int_max"], p["yaw_pi_int_max"]))


def _initialise_window(model, data, params, controller_values,
                       fields, hip_vel_l, hip_vel_r, index: int) -> SimController:
    robot = params.robot
    q_l = firmware_hip_to_sim_q(float(fields["hip_l_pos_rad"][index]), robot)
    q_r = firmware_hip_to_sim_q(float(fields["hip_r_pos_rad"][index]), robot)
    init_sim(model, data, params, q_hip_init=0.5 * (q_l + q_r))
    _set_leg_configuration(model, data, robot, "L", q_l)
    _set_leg_configuration(model, data, robot, "R", q_r)

    root_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    root_qpos = model.jnt_qposadr[root_joint]
    root_dof = model.jnt_dofadr[root_joint]
    roll = float(fields["roll_rad"][index])
    pitch = float(fields["pitch_rad"][index])
    yaw = float(fields["yaw_rad"][index])
    data.qpos[root_qpos + 3:root_qpos + 7] = _quat_from_rpy(roll, pitch, yaw)
    speed = float(fields["wheel_vel_avg"][index])
    data.qvel[root_dof + 0] = speed * math.cos(yaw)
    data.qvel[root_dof + 1] = speed * math.sin(yaw)
    data.qvel[root_dof + 3] = float(fields["roll_rate_rads"][index])
    data.qvel[root_dof + 4] = float(fields["pitch_rate_rads"][index])
    data.qvel[root_dof + 5] = float(fields["yaw_rate_rads"][index])

    for side, turns_s, hip_vel in (
        ("L", fields["wm_l_vel_turns_s"], hip_vel_l),
        ("R", fields["wm_r_vel_turns_s"], hip_vel_r),
    ):
        wheel_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"wheel_spin_{side}")
        hip_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_{side}")
        data.qvel[model.jnt_dofadr[wheel_joint]] = (
            float(turns_s[index]) * 2.0 * math.pi)
        data.qvel[model.jnt_dofadr[hip_joint]] = float(hip_vel[index])
    mujoco.mj_forward(model, data)

    controller = SimController(model, data, params, rng_seed=index)
    controller.firmware_ctrl.update_params(controller_values)
    q_cmd_l = firmware_hip_to_sim_q(
        float(fields["hip_l_cmd_pos_rad"][index]), robot)
    q_cmd_r = firmware_hip_to_sim_q(
        float(fields["hip_r_cmd_pos_rad"][index]), robot)
    _seed_controller(controller, fields, index, 0.5 * (q_cmd_l + q_cmd_r))
    sensor_seed = (
        pitch, float(fields["pitch_rate_rads"][index]),
        speed / robot.wheel_r, 9.81,
    )
    controller.sens_buf.reset(sensor_seed)
    controller.ctrl_buf.reset((
        float(fields["whl_tau_l"][index]),
        float(fields["whl_tau_r"][index]),
    ))
    controller._tau_hist = collections.deque(  # replay state seed
        [float(fields["tau_sym"][index])] * controller.n_sens,
        maxlen=max(controller.n_sens, 1),
    )
    controller.hip_reported_torque_L = float(fields["hip_l_torque_nm"][index])
    controller.hip_reported_torque_R = float(fields["hip_r_torque_nm"][index])
    return controller


def _update_recorded_params(controller: SimController,
                            timeline: dict[str, np.ndarray], index: int,
                            previous: dict[str, float]) -> None:
    updates = {}
    for name, values in timeline.items():
        value = _wire_value(name, float(values[index]))
        if previous.get(name) != value:
            updates[name] = value
            previous[name] = value
    if updates:
        controller.firmware_ctrl.update_params(updates)


def _open_loop_tick(model, data, controller: SimController,
                    fields, index: int) -> dict[str, float]:
    params = controller.params
    robot = params.robot
    p = controller.firmware_ctrl.params
    alpha = (float(fields["gain_sched_alpha"][index])
             if "gain_sched_alpha" in fields
             else hip_q_to_alpha(0.5 * (
                 data.qpos[controller.s_hip_L] + data.qpos[controller.s_hip_R]), robot))
    scale = params.motors.wheel.command_torque_scale
    data.ctrl[controller.act_wheel_L] = motor_taper(
        float(fields["whl_tau_l"][index]) * scale,
        data.qvel[controller.d_whl_L], controller.v_batt,
        params.motors, params.battery)
    data.ctrl[controller.act_wheel_R] = motor_taper(
        float(fields["whl_tau_r"][index]) * scale,
        data.qvel[controller.d_whl_R], controller.v_batt,
        params.motors, params.battery)

    tff = (p["hip_running_tff_ret"] + alpha
           * (p["hip_running_tff_ext"] - p["hip_running_tff_ret"]))
    q_cmd_l = firmware_hip_to_sim_q(
        float(fields["hip_l_cmd_pos_rad"][index]), robot)
    q_cmd_r = firmware_hip_to_sim_q(
        float(fields["hip_r_cmd_pos_rad"][index]), robot)
    reported_l = float(np.clip(
        p["hip_running_kp"] * (q_cmd_l - data.qpos[controller.s_hip_L])
        - p["hip_running_kd"] * data.qvel[controller.d_hip_L] + tff,
        -params.motors.hip.torque_limit, params.motors.hip.torque_limit))
    reported_r = float(np.clip(
        p["hip_running_kp"] * (q_cmd_r - data.qpos[controller.s_hip_R])
        - p["hip_running_kd"] * data.qvel[controller.d_hip_R] + tff,
        -params.motors.hip.torque_limit, params.motors.hip.torque_limit))
    hip_scale = params.motors.hip.torque_scale(alpha)
    data.ctrl[controller.act_hip_L] = reported_l * hip_scale
    data.ctrl[controller.act_hip_R] = reported_r * hip_scale
    pitch, pitch_rate = get_pitch_and_rate(
        data, controller.box_bid, controller.d_pitch)
    _, roll_rate = get_roll_and_rate(data, controller.box_bid, controller.d_roll)
    del roll_rate
    wheel_vel = 0.5 * (
        data.qvel[controller.d_whl_L] + data.qvel[controller.d_whl_R]) * robot.wheel_r
    return {
        "pitch_rad": float(pitch), "pitch_rate_rads": float(pitch_rate),
        "wheel_vel_avg": float(wheel_vel),
        "yaw_rate_rads": float(data.qvel[controller.d_yaw]),
        "hip_l_pos_rad": sim_q_to_firmware_hip(
            float(data.qpos[controller.s_hip_L]), robot),
        "hip_r_pos_rad": sim_q_to_firmware_hip(
            float(data.qpos[controller.s_hip_R]), robot),
        "hip_l_torque_nm": reported_l, "hip_r_torque_nm": reported_r,
    }


def _closed_loop_tick(model, data, controller: SimController,
                      fields, index: int) -> dict[str, float]:
    robot = controller.params.robot
    q_cmd_l = firmware_hip_to_sim_q(
        float(fields["hip_l_cmd_pos_rad"][index]), robot)
    q_cmd_r = firmware_hip_to_sim_q(
        float(fields["hip_r_cmd_pos_rad"][index]), robot)
    q_cmd = 0.5 * (q_cmd_l + q_cmd_r)
    # WLOG stores the post-slew MIT setpoint, not raw CH3.  Seed the slew state
    # to that observation so replay does not rate-limit an already limited signal.
    controller.firmware_ctrl.hip_cmd_rlt = hip_q_to_alpha(q_cmd, robot)
    p = controller.firmware_ctrl.params
    tick = controller.tick(
        model, data,
        v_target_ms=float(fields["v_ref"][index]),
        omega_target=float(fields["omega_cmd_rds"][index]),
        q_hip_target=q_cmd,
        use_lqr=True,
        use_velocity_pi=p["vel_pi_en"] >= 0.5,
        use_yaw_pi=p["yaw_pi_en"] >= 0.5,
        use_impedance=False,
        use_roll_leveling=p["roll_ctrl_en"] >= 0.5,
        use_suspension=True, use_ff1=True, use_ff2=True,
    )
    return {
        "pitch_rad": float(tick["pitch"]),
        "pitch_rate_rads": float(tick["pitch_rate"]),
        "wheel_vel_avg": float(tick["v_measured"]),
        "yaw_rate_rads": float(tick["yaw_rate"]),
        "hip_l_pos_rad": sim_q_to_firmware_hip(tick["hip_q_L"], robot),
        "hip_r_pos_rad": sim_q_to_firmware_hip(tick["hip_q_R"], robot),
        "tau_sym": float(tick["tau_sym"]), "tau_yaw": float(tick["tau_yaw"]),
        "hip_l_torque_nm": float(tick["tau_hip_L"]),
        "hip_r_torque_nm": float(tick["tau_hip_R"]),
    }


def replay(path: Path, *, mode: str = "closed", plant_params=None,
           output_csv: Path | None = None, window_s: float | None = None,
           max_windows: int | None = None) -> dict:
    """Run a windowed MuJoCo replay; ``plant_params`` is rejected deliberately."""
    if mode not in {"open", "closed"}:
        raise ValueError("mode must be 'open' or 'closed'")
    if plant_params is not None:
        raise ValueError("MuJoCo replay uses SimParams plant fields, not AnalyticalPlant")
    if window_s is None:
        window_s = 0.1 if mode == "open" else 2.0
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")

    path = Path(path)
    run = decode_wlog(path)
    fields = run.fields
    sidecar = load_matching_sidecar(path)
    controller_values, timeline = _controller_timeline(run, sidecar)
    params = replace(
        DEFAULT_PARAMS,
        firmware_params=tuple(sorted(controller_values.items())),
    )
    windows = _running_windows(
        fields["robot_state"], run.sample_rate_hz, window_s, max_windows)
    if not windows:
        raise ValueError(f"{path}: no contiguous RUNNING windows")

    hip_vel_l = _gradient(fields["hip_l_pos_rad"], run.t_s)
    hip_vel_r = _gradient(fields["hip_r_pos_rad"], run.t_s)
    score_channels = list(_STATE_CHANNELS)
    score_channels.extend(_CONTROL_CHANNELS if mode == "closed" else _CONTROL_CHANNELS[2:])
    reference = {name: [] for name in score_channels}
    predicted = {name: [] for name in score_channels}
    csv_rows = []
    path_errors = []
    requested = sum(max(0, window.size - 1) for window in windows)
    fallen_windows = 0
    model, data = build_model_and_data(params)

    for window_id, window in enumerate(windows):
        start = int(window[0])
        controller = _initialise_window(
            model, data, params, controller_values, fields,
            hip_vel_l, hip_vel_r, start)
        previous_params: dict[str, float] = {}
        sim_x0 = float(data.qpos[controller.s_root])
        sim_y0 = float(data.qpos[controller.s_root + 1])
        ref_x = ref_y = 0.0
        fell = False

        for local_index, raw_index in enumerate(window):
            index = int(raw_index)
            _update_recorded_params(controller, timeline, index, previous_params)
            values = (_closed_loop_tick(model, data, controller, fields, index)
                      if mode == "closed"
                      else _open_loop_tick(model, data, controller, fields, index))

            if local_index > 0:
                for name in score_channels:
                    reference[name].append(float(fields[name][index]))
                    predicted[name].append(float(values[name]))
                sim_x = float(data.qpos[controller.s_root]) - sim_x0
                sim_y = float(data.qpos[controller.s_root + 1]) - sim_y0
                path_errors.append(math.hypot(sim_x - ref_x, sim_y - ref_y))
                if output_csv is not None:
                    csv_rows.append([
                        window_id, float(run.t_s[index]), ref_x, ref_y, sim_x, sim_y,
                        *(float(fields[name][index]) for name in score_channels),
                        *(float(values[name]) for name in score_channels),
                    ])

            if local_index + 1 >= window.size:
                continue
            next_index = int(window[local_index + 1])
            dt = min(0.02, max(0.0002,
                float(run.t_s[next_index] - run.t_s[index])))
            yaw = float(fields["yaw_rad"][index])
            speed = float(fields["wheel_vel_avg"][index])
            ref_x += speed * math.cos(yaw) * dt
            ref_y += speed * math.sin(yaw) * dt
            steps = max(1, int(round(dt / model.opt.timestep)))
            for _ in range(steps):
                mujoco.mj_step(model, data)
            pitch, _ = get_pitch_and_rate(
                data, controller.box_bid, controller.d_pitch)
            if not np.isfinite(data.qpos).all() or abs(pitch) > params.thresholds.fall_rad:
                fallen_windows += 1
                fell = True
                break
        if fell:
            continue

    arrays_ref = {name: np.asarray(values, dtype=np.float64)
                  for name, values in reference.items()}
    arrays_pred = {name: np.asarray(values, dtype=np.float64)
                   for name, values in predicted.items()}
    if not arrays_ref or not next(iter(arrays_ref.values())).size:
        raise RuntimeError("replay produced no scored samples")
    nrmse = {name: _nrmse(arrays_ref[name], arrays_pred[name])
             for name in score_channels}
    rmse = {name: _rmse(arrays_ref[name], arrays_pred[name])
            for name in score_channels}

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "window", "log_time_s", "dead_reckoned_x_m", "dead_reckoned_y_m",
                "sim_x_m", "sim_y_m",
                *(f"recorded_{name}" for name in score_channels),
                *(f"sim_{name}" for name in score_channels),
            ])
            writer.writerows(csv_rows)

    scored = int(next(iter(arrays_ref.values())).size)
    return {
        "ok": True, "mode": mode, "plant": "mujoco",
        "wlog": str(path), "samples": run.count,
        "running_samples": int(np.count_nonzero(
            fields["robot_state"].astype(np.int64) == STATE_RUNNING)),
        "window_s": window_s, "window_count": len(windows),
        "fallen_windows": fallen_windows,
        "requested_scored_samples": requested, "scored_samples": scored,
        "coverage_fraction": scored / requested if requested else 0.0,
        "controller_parameter_count": len(controller_values),
        "controller_snapshot_sha256": control_snapshot_sha256(controller_values),
        "controller_source": (
            str(sidecar.path) if sidecar is not None
            else "robot-match locked export (no sidecar present)"),
        "controller_fit_allowed": False,
        "nrmse": nrmse, "rmse": rmse,
        "worst_nrmse": max(nrmse.values()),
        "dead_reckoned_xy_rmse_m": float(np.sqrt(np.mean(
            np.square(path_errors)))) if path_errors else None,
        "path_reference": (
            "wheel-speed/yaw dead reckoning only; WLOG has no global XY measurement"),
        "plant_parameters_are_provisional": True,
        **({"overlay_csv": str(output_csv)} if output_csv is not None else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wlog", type=Path)
    parser.add_argument(
        "--mode", choices=("open", "closed", "both"), default="closed")
    parser.add_argument(
        "--window-s", type=float,
        help="reset interval (default: 0.1 s open-loop, 2.0 s closed-loop)")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.mode == "both":
        if args.output_csv is not None:
            parser.error("--output-csv requires --mode open or --mode closed")
        report = {
            mode: replay(
                args.wlog, mode=mode, window_s=args.window_s,
                max_windows=args.max_windows)
            for mode in ("open", "closed")
        }
    else:
        report = replay(
            args.wlog, mode=args.mode, output_csv=args.output_csv,
            window_s=args.window_s, max_windows=args.max_windows)
    text = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
