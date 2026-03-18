"""run_scenario_visual.py — Visual replay of behavioral test scenarios in MuJoCo viewer.

Usage:
    python run_scenario_visual.py --scenario self_balance --q 100 --r 0.1 --blend 1.0
    python run_scenario_visual.py --scenario self_balance --eval-id 216  # Pull Q, R from log
    python run_scenario_visual.py --scenario self_balance --eval-id best  # Pull best from log

Displays the scenario in real-time with MuJoCo viewer.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim_config import *
from physics import solve_ik, get_equilibrium_pitch, build_xml, build_assets
from motor_models import MotorModel
from lqr_design import compute_lqr_gain


def load_params_from_log(eval_id, log_file="optimization_log.csv"):
    """
    Load Q_pitch and R from optimization log by eval_id.

    Args:
        eval_id: Integer eval ID, or "best" for best fitness
        log_file: Path to optimization_log.csv

    Returns:
        Tuple (Q_pitch, R, eval_info_dict)
    """
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Log file is empty")

    if eval_id == "best":
        # Find row with minimum fitness
        best_row = min(rows, key=lambda r: float(r['fitness']))
        eval_id_str = best_row['eval_id']
    else:
        # Find row with matching eval_id
        eval_id_str = str(eval_id)
        best_row = None
        for row in rows:
            if row['eval_id'] == eval_id_str:
                best_row = row
                break
        if best_row is None:
            raise ValueError(f"eval_id {eval_id} not found in log")

    Q_pitch = float(best_row['Q_pitch'])
    R = float(best_row['R'])
    fitness = float(best_row['fitness'])

    info = {
        'eval_id': eval_id_str,
        'Q_pitch': Q_pitch,
        'R': R,
        'fitness': fitness,
        'rms_pitch_deg': float(best_row.get('rms_pitch_wobble_deg', 0)),
        'max_wheel_pos_drift_m': float(best_row.get('max_wheel_pos_drift_m', 0)),
    }

    return Q_pitch, R, info


def get_wheel_vel_target(sim_t):
    """Return target wheel velocity for drive scenario."""
    V_MAX = 1.0  # m/s
    if sim_t < 5.0:
        # Ramp forward: 0 → 1.0 m/s
        return (sim_t / 5.0) * V_MAX
    elif sim_t < 10.0:
        # Hold forward at 1.0 m/s
        return V_MAX
    elif sim_t < 15.0:
        # Ramp reverse: 1.0 → -1.0 m/s
        return V_MAX - ((sim_t - 10.0) / 5.0) * 2.0 * V_MAX
    else:
        # Hold backward
        return -V_MAX


def run_scenario_visual(scenario="self_balance", q_pitch=100.0, r_val=0.1, blend=1.0, with_obstacle=False):
    """
    Run a scenario with MuJoCo viewer.

    Args:
        scenario: Scenario name ('self_balance' or 'drive')
        q_pitch: LQR Q[0,0]
        r_val: LQR R
        blend: LQR blend factor
        with_obstacle: Add obstacle for drive scenario
    """
    if scenario == "self_balance":
        duration_s = 30.0
    elif scenario == "drive":
        duration_s = 20.0
    else:
        raise ValueError(f"Scenario '{scenario}' not implemented. Use 'self_balance' or 'drive'.")
    K = compute_lqr_gain(Q_pitch=q_pitch, R=r_val)

    # Build physics
    p = ROBOT
    extra_ramps = []
    if scenario == "drive" or with_obstacle:
        # 2cm obstacle at x=2.0m for drive scenario
        extra_ramps = [(2.0, 0.0, 0.05, 0.25, 0.01, 0, "0.6 0.4 0.2")]
    xml = build_xml(p, extra_ramps=extra_ramps)
    model = mujoco.MjModel.from_xml_string(xml, assets=build_assets())
    data = mujoco.MjData(model)

    # Fix 4-bar constraints
    for eq_name in ["4bar_close_L", "4bar_close_R"]:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        model.eq_data[eq_id, 0:3] = [-p['Lc'], 0.0, 0.0]
        model.eq_data[eq_id, 3:6] = [0.0, 0.0, p['L_stub']]

    # Helper functions
    def jqp(n):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def jdof(n):
        return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    s_root = jqp("root_free")
    s_hip_L = jqp("hip_L")
    s_hip_R = jqp("hip_R")
    s_hF_L = jqp("hinge_F_L")
    s_hF_R = jqp("hinge_F_R")
    s_knee_L = jqp("knee_joint_L")
    s_knee_R = jqp("knee_joint_R")
    d_root = jdof("root_free")
    d_pitch = d_root + 4
    d_hip_L = jdof("hip_L")
    d_hip_R = jdof("hip_R")
    d_whl_L = jdof("wheel_spin_L")
    d_whl_R = jdof("wheel_spin_R")

    box_bid = bid("box")
    wheel_bid_L = bid("wheel_asm_L")
    wheel_bid_R = bid("wheel_asm_R")

    # Motors
    motor_hip_L = MotorModel(HIP_TORQUE_LIMIT, OMEGA_MAX, HIP_TAU_ELEC, HIP_B_FRICTION)
    motor_hip_R = MotorModel(HIP_TORQUE_LIMIT, OMEGA_MAX, HIP_TAU_ELEC, HIP_B_FRICTION)
    motor_whl_L = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_TAU_ELEC, WHEEL_B_FRICTION)
    motor_whl_R = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD, WHEEL_TAU_ELEC, WHEEL_B_FRICTION)

    def _init():
        """Initialize to neutral stance."""
        mujoco.mj_resetData(model, data)
        ik = solve_ik(Q_NEUTRAL, p)
        if not ik:
            raise RuntimeError("IK failed at Q_NEUTRAL")
        for s_hF, s_hip, s_knee in [
            (s_hF_L, s_hip_L, s_knee_L),
            (s_hF_R, s_hip_R, s_knee_R),
        ]:
            data.qpos[s_hF] = ik['q_coupler_F']
            data.qpos[s_hip] = ik['q_hip']
            data.qpos[s_knee] = ik['q_knee']
        mujoco.mj_forward(model, data)
        wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
        data.qpos[s_root + 2] += WHEEL_R - wz
        theta = get_equilibrium_pitch(p, Q_NEUTRAL)
        data.qpos[s_root + 3] = math.cos(theta / 2)
        data.qpos[s_root + 4] = 0.0
        data.qpos[s_root + 5] = math.sin(theta / 2)
        data.qpos[s_root + 6] = 0.0
        mujoco.mj_forward(model, data)

    _init()

    # State
    wheel_pos_L = 0.0
    wheel_pos_R = 0.0
    pitch_integral = 0.0

    print(f"\n{'='*70}")
    scenario_label = "Self-Balance (30s)" if scenario == "self_balance" else "Drive (20s)"
    print(f"SCENARIO: {scenario_label}")
    print(f"Q[0,0]={q_pitch:.1f}  R={r_val:.2f}  LQR Blend={blend:.2f}")
    if scenario == "drive" and with_obstacle:
        print(f"Obstacle: 2cm tall at x=2.0m")
    print(f"{'='*70}\n")
    print("Running in MuJoCo viewer...")
    print("Close the viewer window to finish.\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 35
        viewer.cam.elevation = -15
        viewer.cam.distance = 2.5
        viewer.cam.lookat = np.array([0.0, 0.0, 0.30])

        while viewer.is_running() and data.time < duration_s:
            _dt = model.opt.timestep

            # State extraction
            q_quat = data.xquat[box_bid]
            pitch_true = math.asin(max(-1.0, min(1.0,
                2 * (q_quat[0]*q_quat[2] - q_quat[3]*q_quat[1]))))
            pitch_rate_true = data.qvel[d_pitch]
            pitch = pitch_true + np.random.normal(0, PITCH_NOISE_STD_RAD)
            pitch_rate = pitch_rate_true + np.random.normal(0, PITCH_RATE_NOISE_STD_RAD_S)

            hip_q = (data.qpos[s_hip_L] + data.qpos[s_hip_R]) / 2.0
            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0

            # Integrate wheel position
            wheel_pos_L += data.qvel[d_whl_L] * _dt
            wheel_pos_R += data.qvel[d_whl_R] * _dt
            wheel_pos = (wheel_pos_L + wheel_pos_R) / 2.0

            # Equilibrium pitch
            pitch_ff = get_equilibrium_pitch(p, hip_q)
            target_pitch = pitch_ff
            pitch_error = pitch - target_pitch
            pitch_integral = np.clip(pitch_integral + pitch_error * _dt, -1.0, 1.0)

            # PID
            u_pid = PITCH_KP * pitch_error + PITCH_KI * pitch_integral + PITCH_KD * pitch_rate

            # LQR (3-STATE: [pitch_err, pitch_rate, wheel_vel])
            # wheel_pos removed — position tracking handled by outer Velocity PI loop
            if scenario == "drive":
                # Drive: damp pitch rate only, allow pitch to vary for velocity tracking
                _lqr_state = np.array([0.0, pitch_rate, wheel_vel])
            else:
                # Self-balance: pitch + rate + velocity
                _lqr_state = np.array([pitch_error, pitch_rate, wheel_vel])
            u_lqr = float(-K @ _lqr_state)

            # Blend
            u_bal = (1.0 - blend) * u_pid + blend * u_lqr

            # Scenario-specific control
            if scenario == "drive":
                # Drive scenario: add velocity error compensation
                wheel_vel_target = get_wheel_vel_target(data.time)
                wheel_vel_linear = wheel_vel * WHEEL_R  # Convert rad/s to m/s
                wheel_vel_error = wheel_vel_target - wheel_vel_linear  # Target - actual
                u_vel = 1.5 * wheel_vel_error  # Positive error → more torque
                u_wheel_total = u_bal + u_vel
            else:
                # Self-balance: use balance control only
                u_wheel_total = u_bal

            # Motors
            u_whl_L = motor_whl_L.step(u_wheel_total, data.qvel[d_whl_L], _dt)
            u_whl_R = motor_whl_R.step(u_wheel_total, data.qvel[d_whl_R], _dt)
            data.ctrl[2] = u_whl_L
            data.ctrl[3] = u_whl_R

            # Hip suspension
            u_hip = HIP_KP_SUSP * (Q_NEUTRAL - hip_q) + HIP_KD_SUSP * (0.0 - data.qvel[d_hip_L])
            u_hip = np.clip(u_hip, -HIP_TORQUE_LIMIT, HIP_TORQUE_LIMIT)
            data.ctrl[0] = motor_hip_L.step(u_hip, data.qvel[d_hip_L], _dt)
            data.ctrl[1] = motor_hip_R.step(u_hip, data.qvel[d_hip_R], _dt)

            # Step
            mujoco.mj_step(model, data)

            # Print status every ~1 second
            if int(data.time * 10) % 10 == 0:
                pitch_deg = math.degrees(pitch)
                print(f"t={data.time:6.2f}s | pitch={pitch_deg:6.2f}° | "
                      f"wheel_pos={wheel_pos:7.4f}m | u={u_bal:6.2f}Nm")

            viewer.sync()

    print(f"\nScenario complete at t={data.time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Visual scenario replay")
    parser.add_argument("--scenario", type=str, default="self_balance",
                        help="Scenario name (self_balance)")
    parser.add_argument("--q", type=float, default=None, help="LQR Q[0,0]")
    parser.add_argument("--r", type=float, default=None, help="LQR R")
    parser.add_argument("--blend", type=float, default=1.0, help="LQR blend (0=PID, 1=LQR)")
    parser.add_argument("--eval-id", type=str, default=None,
                        help="Load Q, R from optimization log by eval_id (or 'best')")
    parser.add_argument("--log", type=str, default="optimization_log.csv",
                        help="Path to optimization log (default: optimization_log.csv)")
    parser.add_argument("--with-obstacle", action="store_true",
                        help="Add 2cm obstacle at 2m for drive scenario")
    args = parser.parse_args()

    try:
        # Load from log if eval-id specified
        if args.eval_id:
            q_pitch, r_val, info = load_params_from_log(args.eval_id, args.log)
            print(f"\nLoaded from log: eval_id={info['eval_id']}")
            print(f"  Q[0,0]={q_pitch:.2f}  R={r_val:.4f}  fitness={info['fitness']:.4f}")
            print(f"  Pitch wobble: {info['rms_pitch_deg']:.3f}°  Drift: {info['max_wheel_pos_drift_m']*1000:.1f}mm\n")
        else:
            # Use command-line args
            q_pitch = args.q if args.q is not None else 100.0
            r_val = args.r if args.r is not None else 0.1

        run_scenario_visual(scenario=args.scenario, q_pitch=q_pitch, r_val=r_val, blend=args.blend, with_obstacle=args.with_obstacle)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
