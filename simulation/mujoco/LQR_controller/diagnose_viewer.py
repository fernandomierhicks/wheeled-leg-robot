"""diagnose_viewer.py — Minimal viewer with detailed error reporting."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import traceback
import mujoco
import mujoco.viewer
import numpy as np
from sim_config import *
from physics import build_xml, build_assets, solve_ik, SimParams
from motor_models import MotorModel

print("="*80)
print("MUJOCO VIEWER DIAGNOSTIC")
print("="*80)

try:
    print("\n[1/5] Building MuJoCo model...")
    p = SimParams()
    xml = build_xml(p)
    model = mujoco.MjModel.from_xml_string(xml, assets=build_assets())
    data = mujoco.MjData(model)
    print("      OK - Model built successfully")

except Exception as e:
    print(f"      FAILED - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/5] Finding body/joint indices...")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    whl_L_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "whl_L")
    whl_R_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "whl_R")
    hip_L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")
    print("      OK - All indices found")

except Exception as e:
    print(f"      FAILED - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[3/5] Setting up motor models and LQR gains...")
    from lqr_design import compute_lqr_gain

    motor_whl_L = MotorModel(T_peak=WHEEL_TORQUE_LIMIT, omega_noload=WHEEL_OMEGA_NOLOAD,
                             tau_elec=WHEEL_TAU_ELEC, B_friction=WHEEL_B_FRICTION)
    motor_whl_R = MotorModel(T_peak=WHEEL_TORQUE_LIMIT, omega_noload=WHEEL_OMEGA_NOLOAD,
                             tau_elec=WHEEL_TAU_ELEC, B_friction=WHEEL_B_FRICTION)

    # Compute K dynamically (like run_scenario_visual.py does)
    K_dyn = compute_lqr_gain(Q_pitch=100.0, R=0.1)
    print(f"      OK - Motors + LQR K = {K_dyn}")

except Exception as e:
    print(f"      FAILED - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[4/5] Proper initialization (IK + equilibrium pitch)...")
    from physics import get_equilibrium_pitch, solve_ik
    import math

    # Get joint indices
    s_hF_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_F_L")
    s_hF_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_F_R")
    s_knee_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint_L")
    s_knee_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint_R")
    s_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")

    wheel_bid_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_L")
    wheel_bid_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_asm_R")

    # Reset and do IK
    mujoco.mj_resetData(model, data)
    ik = solve_ik(Q_NEUTRAL, p)

    # Set leg joint angles
    data.qpos[s_hF_L] = ik['q_coupler_F']
    data.qpos[s_knee_L] = ik['q_knee']
    data.qpos[s_hF_R] = ik['q_coupler_F']
    data.qpos[s_knee_R] = ik['q_knee']
    data.qpos[hip_L_id] = ik['q_hip']
    hip_R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
    data.qpos[hip_R_id] = ik['q_hip']

    mujoco.mj_forward(model, data)

    # Position wheels on ground
    wz = (data.xpos[wheel_bid_L][2] + data.xpos[wheel_bid_R][2]) / 2.0
    data.qpos[s_root + 2] += WHEEL_R - wz

    # Set equilibrium pitch (quaternion)
    theta = get_equilibrium_pitch(p, Q_NEUTRAL)
    data.qpos[s_root + 3] = math.cos(theta / 2)
    data.qpos[s_root + 4] = 0.0
    data.qpos[s_root + 5] = math.sin(theta / 2)
    data.qpos[s_root + 6] = 0.0
    mujoco.mj_forward(model, data)

    print("      OK - Initialization complete")

    print("\n[5/5] Initializing MuJoCo viewer...")
    print("      (This is where crashes usually happen)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("      OK - Viewer launched successfully")

        print(f"\n[6/6] Running simulation loop...")
        print("      Starting 10-second test run")
        print("      (Press Ctrl+C to stop)\n")

        step_count = 0

        while viewer.is_running() and step_count < 10000:  # 10 seconds at 1000 Hz
            try:
                # Get state from quaternion (correct way)
                q_quat = data.xquat[body_id]
                pitch = np.arcsin(np.clip(2 * (q_quat[0]*q_quat[2] - q_quat[3]*q_quat[1]), -1, 1))
                d_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_DOF, "root_free") + 4
                pitch_rate = data.qvel[d_pitch]
                wheel_vel_L = data.qvel[whl_L_jid]
                wheel_vel_R = data.qvel[whl_R_jid]
                wheel_vel_avg = (wheel_vel_L + wheel_vel_R) / 2.0

                # LQR control (using dynamically computed K)
                lqr_state = np.array([pitch, pitch_rate, wheel_vel_avg])
                torque_cmd = float(-K_dyn @ lqr_state)

                # Apply torques
                tau_L = motor_whl_L.step(torque_cmd, wheel_vel_L, model.opt.timestep)
                tau_R = motor_whl_R.step(torque_cmd, wheel_vel_R, model.opt.timestep)
                data.ctrl[2] = tau_L
                data.ctrl[3] = tau_R

                # Step
                mujoco.mj_step(model, data)
                viewer.sync()

                # Progress
                if step_count % 1000 == 0:
                    sim_time = data.time
                    print(f"      t={sim_time:.2f}s - pitch={np.degrees(pitch):6.2f}° u={torque_cmd:6.3f}Nm [OK]")

                step_count += 1

            except Exception as e:
                print(f"\n      ERROR at step {step_count} (t={data.time:.2f}s):")
                print(f"      {e}")
                traceback.print_exc()
                break

        print(f"\n      Completed {step_count} steps (t={data.time:.2f}s)")
        print("      VIEWER RAN SUCCESSFULLY")

except KeyboardInterrupt:
    print("\n      User interrupted (Ctrl+C)")

except Exception as e:
    print(f"\n      VIEWER CRASHED: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
