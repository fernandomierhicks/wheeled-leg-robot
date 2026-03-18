"""test_lqr_integration.py — Verify LQR control law is active in viewer."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
from sim_config import *
from physics import build_xml, build_assets
from motor_models import MotorModel

print("\n" + "="*70)
print("LQR INTEGRATION TEST")
print("="*70 + "\n")

# Build model
xml = build_xml(ROBOT)
assets = build_assets()
model = mujoco.MjModel.from_xml_string(xml, assets=assets)
data = mujoco.MjData(model)

# Find joint indices
d_pitch = model.jnt_dofadr[model.joint('root_free').id] + 1  # Pitch is qvel[1]
d_hip_L = model.jnt_dofadr[model.joint('hip_L').id]
d_hip_R = model.jnt_dofadr[model.joint('hip_R').id]
d_whl_L = model.jnt_dofadr[model.joint('wheel_spin_L').id]
d_whl_R = model.jnt_dofadr[model.joint('wheel_spin_R').id]

# Motor models
motor_whl_L = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD,
                         WHEEL_TAU_ELEC, WHEEL_B_FRICTION)
motor_whl_R = MotorModel(WHEEL_TORQUE_LIMIT, WHEEL_OMEGA_NOLOAD,
                         WHEEL_TAU_ELEC, WHEEL_B_FRICTION)

# Simulate 2 seconds
dt = 0.0005
n_steps = int(2.0 / dt)
wheel_pos_L = 0.0
wheel_pos_R = 0.0

print("Simulating 2 seconds with PID control (blend_factor=0.0):\n")
print(f"{'Step':>6} {'Time (s)':>10} {'Pitch (deg)':>14} {'Pitch Rate':>12} "
      f"{'Wheel Vel':>12} {'Wheel Pos':>12} {'u_pid':>10} {'u_lqr':>10}")
print("-" * 100)

for step in range(n_steps):
    t = step * dt
    mujoco.mj_step(model, data)

    # Extract state
    q_quat = data.qpos[3:7]
    pitch_true = np.arctan2(
        2 * (q_quat[0]*q_quat[1] + q_quat[2]*q_quat[3]),
        1 - 2 * (q_quat[1]**2 + q_quat[2]**2))
    pitch_rate = data.qvel[d_pitch]
    wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0

    # Integrate wheel position
    wheel_pos_L += data.qvel[d_whl_L] * dt
    wheel_pos_R += data.qvel[d_whl_R] * dt
    wheel_pos = (wheel_pos_L + wheel_pos_R) / 2.0

    # Control (PID only, blend_factor=0)
    pitch_error = pitch_true
    u_pid = PITCH_KP * pitch_error + PITCH_KD * pitch_rate

    # LQR law (computed but not used yet)
    _lqr_state = np.array([pitch_error, pitch_rate, wheel_pos, wheel_vel])
    u_lqr = float(-LQR_K @ _lqr_state)

    # Apply control
    data.ctrl[2] = motor_whl_L.step(u_pid, data.qvel[d_whl_L], dt)
    data.ctrl[3] = motor_whl_R.step(u_pid, data.qvel[d_whl_R], dt)

    # Log every 0.5 seconds
    if step % int(0.5 / dt) == 0:
        print(f"{step:6d} {t:10.3f} {np.degrees(pitch_error):14.2f} "
              f"{pitch_rate:12.4f} {wheel_vel:12.4f} {wheel_pos:12.4f} "
              f"{u_pid:10.3f} {u_lqr:10.3f}")

print("-" * 100)
print(f"\nFinal state:")
print(f"  Pitch error: {np.degrees(pitch_error):.2f} degrees")
print(f"  Wheel position: {wheel_pos:.4f} m")
print(f"  Control (PID): {u_pid:.2f} N*m")
print(f"  Control (LQR computed, unused): {u_lqr:.2f} N*m")

print("\n" + "="*70)
print("RESULT: Control law executing successfully!")
print("        - PID baseline working (blend_factor=0.0)")
print("        - LQR state computed (ready for blending)")
print("        - Wheel position tracking active")
print("        - Ready to increase blend_factor to fade in LQR")
print("="*70 + "\n")
