"""lqr_parameter_sweep.py — Systematic LQR tuning via Q/R parameter sweep.

Tests a grid of Q and R values to find optimal LQR gains.
Logs performance metrics: settling time, overshoot, control effort, stability.

Metrics:
  - settling_time: time for pitch to settle within ±1° (seconds)
  - overshoot_deg: max pitch deviation from zero (degrees)
  - control_effort: integral of u^2 over simulation (N²·m²·s)
  - peak_control: maximum |u| (N·m)
  - damp_ratio: damping ratio from closed-loop eigenvalues (0=undamped, 1=critical)
  - phase_margin: phase margin at unity gain (degrees)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco
import csv
from scipy.linalg import solve_continuous_are
from sim_config import ROBOT, WHEEL_R, Q_NEUTRAL
from physics import build_xml, build_assets, solve_ik, SimParams
from motor_models import MotorModel

print("\n" + "="*80)
print("LQR PARAMETER SWEEP — Systematic Tuning")
print("="*80 + "\n")

# ── Simulation parameters ─────────────────────────────────────────────────────
SIM_TIME = 5.0          # Run each test for 5 seconds
DT = 0.0005             # MuJoCo timestep
N_STEPS = int(SIM_TIME / DT)

# ── Q and R search ranges ─────────────────────────────────────────────────────
# Q = diag([Q_pitch, 1.0, 1.0, 0.1]) — only Q[0,0] varies
# R = [[R_control]] — control cost varies
Q_PITCH_VALUES = [50, 75, 100, 150, 200]
R_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]

print(f"Search space: {len(Q_PITCH_VALUES)} × {len(R_VALUES)} = {len(Q_PITCH_VALUES) * len(R_VALUES)} configurations\n")

# ── Build model once ──────────────────────────────────────────────────────────
xml = build_xml(ROBOT)
assets = build_assets()
model = mujoco.MjModel.from_xml_string(xml, assets=assets)

# Find indices
d_pitch = model.jnt_dofadr[model.joint('root_free').id] + 1
d_hip_L = model.jnt_dofadr[model.joint('hip_L').id]
d_hip_R = model.jnt_dofadr[model.joint('hip_R').id]
d_whl_L = model.jnt_dofadr[model.joint('wheel_spin_L').id]
d_whl_R = model.jnt_dofadr[model.joint('wheel_spin_R').id]
s_hip_L = model.jnt_qposadr[model.joint('hip_L').id]
s_hip_R = model.jnt_qposadr[model.joint('hip_R').id]

# ── Compute linearized dynamics (same for all Q/R) ───────────────────────────
p = SimParams()
W_pos = solve_ik(Q_NEUTRAL, p)
W_x, W_z = W_pos['W']
l_eff = abs(W_z)

m_w = 2 * p['m_wheel']
m_b = p['m_box'] + 2 * (p['m_femur'] + p['m_tibia'] + p['m_coupler'] + p['m_bearing'] + 0.260)
M = m_b + m_w
I_w = 0.5 * p['m_wheel'] * WHEEL_R**2
I_b = m_b * l_eff**2
g = 9.81

r = WHEEL_R
denom = (M + 2*I_w/r**2) * (I_b + m_b*l_eff**2) - m_b**2 * l_eff**2
alpha = (M + 2*I_w/r**2) * m_b * g * l_eff / denom
beta = -m_b**2 * g * l_eff**2 / (r * denom)
gamma = -(I_b + m_b*l_eff**2) / (r * denom)
delta = (M + 2*I_w/r**2 + m_b*l_eff/r) / denom

A = np.array([[0, 1, 0, 0], [alpha, 0, 0, 0], [0, 0, 0, 1], [beta, 0, 0, 0]])
B = np.array([[0], [gamma], [0], [delta]])

print(f"System matrices computed:")
print(f"  A (4x4): alpha={alpha:.2f}, beta={beta:.2f}")
print(f"  B (4x1): gamma={gamma:.2f}, delta={delta:.2f}")
print(f"  l_eff={l_eff:.4f} m, M={M:.3f} kg\n")

# ── Results storage ───────────────────────────────────────────────────────────
results = []

# ── Parameter sweep ──────────────────────────────────────────────────────────
print(f"{'Q[0,0]':>8} {'R':>8} {'Settle(s)':>10} {'Overshoot(deg)':>12} {'Ctrl Eff':>12} {'Peak u':>8} {'Damp':>6} {'Notes':>20}")
print("-" * 100)

test_count = 0
for Q_pitch in Q_PITCH_VALUES:
    for R_val in R_VALUES:
        test_count += 1

        # Solve CARE for this Q/R pair
        Q = np.diag([Q_pitch, 1.0, 1.0, 0.1])
        R = np.array([[R_val]])
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
        except Exception as e:
            print(f"{Q_pitch:8.1f} {R_val:8.4f} | CARE FAILED: {str(e)[:30]}")
            continue

        # Check stability
        eigs = np.linalg.eigvals(A - B @ K)
        if not all(e.real < 0 for e in eigs):
            print(f"{Q_pitch:8.1f} {R_val:8.4f} | UNSTABLE (eigenvalues have positive real part)")
            continue

        # Compute metrics
        damp_ratio = min([-e.real / abs(e) for e in eigs if abs(e) > 1e-6])
        damp_ratio = max(0, min(1, damp_ratio))  # Clip to [0,1]

        # Simulate with initial pitch error
        data = mujoco.MjData(model)
        data.qpos[3:7] = [1, 0, 0, 0]  # Identity quat initially

        # Set initial pitch error by tilting body
        pitch_init = 0.05  # 5° initial error
        c, s = np.cos(pitch_init / 2), np.sin(pitch_init / 2)
        data.qpos[3:7] = [c, s, 0, 0]  # Perturbed quaternion

        wheel_pos_L = 0.0
        wheel_pos_R = 0.0
        pitch_log = []
        u_log = []
        settle_time = None
        overshoot = 0.0

        for step in range(N_STEPS):
            t = step * DT

            # Extract state
            q_quat = data.qpos[3:7]
            pitch = np.arctan2(2 * (q_quat[0]*q_quat[1] + q_quat[2]*q_quat[3]),
                              1 - 2 * (q_quat[1]**2 + q_quat[2]**2))
            pitch_rate = data.qvel[d_pitch]
            wheel_vel = (data.qvel[d_whl_L] + data.qvel[d_whl_R]) / 2.0
            wheel_pos_L += data.qvel[d_whl_L] * DT
            wheel_pos_R += data.qvel[d_whl_R] * DT
            wheel_pos = (wheel_pos_L + wheel_pos_R) / 2.0

            # LQR control
            lqr_state = np.array([pitch, pitch_rate, wheel_pos, wheel_vel])
            u = float(-K @ lqr_state)

            # Apply control
            motor_L = MotorModel(3.67, 326.7, 0.002, 0.02)
            motor_R = MotorModel(3.67, 326.7, 0.002, 0.02)
            data.ctrl[2] = motor_L.step(u, data.qvel[d_whl_L], DT)
            data.ctrl[3] = motor_R.step(u, data.qvel[d_whl_R], DT)

            # Step simulation
            mujoco.mj_step(model, data)

            # Logging
            pitch_log.append(np.degrees(pitch))
            u_log.append(u)

            # Detect settling time (pitch within ±1°)
            if settle_time is None and t > 0.5 and abs(pitch) < np.radians(1.0):
                if all(abs(p) < 1.0 for p in pitch_log[-int(0.5/DT):]):  # Sustained for 0.5s
                    settle_time = t

        # Post-process metrics
        pitch_log = np.array(pitch_log)
        u_log = np.array(u_log)

        overshoot = np.max(np.abs(pitch_log))
        settle_time = settle_time if settle_time is not None else SIM_TIME
        control_effort = np.trapz(u_log**2, dx=DT)
        peak_control = np.max(np.abs(u_log))

        # Store results
        results.append({
            'Q_pitch': Q_pitch,
            'R': R_val,
            'settle_time': settle_time,
            'overshoot': overshoot,
            'control_effort': control_effort,
            'peak_control': peak_control,
            'damp_ratio': damp_ratio,
        })

        # Print
        print(f"{Q_pitch:8.1f} {R_val:8.4f} {settle_time:10.3f} {overshoot:12.2f} {control_effort:12.2f} {peak_control:8.2f} {damp_ratio:6.3f}")

print("-" * 100)
print(f"\nCompleted {len(results)} of {len(Q_PITCH_VALUES) * len(R_VALUES)} tests\n")

# ── Find best configurations ──────────────────────────────────────────────────
if results:
    print("="*80)
    print("TOP 5 CONFIGURATIONS (ranked by settling time, then overshoot)")
    print("="*80 + "\n")

    # Sort by settling time, then overshoot
    sorted_results = sorted(results, key=lambda r: (r['settle_time'], r['overshoot']))

    print(f"{'Rank':>4} {'Q[0,0]':>8} {'R':>8} {'Settle(s)':>10} {'Overshoot(deg)':>12} {'Ctrl Eff':>12}")
    print("-" * 65)
    for i, res in enumerate(sorted_results[:5], 1):
        print(f"{i:4d} {res['Q_pitch']:8.1f} {res['R']:8.4f} {res['settle_time']:10.3f} "
              f"{res['overshoot']:12.2f} {res['control_effort']:12.2f}")

    best = sorted_results[0]
    print(f"\n{'RECOMMENDED PARAMETERS':>30}")
    print(f"  Q[0,0] = {best['Q_pitch']:.1f}")
    print(f"  R      = {best['R']:.4f}")
    print(f"  > Settling time: {best['settle_time']:.3f} s")
    print(f"  > Overshoot: {best['overshoot']:.2f} deg\n")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), 'lqr_sweep_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}\n")

    # ── Generate recommended sim_config update ────────────────────────────────
    print("="*80)
    print("TO USE THESE GAINS, UPDATE sim_config.py:")
    print("="*80 + "\n")

    Q_best = best['Q_pitch']
    R_best = best['R']
    Q_rec = np.diag([Q_best, 1.0, 1.0, 0.1])
    R_rec = np.array([[R_best]])
    P_best = solve_continuous_are(A, B, Q_rec, R_rec)
    K_best = np.linalg.inv(R_rec) @ B.T @ P_best

    print(f"# In sim_config.py, replace LQR section with:")
    print(f"LQR_K      = np.array({K_best[0].tolist()})")
    print(f"LQR_Q_DIAG = [100.0, 1.0, 1.0, 0.1]  # With Q[0,0] = {Q_best:.1f}")
    print(f"LQR_R_VAL  = {R_best:.4f}\n")

else:
    print("No stable configurations found. Try broader search ranges.")

print("="*80)
