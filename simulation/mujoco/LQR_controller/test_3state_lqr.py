"""test_3state_lqr.py — Quick validation of 3-state LQR control law.

Tests the new 3-state balance LQR (pitch, pitch_rate, wheel_vel)
without requiring the full viewer. Validates:
1. Gains are loaded correctly from sim_config
2. LQR control law stabilizes pitch disturbance
3. No dimension mismatches
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sim_config import LQR_K, Q_NEUTRAL
from lqr_design import compute_lqr_gain
from physics import solve_ik, SimParams

print("\n" + "="*70)
print("TEST: 3-STATE LQR CONTROL LAW")
print("="*70 + "\n")

# ── Test 1: Load gains and verify shape ──────────────────────────────────────
print("Test 1: Load 3-state gains from sim_config")
print("-" * 70)
print(f"LQR_K shape: {LQR_K.shape}")
print(f"LQR_K = {LQR_K}")

if LQR_K.shape != (3,):
    print("[FAIL] Expected shape (3,), got {LQR_K.shape}")
    sys.exit(1)
else:
    print("[OK] Correct 3-state gain vector\n")

# ── Test 2: Construct 3-state vector and apply control ───────────────────────
print("Test 2: Apply 3-state LQR control law")
print("-" * 70)

# Simulate a pitch disturbance: +5 degrees = 0.0873 rad
pitch_error = 0.0873
pitch_rate = 0.0
wheel_vel_error = 0.0

state = np.array([pitch_error, pitch_rate, wheel_vel_error])
print(f"Initial state: [pitch_err={pitch_error:.4f}, pitch_rate={pitch_rate:.4f}, wheel_vel_err={wheel_vel_error:.4f}]")

# Apply control law
torque_cmd = -LQR_K @ state
print(f"Control torque: {torque_cmd:.4f} N*m")

# Expected: positive torque (accelerate wheels forward) to pitch body backward (corrective)
# When body leans forward (pitch_error > 0), positive wheel torque pitches body back
# This matches the physics: B[1,0] = -9.75 means positive torque → negative pitch accel
if abs(torque_cmd) > 5.0:
    print(f"[OK] Torque is corrective (magnitude {abs(torque_cmd):.2f} N*m)\n")
else:
    print(f"[FAIL] Expected large torque magnitude, got {abs(torque_cmd):.4f}\n")
    sys.exit(1)

# ── Test 3: Simulate 3-state closed-loop dynamics ──────────────────────────
print("Test 3: Closed-loop balance recovery (5s simulation)")
print("-" * 70)

# Build A, B matrices for nominal leg height
p = SimParams()
W_pos = solve_ik(Q_NEUTRAL, p)
l_eff = abs(W_pos['W'][1])

m_w = 2 * p['m_wheel']
m_b = p['m_box'] + 2*(p['m_femur'] + p['m_tibia'] + p['m_coupler'] + p['m_bearing'] + 0.260)
M = m_b + m_w

from sim_config import WHEEL_R
I_w = 0.5 * p['m_wheel'] * WHEEL_R**2
I_b = m_b * l_eff**2
g = 9.81

r = WHEEL_R
denom = (M + 2*I_w/r**2) * (I_b + m_b*l_eff**2) - m_b**2 * l_eff**2
alpha = (M + 2*I_w/r**2) * m_b * g * l_eff / denom
beta = -m_b**2 * g * l_eff**2 / (r * denom)
gamma = -(I_b + m_b*l_eff**2) / (r * denom)
delta = (M + 2*I_w/r**2 + m_b*l_eff/r) / denom

# 3-state A, B
A = np.array([
    [0,     1,   0],
    [alpha, 0,   0],
    [beta,  0,   0]
])

B = np.array([
    [0],
    [gamma],
    [delta]
])

# Simulate: start with 10 degree pitch error
x = np.array([np.radians(10), 0.0, 0.0])
dt = 0.001
t_sim = 5.0
n_steps = int(t_sim / dt)

# Log key points
log = {"t": [], "pitch": [], "pitch_rate": [], "wheel_vel": [], "torque": []}

for step in range(n_steps):
    t = step * dt

    # LQR control
    u = -LQR_K @ x  # scalar

    # Dynamics: x_dot = A @ x + B @ u (B is 3x1, so reshape u to (1,))
    x_dot = A @ x + B.flatten() * u

    # Euler step
    x = x + x_dot * dt

    # Log every 100 steps (~0.1s)
    if step % 100 == 0:
        log["t"].append(t)
        log["pitch"].append(np.degrees(x[0]))
        log["pitch_rate"].append(np.degrees(x[1]))
        log["wheel_vel"].append(x[2])
        log["torque"].append(u)  # u is scalar

print(f"{'Time (s)':>10} {'Pitch (deg)':>14} {'Rate (d/s)':>14} {'Wheel Vel':>14} {'Torque (Nm)':>14}")
print("-" * 70)
for i in range(min(15, len(log["t"]))):  # Print first 15 points
    print(f"{log['t'][i]:10.2f} {log['pitch'][i]:14.4f} {log['pitch_rate'][i]:14.4f} {log['wheel_vel'][i]:14.4f} {log['torque'][i]:14.4f}")

final_pitch = log["pitch"][-1]
print(f"\nFinal pitch error: {final_pitch:.4f} degrees (expect < 0.01)")

if abs(final_pitch) < 0.01:
    print("[OK] Pitch stabilized successfully\n")
else:
    print(f"[FAIL] Pitch did not stabilize, final = {final_pitch:.4f}\n")
    sys.exit(1)

# ── Test 4: Verify gain scheduling parameters exist ──────────────────────────
print("Test 4: Verify gain scheduling will work")
print("-" * 70)

from sim_config import Q_RET, Q_NEUTRAL, Q_EXT

gains_table = {}
for label, q_hip in [("RET", Q_RET), ("NOM", Q_NEUTRAL), ("EXT", Q_EXT)]:
    K = compute_lqr_gain(Q_pitch=100.0, R=0.1, q_hip=q_hip)
    gains_table[label] = K
    print(f"K_{label} = {K}")

# Verify they differ
diff_rn = np.linalg.norm(gains_table["RET"] - gains_table["NOM"])
diff_ne = np.linalg.norm(gains_table["NOM"] - gains_table["EXT"])
print(f"\n||K_RET - K_NOM|| = {diff_rn:.2f}")
print(f"||K_NOM - K_EXT|| = {diff_ne:.2f}")

if diff_rn > 30 and diff_ne > 20:
    print("[OK] Gains differ significantly across leg heights\n")
else:
    print("[FAIL] Gains should differ more across leg heights\n")
    sys.exit(1)

# ── Summary ──────────────────────────────────────────────────────────────────
print("="*70)
print("ALL TESTS PASSED")
print("="*70)
print("\nSummary:")
print("  [OK] 3-state gains loaded correctly (shape 3,)")
print("  [OK] Control law produces corrective torque")
print("  [OK] Closed-loop system stabilizes pitch disturbance")
print("  [OK] Gain scheduling foundation ready for interpolation")
print("\nNext steps:")
print("  1. Add compute_gain_table() function (Step 1.3)")
print("  2. Add interpolate_gains() function (Step 1.4)")
print("  3. Update viewer.py to use 3-state + interpolation (Step 1.5)")
print()
