"""
Step07 4-bar mechanism — MuJoCo simulation.

Uses chain topology (ground→femur→tibia→coupler, connect closes loop).
Link dimensions from simulation/2d/step07_4bar_proposed.py.

Controls: Ctrl + right-click drag on any link to apply force.
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
from scipy.optimize import brentq

# ── Step07 geometry ────────────────────────────────────────────────────────
L_femur = 0.100   # A→C  [m]
L_stub  = 0.015   # C→E  [m]
L_tibia = 0.115   # C→W  [m]
Lc      = 0.110   # E→F  [m]
F_X     = 0.015   # F relative to A, X component
F_Z     = 0.026   # F relative to A, Z component
HIP_INIT_DEG = 25.0

# ── Step07 FK (matches step07_4bar_proposed.py exactly) ───────────────────
def step07_fk(hip_deg):
    """Returns C, E, W, phi in step07 2D coords (x=forward, y=up, A at origin)."""
    th = np.radians(hip_deg)
    C  = np.array([L_femur * np.sin(th), -L_femur * np.cos(th)])
    F  = np.array([F_X, F_Z])

    def residual(phi):
        E = C + L_stub * np.array([-np.sin(phi), np.cos(phi)])
        return float(np.dot(F - E, F - E) - Lc**2)

    phi_sweep = np.linspace(-np.pi, np.pi, 720)
    r_sweep   = [residual(p) for p in phi_sweep]
    brackets  = [(phi_sweep[i], phi_sweep[i+1])
                 for i in range(len(phi_sweep) - 1)
                 if np.sign(r_sweep[i]) != np.sign(r_sweep[i+1])]
    if not brackets:
        return None, None, None, None

    phi_ideal = -th
    roots = [brentq(residual, lo, hi, xtol=1e-9) for lo, hi in brackets]
    phi   = min(roots, key=lambda r: abs(r - phi_ideal))

    E = C + L_stub  * np.array([-np.sin(phi),  np.cos(phi)])
    W = C + L_tibia * np.array([ np.sin(phi), -np.cos(phi)])
    return C, E, W, phi


def to_mujoco_qpos(hip_deg):
    """
    Convert step07 hip angle to MuJoCo joint angles [hinge_hip, hinge_knee, hinge_F].

    Topology: ground→femur(hinge_hip)→tibia(hinge_knee), ground→coupler(hinge_F).

    Coordinate mapping: step07 (x=fwd, y=up) → MuJoCo XZ (X=fwd, Z=up, rot about +Y).
    MuJoCo R_Y(q)*[1,0,0] = (cos q, 0, −sin q)  in world XZ.

    Femur direction in step07 at hip_deg: (sin θ, −cos θ) → MuJoCo XZ same values.
    → cos(q_hip) = sin θ,  −sin(q_hip) = −cos θ   → q_hip = π/2 − θ
    """
    C, E, W, phi = step07_fk(hip_deg)
    if C is None:
        return None

    theta  = np.radians(hip_deg)
    q_hip  = np.pi / 2 - theta

    # Tibia stub direction in step07: (−sin φ, cos φ)
    # MuJoCo tibia +Z in world: R_Y(q_hip+q_knee)*[0,0,1] = (sin total, 0, cos total)
    # → sin total = −sin φ,  cos total = cos φ  → total = −φ
    q_knee = (-phi) - q_hip

    # Coupler: hangs from ground at F, +X points toward E.
    # E in MuJoCo world: (E[0], 0, E[1])
    # F in MuJoCo world: (F_X, 0, F_Z)
    # Coupler +X in world: (cos q_F, 0, −sin q_F) = (E−F)/|E−F|
    E_m = np.array([E[0], 0.0, E[1]])
    F_m = np.array([F_X,  0.0, F_Z])
    d   = E_m - F_m
    q_coupler_F = np.arctan2(-d[2], d[0])

    return q_hip, q_knee, q_coupler_F


# ── Load ───────────────────────────────────────────────────────────────────
XML_PATH = os.path.join(os.path.dirname(__file__), "step07.xml")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# Fix connect constraint anchors — MuJoCo computes them from qpos=0 which is
# not a valid 4-bar configuration, so both anchors are wrong.
# eq_data[0:3] = anchor in body1 (coupler) frame → E is at coupler tip (Lc, 0, 0)
# eq_data[3:6] = anchor in body2 (tibia)   frame → E is at stub tip (0, 0, L_stub)
eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "4bar_close")
model.eq_data[eq_id, 0:3] = [Lc,     0.0, 0.0   ]
model.eq_data[eq_id, 3:6] = [0.0,    0.0, L_stub ]

# Set initial qpos to valid closed configuration at HIP_INIT_DEG.
qpos = to_mujoco_qpos(HIP_INIT_DEG)
if qpos is not None:
    data.qpos[0], data.qpos[1], data.qpos[2] = qpos
    print(f"hip={qpos[0]:.4f}  knee={qpos[1]:.4f}  coupler={qpos[2]:.4f}  rad")
else:
    print("WARNING: no FK solution at HIP_INIT_DEG")

mujoco.mj_forward(model, data)

# Hip qpos range: step07 hip 10.5°–76° (2° margin inside singularity limits 8.5°–78°)
Q_LO  = np.pi/2 - np.radians(76.0)   # 0.244 rad  (leg extended)
Q_HI  = np.pi/2 - np.radians(10.5)   # 1.388 rad  (leg retracted)
Q_MID = (Q_LO + Q_HI) / 2
Q_AMP = (Q_HI - Q_LO) / 2
FREQ  = 0.5   # Hz — full cycle every 2 seconds

import time
data.ctrl[0] = qpos[0] if qpos else Q_MID   # start actuator at initial angle

# ── Simulate ───────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.perf_counter()
    while viewer.is_running():
        t = time.perf_counter() - t0
        data.ctrl[0] = Q_MID + Q_AMP * np.sin(2 * np.pi * FREQ * t)
        mujoco.mj_step(model, data)
        viewer.sync()
