"""lqr_design.py — 3-State LQR Controller Design + Gain Scheduling

Linearized inverted pendulum model with gain scheduling across 3 leg positions.
State: [pitch − equilibrium_pitch, pitch_rate, wheel_vel − v_ref]

Uses scipy.linalg.solve_continuous_are to compute K from Q, R weights.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
import math

from sim_config import (
    ROBOT, WHEEL_R, MOTOR_MASS,
    Q_RET, Q_NOM, Q_EXT,
)
from physics import solve_ik


# ── Constants ─────────────────────────────────────────────────────────────────
G = 9.81  # [m/s²]


# ── Continuous-time A/B matrices from linearized dynamics ──────────────────────
def _compute_coefficients(l_eff: float, m_b: float, m_w: float) -> tuple:
    """Compute linearized model coefficients from mass/inertia.

    Returns (alpha, beta, gamma, delta) for A/B matrix construction.
    """
    r = WHEEL_R
    I_w = 0.5 * (m_w / 2.0) * WHEEL_R**2  # moment of inertia of one wheel
    I_w_total = 2 * I_w
    I_b = m_b * l_eff**2

    M = m_b + m_w

    denom = (M + 2 * I_w_total / r**2) * (I_b + m_b * l_eff**2) - m_b**2 * l_eff**2

    if abs(denom) < 1e-6:
        raise ValueError(f"Singular system: denom={denom}")

    alpha = (M + 2 * I_w_total / r**2) * m_b * G * l_eff / denom
    beta  = -m_b**2 * G * l_eff**2 / (r * denom)
    gamma = -(I_b + m_b * l_eff**2) / (r * denom)
    delta = (M + 2 * I_w_total / r**2 + m_b * l_eff / r) / denom

    return alpha, beta, gamma, delta


def _build_continuous_matrices(l_eff: float, m_b: float, m_w: float) -> tuple:
    """Build continuous A/B matrices.

    State: x = [pitch, pitch_rate, wheel_vel]
    Returns (A, B) as numpy arrays.
    """
    alpha, beta, gamma, delta = _compute_coefficients(l_eff, m_b, m_w)

    A = np.array([
        [0.0, 1.0, 0.0],
        [alpha, 0.0, 0.0],
        [beta, 0.0, 0.0]
    ])

    B = np.array([
        [0.0],
        [gamma],
        [delta]
    ])

    return A, B


def compute_lqr_gain(q_hip: float, p: dict, Q_diag: list, R_val: float) -> np.ndarray:
    """Compute LQR gain K at a given hip angle.

    Args:
        q_hip: Hip joint angle [rad]
        p: Robot geometry dict (ROBOT)
        Q_diag: Diagonal weights [Q_pitch, Q_pitch_rate, Q_vel]
        R_val: Scalar control effort weight

    Returns:
        K as 1D array [k_pitch, k_pitch_rate, k_wheel_vel]
    """
    ik = solve_ik(q_hip, p)
    if ik is None:
        raise RuntimeError(f"IK failed at q_hip={q_hip:.3f}")

    l_eff = abs(ik['W_z'])

    # Compute body mass (excluding wheels)
    m_b = (p['m_box'] +
           2 * (p['m_femur'] + p['m_tibia'] + p['m_coupler'] + p['m_bearing']) +
           2 * MOTOR_MASS)
    m_w = 2 * p['m_wheel']

    A, B = _build_continuous_matrices(l_eff, m_b, m_w)

    # Cost matrices
    Q = np.diag(Q_diag)
    R = np.array([[R_val]])

    # Solve continuous-time ARE
    P = solve_continuous_are(A, B, Q, R)

    # K = R^{-1} B^T P
    K = np.linalg.solve(R, B.T @ P)  # shape (1, 3)

    return K.flatten()  # (3,)


def compute_gain_table(p: dict, Q_diag: list, R_val: float) -> dict:
    """Precompute LQR gains at 3 leg positions.

    Args:
        p: Robot geometry dict
        Q_diag: Cost weights
        R_val: Control effort weight

    Returns:
        {'retracted': K_ret, 'nominal': K_nom, 'extended': K_ext}
    """
    K_ret = compute_lqr_gain(Q_RET, p, Q_diag, R_val)
    K_nom = compute_lqr_gain(Q_NOM, p, Q_diag, R_val)
    K_ext = compute_lqr_gain(Q_EXT, p, Q_diag, R_val)

    return {
        'retracted': K_ret,
        'nominal': K_nom,
        'extended': K_ext,
    }


def interpolate_gains(K_table: dict, q_hip: float) -> np.ndarray:
    """Linear interpolation of LQR gain across leg stroke.

    Args:
        K_table: dict with 'retracted', 'nominal', 'extended' entries
        q_hip: Current hip angle [rad]

    Returns:
        Interpolated K as (3,) array
    """
    # Linear interpolation parameter [0, 1]
    alpha = (q_hip - Q_RET) / (Q_EXT - Q_RET)
    alpha = np.clip(alpha, 0.0, 1.0)

    K_ret = K_table['retracted']
    K_ext = K_table['extended']

    K = (1 - alpha) * K_ret + alpha * K_ext

    return K


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LQR Gain Design — Self-Test")
    print("=" * 70)

    Q_diag = [100.0, 10.0, 1.0]
    R_val = 1.0

    print(f"\nCost weights:")
    print(f"  Q_pitch={Q_diag[0]}, Q_pitch_rate={Q_diag[1]}, Q_vel={Q_diag[2]}")
    print(f"  R={R_val}")

    print(f"\nComputing gain table at Q_RET, Q_NOM, Q_EXT...")
    K_table = compute_gain_table(ROBOT, Q_diag, R_val)

    for pos, K in K_table.items():
        print(f"\n  {pos:12s}: K = {K}")

    print(f"\nInterpolation test:")
    test_angles = [Q_RET, Q_NOM, Q_EXT]
    for q in test_angles:
        K = interpolate_gains(K_table, q)
        print(f"  q_hip={q:7.4f}: K={K}")

    print("\n" + "=" * 70)
    print("[OK] LQR design self-test passed")
