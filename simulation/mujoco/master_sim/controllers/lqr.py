"""lqr.py — 3-State LQR controller with gain scheduling.

Linearized inverted pendulum model.
State: [pitch − equilibrium_pitch − theta_ref, pitch_rate, wheel_vel − v_ref]

Uses scipy.linalg.solve_continuous_are to compute K from Q, R weights.
Gain table is precomputed at 3 leg positions and interpolated at runtime.
"""
import math
import numpy as np
from scipy.linalg import solve_continuous_are, expm

from master_sim.params import RobotGeometry, LQRGains, WheelMotorParams
from master_sim.physics import solve_ik, get_equilibrium_pitch


G = 9.81  # [m/s²]


# ── Linearized dynamics ──────────────────────────────────────────────────────

def _compute_coefficients(l_eff: float, m_b: float, m_w: float,
                          wheel_r: float) -> tuple:
    """Compute linearized model coefficients from mass/inertia.

    Returns (alpha, beta, gamma, delta) for A/B matrix construction.
    """
    r = wheel_r
    I_w = 0.5 * (m_w / 2.0) * r**2   # moment of inertia of one wheel
    I_w_total = 2 * I_w
    I_b = m_b * l_eff**2

    M = m_b + m_w

    denom = ((M + 2 * I_w_total / r**2) * (I_b + m_b * l_eff**2)
             - m_b**2 * l_eff**2)

    if abs(denom) < 1e-6:
        raise ValueError(f"Singular system: denom={denom}")

    alpha = (M + 2 * I_w_total / r**2) * m_b * G * l_eff / denom
    beta  = -m_b**2 * G * l_eff**2 / (r * denom)
    gamma = -(I_b + m_b * l_eff**2) / (r * denom)
    delta = (M + 2 * I_w_total / r**2 + m_b * l_eff / r) / denom

    return alpha, beta, gamma, delta


def _build_continuous_matrices(l_eff: float, m_b: float, m_w: float,
                               wheel_r: float) -> tuple:
    """Build continuous A/B matrices.

    State: x = [pitch, pitch_rate, wheel_vel]
    Returns (A, B) as numpy arrays.
    """
    alpha, beta, gamma, delta = _compute_coefficients(l_eff, m_b, m_w, wheel_r)

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


# ── Gain computation ─────────────────────────────────────────────────────────

def compute_lqr_gain(q_hip: float, robot: RobotGeometry,
                     Q_diag: list, R_val: float) -> np.ndarray:
    """Compute LQR gain K at a given hip angle.

    Args:
        q_hip: Hip joint angle [rad]
        robot: RobotGeometry instance
        Q_diag: Diagonal weights [Q_pitch, Q_pitch_rate, Q_vel]
        R_val: Scalar control effort weight

    Returns:
        K as 1D array [k_pitch, k_pitch_rate, k_wheel_vel]
    """
    p = robot.as_dict()
    ik = solve_ik(q_hip, p)
    if ik is None:
        raise RuntimeError(f"IK failed at q_hip={q_hip:.3f}")

    l_eff = abs(ik['W_z'])

    # Body mass (excluding wheels)
    m_b = (robot.m_box
           + 2 * (robot.m_femur + robot.m_tibia + robot.m_coupler + robot.m_bearing)
           + 2 * robot.motor_mass)
    m_w = 2 * robot.m_wheel

    A, B = _build_continuous_matrices(l_eff, m_b, m_w, robot.wheel_r)

    Q = np.diag(Q_diag)
    R = np.array([[R_val]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # shape (1, 3)

    return K.flatten()  # (3,)


def compute_gain_table(robot: RobotGeometry,
                       lqr_gains: LQRGains) -> dict:
    """Precompute LQR gains at 3 leg positions (retracted, nominal, extended).

    Returns:
        {'retracted': K_ret, 'nominal': K_nom, 'extended': K_ext}
    """
    Q_diag = [lqr_gains.Q_pitch, lqr_gains.Q_pitch_rate, lqr_gains.Q_vel]
    R_val = lqr_gains.R

    K_ret = compute_lqr_gain(robot.Q_RET, robot, Q_diag, R_val)
    K_nom = compute_lqr_gain(robot.Q_NOM, robot, Q_diag, R_val)
    K_ext = compute_lqr_gain(robot.Q_EXT, robot, Q_diag, R_val)

    return {
        'retracted': K_ret,
        'nominal': K_nom,
        'extended': K_ext,
    }


def interpolate_gains(K_table: dict, q_hip: float,
                      robot: RobotGeometry) -> np.ndarray:
    """Linear interpolation of LQR gain across leg stroke.

    Args:
        K_table: dict with 'retracted', 'nominal', 'extended' entries
        q_hip: Current hip angle [rad]
        robot: RobotGeometry (for Q_RET, Q_EXT)

    Returns:
        Interpolated K as (3,) array
    """
    alpha = (q_hip - robot.Q_RET) / (robot.Q_EXT - robot.Q_RET)
    alpha = np.clip(alpha, 0.0, 1.0)

    K_ret = K_table['retracted']
    K_ext = K_table['extended']

    return (1 - alpha) * K_ret + alpha * K_ext


def compute_AB_table(robot: RobotGeometry) -> dict:
    """Precompute continuous A, B matrices at 3 leg positions for state prediction.

    Returns dict with 'retracted', 'nominal', 'extended' keys mapping to (A, B) tuples.
    """
    m_b = (robot.m_box
           + 2 * (robot.m_femur + robot.m_tibia + robot.m_coupler + robot.m_bearing)
           + 2 * robot.motor_mass)
    m_w = 2 * robot.m_wheel

    def _ab_at(q_hip):
        ik = solve_ik(q_hip, robot.as_dict())
        l_eff = abs(ik['W_z'])
        return _build_continuous_matrices(l_eff, m_b, m_w, robot.wheel_r)

    return {
        'retracted': _ab_at(robot.Q_RET),
        'nominal':   _ab_at(robot.Q_NOM),
        'extended':  _ab_at(robot.Q_EXT),
    }


def interpolate_AB(AB_table: dict, q_hip: float,
                   robot: RobotGeometry) -> tuple:
    """Interpolate A, B matrices across leg stroke (same scheme as gain interpolation)."""
    alpha = np.clip((q_hip - robot.Q_RET) / (robot.Q_EXT - robot.Q_RET), 0.0, 1.0)
    A_ret, B_ret = AB_table['retracted']
    A_ext, B_ext = AB_table['extended']
    return (1 - alpha) * A_ret + alpha * A_ext, (1 - alpha) * B_ret + alpha * B_ext


def discretize_AB(A: np.ndarray, B: np.ndarray, dt: float) -> tuple:
    """Discretize continuous (A, B) via zero-order hold (matrix exponential).

    Uses the standard block-matrix method:
        expm([[A, B], [0, 0]] * dt) = [[A_d, B_d], [0, I]]

    Returns (A_d, B_d).
    """
    n, m = A.shape[0], B.shape[1]
    block = np.zeros((n + m, n + m))
    block[:n, :n] = A
    block[:n, n:n + m] = B
    Md = expm(block * dt)
    return Md[:n, :n], Md[:n, n:n + m]


def lqr_torque(pitch: float, pitch_rate: float, wheel_vel: float,
               hip_q_avg: float, K_table: dict, robot: RobotGeometry,
               wheel: WheelMotorParams,
               v_ref: float = 0.0,
               theta_ref: float = 0.0) -> float:
    """3-state LQR balance controller with gain scheduling.

    State: [pitch − pitch_ff − theta_ref, pitch_rate, wheel_vel − v_ref]

    theta_ref (lean command from VelocityPI):
      +  -> lean forward  -> drive forward
      0  -> static balance (default)
      −  -> lean backward -> drive backward

    v_ref: wheel angular velocity feedforward [rad/s]

    Returns tau_wheel (symmetric) for both wheels.
    """
    pitch_ff = get_equilibrium_pitch(robot, hip_q_avg)
    K = interpolate_gains(K_table, hip_q_avg, robot)

    x = np.array([pitch - pitch_ff - theta_ref, pitch_rate, wheel_vel - v_ref])
    u = float(-np.dot(K, x))

    tau_wheel = float(np.clip(u, -wheel.torque_limit, wheel.torque_limit))
    return tau_wheel


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from master_sim.defaults import DEFAULT_PARAMS

    P = DEFAULT_PARAMS
    robot = P.robot
    lqr = P.gains.lqr

    print("LQR Gain Design — Self-Test")
    print("=" * 70)
    print(f"\nCost weights:")
    print(f"  Q_pitch={lqr.Q_pitch}, Q_pitch_rate={lqr.Q_pitch_rate}, "
          f"Q_vel={lqr.Q_vel}")
    print(f"  R={lqr.R}")

    print(f"\nComputing gain table at Q_RET, Q_NOM, Q_EXT...")
    K_table = compute_gain_table(robot, lqr)

    for pos, K in K_table.items():
        print(f"  {pos:12s}: K = {K}")

    print(f"\nInterpolation test:")
    for q in [robot.Q_RET, robot.Q_NOM, robot.Q_EXT]:
        K = interpolate_gains(K_table, q, robot)
        print(f"  q_hip={q:7.4f}: K={K}")

    print(f"\nlqr_torque test (static balance at Q_NOM):")
    pitch_ff = get_equilibrium_pitch(robot, robot.Q_NOM)
    tau = lqr_torque(pitch_ff + 0.05, 0.0, 0.0, robot.Q_NOM,
                     K_table, robot, P.motors.wheel)
    print(f"  pitch=eq+0.05rad -> tau_wheel={tau:.4f} Nm")

    print("\n" + "=" * 70)
    print("[OK] LQR design self-test passed")
