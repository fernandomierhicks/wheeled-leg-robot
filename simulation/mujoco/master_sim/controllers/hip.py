"""hip.py — Hip joint controllers: position servo + impedance with roll leveling.

Two modes used by different scenarios:
  - Position servo PD (S1–S7): stiff tracking of commanded hip angle
  - Impedance + roll leveling (S8): compliant spring-damper with differential
    hip offset to keep body roll at zero on uneven terrain

Sign convention for roll leveling:
  positive roll = left side UP (right-hand rule about +X forward)
  delta_q > 0 -> retract left leg, extend right leg -> levels body
"""
import math
import numpy as np

from master_sim.params import RobotGeometry, HipMotorParams, SuspensionGains


def hip_position_torque(q_hip: float, dq_hip: float, q_target: float,
                        hip: HipMotorParams,
                        dq_target: float = 0.0) -> float:
    """Stiff PD position servo for hip joint (S1–S7).

    Args:
        q_hip: current hip angle [rad]
        dq_hip: current hip velocity [rad/s]
        q_target: target hip angle [rad]
        hip: HipMotorParams (for Kp, Kd, torque limit)
        dq_target: target hip velocity [rad/s] (feed-forward)

    Returns:
        Clamped hip torque [Nm]
    """
    tau = hip.position_Kp * (q_target - q_hip) + hip.position_Kd * (dq_target - dq_hip)
    return float(np.clip(tau, -hip.torque_limit,
                         hip.torque_limit))


def hip_impedance_torque(q_hip: float, dq_hip: float, q_nom: float,
                         suspension: SuspensionGains,
                         hip: HipMotorParams) -> float:
    """Spring-damper impedance controller for hip joint (S8).

    Args:
        q_hip: current hip angle [rad]
        dq_hip: current hip velocity [rad/s]
        q_nom: equilibrium angle (with roll offset applied) [rad]
        suspension: SuspensionGains (K_s, B_s)
        hip: HipMotorParams (for torque limit)

    Returns:
        Clamped hip torque [Nm]
    """
    tau = -(suspension.K_s * (q_hip - q_nom) + suspension.B_s * dq_hip)
    return float(np.clip(tau, -hip.impedance_torque_limit,
                         hip.impedance_torque_limit))


def roll_leveling_offsets(roll: float, roll_rate: float,
                          q_hip_sym: float,
                          suspension: SuspensionGains,
                          robot: RobotGeometry) -> tuple:
    """Compute per-leg hip setpoints with roll leveling offset.

    Args:
        roll: body roll angle [rad] (positive = left side up)
        roll_rate: body roll rate [rad/s]
        q_hip_sym: symmetric (base) hip setpoint [rad]
        suspension: SuspensionGains (K_roll, D_roll)
        robot: RobotGeometry (for joint limits)

    Returns:
        (q_nom_L, q_nom_R) — clamped per-leg setpoints [rad]
    """
    hip_safe_min, hip_safe_max = SuspensionGains.hip_safe_range(robot)
    delta_q = suspension.K_roll * roll + suspension.D_roll * roll_rate
    q_nom_L = float(np.clip(q_hip_sym + delta_q, hip_safe_min, hip_safe_max))
    q_nom_R = float(np.clip(q_hip_sym - delta_q, hip_safe_min, hip_safe_max))
    return q_nom_L, q_nom_R


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from master_sim.defaults import DEFAULT_PARAMS

    P = DEFAULT_PARAMS
    robot = P.robot
    hip = P.motors.hip
    susp = P.gains.suspension

    print("Hip Controller — Self-Test")
    print("=" * 60)

    # Position servo test
    tau = hip_position_torque(
        q_hip=robot.Q_NOM + 0.1, dq_hip=0.0,
        q_target=robot.Q_NOM, hip=hip)
    print(f"  Position PD: q_hip=Q_NOM+0.1, target=Q_NOM -> tau={tau:.3f} Nm")

    # Impedance test
    tau = hip_impedance_torque(
        q_hip=robot.Q_NOM + 0.1, dq_hip=0.5,
        q_nom=robot.Q_NOM, suspension=susp, hip=hip)
    print(f"  Impedance:   q_hip=Q_NOM+0.1, dq=0.5 -> tau={tau:.3f} Nm")

    # Roll leveling test
    q_L, q_R = roll_leveling_offsets(
        roll=math.radians(5.0), roll_rate=0.0,
        q_hip_sym=robot.Q_NOM, suspension=susp, robot=robot)
    print(f"  Roll level:  roll=5°, q_sym=Q_NOM -> q_L={math.degrees(q_L):.1f}° "
          f"q_R={math.degrees(q_R):.1f}°")

    # Verify clamping at joint limits
    q_L, q_R = roll_leveling_offsets(
        roll=math.radians(90.0), roll_rate=0.0,
        q_hip_sym=robot.Q_NOM, suspension=susp, robot=robot)
    hip_min, hip_max = SuspensionGains.hip_safe_range(robot)
    assert q_L >= hip_min and q_L <= hip_max, "Left setpoint out of safe range!"
    assert q_R >= hip_min and q_R <= hip_max, "Right setpoint out of safe range!"
    print(f"  Clamp test:  roll=90° -> q_L={math.degrees(q_L):.1f}° "
          f"q_R={math.degrees(q_R):.1f}° (within safe range)")

    print("=" * 60)
    print("[OK] Hip controller self-test passed")
