"""motor.py — Motor torque taper (back-EMF) and current calculation.

Ported from latency_sensitivity/scenarios.py (motor_taper, _motor_currents).
Models both hip (AK45-10 via gearbox) and wheel (5065 direct drive) motors.
"""
import numpy as np

from master_sim.params import MotorParams, BatteryParams


def motor_taper(tau_cmd: float, omega_wheel: float,
                v_batt: float, motors: MotorParams,
                battery: BatteryParams = None) -> float:
    """Clamp wheel torque by linear back-EMF taper, voltage-scaled.

    omega_noload scales with v_batt / V_nom — lower battery voltage reduces
    available top speed and torque at high RPM.

    Parameters
    ----------
    tau_cmd     : commanded wheel torque [N·m]
    omega_wheel : wheel angular velocity [rad/s]
    v_batt      : current battery terminal voltage [V]
    motors      : MotorParams (wheel KV, current limit)
    battery     : BatteryParams for V_nom reference (defaults to BatteryParams())
    """
    if battery is None:
        battery = BatteryParams()
    whl = motors.wheel
    omega_noload = whl.omega_noload(battery.V_nom) * (v_batt / battery.V_nom)
    taper = max(0.0, 1.0 - abs(omega_wheel) / omega_noload)
    t_max = whl.torque_limit * taper
    return float(np.clip(tau_cmd, -t_max, t_max))


def motor_currents(tau_whl_L: float, tau_whl_R: float,
                   tau_hip_L: float, tau_hip_R: float,
                   motors: MotorParams,
                   I_quiescent: float = 0.30) -> float:
    """Sum all motor currents plus quiescent electronics load [A].

    Uses commanded (clamped) torques as proxy for actual phase current.

    Parameters
    ----------
    tau_whl_L/R : wheel torques [N·m] (direct drive)
    tau_hip_L/R : hip output torques [N·m] (after 10:1 gearbox)
    motors      : MotorParams (Kt values for both motor types)
    I_quiescent : always-on electronics current [A]

    Returns
    -------
    Total current draw [A].
    """
    I_whl = (abs(tau_whl_L) + abs(tau_whl_R)) / motors.wheel.Kt
    I_hip = (abs(tau_hip_L) + abs(tau_hip_R)) / motors.hip.Kt_output
    return I_whl + I_hip + I_quiescent
