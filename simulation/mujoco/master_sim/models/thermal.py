"""thermal.py — Lumped 2-node motor thermal model.

Ported from latency_sensitivity/thermal_model.py.
Models both hip (AK45-10) and wheel (5065) motors.

Physics:  2 nodes per motor — winding (T_w) and case/stator (T_c)
          Heat flows: I²·R_eff → winding ─(R_th_wc)─> case ─(R_th_ca)─> ambient
"""
from master_sim.params import MotorParams, HipMotorParams, WheelMotorParams


class MotorThermalModel:
    """Single-motor 2-node thermal model.

    Parameters
    ----------
    R_eff     : effective winding resistance [Ω]
    C_winding : thermal mass of copper winding [J/°C]
    C_case    : thermal mass of stator + housing [J/°C]
    R_th_wc   : thermal resistance winding→case [°C/W]
    R_th_ca   : thermal resistance case→ambient [°C/W]
    T_max_c   : max rated winding temperature [°C]
    T_amb_c   : ambient temperature [°C]
    """

    def __init__(self, R_eff: float, C_winding: float, C_case: float,
                 R_th_wc: float, R_th_ca: float,
                 T_max_c: float, T_amb_c: float = 25.0) -> None:
        self.R_eff = R_eff
        self.C_winding = C_winding
        self.C_case = C_case
        self.R_th_wc = R_th_wc
        self.R_th_ca = R_th_ca
        self.T_max_c = T_max_c
        self.T_amb_c = T_amb_c
        self._T_w = T_amb_c
        self._T_c = T_amb_c
        self._P = 0.0

    def reset(self) -> None:
        self._T_w = self.T_amb_c
        self._T_c = self.T_amb_c
        self._P = 0.0

    def step(self, dt: float, I: float) -> None:
        """Advance thermal state by dt [s] under phase current I [A]."""
        P = I * I * self.R_eff
        self._P = P

        flux_wc = (self._T_w - self._T_c) / self.R_th_wc
        flux_ca = (self._T_c - self.T_amb_c) / self.R_th_ca

        self._T_w += (P - flux_wc) / self.C_winding * dt
        self._T_c += (flux_wc - flux_ca) / self.C_case * dt

    @property
    def T_winding(self) -> float:
        return self._T_w

    @property
    def T_case(self) -> float:
        return self._T_c

    @property
    def T_margin(self) -> float:
        """Headroom before winding hits rated max [°C]. Negative = over-temp."""
        return self.T_max_c - self._T_w

    @property
    def P_copper(self) -> float:
        return self._P


def _make_thermal(motor_params, T_amb: float) -> MotorThermalModel:
    """Create a MotorThermalModel from a HipMotorParams or WheelMotorParams."""
    return MotorThermalModel(
        R_eff=motor_params.R_eff,
        C_winding=motor_params.C_winding,
        C_case=motor_params.C_case,
        R_th_wc=motor_params.R_th_wc,
        R_th_ca=motor_params.R_th_ca,
        T_max_c=motor_params.T_max_C,
        T_amb_c=T_amb,
    )


class RobotThermalModel:
    """Tracks winding temperatures for all 4 motors (2 hip + 2 wheel).

    Usage::
        thermal = RobotThermalModel(params.motors)
        thermal.reset()
        thermal.step(dt, tau_whl_L, tau_whl_R, tau_hip_L, tau_hip_R)
        print(thermal.wheel_L.T_winding)
    """

    def __init__(self, motors: MotorParams) -> None:
        self._motors = motors
        T_amb = motors.T_amb_C
        self.wheel_L = _make_thermal(motors.wheel, T_amb)
        self.wheel_R = _make_thermal(motors.wheel, T_amb)
        self.hip_L = _make_thermal(motors.hip, T_amb)
        self.hip_R = _make_thermal(motors.hip, T_amb)

    def reset(self) -> None:
        for m in (self.wheel_L, self.wheel_R, self.hip_L, self.hip_R):
            m.reset()

    def step(self, dt: float,
             tau_whl_L: float, tau_whl_R: float,
             tau_hip_L: float, tau_hip_R: float) -> None:
        """Advance all motor thermal states by dt [s].

        Torques are converted to currents internally:
          I_wheel = |τ| / Kt_wheel
          I_hip   = |τ_output| / Kt_output_shaft
        """
        Kt_whl = self._motors.wheel.Kt
        Kt_hip = self._motors.hip.Kt_output
        self.wheel_L.step(dt, abs(tau_whl_L) / Kt_whl)
        self.wheel_R.step(dt, abs(tau_whl_R) / Kt_whl)
        self.hip_L.step(dt, abs(tau_hip_L) / Kt_hip)
        self.hip_R.step(dt, abs(tau_hip_R) / Kt_hip)

    def peak_winding_temp(self) -> float:
        """Hottest winding temperature across all 4 motors [°C]."""
        return max(m.T_winding for m in
                   (self.wheel_L, self.wheel_R, self.hip_L, self.hip_R))

    def min_margin(self) -> float:
        """Smallest temperature margin [°C]. Negative = over-temp."""
        return min(m.T_margin for m in
                   (self.wheel_L, self.wheel_R, self.hip_L, self.hip_R))
