"""thermal_model.py — Lumped 2-node motor thermal model.

Models resistive (copper) losses → winding temperature rise for all 4 motors:
  - Wheel L / R  (5065 130KV outrunner, direct drive)
  - Hip  L / R   (CubeMars AK45-10 KV75, 10:1 planetary)

Physics:  2 nodes per motor — winding (T_w) and case/stator (T_c)
          Heat flows: I²·R_eff → winding ─(R_th_wc)─> case ─(R_th_ca)─> ambient

ODE (Euler-integrated at control rate):
    C_w · dT_w/dt = P_copper − (T_w − T_c) / R_th_wc
    C_c · dT_c/dt = (T_w − T_c) / R_th_wc − (T_c − T_amb) / R_th_ca

Parameters are estimates from motor-class datasheets and thermal analysis.
They are deliberately exposed in sim_config.py for easy adjustment once
physical measurements (thermistor readings) are available.

Current → torque mapping mirrors the battery model in scenarios.py:
    I_wheel = |τ| / WHEEL_KT            (direct-drive, no gearbox)
    I_hip   = |τ_output| / HIP_KT_OUTPUT  (10:1 gearbox torque × Kt_motor)
"""

from sim_config import (
    WHEEL_KT,
    HIP_KT_OUTPUT,
    # Wheel motor thermal params
    WHEEL_R_EFF,
    WHEEL_C_WINDING, WHEEL_C_CASE,
    WHEEL_R_TH_WC, WHEEL_R_TH_CA,
    WHEEL_T_MAX_C,
    # Hip motor thermal params
    HIP_R_EFF,
    HIP_C_WINDING, HIP_C_CASE,
    HIP_R_TH_WC, HIP_R_TH_CA,
    HIP_T_MAX_C,
    # Ambient
    MOTOR_T_AMB_C,
)


class MotorThermalModel:
    """Single-motor 2-node thermal model.

    Parameters
    ----------
    R_eff    : float  Effective winding resistance for copper-loss calc [Ω].
                      For 3-phase star: R_eff = 3 × R_phase (matches P = I²·R_eff
                      when I is the per-phase RMS current, T = Kt·I).
    C_winding: float  Thermal mass of copper winding [J/°C].
    C_case   : float  Thermal mass of stator + housing [J/°C].
    R_th_wc  : float  Thermal resistance winding→case [°C/W].
    R_th_ca  : float  Thermal resistance case→ambient [°C/W].
    T_max_c  : float  Max rated winding temperature [°C] (for margin reporting).
    T_amb_c  : float  Ambient temperature [°C].
    """

    def __init__(self, R_eff: float, C_winding: float, C_case: float,
                 R_th_wc: float, R_th_ca: float,
                 T_max_c: float, T_amb_c: float = 25.0) -> None:
        self.R_eff    = R_eff
        self.C_winding = C_winding
        self.C_case   = C_case
        self.R_th_wc  = R_th_wc
        self.R_th_ca  = R_th_ca
        self.T_max_c  = T_max_c
        self.T_amb_c  = T_amb_c
        self._T_w = T_amb_c   # winding temperature [°C]
        self._T_c = T_amb_c   # case temperature [°C]
        self._P   = 0.0       # last copper loss [W]

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore initial conditions (call at start of each scenario)."""
        self._T_w = self.T_amb_c
        self._T_c = self.T_amb_c
        self._P   = 0.0

    def step(self, dt: float, I: float) -> None:
        """Advance thermal state by dt [s] under phase current I [A].

        I is the magnitude of per-phase RMS current (= |τ| / Kt).
        Call once per control tick (same rate as BatteryModel.step).
        """
        P = I * I * self.R_eff          # copper loss [W]
        self._P = P

        flux_wc = (self._T_w - self._T_c) / self.R_th_wc   # [W] winding→case
        flux_ca = (self._T_c - self.T_amb_c) / self.R_th_ca # [W] case→ambient

        dT_w = (P      - flux_wc) / self.C_winding
        dT_c = (flux_wc - flux_ca) / self.C_case

        self._T_w += dT_w * dt
        self._T_c += dT_c * dt

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def T_winding(self) -> float:
        """Winding temperature [°C] — the hot spot."""
        return self._T_w

    @property
    def T_case(self) -> float:
        """Case/stator surface temperature [°C]."""
        return self._T_c

    @property
    def T_margin(self) -> float:
        """Headroom before winding hits rated max [°C]. Negative = over-temp."""
        return self.T_max_c - self._T_w

    @property
    def P_copper(self) -> float:
        """Copper loss in last step [W]."""
        return self._P


class RobotThermalModel:
    """Tracks winding temperatures for all 4 motors simultaneously.

    Usage::

        thermal = RobotThermalModel()
        thermal.reset()
        # inside control loop:
        thermal.step(dt,
                     tau_whl_L=data.ctrl[act_wheel_L],
                     tau_whl_R=data.ctrl[act_wheel_R],
                     tau_hip_L=data.ctrl[act_hip_L],
                     tau_hip_R=data.ctrl[act_hip_R])
        print(thermal.wheel_L.T_winding)

    Torques are used to compute per-motor currents internally, matching the
    approach used by ``_motor_currents()`` in scenarios.py.
    """

    def __init__(self) -> None:
        kw_wheel = dict(
            R_eff    = WHEEL_R_EFF,
            C_winding= WHEEL_C_WINDING,
            C_case   = WHEEL_C_CASE,
            R_th_wc  = WHEEL_R_TH_WC,
            R_th_ca  = WHEEL_R_TH_CA,
            T_max_c  = WHEEL_T_MAX_C,
            T_amb_c  = MOTOR_T_AMB_C,
        )
        kw_hip = dict(
            R_eff    = HIP_R_EFF,
            C_winding= HIP_C_WINDING,
            C_case   = HIP_C_CASE,
            R_th_wc  = HIP_R_TH_WC,
            R_th_ca  = HIP_R_TH_CA,
            T_max_c  = HIP_T_MAX_C,
            T_amb_c  = MOTOR_T_AMB_C,
        )
        self.wheel_L = MotorThermalModel(**kw_wheel)
        self.wheel_R = MotorThermalModel(**kw_wheel)
        self.hip_L   = MotorThermalModel(**kw_hip)
        self.hip_R   = MotorThermalModel(**kw_hip)

    def reset(self) -> None:
        """Reset all motors to ambient (call at start of each scenario)."""
        for m in (self.wheel_L, self.wheel_R, self.hip_L, self.hip_R):
            m.reset()

    def step(self, dt: float,
             tau_whl_L: float, tau_whl_R: float,
             tau_hip_L: float, tau_hip_R: float) -> None:
        """Advance all motor thermal states by dt [s].

        Parameters match the ctrl outputs written by the controller,
        same torques passed to ``_motor_currents()`` for the battery model.
        """
        self.wheel_L.step(dt, abs(tau_whl_L) / WHEEL_KT)
        self.wheel_R.step(dt, abs(tau_whl_R) / WHEEL_KT)
        self.hip_L.step(dt,   abs(tau_hip_L) / HIP_KT_OUTPUT)
        self.hip_R.step(dt,   abs(tau_hip_R) / HIP_KT_OUTPUT)

    def peak_winding_temp(self) -> float:
        """Hottest winding temperature across all 4 motors [°C]."""
        return max(m.T_winding for m in
                   (self.wheel_L, self.wheel_R, self.hip_L, self.hip_R))

    def min_margin(self) -> float:
        """Smallest temperature margin across all motors [°C]. Negative = over-temp."""
        return min(m.T_margin for m in
                   (self.wheel_L, self.wheel_R, self.hip_L, self.hip_R))
