"""motor_models.py — Realistic per-axis motor output models.

Each MotorModel applies three effects per timestep:
  1. Back-EMF torque-speed derating  — linear taper from T_peak at ω=0 to
                                       T=0 at ω_noload (rated no-load speed).
  2. Electrical / driver lag         — first-order filter (time constant τ_elec)
                                       that models the FOC current-loop + CAN
                                       round-trip delay.
  3. Viscous bearing / winding drag  — B_friction × ω opposes rotation.

Motor specs used:
  Wheel  5065 130KV outrunner, 24 V, direct drive
           ω_noload = 130 × 24 × 2π/60 = 326.7 rad/s  →  24.5 m/s at wheel rim
           T_peak   = Kt × I_max = (9.55/130) × 50 A = 3.67 N·m
           τ_elec   ≈ 2 ms  (CAN transport <0.5 ms + ODrive FOC electrical rise ~1 ms)
           B_fric   ≈ 0.02 N·m·s/rad  (outrunner bearing drag)

  Hip    CubeMars AK45-10  KV75, 24 V, 10:1 planetary
           ω_noload = 75 × 24 × 2π/60 / 10 = 18.85 rad/s  (≡ OMEGA_MAX)
           τ_elec   ≈ 2 ms  (CAN transport <0.5 ms + integrated FOC electrical rise ~1 ms)
           B_fric   ≈ 0.02 N·m·s/rad  (planetary gearbox viscous drag)

  Note on CAN latency: a single torque command arrives in <0.5 ms at 1 Mbps
  (130-bit frame).  Once received the motor's internal FOC closes the current
  loop at ~10 kHz, so current (= torque) rises with the winding τ_e = L/R
  ≈ 0.2–1 ms.  The first-order filter here is an approximation of the combined
  one-shot transport delay + electrical rise, not a recurring per-step lag.
"""


class MotorModel:
    """Single-axis brushless motor with realistic output saturation and lag.

    Parameters
    ----------
    T_peak       : float  Peak torque [N·m] at zero speed.
    omega_noload : float  No-load speed [rad/s] at rated voltage — torque
                          tapers linearly to zero at this speed.
    tau_elec     : float  First-order time constant [s] for the combined
                          electrical lag of FOC current loop + driver comms.
    B_friction   : float  Viscous friction coefficient [N·m·s/rad].
    """

    def __init__(self, T_peak: float, omega_noload: float,
                 tau_elec: float, B_friction: float = 0.0) -> None:
        self.T_peak       = T_peak
        self.omega_noload = omega_noload
        self.tau_elec     = tau_elec
        self.B_friction   = B_friction
        self._T_filt      = 0.0   # internal state: filtered torque

    # ------------------------------------------------------------------
    def step(self, T_cmd: float, omega: float, dt: float) -> float:
        """Compute actual delivered torque for one simulation timestep.

        Parameters
        ----------
        T_cmd : Torque requested by the controller [N·m].
        omega : Current joint angular velocity [rad/s].
        dt    : Simulation timestep [s].

        Returns
        -------
        T_out : Torque actually applied to the joint [N·m].
        """
        # 1. Back-EMF derating — available torque falls linearly with speed
        speed_ratio = abs(omega) / self.omega_noload
        T_avail = self.T_peak * max(0.0, 1.0 - speed_ratio)
        T_sat   = max(-T_avail, min(T_avail, T_cmd))

        # 2. First-order lag: alpha = dt / (τ + dt)
        alpha         = dt / (self.tau_elec + dt)
        self._T_filt += alpha * (T_sat - self._T_filt)

        # 3. Viscous drag (opposes motion, always active)
        T_drag = -self.B_friction * omega

        return self._T_filt + T_drag

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal filter state (call on sim restart)."""
        self._T_filt = 0.0
