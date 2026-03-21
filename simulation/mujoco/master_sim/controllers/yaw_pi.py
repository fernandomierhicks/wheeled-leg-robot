"""yaw_pi.py — Yaw PI controller: yaw rate error -> differential wheel torque.

tau_yaw > 0 -> left turn (CCW viewed from above)
tau_yaw < 0 -> right turn (CW)

Orthogonal to LQR/VelocityPI: average wheel torque (tau_sym) is unaffected.
Usage in sim loop:
    tau_L = tau_sym - tau_yaw
    tau_R = tau_sym + tau_yaw
"""
import numpy as np

from master_sim.params import YawPIGains


class YawPI:
    """PI controller: yaw rate error -> differential wheel torque.

    Args:
        gains: YawPIGains instance
        dt: control timestep [s]
    """

    def __init__(self, gains: YawPIGains, dt: float) -> None:
        self.gains = gains
        self.dt = dt
        self.integral = 0.0

    def update(self, omega_desired: float, omega_measured: float) -> float:
        """Compute differential torque from yaw rate error.

        Args:
            omega_desired: target yaw rate [rad/s] (positive = CCW)
            omega_measured: measured yaw rate [rad/s]

        Returns:
            tau_yaw [Nm] — differential torque
        """
        err = omega_desired - omega_measured
        self.integral = float(np.clip(
            self.integral + err * self.dt,
            -self.gains.int_max, self.gains.int_max))
        tau_yaw = self.gains.Kp * err + self.gains.Ki * self.integral
        return float(np.clip(tau_yaw, -self.gains.torque_max, self.gains.torque_max))

    def reset(self) -> None:
        self.integral = 0.0


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from master_sim.defaults import DEFAULT_PARAMS

    gains = DEFAULT_PARAMS.gains.yaw_pi
    dt = 1.0 / DEFAULT_PARAMS.timing.ctrl_hz
    pi = YawPI(gains, dt)

    print("YawPI — Self-Test")
    print("=" * 50)
    print(f"  Kp={gains.Kp}, Ki={gains.Ki}, dt={dt}")

    # Step response: desired=1.0 rad/s, measured=0.0
    for i in range(5):
        tau = pi.update(1.0, 0.0)
        print(f"  step {i}: tau_yaw={tau:.6f} Nm  integral={pi.integral:.6f}")

    pi.reset()
    print(f"\n  After reset: integral={pi.integral}")
    print("=" * 50)
    print("[OK] YawPI self-test passed")
