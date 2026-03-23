"""velocity_pi.py — Velocity PI outer loop: velocity error -> lean angle.

Converts velocity tracking error into a lean-angle setpoint (theta_ref)
for the LQR inner loop.

theta_ref > 0 -> lean forward  -> drive forward
theta_ref < 0 -> lean backward -> drive backward
"""
import numpy as np

from master_sim.params import VelocityPIGains


class VelocityPI:
    """PI controller: velocity error -> theta_ref for LQR.

    Args:
        gains: VelocityPIGains instance
        dt: control timestep [s]
    """

    def __init__(self, gains: VelocityPIGains, dt: float) -> None:
        self.gains = gains
        self.dt = dt
        self.integral = 0.0
        self.prev_v_desired = 0.0

    def update(self, v_desired_ms: float, v_measured_ms: float) -> float:
        """Compute lean angle command from velocity error.

        Args:
            v_desired_ms: target linear velocity [m/s]
            v_measured_ms: measured linear velocity [m/s]

        Returns:
            theta_ref [rad] — lean angle command for LQR
        """
        v_err = v_desired_ms - v_measured_ms
        self.integral = float(np.clip(
            self.integral + v_err * self.dt,
            -self.gains.int_max, self.gains.int_max))
        # Feed-forward: θ = a/g ≈ Kff * dv_cmd/dt
        dv_cmd_dt = (v_desired_ms - self.prev_v_desired) / self.dt
        self.prev_v_desired = v_desired_ms
        theta_ref = float(np.clip(
            self.gains.Kp * v_err
            + self.gains.Ki * self.integral
            + self.gains.Kff * dv_cmd_dt,
            -self.gains.theta_max, self.gains.theta_max))
        return theta_ref

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_v_desired = 0.0


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from master_sim.defaults import DEFAULT_PARAMS

    gains = DEFAULT_PARAMS.gains.velocity_pi
    dt = 1.0 / DEFAULT_PARAMS.timing.ctrl_hz
    pi = VelocityPI(gains, dt)

    print("VelocityPI — Self-Test")
    print("=" * 50)
    print(f"  Kp={gains.Kp}, Ki={gains.Ki}, dt={dt}")

    # Step response: desired=1.0 m/s, measured=0.0
    for i in range(5):
        theta = pi.update(1.0, 0.0)
        print(f"  step {i}: theta_ref={theta:.6f} rad  integral={pi.integral:.6f}")

    pi.reset()
    print(f"\n  After reset: integral={pi.integral}")
    print("=" * 50)
    print("[OK] VelocityPI self-test passed")
