"""plot_motor_curve.py - Torque-speed curve for the Maytech MTO5065-70-HA-C (70 KV) wheel motor.

Usage:
    python plot_motor_curve.py
    python plot_motor_curve.py --kv 72.5        # measured Kv override
    python plot_motor_curve.py --kt 0.131        # measured Kt override
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "simulation", "mujoco", "baseline1_leg_analysis"))
from motor_models import MotorModel  # noqa: E402

I_MAX        = 50.0    # [A]   ODESC 3.6 current limit
V_BUS        = 24.0    # [V]   6S LiPo nominal
TAU_ELEC     = 0.002   # [s]
B_FRICTION   = 0.001   # [Nm.s/rad]  typical deep-groove ball bearing drag for 5065-class outrunner
WHEEL_RADIUS = 0.075   # [m]


def derive_params(kv, kt_override=None):
    kt = kt_override if kt_override is not None else 9.55 / kv
    t_peak = kt * I_MAX
    omega_noload = kv * V_BUS * (2 * math.pi / 60)
    return kt, t_peak, omega_noload


def run_to_steady(motor, T_cmd, omega, dt=1e-4, duration=0.15):
    motor.reset()
    t_out = 0.0
    for _ in range(int(duration / dt)):
        t_out = motor.step(T_cmd, omega, dt)
    return t_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", type=float, default=70.0)
    parser.add_argument("--kt", type=float, default=None)
    args = parser.parse_args()

    kt, t_peak, omega_noload = derive_params(args.kv, args.kt)
    motor = MotorModel(t_peak, omega_noload, TAU_ELEC, B_FRICTION)

    # Sweep speed from 0 to omega_noload
    N = 200
    omegas = np.linspace(0, omega_noload, N)
    rim_speeds = omegas * WHEEL_RADIUS

    # Ideal taper (no lag/drag — the underlying motor capability)
    taper = np.array([t_peak * max(0.0, 1.0 - w / omega_noload) for w in omegas])

    # Net output including drag (what is actually delivered to the wheel)
    net = np.array([run_to_steady(motor, t_peak, w) for w in omegas])

    # Power curve (from taper, represents motor shaft power)
    power = taper * omegas

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Torque curves ---
    ax1.plot(rim_speeds, taper, color="C0", lw=2.5, label="Available torque (back-EMF taper)")
    ax1.plot(rim_speeds, net,   color="C0", lw=1.5, ls="--", alpha=0.7,
             label="Net output torque (incl. bearing drag)")
    ax1.fill_between(rim_speeds, net, 0, alpha=0.08, color="C0")

    ax1.set_xlabel("Wheel rim speed  [m/s]", fontsize=12)
    ax1.set_ylabel("Torque  [Nm]", fontsize=12, color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xlim(0, rim_speeds[-1] * 1.05)
    ax1.set_ylim(-0.5, t_peak * 1.25)
    ax1.axhline(0, color="black", lw=0.7)
    ax1.grid(True, alpha=0.25)

    # --- Power curve (secondary y-axis) ---
    ax2 = ax1.twinx()
    ax2.plot(rim_speeds, power, color="C3", lw=1.5, ls=":", label="Motor shaft power")
    ax2.set_ylabel("Power  [W]", fontsize=12, color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0, power.max() * 2.2)

    # --- Top axis: angular speed ---
    ax3 = ax1.twiny()
    ax3.set_xlim(0, omegas[-1] * 1.05)
    ax3.set_xlabel("Motor angular speed  [rad/s]", fontsize=11)

    # --- Annotations ---
    # Max torque point
    ax1.annotate(
        f"Max torque\n{t_peak:.2f} Nm  (@  0 m/s)",
        xy=(0, t_peak), xytext=(rim_speeds[-1] * 0.15, t_peak * 1.1),
        arrowprops=dict(arrowstyle="->", color="C0"),
        fontsize=10, color="C0"
    )

    # Max speed point
    ax1.annotate(
        f"Max speed\n{rim_speeds[-1]:.1f} m/s\n({omega_noload:.0f} rad/s)",
        xy=(rim_speeds[-1], 0), xytext=(rim_speeds[-1] * 0.72, t_peak * 0.22),
        arrowprops=dict(arrowstyle="->", color="navy"),
        fontsize=10, color="navy"
    )

    # Peak power annotation
    p_max = power.max()
    idx_p = np.argmax(power)
    ax2.annotate(
        f"Peak power\n{p_max:.0f} W",
        xy=(rim_speeds[idx_p], p_max),
        xytext=(rim_speeds[idx_p] * 0.55, p_max * 1.25),
        arrowprops=dict(arrowstyle="->", color="C3"),
        fontsize=10, color="C3"
    )

    # ODESC current limit note
    ax1.text(rim_speeds[-1] * 0.02, t_peak * 0.05,
             f"ODESC limit: {I_MAX:.0f} A  |  6S @ {V_BUS:.0f} V",
             fontsize=9, color="grey")

    # --- Title & legend ---
    src = f"KV={args.kv:.0f} RPM/V, Kt={kt:.4f} Nm/A"
    if args.kt:
        src += "  (measured Kt)"
    ax1.set_title(f"Maytech MTO5065-70-HA-C  --  {src}", fontsize=12, pad=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()

    outpath = os.path.join(_SCRIPT_DIR, "motor_curve.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved -> {outpath}")
    plt.show()


if __name__ == "__main__":
    main()
