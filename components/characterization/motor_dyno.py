"""motor_dyno.py - Simulated dynamometer for the Maytech MTO5065-70-HA-C (70 KV) wheel motor.

Validates that the MotorModel class matches the datasheet specs before committing
to updating sim constants or buying hardware.

Usage:
    python motor_dyno.py                   # datasheet values (70 KV)
    python motor_dyno.py --kv 72.5         # measured Kv override
    python motor_dyno.py --kt 0.131        # measured Kt override (skips Kv derivation)

Tests:
    1a. Torque-speed curve        - linear taper from T_peak at stall to 0 at omega_noload
    1b. Electrical step response  - 63% rise time == tau_elec (2 ms)
    1c. Operating point table     - T_peak and omega_noload at 6S and 12S
    1d. Drag self-check           - unpowered motor decelerates to rest

Outputs:
    motor_dyno_results.png   - saved in same directory as this script
    Console PASS / FAIL summary
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless - no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Add baseline1 to path so we can import MotorModel directly
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "simulation", "mujoco", "baseline1_leg_analysis"))
from motor_models import MotorModel  # noqa: E402

# ---------------------------------------------------------------------------
# Motor constants - edit these to match measured values once hardware arrives
# ---------------------------------------------------------------------------
I_MAX        = 50.0    # [A]   ODESC 3.6 current limit (also the motor max)
V_BUS_6S     = 24.0    # [V]   6S LiPo nominal
V_BUS_12S    = 44.4    # [V]   12S LiPo nominal
TAU_ELEC     = 0.002   # [s]   FOC current-loop + CAN transport
B_FRICTION   = 0.001   # [Nm.s/rad]  typical deep-groove ball bearing drag for 5065-class outrunner
WHEEL_RADIUS = 0.075   # [m]   75 mm radius -> 150 mm OD
# Wheel inertia: 0.5 * m * r^2 = 0.5 * 0.15 kg * 0.075^2 = 0.000422 kg.m^2
J_WHEEL      = 0.000422  # [kg.m^2]  realistic 150mm PLA/TPU wheel

_PASS = True   # global; set False on any failure


def derive_params(kv_rpm_per_v, kt_override=None, v_bus=V_BUS_6S):
    """Return (Kt, T_peak, omega_noload) from KV rating (or measured Kt)."""
    kt = kt_override if kt_override is not None else 9.55 / kv_rpm_per_v
    t_peak = kt * I_MAX
    omega_noload = kv_rpm_per_v * v_bus * (2 * math.pi / 60)
    return kt, t_peak, omega_noload


def run_to_steady(motor, T_cmd, omega, dt=1e-4, duration=0.15):
    """Step the motor model for `duration` seconds at fixed omega; return final torque."""
    motor.reset()
    t_out = 0.0
    for _ in range(int(duration / dt)):
        t_out = motor.step(T_cmd, omega, dt)
    return t_out


# ---------------------------------------------------------------------------
# Test 1a - Torque-speed curve
# ---------------------------------------------------------------------------
def test_torque_speed_curve(motor, t_peak, omega_noload, label):
    global _PASS
    print(f"\n[1a] Torque-speed curve - {label}")

    N = 60
    omegas = np.linspace(0.0, omega_noload * 1.05, N)
    measured = []
    for omega in omegas:
        t_out = run_to_steady(motor, t_peak, omega)
        measured.append(t_out)
    measured = np.array(measured)

    # Expected: linear taper (drag is separate)
    expected = np.array([
        t_peak * max(0.0, 1.0 - abs(w) / omega_noload) for w in omegas
    ])

    # Strip drag from measured for taper comparison
    drag = B_FRICTION * omegas
    measured_no_drag = measured + drag  # add back because drag is subtracted inside step()

    mask = omegas <= omega_noload
    max_err_pct = np.max(np.abs(measured_no_drag[mask] - expected[mask])) / t_peak * 100

    # Key checkpoints
    t_stall  = run_to_steady(motor, t_peak, 0.0)
    t_noload = run_to_steady(motor, t_peak, omega_noload)
    t_half   = run_to_steady(motor, t_peak, omega_noload / 2)

    drag_stall  = B_FRICTION * 0.0
    drag_half   = B_FRICTION * (omega_noload / 2)
    drag_noload = B_FRICTION * omega_noload
    print(f"  Stall torque      : {t_stall:.4f} Nm  (taper={t_peak:.4f}, drag={drag_stall:.4f})")
    print(f"  Half-speed torque : {t_half:.4f} Nm  (taper={t_peak/2:.4f}, drag=-{drag_half:.4f})")
    print(f"  No-load torque    : {t_noload:.4f} Nm  (taper=0.0000, drag=-{drag_noload:.4f})")
    print(f"  Max taper error   : {max_err_pct:.2f}%  (pass < 2%)")

    passed = max_err_pct < 2.0
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        _PASS = False

    return omegas, measured, expected


# ---------------------------------------------------------------------------
# Test 1b - Electrical step response
# ---------------------------------------------------------------------------
def test_step_response(motor, t_peak):
    global _PASS
    print("\n[1b] Electrical step response")

    dt = 1e-5   # 10 us - fine enough to resolve 2 ms rise
    duration = 0.020
    steps = int(duration / dt)
    motor.reset()

    times = []
    torques = []
    for i in range(steps):
        t = i * dt
        t_out = motor.step(t_peak, 0.0, dt)
        times.append(t)
        torques.append(t_out)

    times = np.array(times)
    torques = np.array(torques)

    # Find 63% rise time
    target = t_peak * 0.632
    idx = np.searchsorted(torques, target)
    t_rise = times[idx] if idx < len(times) else float("nan")

    err_ms = abs(t_rise - TAU_ELEC) * 1000
    print(f"  63% rise time : {t_rise*1000:.2f} ms  (expected {TAU_ELEC*1000:.1f} ms)")
    print(f"  Error         : {err_ms:.3f} ms  (pass < 0.5 ms)")

    passed = err_ms < 0.5
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        _PASS = False

    return times, torques


# ---------------------------------------------------------------------------
# Test 1c - Operating point table (6S and 12S)
# ---------------------------------------------------------------------------
def test_operating_points(kv_rpm_per_v, kt_override=None):
    global _PASS
    print("\n[1c] Operating point table")

    rows = []
    for label, v_bus in [("6S (24.0V)", V_BUS_6S), ("12S (44.4V)", V_BUS_12S)]:
        kt, t_peak, omega_noload = derive_params(kv_rpm_per_v, kt_override, v_bus)
        rim_speed = omega_noload * WHEEL_RADIUS
        print(f"  {label}:  Kt={kt:.4f} Nm/A  T_peak={t_peak:.3f} Nm  "
              f"omega_noload={omega_noload:.1f} rad/s  v_rim={rim_speed:.1f} m/s")
        rows.append((label, kt, t_peak, omega_noload, rim_speed))

    t6s  = rows[0][2]
    t12s = rows[1][2]
    print(f"\n  T_peak at 6S  = {t6s:.3f} Nm")
    print(f"  T_peak at 12S = {t12s:.3f} Nm  (same - torque is current-limited, not voltage-limited)")
    print(f"  T_peak 6S >= 3 Nm (balance margin): {'PASS' if t6s >= 3.0 else 'FAIL'}")

    passed = abs(t6s - t12s) < 0.01 and t6s >= 3.0
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        _PASS = False

    return rows


# ---------------------------------------------------------------------------
# Test 1d - Drag self-check (unpowered motor decelerates to rest)
# ---------------------------------------------------------------------------
def test_drag_deceleration(motor):
    global _PASS
    print("\n[1d] Drag self-check (unpowered deceleration)")

    dt = 1e-3
    omega = 50.0   # start at 50 rad/s
    motor.reset()

    times, omegas = [0.0], [omega]
    for i in range(10000):  # 10 s (5x the J/B time constant at B=0.001)
        t_out = motor.step(0.0, omega, dt)
        omega += (t_out / J_WHEEL) * dt   # Euler: alpha = T/J
        if omega < 0:
            omega = 0.0
        times.append((i + 1) * dt)
        omegas.append(omega)

    tau_mech = J_WHEEL / B_FRICTION   # mechanical time constant [s]
    final_omega = omegas[-1]
    print(f"  Mech. time constant (J/B) : {tau_mech:.3f} s")
    print(f"  Start omega : 50.0 rad/s")
    print(f"  Final omega : {final_omega:.4f} rad/s  (expected ~0)")
    print(f"  Decelerates to rest       : {'PASS' if final_omega < 0.1 else 'FAIL'}")
    print(f"  No self-acceleration      : {'PASS' if max(omegas) <= 50.0 else 'FAIL (self-accelerated!)'}")

    passed = final_omega < 0.1 and max(omegas) <= 50.0
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        _PASS = False

    return np.array(times), np.array(omegas)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(curve_data, step_data, drag_data, kv, kt, t_peak, omega_noload, outpath):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Simulated Dynamometer - Maytech MTO5065-70-HA-C\n"
        f"KV={kv:.1f} RPM/V  Kt={kt:.4f} Nm/A  "
        f"T_peak={t_peak:.3f} Nm  omega_noload={omega_noload:.1f} rad/s",
        fontsize=11
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # 1a - torque-speed curve
    ax1 = fig.add_subplot(gs[0, 0])
    omegas, measured, expected = curve_data
    ax1.plot(omegas, expected, "k--", lw=1.5, label="Ideal linear taper")
    ax1.plot(omegas, measured, "C0",  lw=2,   label="MotorModel output")
    ax1.axvline(omega_noload, color="red", lw=1, ls=":", label=f"omega_noload={omega_noload:.0f} rad/s")
    ax1.axhline(t_peak, color="grey", lw=0.8, ls=":")
    ax1.set_xlabel("omega [rad/s]")
    ax1.set_ylabel("Torque [Nm]")
    ax1.set_title("1a - Torque-speed curve")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, omegas[-1])
    ax1.set_ylim(-0.3, t_peak * 1.15)
    ax1.grid(True, alpha=0.3)

    # Rim speed secondary axis
    ax1b = ax1.twiny()
    ax1b.set_xlim(0, omegas[-1] * WHEEL_RADIUS)
    ax1b.set_xlabel("Rim speed [m/s]", fontsize=8)

    # 1b - step response
    ax2 = fig.add_subplot(gs[0, 1])
    times_s, torques_s = step_data
    ax2.plot(times_s * 1000, torques_s, "C1", lw=2)
    ax2.axhline(t_peak * 0.632, color="grey", lw=1, ls="--", label="63% level")
    ax2.axvline(TAU_ELEC * 1000, color="red", lw=1, ls=":",
                label=f"tau_elec={TAU_ELEC*1000:.0f} ms")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_title("1b - Electrical step response (omega=0)")
    ax2.set_xlim(0, 12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 1c - operating points bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    bar_labels = ["6S (24V)\nT_peak [Nm]", "12S (44.4V)\nT_peak [Nm]",
                  "6S omega_noload\n[rad/s / 10]", "12S omega_noload\n[rad/s / 10]"]
    _kt6,  t6,  w6  = derive_params(kv, None, V_BUS_6S)
    _kt12, t12, w12 = derive_params(kv, None, V_BUS_12S)
    values = [t6, t12, w6 / 10, w12 / 10]
    colors = ["C0", "C0", "C2", "C2"]
    bars = ax3.bar(bar_labels, values, color=colors, alpha=0.75, edgecolor="black", lw=0.7)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("Nm  /  (rad/s / 10)")
    ax3.set_title("1c - Operating points at 6S vs 12S")
    ax3.grid(True, axis="y", alpha=0.3)

    # 1d - drag deceleration
    ax4 = fig.add_subplot(gs[1, 1])
    times_d, omegas_d = drag_data
    ax4.plot(times_d, omegas_d, "C3", lw=2)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("omega [rad/s]")
    ax4.set_title("1d - Unpowered deceleration (T_cmd=0, omega0=50 rad/s)")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.grid(True, alpha=0.3)

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Motor dynamometer for MTO5065-70-HA-C")
    parser.add_argument("--kv", type=float, default=70.0,
                        help="KV rating [RPM/V]")
    parser.add_argument("--kt", type=float, default=None,
                        help="Measured Kt [Nm/A] (overrides KV derivation)")
    args = parser.parse_args()

    kt, t_peak, omega_noload = derive_params(args.kv, args.kt, V_BUS_6S)

    print("=" * 60)
    print("Maytech MTO5065-70-HA-C - Simulated Dynamometer")
    print("=" * 60)
    print(f"  KV            : {args.kv:.1f} RPM/V  {'(datasheet)' if args.kt is None else ''}")
    src = "(from KV)" if args.kt is None else "(measured override)"
    print(f"  Kt            : {kt:.4f} Nm/A  {src}")
    print(f"  T_peak @ 50A  : {t_peak:.3f} Nm")
    print(f"  omega_noload  : {omega_noload:.1f} rad/s  ({omega_noload * WHEEL_RADIUS:.1f} m/s rim)")
    print(f"  tau_elec      : {TAU_ELEC*1000:.0f} ms")
    print(f"  B_friction    : {B_FRICTION} Nm.s/rad")

    motor = MotorModel(t_peak, omega_noload, TAU_ELEC, B_FRICTION)

    curve_data = test_torque_speed_curve(motor, t_peak, omega_noload, f"{args.kv:.0f}KV @ 24V")
    step_data  = test_step_response(motor, t_peak)
    _op_rows   = test_operating_points(args.kv, args.kt)
    drag_data  = test_drag_deceleration(motor)

    outpath = os.path.join(_SCRIPT_DIR, "motor_dyno_results.png")
    make_plot(curve_data, step_data, drag_data, args.kv, kt, t_peak, omega_noload, outpath)

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if _PASS else 'SOME TESTS FAILED'}")
    print("=" * 60)

    if _PASS:
        print("\nConstants to update in all 3 sim_config.py files:")
        print(f"  WHEEL_KV_MEASURED  = {args.kv}")
        print(f"  WHEEL_KT           = {kt:.4f}   # Nm/A")
        print(f"  WHEEL_TORQUE_LIMIT = {t_peak:.4f}  # Nm  (Kt x 50A)")
        print(f"  WHEEL_OMEGA_NOLOAD = {omega_noload:.1f}  # rad/s  (KV x 24V x 2pi/60)")

    return 0 if _PASS else 1


if __name__ == "__main__":
    sys.exit(main())
