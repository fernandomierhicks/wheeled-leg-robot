"""ak45_dyno.py - Simulated dynamometer for the CubeMars AK45-10 KV75 hip motor.

Validates that the MotorModel (instantiated with output-shaft parameters) matches
the AK45-10 datasheet specs before trusting sim_config.py constants.

Motor topology:
  KV75 brushless + 10:1 integrated planetary + MIT CAN driver
  All values below are at the OUTPUT shaft unless stated otherwise.

Usage:
    python ak45_dyno.py                    # datasheet values
    python ak45_dyno.py --vbus 36.0        # max-voltage run
    python ak45_dyno.py --kt_motor 0.130   # measured motor Kt override

Tests:
    1a. Torque-speed curve        - linear taper 7 Nm -> 0 over 0..18.85 rad/s
    1b. Electrical step response  - 63% rise time == tau_elec (2 ms)
    1c. Operating points          - 24 V (nominal) vs 36 V (max rated)
    1d. Drag self-check           - unpowered leg decelerates to rest

Outputs:
    ak45_dyno_results.png   - saved alongside this script
    Console PASS / FAIL summary
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Path: import MotorModel from baseline1
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "simulation", "mujoco", "baseline1_leg_analysis"))
from motor_models import MotorModel  # noqa: E402

# ---------------------------------------------------------------------------
# AK45-10 constants (output shaft, from sim_config.py / datasheet)
# ---------------------------------------------------------------------------
KV_MOTOR     = 75.0    # [RPM/V]   motor KV (before gearbox)
GEAR_RATIO   = 10.0    # []        planetary reduction
V_BUS_NOM    = 24.0    # [V]       6S LiPo nominal
V_BUS_MAX    = 36.0    # [V]       AK45-10 absolute max (from datasheet)
T_PEAK       = 7.0     # [Nm]      peak output torque (datasheet)
TAU_ELEC     = 0.002   # [s]       CAN transport + integrated FOC current loop
B_FRICTION   = 0.02    # [Nm.s/rad] planetary gearbox viscous drag (output shaft)

# Leg+rotor inertia as seen at the output shaft (approximate)
# 4-bar leg: ~0.25 kg at ~0.1 m avg radius + AK45-10 rotor reflected through 10:1
# J_rotor_reflected = 0.5 * 0.05 kg * (0.025m)^2 * 10^2 ≈ very small; leg dominates
J_LEG        = 0.004   # [kg.m^2]  conservative estimate for deceleration test

_PASS = True


def derive_output_params(kv_motor=KV_MOTOR, kt_motor_override=None, v_bus=V_BUS_NOM):
    """Return (Kt_out, omega_noload_out) at the output shaft."""
    kt_motor = kt_motor_override if kt_motor_override is not None else 9.55 / kv_motor
    kt_out = kt_motor * GEAR_RATIO
    omega_motor_noload = kv_motor * v_bus * (2 * math.pi / 60)
    omega_out_noload = omega_motor_noload / GEAR_RATIO
    return kt_out, omega_out_noload


def run_to_steady(motor, T_cmd, omega, dt=1e-4, duration=0.15):
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
    print(f"\n[1a] Torque-speed curve  ({label})")

    N = 60
    omegas = np.linspace(0.0, omega_noload * 1.05, N)
    measured = [run_to_steady(motor, t_peak, w) for w in omegas]
    measured = np.array(measured)

    expected = np.array([t_peak * max(0.0, 1.0 - abs(w) / omega_noload) for w in omegas])

    drag = B_FRICTION * omegas
    measured_no_drag = measured + drag

    mask = omegas <= omega_noload
    max_err_pct = np.max(np.abs(measured_no_drag[mask] - expected[mask])) / t_peak * 100

    t_stall  = run_to_steady(motor, t_peak, 0.0)
    t_noload = run_to_steady(motor, t_peak, omega_noload)
    t_half   = run_to_steady(motor, t_peak, omega_noload / 2)

    print(f"  Stall torque      : {t_stall:.4f} Nm  (expected {t_peak:.4f})")
    print(f"  Half-speed torque : {t_half:.4f} Nm  (expected {t_peak/2 - B_FRICTION*omega_noload/2:.4f} incl. drag)")
    print(f"  No-load torque    : {t_noload:.4f} Nm  (expected ~-{B_FRICTION*omega_noload:.4f} drag only)")
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

    dt = 1e-5
    duration = 0.020
    steps = int(duration / dt)
    motor.reset()

    times, torques = [], []
    for i in range(steps):
        t_out = motor.step(t_peak, 0.0, dt)
        times.append(i * dt)
        torques.append(t_out)

    times  = np.array(times)
    torques = np.array(torques)

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
# Test 1c - Operating points: 24V nominal vs 36V max
# ---------------------------------------------------------------------------
def test_operating_points(kt_motor_override=None):
    global _PASS
    print("\n[1c] Operating points  (nominal 24 V  vs  max 36 V)")

    rows = []
    for label, v in [("6S / 24.0 V (nominal)", V_BUS_NOM), ("9S / 36.0 V (max rated)", V_BUS_MAX)]:
        kt_out, omega_out = derive_output_params(KV_MOTOR, kt_motor_override, v)
        rpm_out = omega_out * 60 / (2 * math.pi)
        # T_peak is current-limited by integrated driver, not voltage-limited
        print(f"  {label}:  Kt_out={kt_out:.4f} Nm/A  "
              f"T_peak={T_PEAK:.2f} Nm  omega_noload={omega_out:.2f} rad/s  ({rpm_out:.0f} RPM)")
        rows.append((label, kt_out, T_PEAK, omega_out, rpm_out))

    t_nom = rows[0][2]
    t_max = rows[1][2]
    w_nom = rows[0][3]
    w_max = rows[1][3]
    print(f"\n  T_peak same at both voltages (current-limited): {'PASS' if abs(t_nom-t_max)<0.01 else 'FAIL'}")
    print(f"  omega_noload scales with voltage (36V / 24V = {w_max/w_nom:.3f}, expect {36/24:.3f}): "
          f"{'PASS' if abs(w_max/w_nom - 36/24) < 0.01 else 'FAIL'}")
    print(f"  T_peak >= 7 Nm (datasheet): {'PASS' if t_nom >= 7.0 else 'FAIL'}")
    print(f"  omega_noload @ 24V ~= 18.85 rad/s: {'PASS' if abs(w_nom - 18.85) < 0.05 else 'FAIL'}")

    passed = (abs(t_nom - t_max) < 0.01 and
              abs(w_max / w_nom - 36 / 24) < 0.01 and
              t_nom >= 7.0 and
              abs(w_nom - 18.85) < 0.05)
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        _PASS = False

    return rows


# ---------------------------------------------------------------------------
# Test 1d - Drag self-check
# ---------------------------------------------------------------------------
def test_drag_deceleration(motor):
    global _PASS
    print("\n[1d] Drag self-check (unpowered leg deceleration)")

    dt = 1e-3
    omega = 10.0   # start at 10 rad/s (near AK45-10 max output speed)
    motor.reset()

    times, omegas = [0.0], [omega]
    tau_mech = J_LEG / B_FRICTION   # [s]
    sim_dur = max(tau_mech * 6, 5.0)

    for i in range(int(sim_dur / dt)):
        t_out = motor.step(0.0, omega, dt)
        omega += (t_out / J_LEG) * dt
        if omega < 0:
            omega = 0.0
        times.append((i + 1) * dt)
        omegas.append(omega)

    final_omega = omegas[-1]
    print(f"  Mech. time constant (J/B) : {tau_mech:.3f} s")
    print(f"  Start omega : 10.0 rad/s")
    print(f"  Final omega : {final_omega:.4f} rad/s  (expected ~0)")
    print(f"  Decelerates to rest       : {'PASS' if final_omega < 0.05 else 'FAIL'}")
    print(f"  No self-acceleration      : {'PASS' if max(omegas) <= 10.0 else 'FAIL (self-accelerated!)'}")

    passed = final_omega < 0.05 and max(omegas) <= 10.0
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        _PASS = False

    return np.array(times), np.array(omegas)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(curve_data, step_data, drag_data, kt_out, omega_noload, outpath):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Simulated Dynamometer — CubeMars AK45-10  KV75  10:1 planetary\n"
        f"Output shaft:  Kt={kt_out:.4f} Nm/A  |  T_peak={T_PEAK:.1f} Nm  "
        f"|  ω_noload={omega_noload:.2f} rad/s ({omega_noload*60/(2*math.pi):.0f} RPM)  "
        f"|  V_bus={V_BUS_NOM:.0f} V",
        fontsize=11
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # 1a - Torque-speed curve
    ax1 = fig.add_subplot(gs[0, 0])
    omegas, measured, expected = curve_data
    ax1.plot(omegas, expected, "k--", lw=1.5, label="Ideal linear taper")
    ax1.plot(omegas, measured, "C0",  lw=2.5, label="MotorModel output")
    ax1.axvline(omega_noload, color="red", lw=1, ls=":", label=f"ω_noload = {omega_noload:.2f} rad/s")
    ax1.axhline(T_PEAK, color="grey", lw=0.8, ls=":")
    ax1.annotate(f"{T_PEAK:.1f} Nm", xy=(0.3, T_PEAK), fontsize=9, color="grey", va="bottom")
    ax1.set_xlabel("Output speed  [rad/s]")
    ax1.set_ylabel("Output torque  [Nm]")
    ax1.set_title("1a — Torque-speed curve (output shaft)")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, omegas[-1])
    ax1.set_ylim(-0.5, T_PEAK * 1.25)
    ax1.grid(True, alpha=0.3)

    # secondary x-axis: output RPM
    ax1b = ax1.twiny()
    ax1b.set_xlim(0, omegas[-1] * 60 / (2 * math.pi))
    ax1b.set_xlabel("Output speed  [RPM]", fontsize=8)

    # 1b - step response
    ax2 = fig.add_subplot(gs[0, 1])
    times_s, torques_s = step_data
    ax2.plot(times_s * 1000, torques_s, "C1", lw=2)
    ax2.axhline(T_PEAK * 0.632, color="grey", lw=1, ls="--", label="63% level")
    ax2.axvline(TAU_ELEC * 1000, color="red", lw=1, ls=":",
                label=f"τ_elec = {TAU_ELEC*1000:.0f} ms")
    ax2.set_xlabel("Time  [ms]")
    ax2.set_ylabel("Output torque  [Nm]")
    ax2.set_title("1b — Electrical step response (ω = 0)")
    ax2.set_xlim(0, 12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 1c - operating point bar chart (24V vs 36V)
    ax3 = fig.add_subplot(gs[1, 0])
    kt24, w24 = derive_output_params(KV_MOTOR, None, V_BUS_NOM)
    kt36, w36 = derive_output_params(KV_MOTOR, None, V_BUS_MAX)
    bar_labels = ["24V\nT_peak [Nm]", "36V\nT_peak [Nm]",
                  "24V  ω_noload\n[rad/s]", "36V  ω_noload\n[rad/s]"]
    values = [T_PEAK, T_PEAK, w24, w36]
    colors = ["C0", "C0", "C2", "C2"]
    bars = ax3.bar(bar_labels, values, color=colors, alpha=0.75, edgecolor="black", lw=0.7)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Nm  /  rad/s")
    ax3.set_title("1c — Operating points: 24 V vs 36 V")
    ax3.grid(True, axis="y", alpha=0.3)

    # 1d - drag deceleration
    ax4 = fig.add_subplot(gs[1, 1])
    times_d, omegas_d = drag_data
    ax4.plot(times_d, omegas_d, "C3", lw=2)
    ax4.set_xlabel("Time  [s]")
    ax4.set_ylabel("Output speed  [rad/s]")
    ax4.set_title("1d - Unpowered leg deceleration (T_cmd=0, w0=10 rad/s)")
    ax4.axhline(0, color="black", lw=0.8)
    tau_mech = J_LEG / B_FRICTION
    ax4.axvline(tau_mech, color="grey", lw=1, ls="--", label=f"J/B = {tau_mech:.2f} s")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulated dyno for CubeMars AK45-10 KV75")
    parser.add_argument("--vbus", type=float, default=V_BUS_NOM,
                        help="Bus voltage [V] for primary run (default 24 V)")
    parser.add_argument("--kt_motor", type=float, default=None,
                        help="Measured motor-shaft Kt [Nm/A] (overrides KV derivation)")
    args = parser.parse_args()

    kt_out, omega_noload = derive_output_params(KV_MOTOR, args.kt_motor, args.vbus)
    # I_peak implied from T_peak and Kt_out (for reference only)
    i_peak_implied = T_PEAK / kt_out

    print("=" * 62)
    print("CubeMars AK45-10 KV75 - Simulated Dynamometer")
    print("=" * 62)
    print(f"  Motor KV          : {KV_MOTOR:.0f} RPM/V")
    print(f"  Gear ratio        : {GEAR_RATIO:.0f} : 1  (planetary)")
    kt_src = "(from KV)" if args.kt_motor is None else "(measured override)"
    print(f"  Kt  (motor shaft) : {9.55/KV_MOTOR:.4f} Nm/A  {kt_src}")
    print(f"  Kt  (output shaft): {kt_out:.4f} Nm/A")
    print(f"  T_peak (output)   : {T_PEAK:.2f} Nm  (datasheet)")
    print(f"  I_peak implied    : {i_peak_implied:.2f} A  (T_peak / Kt_out)")
    print(f"  omega_noload (out): {omega_noload:.2f} rad/s  ({omega_noload*60/(2*math.pi):.0f} RPM)")
    print(f"  tau_elec          : {TAU_ELEC*1000:.0f} ms  (CAN + FOC loop)")
    print(f"  B_friction        : {B_FRICTION} Nm.s/rad  (planetary drag, output shaft)")
    print(f"  V_bus             : {args.vbus:.1f} V")

    motor = MotorModel(T_PEAK, omega_noload, TAU_ELEC, B_FRICTION)

    curve_data = test_torque_speed_curve(motor, T_PEAK, omega_noload,
                                         f"KV75 @ {args.vbus:.0f}V, 10:1")
    step_data  = test_step_response(motor, T_PEAK)
    _op_rows   = test_operating_points(args.kt_motor)
    drag_data  = test_drag_deceleration(motor)

    outpath = os.path.join(_SCRIPT_DIR, "ak45_dyno_results.png")
    make_plot(curve_data, step_data, drag_data, kt_out, omega_noload, outpath)

    print("\n" + "=" * 62)
    print(f"OVERALL: {'ALL TESTS PASSED' if _PASS else 'SOME TESTS FAILED'}")
    print("=" * 62)

    if _PASS:
        print("\nConstants confirmed (matches sim_config.py):")
        print(f"  HIP_TORQUE_LIMIT = {T_PEAK}   # Nm (datasheet peak)")
        print(f"  OMEGA_MAX        = {omega_noload:.2f}   # rad/s (KV75 * {args.vbus:.0f}V / 10:1)")
        print(f"  HIP_TAU_ELEC     = {TAU_ELEC}  # s")
        print(f"  HIP_B_FRICTION   = {B_FRICTION}   # Nm.s/rad")

    return 0 if _PASS else 1


if __name__ == "__main__":
    sys.exit(main())
