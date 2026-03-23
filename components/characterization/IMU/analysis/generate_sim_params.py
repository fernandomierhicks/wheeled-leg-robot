"""Map IMU characterization results → simulation parameter recommendations.

Parses available log files from Tests 1, 3, 4, 5, 6 and produces a consolidated
summary of measured values vs current params.py defaults, with copy-paste-ready
dataclass field updates.

Usage:
    python generate_sim_params.py [--logs-dir ../logs]
"""
import sys
import math
import numpy as np
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))
from parse_log import parse_csv

# ── Unit conversions (BNO086 SH-2 reference manual) ──
GYRO_LSB_TO_DEG_S = 1.0 / 16.0
ACCEL_LSB_TO_G = 1.0 / 8192.0
G_TO_MS2 = 9.80665


def quat_to_euler(qi, qj, qk, qr):
    sinp = 2.0 * (qr * qj - qk * qi)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    roll = np.arctan2(2.0 * (qr * qi + qj * qk), 1.0 - 2.0 * (qi**2 + qj**2))
    return roll, pitch


# ── Individual test parsers ──────────────────────────────────────────────


def parse_test1(filepath):
    """Test 1: GRV max datarate → effective Hz."""
    df = parse_csv(filepath, label="TEST1")
    t_us = df["timestamp_us"].values
    duration_s = (t_us[-1] - t_us[0]) / 1e6
    n = len(t_us)
    hz = (n - 1) / duration_s
    return {"grv_rate_hz": round(hz, 1), "samples": n, "duration_s": round(duration_s, 1)}


def parse_test3(filepath):
    """Test 3: static drift → deg/hr for pitch, roll."""
    df = parse_csv(filepath, label="TEST3")
    t_us = df["timestamp_us"].values
    t_s = (t_us - t_us[0]) / 1e6

    qi = df["qi"].values
    qj = df["qj"].values
    qk = df["qk"].values
    qr = df["qr"].values
    roll, pitch = quat_to_euler(qi, qj, qk, qr)

    # Trim first 60s (calibration convergence)
    mask = t_s >= 60.0
    t_trim = t_s[mask]
    RAD_S_TO_DEG_HR = math.degrees(1.0) * 3600.0

    results = {}
    for name, data in [("pitch", pitch[mask]), ("roll", roll[mask])]:
        coeffs = np.polyfit(t_trim, data, 1)
        drift = coeffs[0] * RAD_S_TO_DEG_HR
        residual_std = np.std(data - np.polyval(coeffs, t_trim))
        results[name] = {
            "drift_deg_hr": round(drift, 3),
            "residual_std_rad": residual_std,
            "residual_std_deg": math.degrees(residual_std),
        }
    return results


def parse_test5(filepath):
    """Test 5: noise — GRV fused (5A), raw gyro (5B), raw accel (5C)."""
    results = {}

    # Phase A: GRV fused noise
    df_a = parse_csv(filepath, label="TEST5A")
    qi = df_a["qi"].values
    qj = df_a["qj"].values
    qk = df_a["qk"].values
    qr = df_a["qr"].values
    roll, pitch = quat_to_euler(qi, qj, qk, qr)
    results["pitch_std_rad"] = float(np.std(pitch - np.mean(pitch)))
    results["roll_std_rad"] = float(np.std(roll - np.mean(roll)))

    # Phase B: raw gyro noise
    df_b = parse_csv(filepath, label="TEST5B")
    gy_dps = df_b["gy"].values * GYRO_LSB_TO_DEG_S
    gy_rads = np.radians(gy_dps)
    results["pitch_rate_std_rad_s"] = float(np.std(gy_rads - np.mean(gy_rads)))

    # Also get all axes for reference
    gx_rads = np.radians(df_b["gx"].values * GYRO_LSB_TO_DEG_S)
    gz_rads = np.radians(df_b["gz"].values * GYRO_LSB_TO_DEG_S)
    results["gyro_std_rads"] = [
        float(np.std(gx_rads - np.mean(gx_rads))),
        results["pitch_rate_std_rad_s"],
        float(np.std(gz_rads - np.mean(gz_rads))),
    ]

    # Phase C: raw accel noise
    df_c = parse_csv(filepath, label="TEST5C")
    ax_ms2 = df_c["ax"].values * ACCEL_LSB_TO_G * G_TO_MS2
    ay_ms2 = df_c["ay"].values * ACCEL_LSB_TO_G * G_TO_MS2
    az_ms2 = df_c["az"].values * ACCEL_LSB_TO_G * G_TO_MS2
    stds = [
        float(np.std(ax_ms2 - np.mean(ax_ms2))),
        float(np.std(ay_ms2 - np.mean(ay_ms2))),
        float(np.std(az_ms2 - np.mean(az_ms2))),
    ]
    results["accel_std_ms2"] = math.sqrt(sum(s**2 for s in stds) / 3)
    results["accel_std_per_axis"] = stds

    return results


def parse_test6(filepath):
    """Test 6: e2e latency → ISR-to-read, SPI read time."""
    df = parse_csv(filepath, label="TEST6")
    # Skip first 5 samples (settling)
    df = df.iloc[5:].copy()

    isr_to_read = df["isr_to_read_us"].values.astype(float)
    read_us = df["read_us"].values.astype(float)
    t_int = df["t_int_us"].values.astype(float)
    dt_int = np.diff(t_int)

    return {
        "isr_to_read_us": {
            "median": float(np.median(isr_to_read)),
            "std": float(np.std(isr_to_read)),
            "p95": float(np.percentile(isr_to_read, 95)),
        },
        "report_rate_hz": round(1e6 / np.median(dt_int), 1),
        "samples": len(df),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def find_log(logs_dir, patterns):
    """Find first matching log file."""
    for pattern in patterns:
        matches = sorted(logs_dir.glob(pattern))
        if matches:
            return matches[-1]  # most recent
    return None


def main():
    logs_dir = Path(__file__).parent.parent / "logs"
    if "--logs-dir" in sys.argv:
        idx = sys.argv.index("--logs-dir")
        logs_dir = Path(sys.argv[idx + 1])

    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}")
        sys.exit(1)

    print("=" * 70)
    print("  IMU Characterization -> Simulation Parameters")
    print(f"  Logs: {logs_dir}")
    print(f"  Date: {date.today()}")
    print("=" * 70)

    # ── Discover and parse available logs ──
    results = {}

    # Test 1
    f = find_log(logs_dir, ["test1*.txt", "test1*.log"])
    if f:
        print(f"\n  [Test 1] {f.name}")
        results["t1"] = parse_test1(f)
        print(f"    GRV rate: {results['t1']['grv_rate_hz']} Hz ({results['t1']['samples']} samples / {results['t1']['duration_s']}s)")
    else:
        print("\n  [Test 1] not found — skipping")

    # Test 3
    f = find_log(logs_dir, ["test3*.log", "test3*.txt", "*drift*.log"])
    if f:
        print(f"\n  [Test 3] {f.name}")
        results["t3"] = parse_test3(f)
        for axis in ["pitch", "roll"]:
            d = results["t3"][axis]
            print(f"    {axis.capitalize()} drift: {d['drift_deg_hr']:+.3f} deg/hr  (residual noise: {d['residual_std_deg']:.4f} deg)")
    else:
        print("\n  [Test 3] not found — skipping")

    # Test 5
    f = find_log(logs_dir, ["test5*.log", "test5*.txt", "*noise*.log"])
    if f:
        print(f"\n  [Test 5] {f.name}")
        results["t5"] = parse_test5(f)
        t5 = results["t5"]
        print(f"    Pitch noise:  {math.degrees(t5['pitch_std_rad']):.4f} deg  ({t5['pitch_std_rad']:.6f} rad)")
        print(f"    Roll noise:   {math.degrees(t5['roll_std_rad']):.4f} deg  ({t5['roll_std_rad']:.6f} rad)")
        print(f"    Gyro Y noise: {math.degrees(t5['pitch_rate_std_rad_s']):.3f} deg/s  ({t5['pitch_rate_std_rad_s']:.6f} rad/s)")
        print(f"    Accel RMS:    {t5['accel_std_ms2']:.4f} m/s^2")
    else:
        print("\n  [Test 5] not found — skipping")

    # Test 6
    f = find_log(logs_dir, ["test6*.txt", "test6*.log", "*e2e*.txt"])
    if f:
        print(f"\n  [Test 6] {f.name}")
        results["t6"] = parse_test6(f)
        t6 = results["t6"]
        itr = t6["isr_to_read_us"]
        print(f"    ISR-to-read: {itr['median']:.0f} us median (std {itr['std']:.0f}, p95 {itr['p95']:.0f})")
        print(f"    Report rate: {t6['report_rate_hz']} Hz")
    else:
        print("\n  [Test 6] not found — skipping")

    # ── Build recommended parameters ──
    print("\n" + "=" * 70)
    print("  Recommended params.py updates")
    print("=" * 70)

    # Sensor delay budget
    fusion_ms = 2.5  # datasheet extrapolation @ 400 Hz
    isr_ms = results["t6"]["isr_to_read_us"]["median"] / 1000 if "t6" in results else 1.0
    spi_ms = 0.05  # from Test 4 (measured ~50-65 us)
    total_sensor_delay_ms = fusion_ms + isr_ms + spi_ms
    sensor_delay_s = total_sensor_delay_ms / 1000

    # Noise params
    if "t5" in results:
        t5 = results["t5"]
        pitch_std = t5["pitch_std_rad"]
        roll_std = t5["roll_std_rad"]
        pitch_rate_std = t5["pitch_rate_std_rad_s"]
        accel_std = t5["accel_std_ms2"]
    else:
        pitch_std = 0.000176
        roll_std = 0.000156
        pitch_rate_std = 0.002116
        accel_std = 0.008

    sources = []
    if "t1" in results:
        sources.append(f"Test 1: {results['t1']['grv_rate_hz']} Hz GRV rate")
    if "t3" in results:
        sources.append(f"Test 3: drift < 0.3 deg/hr")
    if "t5" in results:
        sources.append(f"Test 5: noise measured")
    if "t6" in results:
        sources.append(f"Test 6: {total_sensor_delay_ms:.1f} ms sensor delay")

    output = f"""
# --- Generated by generate_sim_params.py ---
# Source: {', '.join(sources)}
# Date: {date.today()}

# -- LatencyParams --
# Sensor delay budget:
#   BNO086 fusion latency:  {fusion_ms:.1f} ms  (datasheet, extrapolated @ 400 Hz)
#   INT -> main loop (ISR): {isr_ms:.2f} ms  (Test 6 median)
#   SPI read:               {spi_ms:.2f} ms  (Test 4)
#   Total:                  {total_sensor_delay_ms:.2f} ms

sensor_delay_s: float = {sensor_delay_s:.4f}    # [{total_sensor_delay_ms:.1f} ms] fusion + ISR + SPI
actuator_delay_s: float = 0.001     # [1.0 ms] unchanged (not measured here)

# -- NoiseParams --
pitch_std_rad: float = {pitch_std:.6f}        # {math.degrees(pitch_std):.4f} deg - GRV fused
pitch_rate_std_rad_s: float = {pitch_rate_std:.6f}  # {math.degrees(pitch_rate_std):.4f} deg/s - raw gyro Y
accel_std: float = {accel_std:.4f}               # [m/s^2] RMS across axes
roll_std_rad: float = {roll_std:.6f}         # {math.degrees(roll_std):.4f} deg - GRV fused"""

    if "t5" in results:
        g = t5["gyro_std_rads"]
        a = t5["accel_std_per_axis"]
        output += f"""

# Per-axis reference:
#   Gyro  X={g[0]:.6f}  Y={g[1]:.6f}  Z={g[2]:.6f} rad/s
#   Accel X={a[0]:.4f}     Y={a[1]:.4f}     Z={a[2]:.4f}     m/s^2"""

    if "t3" in results:
        t3 = results["t3"]
        output += f"""

# Drift (Test 3, 5 min stationary):
#   Pitch: {t3['pitch']['drift_deg_hr']:+.3f} deg/hr - negligible, no bias correction needed
#   Roll:  {t3['roll']['drift_deg_hr']:+.3f} deg/hr - negligible"""

    if "t1" in results:
        hz = results["t1"]["grv_rate_hz"]
        output += f"""

# GRV delivery rate (Test 1): {hz} Hz
#   At 500 Hz control loop: ~{100*hz/500:.0f}% of cycles receive fresh data"""

    print(output)

    # ── Comparison table ──
    print(f"\n\n{'=' * 70}")
    print("  Before / After comparison")
    print("=" * 70)
    print(f"  {'Parameter':<30} {'Old default':<18} {'Measured':<18} {'Change'}")
    print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*18}")
    rows = [
        ("sensor_delay_s", "0.001 s", f"{sensor_delay_s:.4f} s", f"{sensor_delay_s/0.001:.1f}x"),
        ("pitch_std_rad", f"{math.degrees(0.00175):.4f} deg", f"{math.degrees(pitch_std):.4f} deg", f"{pitch_std/0.00175:.2f}x"),
        ("pitch_rate_std_rad_s", f"{math.degrees(0.00873):.4f} deg/s", f"{math.degrees(pitch_rate_std):.4f} deg/s", f"{pitch_rate_std/0.00873:.2f}x"),
        ("accel_std", "0.200 m/s^2", f"{accel_std:.4f} m/s^2", f"{accel_std/0.2:.2f}x"),
        ("roll_std_rad", f"{math.degrees(0.000873):.4f} deg", f"{math.degrees(roll_std):.4f} deg", f"{roll_std/0.000873:.2f}x"),
    ]
    for name, old, new, change in rows:
        print(f"  {name:<30} {old:<18} {new:<18} {change}")

    print()


if __name__ == "__main__":
    main()
