"""Analyze Test 3 drift data: GRV quaternion → Euler drift rate over 5 minutes.

Outputs:
  - Pitch/roll/yaw drift rate (deg/hr) via linear regression
  - Time series of Euler angles
  - Accuracy timeline

Usage:
    python analyze_drift.py <logfile>
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from parse_log import parse_csv


def quat_to_euler(qi, qj, qk, qr):
    """Quaternion (i,j,k,real) -> (roll, pitch, yaw) in radians."""
    sinp = 2.0 * (qr * qj - qk * qi)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    roll = np.arctan2(2.0 * (qr * qi + qj * qk), 1.0 - 2.0 * (qi**2 + qj**2))
    yaw = np.arctan2(2.0 * (qr * qk + qi * qj), 1.0 - 2.0 * (qj**2 + qk**2))
    return roll, pitch, yaw


def analyze_drift(filepath):
    print("=" * 60)
    print("Test 3: Static GRV Drift (5 min)")
    print("=" * 60)

    df = parse_csv(filepath, label="TEST3")
    print(f"  Samples: {len(df)}")

    t_us = df["timestamp_us"].values
    t_s = (t_us - t_us[0]) / 1e6
    dt_s = np.median(np.diff(t_us)) / 1e6
    rate_hz = 1.0 / dt_s
    duration_s = t_s[-1]
    print(f"  Effective rate: {rate_hz:.1f} Hz (dt = {dt_s*1e3:.2f} ms)")
    print(f"  Duration: {duration_s:.1f} s ({duration_s/60:.1f} min)")

    qi = df["qi"].values
    qj = df["qj"].values
    qk = df["qk"].values
    qr = df["qr"].values
    accuracy = df["accuracy"].values

    roll, pitch, yaw = quat_to_euler(qi, qj, qk, qr)

    # Unwrap yaw to handle +-pi wrapping
    yaw = np.unwrap(yaw)

    # Trim first 60s for drift regression (calibration convergence)
    trim_s = 60.0
    mask = t_s >= trim_s
    t_trim = t_s[mask]
    pitch_trim = pitch[mask]
    roll_trim = roll[mask]
    yaw_trim = yaw[mask]
    print(f"\n  Trimmed first {trim_s:.0f}s — {np.sum(mask)} samples remain for regression")

    # Linear regression: drift = slope (rad/s) → convert to deg/hr
    RAD_S_TO_DEG_HR = np.degrees(1.0) * 3600.0

    results = {}
    for name, data in [("Pitch", pitch_trim), ("Roll", roll_trim), ("Yaw", yaw_trim)]:
        coeffs = np.polyfit(t_trim, data, 1)
        slope_rad_s = coeffs[0]
        drift_deg_hr = slope_rad_s * RAD_S_TO_DEG_HR
        residual = data - np.polyval(coeffs, t_trim)
        noise_std_rad = np.std(residual)
        results[name] = {
            "drift_deg_hr": drift_deg_hr,
            "slope_rad_s": slope_rad_s,
            "noise_std_rad": noise_std_rad,
            "noise_std_deg": np.degrees(noise_std_rad),
        }
        print(f"\n  {name}:")
        print(f"    Drift rate:    {drift_deg_hr:+.3f} deg/hr ({slope_rad_s:+.3e} rad/s)")
        print(f"    Residual std:  {np.degrees(noise_std_rad):.4f} deg ({noise_std_rad:.6f} rad)")

    # Accuracy timeline
    print(f"\n  Accuracy distribution:")
    for level in range(4):
        count = np.sum(accuracy == level)
        pct = 100.0 * count / len(accuracy)
        labels = ["Unreliable", "Low", "Medium", "High"]
        print(f"    {level} ({labels[level]}): {count} ({pct:.1f}%)")

    # --- Plots ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Pitch
    axes[0].plot(t_s, np.degrees(pitch), linewidth=0.3)
    if np.sum(mask) > 1:
        coeffs_p = np.polyfit(t_trim, pitch_trim, 1)
        axes[0].plot(t_trim, np.degrees(np.polyval(coeffs_p, t_trim)), "r-",
                     linewidth=1.5, label=f"fit: {results['Pitch']['drift_deg_hr']:+.3f} deg/hr")
    axes[0].axvline(trim_s, color="gray", linestyle="--", alpha=0.5, label=f"trim @ {trim_s:.0f}s")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].set_title("GRV Pitch Drift")
    axes[0].legend(loc="upper right")

    # Roll
    axes[1].plot(t_s, np.degrees(roll), linewidth=0.3, color="C1")
    if np.sum(mask) > 1:
        coeffs_r = np.polyfit(t_trim, roll_trim, 1)
        axes[1].plot(t_trim, np.degrees(np.polyval(coeffs_r, t_trim)), "r-",
                     linewidth=1.5, label=f"fit: {results['Roll']['drift_deg_hr']:+.3f} deg/hr")
    axes[1].axvline(trim_s, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Roll (deg)")
    axes[1].set_title("GRV Roll Drift")
    axes[1].legend(loc="upper right")

    # Yaw
    axes[2].plot(t_s, np.degrees(yaw), linewidth=0.3, color="C2")
    if np.sum(mask) > 1:
        coeffs_y = np.polyfit(t_trim, yaw_trim, 1)
        axes[2].plot(t_trim, np.degrees(np.polyval(coeffs_y, t_trim)), "r-",
                     linewidth=1.5, label=f"fit: {results['Yaw']['drift_deg_hr']:+.3f} deg/hr")
    axes[2].axvline(trim_s, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Yaw (deg)")
    axes[2].set_title("GRV Yaw Drift (expected — no magnetometer)")
    axes[2].legend(loc="upper right")

    # Accuracy
    axes[3].plot(t_s, accuracy, linewidth=0.5, color="C3")
    axes[3].set_ylabel("Accuracy (0-3)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("BNO086 Calibration Accuracy")
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(["0 Unreliable", "1 Low", "2 Medium", "3 High"])

    plt.tight_layout()
    out_path = Path(filepath).parent / "drift_timeseries.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")
    plt.show()

    # --- Summary for sim params ---
    print("\n" + "=" * 60)
    print("Implications for simulation")
    print("=" * 60)
    p = results["Pitch"]
    r = results["Roll"]
    y = results["Yaw"]
    print(f"  Pitch drift: {p['drift_deg_hr']:+.3f} deg/hr — ", end="")
    if abs(p["drift_deg_hr"]) < 1.0:
        print("negligible for balance control")
    else:
        print("WARNING: significant — may need bias correction")
    print(f"  Roll drift:  {r['drift_deg_hr']:+.3f} deg/hr — ", end="")
    if abs(r["drift_deg_hr"]) < 1.0:
        print("negligible for balance control")
    else:
        print("WARNING: significant — may need bias correction")
    print(f"  Yaw drift:   {y['drift_deg_hr']:+.3f} deg/hr — expected (game rotation vector, no mag)")
    print(f"\n  Pitch residual noise: {p['noise_std_deg']:.4f} deg ({p['noise_std_rad']:.6f} rad)")
    print(f"  Roll  residual noise: {r['noise_std_deg']:.4f} deg ({r['noise_std_rad']:.6f} rad)")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_drift.py <logfile>")
        print("  logfile: serial capture from Test 3 (contains [TEST3] CSV block)")
        sys.exit(1)

    analyze_drift(sys.argv[1])


if __name__ == "__main__":
    main()
