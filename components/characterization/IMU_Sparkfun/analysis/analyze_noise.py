"""Analyze Test 5 noise data: GRV fused pitch/roll noise + raw gyro/accel noise.

Outputs:
  - GRV-derived pitch/roll std-dev (→ pitch_std_rad, roll_std_rad)
  - Raw gyro noise std-dev per axis (→ pitch_rate_std_rad_s)
  - Raw accel noise std-dev per axis (→ accel_std)
  - Allan deviation plots (gyro + accel)
  - Power spectral density plots (gyro + accel)

Usage:
    python analyze_noise.py <logfile>
"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from parse_log import parse_csv


# --- Unit conversions (BNO086 SH-2 reference manual) ---
GYRO_LSB_TO_DEG_S = 1.0 / 16.0       # Table 6-22
ACCEL_LSB_TO_G = 1.0 / 8192.0         # Table 6-20
G_TO_MS2 = 9.80665


def quat_to_euler(qi, qj, qk, qr):
    """Quaternion (i,j,k,real) → (roll, pitch, yaw) in radians.

    Uses aerospace convention matching the sim:
      pitch = asin(2(qr*qj - qk*qi))
      roll  = atan2(2(qr*qi + qj*qk), 1 - 2(qi² + qj²))
      yaw   = atan2(2(qr*qk + qi*qj), 1 - 2(qj² + qk²))
    """
    sinp = 2.0 * (qr * qj - qk * qi)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    roll = np.arctan2(2.0 * (qr * qi + qj * qk), 1.0 - 2.0 * (qi**2 + qj**2))
    yaw = np.arctan2(2.0 * (qr * qk + qi * qj), 1.0 - 2.0 * (qj**2 + qk**2))
    return roll, pitch, yaw


def allan_deviation(data, dt):
    """Overlapping Allan deviation.

    Returns (taus, adevs) arrays.
    On a log-log plot:
      slope = -0.5  →  angle/velocity random walk (white noise)
      slope =  0    →  bias instability (flicker noise)
    """
    N = len(data)
    max_m = N // 4
    if max_m < 2:
        return np.array([]), np.array([])

    taus = []
    adevs = []
    for k in range(1, int(np.log2(max_m)) + 1):
        m = 2**k
        tau = m * dt
        # Truncate to multiple of m and reshape into clusters
        n_clusters = N // m
        truncated = data[: n_clusters * m].reshape(n_clusters, m)
        cluster_means = truncated.mean(axis=1)
        # Allan variance = 0.5 * mean of squared first-differences
        diffs = np.diff(cluster_means)
        adev = np.sqrt(0.5 * np.mean(diffs**2))
        taus.append(tau)
        adevs.append(adev)

    return np.array(taus), np.array(adevs)


def analyze_grv(filepath):
    """Analyze Phase A: GRV fused noise."""
    print("=" * 60)
    print("Phase A: GRV Fused Noise")
    print("=" * 60)

    df = parse_csv(filepath, label="TEST5A")
    print(f"  Samples: {len(df)}")

    t_us = df["timestamp_us"].values
    dt_s = np.median(np.diff(t_us)) / 1e6
    rate_hz = 1.0 / dt_s
    print(f"  Effective rate: {rate_hz:.1f} Hz (dt = {dt_s*1e3:.2f} ms)")

    qi = df["qi"].values
    qj = df["qj"].values
    qk = df["qk"].values
    qr = df["qr"].values

    roll, pitch, yaw = quat_to_euler(qi, qj, qk, qr)

    # Remove mean (we care about noise around the mean, not absolute angle)
    pitch_centered = pitch - np.mean(pitch)
    roll_centered = roll - np.mean(roll)

    pitch_std_rad = np.std(pitch_centered)
    roll_std_rad = np.std(roll_centered)
    yaw_std_rad = np.std(yaw - np.mean(yaw))

    print(f"\n  Pitch noise std: {np.degrees(pitch_std_rad):.4f} deg  ({pitch_std_rad:.6f} rad)")
    print(f"  Roll  noise std: {np.degrees(roll_std_rad):.4f} deg  ({roll_std_rad:.6f} rad)")
    print(f"  Yaw   noise std: {np.degrees(yaw_std_rad):.4f} deg  (expected to be larger — no mag)")

    # Time series plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t_s = (t_us - t_us[0]) / 1e6

    axes[0].plot(t_s, np.degrees(pitch_centered), linewidth=0.3)
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].set_title(f"GRV Pitch Noise — std = {np.degrees(pitch_std_rad):.4f} deg")
    axes[0].axhline(0, color="gray", linewidth=0.5)

    axes[1].plot(t_s, np.degrees(roll_centered), linewidth=0.3, color="C1")
    axes[1].set_ylabel("Roll (deg)")
    axes[1].set_title(f"GRV Roll Noise — std = {np.degrees(roll_std_rad):.4f} deg")
    axes[1].axhline(0, color="gray", linewidth=0.5)

    axes[2].plot(t_s, np.degrees(yaw - np.mean(yaw)), linewidth=0.3, color="C2")
    axes[2].set_ylabel("Yaw (deg)")
    axes[2].set_title(f"GRV Yaw Noise — std = {np.degrees(yaw_std_rad):.4f} deg")
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_grv_timeseries.png", dpi=150)
    plt.show()

    return pitch_std_rad, roll_std_rad


def analyze_gyro(filepath):
    """Analyze Phase B: Raw gyro noise (gyro-only capture)."""
    print("\n" + "=" * 60)
    print("Phase B: Raw Gyro Noise (gyro-only)")
    print("=" * 60)

    df = parse_csv(filepath, label="TEST5B")
    print(f"  Samples: {len(df)}")

    t_us = df["timestamp_us"].values
    dt_s = np.median(np.diff(t_us)) / 1e6
    rate_hz = 1.0 / dt_s
    print(f"  Effective rate: {rate_hz:.1f} Hz (dt = {dt_s*1e3:.2f} ms)")

    # Convert raw LSBs to SI units
    gx_dps = df["gx"].values * GYRO_LSB_TO_DEG_S
    gy_dps = df["gy"].values * GYRO_LSB_TO_DEG_S
    gz_dps = df["gz"].values * GYRO_LSB_TO_DEG_S
    gx_rads = np.radians(gx_dps)
    gy_rads = np.radians(gy_dps)
    gz_rads = np.radians(gz_dps)

    # Remove mean (bias) — we measure noise around the mean
    gx_c = gx_dps - np.mean(gx_dps)
    gy_c = gy_dps - np.mean(gy_dps)
    gz_c = gz_dps - np.mean(gz_dps)

    # Noise std-dev
    gyro_std_dps = [np.std(gx_c), np.std(gy_c), np.std(gz_c)]
    gyro_std_rads = [np.std(gx_rads - np.mean(gx_rads)),
                     np.std(gy_rads - np.mean(gy_rads)),
                     np.std(gz_rads - np.mean(gz_rads))]

    print("\n  Gyro noise std (deg/s):")
    for ax_name, s in zip(["X", "Y", "Z"], gyro_std_dps):
        print(f"    {ax_name}: {s:.4f} deg/s")
    print(f"    RMS: {np.sqrt(np.mean(np.array(gyro_std_dps)**2)):.4f} deg/s")

    print("\n  Gyro noise std (rad/s):")
    for ax_name, s in zip(["X", "Y", "Z"], gyro_std_rads):
        print(f"    {ax_name}: {s:.6f} rad/s")

    # --- Allan Deviation ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for label, data in zip(["Gx", "Gy", "Gz"], [gx_dps, gy_dps, gz_dps]):
        taus, adevs = allan_deviation(data - np.mean(data), dt_s)
        if len(taus) > 0:
            ax.loglog(taus, adevs, "o-", label=label, markersize=3)

    ax.set_xlabel("Cluster time τ (s)")
    ax.set_ylabel("Allan deviation (deg/s)")
    ax.set_title("Gyro Allan Deviation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    if len(taus) > 0:
        ref_tau = taus[0]
        ref_adev = adevs[0]
        tau_ref = np.array([taus[0], taus[-1]])
        ax.loglog(tau_ref, ref_adev * np.sqrt(ref_tau / tau_ref),
                  "--", color="gray", alpha=0.5, label="slope=-0.5 (white)")

    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_gyro_allan.png", dpi=150)
    plt.show()

    # --- PSD ---
    from scipy.signal import welch
    fs = rate_hz
    nperseg = min(1024, len(gx_dps) // 4)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for label, data in zip(["Gx", "Gy", "Gz"], [gx_c, gy_c, gz_c]):
        f, psd = welch(data, fs=fs, nperseg=nperseg)
        ax.semilogy(f, np.sqrt(psd), label=label, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Noise density (deg/s/√Hz)")
    ax.set_title("Gyro Power Spectral Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_gyro_psd.png", dpi=150)
    plt.show()

    # --- Time series ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    t_s = (t_us - t_us[0]) / 1e6
    for label, data, color in zip(["Gx", "Gy", "Gz"], [gx_c, gy_c, gz_c],
                                   ["C0", "C1", "C2"]):
        ax.plot(t_s, data, linewidth=0.2, color=color, label=label, alpha=0.7)
    ax.set_ylabel("Gyro (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Raw Gyro Noise (mean-removed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_gyro_timeseries.png", dpi=150)
    plt.show()

    return gyro_std_rads


def analyze_accel(filepath):
    """Analyze Phase C: Raw accel noise (accel-only capture)."""
    print("\n" + "=" * 60)
    print("Phase C: Raw Accel Noise (accel-only)")
    print("=" * 60)

    df = parse_csv(filepath, label="TEST5C")
    print(f"  Samples: {len(df)}")

    t_us = df["timestamp_us"].values
    dt_s = np.median(np.diff(t_us)) / 1e6
    rate_hz = 1.0 / dt_s
    print(f"  Effective rate: {rate_hz:.1f} Hz (dt = {dt_s*1e3:.2f} ms)")

    ax_ms2 = df["ax"].values * ACCEL_LSB_TO_G * G_TO_MS2
    ay_ms2 = df["ay"].values * ACCEL_LSB_TO_G * G_TO_MS2
    az_ms2 = df["az"].values * ACCEL_LSB_TO_G * G_TO_MS2

    ax_c = ax_ms2 - np.mean(ax_ms2)
    ay_c = ay_ms2 - np.mean(ay_ms2)
    az_c = az_ms2 - np.mean(az_ms2)

    accel_std_ms2 = [np.std(ax_c), np.std(ay_c), np.std(az_c)]

    print("\n  Accel noise std (m/s²):")
    for ax_name, s in zip(["X", "Y", "Z"], accel_std_ms2):
        print(f"    {ax_name}: {s:.4f} m/s²")
    print(f"    RMS: {np.sqrt(np.mean(np.array(accel_std_ms2)**2)):.4f} m/s²")

    # --- Allan Deviation ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for label, data in zip(["Ax", "Ay", "Az"], [ax_ms2, ay_ms2, az_ms2]):
        taus_a, adevs_a = allan_deviation(data - np.mean(data), dt_s)
        if len(taus_a) > 0:
            ax.loglog(taus_a, adevs_a, "o-", label=label, markersize=3)
    ax.set_xlabel("Cluster time τ (s)")
    ax.set_ylabel("Allan deviation (m/s²)")
    ax.set_title("Accel Allan Deviation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_accel_allan.png", dpi=150)
    plt.show()

    # --- PSD ---
    from scipy.signal import welch
    fs = rate_hz
    nperseg = min(1024, len(ax_ms2) // 4)

    fig, ax_plt = plt.subplots(1, 1, figsize=(8, 6))
    for label, data in zip(["Ax", "Ay", "Az"], [ax_c, ay_c, az_c]):
        f, psd = welch(data, fs=fs, nperseg=nperseg)
        ax_plt.semilogy(f, np.sqrt(psd), label=label, linewidth=0.8)
    ax_plt.set_xlabel("Frequency (Hz)")
    ax_plt.set_ylabel("Noise density (m/s²/√Hz)")
    ax_plt.set_title("Accel Power Spectral Density")
    ax_plt.legend()
    ax_plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_accel_psd.png", dpi=150)
    plt.show()

    # --- Time series ---
    fig, ax_plt = plt.subplots(1, 1, figsize=(14, 4))
    t_s = (t_us - t_us[0]) / 1e6
    for label, data, color in zip(["Ax", "Ay", "Az"], [ax_c, ay_c, az_c],
                                   ["C0", "C1", "C2"]):
        ax_plt.plot(t_s, data, linewidth=0.2, color=color, label=label, alpha=0.7)
    ax_plt.set_ylabel("Accel (m/s²)")
    ax_plt.set_xlabel("Time (s)")
    ax_plt.set_title("Raw Accel Noise (mean-removed)")
    ax_plt.legend()
    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "noise_accel_timeseries.png", dpi=150)
    plt.show()

    return accel_std_ms2


def print_sim_params(pitch_std_rad, roll_std_rad, gyro_std_rads, accel_std_ms2):
    """Print recommended simulation parameter updates."""
    print("\n" + "=" * 60)
    print("Recommended params.py updates")
    print("=" * 60)

    # For pitch_rate, use Y-axis gyro (pitch axis in our coordinate frame)
    # but report all axes so user can choose
    gyro_rms = math.sqrt(sum(s**2 for s in gyro_std_rads) / 3)
    accel_rms = math.sqrt(sum(s**2 for s in accel_std_ms2) / 3)

    print(f"""
# Measured noise parameters for simulation/mujoco/master_sim/params.py:

class NoiseParams:
    pitch_std_rad     = {pitch_std_rad:.6f}   # {math.degrees(pitch_std_rad):.4f} deg — from GRV fused (was 0.1 deg)
    roll_std_rad      = {roll_std_rad:.6f}   # {math.degrees(roll_std_rad):.4f} deg — from GRV fused (was 0.05 deg)
    pitch_rate_std_rad_s = {gyro_std_rads[1]:.6f}   # Y-axis gyro (was 0.5 deg/s = {math.radians(0.5):.6f} rad/s)
    accel_std         = {accel_rms:.4f}         # RMS across axes (was 0.2 m/s²)

# Individual axis values for reference:
#   Gyro  X={gyro_std_rads[0]:.6f}  Y={gyro_std_rads[1]:.6f}  Z={gyro_std_rads[2]:.6f} rad/s
#   Accel X={accel_std_ms2[0]:.4f}  Y={accel_std_ms2[1]:.4f}  Z={accel_std_ms2[2]:.4f} m/s²
""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_noise.py <logfile>")
        print("  logfile: serial capture from Test 5 (contains TEST5A + TEST5B + TEST5C blocks)")
        sys.exit(1)

    filepath = sys.argv[1]

    pitch_std_rad, roll_std_rad = analyze_grv(filepath)
    gyro_std_rads = analyze_gyro(filepath)
    accel_std_ms2 = analyze_accel(filepath)
    print_sim_params(pitch_std_rad, roll_std_rad, gyro_std_rads, accel_std_ms2)


if __name__ == "__main__":
    main()
