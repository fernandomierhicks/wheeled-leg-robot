"""Analyze Tests 1 & 2 datarate data: effective Hz, jitter, inter-sample histogram.

Supports two log formats:
  - Test 1 (GRV):  columns idx, timestamp_us, qi, qj, qk, qr
  - Test 2 (raw):  columns idx, timestamp_us, type, x, y, z

Usage:
    python analyze_datarate.py <logfile> [--test 1|2]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_log import parse_csv


def analyze_datarate(filepath, test_num=1):
    label = f"TEST{test_num}"
    print("=" * 60)
    print(f"Test {test_num}: {'GRV' if test_num == 1 else 'Raw Gyro/Accel'} Datarate")
    print("=" * 60)

    df = parse_csv(filepath, label=label)
    print(f"  Samples: {len(df)}")

    t_us = df["timestamp_us"].values
    t_s = (t_us - t_us[0]) / 1e6
    duration_s = t_s[-1]
    n = len(t_us)

    # Inter-sample deltas
    dt_us = np.diff(t_us).astype(np.float64)
    dt_ms = dt_us / 1e3

    effective_hz = (n - 1) / duration_s
    print(f"  Duration: {duration_s:.2f} s")
    print(f"  Effective rate: {effective_hz:.1f} Hz")

    # Statistics
    print(f"\n  Inter-sample delta (ms):")
    print(f"    Mean:   {np.mean(dt_ms):.3f}")
    print(f"    Median: {np.median(dt_ms):.3f}")
    print(f"    Std:    {np.std(dt_ms):.3f}")
    print(f"    Min:    {np.min(dt_ms):.3f}")
    print(f"    Max:    {np.max(dt_ms):.3f}")
    print(f"    P5:     {np.percentile(dt_ms, 5):.3f}")
    print(f"    P95:    {np.percentile(dt_ms, 95):.3f}")
    print(f"    P99:    {np.percentile(dt_ms, 99):.3f}")

    # Bin into 1ms buckets for distribution table
    dt_rounded = np.round(dt_us / 1000).astype(int)
    unique, counts = np.unique(dt_rounded, return_counts=True)
    print(f"\n  Inter-sample distribution (1ms bins):")
    print(f"    {'Delta (ms)':<12} {'Count':<8} {'Percentage':<10} ")
    for val, cnt in zip(unique, counts):
        pct = 100.0 * cnt / len(dt_rounded)
        if pct >= 0.5:  # only show bins with >=0.5%
            print(f"    {val:<12} {cnt:<8} {pct:.1f}%")

    # --- Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Histogram
    bin_edges = np.arange(0, np.max(dt_ms) + 1, 0.5)
    axes[0].hist(dt_ms, bins=bin_edges, edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Inter-sample delta (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Test {test_num} — Inter-sample Timing Histogram ({effective_hz:.1f} Hz effective)")
    axes[0].axvline(np.median(dt_ms), color="r", linestyle="--", label=f"median = {np.median(dt_ms):.2f} ms")
    axes[0].legend()

    # Time series of deltas
    axes[1].plot(t_s[1:], dt_ms, linewidth=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Delta (ms)")
    axes[1].set_title("Inter-sample Delta Over Time")
    axes[1].axhline(np.median(dt_ms), color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = Path(filepath).parent / f"datarate_test{test_num}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")
    plt.show()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Implications")
    print("=" * 60)
    print(f"  Effective rate: {effective_hz:.1f} Hz")
    if test_num == 1:
        print(f"  At 500 Hz control loop: ~{100*effective_hz/500:.0f}% of cycles get fresh GRV data")
        print(f"  Jitter (std): {np.std(dt_ms):.2f} ms — ", end="")
        if np.std(dt_ms) < 1.0:
            print("acceptable for control loop")
        else:
            print("significant — consider timestamp-based interpolation")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_datarate.py <logfile> [--test 1|2]")
        sys.exit(1)

    filepath = sys.argv[1]
    test_num = 1
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        test_num = int(sys.argv[idx + 1])

    analyze_datarate(filepath, test_num)


if __name__ == "__main__":
    main()
