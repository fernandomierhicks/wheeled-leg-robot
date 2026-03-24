"""
Analyze IMU speed characterization logs from the Adafruit BNO08x tests.

Usage:
    python analyze_speed.py <logfile>

Extracts CSV blocks for TEST1 (GRV), TEST2 (raw gyro), TEST3 (cal gyro)
and prints rate statistics + inter-sample jitter.
"""

import sys
import re
import numpy as np
from pathlib import Path


def extract_csv_blocks(text):
    """Extract all [TESTn] CSV blocks from a log file."""
    blocks = {}
    pattern = r'\[TEST(\d)\] ---CSV_START---\n(.*?)\[TEST\1\] ---CSV_END---'
    for match in re.finditer(pattern, text, re.DOTALL):
        test_id = int(match.group(1))
        csv_text = match.group(2).strip()
        # Strip [TESTn] prefixes from lines if present
        lines = []
        for line in csv_text.split('\n'):
            line = re.sub(r'^\[TEST\d\]\s*', '', line).strip()
            if line:
                lines.append(line)
        blocks[test_id] = lines
    return blocks


def analyze_timestamps(test_name, lines):
    """Analyze inter-sample timing from CSV lines with timestamp_us column."""
    header = lines[0].split(',')
    ts_col = header.index('timestamp_us')

    timestamps = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) > ts_col:
            timestamps.append(int(parts[ts_col]))

    if len(timestamps) < 2:
        print(f"  {test_name}: Not enough samples ({len(timestamps)})")
        return

    ts = np.array(timestamps, dtype=np.float64)
    dt = np.diff(ts)  # inter-sample deltas in us

    elapsed_s = (ts[-1] - ts[0]) / 1e6
    effective_hz = (len(ts) - 1) / elapsed_s

    print(f"\n  {test_name}:")
    print(f"    Samples:       {len(ts)}")
    print(f"    Duration:      {elapsed_s:.2f} s")
    print(f"    Effective Hz:  {effective_hz:.1f}")
    print(f"    Mean dt:       {np.mean(dt):.0f} us ({1e6/np.mean(dt):.1f} Hz)")
    print(f"    Std dt:        {np.std(dt):.0f} us")
    print(f"    Min dt:        {np.min(dt):.0f} us")
    print(f"    Max dt:        {np.max(dt):.0f} us")
    print(f"    P5/P50/P95:    {np.percentile(dt,5):.0f} / {np.percentile(dt,50):.0f} / {np.percentile(dt,95):.0f} us")
    print(f"    P99:           {np.percentile(dt,99):.0f} us")

    # Check what % of 500Hz control cycles would get fresh data
    pct_under_2ms = 100.0 * np.sum(dt <= 2000) / len(dt)
    pct_under_3ms = 100.0 * np.sum(dt <= 3000) / len(dt)
    print(f"    dt <= 2ms:     {pct_under_2ms:.1f}%")
    print(f"    dt <= 3ms:     {pct_under_3ms:.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_speed.py <logfile>")
        sys.exit(1)

    logfile = Path(sys.argv[1])
    text = logfile.read_text()

    blocks = extract_csv_blocks(text)

    test_names = {
        1: "Game Rotation Vector (GRV)",
        2: "Raw Gyroscope Y",
        3: "Calibrated Gyroscope Y",
    }

    print(f"=== IMU Speed Analysis: {logfile.name} ===")

    if not blocks:
        print("No CSV blocks found in log file.")
        sys.exit(1)

    for test_id in sorted(blocks.keys()):
        name = test_names.get(test_id, f"Test {test_id}")
        analyze_timestamps(name, blocks[test_id])

    print("\n=== Summary ===")
    for test_id in sorted(blocks.keys()):
        name = test_names.get(test_id, f"Test {test_id}")
        lines = blocks[test_id]
        header = lines[0].split(',')
        ts_col = header.index('timestamp_us')
        timestamps = [int(line.split(',')[ts_col]) for line in lines[1:] if line.split(',')[ts_col].strip()]
        if len(timestamps) >= 2:
            ts = np.array(timestamps, dtype=np.float64)
            hz = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e6)
            print(f"  TEST{test_id} {name:35s} → {hz:.1f} Hz")


if __name__ == "__main__":
    main()
