"""Common CSV parser for all IMU characterization serial captures.

Extracts data between ---CSV_START--- and ---CSV_END--- markers.
Supports sub-test labels (e.g. TEST5A, TEST5B) via the `label` parameter.
"""
import re
from io import StringIO
import pandas as pd


def parse_csv(filepath: str, label: str = None) -> pd.DataFrame:
    """Extract CSV data between ---CSV_START--- and ---CSV_END--- markers.

    Args:
        filepath: Path to serial capture log file.
        label: Optional sub-test label (e.g. "TEST5A"). If given, only extract
               from markers prefixed with [label]. If None, extracts the first
               CSV block found.

    Returns:
        pandas DataFrame with columns from the CSV header.
    """
    lines = open(filepath, "r").readlines()
    in_csv = False
    csv_lines = []

    for line in lines:
        stripped = line.strip()

        # Check for start marker, optionally matching label
        if "---CSV_START---" in stripped:
            if label is None or f"[{label}]" in stripped:
                in_csv = True
                continue

        if "---CSV_END---" in stripped:
            if in_csv:
                break  # stop at first matching end marker
            continue

        if in_csv and stripped:
            # Strip [TESTxx] prefix if present
            cleaned = re.sub(r"^\[TEST\w*\]\s*", "", stripped)
            csv_lines.append(cleaned)

    if not csv_lines:
        tag = f" (label={label})" if label else ""
        raise ValueError(f"No CSV data found in {filepath}{tag}")

    return pd.read_csv(StringIO("\n".join(csv_lines)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_log.py <logfile> [label]")
        sys.exit(1)

    filepath = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else None
    df = parse_csv(filepath, label)
    print(f"Parsed {len(df)} rows, columns: {list(df.columns)}")
    print(df.head())
