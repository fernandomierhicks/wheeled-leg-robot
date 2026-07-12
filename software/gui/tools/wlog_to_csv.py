"""wlog_to_csv.py — decode a .wlog high-datarate SD log into a CSV.

Standalone, no Qt. Reads WlogHeader + repeated LogRecord (t_micros + embedded
235-byte TelemetryPayload blob) and decodes each record with the same
split-telemetry field layout the live GUI uses (telem_format.py), then writes
one row per sample to CSV via pandas.

Usage:
    python wlog_to_csv.py LOG0001.WLOG [out.csv]
"""

import struct
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.telem_format import TELEM_VERSION, decode_telem_full  # noqa: E402

# WlogHeader (32 bytes, packed) — shared/comm_protocol.h
#   char magic[8]; uint8_t format_version, telem_version; uint16_t record_size,
#   sample_rate_hz; uint32_t start_millis; uint8_t reserved[14];
_FMT_HEADER = "<8sBBHHI14s"
assert struct.calcsize(_FMT_HEADER) == 32, "WlogHeader format mismatch"

_T_MICROS_SIZE = 4  # uint32_t t_micros — first field of LogRecord


def read_wlog(path: Path):
    """Yield one dict per LogRecord decoded from the .wlog file at `path`."""
    with open(path, "rb") as f:
        header_bytes = f.read(32)
        if len(header_bytes) < 32:
            raise ValueError(f"{path}: file too short for WlogHeader")
        magic, fmt_ver, telem_ver, record_size, sample_hz, start_ms, _reserved = \
            struct.unpack(_FMT_HEADER, header_bytes)
        magic = magic.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        if magic != "WLRLOG":
            raise ValueError(f"{path}: bad magic {magic!r} — not a .wlog file")

        print(f"{path.name}: format_v{fmt_ver} telem_v{telem_ver} "
              f"record_size={record_size}B sample_rate={sample_hz}Hz start_millis={start_ms}",
              file=sys.stderr)
        if telem_ver != TELEM_VERSION:
            print(f"WARNING: log telem_version={telem_ver} != decoder TELEM_VERSION={TELEM_VERSION} "
                  "— fields may be misread (decoder only supports the current version)",
                  file=sys.stderr)

        telem_size = record_size - _T_MICROS_SIZE
        n = 0
        while True:
            rec = f.read(record_size)
            if len(rec) < record_size:
                if rec:
                    print(f"WARNING: trailing {len(rec)}-byte partial record ignored "
                          "(reset or power loss mid-write?)", file=sys.stderr)
                break
            t_micros = struct.unpack_from("<I", rec, 0)[0]
            telem = decode_telem_full(rec[_T_MICROS_SIZE:_T_MICROS_SIZE + telem_size])
            n += 1
            yield {"t_micros": t_micros, **telem}
        print(f"{path.name}: {n} records decoded", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_path.with_suffix(".csv")

    rows = list(read_wlog(in_path))
    if not rows:
        print("No records decoded — nothing to write.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}", file=sys.stderr)

    # Loop-timing sanity check: t_micros deltas should sit near 1e6/sample_rate_hz
    # (2000 us at 500 Hz), with only rare, small overruns above the 2 ms tick budget.
    dt = df["t_micros"].diff().dropna()
    if not dt.empty:
        print(
            f"t_micros delta [us]: mean={dt.mean():.1f}  p95={dt.quantile(0.95):.1f}  "
            f"max={dt.max():.1f}  min={dt.min():.1f}  n_over_2200us={(dt > 2200).sum()}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
