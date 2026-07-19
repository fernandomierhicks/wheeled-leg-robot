"""analyze_wifi_capture.py — offline metrics for wifi_capture.py .jsonl captures.

Standalone, no Qt dependency. Ingests one or more .jsonl captures (+ their
ping/netsh probe sidecars, found next to each .jsonl by wifi_capture.py's
naming convention) and prints/accumulates a comparable per-file summary:

  - Datagram-level loss (seq gaps on the shared telemetry-link seq counter,
    i.e. across TELEM_A + TELEM_B + TELEM_FULL_WIFI, since they share one
    CommLink _seq_tx on the ESP32) vs. tick-level loss (missing/unpaired
    ticks) — the divergence is the quantitative signature of the A/B-pairing
    amplification hypothesis.
  - Out-of-order datagram count.
  - Inter-arrival jitter between complete ticks: p50/p95/p99/max/mean/stdev.
  - Clumping metric: % of ticks arriving <2 ms after the previous one, plus
    the max gap — the DTIM-buffering signature.
  - RSSI-vs-loss correlation (nearest-preceding WIFI_DIAG sample per event).
  - Probe correlation: ping RTT and PC-side RSSI/PHY-rate vs. loss/jitter.
  - WiFi diag summary: RSSI range, channel stability, reconnects, UDP send
    failures, max loop/UDP-send time, observed build_variant_flags.

Usage:
    python analyze_wifi_capture.py logs/wifi_baseline.jsonl [more.jsonl ...] [--csv logs/wifi_summary.csv]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TELEM_LINK_PTYPES = {0x10, 0x11, 0x15}  # share one CommLink seq counter (g_telem_udp)
_DIAG_LINK_PTYPES  = {0x14}              # separate seq counter (g_wifi_diag)

_VARIANT_BITS = [
    (1 << 0, "unicast"), (1 << 1, "combined"), (1 << 2, "tx_power_max"),
    (1 << 3, "neo_disabled"), (1 << 4, "display_disabled"), (1 << 5, "transport_gating"),
]
_TRANSPORT_NAMES = {0: "both/legacy", 1: "usb_only", 2: "wifi_only"}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _seq_gap_stats(df: pd.DataFrame) -> dict:
    """Seq gaps + out-of-order count for one shared-seq-counter link group,
    sorted by arrival time. raw_diff = (cur-prev) mod 256; 0=dup, 1=in-order,
    2..128=forward gap of (raw_diff-1), >128=treated as out-of-order (the
    seq actually went backwards, which wraps to a large positive mod value)."""
    if len(df) < 2:
        return {"datagrams": len(df), "seq_gaps": 0, "out_of_order": 0, "duplicates": 0}
    d = df.sort_values("recv_ts")
    seqs = d["seq"].to_numpy(dtype=int)
    raw_diff = np.diff(seqs) % 256
    gaps = int(np.where((raw_diff >= 2) & (raw_diff <= 128), raw_diff - 1, 0).sum())
    out_of_order = int((raw_diff > 128).sum())
    duplicates = int((raw_diff == 0).sum())
    return {"datagrams": len(df), "seq_gaps": gaps, "out_of_order": out_of_order, "duplicates": duplicates}


def _jitter_stats(deltas_ms: np.ndarray) -> dict:
    if len(deltas_ms) == 0:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None,
                "mean_ms": None, "stdev_ms": None}
    return {
        "p50_ms":   float(np.percentile(deltas_ms, 50)),
        "p95_ms":   float(np.percentile(deltas_ms, 95)),
        "p99_ms":   float(np.percentile(deltas_ms, 99)),
        "max_ms":   float(np.max(deltas_ms)),
        "mean_ms":  float(np.mean(deltas_ms)),
        "stdev_ms": float(np.std(deltas_ms)),
    }


def _completed_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per completed telemetry tick, arrival time = recv_ts of the
    datagram that completed it (TELEM_B with pair_ok, or TELEM_FULL_WIFI)."""
    parts = []
    if "ptype" in df.columns:
        b_ok = df[(df["ptype"] == 0x11) & (df.get("pair_ok") == True)]  # noqa: E712
        if len(b_ok):
            parts.append(b_ok[["recv_ts"]])
        full = df[df["ptype"] == 0x15]
        if len(full):
            parts.append(full[["recv_ts"]])
    if not parts:
        return pd.DataFrame(columns=["recv_ts"])
    return pd.concat(parts, ignore_index=True).sort_values("recv_ts").reset_index(drop=True)


def _expected_tick_count(df: pd.DataFrame) -> int | None:
    """Ground truth from Teensy timestamp_ms (50 Hz -> 20 ms/tick), present on
    both TELEM_A and TELEM_FULL_WIFI decoded rows regardless of variant."""
    if "timestamp_ms" not in df.columns:
        return None
    ts = df["timestamp_ms"].dropna()
    if len(ts) < 2:
        return None
    span_ms = ts.max() - ts.min()
    return int(round(span_ms / 20.0)) + 1


def _diag_summary(diag: pd.DataFrame) -> dict:
    if diag.empty:
        return {}
    out = {
        "rssi_min_dbm": int(diag["rssi_dbm"].min()) if "rssi_dbm" in diag else None,
        "rssi_max_dbm": int(diag["rssi_dbm"].max()) if "rssi_dbm" in diag else None,
        "channel_stable": bool(diag["wifi_channel"].nunique() <= 1) if "wifi_channel" in diag else None,
        "channels_seen": sorted(diag["wifi_channel"].dropna().unique().tolist()) if "wifi_channel" in diag else [],
        "wifi_reconnect_count_max": int(diag["wifi_reconnect_count"].max()) if "wifi_reconnect_count" in diag else None,
        "udp_send_fail_count_max": int(diag["udp_send_fail_count"].max()) if "udp_send_fail_count" in diag else None,
        "loop_max_us_max": int(diag["loop_max_us"].max()) if "loop_max_us" in diag else None,
        "udp_send_max_us_max": int(diag["udp_send_max_us"].max()) if "udp_send_max_us" in diag else None,
    }
    if "build_variant_flags" in diag and len(diag):
        flags = int(diag["build_variant_flags"].iloc[-1])
        out["build_variant_flags"] = flags
        out["build_variant_names"] = [name for bit, name in _VARIANT_BITS if flags & bit] or ["default"]
    if "active_telem_transport" in diag and len(diag):
        transport = int(diag["active_telem_transport"].iloc[-1])
        out["active_telem_transport"] = _TRANSPORT_NAMES.get(transport, str(transport))
    return out


def _rssi_vs_loss(telem_link: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest-preceding RSSI to each seq-gap event, bucketed by 5 dBm."""
    if diag.empty or "rssi_dbm" not in diag.columns or len(telem_link) < 2:
        return pd.DataFrame()
    d = telem_link.sort_values("recv_ts").reset_index(drop=True)
    seqs = d["seq"].to_numpy(dtype=int)
    raw_diff = np.diff(seqs) % 256
    gap_sizes = np.where((raw_diff >= 2) & (raw_diff <= 128), raw_diff - 1, 0)
    gap_events = pd.DataFrame({
        "recv_ts": d["recv_ts"].to_numpy()[1:],
        "lost": gap_sizes,
    })
    gap_events = gap_events[gap_events["lost"] > 0]
    if gap_events.empty:
        return pd.DataFrame()
    diag_sorted = diag.sort_values("recv_ts")
    merged = pd.merge_asof(gap_events.sort_values("recv_ts"), diag_sorted[["recv_ts", "rssi_dbm"]],
                            on="recv_ts", direction="backward")
    merged = merged.dropna(subset=["rssi_dbm"])
    if merged.empty:
        return pd.DataFrame()
    merged["rssi_bucket"] = (merged["rssi_dbm"] // 5 * 5).astype(int)
    return merged.groupby("rssi_bucket")["lost"].sum().sort_index()


_PING_RE  = re.compile(r"time[=<](\d+(?:\.\d+)?)ms")
_LOSS_RE  = re.compile(r"\((\d+)% loss\)")
_SIGNAL_RE = re.compile(r"Signal\s*:\s*(\d+)%")
_RATE_RE   = re.compile(r"Receive rate \(Mbps\)\s*:\s*([\d.]+)")


def _parse_ping_sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    rtts = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PING_RE.search(line)
        if m:
            rtts.append(float(m.group(1)))
    if not rtts:
        return {}
    rtts = np.array(rtts)
    return {
        "ping_rtt_mean_ms": float(rtts.mean()),
        "ping_rtt_p95_ms":  float(np.percentile(rtts, 95)),
        "ping_rtt_max_ms":  float(rtts.max()),
        "ping_samples":     len(rtts),
    }


def _parse_netsh_if_sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    signals = [int(m.group(1)) for m in _SIGNAL_RE.finditer(text)]
    rates = [float(m.group(1)) for m in _RATE_RE.finditer(text)]
    out = {}
    if signals:
        out["pc_signal_pct_mean"] = float(np.mean(signals))
        out["pc_signal_pct_min"]  = int(np.min(signals))
    if rates:
        out["pc_rx_rate_mbps_mean"] = float(np.mean(rates))
        out["pc_rx_rate_mbps_min"]  = float(np.min(rates))
    return out


def analyze_one(path: Path) -> dict:
    df = load_jsonl(path)
    summary: dict = {"file": path.name, "n_events": len(df)}
    if df.empty:
        summary["error"] = "no events decoded"
        return summary

    # .astype(bool) guards against object dtype (True/False/NaN mixed from JSON,
    # which pandas often leaves as object) — `~` on an object Series does a
    # Python-level bitwise invert (~True == -2) instead of logical NOT.
    summary["malformed"] = int(df.get("malformed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    summary["crc_fail"]  = int((~df.get("crc_ok", pd.Series(dtype=bool)).fillna(True).astype(bool)).sum())

    telem_link = df[df["ptype"].isin(_TELEM_LINK_PTYPES)] if "ptype" in df.columns else df.iloc[0:0]
    diag       = df[df["ptype"].isin(_DIAG_LINK_PTYPES)] if "ptype" in df.columns else df.iloc[0:0]

    # Datagram-level loss (shared telemetry-link seq counter)
    dg = _seq_gap_stats(telem_link)
    summary["datagram_count"]      = dg["datagrams"]
    summary["datagram_seq_gaps"]   = dg["seq_gaps"]
    summary["datagram_out_of_order"] = dg["out_of_order"]
    summary["datagram_duplicates"] = dg["duplicates"]
    summary["datagram_loss_pct"] = (
        100.0 * dg["seq_gaps"] / (dg["seq_gaps"] + dg["datagrams"]) if (dg["seq_gaps"] + dg["datagrams"]) else None
    )

    # Tick-level loss (A/B pairing + completeness vs. Teensy timestamp_ms ground truth)
    ticks = _completed_ticks(df)
    pair_b = df[df["ptype"] == 0x11] if "ptype" in df.columns else df.iloc[0:0]
    pair_drops = int((~pair_b.get("pair_ok", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if len(pair_b) else 0
    summary["ab_pair_drops"] = pair_drops if len(pair_b) else None
    expected = _expected_tick_count(df)
    summary["expected_ticks"]  = expected
    summary["completed_ticks"] = len(ticks)
    if expected:
        summary["tick_loss_pct"] = max(0.0, 100.0 * (1 - len(ticks) / expected))
    else:
        summary["tick_loss_pct"] = None

    # Inter-arrival jitter + clumping between completed ticks
    if len(ticks) >= 2:
        deltas_ms = np.diff(ticks["recv_ts"].to_numpy()) * 1000.0
        summary.update({f"jitter_{k}": v for k, v in _jitter_stats(deltas_ms).items()})
        summary["clump_pct_lt_2ms"] = float((deltas_ms < 2.0).mean() * 100.0)
        summary["clump_max_gap_ms"] = float(deltas_ms.max())
    else:
        summary.update({f"jitter_{k}": None for k in
                         ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms", "stdev_ms")})
        summary["clump_pct_lt_2ms"] = None
        summary["clump_max_gap_ms"] = None

    # RSSI-vs-loss correlation
    rssi_loss = _rssi_vs_loss(telem_link, diag)
    if len(rssi_loss):
        worst_bucket = int(rssi_loss.idxmin())
        summary["rssi_bucket_worst_loss_dbm"] = worst_bucket
        summary["rssi_bucket_worst_loss_count"] = int(rssi_loss.min())

    # WiFi diag summary
    summary.update(_diag_summary(diag))

    # Probe sidecars
    summary.update(_parse_ping_sidecar(path.with_suffix(".ping.log")))
    summary.update(_parse_netsh_if_sidecar(path.with_suffix(".netsh_if.log")))

    return summary


def _print_summary(s: dict):
    print(f"\n=== {s['file']} ===")
    if "error" in s:
        print(f"  ERROR: {s['error']}")
        return
    print(f"  events: {s['n_events']}  malformed: {s['malformed']}  crc_fail: {s['crc_fail']}")
    print(f"  datagram-level: {s['datagram_count']} datagrams, "
          f"{s['datagram_seq_gaps']} lost ({s['datagram_loss_pct']:.2f}%), "
          f"{s['datagram_out_of_order']} out-of-order, {s['datagram_duplicates']} dup")
    tl = s.get("tick_loss_pct")
    print(f"  tick-level: {s['completed_ticks']}/{s.get('expected_ticks')} ticks "
          f"({tl:.2f}% lost)" if tl is not None else
          f"  tick-level: {s['completed_ticks']} ticks (expected unknown)")
    if s.get("ab_pair_drops") is not None:
        print(f"  A/B pair drops: {s['ab_pair_drops']}")
    if s.get("jitter_p50_ms") is not None:
        print(f"  jitter: p50={s['jitter_p50_ms']:.1f}ms p95={s['jitter_p95_ms']:.1f}ms "
              f"p99={s['jitter_p99_ms']:.1f}ms max={s['jitter_max_ms']:.1f}ms "
              f"mean={s['jitter_mean_ms']:.1f}ms stdev={s['jitter_stdev_ms']:.1f}ms")
        print(f"  clumping: {s['clump_pct_lt_2ms']:.1f}% of ticks <2ms after previous, "
              f"max gap {s['clump_max_gap_ms']:.0f}ms")
    if s.get("rssi_min_dbm") is not None:
        print(f"  RSSI: {s['rssi_min_dbm']}..{s['rssi_max_dbm']} dBm, "
              f"channel_stable={s.get('channel_stable')}, channels={s.get('channels_seen')}")
        print(f"  reconnects={s.get('wifi_reconnect_count_max')}  "
              f"udp_send_fails={s.get('udp_send_fail_count_max')}  "
              f"loop_max={s.get('loop_max_us_max')}us  udp_send_max={s.get('udp_send_max_us_max')}us")
        print(f"  build variant: {s.get('build_variant_names')}  "
              f"active_transport={s.get('active_telem_transport')}")
    if s.get("rssi_bucket_worst_loss_dbm") is not None:
        print(f"  worst-loss RSSI bucket: {s['rssi_bucket_worst_loss_dbm']} dBm "
              f"({s['rssi_bucket_worst_loss_count']} datagrams lost)")
    if s.get("ping_samples"):
        print(f"  ping RTT: mean={s['ping_rtt_mean_ms']:.1f}ms p95={s['ping_rtt_p95_ms']:.1f}ms "
              f"max={s['ping_rtt_max_ms']:.1f}ms (n={s['ping_samples']})")
    if s.get("pc_signal_pct_mean") is not None:
        print(f"  PC-side WiFi: signal {s['pc_signal_pct_min']}-{s['pc_signal_pct_mean']:.0f}% "
              f"(mean), rx_rate mean={s.get('pc_rx_rate_mbps_mean', 0):.0f} Mbps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="one or more wifi_capture.py .jsonl files")
    ap.add_argument("--csv", type=str, default=None,
                     help="append a comparison row per file to this CSV (created if missing)")
    args = ap.parse_args()

    summaries = []
    for p in args.captures:
        path = Path(p)
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            continue
        s = analyze_one(path)
        summaries.append(s)
        _print_summary(s)

    if args.csv and summaries:
        csv_path = Path(args.csv)
        new_df = pd.DataFrame(summaries)
        if csv_path.exists():
            old_df = pd.read_csv(csv_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(csv_path, index=False)
        print(f"\n[analyze_wifi_capture] appended to {csv_path} ({len(new_df)} row(s))", file=sys.stderr)


if __name__ == "__main__":
    main()
