"""Collect a repeatable GUI-visible transport baseline through the operator API.

This intentionally observes the exact data path used by the GUI. It never opens
a serial port itself. Example:

    python tools/link_baseline.py --duration 3 --sources teensy esp32 wifi
"""

import argparse
import json
import math
import statistics
import sys
import time

try:
    from .robot_ctl import _request
except ImportError:  # direct script execution: python tools/link_baseline.py
    from robot_ctl import _request


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _counter_snapshot(telem):
    names = (
        "link_crc_drops", "link_seq_gaps", "link_pair_drops",
        "uart_rx_drops", "uart_seq_gaps", "wifi_tx_drops",
        "wifi_rx_drops", "uplink_queue_drops", "loop_max_us",
    )
    return {name: telem[name] for name in names if name in telem}


def measure_source(source, duration_s, sample_hz, rtt_count):
    selected = _request({"cmd": "source_select", "source": source})
    if not selected.get("ok"):
        return {"ok": False, "source": source, "error": selected.get("error", "select failed")}

    time.sleep(0.25)
    samples = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        response = _request({"cmd": "telem"})
        now = time.monotonic()
        if response.get("ok"):
            telem = response.get("telem") or {}
            samples.append({
                "host_s": now,
                "age_ms": response.get("age_ms"),
                "fresh": response.get("fresh"),
                "timestamp_ms": telem.get("timestamp_ms"),
                "state": telem.get("state_name"),
                "fault": telem.get("fault_name"),
                "counters": _counter_snapshot(telem),
            })
        time.sleep(max(0.01, 1.0 / sample_hz))

    rtts = []
    command_errors = []
    for _ in range(rtt_count):
        start = time.perf_counter()
        response = _request({"cmd": "param_get", "id": "buzzer_volume"})
        elapsed = (time.perf_counter() - start) * 1000.0
        if response.get("ok"):
            rtts.append(elapsed)
        else:
            command_errors.append(response.get("error", "unknown error"))

    timestamps = [s["timestamp_ms"] for s in samples if isinstance(s["timestamp_ms"], (int, float))]
    unique_timestamps = list(dict.fromkeys(timestamps))
    telemetry_deltas = [b - a for a, b in zip(unique_timestamps, unique_timestamps[1:]) if b >= a]
    ages = [s["age_ms"] for s in samples if isinstance(s["age_ms"], (int, float))]
    fresh_count = sum(bool(s["fresh"]) for s in samples)
    first_counters = samples[0]["counters"] if samples else {}
    last_counters = samples[-1]["counters"] if samples else {}
    counter_delta = {
        key: last_counters[key] - first_counters.get(key, last_counters[key])
        for key in last_counters
        if isinstance(last_counters[key], (int, float))
    }

    ok = bool(samples and len(unique_timestamps) >= 2 and fresh_count == len(samples) and not command_errors)
    return {
        "ok": ok,
        "source": source,
        "samples": len(samples),
        "fresh_samples": fresh_count,
        "unique_robot_timestamps": len(unique_timestamps),
        "telemetry_period_ms": {
            "median": round(statistics.median(telemetry_deltas), 3) if telemetry_deltas else None,
            "p95": percentile(telemetry_deltas, 0.95),
            "max": max(telemetry_deltas) if telemetry_deltas else None,
        },
        "gui_age_ms": {"p50": percentile(ages, 0.5), "p95": percentile(ages, 0.95), "max": max(ages) if ages else None},
        "command_rtt_ms": {"count": len(rtts), "p50": percentile(rtts, 0.5), "p95": percentile(rtts, 0.95), "max": round(max(rtts), 3) if rtts else None},
        "command_errors": command_errors,
        "counter_start": first_counters,
        "counter_end": last_counters,
        "counter_delta": counter_delta,
        "states_seen": sorted({s["state"] for s in samples if s["state"]}),
        "faults_seen": sorted({s["fault"] for s in samples if s["fault"]}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--rtt-count", type=int, default=5)
    parser.add_argument("--sources", nargs="+", default=["teensy", "esp32", "wifi"])
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    before = _request({"cmd": "source_list"})
    report = {
        "ok": False,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "health": _request({"cmd": "health"}),
        "requested_sources": args.sources,
        "results": [],
    }
    try:
        for source in args.sources:
            report["results"].append(measure_source(source, args.duration, args.sample_hz, args.rtt_count))
    finally:
        original = before.get("override") if before.get("ok") else None
        _request({"cmd": "source_select", "source": original or "auto"})

    available_results = [result for result in report["results"] if "select" not in result.get("error", "")]
    report["ok"] = all(result.get("ok") for result in report["results"]) if args.require_all else any(
        result.get("ok") for result in available_results
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
