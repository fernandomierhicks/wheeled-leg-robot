"""automation_runner.py — scripted, unattended WiFi test scenarios driven
through the real GUI transports (WifiTransport, comm_commands) rather than a
separate protocol simulator. Used via `main.py --automation <scenario.json>`
so a test run can be launched headlessly, run for a fixed duration, and exit
with a written report — no human clicking required.

Scenario JSON fields (all optional except duration_s):
    name                  str   — used in the default report filename
    duration_s            float — timed run duration, starting once WiFi telemetry
                                  is first seen (see bootstrap_timeout_s below) —
                                  boot/reconnect time isn't charged against it
    param_dump_period_s   float — interval between CMD_ID_PARAM_GET(0xFFFF) dumps (default 5)
    tcp_churn_period_s    float — if set, force-close/reopen the WiFi TCP command
                                  socket on this interval to reproduce the
                                  '[TCP] client:'-accept-adjacent crash pattern
    bootstrap_timeout_s   float — max time to wait for the first WiFi telemetry
                                  packet before starting the timed window anyway
                                  (default 30)
    snapshot_period_s     float — interval for periodic counter snapshots, so the
                                  report shows a trend over time, not just start/end
                                  totals (default 30)
    corrupt_injections     list — stress-test entries, each {"t_s": float, "target":
                                  0|1, "count": int}: at t_s seconds into the timed
                                  window, sends CMD_ID_TEST_INJECT_CORRUPT asking the
                                  firmware to deliberately flip the CRC-8 on its next
                                  `count` outgoing TELEM_A frames — target 0 = Teensy's
                                  UART send to the ESP32, target 1 = ESP32's WiFi UDP
                                  send to the GUI. Verifies the receiving side's CRC-8
                                  check actually detects and drops a bad frame (watch
                                  injection_results in the report for observed vs.
                                  expected drop deltas), not just "hasn't seen
                                  corruption yet". See comm_commands.send_test_inject_corrupt().
    report_path           str   — output path, relative to software/gui/; default
                                  "logs/<name>_report.json"

Packet-loss measurement note: the GUI's PacketDecoder tracks link_crc_drops/
link_seq_gaps/link_pair_drops per-transport, but TELEM (via g_telem_udp) and
WIFI_DIAG (via g_wifi_diag) are separate CommLink instances with independent
seq counters that both land on the same UDP port/decoder (main.cpp: both
beginSend() to TELEM_UDP_PORT) — a WIFI_DIAG frame interleaved into the TELEM
stream makes the shared seq-gap arithmetic compare two unrelated counters.
So this runner does NOT use link_seq_gaps for loss measurement. Instead it
tracks TELEM's own `loop_count` field (increments by exactly 10 per telemetry
tick at the Teensy's 500 Hz clock) — any bigger jump between two received
TELEM packets means a UDP datagram (or an unpaired TELEM_A/B half) was lost
in transit, independent of any WIFI_DIAG interleaving. link_seq_gaps etc. are
still recorded in the report for reference, clearly labeled as approximate.
"""

import json
import os
import statistics
import time

from PyQt6.QtCore import QObject, QTimer

from .telemetry_bus import TelemetryBus
from .source_manager import SourceManager
from .comm_commands import send_param_get_all, send_test_inject_corrupt
from .wifi_transport import WifiTransport

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # software/gui/
_TELEM_TICK_LOOP_COUNTS = 10  # 500 Hz control loop / 50 Hz telemetry send rate


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class _DeltaCounter:
    """Tracks first/min/max/last of a cumulative counter seen across packets."""
    def __init__(self):
        self.start = None
        self.last = None
        self.min = None
        self.max = None

    def update(self, value):
        if value is None:
            return
        if self.start is None:
            self.start = value
        self.last = value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def delta(self):
        if self.start is None or self.last is None:
            return None
        return self.last - self.start

    def as_dict(self):
        return {"start": self.start, "end": self.last, "delta": self.delta()}


class AutomationRunner(QObject):
    """Forces WiFi as the active transport, drives realistic command traffic
    for a fixed duration, and writes a detailed pass/fail report covering
    telemetry rate/jitter/loss and both directions of the Teensy<->ESP32 UART
    link. Reboot detection is telemetry-based: a drop in wifi_esp_uptime_ms
    between WIFI_DIAG packets means the ESP32 restarted."""

    def __init__(self, scenario_path: str, app):
        super().__init__()
        self._app = app
        with open(scenario_path) as f:
            self._scenario = json.load(f)

        self._name                = self._scenario.get("name", "scenario")
        self._duration_s          = float(self._scenario.get("duration_s", 150))
        self._dump_period_s       = float(self._scenario.get("param_dump_period_s", 5))
        self._churn_period_s      = self._scenario.get("tcp_churn_period_s")
        self._bootstrap_timeout_s = float(self._scenario.get("bootstrap_timeout_s", 30))
        self._snapshot_period_s   = float(self._scenario.get("snapshot_period_s", 30))
        self._corrupt_injections  = self._scenario.get("corrupt_injections", [])

        self._latest: dict = {}
        self._last_uptime_ms: int | None = None
        self._reboot_events: list[dict] = []
        self._churn_count = 0
        self._start_wall = time.time()
        self._timed_window_started = False
        self._wifi_never_connected = False

        # Telemetry rate/jitter/loss (TELEM packets, ptype 0x01, only)
        self._telem_count = 0
        self._telem_arrival_deltas_ms: list[float] = []
        self._last_telem_arrival: float | None = None
        self._first_loop_count: int | None = None
        self._last_loop_count: int | None = None
        self._missed_ticks = 0

        # UART link counters (firmware-computed, immune to the GUI decode
        # interleaving issue described in the module docstring)
        self._uart_esp32_to_teensy_rx  = _DeltaCounter()  # TELEM.uart_rx_drops  (Teensy's view)
        self._uart_esp32_to_teensy_seq = _DeltaCounter()  # TELEM.uart_seq_gaps
        self._uart_teensy_to_esp32_crc = _DeltaCounter()  # WIFI_DIAG.wifi_uart_crc_drops (ESP32's view)
        self._uart_teensy_to_esp32_seq = _DeltaCounter()  # WIFI_DIAG.wifi_uart_seq_gaps

        # Link-up flag transitions (True -> False is the interesting event)
        self._last_esp32_link_ok: bool | None = None
        self._last_teensy_link_up: bool | None = None
        self._link_down_events: list[dict] = []

        # WiFi link quality / GUI-side decode counters (reference only, see docstring)
        self._rssi_samples: list[int] = []
        self._free_heap = _DeltaCounter()
        self._wifi_reconnect = _DeltaCounter()
        self._udp_send_fail = _DeltaCounter()
        self._uplink_queue_drops = _DeltaCounter()
        self._link_crc_drops = _DeltaCounter()
        self._link_seq_gaps = _DeltaCounter()
        self._link_pair_drops = _DeltaCounter()

        self._snapshots: list[dict] = []
        self._injection_results: list[dict] = []
        self._injection_timers: list[QTimer] = []  # kept alive; see _schedule_injections

        self._dump_timer = QTimer(self)
        self._dump_timer.setInterval(max(50, int(self._dump_period_s * 1000)))
        self._dump_timer.timeout.connect(send_param_get_all)

        self._churn_timer = None
        if self._churn_period_s:
            self._churn_timer = QTimer(self)
            self._churn_timer.setInterval(max(50, int(float(self._churn_period_s) * 1000)))
            self._churn_timer.timeout.connect(self._do_churn)

        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setInterval(max(50, int(self._snapshot_period_s * 1000)))
        self._snapshot_timer.timeout.connect(self._take_snapshot)

        self._end_timer = QTimer(self)
        self._end_timer.setSingleShot(True)
        self._end_timer.timeout.connect(self._finish)

        self._bootstrap_timer = QTimer(self)
        self._bootstrap_timer.setSingleShot(True)
        self._bootstrap_timer.setInterval(max(50, int(self._bootstrap_timeout_s * 1000)))
        self._bootstrap_timer.timeout.connect(self._on_bootstrap_timeout)
        self._bootstrap_timer.start()

        TelemetryBus.instance().packet.connect(self._on_packet)
        SourceManager.instance().set_override("wifi")

        print(f"[Automation] '{self._name}' waiting for first WiFi packet "
              f"(timeout {self._bootstrap_timeout_s}s)...", flush=True)

    def _start_timed_window(self):
        if self._timed_window_started:
            return
        self._timed_window_started = True
        self._start_wall = time.time()
        self._dump_timer.start()
        if self._churn_timer is not None:
            self._churn_timer.start()
        self._snapshot_timer.start()
        self._schedule_injections()
        self._end_timer.setInterval(max(50, int(self._duration_s * 1000)))
        self._end_timer.start()
        print(f"[Automation] '{self._name}' timed window started: duration={self._duration_s}s "
              f"dump_period={self._dump_period_s}s churn_period={self._churn_period_s} "
              f"snapshot_period={self._snapshot_period_s}s "
              f"corrupt_injections={len(self._corrupt_injections)}", flush=True)

    def _schedule_injections(self):
        for spec in self._corrupt_injections:
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(max(0, int(float(spec.get("t_s", 0)) * 1000)))
            t.timeout.connect(lambda spec=spec: self._run_injection(spec))
            t.start()
            self._injection_timers.append(t)

    def _run_injection(self, spec: dict):
        target = int(spec.get("target", 0))
        count = int(spec.get("count", 1))
        # target 0 (UART): ESP32's own WIFI_DIAG.wifi_uart_crc_drops is the
        # ground truth for "did the ESP32's CRC check catch it".
        # target 1 (WiFi): this GUI's own link_crc_drops (PacketDecoder for
        # the "wifi" transport) is the ground truth — see comm_commands.
        counter = self._uart_teensy_to_esp32_crc if target == 0 else self._link_crc_drops
        before = counter.last
        print(f"[Automation] injecting {count} corrupt frame(s), target="
              f"{'UART' if target == 0 else 'WiFi'}, crc_drops before={before}", flush=True)
        send_test_inject_corrupt(count, target)

        def check():
            after = counter.last
            observed = None if (before is None or after is None) else after - before
            result = {
                "t_s": round(time.time() - self._start_wall, 2),
                "target": "uart" if target == 0 else "wifi",
                "expected_count": count,
                "crc_drops_before": before,
                "crc_drops_after": after,
                "observed_delta": observed,
                "detected_all": observed == count,
            }
            self._injection_results.append(result)
            print(f"[Automation] injection result: {result}", flush=True)

        check_timer = QTimer(self)
        check_timer.setSingleShot(True)
        check_timer.setInterval(2000)  # give the frame(s) time to send + be counted
        check_timer.timeout.connect(check)
        check_timer.start()
        self._injection_timers.append(check_timer)

    def _on_bootstrap_timeout(self):
        if self._timed_window_started:
            return
        print(f"[Automation] '{self._name}': no WiFi telemetry seen within "
              f"{self._bootstrap_timeout_s}s — starting timed window anyway.", flush=True)
        self._wifi_never_connected = True
        self._start_timed_window()

    def _on_packet(self, info: dict):
        self._latest.update({k: v for k, v in info.items() if v is not None})

        if not self._timed_window_started and info.get("wifi_esp_uptime_ms") is not None:
            self._bootstrap_timer.stop()
            self._start_timed_window()

        uptime = info.get("wifi_esp_uptime_ms")
        if uptime is not None:
            if self._last_uptime_ms is not None and uptime < self._last_uptime_ms:
                self._reboot_events.append({
                    "t_wall_s":       round(time.time() - self._start_wall, 2),
                    "prev_uptime_ms": self._last_uptime_ms,
                    "new_uptime_ms":  uptime,
                })
            self._last_uptime_ms = uptime

        if not self._timed_window_started:
            return  # don't count pre-bootstrap traffic toward the timed stats

        ptype = info.get("ptype")

        if ptype == 0x01:  # merged TELEM (TELEM_A + TELEM_B, correctly paired)
            self._telem_count += 1
            now = time.monotonic()
            if self._last_telem_arrival is not None:
                self._telem_arrival_deltas_ms.append((now - self._last_telem_arrival) * 1000.0)
            self._last_telem_arrival = now

            lc = info.get("loop_count")
            if lc is not None:
                if self._first_loop_count is None:
                    self._first_loop_count = lc
                if self._last_loop_count is not None:
                    step = (lc - self._last_loop_count) & 0xFFFFFFFF
                    if step > _TELEM_TICK_LOOP_COUNTS:
                        self._missed_ticks += round(step / _TELEM_TICK_LOOP_COUNTS) - 1
                self._last_loop_count = lc

            self._uart_esp32_to_teensy_rx.update(info.get("uart_rx_drops"))
            self._uart_esp32_to_teensy_seq.update(info.get("uart_seq_gaps"))

            link_ok = info.get("esp32_link_ok")
            if link_ok is not None:
                if self._last_esp32_link_ok is True and link_ok is False:
                    self._link_down_events.append({
                        "t_wall_s": round(time.time() - self._start_wall, 2),
                        "which": "esp32_link_ok (Teensy's view of ESP32 heartbeat)",
                    })
                self._last_esp32_link_ok = link_ok

        elif ptype == 0x14:  # WIFI_DIAG
            self._uart_teensy_to_esp32_crc.update(info.get("wifi_uart_crc_drops"))
            self._uart_teensy_to_esp32_seq.update(info.get("wifi_uart_seq_gaps"))
            self._free_heap.update(info.get("wifi_free_heap"))
            self._wifi_reconnect.update(info.get("wifi_reconnect_count"))
            self._udp_send_fail.update(info.get("wifi_udp_send_fail_count"))
            self._uplink_queue_drops.update(info.get("wifi_uplink_queue_drops"))
            rssi = info.get("wifi_rssi_dbm")
            if rssi:
                self._rssi_samples.append(rssi)

            link_up = info.get("wifi_teensy_link_up")
            if link_up is not None:
                if self._last_teensy_link_up is True and link_up is False:
                    self._link_down_events.append({
                        "t_wall_s": round(time.time() - self._start_wall, 2),
                        "which": "wifi_teensy_link_up (ESP32's view of Teensy heartbeat)",
                    })
                self._last_teensy_link_up = link_up

        # GUI-side decode counters — reference only, see module docstring
        self._link_crc_drops.update(info.get("link_crc_drops"))
        self._link_seq_gaps.update(info.get("link_seq_gaps"))
        self._link_pair_drops.update(info.get("link_pair_drops"))

    def _take_snapshot(self):
        self._snapshots.append({
            "t_s":                    round(time.time() - self._start_wall, 1),
            "telem_count":            self._telem_count,
            "missed_ticks_cum":       self._missed_ticks,
            "uart_rx_drops":          self._uart_esp32_to_teensy_rx.last,
            "uart_seq_gaps":          self._uart_esp32_to_teensy_seq.last,
            "wifi_uart_crc_drops":    self._uart_teensy_to_esp32_crc.last,
            "wifi_uart_seq_gaps":     self._uart_teensy_to_esp32_seq.last,
            "rssi_dbm":               self._latest.get("wifi_rssi_dbm"),
            "free_heap":              self._latest.get("wifi_free_heap"),
            "esp32_link_ok":          self._last_esp32_link_ok,
            "wifi_teensy_link_up":    self._last_teensy_link_up,
        })

    def _do_churn(self):
        self._churn_count += 1
        WifiTransport.instance().force_reconnect()

    def _finish(self):
        self._take_snapshot()  # final snapshot at end-of-run
        elapsed_s = time.time() - self._start_wall

        deltas = self._telem_arrival_deltas_ms
        sorted_deltas = sorted(deltas)
        jitter = {
            "n":      len(deltas),
            "p50_ms": _percentile(sorted_deltas, 50),
            "p95_ms": _percentile(sorted_deltas, 95),
            "p99_ms": _percentile(sorted_deltas, 99),
            "max_ms": max(sorted_deltas) if sorted_deltas else None,
            "mean_ms": statistics.mean(deltas) if deltas else None,
            "stdev_ms": statistics.pstdev(deltas) if len(deltas) > 1 else None,
        }

        expected_from_loop_count = None
        if self._first_loop_count is not None and self._last_loop_count is not None:
            expected_from_loop_count = (
                (self._last_loop_count - self._first_loop_count) & 0xFFFFFFFF
            ) // _TELEM_TICK_LOOP_COUNTS + 1

        expected_from_duration = round(elapsed_s * 50)
        actual_hz = self._telem_count / elapsed_s if elapsed_s > 0 else None

        reasons = []
        if len(self._reboot_events) > 0:
            reasons.append(f"{len(self._reboot_events)} ESP32 reboot(s) detected")
        if self._wifi_never_connected:
            reasons.append("WiFi telemetry never arrived")
        if self._missed_ticks > 0:
            reasons.append(f"{self._missed_ticks} missed telemetry tick(s) (WiFi datagram loss)")
        if (self._uart_esp32_to_teensy_rx.delta() or 0) > 0:
            reasons.append("UART CRC drops (ESP32->Teensy direction)")
        if (self._uart_esp32_to_teensy_seq.delta() or 0) > 0:
            reasons.append("UART seq gaps (ESP32->Teensy direction)")
        if (self._uart_teensy_to_esp32_crc.delta() or 0) > 0:
            reasons.append("UART CRC drops (Teensy->ESP32 direction)")
        if (self._uart_teensy_to_esp32_seq.delta() or 0) > 0:
            reasons.append("UART seq gaps (Teensy->ESP32 direction)")
        if len(self._link_down_events) > 0:
            reasons.append(f"{len(self._link_down_events)} link-down event(s)")

        report = {
            "name": self._name,
            "duration_s": self._duration_s,
            "actual_elapsed_s": round(elapsed_s, 2),
            "pass": len(reasons) == 0,
            "fail_reasons": reasons,

            "reboot_count": len(self._reboot_events),
            "reboot_events": self._reboot_events,
            "wifi_never_connected": self._wifi_never_connected,
            "tcp_churn_count": self._churn_count,

            "telemetry": {
                "received_count": self._telem_count,
                "expected_count_from_loop_count": expected_from_loop_count,
                "expected_count_from_duration_at_50hz": expected_from_duration,
                "actual_hz": round(actual_hz, 2) if actual_hz else None,
                "missed_ticks": self._missed_ticks,
                "missed_ticks_pct": (
                    round(100 * self._missed_ticks / expected_from_loop_count, 4)
                    if expected_from_loop_count else None
                ),
                "inter_arrival_jitter": jitter,
            },

            "uart_esp32_to_teensy": {  # Teensy's view (TELEM.uart_rx_drops/uart_seq_gaps)
                "rx_drops": self._uart_esp32_to_teensy_rx.as_dict(),
                "seq_gaps": self._uart_esp32_to_teensy_seq.as_dict(),
            },
            "uart_teensy_to_esp32": {  # ESP32's view (WIFI_DIAG.wifi_uart_crc_drops/wifi_uart_seq_gaps)
                "crc_drops": self._uart_teensy_to_esp32_crc.as_dict(),
                "seq_gaps": self._uart_teensy_to_esp32_seq.as_dict(),
            },
            "link_down_events": self._link_down_events,

            "wifi_link_quality": {
                "rssi_dbm_min": min(self._rssi_samples) if self._rssi_samples else None,
                "rssi_dbm_max": max(self._rssi_samples) if self._rssi_samples else None,
                "rssi_dbm_mean": round(statistics.mean(self._rssi_samples), 1) if self._rssi_samples else None,
                "free_heap": self._free_heap.as_dict(),
                "wifi_reconnect_count": self._wifi_reconnect.as_dict(),
                "udp_send_fail_count": self._udp_send_fail.as_dict(),
                "uplink_queue_drops": self._uplink_queue_drops.as_dict(),
            },

            "gui_decode_counters_reference_only": {
                "note": "link_crc_drops/link_seq_gaps mix TELEM's and WIFI_DIAG's independent "
                        "seq counters on one decoder (see module docstring) — not used for "
                        "pass/fail, shown for reference only.",
                "link_crc_drops": self._link_crc_drops.as_dict(),
                "link_seq_gaps": self._link_seq_gaps.as_dict(),
                "link_pair_drops": self._link_pair_drops.as_dict(),
            },

            # Deliberate stress-test corruption (see corrupt_injections in the module
            # docstring). NOT folded into "pass"/fail_reasons above: a successful
            # injection *should* show up as a real crc_drops/missed_ticks increment
            # (that's the corrupted frame being correctly detected and dropped) —
            # judge each injection by its own detected_all flag here instead.
            "injection_results": self._injection_results,

            "snapshots": self._snapshots,
        }
        rel_path = self._scenario.get("report_path") or f"logs/{self._name}_report.json"
        out_path = os.path.join(_GUI_DIR, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Automation] Report written to {out_path}", flush=True)
        print(json.dumps({k: v for k, v in report.items() if k != "snapshots"}, indent=2), flush=True)
        # Stop the background UDP thread before quit() starts deleting Qt objects
        # it's still emitting into — see WifiTransport._stop_requested.
        WifiTransport.instance().stop()
        self._app.quit()
