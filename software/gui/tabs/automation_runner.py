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
    report_path           str   — output path, relative to software/gui/; default
                                  "logs/<name>_report.json"
"""

import json
import os
import time

from PyQt6.QtCore import QObject, QTimer

from .telemetry_bus import TelemetryBus
from .source_manager import SourceManager
from .comm_commands import send_param_get_all
from .wifi_transport import WifiTransport

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # software/gui/


class AutomationRunner(QObject):
    """Forces WiFi as the active transport, drives realistic command traffic
    for a fixed duration, and writes a pass/fail report. Reboot detection is
    telemetry-based: a drop in wifi_esp_uptime_ms between WIFI_DIAG packets
    means the ESP32 restarted — no raw log grepping needed."""

    def __init__(self, scenario_path: str, app):
        super().__init__()
        self._app = app
        with open(scenario_path) as f:
            self._scenario = json.load(f)

        self._name             = self._scenario.get("name", "scenario")
        self._duration_s        = float(self._scenario.get("duration_s", 150))
        self._dump_period_s     = float(self._scenario.get("param_dump_period_s", 5))
        self._churn_period_s    = self._scenario.get("tcp_churn_period_s")
        self._bootstrap_timeout_s = float(self._scenario.get("bootstrap_timeout_s", 30))

        self._latest: dict = {}
        self._last_uptime_ms: int | None = None
        self._reboot_events: list[dict] = []
        self._churn_count = 0
        self._start_wall = time.time()
        self._timed_window_started = False
        self._wifi_never_connected = False

        self._dump_timer = QTimer(self)
        self._dump_timer.setInterval(max(50, int(self._dump_period_s * 1000)))
        self._dump_timer.timeout.connect(send_param_get_all)

        self._churn_timer = None
        if self._churn_period_s:
            self._churn_timer = QTimer(self)
            self._churn_timer.setInterval(max(50, int(float(self._churn_period_s) * 1000)))
            self._churn_timer.timeout.connect(self._do_churn)

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
              f"(timeout {self._bootstrap_timeout_s}s)...")

    def _start_timed_window(self):
        if self._timed_window_started:
            return
        self._timed_window_started = True
        self._start_wall = time.time()
        self._dump_timer.start()
        if self._churn_timer is not None:
            self._churn_timer.start()
        self._end_timer.setInterval(max(50, int(self._duration_s * 1000)))
        self._end_timer.start()
        print(f"[Automation] '{self._name}' timed window started: duration={self._duration_s}s "
              f"dump_period={self._dump_period_s}s churn_period={self._churn_period_s}")

    def _on_bootstrap_timeout(self):
        if self._timed_window_started:
            return
        print(f"[Automation] '{self._name}': no WiFi telemetry seen within "
              f"{self._bootstrap_timeout_s}s — starting timed window anyway.")
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

    def _do_churn(self):
        self._churn_count += 1
        WifiTransport.instance().force_reconnect()

    def _finish(self):
        report = {
            "name":                       self._name,
            "duration_s":                 self._duration_s,
            "reboot_count":                len(self._reboot_events),
            "reboot_events":               self._reboot_events,
            "tcp_churn_count":             self._churn_count,
            "final_wifi_esp_uptime_ms":    self._latest.get("wifi_esp_uptime_ms"),
            "final_wifi_free_heap":        self._latest.get("wifi_free_heap"),
            "final_wifi_min_free_heap":    self._latest.get("wifi_min_free_heap"),
            "wifi_reconnect_count":        self._latest.get("wifi_reconnect_count"),
            "wifi_udp_send_fail_count":    self._latest.get("wifi_udp_send_fail_count"),
            "wifi_uart_crc_drops":         self._latest.get("wifi_uart_crc_drops"),
            "wifi_uart_seq_gaps":          self._latest.get("wifi_uart_seq_gaps"),
            "wifi_uplink_queue_drops":     self._latest.get("wifi_uplink_queue_drops"),
            "wifi_teensy_link_up":         self._latest.get("wifi_teensy_link_up"),
            "wifi_never_connected":        self._wifi_never_connected,
            "pass": len(self._reboot_events) == 0 and not self._wifi_never_connected,
        }
        rel_path = self._scenario.get("report_path") or f"logs/{self._name}_report.json"
        out_path = os.path.join(_GUI_DIR, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Automation] Report written to {out_path}")
        print(json.dumps(report, indent=2))
        self._app.quit()
