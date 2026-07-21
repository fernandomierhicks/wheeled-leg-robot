"""wifi_diag_tab.py — WiFi Diagnostics tab.

Surfaces the WifiDiagPayload (WIFI_DIAG V2, comm_protocol.h) fields not
already shown on the Dashboard's LinkHealthWidget (main.py), plus a rolling
event log that fires on ESP32 reboots and WiFi reconnects — so diagnosing a
reboot loop is "read this tab" instead of grepping an 800+ MB raw serial log.
A reboot is detected purely from telemetry: wifi_esp_uptime_ms decreasing
between WIFI_DIAG packets means the ESP32 restarted since the last one.

Also has a "Load Report..." button to view the JSON report written by an
`AutomationRunner` scenario (tabs/automation_runner.py, main.py --automation).
"""

import json
import os
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from .telemetry_bus import TelemetryBus
from .theme import BG, BORDER, DIM, GREEN, MONO, RED, SURFACE, TEXT, YELLOW

_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

_ROWS = [
    ("wifi_esp_uptime_ms",       "ESP32 uptime",   lambda v: f"{v / 1000:.1f} s"),
    ("wifi_free_heap",           "Free heap",      lambda v: f"{v} B"),
    ("wifi_min_free_heap",       "Min free heap",  lambda v: f"{v} B"),
    ("wifi_reconnect_count",     "WiFi reconnects", str),
    ("wifi_udp_send_fail_count", "UDP send fails", str),
    ("wifi_loop_max_us",         "Loop max",       lambda v: f"{v} us"),
    ("wifi_channel",             "Channel",        str),
    ("wifi_status",              "WiFi status",    str),
    ("wifi_tx_power_raw",        "TX power (raw)", str),
]


class WifiDiagTab(QWidget):
    def __init__(self):
        super().__init__()

        self._latest: dict = {}
        self._last_uptime_ms: int | None = None
        self._last_reconnect_count: int | None = None

        # ── Live fields panel ───────────────────────────────────────────────
        fields_widget = QWidget()
        fields_widget.setStyleSheet(
            f"QWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 3px; }}"
            f"QLabel {{ border: none; }}"
        )
        grid = QGridLayout(fields_widget)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)

        title = QLabel("WiFi Diagnostics (WIFI_DIAG V2)")
        title.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")
        grid.addWidget(title, 0, 0, 1, 2)

        self._value_lbls: dict[str, QLabel] = {}
        for row, (key, name, _fmt) in enumerate(_ROWS, start=1):
            k = QLabel(name)
            k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            v = QLabel("—")
            v.setStyleSheet(f"color: {TEXT}; font-size: 11px; font-family: {MONO};")
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(k, row, 0)
            grid.addWidget(v, row, 1)
            self._value_lbls[key] = v

        grid.setRowStretch(len(_ROWS) + 1, 1)

        # ── Event log (reboots / reconnects) ────────────────────────────────
        log_widget = QWidget()
        log_lay = QVBoxLayout(log_widget)
        log_lay.setContentsMargins(0, 0, 0, 0)

        log_title = QLabel("Reboot / reconnect events")
        log_title.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")
        log_lay.addWidget(log_title)

        self._event_list = QListWidget()
        self._event_list.setStyleSheet(
            f"QListWidget {{ background: {BG}; border: 1px solid {BORDER};"
            f" font-family: {MONO}; font-size: 11px; }}"
        )
        log_lay.addWidget(self._event_list, stretch=1)

        clear_btn = QPushButton("Clear log")
        clear_btn.clicked.connect(self._event_list.clear)
        log_lay.addWidget(clear_btn)

        # ── Automation report viewer ─────────────────────────────────────────
        report_widget = QWidget()
        report_lay = QVBoxLayout(report_widget)
        report_lay.setContentsMargins(0, 0, 0, 0)

        report_header = QHBoxLayout()
        report_title = QLabel("Automation report")
        report_title.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")
        report_header.addWidget(report_title)
        report_header.addStretch()
        load_btn = QPushButton("Load Report...")
        load_btn.clicked.connect(self._on_load_report)
        report_header.addWidget(load_btn)
        report_lay.addLayout(report_header)

        self._report_view = QPlainTextEdit()
        self._report_view.setReadOnly(True)
        self._report_view.setStyleSheet(
            f"QPlainTextEdit {{ background: {BG}; border: 1px solid {BORDER};"
            f" color: {TEXT}; font-family: {MONO}; font-size: 11px; }}"
        )
        report_lay.addWidget(self._report_view, stretch=1)

        # ── Layout ───────────────────────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(log_widget)
        right_split.addWidget(report_widget)
        right_split.setSizes([300, 300])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(fields_widget)
        splitter.addWidget(right_split)
        splitter.setSizes([320, 600])
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(splitter)

        TelemetryBus.instance().packet.connect(self._on_packet)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start()

    # ── Telemetry handling ───────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        if info.get("ptype") == 0x01 and not self.isVisible():
            return
        self._latest.update({k: v for k, v in info.items() if v is not None})

        uptime = info.get("wifi_esp_uptime_ms")
        if uptime is not None:
            if self._last_uptime_ms is not None and uptime < self._last_uptime_ms:
                self._log_event(
                    f"ESP32 REBOOT — uptime {self._last_uptime_ms} ms -> {uptime} ms", bad=True
                )
            self._last_uptime_ms = uptime

        rc = info.get("wifi_reconnect_count")
        if rc is not None:
            if self._last_reconnect_count is not None and rc > self._last_reconnect_count:
                self._log_event(f"WiFi reconnect count {self._last_reconnect_count} -> {rc}", bad=True)
            self._last_reconnect_count = rc

    def _log_event(self, text: str, bad: bool):
        ts = time.strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{ts}] {text}")
        item.setForeground(Qt.GlobalColor.red if bad else Qt.GlobalColor.white)
        self._event_list.addItem(item)
        self._event_list.scrollToBottom()

    def _refresh(self):
        f = self._latest
        for key, _name, fmt in _ROWS:
            val = f.get(key)
            self._value_lbls[key].setText(fmt(val) if val is not None else "—")

    # ── Report viewer ────────────────────────────────────────────────────────

    def _on_load_report(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load automation report", _LOGS_DIR, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            self._report_view.setPlainText(f"Failed to load {path}: {e}")
            return
        self._report_view.setPlainText(json.dumps(data, indent=2))
