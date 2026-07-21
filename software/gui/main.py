import sys
import os
import time
import psutil
from collections import deque

def _kill_other_instances():
    current_pid = os.getpid()
    current_script = os.path.abspath(__file__)
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['pid'] == current_pid:
            continue
        cmdline = proc.info.get('cmdline') or []
        if any(os.path.abspath(arg) == current_script for arg in cmdline if arg):
            proc.kill()

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QSplitter, QWidget,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox, QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QAction, QIcon

from tabs.theme import APP_STYLE, BORDER, TEXT, DIM, GREEN, ORANGE, RED, BLUE, YELLOW, WHITE
from tabs.robot_log_widget import RobotLogWidget
from tabs.flash_monitor import FlashMonitorTab
from tabs.hip_motors import HipMotorsTab
from tabs.imu_tab import ImuTab, ImuMiniWidget
from tabs.params_tab import ParamsTab
from tabs.raw_data_tab import RawDataTab
from tabs.robot_visualizer_tab import RobotVisualizerTab
from tabs.radio_tab import RadioTab
from tabs.wheel_motors import WheelMotorsTab
from tabs.controllers_tab import ControllersTab
from tabs.log_playback import LogsTab, LogPlaybackController
from tabs.log_transfer import LogTransferManager
from tabs.log_analyzer_tab import LogAnalyzerTab
from tabs.wifi_diag_tab import WifiDiagTab
from tabs.telemetry_bus import TelemetryBus
from tabs.source_manager import SourceManager, TRANSPORT_LABEL
from tabs.comm_commands import (
    send_set_mode, send_reboot, send_reliable, CommandAlertBus,
    STATE_STARTUP, STATE_STANDBY, STATE_ESTOP, STATE_CMD_REJECT,
)

_BG = "#0b0b18"

_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")

# ── Test-val mini chart ───────────────────────────────────────────────────────

class TestValMiniWidget(QWidget):
    """Rolling chart of the 2 Hz sine test_val field — dashboard comms health check."""

    _BUF = 200  # samples; at 50 Hz telemetry = 4 s of history

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(230)

        self._buf = deque([0.0] * self._BUF, maxlen=self._BUF)

        plot = pg.PlotWidget()
        plot.setBackground(_BG)
        plot.setFixedHeight(170)
        plot.setYRange(-1.2, 1.2)
        plot.hideAxis("bottom")
        plot.getAxis("left").setStyle(tickFont=pg.Qt.QtGui.QFont("Consolas", 8))
        plot.getAxis("left").setTextPen(pg.mkPen(DIM))
        plot.showGrid(x=False, y=True, alpha=0.12)
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)

        self._curve = plot.plot(list(self._buf), pen=pg.mkPen(GREEN, width=1.5))

        self._lbl = QLabel("~  —")
        self._lbl.setStyleSheet(
            f"color: {DIM}; font-size: 11px; font-family: Consolas;"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(4)
        lay.addWidget(plot)
        lay.addWidget(self._lbl)

        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        val = info.get("test_val")
        if val is None:
            return
        self._buf.append(val)
        self._curve.setData(list(self._buf))
        self._lbl.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-family: Consolas;"
        )
        self._lbl.setText(f"~  {val:+.4f}")

# ── ToF distance mini widget ─────────────────────────────────────────────────

class TofMiniWidget(QWidget):
    """Four VL53L1X distance bars: sensors 0-1 forward (green), 2-3 backward (orange)."""

    _MAX_MM    = 2000   # display range [mm]
    _WARN_MM   = 400    # highlight threshold [mm]
    _NO_DATA   = 0xFFFF

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(200)

        self._bars:   list[QFrame]  = []
        self._labels: list[QLabel]  = []
        self._stale_lbl = QLabel("TOF — no data")
        self._stale_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; font-family: Consolas;")
        self._stale_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(6)
        bar_row.setContentsMargins(0, 0, 0, 0)

        _COLORS = [GREEN, GREEN, ORANGE, ORANGE]
        _NAMES  = ["F-L", "F-R", "R-L", "R-R"]
        for i in range(4):
            col = _COLORS[i]
            name_lbl = QLabel(_NAMES[i])
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet(f"color: {DIM}; font-size: 9px;")

            bar_bg = QFrame()
            bar_bg.setFixedWidth(30)
            bar_bg.setFixedHeight(120)
            bar_bg.setStyleSheet(f"background: #111; border: 1px solid {BORDER};")

            bar_fill = QFrame(bar_bg)
            bar_fill.setFixedWidth(28)
            bar_fill.move(1, 1)
            bar_fill.setFixedHeight(0)
            bar_fill.setStyleSheet(f"background: {col};")

            dist_lbl = QLabel("—")
            dist_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dist_lbl.setStyleSheet(f"color: {DIM}; font-size: 9px; font-family: Consolas;")

            col_lay = QVBoxLayout()
            col_lay.setSpacing(2)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.addWidget(name_lbl)
            col_lay.addWidget(bar_bg)
            col_lay.addWidget(dist_lbl)
            bar_row.addLayout(col_lay)

            self._bars.append(bar_fill)
            self._labels.append(dist_lbl)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addLayout(bar_row)
        lay.addWidget(self._stale_lbl)

        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        dists = info.get("tof_dist_mm")
        if dists is None:
            return

        stale = info.get("tof_stale", False)
        age   = info.get("tof_age_ms", self._NO_DATA)
        self._stale_lbl.setText(
            "TOF stale" if stale else f"TOF  age {age} ms"
        )
        self._stale_lbl.setStyleSheet(
            f"color: {RED if stale else DIM}; font-size: 10px; font-family: Consolas;"
        )

        _WARN_COLORS = [RED, RED, RED, RED]
        _OK_COLORS   = [GREEN, GREEN, ORANGE, ORANGE]
        for i, d in enumerate(dists):
            bar    = self._bars[i]
            lbl    = self._labels[i]
            parent = bar.parent()
            if d == self._NO_DATA:
                bar.setFixedHeight(0)
                lbl.setText("—")
                lbl.setStyleSheet(f"color: {DIM}; font-size: 9px; font-family: Consolas;")
            else:
                frac   = max(0.0, 1.0 - d / TofMiniWidget._MAX_MM)
                height = int(frac * (parent.height() - 2))
                bar.setFixedHeight(max(1, height))
                bar.move(1, parent.height() - 1 - bar.height())
                warn   = d < TofMiniWidget._WARN_MM
                color  = _WARN_COLORS[i] if warn else _OK_COLORS[i]
                bar.setStyleSheet(f"background: {color};")
                lbl.setText(f"{d} mm")
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 9px; font-weight: {'bold' if warn else 'normal'};"
                    f" font-family: Consolas;"
                )


# ── Link health mini widget (UARTplat.md Phase 4, §5.2) ───────────────────────
# Thresholds kept in one place so they're easy to retune without hunting
# through the widget body.
_LINK_WINDOW_S           = 10.0  # "any increase" / count-based checks look back this far
_LINK_GAP_RATE_WINDOW_S  = 5.0   # GUI-side seq-gap RATE check window
_LINK_ESP32_HB_MAX_MS    = 1000
_LINK_UDP_FAIL_MAX       = 5     # max allowed growth over _LINK_WINDOW_S
_LINK_SEND_MAX_US        = 20000
_LINK_RSSI_MIN_DBM       = -75
_LINK_GAP_RATE_MAX       = 5.0   # gaps/s, over _LINK_GAP_RATE_WINDOW_S


class _WindowedCounter:
    """Tracks a monotonically-increasing counter over a trailing window and
    reports how much it grew (delta) and its average per-second rate — the
    two shapes every threshold in the table below needs."""

    def __init__(self, window_s: float):
        self._window_s = window_s
        self._samples: deque[tuple[float, int]] = deque()

    def update(self, value: int) -> tuple[int, float]:
        now = time.monotonic()
        self._samples.append((now, value))
        while len(self._samples) > 1 and now - self._samples[0][0] > self._window_s:
            self._samples.popleft()
        t0, v0 = self._samples[0]
        elapsed = now - t0
        delta = value - v0
        rate = delta / elapsed if elapsed > 0 else 0.0
        return delta, rate


class LinkHealthWidget(QWidget):
    """Dashboard "Link" panel: UART/WiFi link health at a glance, with
    degradation alerts surfaced via the `alert` signal (one line, "" to clear).
    Fed by both TELEM (v9) and WIFI_DIAG (v2) packets on TelemetryBus;
    display refreshes at 1 Hz. See UARTplat.md Phase 4, §5.2."""

    alert = pyqtSignal(str)

    _ROWS = [
        ("uart_te",     "UART T→E"),
        ("uart_et",     "UART E→T"),
        ("esp32_hb",    "ESP32 hb age"),
        ("teensy_link", "Teensy link"),
        ("uplink_q",    "Uplink drops"),
        ("udp_fail",    "UDP fails"),
        ("send_max",    "Send max"),
        ("gui_link",    "GUI link"),
        ("rssi",        "RSSI"),
    ]

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(260)

        self._t_uart_te_crc = _WindowedCounter(_LINK_WINDOW_S)
        self._t_uart_te_gap = _WindowedCounter(_LINK_WINDOW_S)
        self._t_uart_et_rx  = _WindowedCounter(_LINK_WINDOW_S)
        self._t_uart_et_seq = _WindowedCounter(_LINK_WINDOW_S)
        self._t_uplink_q    = _WindowedCounter(_LINK_WINDOW_S)
        self._t_udp_fail    = _WindowedCounter(_LINK_WINDOW_S)
        self._t_gui_crc     = _WindowedCounter(_LINK_WINDOW_S)
        self._t_gui_gap     = _WindowedCounter(_LINK_GAP_RATE_WINDOW_S)

        self._latest: dict = {}
        self._last_alert_message = ""

        lbl_title = QLabel("Link Health")
        lbl_title.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self._value_lbls: dict[str, QLabel] = {}
        for row, (key, name) in enumerate(self._ROWS):
            k = QLabel(name)
            k.setStyleSheet(f"color: {DIM}; font-size: 10px;")
            v = QLabel("—")
            v.setStyleSheet(f"color: {TEXT}; font-size: 10px; font-family: Consolas;")
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(k, row, 0)
            grid.addWidget(v, row, 1)
            self._value_lbls[key] = v

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(4)
        lay.addWidget(lbl_title)
        lay.addLayout(grid)
        lay.addStretch(1)

        TelemetryBus.instance().packet.connect(self._on_packet)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start()

    def _on_packet(self, info: dict):
        # Merge every packet's fields into the latest-known state — TELEM and
        # WIFI_DIAG carry disjoint keys, so later packets never clobber the
        # other type's fields with missing/None.
        self._latest.update({k: v for k, v in info.items() if v is not None})

    def _set(self, key: str, text: str, bad: bool):
        lbl = self._value_lbls[key]
        lbl.setText(text)
        weight = "font-weight: bold;" if bad else ""
        lbl.setStyleSheet(
            f"color: {RED if bad else TEXT}; font-size: 10px; font-family: Consolas; {weight}"
        )

    def _refresh(self):
        f = self._latest
        problems: list[str] = []

        crc = f.get("wifi_uart_crc_drops")
        gap = f.get("wifi_uart_seq_gaps")
        if crc is not None and gap is not None:
            d_crc, _ = self._t_uart_te_crc.update(crc)
            d_gap, _ = self._t_uart_te_gap.update(gap)
            bad = d_crc > 0 or d_gap > 0
            self._set("uart_te", f"crc {crc}  gap {gap}", bad)
            if bad:
                problems.append("UART T→E drops")
        else:
            self._set("uart_te", "—", False)

        rx  = f.get("uart_rx_drops")
        seq = f.get("uart_seq_gaps")
        if rx is not None and seq is not None:
            d_rx, _  = self._t_uart_et_rx.update(rx)
            d_seq, _ = self._t_uart_et_seq.update(seq)
            bad = d_rx > 0 or d_seq > 0
            self._set("uart_et", f"crc {rx}  gap {seq}", bad)
            if bad:
                problems.append("UART E→T drops")
        else:
            self._set("uart_et", "—", False)

        age = f.get("esp32_status_age_ms")
        if age is not None:
            bad = age > _LINK_ESP32_HB_MAX_MS
            self._set("esp32_hb", f"{age} ms", bad)
            if bad:
                problems.append("ESP32 heartbeat stale")
        else:
            self._set("esp32_hb", "—", False)

        up    = f.get("wifi_teensy_link_up")
        since = f.get("wifi_ms_since_teensy")
        if up is not None:
            bad = not up
            txt = "UP" if up else "DOWN"
            if since is not None:
                txt += f" ({since} ms)"
            self._set("teensy_link", txt, bad)
            if bad:
                problems.append("Teensy link down (ESP32 view)")
        else:
            self._set("teensy_link", "—", False)

        uq = f.get("wifi_uplink_queue_drops")
        if uq is not None:
            d_uq, _ = self._t_uplink_q.update(uq)
            bad = d_uq > 0
            self._set("uplink_q", str(uq), bad)
            if bad:
                problems.append("Uplink queue drops")
        else:
            self._set("uplink_q", "—", False)

        uf = f.get("wifi_udp_send_fail_count")
        if uf is not None:
            d_uf, _ = self._t_udp_fail.update(uf)
            bad = d_uf > _LINK_UDP_FAIL_MAX
            self._set("udp_fail", str(uf), bad)
            if bad:
                problems.append("UDP send failures rising")
        else:
            self._set("udp_fail", "—", False)

        tcp_us = f.get("wifi_tcp_send_max_us")
        udp_us = f.get("wifi_udp_send_max_us")
        if tcp_us is not None or udp_us is not None:
            worst = max(tcp_us or 0, udp_us or 0)
            bad = worst > _LINK_SEND_MAX_US
            self._set("send_max", f"{worst} us", bad)
            if bad:
                problems.append("TCP/UDP send blocking")
        else:
            self._set("send_max", "—", False)

        gcrc  = f.get("link_crc_drops")
        ggap  = f.get("link_seq_gaps")
        gpair = f.get("link_pair_drops")
        if gcrc is not None:
            d_crc, _    = self._t_gui_crc.update(gcrc)
            _, gap_rate = self._t_gui_gap.update(ggap or 0)
            bad = d_crc > 0 or gap_rate > _LINK_GAP_RATE_MAX
            self._set("gui_link", f"crc {gcrc}  gap {ggap}  pair {gpair}", bad)
            if bad:
                problems.append("GUI-side decode errors")
        else:
            self._set("gui_link", "—", False)

        rssi = f.get("wifi_rssi_dbm")
        if rssi is not None:
            bad = rssi < _LINK_RSSI_MIN_DBM
            self._set("rssi", f"{rssi} dBm", bad)
            if bad:
                problems.append("Weak WiFi signal")
        else:
            self._set("rssi", "—", False)

        message = f"LINK DEGRADED: {', '.join(problems)}" if problems else ""
        if message != self._last_alert_message:
            self._last_alert_message = message
            self.alert.emit(message)


# ── Placeholder tabs ──────────────────────────────────────────────────────────

class _PlaceholderTab(QWidget):
    def __init__(self, name: str):
        super().__init__()
        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 22))
        lbl.setStyleSheet(f"color: {BORDER};")
        QVBoxLayout(self).addWidget(lbl)

class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        imu      = ImuMiniWidget()
        test_val = TestValMiniWidget()
        tof      = TofMiniWidget()
        self.link_health = LinkHealthWidget()

        top = QHBoxLayout()
        top.setContentsMargins(8, 8, 0, 0)
        top.setSpacing(12)
        top.addWidget(imu)
        top.addWidget(test_val)
        top.addWidget(tof)
        top.addWidget(self.link_health)
        top.addStretch(7)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(top)
        outer.addStretch(1)

def _send_set_mode_reliable(target: int, label: str, on_success=None, on_fail=None):
    """SET_MODE via ReliableCommand: confirmed once telemetry's robot_state
    reaches `target`, or reported (not retried) if the state machine bounces
    it to STATE_CMD_REJECT instead. See UARTplat.md Phase 5.
    on_success/on_fail let a caller (e.g. tabs/remote_control.py) bridge this
    fire-and-forget confirmation into a synchronous wait; existing call sites
    omit both and keep today's behavior (CommandAlertBus on failure, silent
    on success)."""
    send_reliable(
        lambda: send_set_mode(target),
        confirm_predicate=lambda info: info.get("robot_state") == target,
        reject_predicate=lambda info: info.get("robot_state") == STATE_CMD_REJECT,
        label=label,
        on_success=on_success,
        on_fail=on_fail,
    )


# ── Fault severity (mirrors comm_protocol.h fault_severity()) ─────────────────
_FAULT_SEVERITY = {
    0x06: "SOFT",        # HUMAN_ESTOP
    0x09: "SOFT",        # WHEEL_RUNAWAY
    0x08: "REPOSITION",  # PITCH_WATCHDOG
    0x05: "RECALIBRATE", # CALIBRATION_TIMEOUT (reposition + re-run calib)
    0x04: "GUI_FIX",     # HIP_LARGE_POS_CMD
    0x01: "REBOOT",      # IMU_ERROR
    0x02: "REBOOT",      # HIP_INIT_TIMEOUT
    0x03: "REBOOT",      # HIP_FEEDBACK_LOST
    0x0A: "REBOOT",      # IMU_LOST
    0x0B: "REBOOT",      # WHEEL_FEEDBACK_LOST
}

# ── Battery status widget ─────────────────────────────────────────────────────

class BatteryStatusWidget(QWidget):
    """Battery icon + voltage label for the status bar. Blinks at 1 Hz when critically low."""
    _VMAX = 25.2   # 4.2 V/cell × 6S
    _VMIN = 21.0   # 3.5 V/cell × 6S (display empty)
    _NBARS = 5
    # A wheel side that's disabled (wheel_l/r_enable=0) never gets its vbus
    # requested by firmware, so it reads a permanent 0.0 — exclude it from the
    # average instead of dragging a real ~24V reading down to ~12V.
    _VBUS_MIN_VALID = 5.0

    def __init__(self):
        super().__init__()
        self.setFixedWidth(72)
        self._vbus: float | None = None
        self._blink_on = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._on_blink)
        TelemetryBus.instance().packet.connect(self._on_packet)

    def _num_bars(self, vbus: float) -> int:
        pct = max(0.0, min(1.0, (vbus - self._VMIN) / (self._VMAX - self._VMIN)))
        return round(pct * self._NBARS)

    def _on_packet(self, info: dict):
        vl = info.get("wm_l_vbus")
        vr = info.get("wm_r_vbus")
        if vl is None or vr is None:
            return
        readings = [v for v in (vl, vr) if v > self._VBUS_MIN_VALID]
        if not readings:
            return
        vbus = sum(readings) / len(readings)
        low = self._num_bars(vbus) <= 1
        if low and not self._blink_timer.isActive():
            self._blink_timer.start()
        elif not low and self._blink_timer.isActive():
            self._blink_timer.stop()
            self._blink_on = True
        self._vbus = vbus
        self.update()

    def set_connected(self, connected: bool):
        if not connected:
            self._blink_timer.stop()
            self._blink_on = True
            self._vbus = None
            self.update()

    def _on_blink(self):
        self._blink_on = not self._blink_on
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        w, h = self.width(), self.height()

        # Icon geometry
        BW, BH = 22, 12
        NW, NH = 3, 6
        bx = 2
        by = (h - BH) // 2

        if self._vbus is None:
            p.setPen(QPen(QColor(DIM)))
            p.drawRect(bx, by, BW - 1, BH - 1)
            p.fillRect(bx + BW, by + (BH - NH) // 2, NW, NH, QColor(DIM))
            p.end()
            return

        bars = self._num_bars(self._vbus)
        low  = bars <= 1
        pct  = bars / self._NBARS
        col  = QColor(GREEN if pct > 0.5 else ORANGE if pct > 0.2 else RED)

        if low and not self._blink_on:
            dim = QColor(100, 0, 0)
            p.setPen(QPen(dim))
            p.drawRect(bx, by, BW - 1, BH - 1)
            p.fillRect(bx + BW, by + (BH - NH) // 2, NW, NH, dim)
        else:
            p.setPen(QPen(QColor(WHITE)))
            p.drawRect(bx, by, BW - 1, BH - 1)
            p.fillRect(bx + BW, by + (BH - NH) // 2, NW, NH, QColor(WHITE))
            p.fillRect(bx + 2, by + 2, BW - 4, BH - 4, QColor(15, 15, 20))
            # 5 bars × 2px wide, 1px gap, 2px left padding
            for i in range(self._NBARS):
                if i < bars:
                    p.fillRect(bx + 4 + i * 3, by + 3, 2, BH - 6, col)

        # Voltage text
        p.setPen(QPen(col))
        p.setFont(QFont("Consolas", 9))
        tx = bx + BW + NW + 4
        p.drawText(tx, 0, w - tx, h,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"{self._vbus:.1f}V")
        p.end()


# ── Radio signal widget ───────────────────────────────────────────────────────

class RadioSignalWidget(QWidget):
    """RC TX signal indicator — 4 ascending vertical bars (cellular-style, not WiFi arcs).
    Green = signal, red = no signal, dim = no telemetry data yet."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(52, 24)
        self._alive: bool | None = None
        TelemetryBus.instance().packet.connect(self._on_packet)

    def _on_packet(self, info: dict):
        alive = info.get("ibus_alive")
        if alive is None:
            return
        new = bool(alive)
        if new != self._alive:
            self._alive = new
            self.update()

    def set_connected(self, connected: bool):
        if not connected and self._alive is not None:
            self._alive = None
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # 4 bars, ascending heights, left-aligned
        n_bars = 4
        bar_w  = 4
        gap    = 3
        total  = n_bars * bar_w + (n_bars - 1) * gap
        x0     = 2
        base_y = h - 4

        if self._alive is None:
            bar_color = QColor(DIM)
            n_lit = 0
        elif self._alive:
            bar_color = QColor(GREEN)
            n_lit = n_bars
        else:
            bar_color = QColor(RED)
            n_lit = 1  # one short bar to suggest "no signal"

        dim_color = QColor(DIM)
        dim_color.setAlpha(60)

        for i in range(n_bars):
            bh = 4 + i * 4   # 4, 8, 12, 16 px
            x  = x0 + i * (bar_w + gap)
            y  = base_y - bh
            col = bar_color if i < n_lit else dim_color
            p.fillRect(x, y, bar_w, bh, col)

        # "RC" label
        lbl_color = QColor(GREEN if self._alive else (RED if self._alive is not None else DIM))
        p.setPen(QPen(lbl_color))
        p.setFont(QFont("Consolas", 8))
        p.drawText(x0 + total + 3, 0, w - (x0 + total + 3), h,
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   "RC")
        p.end()


# ── Status bar ────────────────────────────────────────────────────────────────

def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


class LogMiniControls(QWidget):
    """Compact SD-logging + playback controls for the status bar — start/stop
    a log or play/pause an open .wlog from any tab, without switching to
    the Logs tab. Both drive the same LogTransferManager/LogPlaybackController
    singletons the Logs tab uses, so state always stays in sync."""

    def __init__(self):
        super().__init__()
        self._xfer = LogTransferManager.instance()
        self._pb   = LogPlaybackController.instance()

        self._rec_lbl = QLabel("● REC")
        self._rec_lbl.setStyleSheet(f"color: {DIM}; font-weight: bold; font-size: 12px;")
        self._rec_lbl.setToolTip("Lit while an SD log is actively recording")

        self._btn_log = QPushButton()
        self._btn_log.setFixedWidth(64)
        self._btn_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_log.clicked.connect(self._on_log_clicked)
        self._style_log_button(False)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(26)
        self._btn_play.setEnabled(False)
        self._btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_play.setToolTip("Open a .wlog in the Logs tab to enable playback")
        self._btn_play.clicked.connect(self._pb.toggle)
        self._btn_play.setStyleSheet(
            f"QPushButton{{background:transparent;color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:2px}}"
            f"QPushButton:disabled{{color:{DIM};border:1px solid {BORDER}}}"
            f"QPushButton:hover{{background:{BORDER}}}"
        )

        self._pb_lbl = QLabel("")
        self._pb_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px; font-family: Consolas;")
        self._pb_lbl.setMaximumWidth(140)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(6)
        lay.addWidget(self._rec_lbl)
        lay.addWidget(self._btn_log)
        lay.addWidget(self._btn_play)
        lay.addWidget(self._pb_lbl)

        self._xfer.logging_state_changed.connect(self._on_logging_state)
        self._pb.file_opened.connect(self._on_file_opened)
        self._pb.playing_changed.connect(self._on_playing_changed)
        self._pb.position_changed.connect(self._on_position_changed)

    def _style_log_button(self, active: bool):
        color = RED if active else GREEN
        self._btn_log.setText("■ Stop" if active else "● Log")
        self._btn_log.setToolTip(
            "Stop the active SD log" if active else
            "Start logging until Stop is clicked (use the Logs tab for a timed duration)"
        )
        self._btn_log.setStyleSheet(
            f"QPushButton{{background:transparent;color:{color};border:1px solid {color};"
            f"border-radius:3px;padding:2px 6px;font-size:11px}}"
            f"QPushButton:hover{{background:{BORDER}}}"
        )

    def _on_log_clicked(self):
        if self._xfer.is_logging():
            self._xfer.stop_logging()
        else:
            self._xfer.start_logging(0)

    def _on_logging_state(self, active: bool):
        self._style_log_button(active)
        self._rec_lbl.setStyleSheet(
            f"color: {RED if active else DIM}; font-weight: bold; font-size: 12px;"
        )

    def _on_file_opened(self, path: str, count: int, sample_rate_hz: int):
        from pathlib import Path
        self._btn_play.setEnabled(True)
        self._btn_play.setToolTip("Play/pause the open .wlog")
        self._pb_lbl.setText(Path(path).name)

    def _on_playing_changed(self, playing: bool):
        self._btn_play.setText("⏸" if playing else "▶")

    def _on_position_changed(self, idx: int, total: int):
        self._pb_lbl.setToolTip(f"{idx} / {total}")


class StatusBar:
    def __init__(self, sb):
        self._source = QPushButton("● —")
        self._source.setFlat(True)
        self._source.setCursor(Qt.CursorShape.PointingHandCursor)
        self._source.setToolTip("Click to select telemetry source")
        self._source.setStyleSheet(
            f"QPushButton{{color:{DIM};background:transparent;border:none;"
            f"font-size:13px;padding:0 4px}}"
            f"QPushButton:hover{{color:{TEXT};text-decoration:underline}}"
        )
        self._source.clicked.connect(self._on_source_clicked)
        self._transport = QLabel("—")
        self._dt        = QLabel("dt: —")
        self._mode      = QLabel("—")
        self._mode.setStyleSheet(f"color: {DIM};")
        self._profile   = QLabel("")
        self._profile.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-weight: bold;")
        self._batt      = BatteryStatusWidget()
        self._radio     = RadioSignalWidget()
        self._conn      = QLabel("● Disconnected")
        self._conn.setStyleSheet(f"color: {RED};")

        self._current_state      = ""
        self._current_fault_code = 0

        self._btn_estop = QPushButton("ESTOP")
        self._btn_estop.setStyleSheet(
            f"QPushButton{{background:{RED};color:white;font-weight:bold;font-size:14px;"
            f"border:1px solid {RED};border-radius:4px;padding:6px 24px;margin:2px 6px}}"
            f"QPushButton:hover{{background:#ff6e60}}"
            f"QPushButton:pressed{{background:#b2362a}}"
        )
        self._btn_estop.clicked.connect(lambda: _send_set_mode_reliable(STATE_ESTOP, "ESTOP"))

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._reset_style_normal = (
            f"QPushButton{{background:#4a1a1a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#4a1a1acc}}"
            f"QPushButton:pressed{{background:#4a1a1a88}}"
            f"QPushButton:disabled{{background:transparent;color:{DIM};"
            f"border:1px solid {BORDER}}}"
        )
        self._reset_style_soft = (
            f"QPushButton{{background:#1a4a1a;color:white;font-weight:bold;"
            f"border:1px solid {GREEN};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#1a6a1a}}"
            f"QPushButton:pressed{{background:#0f3a0f}}"
        )
        self._reset_style_orange = (
            f"QPushButton{{background:#4a2a00;color:white;"
            f"border:1px solid {ORANGE};border-radius:3px;padding:2px 10px}}"
            f"QPushButton:hover{{background:#6a3a00}}"
            f"QPushButton:pressed{{background:#3a1a00}}"
        )
        self._btn_reset.setStyleSheet(self._reset_style_normal)

        self._btn_reboot = QPushButton("Reboot")
        self._btn_reboot.setStyleSheet(self._btn_reset.styleSheet())
        self._btn_reboot.setEnabled(False)
        self._btn_reboot.clicked.connect(self._on_reboot_clicked)

        for w in (self._source, _vsep(), self._transport, _vsep(), self._dt, _vsep(), self._mode, self._profile, _vsep(), self._batt, _vsep(), self._radio):
            sb.addWidget(w)
        self._mismatch_lbl = QLabel("FIRMWARE MISMATCH — reflash ESP32/Teensy")
        self._mismatch_lbl.setStyleSheet(
            f"color: white; background: {RED}; font-weight: bold;"
            f" padding: 2px 10px; border-radius: 3px;"
        )
        self._mismatch_lbl.setVisible(False)

        # Link-degradation banner (UARTplat.md Phase 4, §5.2) — same visual
        # pattern as the firmware-mismatch banner above, driven by
        # LinkHealthWidget.alert instead of a version check.
        self._link_alert_lbl = QLabel("")
        self._link_alert_lbl.setStyleSheet(
            f"color: white; background: {ORANGE}; font-weight: bold;"
            f" padding: 2px 10px; border-radius: 3px;"
        )
        self._link_alert_lbl.setVisible(False)

        self._log_controls = LogMiniControls()

        sb.addPermanentWidget(self._mismatch_lbl)
        sb.addPermanentWidget(self._link_alert_lbl)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._log_controls)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._btn_estop)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._btn_reset)
        sb.addPermanentWidget(self._btn_reboot)
        sb.addPermanentWidget(_vsep())
        sb.addPermanentWidget(self._conn)

    def _on_reset_clicked(self):
        if self._current_state != "ESTOP":
            return
        severity = _FAULT_SEVERITY.get(self._current_fault_code, "REBOOT")
        if severity == "SOFT":
            _send_set_mode_reliable(STATE_STANDBY, "Clear ESTOP")
        elif severity in ("REPOSITION", "RECALIBRATE"):
            extra = ("\n\nYou will need to re-trigger calibration (CH5 or GUI) after reset."
                     if severity == "RECALIBRATE" else "")
            reply = QMessageBox.question(
                None, "Confirm Reset",
                f"Robot fell. Stand it upright, then click Yes to reset.{extra}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _send_set_mode_reliable(STATE_STARTUP, "Reset")
        elif severity == "GUI_FIX":
            reply = QMessageBox.question(
                None, "Fix Parameter First",
                "A parameter value caused this ESTOP.\n"
                "Please fix the offending parameter in the Params tab, then click Yes to reset.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _send_set_mode_reliable(STATE_STARTUP, "Reset")
        # REBOOT: button is disabled; should not be reached

    def _on_reboot_clicked(self):
        reply = QMessageBox.question(
            None, "Reboot Teensy",
            "This will fully reset the Teensy MCU and re-run setup() from scratch.\n"
            "All motors and state will drop momentarily. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Confirmed by loop_count resetting after a fresh boot (UARTplat.md
            # Phase 5) — a real reset always restarts it near 0.
            send_reliable(
                send_reboot,
                confirm_predicate=lambda info: info.get("loop_count", 10**9) < 500,
                label="Reboot",
            )

    def _on_source_clicked(self):
        from tabs.source_manager import SourceManager, PRIORITY
        sm = SourceManager.instance()
        menu = QMenu()
        menu.setStyleSheet(
            f"QMenu{{background:#1a1a2e;color:{TEXT};border:1px solid {BORDER};"
            f"font-size:13px}}"
            f"QMenu::item:selected{{background:{BORDER}}}"
            f"QMenu::item:disabled{{color:{DIM}}}"
        )

        auto_action = menu.addAction("Auto (priority order)")
        auto_action.setCheckable(True)
        auto_action.setChecked(sm.override is None)
        menu.addSeparator()

        for device in PRIORITY:
            label = device.upper()
            connected = device in sm.connected
            action = menu.addAction(f"● {label}" if connected else f"○ {label} (not connected)")
            action.setCheckable(True)
            action.setChecked(sm.override == device)
            action.setEnabled(connected)
            action.setData(device)

        chosen = menu.exec(self._source.mapToGlobal(self._source.rect().bottomLeft()))
        if chosen is None:
            return
        if chosen is auto_action:
            sm.set_override(None)
        elif chosen.data():
            sm.set_override(chosen.data())

    def set_source(self, src: str):
        from tabs.source_manager import SourceManager
        sm = SourceManager.instance()
        color = {
            "TEENSY": BLUE,
            "ESP32":  ORANGE,
        }.get(src, DIM)
        is_override = sm.override is not None
        label = f"● {src}" if src != "—" else "● —"
        if is_override:
            label += " ✎"
        self._source.setStyleSheet(
            f"QPushButton{{color:{color};background:transparent;border:none;"
            f"font-size:13px;padding:0 4px}}"
            f"QPushButton:hover{{color:{color};text-decoration:underline}}"
        )
        self._source.setText(label)
        tip = "Click to select telemetry source"
        if is_override:
            tip += f" (override active — Auto uses {', '.join(p.upper() for p in ['esp32','teensy'])} priority)"
        self._source.setToolTip(tip)

    def set_transport(self, transport: str):
        self._transport.setText(transport)

    def set_dt(self, dt_ms: float):
        self._dt.setText(f"dt: {dt_ms:.1f} ms")

    def set_mode(self, state: str, fault: str = "", fault_description: str = "", fault_code: int = 0):
        self._current_state      = state
        self._current_fault_code = fault_code
        if state == "—":
            self._mode.setStyleSheet(f"color: {DIM};")
            self._mode.setText("—")
            self._mode.setToolTip("")
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_normal)
            self._btn_reset.setEnabled(False)
            self._btn_reboot.setEnabled(False)
            return
        color = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE, "STANDBY": YELLOW, "STARTUP": WHITE, "STANDING_UP": "#ff3c00"}.get(state, DIM)
        self._mode.setStyleSheet(f"color: {color}; font-weight: bold;")
        is_fault = state == "ESTOP" and fault and fault != "NONE"
        label = f"{state}  [{fault}]" if is_fault else state
        self._mode.setText(label)
        self._mode.setToolTip(fault_description if is_fault else "")
        self._btn_reboot.setEnabled(True)

        if state != "ESTOP":
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_normal)
            self._btn_reset.setEnabled(False)
            self._btn_reset.setToolTip("")
            return

        severity = _FAULT_SEVERITY.get(fault_code, "REBOOT")
        if severity == "SOFT":
            self._btn_reset.setText("Clear ESTOP")
            self._btn_reset.setStyleSheet(self._reset_style_soft)
            self._btn_reset.setEnabled(True)
            self._btn_reset.setToolTip("Click to clear ESTOP and return to STANDBY (no re-init)")
        elif severity in ("REPOSITION", "RECALIBRATE"):
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_orange)
            self._btn_reset.setEnabled(True)
            tip = ("Stand robot upright first, then click Reset" +
                   (" — recalibration required after reset" if severity == "RECALIBRATE" else ""))
            self._btn_reset.setToolTip(tip)
        elif severity == "GUI_FIX":
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_orange)
            self._btn_reset.setEnabled(True)
            self._btn_reset.setToolTip("Fix the offending parameter in the Params tab before resetting")
        else:  # REBOOT
            self._btn_reset.setText("Reset")
            self._btn_reset.setStyleSheet(self._reset_style_normal)
            self._btn_reset.setEnabled(False)
            self._btn_reset.setToolTip("Hardware fault — power cycle robot and reboot required")

    def set_profile(self, profile: int):
        label = f"  P{profile + 1}" if isinstance(profile, int) and 0 <= profile <= 2 else ""
        color = {0: DIM, 1: GREEN, 2: ORANGE}.get(profile, DIM)
        self._profile.setStyleSheet(f"color: {color}; font-family: Consolas; font-weight: bold;")
        self._profile.setText(label)

    def set_link_state(self, state: str, tooltip: str = ""):
        """state: 'connected' | 'esp32_only' | 'disconnected' (UARTplat.md Phase 4, §5.1)."""
        if state == "connected":
            self._conn.setStyleSheet(f"color: {GREEN};")
            self._conn.setText("● Connected")
            self._conn.setToolTip("")
        elif state == "esp32_only":
            self._conn.setStyleSheet(f"color: {ORANGE}; font-weight: bold;")
            self._conn.setText("● ESP32 ONLY — NO TEENSY")
            self._conn.setToolTip(tooltip)
            self._batt.set_connected(False)
            self._radio.set_connected(False)
        else:
            self._conn.setStyleSheet(f"color: {RED};")
            self._conn.setText("● Disconnected")
            self._conn.setToolTip("")
            self._mismatch_lbl.setVisible(False)
            self._batt.set_connected(False)
            self._radio.set_connected(False)

    def set_link_alert(self, message: str):
        if message:
            self._link_alert_lbl.setText(message)
            self._link_alert_lbl.setVisible(True)
        else:
            self._link_alert_lbl.setVisible(False)

    def set_version_mismatch(self, got: int, expected: int):
        self._mismatch_lbl.setText(
            f"FIRMWARE MISMATCH (telem v{got} != v{expected}) — reflash ESP32/Teensy"
        )
        self._mismatch_lbl.setVisible(True)

    def clear_version_mismatch(self):
        self._mismatch_lbl.setVisible(False)

def _as_list(value) -> list:
    """QSettings round-trips a single-item list as a bare string on some
    platforms/formats — normalize back to a list so callers can iterate."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


# ── Floating tab window ───────────────────────────────────────────────────────

class FloatingTabWindow(QMainWindow):
    """A tab popped out of the main window into its own top-level window.

    Closing the window docks the tab back (to the left pane) rather than
    destroying it — every tab holds live TelemetryBus subscriptions that
    can't be recreated, since MainWindow only constructs each one once.
    """

    def __init__(self, title: str, widget: QWidget, main_window: "MainWindow"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 700)
        self._main_window = main_window
        self._title = title
        self._widget = widget
        self._returned = False
        self.setCentralWidget(widget)
        # QTabWidget explicitly hides pages that aren't the active tab, and
        # that hidden flag persists across reparenting — force it visible.
        widget.show()

        self.menuBar().addAction("Put Back in Main Window", self.close)

    def closeEvent(self, event):
        # Reached both from the menu action above and from the OS close (X)
        # button — either way, return the tab to the main window instead of
        # destroying it (it holds live TelemetryBus subscriptions that can't
        # be recreated). Guarded so a second close() call is a harmless no-op
        # rather than re-triggering the dock logic.
        if not self._returned:
            self._returned = True
            self._main_window._dock_floating_tab(self)
        event.accept()

    def take_widget(self) -> QWidget:
        """Detach and return the hosted widget without destroying it."""
        widget = self.centralWidget()
        self.setCentralWidget(QWidget())  # leave something trivial in its place
        return widget


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    # Tri-state header thresholds (UARTplat.md Phase 4, §5.1).
    _TELEM_FRESH_MS     = 3000  # "Connected" — unchanged from the old disconnect timer
    _WIFI_DIAG_FRESH_MS = 1500  # "ESP32 ONLY" — matches firmware TEENSY_LINK_TIMEOUT_MS

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheeled-Leg Robot")
        self.resize(1400, 900)

        # Wire up singletons BEFORE creating tabs — DevicePanel._auto_scan()
        # fires synchronously during tab construction and emits port_opened;
        # SourceManager must already exist to catch those signals.
        self.status = StatusBar(self.statusBar())

        self._last_ts_ms: float | None = None

        # Tri-state connection header (UARTplat.md Phase 4, §5.1): telemetry
        # and WIFI_DIAG go stale independently (WIFI_DIAG now flows from the
        # ESP32 at 5 Hz whether or not the Teensy is talking to it — see
        # Phase 1/3), so a single "any packet restarts a 3 s timer" clock
        # can't tell "Teensy silent" apart from "fully disconnected". Instead,
        # _on_packet() timestamps the two packet families separately and this
        # timer re-evaluates the tri-state on a fixed cadence.
        self._last_telem_ms:     float | None = None
        self._last_wifi_diag_ms: float | None = None
        self._last_wifi_diag_info: dict = {}
        self._link_timer = QTimer(self)
        self._link_timer.setInterval(250)
        self._link_timer.timeout.connect(self._update_link_state)
        self._link_timer.start()

        TelemetryBus.instance().packet.connect(self._on_packet)

        sm = SourceManager.instance()
        sm.source_changed.connect(self._on_source_changed)

        from tabs.wifi_transport import WifiTransport
        WifiTransport.instance().start()

        self._log_pane = RobotLogWidget()
        self._log_pane.setContentsMargins(8, 2, 8, 6)

        # QFrame (not QWidget) so a thick border can be toggled reliably via
        # QSS as the obvious "you are looking at replayed data" indicator —
        # top-level window chrome borders are OS-dependent and unreliable.
        central = QFrame()
        central.setObjectName("central")
        self._central_style_live = f"QFrame#central {{ background: {_BG}; border: 3px solid transparent; }}"
        self._central_style_playback = f"QFrame#central {{ background: {_BG}; border: 3px solid {ORANGE}; }}"
        central.setStyleSheet(self._central_style_live)

        # Up to 2 tab panes side-by-side in a splitter, with the robot log
        # always pinned full-width below — right-click a tab for "Split
        # Right"/"Move to ... Pane"/"Float in New Window" (see
        # _show_tab_context_menu). The right pane is created on demand and
        # collapses away once its last tab leaves, so with nothing split or
        # floated this looks exactly like a single plain QTabWidget.
        self._split = QSplitter(Qt.Orientation.Horizontal)
        self._left_pane = self._make_tab_pane()
        self._right_pane = None
        self._floating: dict[QWidget, FloatingTabWindow] = {}
        self._split.addWidget(self._left_pane)

        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        central_lay.addWidget(self._split, 1)
        central_lay.addWidget(self._log_pane, 0)
        self.setCentralWidget(central)
        self._central = central

        dashboard_tab = DashboardTab()
        dashboard_tab.link_health.alert.connect(self.status.set_link_alert)
        CommandAlertBus.instance().alert.connect(self.status.set_link_alert)

        tab_defs = [
            ("Visualizer",      RobotVisualizerTab()),
            ("Dashboard",       dashboard_tab),
            ("IMU",             ImuTab()),
            ("Raw Data",        RawDataTab()),
            ("Hip Motors",      HipMotorsTab()),
            ("Parameters",      ParamsTab()),
            ("Wheel Motors",    WheelMotorsTab()),
            ("Controllers",     ControllersTab()),
            ("Radio",           RadioTab()),
            ("WiFi",            WifiDiagTab()),
            ("Logs",            LogsTab()),
            ("Log Analyzer",    LogAnalyzerTab()),
            ("Flash & Monitor", FlashMonitorTab()),
        ]
        for title, widget in tab_defs:
            self._left_pane.addTab(widget, title)
        self._tab_widgets = {title: widget for title, widget in tab_defs}

        self._restore_layout()

        self._base_title = self.windowTitle()
        TelemetryBus.instance().playback_state_changed.connect(self._on_playback_state)

        # 10 Hz heartbeat feeding the firmware MANUAL-mode GUI watchdog (500 ms):
        # if the GUI dies, pings stop and the robot exits MANUAL / idles wheels.
        from tabs.comm_commands import send_ping
        self._ping_timer = QTimer(self)
        self._ping_timer.setInterval(100)
        self._ping_timer.timeout.connect(send_ping)
        self._ping_timer.start()

        self._on_source_changed(sm.active)

    def _on_playback_state(self, active: bool):
        self._central.setStyleSheet(self._central_style_playback if active else self._central_style_live)
        self.setWindowTitle(f"[PLAYBACK] {self._base_title}" if active else self._base_title)

    # ── Split-pane / floating tab management ──────────────────────────────

    def _make_tab_pane(self) -> QTabWidget:
        pane = QTabWidget()
        pane.setMovable(True)
        pane.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        pane.tabBar().customContextMenuRequested.connect(
            lambda pos, pane=pane: self._show_tab_context_menu(pane, pos)
        )
        return pane

    def _show_tab_context_menu(self, pane: QTabWidget, pos):
        idx = pane.tabBar().tabAt(pos)

        menu = QMenu(self)
        if idx != -1:
            widget = pane.widget(idx)
            title = pane.tabText(idx)

            if pane is self._left_pane:
                label = "Split Right →" if self._right_pane is None else "Move to Right Pane"
                menu.addAction(label, lambda: self._move_tab_to_pane(widget, title, "right"))
            else:
                menu.addAction("Move to Left Pane", lambda: self._move_tab_to_pane(widget, title, "left"))
            # Visualizer/IMU embed a pyqtgraph OpenGL view, which doesn't survive
            # being reparented into a new top-level window's GL context —
            # observed to throw GL errors or hard-crash the app.
            if title not in ("Visualizer", "IMU"):
                menu.addAction("Float in New Window", lambda: self._float_tab(widget, title))
            menu.addSeparator()

        menu.addAction("Restore Default Layout", self._restore_default_layout)
        menu.exec(pane.tabBar().mapToGlobal(pos))

    def _restore_default_layout(self):
        """Undo split panes / floated windows and put every tab back in a
        single left pane, in the original registration order."""
        for win in list(self._floating.values()):
            win.close()  # closeEvent docks the tab back into the left pane

        if self._right_pane is not None:
            for title in [self._right_pane.tabText(i) for i in range(self._right_pane.count())]:
                self._move_tab_to_pane(self._tab_widgets[title], title, "left")

        self._reorder_pane_tabs(self._left_pane, list(self._tab_widgets.keys()))
        self._left_pane.setCurrentIndex(0)

    def _remove_tab_from_current_location(self, widget: QWidget):
        for pane in (self._left_pane, self._right_pane):
            if pane is None:
                continue
            idx = pane.indexOf(widget)
            if idx != -1:
                pane.removeTab(idx)
                return
        win = self._floating.pop(widget, None)
        if win is not None:
            win.take_widget()
            win._returned = True  # caller is placing the widget elsewhere itself
            win.close()

    def _move_tab_to_pane(self, widget: QWidget, title: str, target: str):
        self._remove_tab_from_current_location(widget)
        if target == "right" and self._right_pane is None:
            self._right_pane = self._make_tab_pane()
            self._split.addWidget(self._right_pane)
            self._split.setSizes([1, 1])
        pane = self._left_pane if target == "left" else self._right_pane
        pane.addTab(widget, title)
        pane.setCurrentWidget(widget)
        self._collapse_empty_pane()

    def _float_tab(self, widget: QWidget, title: str):
        self._remove_tab_from_current_location(widget)
        win = FloatingTabWindow(title, widget, self)
        win.show()
        self._floating[widget] = win
        self._collapse_empty_pane()

    def _dock_floating_tab(self, win: "FloatingTabWindow"):
        title = win._title
        widget = win.take_widget()
        self._floating.pop(widget, None)
        self._left_pane.addTab(widget, title)
        self._left_pane.setCurrentWidget(widget)

    def _collapse_empty_pane(self):
        if self._right_pane is not None and self._right_pane.count() == 0:
            self._right_pane.setParent(None)
            self._right_pane.deleteLater()
            self._right_pane = None
        if self._left_pane.count() == 0 and self._right_pane is not None:
            self._left_pane, self._right_pane = self._right_pane, self._left_pane
            self._right_pane.setParent(None)
            self._right_pane.deleteLater()
            self._right_pane = None

    # ── Split/float layout persistence ────────────────────────────────────

    def closeEvent(self, event):
        self._save_layout()
        super().closeEvent(event)

    def _reorder_pane_tabs(self, pane: QTabWidget, order: list[str]):
        for target_idx, title in enumerate(order):
            for i in range(pane.count()):
                if pane.tabText(i) == title:
                    if i != target_idx:
                        pane.tabBar().moveTab(i, target_idx)
                    break

    def _save_layout(self):
        settings = QSettings("WheeledLegRobot", "GUI")
        settings.setValue("mainWindow/geometry", self.saveGeometry())
        settings.setValue(
            "layout/left_tabs",
            [self._left_pane.tabText(i) for i in range(self._left_pane.count())],
        )
        settings.setValue(
            "layout/right_tabs",
            [self._right_pane.tabText(i) for i in range(self._right_pane.count())]
            if self._right_pane is not None else [],
        )
        if self._right_pane is not None:
            settings.setValue("layout/splitter_sizes", self._split.sizes())
        settings.setValue("layout/floating_tabs", [win._title for win in self._floating.values()])
        for win in self._floating.values():
            settings.setValue(f"layout/floating_geometry/{win._title}", win.saveGeometry())

    def _restore_layout(self):
        settings = QSettings("WheeledLegRobot", "GUI")
        geometry = settings.value("mainWindow/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        for title in _as_list(settings.value("layout/right_tabs")):
            widget = self._tab_widgets.get(title)
            if widget is not None:
                self._move_tab_to_pane(widget, title, "right")

        sizes = settings.value("layout/splitter_sizes")
        if sizes and self._right_pane is not None:
            try:
                self._split.setSizes([int(s) for s in sizes])
            except (TypeError, ValueError):
                pass

        for title in _as_list(settings.value("layout/floating_tabs")):
            widget = self._tab_widgets.get(title)
            if widget is None:
                continue
            self._float_tab(widget, title)
            win = self._floating.get(widget)
            win_geometry = settings.value(f"layout/floating_geometry/{title}")
            if win is not None and win_geometry is not None:
                win.restoreGeometry(win_geometry)

        self._reorder_pane_tabs(self._left_pane, _as_list(settings.value("layout/left_tabs")))
        if self._right_pane is not None:
            self._reorder_pane_tabs(self._right_pane, _as_list(settings.value("layout/right_tabs")))

    def _on_source_changed(self, device: str):
        # A source change means we're now looking at a different device's
        # stream — any freshness state from the previous one is irrelevant,
        # so the tri-state header shouldn't coast on it (UARTplat.md Phase 4).
        self._last_telem_ms     = None
        self._last_wifi_diag_ms = None
        if device:
            self.status.set_source(device.upper())
            self.status.set_transport(TRANSPORT_LABEL.get(device, "Unknown"))
        else:
            self.status.set_link_state("disconnected")
            self.status.set_source("—")
            self.status.set_transport("—")
            self.status.set_mode("—")
            self._last_ts_ms = None

        # Tell the ESP32 which link the GUI is now reading so it can suppress
        # WiFi UDP telemetry sends when USB is active (§2b transport gating;
        # a no-op unless the ESP32 is built with WIFI_TRANSPORT_GATING=1).
        # "teensy" bypasses the ESP32 entirely, so there's nothing useful to tell it.
        from tabs.comm_commands import send_set_telem_transport
        if device == "esp32":
            send_set_telem_transport(True)
        elif device == "wifi":
            send_set_telem_transport(False)

    def _on_packet(self, info: dict):
        now = time.monotonic() * 1000.0
        if info.get("ptype") == 0x14:
            # WIFI_DIAG: independent heartbeat from the ESP32, carries none of
            # the telemetry fields below — just record freshness for the
            # tri-state header/link-health widget and stop.
            self._last_wifi_diag_ms   = now
            self._last_wifi_diag_info = info
            return
        self._last_telem_ms = now

        if info.get("version_mismatch"):
            self.status.set_version_mismatch(
                info.get("got_version", "?"), info.get("expected_version", "?")
            )
            return
        if info.get("ptype") == 0x01:
            self.status.clear_version_mismatch()
        ts = info.get("timestamp_ms")
        if ts is not None and self._last_ts_ms is not None:
            self.status.set_dt(ts - self._last_ts_ms)
        if ts is not None:
            self._last_ts_ms = ts
        state = info.get("state_name")
        if state:
            self.status.set_mode(state, info.get("fault_name", ""), info.get("fault_description", ""),
                                 info.get("fault_code", 0))
        profile = info.get("active_profile")
        if profile is not None:
            self.status.set_profile(profile)

    def _update_link_state(self):
        """Re-evaluate the tri-state connection header. Runs on a fixed
        cadence rather than being restarted per-packet, since the whole point
        is noticing when packets *stop* arriving (see the comment in
        __init__). UARTplat.md Phase 4, §5.1."""
        now = time.monotonic() * 1000.0
        telem_age = (now - self._last_telem_ms) if self._last_telem_ms is not None else float("inf")
        diag_age  = (now - self._last_wifi_diag_ms) if self._last_wifi_diag_ms is not None else float("inf")
        if telem_age < self._TELEM_FRESH_MS:
            self.status.set_link_state("connected")
        elif diag_age < self._WIFI_DIAG_FRESH_MS:
            d = self._last_wifi_diag_info
            up    = d.get("wifi_teensy_link_up")
            since = d.get("wifi_ms_since_teensy")
            tooltip = f"Teensy link: {'UP' if up else 'DOWN'}"
            if since is not None:
                tooltip += f"  ({since} ms since last Teensy packet)"
            self.status.set_link_state("esp32_only", tooltip)
        else:
            self.status.set_link_state("disconnected")

# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--automation", type=str, default=None,
                     help="Path to a JSON automation scenario (see tabs/automation_runner.py) "
                          "— runs the scenario unattended against the real GUI and exits.")
    args, _ = ap.parse_known_args()

    _kill_other_instances()

    if sys.platform == "win32":
        # Without an explicit AppUserModelID, Windows groups this process under
        # python.exe's own taskbar icon instead of the one we set below.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WheeledLegRobot.GUI")

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    app.setWindowIcon(QIcon(_ICON_PATH))
    win = MainWindow()
    win.setWindowIcon(QIcon(_ICON_PATH))
    win.show()

    from tabs.remote_control import RemoteControlServer
    win._remote_control = RemoteControlServer(win)

    if args.automation:
        from tabs.automation_runner import AutomationRunner
        # Held as a window attribute — a bare signal/timer connection does not
        # keep a parentless QObject alive in PyQt6 (bit us before, see
        # ReliableCommand's _in_flight set, UARTplat.md Phase 5).
        win._automation_runner = AutomationRunner(args.automation, app)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
