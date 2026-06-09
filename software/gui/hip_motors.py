"""hip_motors.py — Hip Motors tab (AK45-10, MIT CAN mode).

Displays live position + command torque for both hip axes.
Controls: Enable / Disable / Zero per-motor and both,
          MIT position command (p°, Kp, Kd, τff).

Telemetry fields used (from TelemetryPayload / comm_protocol.h):
    hip_l_pos_rad, hip_r_pos_rad   — hip encoder position [rad]
    cmd_l, cmd_r                   — torque command sent to each hip [N·m]

Commands are framed via the CommLink protocol (shared/comm_protocol.h) and
written to the Teensy serial port through SerialPortManager.
"""

import math
import struct
from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QVBoxLayout, QWidget, QSizePolicy,
)

from port_manager import SerialPortManager
from telemetry_bus import TelemetryBus
from theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

_BUF = 750          # rolling chart samples (~1.5 s at 500 Hz)
HIP_TORQUE_MAX = 7.0  # N·m — used for limit lines

# ── CommLink frame constants (shared/comm_protocol.h) ─────────────────────────
_COMM_START    = 0xFF
_COMM_END      = 0xFE
_COMM_SRC_PC   = 0x03
_COMM_TYPE_CMD = 0x02
_CMD_PAYLOAD_V = 1

# Command IDs (comm_protocol.h CMD_ID_*)
_CMD_ID_SET_MODE = 0x01
_CMD_ID_HIP      = 0x05

# Hip motor IDs
_HIP_MOTOR_BOTH  = 0x00
_HIP_MOTOR_L     = 0x01
_HIP_MOTOR_R     = 0x02

# Hip sub-commands
_HIP_CMD_DISABLE = 0x00
_HIP_CMD_ENABLE  = 0x01
_HIP_CMD_ZERO    = 0x02
_HIP_CMD_MIT     = 0x03

# Robot state IDs (robot_state.h RobotStateEnum)
_STATE_STANDBY = 2
_STATE_MANUAL  = 5

# Keep old alias so nothing else breaks
_CMD_HIP_MOTOR = _CMD_ID_HIP

_seq = [0]  # rolling Tx sequence counter


# ── CommLink framing helpers ──────────────────────────────────────────────────

def _build_frame(payload: bytes) -> bytes:
    """Wrap payload in a CommLink COMMAND frame."""
    seq = _seq[0] & 0xFF
    _seq[0] += 1
    plen = len(payload)
    header = bytes([_COMM_TYPE_CMD, _CMD_PAYLOAD_V, _COMM_SRC_PC,
                    seq, plen & 0xFF, (plen >> 8) & 0xFF])
    crc = 0
    for b in header + payload:
        crc ^= b
    return bytes([_COMM_START]) + header + payload + bytes([crc, _COMM_END])


def _send(frame: bytes):
    """Write a frame to the Teensy serial port (no-op if port is closed)."""
    pm = SerialPortManager.instance()
    with pm._lock:
        s = pm._open.get("teensy")
    if s and s.is_open:
        try:
            s.write(frame)
        except Exception:
            pass


# ── Small UI helpers ──────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _readout(parent_lay: QVBoxLayout, label: str,
             color: str = TEXT) -> QLabel:
    """Add a dim-label / value-label row; return the value QLabel."""
    row = QHBoxLayout()
    row.setSpacing(6)
    k = QLabel(label + ":")
    k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    v = QLabel("—")
    v.setStyleSheet(
        f"color: {color}; font-size: 13px; font-weight: bold;"
        f" font-family: Consolas;"
    )
    row.addWidget(k)
    row.addStretch()
    row.addWidget(v)
    parent_lay.addLayout(row)
    return v


def _colored_btn(label: str, bg: str) -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton{{background:{bg};color:white;"
        f"border:1px solid {BORDER};border-radius:3px;padding:4px 10px}}"
        f"QPushButton:hover{{background:{bg}cc}}"
        f"QPushButton:pressed{{background:{bg}88}}"
    )
    return b


def _spin(lo: float, hi: float, step: float,
          val: float, dec: int) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(dec)
    s.setFixedWidth(72)
    s.setStyleSheet(
        f"QDoubleSpinBox{{background:{BG};color:{TEXT};"
        f"font-family:Consolas;font-size:10px;"
        f"border:1px solid {BORDER};border-radius:3px;padding:1px 4px}}"
        f"QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{{width:14px}}"
    )
    return s


# ── Per-motor panel ───────────────────────────────────────────────────────────

class _MotorPanel(QWidget):
    """Live readout + mini chart + controls for one AK45 hip axis."""

    def __init__(self, motor_id: int, title: str, node_hex: str):
        super().__init__()
        self._mid = motor_id
        self._pos_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._tau_buf: deque = deque([0.0] * _BUF, maxlen=_BUF)

        self.setObjectName("MotorPanel")
        self.setStyleSheet(
            f"#MotorPanel {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
            f"QLabel, QPushButton, QDoubleSpinBox {{border: none;}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        # Header
        hdr = QLabel(
            f"<b>{title}</b>"
            f"  <span style='color:{DIM};font-size:10px'>({node_hex})</span>"
        )
        hdr.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        lay.addWidget(hdr)
        lay.addWidget(_hline())

        # Live readouts
        self._lbl_pos = _readout(lay, "Position", BLUE)
        self._lbl_tau = _readout(lay, "Cmd torque", ORANGE)
        lay.addWidget(_hline())

        # Mini chart — pos (blue) + tau (orange)
        self._chart = pg.PlotWidget()
        self._chart.setBackground(BG)
        self._chart.setMaximumHeight(115)
        self._chart.setMinimumHeight(80)
        self._chart.showGrid(x=True, y=True, alpha=0.12)
        self._chart.setXRange(0, _BUF)
        self._chart.getAxis("bottom").setStyle(showValues=False)
        leg = self._chart.addLegend(offset=(4, 4), verSpacing=-4)
        self._crv_pos = self._chart.plot(
            list(self._pos_buf), pen=pg.mkPen(BLUE,   width=1.5), name="pos (°)")
        self._crv_tau = self._chart.plot(
            list(self._tau_buf), pen=pg.mkPen(ORANGE, width=1.2), name="τ (N·m)")
        lay.addWidget(self._chart)
        lay.addWidget(_hline())

        # ── Controls container (disabled until MANUAL mode) ───────────────────
        self._ctrl = QWidget()
        ctrl_lay = QVBoxLayout(self._ctrl)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(6)

        # Enable / Disable / Zero
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        for lbl, cv, col in [("Enable",  _HIP_CMD_ENABLE,  "#2e7d32"),
                              ("Disable", _HIP_CMD_DISABLE, "#7d2e2e"),
                              ("Zero",    _HIP_CMD_ZERO,    "#4a4a2e")]:
            b = _colored_btn(lbl, col)
            b.clicked.connect(lambda _, m=motor_id, c=cv: self._simple_cmd(m, c))
            btn_row.addWidget(b)
        btn_row.addStretch()
        ctrl_lay.addLayout(btn_row)
        ctrl_lay.addWidget(_hline())

        # ── Presets: fill Kp / Kd ────────────────────────────────────────────
        _PRESETS = [
            ("Springy", 15.0,  1.5, "#3a5a3a"),
            ("Normal",  50.0,  3.0, "#3a3a5a"),
            ("Stiff",  200.0,  4.5, "#5a3a2a"),
        ]
        pre_row = QHBoxLayout()
        pre_row.setSpacing(6)
        pre_lbl = QLabel("Preset:")
        pre_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        pre_row.addWidget(pre_lbl)
        for name, kp_val, kd_val, bg in _PRESETS:
            b = QPushButton(name)
            b.setStyleSheet(
                f"QPushButton{{background:{bg};color:white;"
                f"border:1px solid {BORDER};border-radius:3px;padding:3px 10px}}"
                f"QPushButton:hover{{background:{bg}cc}}"
            )
            b.clicked.connect(lambda _, kp=kp_val, kd=kd_val: self._apply_preset(kp, kd))
            pre_row.addWidget(b)
        pre_row.addStretch()
        ctrl_lay.addLayout(pre_row)

        # ── MIT spinboxes ─────────────────────────────────────────────────────
        mit_row = QHBoxLayout()
        mit_row.setSpacing(4)
        self._sp_p  = _spin(-716, 716, 1.0, 0.0, 1)
        self._sp_kp = _spin(0, 500,   5.0, 50.0, 1)
        self._sp_kd = _spin(0,   5,   0.1,  3.0, 2)
        self._sp_tf = _spin(-8,  8,   0.1,  0.0, 2)
        for lbl_txt, sp in [("p°", self._sp_p), ("Kp", self._sp_kp),
                             ("Kd", self._sp_kd), ("τff", self._sp_tf)]:
            k = QLabel(lbl_txt)
            k.setStyleSheet(f"color: {DIM}; font-size: 10px;")
            mit_row.addWidget(k)
            mit_row.addWidget(sp)
        mit_row.addStretch()
        ctrl_lay.addLayout(mit_row)

        # ── Angle limits + Send ───────────────────────────────────────────────
        lim_row = QHBoxLayout()
        lim_row.setSpacing(4)
        lim_lbl = QLabel("Limits:")
        lim_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        lim_row.addWidget(lim_lbl)
        lo_lbl = QLabel("min")
        lo_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_min = _spin(-716, 716, 5.0, -90.0, 1)
        hi_lbl = QLabel("max")
        hi_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        self._sp_max = _spin(-716, 716, 5.0, +90.0, 1)
        lim_row.addWidget(lo_lbl)
        lim_row.addWidget(self._sp_min)
        lim_row.addWidget(hi_lbl)
        lim_row.addWidget(self._sp_max)
        lim_row.addSpacing(8)
        btn_mit = QPushButton("Send MIT")
        btn_mit.setStyleSheet(
            f"QPushButton{{background:#1a4a7a;color:white;"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 10px}}"
            f"QPushButton:hover{{background:#2a5a8a}}"
        )
        btn_mit.clicked.connect(self._send_mit)
        lim_row.addWidget(btn_mit)
        lim_row.addStretch()
        ctrl_lay.addLayout(lim_row)

        self._ctrl.setEnabled(False)
        lay.addWidget(self._ctrl)
        lay.addStretch()

    def set_controls_enabled(self, enabled: bool):
        self._ctrl.setEnabled(enabled)

    # ── command helpers ───────────────────────────────────────────────────────

    def _simple_cmd(self, motor_id: int, hip_sub: int):
        payload = struct.pack("<BBB", _CMD_ID_HIP, motor_id, hip_sub)
        _send(_build_frame(payload))

    def _apply_preset(self, kp: float, kd: float):
        self._sp_kp.setValue(kp)
        self._sp_kd.setValue(kd)

    def _send_mit(self):
        # Clamp p° to [min, max] and snap the spinbox so it stays in range visually
        lo_deg  = self._sp_min.value()
        hi_deg  = self._sp_max.value()
        p_deg   = max(lo_deg, min(hi_deg, self._sp_p.value()))
        self._sp_p.setValue(p_deg)
        p_rad   = math.radians(p_deg)
        payload = struct.pack("<BBBfffff",
                              _CMD_ID_HIP, self._mid, _HIP_CMD_MIT,
                              p_rad, 0.0,
                              self._sp_kp.value(), self._sp_kd.value(),
                              self._sp_tf.value())
        _send(_build_frame(payload))

    # ── data update (called from HipMotorsTab._on_packet) ────────────────────

    def update_data(self, pos_rad: float, tau_nm: float):
        pos_deg = math.degrees(pos_rad)
        self._lbl_pos.setText(f"{pos_deg:+.1f}°")
        self._lbl_tau.setText(f"{tau_nm:+.2f} N·m")

        self._pos_buf.append(pos_deg)
        self._tau_buf.append(tau_nm)
        self._crv_pos.setData(list(self._pos_buf))
        self._crv_tau.setData(list(self._tau_buf))

        # auto-fit mini chart Y to both traces together
        lo = min(min(self._pos_buf), min(self._tau_buf))
        hi = max(max(self._pos_buf), max(self._tau_buf))
        span = max(hi - lo, 5.0)
        mid  = (lo + hi) / 2
        self._chart.setYRange(mid - span * 0.6, mid + span * 0.6, padding=0.05)


# ── Main tab ──────────────────────────────────────────────────────────────────

class HipMotorsTab(QWidget):
    def __init__(self):
        super().__init__()

        self._panel_L = _MotorPanel(1, "Hip L", "0x41")
        self._panel_R = _MotorPanel(2, "Hip R", "0x42")

        # ── Centre: Both controls ─────────────────────────────────────────────
        both = QWidget()
        both.setEnabled(False)
        both.setFixedWidth(120)
        both_lay = QVBoxLayout(both)
        both_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        both_lay.setContentsMargins(4, 0, 4, 0)
        both_lay.setSpacing(10)
        lbl = QLabel("Both")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {DIM}; font-size: 11px; font-weight: bold;")
        both_lay.addWidget(lbl)
        for lbl_txt, cv, col in [
                ("Enable\nBoth",  _HIP_CMD_ENABLE,  "#2e7d32"),
                ("Disable\nBoth", _HIP_CMD_DISABLE, "#7d2e2e"),
                ("Zero\nBoth",    _HIP_CMD_ZERO,    "#4a4a2e")]:
            b = _colored_btn(lbl_txt, col)
            b.clicked.connect(lambda _, c=cv: self._both_cmd(c))
            both_lay.addWidget(b)

        self._both = both

        # ── Top row ───────────────────────────────────────────────────────────
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(8)
        top_lay.addWidget(self._panel_L, stretch=1)
        top_lay.addWidget(both)
        top_lay.addWidget(self._panel_R, stretch=1)

        # ── Bottom: position + torque charts (both motors together) ───────────
        self._pos_L: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._pos_R: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._tau_L: deque = deque([0.0] * _BUF, maxlen=_BUF)
        self._tau_R: deque = deque([0.0] * _BUF, maxlen=_BUF)

        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(BG)

        def _chart(row, col, title, ylabel):
            p = glw.addPlot(row=row, col=col)
            p.setTitle(f'<span style="color:{TEXT}">{title}</span>', size="10pt")
            p.setLabel("left", f'<span style="color:{DIM}">{ylabel}</span>')
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setXRange(0, _BUF)
            p.getAxis("bottom").setStyle(showValues=False)
            p.addLegend(offset=(5, 5), verSpacing=-4)
            return p

        p_pos = _chart(0, 0, "Hip Position", "deg")
        p_pos.enableAutoRange(axis="y", enable=True)
        self._ln_pos_L = p_pos.plot(
            list(self._pos_L), pen=pg.mkPen(BLUE,  width=1.5), name="L")
        self._ln_pos_R = p_pos.plot(
            list(self._pos_R), pen=pg.mkPen(GREEN, width=1.5), name="R")

        p_tau = _chart(0, 1, "Cmd Torque", "N·m")
        p_tau.setYRange(-HIP_TORQUE_MAX * 1.1, HIP_TORQUE_MAX * 1.1, padding=0)
        p_tau.disableAutoRange()
        for sign in (1, -1):
            p_tau.addItem(pg.InfiniteLine(
                pos=sign * HIP_TORQUE_MAX, angle=0,
                pen=pg.mkPen(RED, width=1, style=Qt.PenStyle.DashLine),
                label=f'{"+" if sign > 0 else "−"}{HIP_TORQUE_MAX:.1f}',
                labelOpts={"color": RED, "anchors": [(0.05, 1.1), (0.05, 1.1)]}))
        self._ln_tau_L = p_tau.plot(
            list(self._tau_L), pen=pg.mkPen(BLUE,  width=1.5), name="L")
        self._ln_tau_R = p_tau.plot(
            list(self._tau_R), pen=pg.mkPen(GREEN, width=1.5), name="R")

        # ── Manual mode bar ───────────────────────────────────────────────────
        mode_bar = QWidget()
        mode_bar.setObjectName("ModeBar")
        mode_bar.setFixedHeight(36)
        mode_bar.setStyleSheet(
            f"#ModeBar {{background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px;}}"
            f"QLabel, QPushButton {{border: none;}}"
        )
        mb_lay = QHBoxLayout(mode_bar)
        mb_lay.setContentsMargins(10, 4, 10, 4)
        mb_lay.setSpacing(10)

        self._lbl_mode = QLabel("STANDBY")
        self._lbl_mode.setStyleSheet(
            f"color: {ORANGE}; font-size: 12px; font-weight: bold;"
            f" font-family: Consolas;"
        )
        mb_lay.addWidget(self._lbl_mode)
        mb_lay.addStretch()

        mode_hint = QLabel("Hip commands are only executed in MANUAL mode")
        mode_hint.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        mb_lay.addWidget(mode_hint)
        mb_lay.addStretch()

        self._btn_enter = _colored_btn("Enter Manual", "#1a4a6a")
        self._btn_exit  = _colored_btn("Exit Manual",  "#4a2a1a")
        self._btn_enter.clicked.connect(lambda: self._set_mode(_STATE_MANUAL))
        self._btn_exit .clicked.connect(lambda: self._set_mode(_STATE_STANDBY))
        mb_lay.addWidget(self._btn_enter)
        mb_lay.addWidget(self._btn_exit)

        # ── Outer splitter ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(glw)
        splitter.setSizes([460, 220])
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(mode_bar)
        lay.addWidget(splitter)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── commands ──────────────────────────────────────────────────────────────

    def _set_mode(self, target: int):
        payload = struct.pack("<BB", _CMD_ID_SET_MODE, target)
        _send(_build_frame(payload))

    def _both_cmd(self, hip_sub: int):
        payload = struct.pack("<BBB", _CMD_ID_HIP, _HIP_MOTOR_BOTH, hip_sub)
        _send(_build_frame(payload))

    # ── telemetry ─────────────────────────────────────────────────────────────

    _STATE_LABELS = {
        0: ("STARTUP",     "#aaaaaa"),
        1: ("CALIBRATION", "#4488ff"),
        2: ("STANDBY",     "#ffcc00"),
        3: ("RUNNING",     "#44ff88"),
        4: ("ESTOP",       "#ff4444"),
        5: ("MANUAL",      "#00ccff"),
    }

    def _on_packet(self, info: dict):
        if info.get("ptype") != 0x01:
            return

        state_id = info.get("robot_state", 0)
        label, color = self._STATE_LABELS.get(state_id, (f"STATE {state_id}", "#aaaaaa"))
        fault = info.get("fault_name", "")
        if state_id == 4 and fault and fault != "NONE":  # ESTOP
            label = f"ESTOP [{fault}]"
        self._lbl_mode.setText(label)
        self._lbl_mode.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: bold; font-family: Consolas;"
        )

        in_manual = (state_id == _STATE_MANUAL)
        self._panel_L.set_controls_enabled(in_manual)
        self._panel_R.set_controls_enabled(in_manual)
        self._both.setEnabled(in_manual)

        pos_l = info.get("hip_l_pos_rad", 0.0)
        pos_r = info.get("hip_r_pos_rad", 0.0)
        tau_l = info.get("cmd_l", 0.0)
        tau_r = info.get("cmd_r", 0.0)

        # Update per-motor panels
        self._panel_L.update_data(pos_l, tau_l)
        self._panel_R.update_data(pos_r, tau_r)

        # Update bottom charts
        self._pos_L.append(math.degrees(pos_l))
        self._pos_R.append(math.degrees(pos_r))
        self._tau_L.append(tau_l)
        self._tau_R.append(tau_r)
        self._ln_pos_L.setData(list(self._pos_L))
        self._ln_pos_R.setData(list(self._pos_R))
        self._ln_tau_L.setData(list(self._tau_L))
        self._ln_tau_R.setData(list(self._tau_R))
