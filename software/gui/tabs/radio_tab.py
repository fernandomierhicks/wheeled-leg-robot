from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QSplitter, QVBoxLayout, QWidget,
)

from .telemetry_bus import TelemetryBus
from .theme import BG, BORDER, DIM, GREEN, RED, SURFACE, TEXT

# Firmware mirrors CH1–CH14 only: TelemetryPayload.ibus_ch[] is sized by the
# payload, not by CRSF's 16 (main.cpp, RC_MIRROR_CH). CH15 (coordinated-turn
# lean) and CH16 are live in firmware but never reach the GUI — watch those on
# the transmitter's own Outputs screen, which shows all 16. See
# firmware/robot_teensy/radio_channels.md → "Where to watch these channels".
NUM_CH = 14
CH_MIN = 1000
CH_MAX = 2000
CH_MID = 1500
_BUF   = 300  # rolling chart samples

_COLORS = [
    "#ff6b6b", "#ffa94d", "#ffe066", "#69db7c",
    "#4dabf7", "#cc5de8", "#f783ac", "#a9e34b",
    "#38d9a9", "#74c0fc", "#da77f2", "#ff8787",
    "#ffd43b", "#63e6be",
]

# (channel, TX15 control, short function, tooltip) — the control column is the
# name the *radio* uses on its Inputs/Outputs screens, so a reading here can be
# cross-checked against the transmitter without a translation step. Keep in
# sync with radio_channels.md and radio/CHANNELS.md.
_CH_INFO = [
    ("CH1",  "Ail",  "Roll",        "Right stick ↔ — roll / lean setpoint, × the profile's roll_max. Only acts when roll_ctrl_en=1 in RUNNING."),
    ("CH2",  "Ele",  "Velocity",    "Right stick ↕ — forward velocity command, × the profile's vel_max."),
    ("CH3",  "Thr",  "Hip height",  "Left stick ↕ — hip height as t ∈ [0,1], slewed by hip_cmd_rate_lim. The non-centring stick: leg height is a held pose."),
    ("CH4",  "Rud",  "Yaw rate",    "Left stick ↔ — yaw rate command, × the profile's yaw_max. Inverted in firmware so stick-left yaws left."),
    ("CH5",  "SB",   "SD log",      "Top row, 2nd from left. Edge-triggered: high starts recording, low stops. A start is refused outside STANDBY/ESTOP, but recording continues through RUNNING."),
    ("CH6",  "SF",   "Jump",        "Momentary shoulder. Rising edge > 1990 µs requests STATE_JUMPING — one jump per edge. Needs RUNNING and jump_enable=1 (default 0)."),
    ("CH7",  "S1",   "Tune A",      "Left dial — live-tune slot 0 of the active gain group. Inert unless live_tune_multi_en=1 and a group button is lit."),
    ("CH8",  "S2",   "Tune B",      "Right dial — live-tune slot 1 of the active gain group. Inert unless live_tune_multi_en=1 and a group button is lit."),
    ("CH9",  "SC",   "Profile",     "Top row, right of the dials. < 1333 = profile 1, 1333–1667 = profile 2, > 1667 = profile 3. Selects vel_max, yaw_max, torque_lim and roll_max."),
    ("CH10", "SD",   "ARM",         "Top row, rightmost. Level-based: > 1990 µs arms into RUNNING (requires calibration); drop disarms to STANDBY. Reads 0 on link loss, so a dead radio can never look armed."),
    ("CH11", "SA",   "Calibrate",   "Top row, leftmost. Rising edge requests CALIBRATION from STANDBY; re-trigger cancels a radio-owned calibration through DISARMING. 1 s lockout, release required between edges."),
    ("CH12", "SE",   "Reset fault", "Latching shoulder. Rising edge in ESTOP: full reset to STARTUP, clearing fault_code and re-running the startup checks. In STANDBY: beep only. Never armed with torque live."),
    ("CH13", "SGHI", "Tune group",  "RGB buttons 1–3, mutually exclusive, encoded as levels on one channel: ~1500 µs none, 1660 group 0, 1830 group 1, 2000 group 2. The lit button is the group indicator."),
    ("CH14", "SJ",   "Latch gains", "RGB button 4. Rising edge fires the same one-shot as writing live_tune_latch from the Params tab. Only picked-up slots are committed."),
]

_CH_NAMES = [f"{ch:<5}{sw:<5}{fn}" for ch, sw, fn, _ in _CH_INFO]


class RadioTab(QWidget):
    def __init__(self):
        super().__init__()

        self._bufs = [deque([CH_MID] * _BUF, maxlen=_BUF) for _ in range(NUM_CH)]
        self._x = list(range(_BUF))

        # ── Rolling chart ──────────────────────────────────────────────────────
        self._plot = pg.PlotWidget()
        self._plot.setBackground(BG)
        self._plot.setTitle("RC Channels vs Time", color=TEXT, size="11pt")
        self._plot.setLabel("left", "µs", color=DIM)
        self._plot.setYRange(CH_MIN - 50, CH_MAX + 50)
        self._plot.setXRange(0, _BUF)
        self._plot.getAxis("bottom").setStyle(showValues=False)
        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)

        mid_line = pg.InfiniteLine(
            pos=CH_MID, angle=0,
            pen=pg.mkPen(BORDER, width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(mid_line)

        self._plot.addLegend(offset=(5, 5))
        self._curves = []
        for i in range(NUM_CH):
            curve = self._plot.plot(
                list(self._bufs[i]),
                pen=pg.mkPen(_COLORS[i], width=1.2),
                name=f"{_CH_INFO[i][0]} {_CH_INFO[i][2]}",
            )
            self._curves.append(curve)

        # ── Bar gauges panel ───────────────────────────────────────────────────
        bars_widget = QWidget()
        bars_widget.setStyleSheet(
            f"QWidget {{ background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 3px; }}"
            f"QLabel {{ border: none; }}"
        )
        bars_layout = QVBoxLayout(bars_widget)
        bars_layout.setContentsMargins(12, 10, 12, 10)
        bars_layout.setSpacing(5)

        self._signal_lbl = QLabel("● No Signal")
        self._signal_lbl.setStyleSheet(
            f"color: {RED}; font-weight: bold; font-size: 12px;"
        )
        bars_layout.addWidget(self._signal_lbl)

        sep = QLabel()
        sep.setFixedHeight(6)
        bars_layout.addWidget(sep)

        self._bars:    list[QProgressBar] = []
        self._val_lbls: list[QLabel]      = []

        for i in range(NUM_CH):
            row = QHBoxLayout()
            row.setSpacing(8)

            name_lbl = QLabel(_CH_NAMES[i])
            name_lbl.setStyleSheet(
                f"color: {_COLORS[i]}; font-size: 10px; font-family: Consolas;"
            )
            name_lbl.setFixedWidth(148)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            name_lbl.setToolTip(_CH_INFO[i][3])

            bar = QProgressBar()
            bar.setRange(CH_MIN, CH_MAX)
            bar.setValue(CH_MID)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            bar.setStyleSheet(
                f"QProgressBar {{ background: {BG}; border: 1px solid {BORDER};"
                f" border-radius: 2px; }}"
                f"QProgressBar::chunk {{ background: {_COLORS[i]}; border-radius: 1px; }}"
            )

            val_lbl = QLabel(f"{CH_MID}")
            val_lbl.setStyleSheet(
                f"color: {TEXT}; font-size: 10px; font-family: Consolas;"
            )
            val_lbl.setFixedWidth(38)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row.addWidget(name_lbl)
            row.addWidget(bar, stretch=1)
            row.addWidget(val_lbl)
            bars_layout.addLayout(row)

            self._bars.append(bar)
            self._val_lbls.append(val_lbl)

        # CH15/CH16 exist in firmware but not in the telemetry mirror. Say so
        # here rather than leaving the list looking complete at 14.
        note_lbl = QLabel(
            "CH15 SK   Turn lean\nCH16 —    spare\nnot mirrored — read them on the radio's Outputs screen"
        )
        note_lbl.setStyleSheet(f"color: {DIM}; font-size: 9px; font-family: Consolas;")
        note_lbl.setToolTip(
            "TelemetryPayload.ibus_ch[] carries CH1–CH14 only (main.cpp, RC_MIRROR_CH).\n"
            "CH15 drives the coordinated-turn lean; CH16 is spare. The transmitter's\n"
            "own Outputs screen shows all 16 channels live, so it is the instrument\n"
            "for verifying these two."
        )
        bars_layout.addSpacing(8)
        bars_layout.addWidget(note_lbl)

        bars_layout.addStretch()

        # ── Outer splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._plot)
        splitter.addWidget(bars_widget)
        splitter.setSizes([820, 360])
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(splitter)

        TelemetryBus.instance().packet.connect(self._on_packet)

        # Redrawing 14 pyqtgraph curves (300 pts each) is expensive — decouple
        # it from raw packet rate (up to 50 Hz) onto a fixed-interval timer,
        # same pattern as controllers_tab.py's chart timer. Buffer updates and
        # bar/label text stay in _on_packet since those are cheap.
        self._chart_timer = QTimer(self)
        self._chart_timer.setInterval(50)  # 20 Hz chart refresh
        self._chart_timer.timeout.connect(self._refresh_charts)
        self._chart_timer.start()

    # ── Telemetry handler ──────────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        if info.get("ptype") == 0x01 and not self.isVisible():
            return
        if info.get("ptype") != 0x01:
            return
        channels = info.get("ibus_ch")
        if channels is None:
            return

        alive = info.get("ibus_alive", False)
        self._signal_lbl.setText("● Signal OK" if alive else "● No Signal")
        self._signal_lbl.setStyleSheet(
            f"color: {GREEN}; font-weight: bold; font-size: 12px;"
            if alive else
            f"color: {RED}; font-weight: bold; font-size: 12px;"
        )

        for i, val in enumerate(channels[:NUM_CH]):
            self._bufs[i].append(val)
            self._bars[i].setValue(val)
            self._val_lbls[i].setText(str(val))

    def _refresh_charts(self) -> None:
        for i in range(NUM_CH):
            self._curves[i].setData(self._x, list(self._bufs[i]))
