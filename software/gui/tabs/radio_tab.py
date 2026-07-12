from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QSplitter, QVBoxLayout, QWidget,
)

from .telemetry_bus import TelemetryBus
from .theme import BG, BORDER, DIM, GREEN, RED, SURFACE, TEXT

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

_CH_NAMES = [
    "CH1  Ail", "CH2  Ele", "CH3  Thr", "CH4  Rud",
    "CH5  SwA", "CH6  SwB", "CH7  SwC", "CH8  SwD",
    "CH9",      "CH10",     "CH11",     "CH12",
    "CH13",     "CH14",
]


class RadioTab(QWidget):
    def __init__(self):
        super().__init__()

        self._bufs = [deque([CH_MID] * _BUF, maxlen=_BUF) for _ in range(NUM_CH)]

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
                name=f"CH{i + 1}",
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
            name_lbl.setFixedWidth(80)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

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

        bars_layout.addStretch()

        # ── Outer splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._plot)
        splitter.addWidget(bars_widget)
        splitter.setSizes([900, 280])
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(splitter)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── Telemetry handler ──────────────────────────────────────────────────────

    def _on_packet(self, info: dict):
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

        x = list(range(_BUF))
        for i in range(NUM_CH):
            self._curves[i].setData(x, list(self._bufs[i]))
