"""params_tab.py — Parameter Registry tab.

Displays all firmware parameters from the ParamRegistry.  On first telemetry
received (or when Refresh is clicked) sends CMD_ID_PARAM_GET 0xFFFF.  Each
PARAM_REPORT response (ptype 0x06) populates or refreshes a row.  Rows are
grouped by subsystem with section headers.  Values are editable; Enter or the
Set button sends CMD_ID_PARAM_SET and the cell flashes green on echo-back.
"""

import struct

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from comm_commands import send_param_get_all, send_param_set
from telemetry_bus import TelemetryBus
from theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

# ── Param flag bits (param_registry.h) ────────────────────────────────────────
_FLAG_PERSISTENT      = 1 << 0
_FLAG_READONLY        = 1 << 1
_FLAG_COMMAND         = 1 << 2
_FLAG_FAULT_ON_BOUNDS = 1 << 3

# ── Group map (param_ids.h GROUP_*) ───────────────────────────────────────────
_GROUP_NAMES = {
    0x00: "System",
    0x01: "Calibration",
    0x02: "Hip",
    0x03: "Wheel",
    0x04: "Control",
    0x05: "Command",
}
_GROUP_COLORS = {
    0x00: "#888888",
    0x01: BLUE,
    0x02: ORANGE,
    0x03: GREEN,
    0x04: "#cc88ff",
    0x05: "#ff88cc",
}


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _flag_text(flags: int) -> str:
    parts = []
    if flags & _FLAG_PERSISTENT:      parts.append("P")
    if flags & _FLAG_READONLY:        parts.append("R")
    if flags & _FLAG_COMMAND:         parts.append("C")
    if flags & _FLAG_FAULT_ON_BOUNDS: parts.append("!")
    return " ".join(parts) or "—"


def _flag_tooltip(flags: int) -> str:
    lines = []
    if flags & _FLAG_PERSISTENT:      lines.append("P — Persistent (saved to flash)")
    if flags & _FLAG_READONLY:        lines.append("R — Read-only (firmware writes only)")
    if flags & _FLAG_COMMAND:         lines.append("C — Command (high-freq setpoint, not saved)")
    if flags & _FLAG_FAULT_ON_BOUNDS: lines.append("! — Fault on bounds violation (triggers ESTOP)")
    return "\n".join(lines) or "No flags"


# ── One param row ─────────────────────────────────────────────────────────────

_EDIT_STYLE_NORMAL = (
    f"QLineEdit{{background:{BG};color:{TEXT};font-family:Consolas;font-size:11px;"
    f"border:1px solid {BORDER};border-radius:2px;padding:1px 4px}}"
    f"QLineEdit:focus{{border:1px solid {BLUE}}}"
    f"QLineEdit:disabled{{background:{SURFACE};color:{DIM};border:1px solid {BORDER}}}"
)
_EDIT_STYLE_PENDING = (
    f"QLineEdit{{background:{BG};color:{ORANGE};font-family:Consolas;font-size:11px;"
    f"border:1px solid {ORANGE};border-radius:2px;padding:1px 4px}}"
)
_EDIT_STYLE_OK = (
    f"QLineEdit{{background:{BG};color:{GREEN};font-family:Consolas;font-size:11px;"
    f"border:1px solid {GREEN};border-radius:2px;padding:1px 4px}}"
)


class _ParamRow(QWidget):
    def __init__(self, param_id: int, name: str, value: float,
                 min_val: float, max_val: float, flags: int):
        super().__init__()
        self._id      = param_id
        self._flags   = flags
        self._group   = (param_id >> 8) & 0xFF
        readonly      = bool(flags & _FLAG_READONLY)

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(8)

        # Name
        lbl = QLabel(name)
        lbl.setFixedWidth(190)
        lbl.setStyleSheet(f"color: {TEXT}; font-family: Consolas; font-size: 11px;")
        lay.addWidget(lbl)

        # Hex ID
        id_lbl = QLabel(f"0x{param_id:04X}")
        id_lbl.setFixedWidth(52)
        id_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(id_lbl)

        # Editable value
        self._edit = QLineEdit(f"{value:.6g}")
        self._edit.setFixedWidth(96)
        self._edit.setStyleSheet(_EDIT_STYLE_NORMAL)
        self._edit.setEnabled(not readonly)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._edit.setValidator(validator)
        self._edit.returnPressed.connect(self._send)
        lay.addWidget(self._edit)

        # Set button
        self._btn = QPushButton("Set")
        self._btn.setFixedWidth(34)
        self._btn.setEnabled(not readonly)
        self._btn.setStyleSheet(
            f"QPushButton{{background:#1a3a5a;color:white;font-size:10px;"
            f"border:1px solid {BORDER};border-radius:2px;padding:2px 4px}}"
            f"QPushButton:hover{{background:#2a4a6a}}"
            f"QPushButton:disabled{{background:{SURFACE};color:{DIM};border:1px solid {BORDER}}}"
        )
        self._btn.clicked.connect(self._send)
        lay.addWidget(self._btn)

        # Range label
        range_lbl = QLabel(f"[{min_val:.4g} … {max_val:.4g}]")
        range_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        range_lbl.setMinimumWidth(130)
        lay.addWidget(range_lbl)

        # Flags
        flag_lbl = QLabel(_flag_text(flags))
        flag_lbl.setFixedWidth(40)
        flag_lbl.setToolTip(_flag_tooltip(flags))
        flag_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(flag_lbl)

        lay.addStretch()

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def group(self) -> int:
        return self._group

    def update_value(self, value: float):
        self._edit.setText(f"{value:.6g}")
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_OK)
        self._flash_timer.start(700)

    def set_group_visible(self, group_filter: int | None):
        self.setVisible(group_filter is None or self._group == group_filter)

    # ── private ───────────────────────────────────────────────────────────────

    def _send(self):
        try:
            val = float(self._edit.text())
        except ValueError:
            return
        send_param_set(self._id, val)
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_PENDING)
        self._flash_timer.start(2500)  # revert if no echo within 2.5 s

    def _clear_flash(self):
        self._edit.setStyleSheet(_EDIT_STYLE_NORMAL)


# ── Group section header ──────────────────────────────────────────────────────

class _GroupHeader(QWidget):
    def __init__(self, group_id: int):
        super().__init__()
        name  = _GROUP_NAMES.get(group_id, f"Group 0x{group_id:02X}")
        color = _GROUP_COLORS.get(group_id, DIM)
        self._group = group_id

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 2)

        lbl = QLabel(name.upper())
        lbl.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        lay.addWidget(lbl)
        lay.addWidget(_hline())

    def set_group_visible(self, group_filter: int | None):
        self.setVisible(group_filter is None or self._group == group_filter)


# ── Main tab ──────────────────────────────────────────────────────────────────

class ParamsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._rows: dict[int, _ParamRow] = {}          # param_id → row widget
        self._headers: dict[int, _GroupHeader] = {}    # group_id → header widget
        self._requested = False

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        grp_lbl = QLabel("Group:")
        grp_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        toolbar.addWidget(grp_lbl)

        self._grp_combo = QComboBox()
        self._grp_combo.addItem("All", None)
        for gid, gname in _GROUP_NAMES.items():
            self._grp_combo.addItem(gname, gid)
        self._grp_combo.setFixedWidth(130)
        self._grp_combo.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self._grp_combo)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(72)
        btn_refresh.setStyleSheet(
            f"QPushButton{{background:{SURFACE};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 8px}}"
            f"QPushButton:hover{{background:{BORDER}}}"
        )
        btn_refresh.clicked.connect(self._request_all)
        toolbar.addWidget(btn_refresh)

        toolbar.addSpacing(8)

        self._lbl_status = QLabel("No params loaded — connect and click Refresh")
        self._lbl_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        toolbar.addWidget(self._lbl_status)
        toolbar.addStretch()

        # ── Column header bar ─────────────────────────────────────────────────
        col_bar = QWidget()
        col_bar.setStyleSheet(f"background: {SURFACE};")
        col_lay = QHBoxLayout(col_bar)
        col_lay.setContentsMargins(6, 3, 6, 3)
        col_lay.setSpacing(8)
        for txt, w in [("Name", 190), ("ID", 52), ("Value", 96), ("Set", 34),
                       ("Range", 130), ("Flags", 40)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; font-weight: bold;")
            col_lay.addWidget(lbl)
        col_lay.addStretch()

        # ── Scrollable row area ───────────────────────────────────────────────
        self._inner = QWidget()
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(0, 0, 0, 0)
        self._inner_lay.setSpacing(0)
        self._inner_lay.addStretch()  # keeps rows packed to top

        scroll = QScrollArea()
        scroll.setWidget(self._inner)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{border: 1px solid {BORDER}; background: {BG};}}"
            f"QScrollBar:vertical{{background:{SURFACE};width:8px}}"
            f"QScrollBar::handle:vertical{{background:{BORDER};border-radius:4px}}"
        )

        # ── Outer layout ──────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addLayout(toolbar)
        lay.addWidget(col_bar)
        lay.addWidget(scroll, stretch=1)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _request_all(self):
        send_param_get_all()
        self._requested = True
        self._lbl_status.setText("Requesting…")
        self._lbl_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")

    def _on_packet(self, info: dict):
        ptype = info.get("ptype")

        # Auto-request on first telemetry if nothing loaded yet
        if ptype == 0x01 and not self._rows and not self._requested:
            self._request_all()
            return

        if ptype != 0x06:
            return

        param_id = info.get("param_id")
        value    = info.get("param_value")
        if param_id is None or value is None:
            return

        if param_id in self._rows:
            self._rows[param_id].update_value(value)
        else:
            self._add_row(
                param_id,
                info.get("param_name", f"0x{param_id:04X}"),
                value,
                info.get("param_min", 0.0),
                info.get("param_max", 0.0),
                info.get("param_flags", 0),
            )

        n = len(self._rows)
        self._lbl_status.setText(f"{n} param{'s' if n != 1 else ''} loaded")
        self._lbl_status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

    def _apply_filter(self):
        gid = self._grp_combo.currentData()
        for row in self._rows.values():
            row.set_group_visible(gid)
        for hdr in self._headers.values():
            hdr.set_group_visible(gid)

    # ── private ───────────────────────────────────────────────────────────────

    def _add_row(self, param_id: int, name: str, value: float,
                 min_val: float, max_val: float, flags: int):
        group = (param_id >> 8) & 0xFF

        # Insert group header the first time we see this group
        if group not in self._headers:
            hdr = _GroupHeader(group)
            # insert before the trailing stretch (last item)
            self._inner_lay.insertWidget(self._inner_lay.count() - 1, hdr)
            self._headers[group] = hdr

        row = _ParamRow(param_id, name, value, min_val, max_val, flags)
        self._inner_lay.insertWidget(self._inner_lay.count() - 1, row)
        self._rows[param_id] = row

        # Apply current filter to the new row
        gid = self._grp_combo.currentData()
        row.set_group_visible(gid)
        if group in self._headers:
            self._headers[group].set_group_visible(gid)
