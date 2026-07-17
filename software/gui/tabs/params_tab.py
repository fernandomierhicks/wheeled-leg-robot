"""params_tab.py — Parameter Registry tab.

Displays all firmware parameters from the ParamRegistry.  On first telemetry
received (or when Refresh is clicked) sends CMD_ID_PARAM_GET 0xFFFF.  Each
PARAM_REPORT response (ptype 0x06) populates or refreshes a row.  Rows are
grouped by subsystem and split into collapsible sub-sections (all start
collapsed).  Values are editable; Enter or the Set button sends
CMD_ID_PARAM_SET and the cell flashes green on echo-back.
"""

import json
import struct

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .comm_commands import send_param_get_all, send_param_reset_defaults, send_param_set
from .telemetry_bus import TelemetryBus
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

# ── Param flag bits (param_registry.h) ────────────────────────────────────────
_FLAG_PERSISTENT      = 1 << 0
_FLAG_READONLY        = 1 << 1
_FLAG_COMMAND         = 1 << 2

# ── Group map (param_ids.h GROUP_*) ───────────────────────────────────────────
_GROUP_NAMES = {
    0x00: "System",
    0x01: "Calibration",
    0x02: "Hip",
    0x03: "Wheel",
    0x04: "Control",
    0x05: "Command",
    0x06: "RC Receiver (iBus)",
}
_GROUP_COLORS = {
    0x00: "#888888",
    0x01: BLUE,
    0x02: ORANGE,
    0x03: GREEN,
    0x04: "#cc88ff",
    0x05: "#ff88cc",
    0x06: "#88ddff",
}

# ── Sub-group definitions (Control group split into logical sections) ──────────
# Each entry: (param_id_range, parent_group_id, sub_group_label)
_SUBGROUPS: list[tuple[range, int, str]] = [
    (range(0x0400, 0x0404), 0x04, "LQR Core"),
    (range(0x0404, 0x040C), 0x04, "Velocity PI"),
    (range(0x040C, 0x0412), 0x04, "Yaw PI"),
    (range(0x0412, 0x0415), 0x04, "Feedforward"),
    (range(0x0415, 0x0420), 0x04, "Jump"),
    (range(0x0420, 0x0423), 0x04, "Sim Injection"),  # enable_sim_pitch, sim_pitch_rate, enable_sim_prate
    (range(0x0500, 0x0504), 0x05, "Radio Scale"),    # radio_hip_cmd, radio_vel_max, radio_yaw_max, radio_pitch_trim
    (range(0x0510, 0x051A), 0x05, "Speed Profiles"), # profile 1/2/3 params + active_profile
]

_SUBGROUP_COLORS: dict[str, str] = {
    "LQR Core":       "#dd99ff",
    "Velocity PI":    "#bb77ee",
    "Yaw PI":         "#9966dd",
    "Feedforward":    "#ccaaff",
    "Jump":           "#ffaa44",
    "Sim Injection":  "#88ddcc",
    "Radio Scale":    "#ff88cc",
    "Speed Profiles": "#ffcc66",
}


def _get_subgroup(param_id: int) -> str | None:
    for r, _, name in _SUBGROUPS:
        if param_id in r:
            return name
    return None


def _hline(color: str = BORDER) -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {color};")
    f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return f


def _flag_text(flags: int) -> str:
    parts = []
    if flags & _FLAG_PERSISTENT:      parts.append("P")
    if flags & _FLAG_READONLY:        parts.append("R")
    if flags & _FLAG_COMMAND:         parts.append("C")
    return " ".join(parts) or "—"


def _flag_tooltip(flags: int) -> str:
    lines = []
    if flags & _FLAG_PERSISTENT:      lines.append("P — Persistent (saved to flash)")
    if flags & _FLAG_READONLY:        lines.append("R — Read-only (firmware writes only)")
    if flags & _FLAG_COMMAND:         lines.append("C — Command (high-freq setpoint, not saved)")
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
        self._id    = param_id
        self._name  = name
        self._flags = flags
        readonly    = bool(flags & _FLAG_READONLY)

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(8)

        lbl = QLabel(name)
        lbl.setFixedWidth(190)
        lbl.setStyleSheet(f"color: {TEXT}; font-family: Consolas; font-size: 11px;")
        lay.addWidget(lbl)

        id_lbl = QLabel(f"0x{param_id:04X}")
        id_lbl.setFixedWidth(52)
        id_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(id_lbl)

        self._edit = QLineEdit(f"{value:.6g}")
        self._edit.setFixedWidth(96)
        self._edit.setStyleSheet(_EDIT_STYLE_NORMAL)
        self._edit.setEnabled(not readonly)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._edit.setValidator(validator)
        self._edit.returnPressed.connect(self._send)
        lay.addWidget(self._edit)

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

        range_lbl = QLabel(f"[{min_val:.4g} … {max_val:.4g}]")
        range_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        range_lbl.setMinimumWidth(130)
        lay.addWidget(range_lbl)

        flag_lbl = QLabel(_flag_text(flags))
        flag_lbl.setFixedWidth(40)
        flag_lbl.setToolTip(_flag_tooltip(flags))
        flag_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(flag_lbl)

        lay.addStretch()

    def update_value(self, value: float):
        self._edit.setText(f"{value:.6g}")
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_OK)
        self._flash_timer.start(700)

    def is_readonly(self) -> bool:
        return bool(self._flags & _FLAG_READONLY)

    def current_value(self) -> float:
        try:
            return float(self._edit.text())
        except ValueError:
            return 0.0

    def export_entry(self) -> dict:
        return {"name": self._name, "value": self.current_value()}

    def _send(self):
        try:
            val = float(self._edit.text())
        except ValueError:
            return
        send_param_set(self._id, val)
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_PENDING)
        self._flash_timer.start(2500)

    def _clear_flash(self):
        self._edit.setStyleSheet(_EDIT_STYLE_NORMAL)


# ── Group section header (top-level, collapsible) ─────────────────────────────

class _GroupHeader(QWidget):
    def __init__(self, group_id: int, on_toggle):
        super().__init__()
        name  = _GROUP_NAMES.get(group_id, f"Group 0x{group_id:02X}")
        color = _GROUP_COLORS.get(group_id, DIM)
        self._group     = group_id
        self._on_toggle = on_toggle

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 2)
        lay.setSpacing(4)

        self._btn = QPushButton("▶")   # starts collapsed
        self._btn.setFixedSize(18, 18)
        self._btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{color};font-size:10px;"
            f"border:none;padding:0}}"
            f"QPushButton:hover{{color:white}}"
        )
        self._btn.clicked.connect(lambda: self._on_toggle(self._group))
        lay.addWidget(self._btn)

        lbl = QLabel(name.upper())
        lbl.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        lay.addWidget(lbl)
        lay.addWidget(_hline(color))

    def set_collapsed(self, collapsed: bool):
        self._btn.setText("▶" if collapsed else "▼")


# ── Sub-group header (indented, collapsible) ──────────────────────────────────

class _SubGroupHeader(QWidget):
    def __init__(self, group_id: int, subgroup: str, on_toggle):
        super().__init__()
        color = _SUBGROUP_COLORS.get(subgroup, DIM)
        self._group     = group_id
        self._subgroup  = subgroup
        self._on_toggle = on_toggle

        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 3, 6, 1)
        lay.setSpacing(4)

        self._btn = QPushButton("▶")   # starts collapsed
        self._btn.setFixedSize(14, 14)
        self._btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{color};font-size:9px;"
            f"border:none;padding:0}}"
            f"QPushButton:hover{{color:white}}"
        )
        self._btn.clicked.connect(lambda: self._on_toggle(self._group, self._subgroup))
        lay.addWidget(self._btn)

        lbl = QLabel(subgroup)
        lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-style: italic;")
        lay.addWidget(lbl)
        lay.addWidget(_hline(color))

    def set_collapsed(self, collapsed: bool):
        self._btn.setText("▶" if collapsed else "▼")


# ── Main tab ──────────────────────────────────────────────────────────────────

class ParamsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._rows:       dict[int, _ParamRow]                   = {}
        self._headers:    dict[int, _GroupHeader]                = {}
        self._subheaders: dict[tuple[int, str], _SubGroupHeader] = {}
        self._collapsed_groups:    set[int]            = set()
        self._collapsed_subgroups: set[tuple[int, str]] = set()
        self._requested = False

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        grp_lbl = QLabel("Filter:")
        grp_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        toolbar.addWidget(grp_lbl)

        self._grp_combo = QComboBox()
        self._grp_combo.addItem("All", None)
        for gid, gname in _GROUP_NAMES.items():
            self._grp_combo.addItem(gname, gid)
        # Separator then sub-group quick-jump entries
        sep_idx = self._grp_combo.count()
        self._grp_combo.addItem("── sections ──", "separator")
        self._grp_combo.model().item(sep_idx).setEnabled(False)
        for _, gid, sgname in _SUBGROUPS:
            self._grp_combo.addItem(f"  {sgname}", (gid, sgname))

        self._grp_combo.setFixedWidth(160)
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

        toolbar.addSpacing(12)

        _neutral_btn_style = (
            f"QPushButton{{background:{SURFACE};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:3px;padding:3px 8px}}"
            f"QPushButton:hover{{background:{BORDER}}}"
        )

        btn_export = QPushButton("Export…")
        btn_export.setFixedWidth(72)
        btn_export.setStyleSheet(_neutral_btn_style)
        btn_export.clicked.connect(self._on_export)
        toolbar.addWidget(btn_export)

        btn_import = QPushButton("Import…")
        btn_import.setFixedWidth(72)
        btn_import.setStyleSheet(_neutral_btn_style)
        btn_import.clicked.connect(self._on_import)
        toolbar.addWidget(btn_import)

        toolbar.addSpacing(12)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setFixedWidth(120)
        btn_reset.setStyleSheet(
            f"QPushButton{{background:{SURFACE};color:{RED};"
            f"border:1px solid {RED};border-radius:3px;padding:3px 8px}}"
            f"QPushButton:hover{{background:{RED};color:{BG}}}"
        )
        btn_reset.clicked.connect(self._on_reset_defaults)
        toolbar.addWidget(btn_reset)

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
        self._inner_lay.addStretch()

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

    def _on_reset_defaults(self):
        reply = QMessageBox.question(
            self, "Reset All Parameters",
            "Reset ALL parameters to firmware defaults?\n\n"
            "This overwrites every editable value — including tuned gains and "
            "calibration-adjacent settings — and cannot be undone. Export first "
            "if you want to keep the current configuration.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        send_param_reset_defaults()
        self._lbl_status.setText("Reset to defaults requested…")
        self._lbl_status.setStyleSheet(f"color: {ORANGE}; font-size: 11px;")

    def _on_export(self):
        if not self._rows:
            QMessageBox.information(self, "Export Parameters",
                                     "No params loaded — connect and click Refresh first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Parameters", "params.json",
                                               "JSON Files (*.json)")
        if not path:
            return
        data = {f"0x{pid:04X}": row.export_entry() for pid, row in sorted(self._rows.items())}
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "Export Parameters", f"Failed to write file:\n{e}")
            return
        self._lbl_status.setText(f"Exported {len(data)} params to {path}")
        self._lbl_status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Parameters", "",
                                               "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Import Parameters", f"Failed to read file:\n{e}")
            return

        to_apply: list[tuple[int, float]] = []
        skipped = 0
        for key, entry in data.items():
            try:
                pid = int(key, 16) if isinstance(key, str) and key.lower().startswith("0x") else int(key)
                value = float(entry["value"]) if isinstance(entry, dict) else float(entry)
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            row = self._rows.get(pid)
            if row is None or row.is_readonly():
                skipped += 1
                continue
            to_apply.append((pid, value))

        if not to_apply:
            QMessageBox.information(self, "Import Parameters",
                                     "No applicable params found in file (unknown IDs or all read-only).")
            return

        msg = f"Apply {len(to_apply)} param(s) from:\n{path}"
        if skipped:
            msg += f"\n\n{skipped} entr{'y' if skipped == 1 else 'ies'} skipped (unknown ID or read-only)."
        reply = QMessageBox.question(
            self, "Import Parameters", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for pid, value in to_apply:
            send_param_set(pid, value)

        self._lbl_status.setText(f"Imported {len(to_apply)} params from {path}")
        self._lbl_status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

    def _on_packet(self, info: dict):
        ptype = info.get("ptype")

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
        flt = self._grp_combo.currentData()
        if flt == "separator":
            return
        # Auto-expand the selected section so filtered items are always visible
        if isinstance(flt, tuple):
            self._collapsed_subgroups.discard(flt)
            self._collapsed_groups.discard(flt[0])
        elif isinstance(flt, int):
            self._collapsed_groups.discard(flt)
        self._reapply_visibility()

    def _on_group_toggle(self, group_id: int):
        if group_id in self._collapsed_groups:
            self._collapsed_groups.discard(group_id)
        else:
            self._collapsed_groups.add(group_id)
        self._reapply_visibility()

    def _on_subgroup_toggle(self, group_id: int, subgroup: str):
        key = (group_id, subgroup)
        if key in self._collapsed_subgroups:
            self._collapsed_subgroups.discard(key)
        else:
            self._collapsed_subgroups.add(key)
        self._reapply_visibility()

    # ── private ───────────────────────────────────────────────────────────────

    def _add_row(self, param_id: int, name: str, value: float,
                 min_val: float, max_val: float, flags: int):
        group    = (param_id >> 8) & 0xFF
        subgroup = _get_subgroup(param_id)

        if group not in self._headers:
            hdr = _GroupHeader(group, self._on_group_toggle)
            self._inner_lay.insertWidget(self._inner_lay.count() - 1, hdr)
            self._headers[group] = hdr
            self._collapsed_groups.add(group)      # start collapsed

        if subgroup is not None:
            key = (group, subgroup)
            if key not in self._subheaders:
                subhdr = _SubGroupHeader(group, subgroup, self._on_subgroup_toggle)
                self._inner_lay.insertWidget(self._inner_lay.count() - 1, subhdr)
                self._subheaders[key] = subhdr
                self._collapsed_subgroups.add(key)  # start collapsed

        row = _ParamRow(param_id, name, value, min_val, max_val, flags)
        self._inner_lay.insertWidget(self._inner_lay.count() - 1, row)
        self._rows[param_id] = row

        self._reapply_visibility()

    def _reapply_visibility(self):
        flt = self._grp_combo.currentData()
        if flt == "separator":
            flt = None

        for param_id, row in self._rows.items():
            group    = (param_id >> 8) & 0xFF
            subgroup = _get_subgroup(param_id)

            if flt is None:
                filter_ok = True
            elif isinstance(flt, tuple):
                filter_ok = (group == flt[0] and subgroup == flt[1])
            else:
                filter_ok = (group == flt)

            group_collapsed = group in self._collapsed_groups
            sub_collapsed   = (subgroup is not None and
                               (group, subgroup) in self._collapsed_subgroups)

            row.setVisible(filter_ok and not group_collapsed and not sub_collapsed)

        for gid, hdr in self._headers.items():
            if flt is None:
                hdr.setVisible(True)
            elif isinstance(flt, tuple):
                hdr.setVisible(flt[0] == gid)
            else:
                hdr.setVisible(gid == flt)
            hdr.set_collapsed(gid in self._collapsed_groups)

        for (gid, sgname), subhdr in self._subheaders.items():
            if flt is None:
                filter_ok = True
            elif isinstance(flt, tuple):
                filter_ok = (flt == (gid, sgname))
            else:
                filter_ok = (gid == flt)

            group_collapsed = gid in self._collapsed_groups
            subhdr.setVisible(filter_ok and not group_collapsed)
            subhdr.set_collapsed((gid, sgname) in self._collapsed_subgroups)
