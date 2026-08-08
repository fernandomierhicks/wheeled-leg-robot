"""params_tab.py — Parameter Registry tab.

Shows every firmware parameter generated from the protocol schema as soon as
the tab is built, with Value/Range/Flags left blank until actually confirmed by
hardware — never a guessed or default value. On first telemetry received (or
when Refresh is clicked) sends CMD_ID_PARAM_GET 0xFFFF; each PARAM_REPORT
response (ptype 0x06) fills in that row's live Value/Range/Flags. Rows are
grouped by subsystem and split into collapsible sub-sections (all start
collapsed). Values are editable; Enter or the Set button sends CMD_ID_PARAM_SET
and the cell flashes green on echo-back.
"""

import json
import math
import struct
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .comm_commands import (
    send_param_get, send_param_get_all, send_param_reset_defaults, send_param_set, send_reliable,
)
from .generated_protocol import PARAM_BY_NAME as _PARAM_BY_NAME
from .generated_protocol import PARAM_DEFS as _PARAM_DEFS
from .telemetry_bus import TelemetryBus
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW

# ── Param flag bits (param_registry.h) ────────────────────────────────────────
_FLAG_PERSISTENT      = 1 << 0
_FLAG_READONLY        = 1 << 1
_FLAG_COMMAND         = 1 << 2

# Range column width — kept narrow (elided with a hover tooltip for the full
# text) so Description gets most of the row.
_RANGE_COL_WIDTH = 80

# Missing-param retry sweep (see ParamsTab._sweep_missing_params). A refresh
# fires a bulk PARAM_GET 0xFFFF (~131 individually-paced PARAM_REPORT
# frames); on a lossy link (WiFi especially) a reliable fraction never
# arrives with nothing re-requesting them. These tune how the follow-up
# per-param retry sweep paces itself.
_SWEEP_FIRST_DELAY_MS = 700   # let the paced bulk dump finish arriving first
_SWEEP_RETRY_DELAY_MS = 600
_SWEEP_MAX_ROUNDS     = 6
# Individual re-GETs are unpaced on firmware (unlike the bulk 0xFFFF dump's
# 4/tick cursor) -- each triggers an immediate synchronous PARAM_REPORT send.
# Capping how many go out per round keeps a single round well under the ~11
# reports (512 B / 45 B) firmware's own dump pacing comment says fits its
# Serial5 TX buffer without blocking the 500 Hz loop; leftovers just roll
# into the next round.
_SWEEP_MAX_PER_ROUND  = 5

# Direct angular quantities are presented in human-friendly degrees while the
# protocol, firmware registry, and parameter export format remain in radians.
# Gains whose units merely contain radians (for example N·m/rad) intentionally
# remain unchanged: they are coefficients, not angle/rate values.
_ANGLE_PARAM_NAMES = frozenset({
    "calib_backoff_rad",
    "calib_range_l_rad",
    "calib_range_r_rad",
    "calib_max_seek_rad",
    "calib_max_rel_rad",
    "sim_pitch_rad",
    "theta_max_fwd_ret",
    "theta_max_bwd_ret",
    "theta_max_fwd_ext",
    "theta_max_bwd_ext",
    "jump_ramp_down",
    "jump_hs_margin",
    "standup_pitch_fwd",
    "standup_pitch_bwd",
    "standup_cap_pitch",
    "lqr_pitch_trim_ret",
    "lqr_pitch_trim_ext",
    "pitch_wd_fwd_ret",
    "pitch_wd_bwd_ret",
    "pitch_wd_fwd_ext",
    "pitch_wd_bwd_ext",
})
_ANGULAR_RATE_PARAM_NAMES = frozenset({
    "calib_seek_speed",
    "calib_move_speed",
    "vel_pi_rate_lim",
    "omega_cmd_rds",
    "jump_omega_max",
    "sim_pitch_rate",
    "standup_cap_rate",
    "radio_yaw_max",
    "profile1_yaw_max",
    "profile2_yaw_max",
    "profile3_yaw_max",
})
_DISPLAY_UNIT_BY_PARAM = {
    **{_PARAM_BY_NAME[name]: "deg" for name in _ANGLE_PARAM_NAMES},
    **{_PARAM_BY_NAME[name]: "deg/s" for name in _ANGULAR_RATE_PARAM_NAMES},
}
_RAD_TO_DEG = 180.0 / math.pi

# ── Group map (param_ids.h GROUP_*) ───────────────────────────────────────────
_GROUP_NAMES = {
    0x00: "System",
    0x01: "Calibration",
    0x02: "Hip",
    0x03: "Wheel",
    0x04: "Control",
    0x05: "Command",
    0x06: "RC Receiver (iBus)",
    0x07: "Safety / Watchdog / Bypass",
    0x08: "Standing Up",
    0x09: "Diagnostics",
}
_GROUP_COLORS = {
    0x00: "#888888",
    0x01: BLUE,
    0x02: ORANGE,
    0x03: GREEN,
    0x04: "#cc88ff",
    0x05: "#ff88cc",
    0x06: "#88ddff",
    0x07: RED,
    0x08: "#ff6644",
    0x09: YELLOW,
}

# Watchdog/bypass params live natively in System/Wheel/Control/Calibration
# (param_ids.h) but are pulled into their own top-level "Safety / Watchdog /
# Bypass" group here since they're reviewed together regardless of which
# subsystem they guard.
_GROUP_SAFETY = 0x07
_WATCHDOG_PARAM_IDS = frozenset({
    0x0009,  # PARAM_WATCHDOG_ENABLE (hardware WDOG1)
    0x0300,  # PARAM_WM_ENC_TIMEOUT_MS (wheel encoder feedback watchdog)
    0x0301,  # PARAM_WM_VEL_SLEW_MAX (encoder-velocity plausibility filter; guards the runaway trip)
    0x0403,  # PARAM_WHEEL_VEL_LIMIT_TURNS_S (soft wheel governor; ESTOP at 2x)
    0x0423,  # PARAM_PITCH_WATCHDOG_ENABLE
    0x0521,  # PARAM_ROLL_WATCHDOG_EN
    0x0522,  # PARAM_ROLL_WATCHDOG_LIMIT (trips FAULT_ROLL_WATCHDOG)
    0x0131,  # PARAM_CALIB_BYPASS_EN (skip switch-calibration requirement for RUNNING)
    0x0429,  # PARAM_RUNNING_WHEEL_BYPASS_EN (skip wheel-enable requirement for RUNNING)
    0x042A,  # PARAM_ALPHA_FORCE_RETRACTED_EN (force gain-sched alpha=0, bypass calib-valid check)
})

# Standing-Up params live natively in GROUP_CONTROL (param_ids.h) but are
# pulled into their own top-level group here — a full recovery subsystem,
# reviewed/tuned as a unit at the same level as Calibration rather than
# buried as a Control subsection.
_GROUP_STANDUP = 0x08
_STANDUP_PARAM_IDS = frozenset(range(0x042C, 0x043C))

# gui_motion_ctrl_en lives natively in GROUP_CONTROL (param_ids.h) but is a
# System-level "who's driving" toggle, not a control-tuning value — pulled
# into System here.
_GROUP_SYSTEM = 0x00
_SYSTEM_OVERRIDE_PARAM_IDS = frozenset({
    0x042B,  # PARAM_GUI_MOTION_CTRL_EN
})

# The roll-controller params were allocated IDs out of the 0x05xx (COMMAND)
# block for lack of room elsewhere, but param_registry.cpp's g_params[]
# explicitly declares their real group as CONTROL/HIP (see
# generated_param_table.inc .group_id). _effective_group()'s default
# (id >> 8) heuristic gets these wrong, so override them explicitly here —
# same pattern as the other override sets in this file. The roll *watchdog*
# pair (0x0521/0x0522) is deliberately absent — it goes to Safety via
# _WATCHDOG_PARAM_IDS, and this set doubles as the "Roll" subgroup membership
# below, so leaving them here would route them to a subgroup of a group they
# no longer belong to.
_ROLL_CONTROL_OVERRIDE_PARAM_IDS = frozenset({
    0x051C,  # PARAM_ROLL_CTRL_EN
    0x051D,  # PARAM_ROLL_KP
    0x051E,  # PARAM_ROLL_KD
    0x051F,  # PARAM_ROLL_OFFSET_MAX
    0x0520,  # PARAM_ROLL_RATE_LIM
    0x052A,  # PARAM_ROLL_KI      (0x052x block — same override/subgroup treatment)
    0x052B,  # PARAM_ROLL_INT_MAX
})
_ROLL_HIP_OVERRIDE_PARAM_IDS = frozenset({
    0x0523,  # PARAM_HIP_ROLL_KP
    0x0524,  # PARAM_HIP_ROLL_KD
})

# Dev/debug params pulled out of their native group into their own top-level
# Diagnostics section.
_GROUP_DIAGNOSTICS = 0x09
_DIAGNOSTICS_PARAM_IDS = frozenset({
    0x000A,  # PARAM_LOOP_PROFILE_ENABLE
})


def _effective_group(param_id: int) -> int:
    if param_id in _WATCHDOG_PARAM_IDS:
        return _GROUP_SAFETY
    if param_id in _STANDUP_PARAM_IDS:
        return _GROUP_STANDUP
    if param_id in _DIAGNOSTICS_PARAM_IDS:
        return _GROUP_DIAGNOSTICS
    if param_id in _SYSTEM_OVERRIDE_PARAM_IDS:
        return _GROUP_SYSTEM
    if param_id in _ROLL_CONTROL_OVERRIDE_PARAM_IDS:
        return 0x04  # GROUP_CONTROL
    if param_id in _ROLL_HIP_OVERRIDE_PARAM_IDS:
        return 0x02  # GROUP_HIP
    return (param_id >> 8) & 0xFF

# ── Sub-group definitions ─────────────────────────────────────────────────────
# Each entry: (param_id_membership, parent_group_id, sub_group_label)
# Membership is usually a contiguous range, but Sim Injection's sim_pitch_rad
# (0x0401) is interleaved inside the LQR Core id block (param_ids.h), so both
# use explicit id sets instead of a range.
_SUBGROUPS: list[tuple[range | frozenset[int], int, str]] = [
    # Switch-based hip calibration.
    (frozenset({
        0x0120,  # calib_seek_speed
        0x0122,  # calib_seek_kp
        0x0124,  # calib_seek_trq_lim
        0x0129,  # calib_seek_timeout
        0x012B,  # calib_max_seek_rad
    }), 0x01, "Retracting"),
    (frozenset({
        0x0121,  # calib_move_speed
        0x0125,  # calib_move_trq_lim
        0x0126,  # calib_backoff_rad
        0x0127,  # calib_range_l_rad
        0x0128,  # calib_range_r_rad
        0x012A,  # calib_release_to
        0x012C,  # calib_max_rel_rad
        0x012E,  # calib_move_kp
    }), 0x01, "Extending"),
    (frozenset({
        0x0123,  # calib_kd
        0x012D,  # calib_trq_trip_ms
        0x012F,  # calib_rampdown_s
    }), 0x01, "Shared"),

    # Balance/control tuning.
    (frozenset({0x0400, 0x0402}), 0x04, "LQR Core"),
    (range(0x0424, 0x0429), 0x04, "LQR Gains"),
    (range(0x0404, 0x040C), 0x04, "Velocity PI"),
    (range(0x040C, 0x0412), 0x04, "Yaw PI"),
    (range(0x0412, 0x0415), 0x04, "Feedforward"),
    (range(0x0415, 0x0420), 0x04, "Jump"),
    (frozenset({0x0401, 0x0420, 0x0421, 0x0422}), 0x04, "Sim Injection"),
    (_ROLL_CONTROL_OVERRIDE_PARAM_IDS, 0x04, "Roll"),
    (range(0x043E, 0x0446), 0x04, "Pitch Envelope"),  # theta_max_*, pitch_wd_* (fwd/bwd x ret/ext)
    (range(0x0500, 0x0504), 0x05, "Radio Scale"),  # radio_hip_cmd, radio_vel_max, radio_yaw_max, live_tune_ch7_val
    (range(0x0510, 0x0513), 0x05, "Profile 1"),    # profile1_vel_max/yaw_max/torque_lim
    (range(0x0513, 0x0516), 0x05, "Profile 2"),    # profile2_vel_max/yaw_max/torque_lim
    (range(0x0516, 0x0519), 0x05, "Profile 3"),    # profile3_vel_max/yaw_max/torque_lim
    # active_profile (0x0519) intentionally left without a subgroup — it's the
    # CH9-selected index, not a per-profile value.
]

_SUBGROUP_COLORS: dict[str, str] = {
    "Retracting":     "#66aaff",
    "Extending":      "#ff9955",
    "Shared":         "#99aabb",
    "LQR Core":       "#dd99ff",
    "LQR Gains":      "#ee88ff",
    "Velocity PI":    "#bb77ee",
    "Yaw PI":         "#9966dd",
    "Feedforward":    "#ccaaff",
    "Jump":           "#ffaa44",
    "Sim Injection":  "#88ddcc",
    "Roll":           "#55ccff",
    "Pitch Envelope": "#ff99aa",
    "Safety":         "#ff5555",
    "Radio Scale":    "#ff88cc",
    "Profile 1":      "#ffdd88",
    "Profile 2":      "#ffcc66",
    "Profile 3":      "#ffaa33",
}


def _get_subgroup(param_id: int) -> str | None:
    for r, _, name in _SUBGROUPS:
        if param_id in r:
            return name
    return None


# Display order for subgroups within a parent group — declaration order in
# _SUBGROUPS, not param-report arrival order (see _ensure_subgroup).
_SUBGROUP_RANK: dict[tuple[int, str], int] = {
    (gid, name): idx for idx, (_, gid, name) in enumerate(_SUBGROUPS)
}

# ── Generated parameter definitions ───────────────────────────────────────────
# _PARAM_DEFS comes from generated_protocol.py, which is generated from
# firmware/robot_teensy/protocol/schema.json. This tab only owns presentation
# details such as effective groups and subgroups; it does not mirror parameter
# names or descriptions by hand.
def _hline(color: str = BORDER) -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {color};")
    f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return f


class _ElidedLabel(QLabel):
    """QLabel that elides long text with '…' to fit its current width, and
    always shows the untruncated text as a hover tooltip."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.set_full_text(text)

    def set_full_text(self, text: str):
        self._full_text = text
        self.setToolTip(text)
        self._reflow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self):
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, self.width())
        super().setText(elided)


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


def _display_description(description: str, unit: str | None) -> str:
    if unit is None:
        return description
    converted = description.replace("[rad/s]", "[deg/s]").replace("[rad]", "[deg]")
    firmware_unit = "rad/s" if unit == "deg/s" else "rad"
    return f"{converted} GUI displays {unit}; firmware stores {firmware_unit}."


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
    def __init__(self, param_id: int, name: str, description: str = "",
                 value: float | None = None, min_val: float | None = None,
                 max_val: float | None = None, flags: int | None = None):
        super().__init__()
        self._id        = param_id
        self._name      = name
        self._flags     = flags if flags is not None else 0
        self._confirmed = value is not None
        self._display_unit = _DISPLAY_UNIT_BY_PARAM.get(param_id)
        self._display_scale = _RAD_TO_DEG if self._display_unit else 1.0
        readonly        = bool(flags & _FLAG_READONLY) if flags is not None else False

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(8)

        lbl = QLabel(
            f"{name} [{self._display_unit}]" if self._display_unit else name
        )
        lbl.setFixedWidth(190)
        if self._display_unit:
            firmware_unit = "rad/s" if self._display_unit == "deg/s" else "rad"
            lbl.setToolTip(
                f"{name}: GUI uses {self._display_unit}; firmware uses {firmware_unit}."
            )
        lbl.setStyleSheet(f"color: {TEXT}; font-family: Consolas; font-size: 11px;")
        lay.addWidget(lbl)

        id_lbl = QLabel(f"0x{param_id:04X}")
        id_lbl.setFixedWidth(52)
        id_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(id_lbl)

        display_value = value * self._display_scale if value is not None else None
        self._edit = QLineEdit(
            f"{display_value:.6g}" if display_value is not None else ""
        )
        if display_value is None:
            # Placeholder, not real text: shows "?" (dimmed) while unconfirmed
            # without landing in .text(), so parsing/current_value()/_send()
            # never have to special-case it.
            self._edit.setPlaceholderText("?")
        self._edit.setFixedWidth(96)
        if self._display_unit:
            self._edit.setToolTip(
                f"Enter {self._display_unit}; converted to radians when sent."
            )
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

        range_txt = (
            self._format_range(min_val, max_val)
            if min_val is not None and max_val is not None else "…"
        )
        self._range_lbl = _ElidedLabel(range_txt)
        self._range_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        self._range_lbl.setFixedWidth(_RANGE_COL_WIDTH)
        lay.addWidget(self._range_lbl)

        self._flag_lbl = QLabel(_flag_text(flags) if flags is not None else "…")
        self._flag_lbl.setFixedWidth(40)
        self._flag_lbl.setToolTip(_flag_tooltip(flags) if flags is not None else "Not read yet")
        self._flag_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(self._flag_lbl)

        desc_lbl = _ElidedLabel(
            _display_description(description, self._display_unit)
        )
        desc_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        lay.addWidget(desc_lbl, stretch=1)

    def update_value(self, value: float, min_val: float, max_val: float, flags: int):
        self._flags     = flags
        self._confirmed = True
        readonly = bool(flags & _FLAG_READONLY)
        self._edit.setEnabled(not readonly)
        self._btn.setEnabled(not readonly)
        self._range_lbl.set_full_text(self._format_range(min_val, max_val))
        self._flag_lbl.setText(_flag_text(flags))
        self._flag_lbl.setToolTip(_flag_tooltip(flags))
        self._edit.setText(f"{value * self._display_scale:.6g}")
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_OK)
        self._flash_timer.start(700)

    def is_readonly(self) -> bool:
        return bool(self._flags & _FLAG_READONLY)

    def is_confirmed(self) -> bool:
        return self._confirmed

    def current_value(self) -> float:
        try:
            return float(self._edit.text()) / self._display_scale
        except ValueError:
            return 0.0

    def export_entry(self) -> dict | None:
        if not self._confirmed:
            return None
        return {"name": self._name, "value": self.current_value()}

    def _send(self):
        try:
            val = float(self._edit.text()) / self._display_scale
        except ValueError:
            return
        pid = self._id
        # Confirmed by a PARAM_REPORT echo for this id — any value, since the
        # firmware clamps out-of-range writes rather than rejecting them
        # (UARTplat.md Phase 5).
        send_reliable(
            lambda: send_param_set(pid, val),
            confirm_predicate=lambda info: info.get("ptype") == 0x06 and info.get("param_id") == pid,
            label=f"PARAM_SET {self._name}",
        )
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_PENDING)
        self._flash_timer.start(2500)

    def _format_range(self, min_val: float, max_val: float) -> str:
        lo = min_val * self._display_scale
        hi = max_val * self._display_scale
        unit = f" {self._display_unit}" if self._display_unit else ""
        return f"[{lo:.4g} … {hi:.4g}]{unit}"

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
        self._group_containers:    dict[int, QWidget]              = {}
        self._subgroup_containers: dict[tuple[int, str], QWidget]  = {}
        self._collapsed_groups:    set[int]            = set()
        self._collapsed_subgroups: set[tuple[int, str]] = set()
        self._requested = False
        # Live search query (lowercased). While non-empty it overrides collapse
        # state so a match is never hidden inside a collapsed section — see
        # _reapply_visibility.
        self._search_q = ""

        # Missing-param retry sweep (see _start_missing_sweep): a bulk
        # PARAM_GET 0xFFFF dump is ~131 individual PARAM_REPORT frames paced
        # out by firmware over tens of ms — on a lossy link (WiFi especially)
        # some fraction reliably never arrive, and nothing was re-requesting
        # them. This timer periodically re-asks (individually) for whichever
        # rows are still unconfirmed after a refresh, until all are confirmed
        # or a retry cap is hit.
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setSingleShot(True)
        self._sweep_timer.timeout.connect(self._sweep_missing_params)
        self._sweep_round = 0

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

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search name, description, or id…")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedWidth(240)
        self._search.setStyleSheet(
            f"QLineEdit {{ background: {SURFACE}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; border-radius: 3px; "
            f"padding: 2px 6px; font-size: 11px; }}"
            f"QLineEdit:focus {{ border: 1px solid {BLUE}; }}"
        )
        self._search.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self._search)

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

        self._lbl_status = QLabel("Loading…")
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
                       ("Range", _RANGE_COL_WIDTH), ("Flags", 40)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet(f"color: {DIM}; font-size: 10px; font-weight: bold;")
            col_lay.addWidget(lbl)
        desc_hdr = QLabel("Description")
        desc_hdr.setStyleSheet(f"color: {DIM}; font-size: 10px; font-weight: bold;")
        col_lay.addWidget(desc_hdr, stretch=1)

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

        self._populate_static_rows()

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _request_all(self):
        send_param_get_all()
        self._requested = True
        self._lbl_status.setText("Requesting…")
        self._lbl_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        self._start_missing_sweep()

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
        self._start_missing_sweep()

    def _start_missing_sweep(self):
        """(Re)start the retry sweep that chases down rows still unconfirmed
        after a bulk PARAM_REPORT dump (PARAM_GET 0xFFFF or a defaults reset,
        both of which trigger firmware's paced full dump). Individually
        re-requests only the still-missing params, a few hundred ms apart,
        until everything is confirmed or _SWEEP_MAX_ROUNDS is hit."""
        self._sweep_round = 0
        self._sweep_timer.start(_SWEEP_FIRST_DELAY_MS)

    def _sweep_missing_params(self):
        missing = [pid for pid, row in self._rows.items() if not row.is_confirmed()]
        if not missing:
            return
        self._sweep_round += 1
        for pid in missing[:_SWEEP_MAX_PER_ROUND]:
            send_param_get(pid)
        if self._sweep_round < _SWEEP_MAX_ROUNDS:
            self._sweep_timer.start(_SWEEP_RETRY_DELAY_MS)
        else:
            confirmed = len(self._rows) - len(missing)
            self._lbl_status.setText(
                f"{confirmed}/{len(self._rows)} params confirmed — {len(missing)} "
                f"never responded after {_SWEEP_MAX_ROUNDS} retries (link issue?)"
            )
            self._lbl_status.setStyleSheet(f"color: {ORANGE}; font-size: 11px;")

    def _on_export(self):
        data = {}
        for pid, row in sorted(self._rows.items()):
            entry = row.export_entry()
            if entry is not None:
                data[f"0x{pid:04X}"] = entry
        if not data:
            QMessageBox.information(self, "Export Parameters",
                                     "No confirmed params to export — connect and click Refresh first.")
            return
        default_path = str(Path(__file__).resolve().parent.parent / "parameter_exports" / "Default gains.json")
        path, _ = QFileDialog.getSaveFileName(self, "Export Parameters", default_path,
                                               "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "Export Parameters", f"Failed to write file:\n{e}")
            return
        self._lbl_status.setText(f"Exported {len(data)} params to {path}")
        self._lbl_status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

    def _on_import(self):
        default_path = str(Path(__file__).resolve().parent.parent / "parameter_exports" / "Default gains.json")
        path, _ = QFileDialog.getOpenFileName(self, "Import Parameters", default_path,
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
        if info.get("ptype") == 0x01 and not self.isVisible():
            return
        ptype = info.get("ptype")

        if ptype == 0x01 and not self._requested:
            self._request_all()
            return

        if ptype != 0x06:
            return

        param_id = info.get("param_id")
        value    = info.get("param_value")
        if param_id is None or value is None:
            return

        min_val = info.get("param_min", 0.0)
        max_val = info.get("param_max", 0.0)
        flags   = info.get("param_flags", 0)

        if param_id in self._rows:
            self._rows[param_id].update_value(value, min_val, max_val, flags)
        else:
            # Not in this GUI's generated schema (e.g. firmware is newer than
            # the GUI) — create it on the fly, already confirmed.
            self._ensure_row(
                param_id,
                info.get("param_name", f"0x{param_id:04X}"),
                "",
                value, min_val, max_val, flags,
            )
            self._reapply_visibility()

        self._update_status()

    def _update_status(self):
        total     = len(self._rows)
        confirmed = sum(1 for r in self._rows.values() if r.is_confirmed())
        if self._search_q:
            hits = sum(1 for pid in self._rows if self._search_match(pid))
            self._lbl_status.setText(
                f"{hits} of {total} params match “{self._search_q}”"
                if hits else f"no params match “{self._search_q}”")
            self._lbl_status.setStyleSheet(
                f"color: {TEXT if hits else ORANGE}; font-size: 11px;")
            return
        if confirmed == 0:
            self._lbl_status.setText(f"{total} params known — connect and click Refresh to read live values")
            self._lbl_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        else:
            self._lbl_status.setText(f"{confirmed}/{total} params confirmed")
            self._lbl_status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

    def _on_search_changed(self, text: str):
        self._search_q = text.strip().lower()
        self._reapply_visibility()
        self._update_status()

    def _search_match(self, param_id: int) -> bool:
        """True if this param matches the live query. Matches on name,
        description, the id in both hex and decimal, and the group/sub-group
        labels — so "extending" finds the calibration extend-phase params and
        "0x0125" or "125" finds one by id."""
        q = self._search_q
        if not q:
            return True
        name, description = _PARAM_DEFS.get(param_id, ("", ""))
        subgroup = _get_subgroup(param_id) or ""
        group = _GROUP_NAMES.get(_effective_group(param_id), "")
        haystack = (f"{name}\n{description}\n{subgroup}\n{group}\n"
                    f"0x{param_id:04x}\n{param_id}").lower()
        # Every whitespace-separated term must appear (AND), so "calib trq"
        # narrows rather than widening.
        return all(term in haystack for term in q.split())

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

    def _ensure_group(self, group: int) -> QVBoxLayout:
        """Lazily create a group's own container (header + its rows/subgroups)
        and slot it into _inner_lay in ascending group-id order. Params for a
        given group don't necessarily arrive contiguously (e.g. Safety pulls
        ids from System/Wheel/Control), so each group needs a stable home to
        insert into rather than always appending at the current tail."""
        if group in self._group_containers:
            return self._group_containers[group].layout()

        container = QWidget()
        clay = QVBoxLayout(container)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(0)
        hdr = _GroupHeader(group, self._on_group_toggle)
        clay.addWidget(hdr)
        self._headers[group] = hdr
        self._collapsed_groups.add(group)  # start collapsed

        insert_before_widget = None
        insert_before_gid = None
        for gid, w in self._group_containers.items():
            if gid > group and (insert_before_gid is None or gid < insert_before_gid):
                insert_before_gid, insert_before_widget = gid, w
        pos = (self._inner_lay.indexOf(insert_before_widget) if insert_before_widget
               else self._inner_lay.count() - 1)  # else: just before the trailing stretch
        self._inner_lay.insertWidget(pos, container)
        self._group_containers[group] = container
        return clay

    def _ensure_subgroup(self, group: int, subgroup: str) -> QVBoxLayout:
        """Same idea as _ensure_group, one level down: slot the subgroup's
        own container into its parent group's layout in _SUBGROUPS order."""
        group_lay = self._ensure_group(group)
        key = (group, subgroup)
        if key in self._subgroup_containers:
            return self._subgroup_containers[key].layout()

        container = QWidget()
        clay = QVBoxLayout(container)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(0)
        subhdr = _SubGroupHeader(group, subgroup, self._on_subgroup_toggle)
        clay.addWidget(subhdr)
        self._subheaders[key] = subhdr
        self._collapsed_subgroups.add(key)  # start collapsed

        rank = _SUBGROUP_RANK.get(key, 0)
        insert_before_widget = None
        insert_before_rank = None
        for (gid2, name2), w in self._subgroup_containers.items():
            if gid2 != group:
                continue
            r2 = _SUBGROUP_RANK.get((gid2, name2), 0)
            if r2 > rank and (insert_before_rank is None or r2 < insert_before_rank):
                insert_before_rank, insert_before_widget = r2, w
        pos = group_lay.indexOf(insert_before_widget) if insert_before_widget else group_lay.count()
        group_lay.insertWidget(pos, container)
        self._subgroup_containers[key] = container
        return clay

    def _ensure_row(self, param_id: int, name: str, description: str = "",
                     value: float | None = None, min_val: float | None = None,
                     max_val: float | None = None, flags: int | None = None) -> _ParamRow:
        if param_id in self._rows:
            return self._rows[param_id]

        group    = _effective_group(param_id)
        subgroup = _get_subgroup(param_id)

        target_lay = (self._ensure_subgroup(group, subgroup) if subgroup is not None
                      else self._ensure_group(group))

        row = _ParamRow(param_id, name, description, value, min_val, max_val, flags)
        target_lay.addWidget(row)
        self._rows[param_id] = row
        return row

    def _populate_static_rows(self):
        for param_id in sorted(_PARAM_DEFS):
            name, description = _PARAM_DEFS[param_id]
            self._ensure_row(param_id, name, description)
        self._reapply_visibility()
        self._update_status()

    def _reapply_visibility(self):
        flt = self._grp_combo.currentData()
        if flt == "separator":
            flt = None

        searching = bool(self._search_q)
        # Which groups / sub-groups still have a visible row, so empty headers
        # can be hidden while searching instead of leaving bare section titles.
        live_groups: set[int] = set()
        live_subgroups: set[tuple[int, str]] = set()

        for param_id, row in self._rows.items():
            group    = _effective_group(param_id)
            subgroup = _get_subgroup(param_id)

            if flt is None:
                filter_ok = True
            elif isinstance(flt, tuple):
                filter_ok = (group == flt[0] and subgroup == flt[1])
            else:
                filter_ok = (group == flt)

            filter_ok = filter_ok and self._search_match(param_id)

            if searching:
                # A search deliberately ignores collapse state: a hit must never
                # be hidden inside a collapsed section the user can't see.
                visible = filter_ok
            else:
                group_collapsed = group in self._collapsed_groups
                sub_collapsed   = (subgroup is not None and
                                   (group, subgroup) in self._collapsed_subgroups)
                visible = filter_ok and not group_collapsed and not sub_collapsed

            row.setVisible(visible)
            if filter_ok:
                live_groups.add(group)
                if subgroup is not None:
                    live_subgroups.add((group, subgroup))

        for gid, hdr in self._headers.items():
            if flt is None:
                visible = True
            elif isinstance(flt, tuple):
                visible = (flt[0] == gid)
            else:
                visible = (gid == flt)
            if searching:
                visible = visible and gid in live_groups
            hdr.setVisible(visible)
            # While searching every section reads as expanded, because that is
            # what the rows below it are actually doing.
            hdr.set_collapsed(not searching and gid in self._collapsed_groups)

        for (gid, sgname), subhdr in self._subheaders.items():
            if flt is None:
                filter_ok = True
            elif isinstance(flt, tuple):
                filter_ok = (flt == (gid, sgname))
            else:
                filter_ok = (gid == flt)

            if searching:
                visible = filter_ok and (gid, sgname) in live_subgroups
            else:
                visible = filter_ok and gid not in self._collapsed_groups
            subhdr.setVisible(visible)
            subhdr.set_collapsed(not searching and
                                 (gid, sgname) in self._collapsed_subgroups)
