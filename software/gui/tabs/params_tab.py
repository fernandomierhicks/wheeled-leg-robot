"""params_tab.py — Parameter Registry tab.

Shows every known firmware parameter (from the static _PARAM_DEFS mirror of
ParamRegistry) as soon as the tab is built, with Value/Range/Flags left blank
until actually confirmed by hardware — never a guessed or default value. On
first telemetry received (or when Refresh is clicked) sends CMD_ID_PARAM_GET
0xFFFF; each PARAM_REPORT response (ptype 0x06) fills in that row's live
Value/Range/Flags. Rows are grouped by subsystem and split into collapsible
sub-sections (all start collapsed). Values are editable; Enter or the Set
button sends CMD_ID_PARAM_SET and the cell flashes green on echo-back.
"""

import json
import struct

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .comm_commands import send_param_get_all, send_param_reset_defaults, send_param_set, send_reliable
from .telemetry_bus import TelemetryBus
from .theme import BG, BLUE, BORDER, DIM, GREEN, ORANGE, RED, SURFACE, TEXT

# ── Param flag bits (param_registry.h) ────────────────────────────────────────
_FLAG_PERSISTENT      = 1 << 0
_FLAG_READONLY        = 1 << 1
_FLAG_COMMAND         = 1 << 2

# Range column width — kept narrow (elided with a hover tooltip for the full
# text) so Description gets most of the row.
_RANGE_COL_WIDTH = 80

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
}

# Watchdog/bypass params live natively in System/Wheel/Control/Calibration
# (param_ids.h) but are pulled into their own top-level "Safety / Watchdog /
# Bypass" group here since they're reviewed together regardless of which
# subsystem they guard.
_GROUP_SAFETY = 0x07
_WATCHDOG_PARAM_IDS = frozenset({
    0x0009,  # PARAM_WATCHDOG_ENABLE (hardware WDOG1)
    0x0300,  # PARAM_WM_ENC_TIMEOUT_MS (wheel encoder feedback watchdog)
    0x0423,  # PARAM_PITCH_WATCHDOG_ENABLE
    0x0113,  # PARAM_CALIB_BYPASS_EN (skip hardstop-calibration requirement for RUNNING)
    0x0429,  # PARAM_RUNNING_WHEEL_BYPASS_EN (skip wheel-enable requirement for RUNNING)
    0x042A,  # PARAM_ALPHA_FORCE_RETRACTED_EN (force gain-sched alpha=0, bypass calib-valid check)
})


def _effective_group(param_id: int) -> int:
    if param_id in _WATCHDOG_PARAM_IDS:
        return _GROUP_SAFETY
    return (param_id >> 8) & 0xFF

# ── Sub-group definitions (Control group split into logical sections) ──────────
# Each entry: (param_id_membership, parent_group_id, sub_group_label)
# Membership is usually a contiguous range, but Sim Injection's sim_pitch_rad
# (0x0401) is interleaved inside the LQR Core id block (param_ids.h), so both
# use explicit id sets instead of a range.
_SUBGROUPS: list[tuple[range | frozenset[int], int, str]] = [
    (frozenset({0x010A}), 0x01, "Left"),   # calib_l_seek_dir
    (frozenset({0x010B}), 0x01, "Right"),  # calib_r_seek_dir
    (frozenset({0x0400, 0x0402, 0x0403}), 0x04, "LQR Core"),
    (range(0x0424, 0x0429), 0x04, "LQR Gains"),
    (range(0x0404, 0x040C), 0x04, "Velocity PI"),
    (range(0x040C, 0x0412), 0x04, "Yaw PI"),
    (range(0x0412, 0x0415), 0x04, "Feedforward"),
    (range(0x0415, 0x0420), 0x04, "Jump"),
    (frozenset({0x0401, 0x0420, 0x0421, 0x0422}), 0x04, "Sim Injection"),
    (range(0x0500, 0x0504), 0x05, "Radio Scale"),  # radio_hip_cmd, radio_vel_max, radio_yaw_max, radio_pitch_trim
    (range(0x0510, 0x0513), 0x05, "Profile 1"),    # profile1_vel_max/yaw_max/torque_lim
    (range(0x0513, 0x0516), 0x05, "Profile 2"),    # profile2_vel_max/yaw_max/torque_lim
    (range(0x0516, 0x0519), 0x05, "Profile 3"),    # profile3_vel_max/yaw_max/torque_lim
    # active_profile (0x0519) intentionally left without a subgroup — it's the
    # CH9-selected index, not a per-profile value.
]

_SUBGROUP_COLORS: dict[str, str] = {
    "Left":           "#66aaff",
    "Right":          "#3377cc",
    "LQR Core":       "#dd99ff",
    "LQR Gains":      "#ee88ff",
    "Velocity PI":    "#bb77ee",
    "Yaw PI":         "#9966dd",
    "Feedforward":    "#ccaaff",
    "Jump":           "#ffaa44",
    "Sim Injection":  "#88ddcc",
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

# ── Static parameter definitions ──────────────────────────────────────────────
# Hand-copied from param_ids.h (id + description) and param_registry.cpp (name
# string). Lets the tab show the full known parameter list — with names and
# descriptions — before any device is connected; live Value/Range/Flags are
# filled in from PARAM_REPORT once read.
# IMPORTANT: keep in sync — when a param is added/changed/removed in either of
# those files, update the matching entry here too.
_PARAM_DEFS: dict[int, tuple[str, str]] = {
    # GROUP_SYSTEM
    0x0000: ("imu_enable", "BNO086 IMU present (SPI). 0 = skip init/poll and stop gating "
             "STARTUP/CALIBRATION/RUNNING on it; also blocks RUNNING since real pitch "
             "feedback is required. Persisted; takes effect at boot (needs reboot)."),
    0x0003: ("buzzer_enable", "Buzzer present. 0 = skip init/poll. Persisted; takes effect at boot."),
    0x0004: ("led_enable", "Status RGB LED present. 0 = skip init/poll. Persisted; takes effect at boot."),
    0x0005: ("hip_l_enable", "Left AK45 hip motor present. 0 = skip CAN traffic and stop gating "
             "STARTUP/CALIBRATION on it. RUNNING/JUMPING blocked unless all 4 motor-enable "
             "flags are set. Persisted; takes effect at boot."),
    0x0006: ("hip_r_enable", "Right AK45 hip motor present. 0 = skip CAN traffic and stop gating "
             "STARTUP/CALIBRATION on it. RUNNING/JUMPING blocked unless all 4 motor-enable "
             "flags are set. Persisted; takes effect at boot."),
    0x0007: ("wheel_l_enable", "Left ODrive wheel motor present. 0 = skip CAN traffic and stop "
             "gating STARTUP/CALIBRATION on it. RUNNING/JUMPING blocked unless all 4 "
             "motor-enable flags are set. Persisted; takes effect at boot."),
    0x0008: ("wheel_r_enable", "Right ODrive wheel motor present. 0 = skip CAN traffic and stop "
             "gating STARTUP/CALIBRATION on it. RUNNING/JUMPING blocked unless all 4 "
             "motor-enable flags are set. Persisted; takes effect at boot."),
    0x0009: ("watchdog_enable", "Enable hardware watchdog (WDOG1) — auto-reboots the MCU if the "
             "main loop stalls. Default off. Takes effect at boot; once armed cannot be "
             "disabled in software until the next reset."),
    0x000A: ("loop_profile_enable", "Debug: log a rolling max of loop()-section timings once/sec. "
             "Not persisted."),

    # GROUP_HIP
    0x0200: ("estop_hip_disable", "1 = exit MIT mode on ESTOP entry and re-enter on reset; "
             "0 = leave MIT running through ESTOP."),
    0x0201: ("hip_running_kp", "MIT position gain used for hip setpoints while RUNNING."),
    0x0202: ("hip_running_kd", "MIT damping gain used for hip setpoints while RUNNING."),
    0x0203: ("hip_running_tff", "MIT feedforward torque used for hip setpoints while RUNNING."),
    0x0204: ("hip_running_ramp_s", "Seconds to ramp kp/tff from 0 to their running value after "
             "entering RUNNING, easing in instead of snapping. kd applies at full value "
             "throughout. 0 = no ramp."),

    # GROUP_CALIB
    0x0100: ("calib_seek_speed", "Ramp speed toward hardstop while seeking [rad/s]."),
    0x0101: ("calib_kp_bottom", "Position gain while seeking SEEK_BOTTOM (retract, "
             "weight-assisted); also reused by CAL_RETURN_HOME's traverse-back move."),
    0x0102: ("calib_kd", "Damping while seeking, both phases."),
    0x0103: ("calib_hold_kp", "Position gain while holding at home."),
    0x0104: ("calib_hold_kd", "Damping while holding at home."),
    0x0105: ("calib_stall_cur_btm", "Current threshold to declare a hardstop while seeking "
             "SEEK_BOTTOM (retract) [A]."),
    0x0106: ("calib_stall_db", "Max position movement per tick to still count as stalled [rad]."),
    0x0107: ("calib_stall_ticks", "Consecutive stalled ticks before declaring a hardstop."),
    0x0108: ("calib_margin", "Safety margin kept from each hardstop [rad]."),
    0x0109: ("calib_safety_bound", "Max CAL_SEEK_BOTTOM travel before fault [rad] — worst case an "
             "unknown start position needs the full joint range to reach the first hardstop. "
             "Geometry-derived; READONLY, edit + reflash to change."),
    0x010A: ("calib_l_seek_dir", "Sign (+1/-1) toward left hip bottom hardstop. Wiring-direction "
             "constant; READONLY, edit + reflash to change."),
    0x010B: ("calib_r_seek_dir", "Sign (+1/-1) toward right hip bottom hardstop. Flipped from "
             "-1.0 — confirmed backwards on the hardware bench. Wiring-direction constant; "
             "READONLY, edit + reflash to change."),
    0x010C: ("calib_done", "1.0 = calibration has completed at least once (persisted across reboots)."),
    0x010D: ("calib_bound_top", "Max CAL_SEEK_TOP travel before fault [rad], measured from the "
             "just-zeroed bottom hardstop. Geometry-derived; READONLY, edit + reflash to change."),
    0x010E: ("calib_kp_top", "Position gain while seeking SEEK_TOP (extend, fights robot weight)."),
    0x010F: ("calib_stall_cur_top", "Current threshold to declare a hardstop while seeking "
             "SEEK_TOP (extend) [A]; higher than the retract threshold since fighting gravity "
             "raises baseline current."),
    0x0110: ("calib_retract_en", "1 = run SEEK_BOTTOM (retract), the prerequisite phase that "
             "establishes the zero reference. 0 skips the whole axis, like a disabled hip "
             "motor. READONLY, edit + reflash to change."),
    0x0111: ("calib_extend_en", "1 = run SEEK_TOP (extend) after retract. 0 holds at "
             "CAL_HOLD_RETRACT instead — no limits computed, calibration_done() never fires "
             "for that axis. READONLY, edit + reflash to change."),
    0x0112: ("calib_rampdown_s", "Seconds to ramp kp/kd from hold values down to zero before "
             "calibration_done() fires, so exiting CALIBRATION never yanks torque to zero in "
             "one tick. 0 = no ramp."),
    0x0113: ("calib_bypass_en", "1 = allow RUNNING mode without a completed hardstop calibration "
             "(hip position limits unenforced — bench-test only). Persisted; default 0."),

    # GROUP_WHEEL
    0x0300: ("wm_enc_timeout_ms", "Wheel encoder feedback watchdog timeout [ms]; increase if CAN "
             "is flaky. ODrive PID gains aren't exposed here — tune via the ODrive USB GUI "
             "(CAN can't reach those properties on 0.5.x)."),

    # GROUP_CONTROL — LQR core / limits
    0x0400: ("lqr_enable", "1 = wheel torque output active; 0 = LQR runs but outputs zero."),
    0x0401: ("sim_pitch_rad", "Pitch value injected in place of real IMU pitch when "
             "enable_sim_pitch=1."),
    0x0402: ("lqr_torque_limit", "|tau_sym| clamp [N·m]; hard max 7.0. READONLY — slewed "
             "automatically from the active CH9 speed profile's torque_lim. Edit "
             "profile1/2/3_torque_lim instead."),
    0x0403: ("wm_vel_limit", "Per-tick soft wheel velocity governor [turns/s]; ESTOP at 2x this."),
    # Velocity PI
    0x0404: ("vel_pi_en", "1 = velocity PI active; 0 = theta_ref fixed at 0."),
    0x0405: ("vel_pi_kp", "Velocity PI proportional gain [rad/(m/s)]."),
    0x0406: ("vel_pi_ki", "Velocity PI integral gain [rad/m]."),
    0x0407: ("vel_pi_kff", "Velocity PI acceleration feedforward gain [s²·rad/m] (≈ 1/g)."),
    0x0408: ("vel_pi_theta_max", "|theta_ref| hard clamp [rad]."),
    0x0409: ("vel_pi_rate_lim", "theta_ref slew rate limit [rad/s]."),
    0x040A: ("vel_pi_int_max", "Velocity PI integrator anti-windup clamp [rad·s]."),
    0x040B: ("v_cmd_ms", "Desired forward velocity setpoint [m/s]; GUI/Phase3 testing."),
    # Yaw PI
    0x040C: ("yaw_pi_en", "1 = yaw PI active; 0 = tau_yaw fixed at 0."),
    0x040D: ("yaw_pi_kp", "Yaw PI proportional gain [N·m/(rad/s)]."),
    0x040E: ("yaw_pi_ki", "Yaw PI integral gain [N·m/rad]."),
    0x040F: ("yaw_pi_torque_max", "|tau_yaw| clamp [N·m]."),
    0x0410: ("yaw_pi_int_max", "Yaw PI integrator anti-windup [N·m·s]."),
    0x0411: ("omega_cmd_rds", "Desired yaw rate [rad/s]; positive = CCW from above."),
    # Feedforward
    0x0412: ("ff1_alpha", "Hip reaction cancel feedforward gain [0–1]; start at 0, ramp up."),
    0x0413: ("ff2_alpha", "Gravity compensation feedforward gain [0–1]; start at 0, ramp up."),
    0x0414: ("ff1_kt_hip", "AK45-10 hip motor output torque constant [N·m/A]; hardware "
             "characteristic, not a tuning knob. READONLY, edit + reflash to change."),
    # Jump
    0x0415: ("jump_enable", "Master gate for jump sequence: 0 = no-op, 1 = execute."),
    0x0416: ("jump_torque_max", "Max hip feedforward torque during EXTEND [N·m]; ramp up from 0."),
    0x0417: ("jump_crouch_time", "CROUCH phase duration [s]."),
    0x0418: ("jump_ramp_up", "EXTEND torque softstart duration [s]."),
    0x0419: ("jump_ramp_down", "Torque→zero zone near the extended limit [rad]."),
    0x041A: ("jump_omega_max", "Hip velocity above which feedforward torque → 0 [rad/s]."),
    0x041B: ("jump_hs_margin", "Hard cutoff margin from the calibrated hardstop [rad]."),
    0x041C: ("jump_kp", "Position gain for CROUCH/RETRACT phases."),
    0x041D: ("jump_kd", "Damping for CROUCH/RETRACT phases."),
    0x041E: ("jump_ext_kd", "Small damping during EXTEND (electrical damping only)."),
    0x041F: ("jump_ext_timeout", "Max time in EXTEND before forced RETRACT [s]."),
    # Sim injection
    0x0420: ("enable_sim_pitch", "1 = use sim_pitch_rad instead of real IMU pitch."),
    0x0421: ("sim_pitch_rate", "Pitch rate injected in place of real IMU pitch rate when "
             "enable_sim_prate=1 [rad/s]."),
    0x0422: ("enable_sim_prate", "1 = use sim_pitch_rate instead of real IMU pitch rate."),
    0x0423: ("pitch_watchdog_en", "1 = ESTOP (FAULT_PITCH_WATCHDOG) if |pitch| > 50° for "
             "> 200 ms. Default on and NOT persisted — always starts back on after reboot so "
             "a bench-test disable can't silently survive into a real run."),
    # LQR gain table
    0x0424: ("lqr_k_pitch_ret", "LQR pitch gain, fully retracted leg. Default -13.0495742."),
    0x0425: ("lqr_k_rate_ret", "LQR pitch-rate gain, fully retracted leg. Default -2.18083692."),
    0x0426: ("lqr_k_pitch_ext", "LQR pitch gain, fully extended leg. Default -7.92908352."),
    0x0427: ("lqr_k_rate_ext", "LQR pitch-rate gain, fully extended leg. Default -1.69084204."),
    0x0428: ("lqr_k_vel", "LQR velocity-error gain, invariant across leg height. Default -7.13051190e-03."),
    0x0429: ("run_wheel_bypass_en", "1 = allow RUNNING mode to arm with wheel_l/r_enable off "
             "(bench-test only, e.g. a software-triggered smoke test with hips also disabled, "
             "no real torque anywhere). Independent of calib_bypass_en, which only covers the "
             "hip check. Not persisted — always boots to 0 (bypass off)."),
    0x042A: ("alpha_force_ret_en", "1 = force the hip gain-schedule blend to alpha=0.0 "
             "(fully-retracted LQR gains), bypassing the hm_limits valid-calibration check "
             "entirely. For hardware tuning with hips zip-tied retracted and disabled, where "
             "real calibration can never complete. Not persisted — always boots to 0."),
    0x042B: ("gui_motion_ctrl_en", "1 = v_cmd_ms/omega_cmd_rds are driven by GUI/CLI param_set "
             "instead of CH2/CH4; radio arming (CH10) is unaffected. Auto-clears (reverting to "
             "radio control) if no GUI command arrives for 300 ms. Not persisted — always boots to 0."),

    # GROUP_COMMAND
    0x0500: ("radio_hip_cmd", "Hip extension command from CH3 [0=retracted, 1=extended]; "
             "firmware-written, stale when the radio link is dead."),
    0x0501: ("radio_vel_max", "Max forward speed mapped from full CH2 deflection [m/s]. READONLY "
             "— copied from the active CH9 profile's vel_max. Edit profile1/2/3_vel_max instead."),
    0x0502: ("radio_yaw_max", "Max yaw rate mapped from full CH4 deflection [rad/s]. READONLY "
             "— copied from the active CH9 profile's yaw_max. Edit profile1/2/3_yaw_max instead."),
    0x0503: ("radio_pitch_trim", "Pitch equilibrium trim from CH7 [rad]; hook only, not yet "
             "applied to LQR."),
    0x0510: ("profile1_vel_max", "Speed profile 1 (slow) max forward speed [m/s]; CH9 "
             "3-position switch selects the active profile."),
    0x0511: ("profile1_yaw_max", "Speed profile 1 (slow) max yaw rate [rad/s]."),
    0x0512: ("profile1_torque_lim", "Speed profile 1 (slow) LQR torque limit [N·m]."),
    0x0513: ("profile2_vel_max", "Speed profile 2 (normal) max forward speed [m/s]."),
    0x0514: ("profile2_yaw_max", "Speed profile 2 (normal) max yaw rate [rad/s]."),
    0x0515: ("profile2_torque_lim", "Speed profile 2 (normal) LQR torque limit [N·m]."),
    0x0516: ("profile3_vel_max", "Speed profile 3 (fast) max forward speed [m/s]."),
    0x0517: ("profile3_yaw_max", "Speed profile 3 (fast) max yaw rate [rad/s]."),
    0x0518: ("profile3_torque_lim", "Speed profile 3 (fast) LQR torque limit [N·m]."),
    0x0519: ("active_profile", "Active speed profile index (0/1/2), selected by CH9; "
             "firmware-written, READONLY."),

    # GROUP_IBUS
    **{0x0600 + i: (f"ibus_ch{i}", f"RC receiver channel {i} raw pulse width [1000–2000 µs]; "
                     "firmware-written, READONLY.")
       for i in range(14)},
    0x060E: ("ibus_alive", "RC link alive: 1.0 = packet received within 500 ms, 0.0 = link lost; "
             "firmware-written, READONLY."),
}


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
        readonly        = bool(flags & _FLAG_READONLY) if flags is not None else False

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

        self._edit = QLineEdit(f"{value:.6g}" if value is not None else "")
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

        range_txt = (f"[{min_val:.4g} … {max_val:.4g}]"
                     if min_val is not None and max_val is not None else "…")
        self._range_lbl = _ElidedLabel(range_txt)
        self._range_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        self._range_lbl.setFixedWidth(_RANGE_COL_WIDTH)
        lay.addWidget(self._range_lbl)

        self._flag_lbl = QLabel(_flag_text(flags) if flags is not None else "…")
        self._flag_lbl.setFixedWidth(40)
        self._flag_lbl.setToolTip(_flag_tooltip(flags) if flags is not None else "Not read yet")
        self._flag_lbl.setStyleSheet(f"color: {DIM}; font-family: Consolas; font-size: 10px;")
        lay.addWidget(self._flag_lbl)

        desc_lbl = _ElidedLabel(description)
        desc_lbl.setStyleSheet(f"color: {DIM}; font-size: 10px;")
        lay.addWidget(desc_lbl, stretch=1)

    def update_value(self, value: float, min_val: float, max_val: float, flags: int):
        self._flags     = flags
        self._confirmed = True
        readonly = bool(flags & _FLAG_READONLY)
        self._edit.setEnabled(not readonly)
        self._btn.setEnabled(not readonly)
        self._range_lbl.set_full_text(f"[{min_val:.4g} … {max_val:.4g}]")
        self._flag_lbl.setText(_flag_text(flags))
        self._flag_lbl.setToolTip(_flag_tooltip(flags))
        self._edit.setText(f"{value:.6g}")
        self._flash_timer.stop()
        self._edit.setStyleSheet(_EDIT_STYLE_OK)
        self._flash_timer.start(700)

    def is_readonly(self) -> bool:
        return bool(self._flags & _FLAG_READONLY)

    def is_confirmed(self) -> bool:
        return self._confirmed

    def current_value(self) -> float:
        try:
            return float(self._edit.text())
        except ValueError:
            return 0.0

    def export_entry(self) -> dict | None:
        if not self._confirmed:
            return None
        return {"name": self._name, "value": self.current_value()}

    def _send(self):
        try:
            val = float(self._edit.text())
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
        data = {}
        for pid, row in sorted(self._rows.items()):
            entry = row.export_entry()
            if entry is not None:
                data[f"0x{pid:04X}"] = entry
        if not data:
            QMessageBox.information(self, "Export Parameters",
                                     "No confirmed params to export — connect and click Refresh first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Parameters", "params.json",
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
            # Not in the static mirror (e.g. GUI not yet updated after a
            # firmware change) — create it on the fly, already confirmed.
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
        if confirmed == 0:
            self._lbl_status.setText(f"{total} params known — connect and click Refresh to read live values")
            self._lbl_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        else:
            self._lbl_status.setText(f"{confirmed}/{total} params confirmed")
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

        for param_id, row in self._rows.items():
            group    = _effective_group(param_id)
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
