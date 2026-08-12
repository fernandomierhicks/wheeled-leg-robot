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
    # Hip extension angles measured from the retract switch — the CROUCH,
    # EXTEND and RETRACT targets. Degrees is the whole point of specifying them
    # this way. jump_retract_angle's negative sentinel ("return to the pre-jump
    # pose") converts to degrees harmlessly: -1 rad reads as -57.3 deg, still
    # obviously not an angle you asked for.
    "jump_crouch_angle",
    "jump_extend_angle",
    "jump_retract_angle",
    "standup_pitch_min",
    "standup_pitch_max",
    "standup_cap_pitch",
    "lqr_pitch_trim_ret",
    "lqr_pitch_trim_ext",
    "lqr_trim_curve",
    "pitch_wd_fwd_ret",
    "pitch_wd_bwd_ret",
    "pitch_wd_fwd_ext",
    "pitch_wd_bwd_ext",
    # A hip position offset, not a body angle — but it is still a direct angular
    # quantity in rad, so degrees read better than the 0.15 rad it defaults to.
    "roll_offset_max",
    # Roll setpoint limits. radio_roll_max is firmware-written from the active
    # CH9 profile, so the three profile values are where these actually get set;
    # all four display together or the comparison is unreadable.
    "radio_roll_max",
    "profile1_roll_max",
    "profile2_roll_max",
    "profile3_roll_max",
})
_ANGULAR_RATE_PARAM_NAMES = frozenset({
    "calib_seek_speed",
    "calib_move_speed",
    "vel_pi_rate_lim",
    "omega_cmd_rds",
    "jump_omega_max",
    # Peak hip speeds for the CROUCH and RETRACT ramps.
    "jump_crouch_speed",
    "jump_retract_speed",
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


def _migrate_import_value(param_id: int, entry, value: float) -> float:
    """Translate values from parameter exports made before a semantic rename."""
    if (
        param_id == _PARAM_BY_NAME["standup_pitch_min"]
        and isinstance(entry, dict)
        and entry.get("name") == "standup_pitch_bwd"
    ):
        # ID 0x042E used to be a positive backward magnitude. It is now the
        # signed lower bound, matching the firmware's param-store v4 migration.
        return -abs(value)
    return value

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
    0x052E,  # PARAM_WHEEL_RUNAWAY_EN (the 2x ESTOP itself; persistent, so a disable outlives the bench session)
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
_STANDUP_DIVERGENCE_PARAM_IDS = frozenset({
    0x0446,  # PARAM_STANDUP_DIVERGE_PITCH_FWD_RAD
    0x0447,  # PARAM_STANDUP_DIVERGE_PITCH_BWD_RAD
})
_STANDUP_PARAM_IDS = (
    frozenset(range(0x042C, 0x043C))
    | _STANDUP_DIVERGENCE_PARAM_IDS
    | frozenset({
        0x052D,  # PARAM_STANDUP_USE_RET_GAINS
        0x052F,  # PARAM_STANDUP_CROUCH_STIFF
        0x0530,  # PARAM_STANDUP_STIFFEN_TIME_S
    })
)

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
# Exactly the roll case above, for the same reason: the jump effort dial and the
# angle/speed trajectory params were allocated out of the 0x05xx block because
# the original 0x0415-0x041F jump block was full, but schema.json declares all
# of them GROUP_CONTROL and generated_param_table.inc carries that through. Left
# to the (id >> 8) heuristic they resolve to COMMAND, which split the Jump panel
# in two — a "Jump" under Control holding the nine original params and a second
# "Jump" under Command holding the seven new ones, since _ensure_subgroup() keys
# its containers on (group, subgroup).
_JUMP_CONTROL_OVERRIDE_PARAM_IDS = frozenset(range(0x0531, 0x0538))
# Full Jump membership, both ID blocks. Doubles as the subgroup membership below
# so the two can never drift apart.
_JUMP_PARAM_IDS = frozenset(range(0x0415, 0x0420)) | _JUMP_CONTROL_OVERRIDE_PARAM_IDS
_BALANCE_TRIM_HIP_PARAM_IDS = frozenset({
    0x043C,  # PARAM_LQR_PITCH_TRIM_RET
    0x043D,  # PARAM_LQR_PITCH_TRIM_EXT
    0x0450,  # PARAM_LQR_PITCH_TRIM_CURVE
})

# ── Bench-mode presets ────────────────────────────────────────────────────────
# One-click sets of the params that decide whether RUNNING will arm, so a bench
# configuration doesn't have to be reassembled by hand from memory. The arm gate
# is state_machine.cpp on_running_guard(): it needs imu_enable, all four
# motor-enable flags, and (per half) either the motor enabled or its bypass set —
# calib_bypass_en covers the hip half, run_wheel_bypass_en the wheel half.
#
# Each bench preset also owns the transition/safety settings made meaningless by
# the hardware it removes. "Fully functional" restores every safety a bench
# session may have switched off, since "nothing bypassed" is its whole point.
#
# alpha_force_ret_en is deliberately cleared by the Full preset only. It is a
# tuning tool (persistent since 2026-08-09 so a session at one leg height
# survives a power cycle), so the bench presets leave whatever you set alone
# rather than clobbering it on every click.
#
# Values are written by NAME and resolved through the generated schema below, so
# a re-generated id can never silently point a preset at the wrong param.
_PRESET_GATE_NOTE = (
    "No Motors cuts enabled actuator outputs live; safety/bypass values also "
    "apply immediately. Every value is read back to confirm the write landed.\n"
    "Bench settings can disable safety checks and most persist across reboot. "
    "Click Full Robot, verify it, and reboot before operating the complete robot."
)
_BENCH_PRESETS: list[tuple[str, str, str, str, dict[str, float]]] = [
    # (key, button label, tooltip title, accent colour, {param name: value})
    ("full", "Full Robot", "Fully functional robot", GREEN, {
        "hip_l_enable": 1.0, "hip_r_enable": 1.0,
        "wheel_l_enable": 1.0, "wheel_r_enable": 1.0,
        "calib_bypass_en": 0.0, "run_wheel_bypass_en": 0.0,
        # Everything active, all safeties active, nothing bypassed.
        "imu_enable": 1.0,
        "pitch_watchdog_en": 1.0,
        "roll_watchdog_en": 1.0,
        "wheel_runaway_en": 1.0,
        "alpha_force_ret_en": 0.0,
    }),
    ("no_hips", "No Hips", "Hips disabled", ORANGE, {
        "hip_l_enable": 0.0, "hip_r_enable": 0.0,
        "wheel_l_enable": 1.0, "wheel_r_enable": 1.0,
        "calib_bypass_en": 1.0, "run_wheel_bypass_en": 0.0,
        # Disabled hips can never be calibrated, and STANDING_UP hard-requires
        # valid hip limits (it does NOT honour calib_bypass_en — its pitch
        # watchdog is masked, so an unknown leg pose has no safety net). Arming
        # with standup_enable=1 would therefore always ESTOP on
        # FAULT_STANDUP_FAILED. Route straight to RUNNING instead.
        "standup_enable": 0.0,
    }),
    ("no_wheels", "No Wheels", "Wheels disabled", ORANGE, {
        "hip_l_enable": 1.0, "hip_r_enable": 1.0,
        "wheel_l_enable": 0.0, "wheel_r_enable": 0.0,
        "calib_bypass_en": 0.0, "run_wheel_bypass_en": 1.0,
    }),
    ("no_motors", "No Motors", "All motors disabled", RED, {
        # Order is a safety property: dict insertion order becomes the write
        # queue. Disable pose watchdogs first, before an already-enabled arm
        # bypass can admit RUNNING. Their 200 ms debounce then cannot beat the
        # preset's paced writes. The two arm bypasses are deliberately last.
        "roll_watchdog_en": 0.0,
        "pitch_watchdog_en": 0.0,
        "wheel_runaway_en": 0.0,
        "standup_enable": 0.0,   # same reason as No Hips — see above
        "hip_l_enable": 0.0, "hip_r_enable": 0.0,
        "wheel_l_enable": 0.0, "wheel_r_enable": 0.0,
        "calib_bypass_en": 1.0, "run_wheel_bypass_en": 1.0,
        # A no-torque bench arm is commonly lying on its side or back. Pose
        # watchdogs are independent of the motor-enable gates, so leaving them
        # active causes an otherwise successful arm to ESTOP after exactly its
        # 200 ms debounce (FAULT_ROLL_WATCHDOG/PITCH_WATCHDOG). Wheel runaway is
        # likewise meaningless with both wheel interfaces disabled. Full Robot
        # explicitly restores all three before real operation. These are
        # ordinary parameter writes; firmware does not infer a separate mode
        # from the four motor-enable values.
    }),
]

# Preset apply → read-back → verify pacing. The read-back is unconditional
# rather than trusting the PARAM_SET echoes: on WiFi a dropped SET and a
# dropped echo look identical from here, and this whole feature exists because
# a param that silently wasn't set is what blocks arming.
_PRESET_SETTLE_MS      = 350   # after the SETs, before the read-back GETs
_PRESET_VERIFY_WAIT_MS = 500   # after the GETs, before judging the values
_PRESET_MAX_ROUNDS     = 4     # re-SET + re-GET attempts for stragglers
_PRESET_TOLERANCE      = 1e-3
# One PARAM_SET per tick rather than the whole preset at once. Firmware chirps
# PARAM_SET_CHIRP (C6, 30 ms) on every param whose value actually *changed*, and
# Buzzer::play() restarts the melody from note 0 on each call — so a burst of
# writes landing within a few ms of each other collapses into a single short
# tick instead of one beep per param. Spacing them past the note length gives
# the audible per-param feedback back, and stops hammering a lossy WiFi link
# with a simultaneous burst. Unchanged params are silent by firmware design.
_PRESET_SEND_SPACING_MS = 90

# Dev/debug params pulled out of their native group into their own top-level
# Diagnostics section.
_GROUP_DIAGNOSTICS = 0x09
_PLANT_ID_PARAM_IDS = frozenset(range(0x044B, 0x0450))
_DIAGNOSTICS_PARAM_IDS = frozenset({
    0x000A,  # PARAM_LOOP_PROFILE_ENABLE
}) | _PLANT_ID_PARAM_IDS

_LQR_BARRIER_PARAM_IDS = frozenset({
    0x0448,  # PARAM_LQR_BARRIER_K
    0x0449,  # PARAM_LQR_BARRIER_THRESH_RET
    0x044A,  # PARAM_LQR_BARRIER_THRESH_EXT
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
    if param_id in _JUMP_CONTROL_OVERRIDE_PARAM_IDS:
        return 0x04  # GROUP_CONTROL
    if param_id in _ROLL_HIP_OVERRIDE_PARAM_IDS:
        return 0x02  # GROUP_HIP
    if param_id in _BALANCE_TRIM_HIP_PARAM_IDS:
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

    # Height-scheduled LQR balance point, presented with the leg geometry.
    (_BALANCE_TRIM_HIP_PARAM_IDS, 0x02, "Pitch Trim vs Leg Height"),

    # Balance/control tuning.
    (frozenset({0x0400, 0x0402}), 0x04, "LQR Core"),
    (range(0x0424, 0x0429), 0x04, "LQR Gains"),
    (range(0x0404, 0x040C), 0x04, "Velocity PI"),
    (range(0x040C, 0x0412), 0x04, "Yaw PI"),
    (range(0x0412, 0x0415), 0x04, "Feedforward"),
    # One panel spanning both ID blocks — see _JUMP_PARAM_IDS.
    (_JUMP_PARAM_IDS, 0x04, "Jump"),
    (frozenset({0x0401, 0x0420, 0x0421, 0x0422}), 0x04, "Sim Injection"),
    (_ROLL_CONTROL_OVERRIDE_PARAM_IDS, 0x04, "Roll"),
    (range(0x043E, 0x0446), 0x04, "Pitch Envelope"),  # theta_max_*, pitch_wd_* (fwd/bwd x ret/ext)
    (_LQR_BARRIER_PARAM_IDS, 0x04, "Backward Pitch Barrier"),

    # Standing-up safety and developer diagnostics.
    (_STANDUP_DIVERGENCE_PARAM_IDS, 0x08, "Divergence Limits"),
    (_PLANT_ID_PARAM_IDS, 0x09, "Plant Identification"),

    # The roll members of these four sets were allocated later, out of the 0x052x
    # tail rather than next to their siblings, so they are listed explicitly —
    # they belong with the radio/profile values they are set alongside, not
    # loose at the bottom of Command.
    (frozenset(range(0x0500, 0x0504)) | frozenset({
        0x0525,  # roll_cmd_rad   (live CH1 setpoint, same as radio_hip_cmd)
        0x0526,  # radio_roll_max (readonly, copied from the active profile)
    }), 0x05, "Radio Scale"),      # + radio_hip_cmd, radio_vel_max, radio_yaw_max, live_tune_ch7_val
    (frozenset(range(0x0510, 0x0513)) | frozenset({0x0527}), 0x05, "Profile 1"),  # vel_max/yaw_max/torque_lim/roll_max
    (frozenset(range(0x0513, 0x0516)) | frozenset({0x0528}), 0x05, "Profile 2"),
    (frozenset(range(0x0516, 0x0519)) | frozenset({0x0529}), 0x05, "Profile 3"),
    # active_profile (0x0519) intentionally left without a subgroup — it's the
    # CH9-selected index, not a per-profile value.
]

_SUBGROUP_COLORS: dict[str, str] = {
    "Retracting":     "#66aaff",
    "Extending":      "#ff9955",
    "Shared":         "#99aabb",
    "Pitch Trim vs Leg Height": "#ffb36b",
    "LQR Core":       "#dd99ff",
    "LQR Gains":      "#ee88ff",
    "Velocity PI":    "#bb77ee",
    "Yaw PI":         "#9966dd",
    "Feedforward":    "#ccaaff",
    "Jump":           "#ffaa44",
    "Sim Injection":  "#88ddcc",
    "Roll":           "#55ccff",
    "Pitch Envelope": "#ff99aa",
    "Backward Pitch Barrier": "#ff7788",
    "Divergence Limits": "#ff8866",
    "Plant Identification": "#ffdd66",
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

        # Bench-mode preset apply → read-back → verify cycle (see _apply_preset).
        self._preset_timer = QTimer(self)
        self._preset_timer.setSingleShot(True)
        self._preset_timer.timeout.connect(self._preset_tick)
        self._preset_send_timer = QTimer(self)
        self._preset_send_timer.setSingleShot(True)
        self._preset_send_timer.timeout.connect(self._preset_send_next)
        self._preset_expect: dict[int, float] = {}
        self._preset_queue: list[tuple[int, float]] = []
        self._preset_title = ""
        self._preset_round = 0
        self._preset_phase = "read"   # "read" = writes done, go re-read; "judge" = compare
        self._preset_box = None

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

        # ── Bench-mode presets (right-aligned) ────────────────────────────────
        # Own status label: _update_status() rewrites _lbl_status on every
        # PARAM_REPORT, which would wipe the apply/verify result the instant the
        # read-back replies started landing.
        self._lbl_preset = QLabel("")
        self._lbl_preset.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        self._lbl_preset.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toolbar.addWidget(self._lbl_preset, stretch=1)

        bench_lbl = QLabel("Bench mode:")
        bench_lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        toolbar.addWidget(bench_lbl)

        for key, label, title, accent, values in _BENCH_PRESETS:
            btn = QPushButton(label)
            btn.setFixedWidth(82)
            btn.setStyleSheet(
                f"QPushButton{{background:{SURFACE};color:{accent};"
                f"border:1px solid {accent};border-radius:3px;padding:3px 8px}}"
                f"QPushButton:hover{{background:{accent};color:{BG}}}"
            )
            btn.setToolTip(self._preset_tooltip(title, values))
            btn.clicked.connect(lambda _checked=False, k=key: self._apply_preset(k))
            toolbar.addWidget(btn)

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

    # ── Bench-mode presets ────────────────────────────────────────────────────

    @staticmethod
    def _preset_tooltip(title: str, values: dict[str, float]) -> str:
        body = "\n".join(f"  {name} = {val:g}" for name, val in values.items())
        return f"{title}\n\nSets {len(values)} params:\n{body}\n\n{_PRESET_GATE_NOTE}"

    def _set_preset_status(self, text: str, color: str):
        self._lbl_preset.setText(text)
        self._lbl_preset.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _preset_popup(self, title: str, text: str, detail: str, icon):
        """Report the apply/verify verdict in a window, since the toolbar label
        is easy to miss.

        Deliberately NON-modal (show(), not exec()): a modal box spins a nested
        Qt event loop, which stalls the RemoteControlServer mid-handler and
        makes robot_ctl.py time out until someone clicks it (see
        software/gui/CLAUDE.md). This pops up and stays put without wedging
        automation or the telemetry UI behind it."""
        prev = getattr(self, "_preset_box", None)
        if prev is not None:
            # WA_DeleteOnClose means the C++ side may already be gone if the box
            # was closed by something other than its own finished signal.
            try:
                prev.close()
            except RuntimeError:
                pass
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.finished.connect(lambda _r: setattr(self, "_preset_box", None))
        self._preset_box = box
        box.show()
        box.raise_()

    def _preset_value_lines(self, pids) -> str:
        """name = value, read back from the robot (not what we asked for)."""
        out = []
        for pid in pids:
            name = _PARAM_DEFS.get(pid, (f"0x{pid:04X}", ""))[0]
            row = self._rows.get(pid)
            if row is None or not row.is_confirmed():
                out.append(f"  {name} = (no reply)")
            else:
                out.append(f"  {name} = {row.current_value():g}")
        return "\n".join(out)

    def _apply_preset(self, key: str):
        """Write a bench-mode preset, then read every param back to prove it
        landed. Firmware reads all of these live, so no reboot is needed."""
        preset = next((p for p in _BENCH_PRESETS if p[0] == key), None)
        if preset is None:
            return
        _, _, title, _, values = preset

        # Resolve names through the generated schema; skip (and report) anything
        # this GUI's schema doesn't know, rather than writing a guessed id.
        expect: dict[int, float] = {}
        unknown: list[str] = []
        for name, val in values.items():
            pid = _PARAM_BY_NAME.get(name)
            if pid is None:
                unknown.append(name)
                continue
            expect[pid] = val
        if not expect:
            self._set_preset_status(f"{title}: no known params to set", RED)
            return

        self._preset_title  = title
        self._preset_expect = expect
        self._preset_round  = 0
        self._preset_phase  = "read"
        self._preset_queue  = list(expect.items())

        msg = f"{title}: applying {len(expect)} params…"
        if unknown:
            msg += f" ({len(unknown)} unknown to this GUI: {', '.join(unknown)})"
        self._set_preset_status(msg, ORANGE)
        self._preset_send_next()

    def _preset_send_next(self):
        """Drain the pending writes one per tick (see _PRESET_SEND_SPACING_MS),
        then hand off to the read-back/verify cycle."""
        if not self._preset_queue:
            self._preset_timer.start(_PRESET_SETTLE_MS)
            return
        pid, val = self._preset_queue.pop(0)
        send_param_set(pid, val)
        self._preset_send_timer.start(_PRESET_SEND_SPACING_MS)

    def _preset_mismatches(self) -> list[int]:
        """Preset ids whose live value doesn't (yet) match what was written.
        An unconfirmed row counts as a mismatch — no reply is not a pass."""
        bad = []
        for pid, want in self._preset_expect.items():
            row = self._rows.get(pid)
            if row is None or not row.is_confirmed():
                bad.append(pid)
            elif abs(row.current_value() - want) > _PRESET_TOLERANCE:
                bad.append(pid)
        return bad

    def _preset_tick(self):
        if not self._preset_expect:
            return
        # "read" runs after every batch of writes — first pass and every retry
        # alike — and always re-asks firmware rather than trusting the PARAM_SET
        # echoes, which is the whole point of verifying.
        if self._preset_phase == "read":
            self._preset_phase = "judge"
            for pid in self._preset_expect:
                send_param_get(pid)
            self._set_preset_status(f"{self._preset_title}: verifying…", ORANGE)
            self._preset_timer.start(_PRESET_VERIFY_WAIT_MS)
            return

        bad = self._preset_mismatches()
        if not bad:
            pids = list(self._preset_expect)
            n = len(pids)
            self._set_preset_status(f"✓ {self._preset_title} — {n}/{n} params verified", GREEN)
            self._preset_popup(
                "Bench Preset Applied",
                f"{self._preset_title}\n\n{n} of {n} params written and confirmed.",
                "Read back from the robot after writing:\n"
                f"{self._preset_value_lines(pids)}",
                QMessageBox.Icon.Information,
            )
            self._preset_expect = {}
            return

        if self._preset_round >= _PRESET_MAX_ROUNDS:
            names = ", ".join(_PARAM_DEFS.get(pid, (f"0x{pid:04X}", ""))[0] for pid in bad)
            total = len(self._preset_expect)
            self._set_preset_status(
                f"✗ {self._preset_title} — {len(bad)} param(s) NOT confirmed: {names}", RED)
            self._preset_popup(
                "Bench Preset NOT Confirmed",
                f"{self._preset_title}\n\n{total - len(bad)} of {total} params confirmed — "
                f"{len(bad)} did NOT take after {_PRESET_MAX_ROUNDS} attempts.",
                "The robot is in a MIXED state — do not assume it is safe to arm.\n"
                "Still wrong (value read back from the robot):\n"
                f"{self._preset_value_lines(bad)}",
                QMessageBox.Icon.Warning,
            )
            self._preset_expect = {}
            return

        # Straggler: re-send just the ones that are wrong (paced, same as the
        # first pass), which then falls back into the read-back round.
        self._preset_round += 1
        self._preset_phase = "read"
        self._set_preset_status(
            f"{self._preset_title}: retry {self._preset_round}, {len(bad)} pending…", ORANGE)
        self._preset_queue = [(pid, self._preset_expect[pid]) for pid in bad]
        self._preset_send_next()

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
                value = _migrate_import_value(pid, entry, value)
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
