from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from telemetry_bus import TelemetryBus
from theme import BORDER, BLUE, DIM, GREEN, ORANGE, RED, SURFACE, TEXT, YELLOW


_NO_DATA      = 0xFFFF
_STATE_COLORS = {"RUNNING": GREEN, "ESTOP": RED, "CALIBRATION": BLUE, "STANDBY": YELLOW}
_WM_STATES    = {1: "IDLE", 8: "CLOSED_LOOP"}
_WM_MODES     = {0: "IDLE", 1: "VEL", 2: "POS", 3: "TRQ"}


class RawDataTab(QWidget):
    def __init__(self):
        super().__init__()

        self._pkt_count  = 0
        self._rate_count = 0
        self._rate_hz    = 0.0
        self._fields: dict[str, QLabel] = {}

        # ── Header bar ────────────────────────────────────────────────────────
        self._pkt_lbl  = _stat("Pkt #",  "—")
        self._rate_lbl = _stat("Rate",   "—")
        self._src_lbl  = _stat("Source", "—")
        self._crc_lbl  = _stat("CRC",    "—")

        hdr = QHBoxLayout()
        hdr.setSpacing(24)
        for lbl in (self._pkt_lbl, self._rate_lbl, self._src_lbl, self._crc_lbl):
            hdr.addWidget(lbl)
        hdr.addStretch()

        # ── Tab widget ────────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabBar::tab {{ padding: 4px 14px; font-size: 11px; }}"
            f"QTabBar::tab:selected {{ color: {BLUE}; font-weight: bold; }}"
        )

        self._build_frame_tab(tabs)
        self._build_imu_tab(tabs)
        self._build_hip_tab(tabs)
        self._build_wheel_tab(tabs)
        self._build_control_tab(tabs)
        self._build_rc_tof_tab(tabs)

        # ── Outer layout ──────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(12)
        lay.addLayout(hdr)
        lay.addWidget(_hline())
        lay.addWidget(tabs, stretch=1)

        # ── Rate timer ────────────────────────────────────────────────────────
        self._rate_timer = QTimer(self)
        self._rate_timer.timeout.connect(self._tick_rate)
        self._rate_timer.start(1000)

        TelemetryBus.instance().packet.connect(self._on_packet)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _make_page(self) -> tuple[QWidget, QGridLayout]:
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        grid = QGridLayout(content)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(6)
        return content, grid

    def _add_tab(self, tabs: QTabWidget, title: str, content: QWidget, row: int):
        content.layout().setRowStretch(row, 1)
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        tabs.addTab(scroll, title)

    def _build_frame_tab(self, tabs: QTabWidget):
        content, g = self._make_page()
        r = 0
        r = self._add_row(g, r, ("START",    "—"),  ("END",      "—"))
        r = self._add_row(g, r, ("TYPE",     "—"),  ("VERSION",  "—"))
        r = self._add_row(g, r, ("SOURCE",   "—"),  ("SEQ",      "—"))
        r = self._add_row(g, r, ("LEN",      "—"),  ("CHECKSUM", "—"))
        self._add_tab(tabs, "Frame", content, r)

    def _build_imu_tab(self, tabs: QTabWidget):
        content, g = self._make_page()
        r = 0
        r = self._section(g, r, "State")
        r = self._add_row(g, r, ("timestamp_ms",        "—"),  ("robot_state",         "—"))
        r = self._add_row(g, r, ("fault_code",          "—"),  ("imu_packet_loss_pct", "—"))
        r = self._add_row(g, r, ("loop_count",          "—"),  ("jump_state",          "—"))
        r = self._add_row(g, r, ("active_profile",      "—"),  ("pitch_trim_rad",      "—"))
        r = self._section(g, r, "Orientation")
        r = self._add_row(g, r, ("pitch_rad",           "—"),  ("roll_rad",            "—"))
        r = self._add_row(g, r, ("yaw_rad",             "—"),  ("",                    ""))
        r = self._section(g, r, "Angular rates")
        r = self._add_row(g, r, ("pitch_rate_rads",     "—"),  ("roll_rate_rads",      "—"))
        r = self._add_row(g, r, ("yaw_rate_rads",       "—"),  ("",                    ""))
        r = self._section(g, r, "Accelerometer")
        r = self._add_row(g, r, ("accel_x_ms2",         "—"),  ("accel_y_ms2",         "—"))
        r = self._add_row(g, r, ("accel_z_ms2",         "—"),  ("",                    ""))
        self._add_tab(tabs, "IMU", content, r)

    def _build_hip_tab(self, tabs: QTabWidget):
        content, g = self._make_page()
        r = 0
        r = self._section(g, r, "Feedback")
        r = self._add_row(g, r, ("hip_l_pos_rad",       "—"),  ("hip_r_pos_rad",       "—"))
        r = self._add_row(g, r, ("hip_l_vel_rads",      "—"),  ("hip_r_vel_rads",      "—"))
        r = self._add_row(g, r, ("hip_l_current_a",     "—"),  ("hip_r_current_a",     "—"))
        r = self._section(g, r, "MIT setpoints")
        r = self._add_row(g, r, ("hip_l_cmd_pos_rad",   "—"),  ("hip_r_cmd_pos_rad",   "—"))
        r = self._add_row(g, r, ("hip_l_cmd_vel_rads",  "—"),  ("hip_r_cmd_vel_rads",  "—"))
        r = self._add_row(g, r, ("hip_l_cmd_kp",        "—"),  ("hip_r_cmd_kp",        "—"))
        r = self._add_row(g, r, ("hip_l_cmd_kd",        "—"),  ("hip_r_cmd_kd",        "—"))
        r = self._add_row(g, r, ("hip_l_cmd_tff",       "—"),  ("hip_r_cmd_tff",       "—"))
        self._add_tab(tabs, "Hip Motors", content, r)

    def _build_wheel_tab(self, tabs: QTabWidget):
        content, g = self._make_page()
        r = 0
        r = self._section(g, r, "Velocity & position")
        r = self._add_row(g, r, ("wm_l_vel_turns_s",    "—"),  ("wm_r_vel_turns_s",    "—"))
        r = self._add_row(g, r, ("wm_l_pos_turns",      "—"),  ("wm_r_pos_turns",      "—"))
        r = self._add_row(g, r, ("wheel_vel_avg",        "—"),  ("",                    ""))
        r = self._section(g, r, "Electrical")
        r = self._add_row(g, r, ("wm_l_vbus",            "—"),  ("wm_r_vbus",           "—"))
        r = self._section(g, r, "Status")
        r = self._add_row(g, r, ("wm_l_state",           "—"),  ("wm_r_state",          "—"))
        r = self._add_row(g, r, ("wm_mode",              "—"),  ("",                    ""))
        r = self._section(g, r, "Errors")
        r = self._add_row(g, r, ("wm_l_error",           "—"),  ("wm_r_error",          "—"))
        r = self._section(g, r, "Torque commands")
        r = self._add_row(g, r, ("whl_tau_l",            "—"),  ("whl_tau_r",           "—"))
        self._add_tab(tabs, "Wheel Motors", content, r)

    def _build_control_tab(self, tabs: QTabWidget):
        content, g = self._make_page()
        r = 0
        r = self._section(g, r, "Velocity PI")
        r = self._add_row(g, r, ("v_ref",                "—"),  ("theta_ref",           "—"))
        r = self._section(g, r, "Yaw PI")
        r = self._add_row(g, r, ("omega_cmd_rds",        "—"),  ("tau_yaw",             "—"))
        r = self._section(g, r, "LQR balance")
        r = self._add_row(g, r, ("tau_sym",              "—"),  ("",                    ""))
        r = self._section(g, r, "Feedforward")
        r = self._add_row(g, r, ("ff1_out",              "—"),  ("ff2_out",             "—"))
        r = self._section(g, r, "Misc")
        r = self._add_row(g, r, ("test_val",             "—"),  ("health_flags",        "—"))
        self._add_tab(tabs, "Control", content, r)

    def _build_rc_tof_tab(self, tabs: QTabWidget):
        content, g = self._make_page()
        r = 0
        r = self._section(g, r, "RC (iBUS)")
        r = self._add_row(g, r, ("ibus_alive",           "—"),  ("",                    ""))
        for i in range(0, 14, 2):
            r = self._add_row(g, r, (f"ibus_ch[{i}]",   "—"),  (f"ibus_ch[{i+1}]",    "—"))
        r = self._section(g, r, "ToF distances")
        r = self._add_row(g, r, ("tof_dist_mm[0]",       "—"),  ("tof_dist_mm[1]",     "—"))
        r = self._add_row(g, r, ("tof_dist_mm[2]",       "—"),  ("tof_dist_mm[3]",     "—"))
        r = self._add_row(g, r, ("tof_front_min_mm",     "—"),  ("tof_rear_min_mm",    "—"))
        self._add_tab(tabs, "RC / ToF", content, r)

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _section(self, grid: QGridLayout, row: int, title: str) -> int:
        if row > 0:
            grid.addWidget(_hline(), row, 0, 1, 4)
            row += 1
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {BLUE}; font-weight: bold; font-size: 11px;")
        grid.addWidget(lbl, row, 0, 1, 4)
        return row + 1

    def _add_row(self, grid: QGridLayout, row: int,
                 left: tuple[str, str], right: tuple[str, str]) -> int:
        for col_offset, (field_name, field_default) in enumerate([left, right]):
            if not field_name:
                continue
            k = QLabel(field_name)
            k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            k.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            k.setMinimumWidth(130)
            v = QLabel(field_default)
            v.setStyleSheet(
                f"color: {TEXT}; font-size: 13px; font-weight: bold;"
                f" font-family: Consolas; background: {SURFACE};"
                f" border: 1px solid {BORDER}; border-radius: 3px;"
                f" padding: 2px 8px;"
            )
            v.setMinimumWidth(140)
            grid.addWidget(k, row, col_offset * 2)
            grid.addWidget(v, row, col_offset * 2 + 1)
            self._fields[field_name] = v
        return row + 1

    # ── Update ────────────────────────────────────────────────────────────────

    def _set(self, key: str, text: str, color: str = TEXT):
        lbl = self._fields.get(key)
        if lbl is None:
            return
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
            f" font-family: Consolas; background: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 3px;"
            f" padding: 2px 8px;"
        )

    def _on_packet(self, info: dict):
        self._pkt_count += 1
        self._rate_count += 1

        # Header bar
        self._pkt_lbl.findChild(QLabel, "val").setText(str(self._pkt_count))
        src = info.get("src_name", "—")
        _set_stat(self._src_lbl, src, BLUE if src == "TEENSY" else ORANGE)
        crc = info.get("crc_ok")
        _set_stat(self._crc_lbl, "OK" if crc else "FAIL", GREEN if crc else RED)

        # Frame tab — always updated
        self._set("START",    "0xAA 0x55")
        self._set("END",      "0xEF")
        self._set("TYPE",     f"{info.get('type_name','—')}  (0x{info.get('ptype',0):02X})")
        self._set("VERSION",  str(info.get("version", "—")))
        self._set("SOURCE",   f"{info.get('src_name','—')}  (0x{info.get('source', 0):02X})")
        self._set("SEQ",      str(info.get("seq", "—")))
        self._set("LEN",      str(info.get("length", "—")))
        chk = info.get("checksum")
        self._set("CHECKSUM", f"0x{chk:02X}" if chk is not None else "—")

        if info.get("ptype") != 0x01:
            return

        f = info

        # IMU tab
        state = f.get("state_name", "—")
        self._set("timestamp_ms",        str(f.get("timestamp_ms", "—")))
        self._set("robot_state",         f"{state}  ({f.get('robot_state', '—')})",
                  _STATE_COLORS.get(state, TEXT))
        fault = f.get("fault_code", 0)
        self._set("fault_code",          f"{f.get('fault_name', '—')}  (0x{fault:02X})",
                  RED if fault else GREEN)
        imu_loss = f.get("imu_packet_loss_pct", 0)
        self._set("imu_packet_loss_pct", f"{imu_loss}%",
                  RED if imu_loss > 10 else (YELLOW if imu_loss > 0 else GREEN))
        self._set("loop_count",          str(f.get("loop_count", "—")))
        self._set("jump_state",          str(f.get("jump_state", "—")))
        ap = f.get("active_profile", "—")
        self._set("active_profile",      f"P{ap + 1}" if isinstance(ap, int) else "—")
        self._set("pitch_trim_rad",      f"{f.get('pitch_trim_rad', 0.0):+.4f}")
        self._set("pitch_rad",           f"{f.get('pitch_rad', 0.0):+.6f}")
        self._set("roll_rad",            f"{f.get('roll_rad', 0.0):+.6f}")
        self._set("yaw_rad",             f"{f.get('yaw_rad', 0.0):+.6f}")
        self._set("pitch_rate_rads",     f"{f.get('pitch_rate_rads', 0.0):+.6f}")
        self._set("roll_rate_rads",      f"{f.get('roll_rate_rads', 0.0):+.6f}")
        self._set("yaw_rate_rads",       f"{f.get('yaw_rate_rads', 0.0):+.6f}")
        self._set("accel_x_ms2",         f"{f.get('accel_x_ms2', 0.0):+.4f}")
        self._set("accel_y_ms2",         f"{f.get('accel_y_ms2', 0.0):+.4f}")
        self._set("accel_z_ms2",         f"{f.get('accel_z_ms2', 0.0):+.4f}")

        # Hip tab
        self._set("hip_l_pos_rad",       f"{f.get('hip_l_pos_rad', 0.0):+.6f}")
        self._set("hip_r_pos_rad",       f"{f.get('hip_r_pos_rad', 0.0):+.6f}")
        self._set("hip_l_vel_rads",      f"{f.get('hip_l_vel_rads', 0.0):+.6f}")
        self._set("hip_r_vel_rads",      f"{f.get('hip_r_vel_rads', 0.0):+.6f}")
        self._set("hip_l_current_a",     f"{f.get('hip_l_current_a', 0.0):+.4f}")
        self._set("hip_r_current_a",     f"{f.get('hip_r_current_a', 0.0):+.4f}")
        self._set("hip_l_cmd_pos_rad",   f"{f.get('hip_l_cmd_pos_rad', 0.0):+.6f}")
        self._set("hip_r_cmd_pos_rad",   f"{f.get('hip_r_cmd_pos_rad', 0.0):+.6f}")
        self._set("hip_l_cmd_vel_rads",  f"{f.get('hip_l_cmd_vel_rads', 0.0):+.6f}")
        self._set("hip_r_cmd_vel_rads",  f"{f.get('hip_r_cmd_vel_rads', 0.0):+.6f}")
        self._set("hip_l_cmd_kp",        f"{f.get('hip_l_cmd_kp', 0.0):.4f}")
        self._set("hip_r_cmd_kp",        f"{f.get('hip_r_cmd_kp', 0.0):.4f}")
        self._set("hip_l_cmd_kd",        f"{f.get('hip_l_cmd_kd', 0.0):.4f}")
        self._set("hip_r_cmd_kd",        f"{f.get('hip_r_cmd_kd', 0.0):.4f}")
        self._set("hip_l_cmd_tff",       f"{f.get('hip_l_cmd_tff', 0.0):+.4f}")
        self._set("hip_r_cmd_tff",       f"{f.get('hip_r_cmd_tff', 0.0):+.4f}")

        # Wheel tab
        self._set("wm_l_vel_turns_s",    f"{f.get('wm_l_vel_turns_s', 0.0):+.4f}")
        self._set("wm_r_vel_turns_s",    f"{f.get('wm_r_vel_turns_s', 0.0):+.4f}")
        self._set("wm_l_pos_turns",      f"{f.get('wm_l_pos_turns', 0.0):+.4f}")
        self._set("wm_r_pos_turns",      f"{f.get('wm_r_pos_turns', 0.0):+.4f}")
        self._set("wheel_vel_avg",       f"{f.get('wheel_vel_avg', 0.0):+.4f}")
        self._set("wm_l_vbus",           f"{f.get('wm_l_vbus', 0.0):.2f} V")
        self._set("wm_r_vbus",           f"{f.get('wm_r_vbus', 0.0):.2f} V")
        wm_l_st = f.get("wm_l_state", 0)
        wm_r_st = f.get("wm_r_state", 0)
        wm_mode = f.get("wm_mode", 0)
        self._set("wm_l_state",          _WM_STATES.get(wm_l_st, str(wm_l_st)))
        self._set("wm_r_state",          _WM_STATES.get(wm_r_st, str(wm_r_st)))
        self._set("wm_mode",             _WM_MODES.get(wm_mode, str(wm_mode)))
        wm_l_err = f.get("wm_l_error", 0)
        wm_r_err = f.get("wm_r_error", 0)
        self._set("wm_l_error",          f"0x{wm_l_err:08X}", RED if wm_l_err else GREEN)
        self._set("wm_r_error",          f"0x{wm_r_err:08X}", RED if wm_r_err else GREEN)
        self._set("whl_tau_l",           f"{f.get('whl_tau_l', 0.0):+.6f}")
        self._set("whl_tau_r",           f"{f.get('whl_tau_r', 0.0):+.6f}")

        # Control tab
        self._set("v_ref",               f"{f.get('v_ref', 0.0):+.6f}")
        self._set("theta_ref",           f"{f.get('theta_ref', 0.0):+.6f}")
        self._set("omega_cmd_rds",       f"{f.get('omega_cmd_rds', 0.0):+.6f}")
        self._set("tau_yaw",             f"{f.get('tau_yaw', 0.0):+.4f}")
        self._set("tau_sym",             f"{f.get('tau_sym', 0.0):+.4f}")
        self._set("ff1_out",             f"{f.get('ff1_out', 0.0):+.4f}")
        self._set("ff2_out",             f"{f.get('ff2_out', 0.0):+.4f}")
        self._set("test_val",            f"{f.get('test_val', 0.0):+.4f}")
        hflags = f.get("health_flags", 0)
        self._set("health_flags",        f"0x{hflags:04X}")

        # RC / ToF tab
        self._set("ibus_alive",          "YES" if f.get("ibus_alive") else "NO",
                  GREEN if f.get("ibus_alive") else RED)
        ibus = f.get("ibus_ch", [0] * 14)
        for i in range(14):
            val = ibus[i] if i < len(ibus) else 0
            self._set(f"ibus_ch[{i}]",   str(val),
                      TEXT if abs(val - 1500) < 20 else ORANGE)
        tof = f.get("tof_dist_mm", [_NO_DATA] * 4)
        for i in range(4):
            d = tof[i] if i < len(tof) else _NO_DATA
            self._set(f"tof_dist_mm[{i}]", "—" if d == _NO_DATA else f"{d} mm")
        tfr = f.get("tof_front_min_mm", _NO_DATA)
        trr = f.get("tof_rear_min_mm",  _NO_DATA)
        self._set("tof_front_min_mm",    "—" if tfr == _NO_DATA else f"{tfr} mm")
        self._set("tof_rear_min_mm",     "—" if trr == _NO_DATA else f"{trr} mm")

    def _tick_rate(self):
        self._rate_hz    = float(self._rate_count)
        self._rate_count = 0
        _set_stat(self._rate_lbl, f"{self._rate_hz:.0f} Hz",
                  GREEN if self._rate_hz > 100 else (ORANGE if self._rate_hz > 0 else DIM))


# ── Small helpers ─────────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


def _stat(label: str, default: str) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    k = QLabel(label + ":")
    k.setStyleSheet(f"color: {DIM}; font-size: 11px;")
    v = QLabel(default)
    v.setObjectName("val")
    v.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: bold; font-family: Consolas;")
    lay.addWidget(k)
    lay.addWidget(v)
    return w


def _set_stat(widget: QWidget, text: str, color: str = TEXT):
    lbl = widget.findChild(QLabel, "val")
    if lbl:
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold; font-family: Consolas;"
        )
