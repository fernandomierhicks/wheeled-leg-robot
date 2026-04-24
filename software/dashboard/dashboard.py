"""dashboard.py — Live telemetry dashboard for the wheeled-leg robot.

Receives 65-byte telemetry packets from the Arduino UNO R4 WiFi
and displays them in a 5x2 pyqtgraph FastChart (same layout as the
simulation visualizer).  Also sends commands back to the robot.

Supports two transport modes:
  - UDP/WiFi (default): auto-discover or specify --robot IP
  - USB-Serial:         specify --serial COM5 (or port name)

Usage:
    python dashboard.py                       # auto-detect USB-Serial (default)
    python dashboard.py --serial COM5         # explicit serial port
    python dashboard.py --robot 192.168.1.42  # WiFi mode with known IP
"""

import argparse
import math
import socket
import struct
import sys
import threading
import time
from collections import deque
from queue import Queue, Empty

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Protocol constants (must match firmware config.h) ────────────────────────
TELEMETRY_PORT = 4210
COMMAND_PORT   = 4211

# Telemetry packet: uint32 ts | uint8 mode | 17 floats | 2 floats (encoder) | uint8 flags
TELEM_FMT  = '<IB17f2fB'
TELEM_SIZE = struct.calcsize(TELEM_FMT)  # 82 bytes
assert TELEM_SIZE == 82

# Command type IDs
CMD_DRIVE          = 1
CMD_MODE           = 2
CMD_GAIN           = 3
CMD_PING           = 4
CMD_HIP            = 5
CMD_ODRIVE_ENABLE  = 7   # 1 byte: ctrl_mode (2=velocity, 3=position)
CMD_ODRIVE_DISABLE = 8   # no payload
CMD_ODRIVE_VEL     = 9   # 1 float: turns/s
CMD_ODRIVE_POS     = 10  # 1 float: turns
CMD_ODRIVE_CLEAR   = 11  # no payload: clear errors on axis 0

# ── Style constants (from simulation visualizer) ────────────────────────────
BG_COLOR   = "#12121e"
BAR_COLOR  = "#1a1a2e"
TICK_COLOR = "#d8d8d8"
LINE_WIDTH = 1.4

# Torque limits for chart limit lines
WHEEL_TORQUE_MAX = 6.825
HIP_TORQUE_MAX   = 7.0

WINDOW_S     = 15.0   # rolling window in seconds
TELEMETRY_HZ = 50     # expected packet rate from firmware
MAXLEN       = int(WINDOW_S * TELEMETRY_HZ) + 200

# ── Mode names ───────────────────────────────────────────────────────────────
MODE_NAMES = {0: "IDLE", 1: "BALANCE", 2: "DRIVE", 3: "JUMP", 4: "STAND_UP", 255: "FAULT"}

# ═════════════════════════════════════════════════════════════════════════════
# UDP RECEIVER THREAD
# ═════════════════════════════════════════════════════════════════════════════

class UDPReceiver(threading.Thread):
    """Background thread: receives telemetry packets, pushes to queue."""

    def __init__(self, data_q: Queue, port: int = TELEMETRY_PORT):
        super().__init__(daemon=True)
        self.data_q = data_q
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.sock.settimeout(0.5)  # allow clean shutdown
        self.running = True
        self.robot_ip = None
        self.last_rx_time = 0.0

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) != TELEM_SIZE:
                continue

            self.robot_ip = addr[0]
            self.last_rx_time = time.monotonic()

            vals = struct.unpack(TELEM_FMT, data)
            # vals: (timestamp_ms, mode, pitch, pitch_rate, roll, yaw,
            #        wheel_vel_avg, v_cmd, theta_ref, tau_sym, tau_yaw,
            #        tau_wheel_L, tau_wheel_R, hip_q_L, tau_hip_L, tau_hip_R, hip_q_R,
            #        dt_us, debug_sine,
            #        wheel_pos_L, wheel_vel_L, status_flags)
            try:
                self.data_q.put_nowait(vals)
            except Exception:
                pass  # drop if queue full

    def stop(self):
        self.running = False
        self.sock.close()


# ═════════════════════════════════════════════════════════════════════════════
# SERIAL RECEIVER THREAD
# ═════════════════════════════════════════════════════════════════════════════

SERIAL_SYNC = bytes([0xAA, 0x55])
SERIAL_FRAME_SIZE = 2 + TELEM_SIZE + 1  # sync(2) + packet + checksum(1)

# Command frame sync bytes (firmware→dashboard direction is 0xAA55; dashboard→firmware is 0xBBCC)
CMD_SYNC = bytes([0xBB, 0xCC])


def _find_serial_port():
    """Auto-detect Arduino UNO R4 WiFi COM port."""
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        vid_pid = f"{p.vid:04X}:{p.pid:04X}" if p.vid else ""
        # UNO R4 WiFi shows as "USB Serial Device" with VID:PID 2341:1002
        if "2341" in vid_pid or "arduino" in desc or "uno" in desc:
            return p.device
    # Fallback: return first available port
    ports = serial.tools.list_ports.comports()
    if ports:
        return ports[0].device
    return None


class SerialReceiver(threading.Thread):
    """Background thread: receives framed telemetry packets from USB-UART.
    Also supports sending command frames to the firmware via write_cmd()."""

    def __init__(self, data_q: Queue, port: str, baud: int = 1000000):
        super().__init__(daemon=True)
        self.data_q = data_q
        self.port_name = port
        self.baud = baud
        self.running = True
        self.robot_ip = "USB-Serial"   # fake IP for status display compatibility
        self.last_rx_time = 0.0
        self._lock = threading.Lock()
        try:
            self._ser = serial.Serial(port, baud, timeout=0.5)
            print(f"[Serial] Opened {port} @ {baud} baud")
        except serial.SerialException as e:
            print(f"[Serial] Failed to open {port}: {e}")
            self._ser = None

    def write_cmd(self, data: bytes):
        """Write raw bytes to serial port (thread-safe). Used by CommandSender."""
        with self._lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(data)
                except serial.SerialException:
                    pass

    def run(self):
        if not self._ser:
            return
        print(f"[Serial] Listening for telemetry on {self.port_name}")
        buf = bytearray()

        while self.running:
            try:
                with self._lock:
                    waiting = self._ser.in_waiting
                chunk = self._ser.read(max(1, waiting))
            except serial.SerialException:
                break
            if not chunk:
                continue

            buf.extend(chunk)

            # Scan for sync + complete frame
            while len(buf) >= SERIAL_FRAME_SIZE:
                idx = buf.find(SERIAL_SYNC)
                if idx < 0:
                    buf = buf[-1:]
                    break
                if idx > 0:
                    del buf[:idx]
                if len(buf) < SERIAL_FRAME_SIZE:
                    break

                pkt_bytes = bytes(buf[2:2 + TELEM_SIZE])
                rx_ck = buf[2 + TELEM_SIZE]

                xor_ck = 0
                for b in pkt_bytes:
                    xor_ck ^= b

                if xor_ck != rx_ck:
                    del buf[:1]
                    continue

                del buf[:SERIAL_FRAME_SIZE]
                self.last_rx_time = time.monotonic()

                vals = struct.unpack(TELEM_FMT, pkt_bytes)
                try:
                    self.data_q.put_nowait(vals)
                except Exception:
                    pass

    def stop(self):
        self.running = False
        if self._ser and self._ser.is_open:
            self._ser.close()


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND SENDER
# ═════════════════════════════════════════════════════════════════════════════

class CommandSender:
    """Sends command packets to the robot via UDP (WiFi) or serial (USB).

    In serial mode, pass serial_rx=<SerialReceiver instance>.  Commands are
    framed as: [0xBB][0xCC][cmd_type][payload...][XOR checksum].
    """

    def __init__(self, robot_ip: str = None, serial_rx=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_ip = robot_ip
        self.serial_rx = serial_rx  # SerialReceiver, if in serial mode

    # ── Internal helpers ────────────────────────────────────────────────────

    def _send_serial(self, cmd_type: int, payload: bytes = b''):
        if not self.serial_rx:
            return
        xor = cmd_type
        for b in payload:
            xor ^= b
        frame = CMD_SYNC + bytes([cmd_type]) + payload + bytes([xor])
        self.serial_rx.write_cmd(frame)

    def _send_udp(self, data: bytes):
        if self.robot_ip:
            self.sock.sendto(data, (self.robot_ip, COMMAND_PORT))

    # ── Public API ──────────────────────────────────────────────────────────

    def send_ping(self, ip: str = None):
        if self.serial_rx:
            self._send_serial(CMD_PING)
        else:
            target = ip or self.robot_ip
            if target:
                self.sock.sendto(struct.pack('<B', CMD_PING), (target, COMMAND_PORT))

    def send_drive(self, v_cmd: float, omega_cmd: float, hip_target: float):
        payload = struct.pack('<fff', v_cmd, omega_cmd, hip_target)
        if self.serial_rx:
            self._send_serial(CMD_DRIVE, payload)
        else:
            self._send_udp(struct.pack('<B', CMD_DRIVE) + payload)

    def send_mode(self, mode: int):
        if self.serial_rx:
            self._send_serial(CMD_MODE, bytes([mode]))
        else:
            self._send_udp(struct.pack('<BB', CMD_MODE, mode))

    def send_hip(self, motor_id: int, hip_cmd: int,
                 p_des: float = 0.0, v_des: float = 0.0,
                 kp: float = 0.0, kd: float = 0.0, t_ff: float = 0.0):
        # HIP command is WiFi-only (AK45 not on single-axis ODrive bench)
        if not self.robot_ip:
            return
        pkt = struct.pack('<BBBfffff', CMD_HIP, motor_id, hip_cmd,
                          p_des, v_des, kp, kd, t_ff)
        self._send_udp(pkt)

    def send_odrive_enable(self, ctrl_mode: int):
        """Enable ODrive in velocity (2) or position (3) mode."""
        if self.serial_rx:
            self._send_serial(CMD_ODRIVE_ENABLE, bytes([ctrl_mode]))
        else:
            self._send_udp(struct.pack('<BB', CMD_ODRIVE_ENABLE, ctrl_mode))

    def send_odrive_disable(self):
        if self.serial_rx:
            self._send_serial(CMD_ODRIVE_DISABLE)
        else:
            self._send_udp(struct.pack('<B', CMD_ODRIVE_DISABLE))

    def send_odrive_clear_errors(self):
        if self.serial_rx:
            self._send_serial(CMD_ODRIVE_CLEAR)
        else:
            self._send_udp(struct.pack('<B', CMD_ODRIVE_CLEAR))

    def send_odrive_velocity(self, vel_turns_s: float):
        """Set ODrive velocity setpoint [turns/s]."""
        payload = struct.pack('<f', vel_turns_s)
        if self.serial_rx:
            self._send_serial(CMD_ODRIVE_VEL, payload)
        else:
            self._send_udp(struct.pack('<B', CMD_ODRIVE_VEL) + payload)

    def send_odrive_position(self, pos_turns: float):
        """Set ODrive position setpoint [turns]."""
        payload = struct.pack('<f', pos_turns)
        if self.serial_rx:
            self._send_serial(CMD_ODRIVE_POS, payload)
        else:
            self._send_udp(struct.pack('<B', CMD_ODRIVE_POS) + payload)

    def send_broadcast_ping(self):
        """Send ping to broadcast address to discover robot."""
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.sendto(struct.pack('<B', CMD_PING), ('255.255.255.255', COMMAND_PORT))


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD GUI
# ═════════════════════════════════════════════════════════════════════════════

def run_dashboard(robot_ip: str = None, serial_port: str = None):
    pg.setConfigOptions(antialias=False, useOpenGL=True, enableExperimental=True)
    app = QtWidgets.QApplication(sys.argv)

    # ── Data pipeline ────────────────────────────────────────────────────────
    data_q = Queue(maxsize=500)

    if serial_port:
        if not HAS_SERIAL:
            print("ERROR: pyserial not installed.  pip install pyserial")
            sys.exit(1)
        if serial_port == "auto":
            serial_port = _find_serial_port()
            if not serial_port:
                print("ERROR: No serial port found. Specify port explicitly.")
                sys.exit(1)
            print(f"[Dashboard] Auto-detected serial port: {serial_port}")
        receiver = SerialReceiver(data_q, serial_port)
        cmd = CommandSender(serial_rx=receiver)  # commands enabled via serial
    else:
        receiver = UDPReceiver(data_q)
        cmd = CommandSender(robot_ip)

    # ── Main window ──────────────────────────────────────────────────────────
    main_win = QtWidgets.QMainWindow()
    main_win.setWindowTitle("Robot Dashboard — Waiting for telemetry…")
    main_win.setStyleSheet(f"background:{BG_COLOR};")
    central = QtWidgets.QWidget()
    main_win.setCentralWidget(central)
    vbox = QtWidgets.QVBoxLayout(central)
    vbox.setContentsMargins(4, 4, 4, 2)
    vbox.setSpacing(2)

    # ── Top area: charts left, 3D axis right ─────────────────────────────────
    top_hbox = QtWidgets.QHBoxLayout()
    top_hbox.setSpacing(4)
    vbox.addLayout(top_hbox, stretch=1)

    glw = pg.GraphicsLayoutWidget()
    glw.setBackground(BG_COLOR)
    top_hbox.addWidget(glw, stretch=1)

    # ── 3D IMU Axis Viewer ─────────────────────────────────────────────────
    axis_frame = QtWidgets.QWidget()
    axis_frame.setFixedWidth(240)
    axis_frame.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
    axis_vbox = QtWidgets.QVBoxLayout(axis_frame)
    axis_vbox.setContentsMargins(4, 4, 4, 4)
    axis_vbox.setSpacing(2)

    axis_title = QtWidgets.QLabel("IMU Orientation")
    axis_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    axis_title.setStyleSheet(
        "color:#e0e0e0; font-family:Consolas,monospace; font-size:11px; font-weight:bold;")
    axis_vbox.addWidget(axis_title)

    axis_view = gl.GLViewWidget()
    axis_view.opts['distance'] = 3.5
    axis_view.opts['elevation'] = 25
    axis_view.opts['azimuth'] = -50
    axis_view.setBackgroundColor(BG_COLOR)
    axis_vbox.addWidget(axis_view, stretch=1)

    # Draw XYZ axes as colored lines
    AXIS_LEN = 1.3
    axis_defs = [
        ('X', np.array([AXIS_LEN, 0, 0]), (1.0, 0.25, 0.25, 1.0)),   # red
        ('Y', np.array([0, AXIS_LEN, 0]), (0.25, 1.0, 0.25, 1.0)),   # green
        ('Z', np.array([0, 0, AXIS_LEN]), (0.3, 0.55, 1.0, 1.0)),    # blue
    ]
    axis_gl_lines = []
    for name, vec, color in axis_defs:
        pts = np.array([[0, 0, 0], vec.tolist()], dtype=np.float32)
        line = gl.GLLinePlotItem(pos=pts, color=color, width=3.0, antialias=True)
        axis_view.addItem(line)
        axis_gl_lines.append((vec, color, line))

    # Faint reference grid on the XY plane
    grid = gl.GLGridItem()
    grid.setSize(3, 3, 1)
    grid.setSpacing(0.5, 0.5, 0.5)
    grid.setColor((255, 255, 255, 25))
    axis_view.addItem(grid)

    # Color legend
    legend_lbl = QtWidgets.QLabel(
        '<span style="color:#ff4444;">X fwd</span> &nbsp; '
        '<span style="color:#44ff44;">Y left</span> &nbsp; '
        '<span style="color:#5599ff;">Z up</span>')
    legend_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    legend_lbl.setStyleSheet("font-family:Consolas; font-size:10px;")
    axis_vbox.addWidget(legend_lbl)

    # Pitch/roll readout
    imu_readout = QtWidgets.QLabel("Pitch: --  Roll: --")
    imu_readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    imu_readout.setStyleSheet(
        "color:#c8c8c8; font-family:Consolas; font-size:10px;")
    axis_vbox.addWidget(imu_readout)

    top_hbox.addWidget(axis_frame)

    # ── Status bar ───────────────────────────────────────────────────────────
    status_row = QtWidgets.QWidget()
    status_row.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
    hbox_status = QtWidgets.QHBoxLayout(status_row)
    hbox_status.setContentsMargins(6, 3, 6, 3)
    hbox_status.setSpacing(0)

    _SL = ("color:#e8e8e8; font-family:Consolas,monospace; "
           "font-size:11px; font-weight:bold; padding:0 14px 0 0;")

    lbl_conn     = QtWidgets.QLabel("Connection: --")
    lbl_mode     = QtWidgets.QLabel("Mode: --")
    lbl_dt       = QtWidgets.QLabel("dt: --")
    lbl_rate     = QtWidgets.QLabel("Rate: --")
    hover_lbl    = QtWidgets.QLabel("")

    for lbl in (lbl_conn, lbl_mode, lbl_dt, lbl_rate, hover_lbl):
        lbl.setStyleSheet(_SL)
        hbox_status.addWidget(lbl)
    hbox_status.addStretch()

    vbox.addWidget(status_row)

    # ── Command bar ──────────────────────────────────────────────────────────
    cmd_row = QtWidgets.QWidget()
    cmd_row.setStyleSheet(f"background:{BAR_COLOR}; border-radius:4px;")
    hbox_cmd = QtWidgets.QHBoxLayout(cmd_row)
    hbox_cmd.setContentsMargins(6, 4, 6, 4)
    hbox_cmd.setSpacing(8)

    _BTN_STYLE = ("QPushButton{background:#3a3a5e;color:white;font-size:11px;"
                  "font-family:Consolas;border-radius:4px;padding:4px 12px}"
                  "QPushButton:hover{background:#5a5a9e}"
                  "QPushButton:pressed{background:#7a7abe}")

    def _mode_btn(name, mode_val):
        btn = QtWidgets.QPushButton(name)
        btn.setStyleSheet(_BTN_STYLE)
        btn.clicked.connect(lambda: cmd.send_mode(mode_val))
        hbox_cmd.addWidget(btn)
        return btn

    _mode_btn("IDLE", 0)
    _mode_btn("BALANCE", 1)
    _mode_btn("DRIVE", 2)

    hbox_cmd.addSpacing(20)

    # Ping button (for discovery)
    btn_ping = QtWidgets.QPushButton("Ping")
    btn_ping.setStyleSheet(_BTN_STYLE)
    btn_ping.clicked.connect(lambda: cmd.send_broadcast_ping())
    hbox_cmd.addWidget(btn_ping)

    hbox_cmd.addSpacing(20)

    # ── Transport toggle button ───────────────────────────────────────────
    _TOGGLE_USB = "QPushButton{background:#2e6e3e;color:white;font-size:11px;font-family:Consolas;border-radius:4px;padding:4px 12px}QPushButton:hover{background:#3e8e5e}QPushButton:pressed{background:#5eae7e}"
    _TOGGLE_WIFI = "QPushButton{background:#3e3e8e;color:white;font-size:11px;font-family:Consolas;border-radius:4px;padding:4px 12px}QPushButton:hover{background:#5e5eae}QPushButton:pressed{background:#7e7ece}"

    btn_transport = QtWidgets.QPushButton()
    # Track mutable state in a list so closures can modify it
    _transport = ["serial" if serial_port else "wifi"]
    _receiver_ref = [receiver]

    def _update_transport_btn():
        if _transport[0] == "serial":
            btn_transport.setText("USB-UART ⇄ WiFi")
            btn_transport.setStyleSheet(_TOGGLE_USB)
            btn_transport.setToolTip("Currently USB-UART — click to switch to WiFi")
        else:
            btn_transport.setText("WiFi ⇄ USB-UART")
            btn_transport.setStyleSheet(_TOGGLE_WIFI)
            btn_transport.setToolTip("Currently WiFi — click to switch to USB-UART")

    def _switch_transport():
        nonlocal receiver
        # Stop current receiver
        _receiver_ref[0].stop()
        # Clear stale data from queue
        while not data_q.empty():
            try:
                data_q.get_nowait()
            except Empty:
                break

        if _transport[0] == "serial":
            # Switch to WiFi
            _transport[0] = "wifi"
            new_rx = UDPReceiver(data_q)
            cmd.robot_ip = None  # will be discovered
            cmd.serial_rx = None
            print("[Dashboard] Switched to WiFi (UDP) transport")
        else:
            # Switch to USB-UART
            if not HAS_SERIAL:
                print("[Dashboard] ERROR: pyserial not installed — can't switch to USB-UART")
                return
            port = _find_serial_port()
            if not port:
                print("[Dashboard] ERROR: No serial port found")
                return
            _transport[0] = "serial"
            new_rx = SerialReceiver(data_q, port)
            cmd.robot_ip = None
            cmd.serial_rx = new_rx
            print(f"[Dashboard] Switched to USB-UART on {port}")

        receiver = new_rx
        _receiver_ref[0] = new_rx
        new_rx.start()
        _update_transport_btn()

        # Send ping if WiFi
        if _transport[0] == "wifi":
            cmd.send_broadcast_ping()

    _update_transport_btn()
    btn_transport.clicked.connect(_switch_transport)
    hbox_cmd.addWidget(btn_transport)

    hbox_cmd.addStretch()
    vbox.addWidget(cmd_row)

    # ── AK45 Motors tab ──────────────────────────────────────────────────────
    _SPIN_STYLE = ("QDoubleSpinBox{background:#1e1e3a;color:white;"
                   "font-family:Consolas;font-size:10px;border:1px solid #444;"
                   "border-radius:3px;padding:1px 4px}"
                   "QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{"
                   "width:12px;background:#3a3a5e}")
    _LBL_LIVE = ("color:#80ff80;font-family:Consolas;font-size:11px;"
                 "font-weight:bold;min-width:120px")
    _LBL_DIM  = "color:#888;font-family:Consolas;font-size:10px;"

    tab_widget = QtWidgets.QTabWidget()
    tab_widget.setStyleSheet(
        "QTabWidget::pane{background:#1a1a2e;border:1px solid #333}"
        "QTabBar::tab{background:#2a2a4e;color:#aaa;font-family:Consolas;"
        "font-size:10px;padding:4px 14px;border-radius:3px 3px 0 0}"
        "QTabBar::tab:selected{background:#3a3a6e;color:white}")
    tab_widget.setMaximumHeight(280)

    # ── ODrive tab (single-axis bench test — velocity / position native modes) ──
    odrive_tab = QtWidgets.QWidget()
    odrive_tab.setStyleSheet(f"background:{BG_COLOR};")
    tab_widget.addTab(odrive_tab, "ODrive Axis 0")

    odrive_hbox = QtWidgets.QHBoxLayout(odrive_tab)
    odrive_hbox.setContentsMargins(10, 8, 10, 8)
    odrive_hbox.setSpacing(12)

    def _od_spin(lo, hi, step, val, dec=3, w=90):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step)
        s.setValue(val); s.setDecimals(dec)
        s.setStyleSheet(_SPIN_STYLE); s.setFixedWidth(w)
        return s

    def _od_lbl(txt, dim=False):
        l = QtWidgets.QLabel(txt)
        l.setStyleSheet(
            "color:#888;font-family:Consolas;font-size:10px;" if dim
            else "color:#c8c8c8;font-family:Consolas;font-size:10px;")
        return l

    # ── Left panel: enable / velocity / position controls ────────────────
    od_ctrl = QtWidgets.QWidget()
    od_ctrl.setStyleSheet(f"background:{BAR_COLOR};border-radius:4px;")
    od_ctrl_vb = QtWidgets.QVBoxLayout(od_ctrl)
    od_ctrl_vb.setContentsMargins(10, 8, 10, 8)
    od_ctrl_vb.setSpacing(6)

    od_title = QtWidgets.QLabel("ODrive v3.6  —  Axis 0 (CAN node 0)")
    od_title.setStyleSheet("color:white;font-family:Consolas;font-size:11px;font-weight:bold;")
    od_ctrl_vb.addWidget(od_title)

    od_status_lbl = QtWidgets.QLabel("Wheel vel: --  rad/s  (-- turns/s)")
    od_status_lbl.setStyleSheet("color:#80ff80;font-family:Consolas;font-size:11px;")
    od_ctrl_vb.addWidget(od_status_lbl)

    od_ctrl_vb.addSpacing(4)

    # Enable row
    od_enable_row = QtWidgets.QHBoxLayout()
    od_enable_row.setSpacing(6)
    for label, ctrl_mode, col in [
            ("Enable Vel", 2, "#1e6e3e"),
            ("Enable Pos", 3, "#1e3e6e"),
            ("Disable",    0, "#7d2e2e"),
            ("Clear Errors", -1, "#6e5e1e")]:
        b = QtWidgets.QPushButton(label)
        b.setStyleSheet(
            f"QPushButton{{background:{col};color:white;font-family:Consolas;"
            f"font-size:10px;border-radius:3px;padding:5px 10px}}"
            f"QPushButton:hover{{background:{col}bb}}")
        _cm = ctrl_mode
        if _cm == 0:
            b.clicked.connect(lambda: cmd.send_odrive_disable())
        elif _cm == -1:
            b.clicked.connect(lambda: cmd.send_odrive_clear_errors())
        else:
            b.clicked.connect(lambda _, m=_cm: cmd.send_odrive_enable(m))
        od_enable_row.addWidget(b)
    od_enable_row.addStretch()
    od_ctrl_vb.addLayout(od_enable_row)

    od_ctrl_vb.addSpacing(6)

    # Velocity control
    od_ctrl_vb.addWidget(_od_lbl("Velocity control  [turns/s]", dim=True))
    od_vel_row = QtWidgets.QHBoxLayout()
    od_vel_row.setSpacing(6)
    od_vel_spin = _od_spin(-5.0, 5.0, 0.25, 0.0, dec=2)
    od_vel_row.addWidget(_od_lbl("vel (t/s)"))
    od_vel_row.addWidget(od_vel_spin)

    od_vel_send = QtWidgets.QPushButton("Send")
    od_vel_send.setStyleSheet(_BTN_STYLE)
    od_vel_send.clicked.connect(lambda: cmd.send_odrive_velocity(od_vel_spin.value()))
    od_vel_row.addWidget(od_vel_send)

    od_vel_stop = QtWidgets.QPushButton("Stop (0)")
    od_vel_stop.setStyleSheet(_BTN_STYLE)
    od_vel_stop.clicked.connect(lambda: (od_vel_spin.setValue(0.0),
                                         cmd.send_odrive_velocity(0.0)))
    od_vel_row.addWidget(od_vel_stop)
    od_vel_row.addStretch()
    od_ctrl_vb.addLayout(od_vel_row)

    od_ctrl_vb.addSpacing(6)

    # Position control
    od_ctrl_vb.addWidget(_od_lbl("Position control  [turns from zero]", dim=True))
    od_pos_row = QtWidgets.QHBoxLayout()
    od_pos_row.setSpacing(6)
    od_pos_spin = _od_spin(-50.0, 50.0, 0.25, 0.0, dec=3)
    od_pos_row.addWidget(_od_lbl("pos (turns)"))
    od_pos_row.addWidget(od_pos_spin)

    od_pos_send = QtWidgets.QPushButton("Send")
    od_pos_send.setStyleSheet(_BTN_STYLE)
    od_pos_send.clicked.connect(lambda: cmd.send_odrive_position(od_pos_spin.value()))
    od_pos_row.addWidget(od_pos_send)

    od_pos_zero = QtWidgets.QPushButton("Go to 0")
    od_pos_zero.setStyleSheet(_BTN_STYLE)
    od_pos_zero.clicked.connect(lambda: (od_pos_spin.setValue(0.0),
                                          cmd.send_odrive_position(0.0)))
    od_pos_row.addWidget(od_pos_zero)
    od_pos_row.addStretch()
    od_ctrl_vb.addLayout(od_pos_row)

    od_ctrl_vb.addStretch()
    odrive_hbox.addWidget(od_ctrl, stretch=1)

    # ── Right panel: velocity time-series plot ────────────────────────────
    od_plot_w = pg.GraphicsLayoutWidget()
    od_plot_w.setBackground(BG_COLOR)
    od_plot_w.setMaximumWidth(420)
    od_pl = od_plot_w.addPlot()
    od_pl.setTitle('<span style="color:#e0e0e0;font-size:9pt">Wheel Position — Axis 0</span>')
    od_pl.setLabel("left", '<span style="color:#c8c8c8;font-size:9pt">turns</span>')
    od_pl.showGrid(x=True, y=True, alpha=0.20)
    od_pl.setXRange(-WINDOW_S, 0, padding=0.02)
    od_pl.disableAutoRange()
    od_ln_vel = od_pl.plot(pen=pg.mkPen('#60d0ff', width=1.4), name="wheel_vel_avg")
    odrive_hbox.addWidget(od_plot_w, stretch=1)

    ak45_tab = QtWidgets.QWidget()
    ak45_tab.setStyleSheet(f"background:{BG_COLOR};")
    tab_widget.addTab(ak45_tab, "AK45 Motors")

    ak45_hbox = QtWidgets.QHBoxLayout(ak45_tab)
    ak45_hbox.setContentsMargins(6, 6, 6, 6)
    ak45_hbox.setSpacing(8)

    # Live label dicts — filled per motor panel, read in _update
    _hip_lbl = {}   # key: (motor_idx, field) → QLabel

    def _make_motor_panel(motor_id: int, title: str):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(f"background:{BAR_COLOR};border-radius:4px;")
        vb = QtWidgets.QVBoxLayout(frame)
        vb.setContentsMargins(6, 6, 6, 6)
        vb.setSpacing(4)

        hdr = QtWidgets.QLabel(title)
        hdr.setStyleSheet("color:white;font-family:Consolas;font-size:11px;font-weight:bold;")
        vb.addWidget(hdr)

        # Live labels
        for field in ("pos_deg", "vel_rads", "torque_nm", "temp_c"):
            row_w = QtWidgets.QWidget()
            row_h = QtWidgets.QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            dim_txt = {"pos_deg": "Pos (deg)", "vel_rads": "Vel (rad/s)",
                       "torque_nm": "Torque (N·m)", "temp_c": "Temp (°C)"}[field]
            dim_lbl = QtWidgets.QLabel(dim_txt)
            dim_lbl.setStyleSheet(_LBL_DIM)
            val_lbl = QtWidgets.QLabel("--")
            val_lbl.setStyleSheet(_LBL_LIVE)
            _hip_lbl[(motor_id, field)] = val_lbl
            row_h.addWidget(dim_lbl)
            row_h.addWidget(val_lbl)
            row_h.addStretch()
            vb.addWidget(row_w)

        # Small plot: position + torque
        mini_glw = pg.GraphicsLayoutWidget()
        mini_glw.setBackground(BG_COLOR)
        mini_glw.setMaximumHeight(100)
        mini_pl = mini_glw.addPlot()
        mini_pl.showGrid(x=True, y=True, alpha=0.15)
        mini_pl.setXRange(-WINDOW_S, 0, padding=0.02)
        mini_pl.disableAutoRange()
        mini_ln_pos = mini_pl.plot(pen=pg.mkPen('#00e5ff', width=1.2), name="pos")
        mini_ln_tau = mini_pl.plot(pen=pg.mkPen('#ffa040', width=1.2), name="τ")
        _hip_lbl[(motor_id, '_mini_pos')] = mini_ln_pos
        _hip_lbl[(motor_id, '_mini_tau')] = mini_ln_tau
        vb.addWidget(mini_glw, stretch=1)

        # Enable / Disable / Zero buttons
        btn_row = QtWidgets.QHBoxLayout()
        for btn_txt, hip_cmd_val, col in [
                ("Enable",  1, "#2e7d32"), ("Disable", 0, "#7d2e2e"), ("Zero", 2, "#4a4a2e")]:
            b = QtWidgets.QPushButton(btn_txt)
            b.setStyleSheet(
                f"QPushButton{{background:{col};color:white;font-family:Consolas;"
                f"font-size:10px;border-radius:3px;padding:3px 8px}}"
                f"QPushButton:hover{{background:{col}cc}}")
            _hcv = hip_cmd_val
            _mid = motor_id
            b.clicked.connect(lambda _, m=_mid, c=_hcv: cmd.send_hip(m, c))
            btn_row.addWidget(b)
        vb.addLayout(btn_row)

        # MIT cmd spinboxes + Send
        mit_row = QtWidgets.QHBoxLayout()
        mit_row.setSpacing(4)

        def _spin(lo, hi, step, val, dec=3):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setSingleStep(step)
            s.setValue(val); s.setDecimals(dec)
            s.setStyleSheet(_SPIN_STYLE)
            s.setFixedWidth(75)
            return s

        sp_p  = _spin(-716, 716, 1.0,  0.0, 1)  # degrees
        sp_kp = _spin(0, 500, 5.0,    50.0, 1)
        sp_kd = _spin(0,   5, 0.1,     3.0, 2)
        sp_t  = _spin(-8,  8, 0.1,     0.0, 2)

        for lbl_txt, sp in [("p°", sp_p), ("Kp", sp_kp), ("Kd", sp_kd), ("τff", sp_t)]:
            lbl = QtWidgets.QLabel(lbl_txt)
            lbl.setStyleSheet(_LBL_DIM)
            mit_row.addWidget(lbl)
            mit_row.addWidget(sp)

        btn_send = QtWidgets.QPushButton("Send")
        btn_send.setStyleSheet(_BTN_STYLE)
        _mid = motor_id
        btn_send.clicked.connect(lambda _, m=_mid: cmd.send_hip(
            m, 3,
            math.radians(sp_p.value()), 0.0,
            sp_kp.value(), sp_kd.value(), sp_t.value()))
        mit_row.addWidget(btn_send)
        mit_row.addStretch()
        vb.addLayout(mit_row)

        return frame

    panel_L = _make_motor_panel(1, "Hip L (ID 0x41)")
    ak45_hbox.addWidget(panel_L, stretch=1)

    # Centre column: Both buttons
    centre = QtWidgets.QWidget()
    centre.setFixedWidth(110)
    centre_vb = QtWidgets.QVBoxLayout(centre)
    centre_vb.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
    centre_vb.setSpacing(6)
    for btn_txt, hip_cmd_val, col in [
            ("Enable\nBoth", 1, "#2e7d32"), ("Disable\nBoth", 0, "#7d2e2e"), ("Zero\nBoth", 2, "#4a4a2e")]:
        b = QtWidgets.QPushButton(btn_txt)
        b.setStyleSheet(
            f"QPushButton{{background:{col};color:white;font-family:Consolas;"
            f"font-size:10px;border-radius:3px;padding:6px 8px}}"
            f"QPushButton:hover{{background:{col}cc}}")
        _hcv = hip_cmd_val
        b.clicked.connect(lambda _, c=_hcv: cmd.send_hip(3, c))
        centre_vb.addWidget(b)
    ak45_hbox.addWidget(centre)

    panel_R = _make_motor_panel(2, "Hip R (ID 0x42)")
    ak45_hbox.addWidget(panel_R, stretch=1)

    vbox.addWidget(tab_widget)

    # ── Style helpers ────────────────────────────────────────────────────────
    TICK_FONT = QtGui.QFont("Consolas", 9)
    TICK_PEN  = pg.mkColor(TICK_COLOR)
    _DASH     = QtCore.Qt.PenStyle.DashLine
    W         = LINE_WIDTH

    def _p(row, col, ttl, ylabel):
        pl = glw.addPlot(row=row, col=col)
        pl.setTitle(
            f'<span style="color:#e0e0e0;font-size:9pt;font-weight:600">{ttl}</span>')
        pl.setLabel(
            "left",
            f'<span style="color:#c8c8c8;font-size:9pt">{ylabel}</span>')
        pl.showGrid(x=True, y=True, alpha=0.20)
        for ax_name in ("left", "bottom"):
            ax = pl.getAxis(ax_name)
            ax.setTextPen(TICK_PEN)
            ax.setPen(pg.mkPen('#555'))
            ax.setStyle(tickFont=TICK_FONT)
        pl.setXRange(-WINDOW_S, 0, padding=0.02)
        pl.disableAutoRange(axis='y')
        return pl

    def _leg(pl, ncols=1):
        leg = pl.addLegend(offset=(6, 6), verSpacing=-4, colCount=ncols)
        leg.setBrush(pg.mkBrush(18, 18, 36, 210))
        leg.setPen(pg.mkPen('#444'))
        leg.setLabelTextColor(pg.mkColor('#cccccc'))
        return leg

    def _limits(pl, val, color='#ff4444'):
        for sign in (+1, -1):
            anchor = (0.05, 1.1) if sign > 0 else (0.05, -0.1)
            il = pg.InfiniteLine(
                pos=sign * val, angle=0,
                pen=pg.mkPen(color, width=1.2, style=_DASH),
                label=f'{"+" if sign > 0 else "−"}{val:.1f}',
                labelOpts={"color": color, "anchors": [anchor, anchor]})
            pl.addItem(il)

    # ── Row 0: Pitch | Pitch Rate ────────────────────────────────────────────
    p_pitch = _p(0, 0, "Pitch", "deg")
    _leg(p_pitch)
    ln_pitch     = p_pitch.plot(pen=pg.mkPen('#60d0ff', width=W), name="pitch")
    ln_pitch_ref = p_pitch.plot(pen=pg.mkPen('#ff6060', width=W, style=_DASH), name="θ_ref")

    p_prate = _p(0, 1, "Pitch Rate", "deg/s")
    _leg(p_prate)
    ln_prate = p_prate.plot(pen=pg.mkPen('#ffa040', width=W), name="pitch rate")

    # ── Row 1: Velocity | Wheel Vel Avg ──────────────────────────────────────
    p_vel = _p(1, 0, "Velocity", "rad/s")
    _leg(p_vel)
    ln_vel  = p_vel.plot(pen=pg.mkPen('#60d0ff', width=W), name="wheel_vel")
    ln_vcmd = p_vel.plot(pen=pg.mkPen('#ff6060', width=W, style=_DASH), name="v_cmd")

    p_control = _p(1, 1, "Control Signals", "N·m / rad")
    _leg(p_control)
    ln_tau_sym = p_control.plot(pen=pg.mkPen('#60d0ff', width=W), name="τ_sym")
    ln_tau_yaw = p_control.plot(pen=pg.mkPen('#ff6060', width=W), name="τ_yaw")
    ln_theta   = p_control.plot(pen=pg.mkPen('#80ff80', width=W, style=_DASH), name="θ_ref")

    # ── Row 2: Hip Position | Roll ───────────────────────────────────────────
    p_hip = _p(2, 0, "Hip Position", "deg")
    _leg(p_hip)
    ln_hip_L = p_hip.plot(pen=pg.mkPen('#00e5ff', width=W), name="L")
    ln_hip_R = p_hip.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    p_roll = _p(2, 1, "Roll", "deg")
    _leg(p_roll)
    ln_roll = p_roll.plot(pen=pg.mkPen('#ffa040', width=W), name="roll")

    # ── Row 3: Wheel Torque | Hip Torque ─────────────────────────────────────
    p_tau = _p(3, 0, "Wheel Torque", "N·m")
    _leg(p_tau)
    _limits(p_tau, WHEEL_TORQUE_MAX)
    p_tau.setYRange(-WHEEL_TORQUE_MAX * 1.1, WHEEL_TORQUE_MAX * 1.1, padding=0)
    ln_tau_L = p_tau.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_tau_R = p_tau.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    p_htau = _p(3, 1, "Hip Torque", "N·m")
    _leg(p_htau)
    _limits(p_htau, HIP_TORQUE_MAX)
    p_htau.setYRange(-HIP_TORQUE_MAX * 1.1, HIP_TORQUE_MAX * 1.1, padding=0)
    ln_htau_L = p_htau.plot(pen=pg.mkPen('#60d0ff', width=W), name="L")
    ln_htau_R = p_htau.plot(pen=pg.mkPen('#80ff80', width=W), name="R")

    # ── Row 4: Loop Timing | (reserved) ─────────────────────────────────────
    p_dt = _p(4, 0, "Loop dt", "µs")
    _leg(p_dt)
    ln_dt = p_dt.plot(pen=pg.mkPen('#60d0ff', width=W), name="dt_us")

    p_sine = _p(4, 1, "Debug Sine (rate check)", "—")
    _leg(p_sine)
    ln_sine = p_sine.plot(pen=pg.mkPen('#60d0ff', width=W), name="sine")

    # ── Mouse hover ──────────────────────────────────────────────────────────
    named_plots = [
        ("Pitch", p_pitch), ("Pitch Rate", p_prate),
        ("Velocity", p_vel), ("Control", p_control),
        ("Hip", p_hip), ("Roll", p_roll),
        ("Wheel Torque", p_tau), ("Hip Torque", p_htau),
        ("Loop dt", p_dt), ("Debug Sine", p_sine),
    ]

    def _on_mouse(evt):
        pos = evt[0]
        for name, pl in named_plots:
            if pl.sceneBoundingRect().contains(pos):
                mp_ = pl.vb.mapSceneToView(pos)
                hover_lbl.setText(f"  {name}   y = {mp_.y():.3f}")
                return
        hover_lbl.setText("")

    _proxy = pg.SignalProxy(
        glw.scene().sigMouseMoved, rateLimit=60, slot=_on_mouse)

    # ── Position window ──────────────────────────────────────────────────────
    try:
        screen = app.primaryScreen()
        rect   = screen.geometry()
        half_w = rect.width() // 2
        main_win.setGeometry(rect.x() + 90, rect.y() + 35,
                             half_w - 100, rect.height() - 70)
    except Exception:
        main_win.resize(960, 1000)
    main_win.show()

    # ── Ring buffers ─────────────────────────────────────────────────────────
    t_buf         = deque(maxlen=MAXLEN)
    pitch_buf     = deque(maxlen=MAXLEN)
    pitch_ref_buf = deque(maxlen=MAXLEN)
    prate_buf     = deque(maxlen=MAXLEN)
    vel_buf       = deque(maxlen=MAXLEN)
    vcmd_buf      = deque(maxlen=MAXLEN)
    tau_sym_buf   = deque(maxlen=MAXLEN)
    tau_yaw_buf   = deque(maxlen=MAXLEN)
    theta_buf     = deque(maxlen=MAXLEN)
    hip_q_L_buf   = deque(maxlen=MAXLEN)
    hip_q_R_buf   = deque(maxlen=MAXLEN)
    roll_buf      = deque(maxlen=MAXLEN)
    yaw_buf       = deque(maxlen=MAXLEN)
    tau_wL_buf    = deque(maxlen=MAXLEN)
    tau_wR_buf    = deque(maxlen=MAXLEN)
    tau_hL_buf    = deque(maxlen=MAXLEN)
    tau_hR_buf    = deque(maxlen=MAXLEN)
    dt_buf        = deque(maxlen=MAXLEN)
    sine_buf      = deque(maxlen=MAXLEN)
    wpos_L_buf    = deque(maxlen=MAXLEN)   # wheel_pos_L [turns]
    wvel_L_buf    = deque(maxlen=MAXLEN)   # wheel_vel_L [turns/s]
    wheel_ok_buf  = deque(maxlen=MAXLEN)   # bool as 0/1
    odrive_state_buf = deque(maxlen=MAXLEN)  # axis_state from heartbeat
    odrive_error_buf = deque(maxlen=MAXLEN)  # has_error flag from heartbeat

    all_bufs = [t_buf, pitch_buf, pitch_ref_buf, prate_buf, vel_buf, vcmd_buf,
                tau_sym_buf, tau_yaw_buf, theta_buf,
                hip_q_L_buf, hip_q_R_buf, roll_buf, yaw_buf,
                tau_wL_buf, tau_wR_buf, tau_hL_buf, tau_hR_buf, dt_buf, sine_buf,
                wpos_L_buf, wvel_L_buf, wheel_ok_buf]

    _pkt_count = [0]
    _last_stat = [0.0]
    _start_time = [None]  # first packet monotonic time
    _last_mode = [0]

    # ── 60 Hz update ─────────────────────────────────────────────────────────
    _MIN_Y_SPAN = 2.0
    _fixed_range_plots = {id(p_tau), id(p_htau)}

    def _update():
        # Drain queue
        while True:
            try:
                item = data_q.get_nowait()
            except Empty:
                break

            (ts_ms, mode, pitch, pitch_rate, roll, yaw,
             wheel_vel_avg, v_cmd, theta_ref,
             tau_sym, tau_yaw,
             tau_wheel_L, tau_wheel_R,
             hip_q_L, tau_hip_L, tau_hip_R, hip_q_R,
             dt_us, debug_sine,
             wheel_pos_L, wheel_vel_L, status_flags) = item
            wheel_ok          = bool(status_flags & 0x01)
            imu_ok            = bool(status_flags & 0x02)
            odrive_axis_state = (status_flags >> 2) & 0x0F
            odrive_has_error  = bool(status_flags & 0x40)

            # Convert timestamp to seconds
            t_s = ts_ms / 1000.0
            if _start_time[0] is None:
                _start_time[0] = t_s

            t_buf.append(t_s)
            pitch_buf.append(math.degrees(pitch))
            pitch_ref_buf.append(math.degrees(theta_ref))
            prate_buf.append(math.degrees(pitch_rate))
            vel_buf.append(wheel_vel_avg)
            vcmd_buf.append(v_cmd)
            tau_sym_buf.append(tau_sym)
            tau_yaw_buf.append(tau_yaw)
            theta_buf.append(math.degrees(theta_ref))
            hip_q_L_buf.append(math.degrees(hip_q_L))
            hip_q_R_buf.append(math.degrees(hip_q_R))
            roll_buf.append(math.degrees(roll))
            yaw_buf.append(math.degrees(yaw))
            tau_wL_buf.append(tau_wheel_L)
            tau_wR_buf.append(tau_wheel_R)
            tau_hL_buf.append(tau_hip_L)
            tau_hR_buf.append(tau_hip_R)
            dt_buf.append(dt_us)
            sine_buf.append(debug_sine)
            wpos_L_buf.append(wheel_pos_L / (2 * math.pi))   # rad → turns
            wvel_L_buf.append(wheel_vel_L / (2 * math.pi))   # rad/s → turns/s
            wheel_ok_buf.append(1.0 if wheel_ok else 0.0)
            odrive_state_buf.append(odrive_axis_state)
            odrive_error_buf.append(1.0 if odrive_has_error else 0.0)

            _last_mode[0] = mode
            _pkt_count[0] += 1

        if len(t_buf) < 2:
            return

        # Update robot IP in command sender when discovered
        if receiver.robot_ip and not cmd.robot_ip:
            cmd.robot_ip = receiver.robot_ip

        # Status labels — throttled to 3 Hz
        now = time.perf_counter()
        if now - _last_stat[0] >= 1.0 / 3.0:
            age = time.monotonic() - receiver.last_rx_time if receiver.last_rx_time else 999
            if age < 1.0:
                lbl_conn.setText(f"Connection: OK ({receiver.robot_ip})")
                lbl_conn.setStyleSheet(_SL + "color:#80ff80;")
                main_win.setWindowTitle(f"Robot Dashboard — {receiver.robot_ip}")
            else:
                lbl_conn.setText("Connection: LOST" if receiver.last_rx_time else "Connection: Waiting…")
                lbl_conn.setStyleSheet(_SL + "color:#ff6060;")

            lbl_mode.setText(f"Mode: {MODE_NAMES.get(_last_mode[0], str(_last_mode[0]))}")

            lbl_dt.setText(f"dt: {dt_buf[-1]:.0f} µs")
            lbl_rate.setText(f"Pkts: {_pkt_count[0]}")
            _last_stat[0] = now

        # Compute visible window
        tb   = np.array(t_buf)
        t_now = float(tb[-1])
        t0    = max(tb[0], t_now - WINDOW_S)
        idx   = int(np.searchsorted(tb, t0))
        xw    = tb[idx:] - t_now  # seconds relative to now

        def _a(buf):
            return np.array(buf)[idx:]

        ln_pitch.setData(xw, _a(pitch_buf))
        ln_pitch_ref.setData(xw, _a(pitch_ref_buf))
        ln_prate.setData(xw, _a(prate_buf))
        ln_vel.setData(xw, _a(vel_buf))
        ln_vcmd.setData(xw, _a(vcmd_buf))
        ln_tau_sym.setData(xw, _a(tau_sym_buf))
        ln_tau_yaw.setData(xw, _a(tau_yaw_buf))
        ln_theta.setData(xw, _a(theta_buf))
        ln_hip_L.setData(xw, _a(hip_q_L_buf))
        ln_hip_R.setData(xw, _a(hip_q_R_buf))
        ln_roll.setData(xw, _a(roll_buf))
        ln_tau_L.setData(xw, _a(tau_wL_buf))
        ln_tau_R.setData(xw, _a(tau_wR_buf))
        ln_htau_L.setData(xw, _a(tau_hL_buf))
        ln_htau_R.setData(xw, _a(tau_hR_buf))
        ln_dt.setData(xw, _a(dt_buf))
        ln_sine.setData(xw, _a(sine_buf))

        # ── ODrive tab live status ──
        _ODRIVE_AXIS_STATES = {
            0: "UNDEFINED", 1: "IDLE", 3: "FULL_CALIB",
            4: "MOTOR_CALIB", 6: "ENC_INDEX_SEARCH",
            7: "ENC_OFFSET_CALIB", 8: "CLOSED_LOOP", 11: "ENC_DIR_FIND",
        }
        if wpos_L_buf:
            pos_t    = wpos_L_buf[-1]
            vel_t    = wvel_L_buf[-1]
            ok       = bool(wheel_ok_buf[-1]) if wheel_ok_buf else False
            ax_st    = int(odrive_state_buf[-1]) if odrive_state_buf else 0
            has_err  = bool(odrive_error_buf[-1]) if odrive_error_buf else False
            ax_str   = _ODRIVE_AXIS_STATES.get(ax_st, f"STATE_{ax_st}")
            err_str  = "  ⚠ ERROR" if has_err else ""
            enc_str  = "OK" if ok else "NO SIGNAL"
            # colour: green=closed loop, amber=idle+no error, red=error or unknown
            ax_col = "#80ff80" if ax_st == 8 else ("#ff6060" if has_err else ("#ffa040" if ax_st == 1 else "#ff6060"))
            od_status_lbl.setStyleSheet(f"color:{ax_col};font-family:Consolas;font-size:11px;")
            od_status_lbl.setText(
                f"Axis: {ax_str}{err_str}   Enc: {enc_str}   "
                f"pos={pos_t:+.4f} t   vel={vel_t:+.4f} t/s")

        od_ln_vel.setData(xw, _a(wpos_L_buf))
        pos_arr = _a(wpos_L_buf)
        if len(pos_arr) > 0:
            lo, hi = float(np.min(pos_arr)), float(np.max(pos_arr))
            span = max(hi - lo, 0.5)
            mid  = (lo + hi) / 2
            od_pl.setYRange(mid - span * 0.6, mid + span * 0.6, padding=0.05)
        od_pl.setLabel("left", '<span style="color:#c8c8c8;font-size:9pt">turns</span>')

        # ── AK45 tab live labels + mini plots ──
        for motor_id, pos_buf, tau_buf in [
                (1, hip_q_L_buf, tau_hL_buf),
                (2, hip_q_R_buf, tau_hR_buf)]:
            if pos_buf:
                _hip_lbl[(motor_id, 'pos_deg')].setText(f"{pos_buf[-1]:+.1f}")
            if tau_buf:
                _hip_lbl[(motor_id, 'torque_nm')].setText(f"{tau_buf[-1]:+.2f}")
            mini_pos = _a(pos_buf)
            mini_tau = _a(tau_buf)
            _hip_lbl[(motor_id, '_mini_pos')].setData(xw, mini_pos)
            _hip_lbl[(motor_id, '_mini_tau')].setData(xw, mini_tau)
            if len(mini_pos) > 0 or len(mini_tau) > 0:
                all_y = []
                if len(mini_pos) > 0: all_y.extend([float(np.min(mini_pos)), float(np.max(mini_pos))])
                if len(mini_tau) > 0: all_y.extend([float(np.min(mini_tau)), float(np.max(mini_tau))])
                lo, hi = min(all_y), max(all_y)
                span = max(hi - lo, _MIN_Y_SPAN)
                mid = (lo + hi) / 2
                _hip_lbl[(motor_id, '_mini_pos')].getViewBox().setYRange(
                    mid - span * 0.6, mid + span * 0.6, padding=0)

        # ── Update 3D IMU axes ──
        if len(pitch_buf) > 0:
            p_rad = math.radians(pitch_buf[-1])
            r_rad = math.radians(roll_buf[-1])
            y_rad = math.radians(yaw_buf[-1]) if len(yaw_buf) > 0 else 0.0
            cp, sp = math.cos(p_rad), math.sin(p_rad)
            cr, sr = math.cos(r_rad), math.sin(r_rad)
            cy, sy = math.cos(y_rad), math.sin(y_rad)
            # R = R_yaw(Z) @ R_pitch(Y) @ R_roll(X)
            R = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp,     cp * sr,                cp * cr               ],
            ])
            for vec0, color, line_item in axis_gl_lines:
                rotated = R @ vec0
                pts = np.array([[0, 0, 0], rotated.tolist()], dtype=np.float32)
                line_item.setData(pos=pts)
            imu_readout.setText(
                f"P: {pitch_buf[-1]:+.1f}\u00b0  R: {roll_buf[-1]:+.1f}\u00b0  Y: {yaw_buf[-1]:+.1f}\u00b0" if len(yaw_buf) > 0 else
                f"Pitch: {pitch_buf[-1]:+.1f}\u00b0  Roll: {roll_buf[-1]:+.1f}\u00b0")

        # Auto-fit Y range (except fixed-range torque plots)
        for _, pl in named_plots:
            if id(pl) in _fixed_range_plots:
                continue
            d_lo, d_hi = float('inf'), float('-inf')
            for item in pl.listDataItems():
                yd = item.yData
                if yd is not None and len(yd) > 0:
                    d_lo = min(d_lo, float(np.min(yd)))
                    d_hi = max(d_hi, float(np.max(yd)))
            if d_lo > d_hi:
                continue
            span = d_hi - d_lo
            if span < _MIN_Y_SPAN:
                mid = (d_lo + d_hi) / 2
                d_lo = mid - _MIN_Y_SPAN / 2
                d_hi = mid + _MIN_Y_SPAN / 2
            margin = max(0.05 * span, 0.1)
            pl.setYRange(d_lo - margin, d_hi + margin, padding=0)

    timer = QtCore.QTimer()
    timer.timeout.connect(_update)
    timer.start(16)  # ~60 Hz

    # ── Start receiver + initial ping ────────────────────────────────────────
    receiver.start()
    if not serial_port:
        if robot_ip:
            cmd.send_ping(robot_ip)
        else:
            cmd.send_broadcast_ping()

    if serial_port:
        print(f"[Dashboard] Listening for telemetry on Serial: {receiver.port_name}")
        print("[Dashboard] Commands: enabled via serial (0xBBCC framing)")
        print("[Dashboard] ODrive tab: Enable=Mode1, Disable=Mode0, τ sends CMD_DRIVE")
    else:
        print(f"[Dashboard] Listening for telemetry on UDP :{TELEMETRY_PORT}")
        print(f"[Dashboard] Sending commands to UDP :{COMMAND_PORT}")
        if robot_ip:
            print(f"[Dashboard] Target robot: {robot_ip}")
        else:
            print("[Dashboard] Broadcast ping sent — waiting for robot…")

    # ── Run Qt event loop ────────────────────────────────────────────────────
    ret = app.exec()
    receiver.stop()
    sys.exit(ret)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Robot telemetry dashboard")
    parser.add_argument("--robot", type=str, default=None,
                        help="Robot IP address (default: auto-discover via broadcast)")
    parser.add_argument("--serial", type=str, default=None, metavar="PORT",
                        help="USB-Serial port (e.g. COM5, /dev/ttyACM0, or 'auto')")
    args = parser.parse_args()

    if args.serial and args.robot:
        print("ERROR: --serial and --robot are mutually exclusive")
        sys.exit(1)

    # Default: auto-detect serial port (use --robot to force WiFi mode)
    serial_port = args.serial if args.serial else ("auto" if not args.robot else None)

    run_dashboard(robot_ip=args.robot, serial_port=serial_port)


if __name__ == "__main__":
    main()
