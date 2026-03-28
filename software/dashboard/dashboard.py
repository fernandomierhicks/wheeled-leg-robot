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

# Telemetry packet: '<' = little-endian, I=uint32, B=uint8, 16f=16 floats
TELEM_FMT  = '<IB16f'
TELEM_SIZE = struct.calcsize(TELEM_FMT)  # 69 bytes
assert TELEM_SIZE == 69

# Command type IDs
CMD_DRIVE = 1
CMD_MODE  = 2
CMD_GAIN  = 3
CMD_PING  = 4

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
            #        tau_wheel_L, tau_wheel_R, hip_q_avg, tau_hip_L, tau_hip_R, dt_us,
            #        debug_sine)
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
SERIAL_FRAME_SIZE = 2 + TELEM_SIZE + 1  # sync(2) + packet(69) + checksum(1) = 72


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
    """Background thread: receives framed telemetry packets from USB-UART."""

    def __init__(self, data_q: Queue, port: str, baud: int = 1000000):
        super().__init__(daemon=True)
        self.data_q = data_q
        self.port_name = port
        self.baud = baud
        self.running = True
        self.robot_ip = "USB-Serial"   # fake IP for status display compatibility
        self.last_rx_time = 0.0
        self._ser = None

    def run(self):
        try:
            self._ser = serial.Serial(self.port_name, self.baud, timeout=0.5)
        except serial.SerialException as e:
            print(f"[Serial] Failed to open {self.port_name}: {e}")
            return

        print(f"[Serial] Listening on {self.port_name} @ {self.baud} baud")
        buf = bytearray()

        while self.running:
            try:
                chunk = self._ser.read(max(1, self._ser.in_waiting))
            except serial.SerialException:
                break
            if not chunk:
                continue

            buf.extend(chunk)

            # Scan for sync + complete frame
            while len(buf) >= SERIAL_FRAME_SIZE:
                idx = buf.find(SERIAL_SYNC)
                if idx < 0:
                    # No sync found — keep last byte (could be partial 0xAA)
                    buf = buf[-1:]
                    break
                if idx > 0:
                    # Discard bytes before sync
                    del buf[:idx]
                if len(buf) < SERIAL_FRAME_SIZE:
                    break  # wait for more data

                # Extract packet and checksum
                pkt_bytes = bytes(buf[2:2 + TELEM_SIZE])
                rx_ck = buf[2 + TELEM_SIZE]

                # Verify XOR checksum
                xor_ck = 0
                for b in pkt_bytes:
                    xor_ck ^= b

                if xor_ck != rx_ck:
                    # Bad checksum — skip this sync and search for next
                    del buf[:1]
                    continue

                # Valid frame — consume it
                del buf[:SERIAL_FRAME_SIZE]
                self.last_rx_time = time.monotonic()

                vals = struct.unpack(TELEM_FMT, pkt_bytes)
                try:
                    self.data_q.put_nowait(vals)
                except Exception:
                    pass  # drop if queue full

    def stop(self):
        self.running = False
        if self._ser and self._ser.is_open:
            self._ser.close()


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND SENDER
# ═════════════════════════════════════════════════════════════════════════════

class CommandSender:
    """Sends UDP command packets to the robot."""

    def __init__(self, robot_ip: str = None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_ip = robot_ip

    def send_ping(self, ip: str = None):
        target = ip or self.robot_ip
        if target:
            self.sock.sendto(struct.pack('<B', CMD_PING), (target, COMMAND_PORT))

    def send_drive(self, v_cmd: float, omega_cmd: float, hip_target: float):
        if not self.robot_ip:
            return
        pkt = struct.pack('<Bfff', CMD_DRIVE, v_cmd, omega_cmd, hip_target)
        self.sock.sendto(pkt, (self.robot_ip, COMMAND_PORT))

    def send_mode(self, mode: int):
        if not self.robot_ip:
            return
        pkt = struct.pack('<BB', CMD_MODE, mode)
        self.sock.sendto(pkt, (self.robot_ip, COMMAND_PORT))

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
        cmd = CommandSender(None)  # no commands over serial (read-only)
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
    ln_hip = p_hip.plot(pen=pg.mkPen('#60d0ff', width=W), name="hip_avg")

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
    hip_buf       = deque(maxlen=MAXLEN)
    roll_buf      = deque(maxlen=MAXLEN)
    yaw_buf       = deque(maxlen=MAXLEN)
    tau_wL_buf    = deque(maxlen=MAXLEN)
    tau_wR_buf    = deque(maxlen=MAXLEN)
    tau_hL_buf    = deque(maxlen=MAXLEN)
    tau_hR_buf    = deque(maxlen=MAXLEN)
    dt_buf        = deque(maxlen=MAXLEN)
    sine_buf      = deque(maxlen=MAXLEN)

    all_bufs = [t_buf, pitch_buf, pitch_ref_buf, prate_buf, vel_buf, vcmd_buf,
                tau_sym_buf, tau_yaw_buf, theta_buf, hip_buf, roll_buf, yaw_buf,
                tau_wL_buf, tau_wR_buf, tau_hL_buf, tau_hR_buf, dt_buf, sine_buf]

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
             hip_q_avg, tau_hip_L, tau_hip_R, dt_us,
             debug_sine) = item

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
            hip_buf.append(math.degrees(hip_q_avg))
            roll_buf.append(math.degrees(roll))
            yaw_buf.append(math.degrees(yaw))
            tau_wL_buf.append(tau_wheel_L)
            tau_wR_buf.append(tau_wheel_R)
            tau_hL_buf.append(tau_hip_L)
            tau_hR_buf.append(tau_hip_R)
            dt_buf.append(dt_us)
            sine_buf.append(debug_sine)

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
        ln_hip.setData(xw, _a(hip_buf))
        ln_roll.setData(xw, _a(roll_buf))
        ln_tau_L.setData(xw, _a(tau_wL_buf))
        ln_tau_R.setData(xw, _a(tau_wR_buf))
        ln_htau_L.setData(xw, _a(tau_hL_buf))
        ln_htau_R.setData(xw, _a(tau_hR_buf))
        ln_dt.setData(xw, _a(dt_buf))
        ln_sine.setData(xw, _a(sine_buf))

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
        print("[Dashboard] Commands: disabled (serial is read-only)")
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
