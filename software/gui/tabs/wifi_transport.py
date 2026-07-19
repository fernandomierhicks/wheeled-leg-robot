"""wifi_transport.py — UDP telemetry receiver + TCP command sender for ESP32 WiFi mode.

Telemetry path: ESP32 broadcasts CommLink frames as UDP datagrams on port 5005.
                WifiTransport listens, feeds bytes into PacketDecoder, and
                advertises "wifi" to SourceManager when packets are flowing.

Command path:   When a UDP packet arrives the sender IP is recorded.
                send() opens a TCP connection to <ip>:5006 (lazily, on first call)
                and writes CommLink command frames — ESP32 forwards them to Teensy.
"""

import socket
import threading
import time

from PyQt6.QtCore import QThread

UDP_PORT     = 5005
TCP_PORT     = 5006
TIMEOUT_S    = 3.0   # declare WiFi dead if no UDP packet in this many seconds


class WifiTransport(QThread):
    _instance: "WifiTransport | None" = None

    @classmethod
    def instance(cls) -> "WifiTransport":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._esp_ip:     str | None   = None
        self._tcp_sock:   socket.socket | None = None
        self._tcp_lock    = threading.Lock()
        self._connected   = False
        # Reconnect backoff (Phase 4, UARTplat.md): a dead ESP32 used to get a
        # fresh connect() attempt on every send() call (up to 10 Hz from the GUI
        # ping), each blocking up to the 1 s socket timeout. Skip attempts until
        # _next_connect_time; reset on a successful send or a fresh ESP32 contact.
        self._fail_count      = 0
        self._next_connect_time = 0.0

        # PacketDecoder is created lazily in run() so it lives on the right thread
        self._decoder = None

        # Set by stop() so run()'s loop can exit before Qt starts tearing down
        # objects during app.quit() — without this, a background emit into an
        # already-deleted TelemetryBus/PacketDecoder raises "wrapped C/C++
        # object ... has been deleted" (or segfaults outright). A human closing
        # the GUI window never raced this; AutomationRunner's programmatic
        # app.quit() does, every time.
        self._stop_requested = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def esp_ip(self) -> str | None:
        """ESP32's current IP, learned from the source address of the last UDP
        telemetry datagram. None until the first packet arrives."""
        return self._esp_ip

    def stop(self, wait_ms: int = 3500):
        """Ask run()'s loop to exit and block until it does (or wait_ms elapses).
        Call before app.quit() in any programmatic-shutdown path — see
        _stop_requested for why this matters."""
        self._stop_requested.set()
        self.wait(wait_ms)

    def send(self, frame: bytes):
        """Write a CommLink frame to the ESP32 via TCP (no-op if not connected)."""
        if not self._esp_ip:
            return
        with self._tcp_lock:
            try:
                if self._tcp_sock is None:
                    if time.monotonic() < self._next_connect_time:
                        return  # backing off after a recent failed connect
                    self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._tcp_sock.settimeout(1.0)
                    self._tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    self._tcp_sock.connect((self._esp_ip, TCP_PORT))
                self._tcp_sock.sendall(frame)
                self._fail_count = 0
            except Exception:
                self._close_tcp()
                self._next_connect_time = time.monotonic() + min(5.0, 0.5 * 2 ** self._fail_count)
                self._fail_count += 1

    def force_reconnect(self):
        """Close the TCP command socket now so the next send() reopens a fresh
        connection — used by automation scenarios to reproduce the
        '[TCP] client:'-accept-adjacent crash pattern (repeated connect/
        disconnect churn) without touching the UDP telemetry listener."""
        with self._tcp_lock:
            self._close_tcp()
            self._fail_count        = 0
            self._next_connect_time = 0.0

    # ── Background thread ─────────────────────────────────────────────────────

    def run(self):
        from .flash_monitor import PacketDecoder
        from .source_manager import SourceManager

        self._decoder = PacketDecoder("wifi")
        self._decoder.packet_decoded.connect(lambda _: None)  # keep signal wired

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # ~1 MB: survive GUI stalls/bursts
        udp.settimeout(TIMEOUT_S)
        try:
            udp.bind(("", UDP_PORT))
        except Exception as e:
            print(f"[WifiTransport] UDP bind failed: {e}")
            return

        sm = SourceManager.instance()

        while not self._stop_requested.is_set():
            try:
                data, (src_ip, _) = udp.recvfrom(4096)
            except socket.timeout:
                if self._connected:
                    self._connected = False
                    self._esp_ip = None
                    with self._tcp_lock:
                        self._close_tcp()
                    sm._on_released("wifi")
                continue
            except Exception as e:
                print(f"[WifiTransport] UDP error: {e}")
                break

            if not self._connected or src_ip != self._esp_ip:
                self._esp_ip = src_ip
                with self._tcp_lock:
                    self._close_tcp()   # reset TCP so it reconnects to new IP
                    self._fail_count        = 0
                    self._next_connect_time = 0.0
                if not self._connected:
                    self._connected = True
                    sm._on_opened("wifi")

            self._decoder.feed(data)

        udp.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _close_tcp(self):
        """Close TCP socket. Must be called with _tcp_lock held."""
        if self._tcp_sock is not None:
            try:
                self._tcp_sock.close()
            except Exception:
                pass
            self._tcp_sock = None
