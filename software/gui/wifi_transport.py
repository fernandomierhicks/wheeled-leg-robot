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

        # PacketDecoder is created lazily in run() so it lives on the right thread
        self._decoder = None

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, frame: bytes):
        """Write a CommLink frame to the ESP32 via TCP (no-op if not connected)."""
        if not self._esp_ip:
            return
        with self._tcp_lock:
            try:
                if self._tcp_sock is None:
                    self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._tcp_sock.settimeout(1.0)
                    self._tcp_sock.connect((self._esp_ip, TCP_PORT))
                self._tcp_sock.sendall(frame)
            except Exception:
                self._close_tcp()

    # ── Background thread ─────────────────────────────────────────────────────

    def run(self):
        from flash_monitor import PacketDecoder
        from source_manager import SourceManager

        self._decoder = PacketDecoder("wifi")
        self._decoder.packet_decoded.connect(lambda _: None)  # keep signal wired

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.settimeout(TIMEOUT_S)
        try:
            udp.bind(("", UDP_PORT))
        except Exception as e:
            print(f"[WifiTransport] UDP bind failed: {e}")
            return

        sm = SourceManager.instance()

        while True:
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
