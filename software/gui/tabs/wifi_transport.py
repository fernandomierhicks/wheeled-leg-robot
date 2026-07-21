"""wifi_transport.py — UDP telemetry receiver + TCP command sender/receiver for ESP32 WiFi mode.

Telemetry path: ESP32 broadcasts CommLink frames as UDP datagrams on port 5005.
                WifiTransport listens, feeds bytes into PacketDecoder, and
                advertises "wifi" to SourceManager when packets are flowing.

Command path:   When a UDP packet arrives the sender IP is recorded.
                send() opens a TCP connection to <ip>:5006 (lazily, on first call)
                and writes CommLink command frames — ESP32 forwards them to Teensy.
                The ESP32 also sends every reply frame (PARAM_REPORT, ACK,
                LOG_INFO/LOG_DATA, CALIB_EVENT, TOF, comm_log LOG messages —
                everything except TELEM_A/B/WIFI_DIAG, which additionally go out
                over UDP too) back over this same TCP connection; run()'s select()
                loop below reads it. Found missing entirely (write-only socket,
                nothing ever called .recv() on it) via WiFi command-guard testing,
                UARTplat.md Phase 9 — before this fix, single-param reads, log
                downloads, calibration feedback, and ToF were silently
                non-functional over WiFi, invisible because regular telemetry
                (UDP-only) looked fine throughout.
"""

import select
import secrets
import socket
import threading
import time

from PyQt6.QtCore import QThread

UDP_PORT     = 5005
TCP_PORT     = 5006
DISCOVERY_PORT = 5007
TELEMETRY_TIMEOUT_S = 1.5
SESSION_RENEW_S = 0.8
SELECT_POLL_S = 0.2


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
        self._session_token = secrets.randbits(32) or 1
        self._last_claim_time = 0.0
        self._last_lease_ack_time = 0.0
        self._last_udp_time = 0.0
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

    @property
    def telemetry_age_ms(self) -> int | None:
        if not self._last_udp_time:
            return None
        return int((time.monotonic() - self._last_udp_time) * 1000)

    @property
    def lease_age_ms(self) -> int | None:
        if not self._last_lease_ack_time:
            return None
        return int((time.monotonic() - self._last_lease_ack_time) * 1000)

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
        # Separate decoder for the TCP command-response byte stream — TCP is a
        # continuous stream (unlike UDP's one-datagram-per-feed() calls), so it
        # needs its own independent parse buffer; reusing self._decoder would
        # interleave UDP datagram bytes and TCP stream bytes into the same _buf.
        # Same device string ("wifi") as self._decoder, so both land on the same
        # TelemetryBus/LogPacketBus via PacketDecoder's own SourceManager gating
        # (flash_monitor.py) — no separate wiring needed. The ESP32's uplink_task
        # sends every frame type over TCP unconditionally (not just command
        # replies) *in addition to* sending TELEM_A/B/WIFI_DIAG over UDP — so
        # without skip_emit_types those three would arrive and get delivered
        # twice (once per transport), doubling the apparent telemetry rate.
        self._tcp_decoder = PacketDecoder("wifi", skip_emit_types=frozenset({0x10, 0x11, 0x14, 0x15}))
        self._tcp_decoder.packet_decoded.connect(lambda _: None)

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # ~1 MB: survive GUI stalls/bursts
        try:
            udp.bind(("", UDP_PORT))
        except Exception as e:
            print(f"[WifiTransport] UDP bind failed: {e}")
            return

        sm = SourceManager.instance()

        while not self._stop_requested.is_set():
            now = time.monotonic()
            if now - self._last_claim_time >= SESSION_RENEW_S:
                claim = f"WLR_CLAIM_V1 {self._session_token:08X}".encode("ascii")
                try:
                    udp.sendto(claim, ("255.255.255.255", DISCOVERY_PORT))
                    if self._esp_ip:
                        udp.sendto(claim, (self._esp_ip, DISCOVERY_PORT))
                except OSError:
                    pass
                self._last_claim_time = now

            with self._tcp_lock:
                tcp_sock = self._tcp_sock
            read_fds = [udp]
            if tcp_sock is not None and tcp_sock.fileno() != -1:
                read_fds.append(tcp_sock)

            try:
                ready, _, errored = select.select(read_fds, [], read_fds, SELECT_POLL_S)
            except Exception as e:
                # Most likely tcp_sock was closed by send() between the snapshot
                # above and this call (rare, self-heals next iteration).
                print(f"[WifiTransport] select() error: {e}")
                time.sleep(0.1)
                continue

            if udp in ready:
                try:
                    data, (src_ip, _) = udp.recvfrom(4096)
                except Exception as e:
                    print(f"[WifiTransport] UDP error: {e}")
                    break

                if data.startswith(b"WLR_ACK_V1 "):
                    parts = data.decode("ascii", errors="ignore").split()
                    if len(parts) >= 2 and parts[1].upper() == f"{self._session_token:08X}":
                        self._last_lease_ack_time = time.monotonic()
                        if src_ip != self._esp_ip:
                            self._esp_ip = src_ip
                            with self._tcp_lock:
                                self._close_tcp()
                                self._fail_count = 0
                                self._next_connect_time = 0.0
                    continue
                if data.startswith(b"WLR_BUSY_V1 "):
                    continue

                self._last_udp_time = time.monotonic()
                if src_ip != self._esp_ip:
                    self._esp_ip = src_ip
                    with self._tcp_lock:
                        self._close_tcp()   # reset TCP so it reconnects to new IP
                        self._fail_count        = 0
                        self._next_connect_time = 0.0
                if not self._connected:
                    self._connected = True
                    sm._on_opened("wifi")

                self._decoder.feed(data)

            if tcp_sock is not None and (tcp_sock in ready or tcp_sock in errored):
                try:
                    data = tcp_sock.recv(4096)
                except Exception:
                    data = None
                if not data:
                    # Peer closed or errored — only close if send() hasn't already
                    # swapped in a different socket since our snapshot above.
                    with self._tcp_lock:
                        if self._tcp_sock is tcp_sock:
                            self._close_tcp()
                else:
                    self._tcp_decoder.feed(data)

            # UDP telemetry and TCP command liveness are independent. A busy or
            # healthy TCP stream must never mask missing telemetry, and UDP loss
            # must not tear down a still-usable command connection.
            if (self._connected and self._last_udp_time and
                    time.monotonic() - self._last_udp_time > TELEMETRY_TIMEOUT_S):
                self._connected = False
                sm._on_released("wifi")

        udp.close()
        with self._tcp_lock:
            self._close_tcp()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _close_tcp(self):
        """Close TCP socket. Must be called with _tcp_lock held."""
        if self._tcp_sock is not None:
            try:
                self._tcp_sock.close()
            except Exception:
                pass
            self._tcp_sock = None
