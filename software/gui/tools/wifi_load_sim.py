"""wifi_load_sim.py — headless GUI-equivalent WiFi load generator.

Mirrors the network-facing load the real GUI puts on the ESP32 when connected
over WiFi: a steady CMD_ID_PING heartbeat (matches MainWindow's 10 Hz ping
timer, main.py), periodic CMD_ID_PARAM_GET 0xFFFF dumps (a deliberately
heavier version of what the Params tab does once on connect — here it repeats
to stress-test the paced param-dump path), and continuous TCP RX drain
(matches PacketDecoder.feed() always consuming whatever arrives). Used to
reproduce the WiFi-under-load flakiness described in UARTplat.md's root-cause
section without needing the actual GUI running, for the Phase 6 baseline-vs-
fixed test campaign (run alongside wifi_capture.py).

pyserial-free, sockets only — no USB access, no dependency on the `tabs`
package (its CRC-8/frame-builder are copied below, not imported, so this
script has zero ties to the GUI's Qt-based code). Learns the ESP32's LAN IP
from the source address of the first UDP telemetry datagram it sees on
:5005 — the same signal WifiTransport (tabs/wifi_transport.py) uses in the
real GUI — rather than parsing USB serial boot output.

Usage:
    python wifi_load_sim.py --duration 900
    python wifi_load_sim.py --duration 300 --ping-hz 10 --dump-period 2
    python wifi_load_sim.py --duration 300 --reboot-teensy-at 60
"""

import argparse
import secrets
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.generated_protocol import (  # noqa: E402
    CMD_ID_PARAM_GET, CMD_ID_PING, CMD_ID_REBOOT,
)

UDP_PORT = 5005
TCP_PORT = 5006
DISCOVERY_PORT = 5007

# ── CommLink frame constants + CRC-8 (shared/comm_protocol.h, shared/CommLink/
# CommLink.cpp) — copied (not imported) so this tool has no dependency on the
# `tabs` package. Keep in sync if the frame format ever changes.
_COMM_START_A  = 0xAA
_COMM_START_B  = 0x55
_COMM_END      = 0xEF
_COMM_SRC_PC   = 0x03
_COMM_TYPE_CMD = 0x02
_CMD_PAYLOAD_V = 2

_CRC8_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x07) & 0xFF if _c & 0x80 else (_c << 1) & 0xFF
    _CRC8_TABLE.append(_c)


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


_seq = [0]  # rolling Tx sequence counter, mirrors comm_commands.py's build_frame()
_request_id = [1]


def _build_frame(payload: bytes) -> bytes:
    """Wrap payload in a CommLink COMMAND frame (CRC-8 checksum) — copied
    from comm_commands.py's build_frame()."""
    request_id = _request_id[0]
    _request_id[0] = (_request_id[0] + 1) & 0xFFFFFFFF or 1
    payload = struct.pack("<I", request_id) + payload
    seq = _seq[0] & 0xFF
    _seq[0] += 1
    plen = len(payload)
    header = bytes([_COMM_TYPE_CMD, _CMD_PAYLOAD_V, _COMM_SRC_PC,
                     seq, plen & 0xFF, (plen >> 8) & 0xFF])
    crc = _crc8(header + payload)
    return bytes([_COMM_START_A, _COMM_START_B]) + header + payload + bytes([crc, _COMM_END])


def _ping_frame() -> bytes:
    return _build_frame(struct.pack("<B", CMD_ID_PING))


def _param_get_all_frame() -> bytes:
    return _build_frame(struct.pack("<BH", CMD_ID_PARAM_GET, 0xFFFF))


def _reboot_frame() -> bytes:
    return _build_frame(struct.pack("<B", CMD_ID_REBOOT))


def _detect_esp32_ip(timeout_s: float = 30.0) -> str:
    """Learn the ESP32's LAN IP from the source address of the first UDP
    telemetry datagram it sends — same signal WifiTransport uses (see module
    docstring). Raises TimeoutError if nothing arrives within timeout_s."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout_s)
    sock.bind(("", UDP_PORT))
    try:
        _, (src_ip, _src_port) = sock.recvfrom(4096)
        return src_ip
    finally:
        sock.close()


class TelemetryLease:
    """Own and renew the same explicit UDP telemetry lease as the GUI."""

    def __init__(self, esp32_ip: str | None):
        self.esp32_ip = esp32_ip
        self._token = f"{secrets.randbits(32) or 1:08X}"
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout_s: float = 8.0) -> str:
        self._thread.start()
        if not self._ready.wait(timeout_s):
            self.stop()
            raise TimeoutError("ESP32 did not acknowledge a telemetry lease")
        if self._error:
            raise self._error
        return self.esp32_ip

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", UDP_PORT))
        sock.settimeout(0.2)
        claim = f"WLR_CLAIM_V1 {self._token}".encode("ascii")
        next_claim = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_claim:
                    sock.sendto(claim, ("255.255.255.255", DISCOVERY_PORT))
                    if self.esp32_ip:
                        sock.sendto(claim, (self.esp32_ip, DISCOVERY_PORT))
                    next_claim = now + 0.8
                try:
                    data, peer = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if data.startswith(f"WLR_ACK_V1 {self._token} ".encode("ascii")):
                    self.esp32_ip = peer[0]
                    self._ready.set()
                elif data.startswith(b"WLR_BUSY_V1 ") and not self._ready.is_set():
                    self._error = RuntimeError("ESP32 WiFi session is owned by another client")
                    self._ready.set()
                    return
        finally:
            sock.close()


class TcpCommandLink:
    """Mirrors wifi_transport.py's WifiTransport.send(): lazy connect,
    exponential backoff on failure (0.5 s doubling to a 5 s cap — same as the
    GUI's Phase 4 fix), and a background thread continuously draining/
    discarding RX (matches PacketDecoder.feed() always consuming whatever the
    ESP32 forwards back over the same TCP connection)."""

    def __init__(self, esp32_ip: str):
        self._ip                = esp32_ip
        self._sock: socket.socket | None = None
        self._lock              = threading.Lock()
        self._fail_count        = 0
        self._next_connect_time = 0.0
        self._stop              = threading.Event()
        self._rx_thread         = threading.Thread(target=self._drain_rx, daemon=True)
        self.rx_bytes           = 0

    def start(self):
        self._rx_thread.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            self._close()
        self._rx_thread.join(timeout=3)

    def send(self, frame: bytes):
        with self._lock:
            try:
                if self._sock is None:
                    if time.monotonic() < self._next_connect_time:
                        return  # backing off after a recent failed connect
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1.0)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.connect((self._ip, TCP_PORT))
                    self._sock = s
                self._sock.sendall(frame)
                self._fail_count = 0
            except OSError:
                self._close()
                self._next_connect_time = time.monotonic() + min(5.0, 0.5 * 2 ** self._fail_count)
                self._fail_count += 1

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _drain_rx(self):
        while not self._stop.is_set():
            with self._lock:
                s = self._sock
            if s is None:
                time.sleep(0.05)
                continue
            try:
                s.settimeout(0.2)
                data = s.recv(4096)
                if data:
                    self.rx_bytes += len(data)
                else:
                    with self._lock:
                        self._close()  # peer closed
            except socket.timeout:
                continue
            except OSError:
                continue


def run(duration_s: float, ping_hz: float, dump_period_s: float,
        reboot_teensy_at: float | None, esp32_ip: str | None):
    lease = TelemetryLease(esp32_ip)
    esp32_ip = lease.start()
    print(f"[wifi_load_sim] ESP32 IP: {esp32_ip} — connecting TCP :{TCP_PORT}", file=sys.stderr)

    link = TcpCommandLink(esp32_ip)
    link.start()

    ping_period = 1.0 / ping_hz
    t_start     = time.monotonic()
    last_ping   = -ping_period       # fire immediately on the first loop tick
    last_dump   = -dump_period_s
    reboot_sent = False
    pings_sent  = 0
    dumps_sent  = 0

    try:
        while True:
            now = time.monotonic() - t_start
            if now >= duration_s:
                break
            if now - last_ping >= ping_period:
                last_ping = now
                link.send(_ping_frame())
                pings_sent += 1
            if now - last_dump >= dump_period_s:
                last_dump = now
                link.send(_param_get_all_frame())
                dumps_sent += 1
            if reboot_teensy_at is not None and not reboot_sent and now >= reboot_teensy_at:
                reboot_sent = True
                print(f"[wifi_load_sim] t={now:.1f}s: sending CMD_ID_REBOOT", file=sys.stderr)
                link.send(_reboot_frame())
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[wifi_load_sim] interrupted", file=sys.stderr)
    finally:
        link.stop()
        lease.stop()

    print(f"[wifi_load_sim] done: {pings_sent} pings, {dumps_sent} param dumps, "
          f"{link.rx_bytes} RX bytes drained", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, required=True, help="run duration in seconds")
    ap.add_argument("--ping-hz", type=float, default=10.0,
                     help="CMD_ID_PING rate in Hz (default 10, matches the GUI's ping timer)")
    ap.add_argument("--dump-period", type=float, default=10.0,
                     help="seconds between CMD_ID_PARAM_GET 0xFFFF dumps (default 10)")
    ap.add_argument("--reboot-teensy-at", type=float, default=None,
                     help="send one CMD_ID_REBOOT this many seconds after start (for the "
                          "ESP32-alive-during-Teensy-reboot test, UARTplat.md Phase 6/§7.4.3)")
    ap.add_argument("--esp32-ip", type=str, default=None,
                     help="skip UDP auto-detect and connect to this IP directly")
    args = ap.parse_args()
    run(args.duration, args.ping_hz, args.dump_period, args.reboot_teensy_at, args.esp32_ip)


if __name__ == "__main__":
    main()
