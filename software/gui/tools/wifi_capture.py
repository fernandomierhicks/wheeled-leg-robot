"""wifi_capture.py — headless WiFi telemetry capture for the WiFi diagnostics campaign.

Standalone, no Qt dependency (same precedent as wlog_to_csv.py / trigger_log_test.py:
only tabs.telem_format's pure decode functions are imported, not PacketDecoder or any
Qt singleton). A UDP datagram is a natural frame boundary, so unlike the live
PacketDecoder this needs no byte-stream resync buffering — each recvfrom() is exactly
one CommLink frame.

Binds UDP 5005 and logs one JSON line per received datagram (decoded TELEM_A/TELEM_B/
TELEM_FULL_WIFI/WIFI_DIAG fields merged in, plus arrival timestamp/seq/crc_ok) to a
.jsonl file. All loss/jitter/clumping/reorder metrics are computed offline by
analyze_wifi_capture.py from these raw per-datagram records — this script only
captures and lightly annotates (e.g. A/B pair_ok) as it goes.

Also spawns concurrent PC-side link probes for the capture duration, logged to
timestamped sidecar files next to the .jsonl:
  - `ping <esp32_ip> -t`                        -> <out>.ping.log
  - `netsh wlan show interfaces` sampled ~0.5 Hz -> <out>.netsh_if.log
  - one `netsh wlan show networks mode=bssid`    -> <out>.netsh_networks.log

Usage:
    python wifi_capture.py --duration 300 --out logs/wifi_baseline.jsonl [--esp32-ip A.B.C.D]
"""

import argparse
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

# Full paths: some Python distributions (e.g. the Windows Store build) don't
# inherit a PATH that resolves bare "netsh"/"ping" via subprocess.
_WINDIR = os.environ.get("WINDIR", r"C:\Windows")
_NETSH  = str(Path(_WINDIR) / "System32" / "netsh.exe")
_PING   = str(Path(_WINDIR) / "System32" / "PING.EXE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.telem_format import (  # noqa: E402
    crc8, decode_telem_a, decode_telem_b, decode_telem_full,
)

_GUI_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _GUI_DIR.parent.parent / "data" / "logs" / "diagnostics" / "wifi"
_DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

UDP_PORT = 5005
DISCOVERY_PORT = 5007
CLAIM_PERIOD_S = 0.8

_COMM_MAGIC = bytes([0xAA, 0x55])
_COMM_END   = 0xEF
_HEADER_SZ  = 8
_OVERHEAD   = 10

_TELEM_A_LEN = 118
_TELEM_B_LEN = 124

_TYPE_NAMES = {
    0x10: "TELEM_A", 0x11: "TELEM_B", 0x14: "WIFI_DIAG", 0x15: "TELEM_FULL_WIFI",
}

# WifiDiagPayload (26 bytes, packed) — shared/comm_protocol.h
_FMT_WIFI_DIAG = "<IbBBBIIHHHHBB"
_WIFI_DIAG_FIELDS = [
    "esp_uptime_ms", "rssi_dbm", "wifi_channel", "wifi_status", "tx_power_raw",
    "free_heap", "min_free_heap", "loop_max_us", "udp_send_max_us",
    "wifi_reconnect_count", "udp_send_fail_count",
    "build_variant_flags", "active_telem_transport",
]
assert struct.calcsize(_FMT_WIFI_DIAG) == 26, "WifiDiagPayload format mismatch"

# V2 additions (12 bytes) appended after the V1 26 bytes — Teensy-link status,
# the "ESP32 alive" carrier that Phase 7's pass/fail table reads CRC/seq-gap
# counters from (see UARTplat.md Phase 3). Length-gated so this script still
# reads a pre-Phase-3 (V1, 26-byte) capture without erroring.
_FMT_WIFI_DIAG_V2_EXT = "<BBHHHHH"
_WIFI_DIAG_V2_FIELDS = [
    "teensy_link_up", "_reserved2", "ms_since_teensy", "uart_crc_drops",
    "uart_seq_gaps", "uplink_queue_drops", "tcp_send_max_us",
]
assert struct.calcsize(_FMT_WIFI_DIAG_V2_EXT) == 12, "WifiDiagPayload V2 extension format mismatch"

# ESP32 USB-serial VID/PID (mirrors tabs/port_manager.py SIGNATURES["esp32"] —
# duplicated locally rather than imported, since port_manager.py pulls in PyQt6).
_ESP32_VID_PID = [(0x2886, 0x0056), (0x303A, 0x1001), (0x10C4, 0xEA60)]


def _parse_frame(data: bytes):
    """Parse exactly one CommLink frame from a UDP datagram. None if malformed."""
    if len(data) < _HEADER_SZ or data[0:2] != _COMM_MAGIC:
        return None
    ptype, version, source, seq = data[2], data[3], data[4], data[5]
    length = data[6] | (data[7] << 8)
    total = _OVERHEAD + length
    if len(data) < total:
        return None
    payload  = data[8:8 + length]
    checksum = data[8 + length]
    end_byte = data[9 + length]
    if end_byte != _COMM_END:
        return None
    crc_ok = crc8(data[2:8 + length]) == checksum
    return {
        "ptype": ptype, "version": version, "source": source, "seq": seq,
        "length": length, "payload": payload, "crc_ok": crc_ok,
    }


def _detect_esp32_ip(timeout_s: float = 8.0) -> str | None:
    """Auto-detect the ESP32's LAN IP by watching its USB serial port for the
    "[WiFi] IP: ..." boot line. Returns None (caller should fall back to
    --esp32-ip) if no ESP32 port is found or nothing is seen within timeout_s."""
    import serial
    import serial.tools.list_ports

    port = None
    for info in serial.tools.list_ports.comports():
        if (info.vid, info.pid) in _ESP32_VID_PID:
            port = info.device
            break
    if port is None:
        print("[wifi_capture] no ESP32 USB serial port found for IP auto-detect", file=sys.stderr)
        return None

    try:
        with serial.Serial(port, 921600, timeout=0.5) as ser:
            deadline = time.time() + timeout_s
            buf = b""
            while time.time() < deadline:
                chunk = ser.read(256)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if "[WiFi] IP:" in text:
                        return text.split("[WiFi] IP:")[-1].strip()
    except Exception as e:
        print(f"[wifi_capture] ESP32 IP auto-detect failed: {e}", file=sys.stderr)
    return None


# ── PC-side link probes ────────────────────────────────────────────────────────

class ProbeSet:
    """Spawns/owns the ping + netsh sidecar probes for one capture run."""

    def __init__(self, esp32_ip: str | None, out_base: Path):
        self._esp32_ip   = esp32_ip
        self._out_base   = out_base
        self._ping_proc: subprocess.Popen | None = None
        self._ping_thread: threading.Thread | None = None
        self._netsh_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        # One-shot channel-congestion snapshot.
        try:
            r = subprocess.run(
                [_NETSH, "wlan", "show", "networks", "mode=bssid"],
                capture_output=True, text=True, timeout=10,
            )
            (self._out_base.with_suffix(".netsh_networks.log")).write_text(
                r.stdout + r.stderr, encoding="utf-8", errors="replace"
            )
        except Exception as e:
            print(f"[wifi_capture] netsh networks snapshot failed: {e}", file=sys.stderr)

        # Continuous ping RTT/loss series.
        if self._esp32_ip:
            try:
                self._ping_proc = subprocess.Popen(
                    [_PING, "-t", self._esp32_ip],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                self._ping_thread = threading.Thread(target=self._pump_ping, daemon=True)
                self._ping_thread.start()
            except Exception as e:
                print(f"[wifi_capture] ping probe failed to start: {e}", file=sys.stderr)
        else:
            print("[wifi_capture] no ESP32 IP — skipping ping probe", file=sys.stderr)

        # PC-side RSSI/channel/PHY-rate sampled ~0.5 Hz.
        self._netsh_thread = threading.Thread(target=self._pump_netsh_interfaces, daemon=True)
        self._netsh_thread.start()

    def _pump_ping(self):
        path = self._out_base.with_suffix(".ping.log")
        with open(path, "a", buffering=1, encoding="utf-8", errors="replace") as f:
            for line in self._ping_proc.stdout:
                f.write(f"{time.time():.3f}\t{line.rstrip()}\n")
                if self._stop.is_set():
                    break

    def _pump_netsh_interfaces(self):
        path = self._out_base.with_suffix(".netsh_if.log")
        with open(path, "a", buffering=1, encoding="utf-8", errors="replace") as f:
            while not self._stop.is_set():
                try:
                    r = subprocess.run(
                        [_NETSH, "wlan", "show", "interfaces"],
                        capture_output=True, text=True, timeout=5,
                    )
                    f.write(f"── {time.time():.3f} ──\n{r.stdout}\n")
                except Exception as e:
                    f.write(f"── {time.time():.3f} ── ERROR: {e}\n")
                self._stop.wait(2.0)  # ~0.5 Hz

    def stop(self):
        self._stop.set()
        if self._ping_proc is not None:
            try:
                self._ping_proc.terminate()
                self._ping_proc.wait(timeout=3)
            except Exception:
                try:
                    self._ping_proc.kill()
                except Exception:
                    pass
        if self._ping_thread is not None:
            self._ping_thread.join(timeout=3)
        if self._netsh_thread is not None:
            self._netsh_thread.join(timeout=3)


# ── Capture loop ─────────────────────────────────────────────────────────────

def run_capture(duration_s: float, out_path: Path, esp32_ip: str | None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.settimeout(1.0)
    sock.bind(("", UDP_PORT))
    token = f"{secrets.randbits(32) or 1:08X}"
    claim = f"WLR_CLAIM_V1 {token}".encode("ascii")
    next_claim = 0.0

    probes = ProbeSet(esp32_ip, out_path)
    probes.start()

    pending_a: dict | None = None
    counts = {"TELEM_A": 0, "TELEM_B": 0, "TELEM_FULL_WIFI": 0, "WIFI_DIAG": 0,
              "malformed": 0, "crc_fail": 0, "pair_drops": 0}

    print(f"[wifi_capture] listening on UDP :{UDP_PORT} for {duration_s:.0f}s -> {out_path}",
          file=sys.stderr)

    t_start = time.time()
    t_last_status = t_start
    with open(out_path, "a", buffering=1, encoding="utf-8") as f:
        while True:
            now = time.time()
            if now - t_start >= duration_s:
                break
            if now >= next_claim:
                # Broadcast discovers an unknown ESP32; directed renewals keep an
                # already-discovered lease alive on networks that filter broadcast.
                sock.sendto(claim, ("255.255.255.255", DISCOVERY_PORT))
                if esp32_ip:
                    sock.sendto(claim, (esp32_ip, DISCOVERY_PORT))
                next_claim = now + CLAIM_PERIOD_S
            try:
                data, (src_ip, _src_port) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[wifi_capture] recv error: {e}", file=sys.stderr)
                break
            recv_ts = time.time()

            if data.startswith(b"WLR_ACK_V1 "):
                esp32_ip = src_ip
                continue
            if data.startswith(b"WLR_BUSY_V1 "):
                print("[wifi_capture] ESP32 telemetry lease is owned by another client",
                      file=sys.stderr)
                continue

            frame = _parse_frame(data)
            event: dict = {"recv_ts": recv_ts, "src_ip": src_ip, "raw_len": len(data)}
            if frame is None:
                event["malformed"] = True
                counts["malformed"] += 1
                f.write(json.dumps(event) + "\n")
                continue

            ptype = frame["ptype"]
            event.update({
                "ptype": ptype,
                "type_name": _TYPE_NAMES.get(ptype, f"0x{ptype:02X}"),
                "version": frame["version"],
                "seq": frame["seq"],
                "length": frame["length"],
                "crc_ok": frame["crc_ok"],
            })
            if not frame["crc_ok"]:
                counts["crc_fail"] += 1
                f.write(json.dumps(event) + "\n")
                continue

            payload = frame["payload"]
            try:
                if ptype == 0x10 and frame["length"] == _TELEM_A_LEN:
                    counts["TELEM_A"] += 1
                    event.update(decode_telem_a(payload))
                    pending_a = {"seq": frame["seq"], "recv_ts": recv_ts}
                elif ptype == 0x11 and frame["length"] == _TELEM_B_LEN:
                    counts["TELEM_B"] += 1
                    event.update(decode_telem_b(payload))
                    if pending_a is not None:
                        pair_ok = ((pending_a["seq"] + 1) & 0xFF) == frame["seq"]
                        event["pair_ok"]     = pair_ok
                        event["a_seq"]       = pending_a["seq"]
                        event["a_recv_ts"]   = pending_a["recv_ts"]
                        if not pair_ok:
                            counts["pair_drops"] += 1
                    else:
                        event["pair_ok"] = False
                        counts["pair_drops"] += 1
                    pending_a = None
                elif ptype == 0x15 and frame["length"] == 247:
                    counts["TELEM_FULL_WIFI"] += 1
                    event.update(decode_telem_full(payload))
                elif ptype == 0x14 and frame["length"] >= 26:
                    counts["WIFI_DIAG"] += 1
                    fields = struct.unpack_from(_FMT_WIFI_DIAG, payload)
                    event.update(dict(zip(_WIFI_DIAG_FIELDS, fields)))
                    if frame["length"] >= 38:
                        v2_fields = struct.unpack_from(_FMT_WIFI_DIAG_V2_EXT, payload, 26)
                        event.update(dict(zip(_WIFI_DIAG_V2_FIELDS, v2_fields)))
            except Exception as e:
                event["decode_error"] = str(e)

            f.write(json.dumps(event) + "\n")

            if now - t_last_status >= 30:
                t_last_status = now
                print(f"[wifi_capture] {now - t_start:.0f}s elapsed: {counts}", file=sys.stderr)

    probes.stop()
    sock.close()
    print(f"[wifi_capture] done: {counts}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, required=True, help="capture duration in seconds")
    ap.add_argument("--out", type=str, default=None,
                     help="output .jsonl path (default: logs/wifi_capture_<timestamp>.jsonl)")
    ap.add_argument("--esp32-ip", type=str, default=None,
                     help="ESP32 LAN IP for the ping probe (auto-detected from USB serial if omitted)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        _DEFAULT_LOG_DIR / f"wifi_capture_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    esp32_ip = args.esp32_ip or _detect_esp32_ip()
    if esp32_ip:
        print(f"[wifi_capture] ESP32 IP: {esp32_ip}", file=sys.stderr)
    else:
        print("[wifi_capture] ESP32 IP unknown — ping probe will be skipped", file=sys.stderr)

    run_capture(args.duration, out_path, esp32_ip)


if __name__ == "__main__":
    main()
