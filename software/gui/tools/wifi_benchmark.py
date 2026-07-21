"""Measure the ESP32 WiFi path without Qt/UI scheduling in the result.

Claims the single telemetry session, receives the combined UDP stream, and
sends correlated v2 PING commands over TCP. The ESP32 must not already have a
GUI lease; close the GUI (or wait for its 3.5 s lease to expire) first.
"""

import argparse
import json
import secrets
import select
import socket
import statistics
import struct
import time

UDP_PORT = 5005
TCP_PORT = 5006
DISCOVERY_PORT = 5007
TELEM_TYPE = 0x15
COMMAND_RESULT_TYPE = 0x16


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def frame(ptype: int, version: int, source: int, seq: int, payload: bytes) -> bytes:
    header = struct.pack("<BBBBH", ptype, version, source, seq, len(payload))
    return b"\xAA\x55" + header + payload + bytes((crc8(header + payload), 0xEF))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int((pct / 100) * len(ordered) + 0.999999) - 1))]


def parse_stream(buffer: bytearray):
    frames = []
    while True:
        start = buffer.find(b"\xAA\x55")
        if start < 0:
            buffer[:] = buffer[-1:]
            break
        if start:
            del buffer[:start]
        if len(buffer) < 10:
            break
        length = buffer[6] | buffer[7] << 8
        total = length + 10
        if length > 512:
            del buffer[0]
            continue
        if len(buffer) < total:
            break
        raw = bytes(buffer[:total])
        del buffer[:total]
        if raw[-1] == 0xEF and crc8(raw[2:-2]) == raw[-2]:
            frames.append((raw[2], raw[3], raw[8:-2]))
    return frames


def run(ip: str, duration: float, ping_hz: float) -> dict:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp.bind(("", UDP_PORT))
    udp.setblocking(False)

    token = f"{secrets.randbits(32) or 1:08X}"
    claim = f"WLR_CLAIM_V1 {token}".encode("ascii")
    deadline = time.monotonic() + 5.0
    next_claim = 0.0
    claimed = False
    while time.monotonic() < deadline and not claimed:
        now = time.monotonic()
        if now >= next_claim:
            udp.sendto(claim, ("255.255.255.255", DISCOVERY_PORT))
            udp.sendto(claim, (ip, DISCOVERY_PORT))
            next_claim = now + 0.5
        readable, _, _ = select.select([udp], [], [], 0.1)
        if readable:
            data, peer = udp.recvfrom(4096)
            if data.startswith(f"WLR_ACK_V1 {token} ".encode("ascii")):
                ip = peer[0]
                claimed = True
            elif data.startswith(b"WLR_BUSY_V1 "):
                raise RuntimeError("ESP32 WiFi session is owned by another client")
    if not claimed:
        raise TimeoutError("no matching ESP32 lease acknowledgement")

    tcp = socket.create_connection((ip, TCP_PORT), timeout=2.0)
    tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    tcp.setblocking(False)

    started = time.monotonic()
    next_ping = started
    next_renew = started + 0.8
    seq = 0
    request_id = 1
    sent: dict[int, float] = {}
    rtts_ms: list[float] = []
    tcp_buffer = bytearray()
    telemetry_times: list[float] = []
    telemetry_seq_gaps = 0
    last_telem_seq = None
    malformed_udp = 0
    diagnostic_frames = 0

    try:
        while time.monotonic() - started < duration:
            now = time.monotonic()
            if now >= next_renew:
                udp.sendto(claim, (ip, DISCOVERY_PORT))
                next_renew += 0.8
            if now >= next_ping:
                payload = struct.pack("<IB", request_id, 0x02)
                tcp.sendall(frame(0x02, 2, 0x03, seq, payload))
                sent[request_id] = now
                request_id += 1
                seq = (seq + 1) & 0xFF
                next_ping += 1.0 / ping_hz

            readable, _, _ = select.select([udp, tcp], [], [], 0.02)
            for ready in readable:
                if ready is udp:
                    data, _ = udp.recvfrom(4096)
                    if data.startswith(b"WLR_"):
                        continue
                    if len(data) < 10 or data[:2] != b"\xAA\x55":
                        malformed_udp += 1
                        continue
                    length = data[6] | data[7] << 8
                    valid = (len(data) == length + 10 and data[-1] == 0xEF
                             and crc8(data[2:-2]) == data[-2])
                    if not valid:
                        malformed_udp += 1
                        continue
                    udp_seq = data[5]
                    if last_telem_seq is not None:
                        telemetry_seq_gaps += (udp_seq - last_telem_seq - 1) & 0xFF
                    last_telem_seq = udp_seq
                    if data[2] == TELEM_TYPE and length == 247:
                        telemetry_times.append(time.monotonic())
                    elif data[2] == 0x14:
                        diagnostic_frames += 1
                    else:
                        malformed_udp += 1
                else:
                    chunk = tcp.recv(4096)
                    if not chunk:
                        raise ConnectionError("ESP32 closed TCP command connection")
                    tcp_buffer.extend(chunk)
                    for ptype, version, payload in parse_stream(tcp_buffer):
                        if ptype == COMMAND_RESULT_TYPE and version == 1 and len(payload) == 8:
                            rid, cmd_id, status, reason, _state = struct.unpack("<IBBBB", payload)
                            sent_at = sent.pop(rid, None)
                            if sent_at is not None and cmd_id == 0x02 and status == 0 and reason == 0:
                                rtts_ms.append((time.monotonic() - sent_at) * 1000.0)
    finally:
        tcp.close()
        udp.close()

    intervals_ms = [(b - a) * 1000.0 for a, b in zip(telemetry_times, telemetry_times[1:])]
    result = {
        "esp32_ip": ip,
        "duration_s": duration,
        "commands_sent": request_id - 1,
        "command_results": len(rtts_ms),
        "command_timeouts": len(sent),
        "command_rtt_ms": {
            "p50": percentile(rtts_ms, 50),
            "p95": percentile(rtts_ms, 95),
            "p99": percentile(rtts_ms, 99),
            "max": max(rtts_ms) if rtts_ms else None,
        },
        "telemetry_frames": len(telemetry_times),
        "telemetry_rate_hz": len(telemetry_times) / duration,
        "telemetry_interval_ms": {
            "mean": statistics.fmean(intervals_ms) if intervals_ms else None,
            "p99": percentile(intervals_ms, 99),
            "max": max(intervals_ms) if intervals_ms else None,
        },
        "telemetry_seq_gaps": telemetry_seq_gaps,
        "diagnostic_frames": diagnostic_frames,
        "malformed_udp": malformed_udp,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esp32-ip", default="192.168.1.151")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--ping-hz", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(run(args.esp32_ip, args.duration, args.ping_hz), indent=2))


if __name__ == "__main__":
    main()
