"""stress_test_arm.py — bench stress test of the mode-change command surface
(CMD_ID_SET_MODE) over USB serial: repeated CALIBRATION/RUNNING requests plus
a fuzzed out-of-range SET_MODE target per round, verifying the robot responds
exactly as state_machine.cpp's req_running()/req_calibration() say it should.

Standalone (pyserial only), same pattern as tools/trigger_log_test.py /
tools/trigger_running_test.py. Imports tabs.telem_format (Qt-free) to decode
TELEM_A so the state enum can't drift out of sync with comm_protocol.h.

SAFETY — this test only makes sense, and is only safe to run unattended-ish,
with all four motor-enable params OFF (hip_l/r_enable, wheel_l/r_enable) so
RUNNING never commands real torque. The script queries these live first and
refuses to proceed if any of them are on, or if calib_bypass_en is off (that
one it won't silently flip — it's persisted and has a bigger footprint than
this test's own throwaway run_wheel_bypass_en flag).

What this test does NOT cover: it cannot simulate physical RC transmitter
input (CH5/CH9/CH10 channel values) — those only reach the GUI via telemetry
(TelemetryPayload::ibus_ch/ibus_alive), with no software injection point in
the current firmware. "Radio commands" here
means the same state transitions the radio triggers (CALIBRATION, RUNNING),
issued instead through CMD_ID_SET_MODE. Profile switching (CH9) has no
non-radio trigger at all right now, so it isn't exercised here.

Usage:
    python stress_test_arm.py [--port COM12] [--rounds 5]
"""

import argparse
import random
import struct
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # software/gui/
from tabs.telem_format import TELEM_VERSION, decode_telem_a  # noqa: E402
from tabs.generated_protocol import (  # noqa: E402
    CMD_ID_PARAM_GET, CMD_ID_PARAM_SET, CMD_ID_SET_MODE,
    PARAM_CALIB_BYPASS_EN, PARAM_HIP_L_ENABLE, PARAM_HIP_R_ENABLE,
    PARAM_IMU_ENABLE, PARAM_RUNNING_WHEEL_BYPASS_EN,
    PARAM_WHEEL_L_ENABLE, PARAM_WHEEL_R_ENABLE,
    STATE_CALIBRATION, STATE_CMD_REJECT, STATE_ESTOP, STATE_NAMES,
    STATE_RUNNING, STATE_STANDBY, STATE_STARTUP,
)

TEENSY_VID_PID = (0x16C0, 0x0483)
BAUD = 115200

# ── CommLink frame constants (shared/comm_protocol.h) ─────────────────────────
START_A, START_B, END = 0xAA, 0x55, 0xEF
SRC_PC = 0x03
TYPE_COMMAND, TYPE_LOG, TYPE_PARAM_REPORT, TYPE_TELEM_A = 0x02, 0x04, 0x06, 0x10
LOG_LEVEL_NAMES = {1: "INFO", 2: "WARN", 3: "ERROR"}

# RobotStateEnum (robot_state.h)

# Deliberately curated (not full-range random): every value here is outside
# the valid 0-7 RobotStateEnum range, so "expect no state change" is always
# an unambiguous pass/fail regardless of which one gets picked.
FUZZ_TARGETS = [9, 42, 128, 255]


# CRC-8 poly 0x07, init 0x00 — MIRROR of crc8() in tabs/telem_format.py and
# shared/CommLink/CommLink.cpp (inlined here to keep this script standalone).
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


def build_frame(ftype: int, version: int, payload: bytes, seq: int) -> bytes:
    plen = len(payload)
    header = bytes([ftype, version, SRC_PC, seq & 0xFF, plen & 0xFF, (plen >> 8) & 0xFF])
    return bytes([START_A, START_B]) + header + payload + bytes([_crc8(header + payload), END])


def find_teensy_port() -> str:
    for p in list_ports.comports():
        if (p.vid, p.pid) == TEENSY_VID_PID:
            return p.device
    raise SystemExit(f"No Teensy found (VID:PID {TEENSY_VID_PID[0]:04X}:{TEENSY_VID_PID[1]:04X}). "
                      "Pass --port explicitly, e.g. --port COM12.")


def parse_frames(buf: bytearray):
    """Yield (type, version, payload) for each complete frame found in buf,
    consuming bytes as it goes. Best-effort — drops anything before a sync."""
    while True:
        i = buf.find(bytes([START_A, START_B]))
        if i < 0:
            buf.clear()
            return
        if i > 0:
            del buf[:i]
        if len(buf) < 8:
            return
        ftype, ver, _src, _seq, len_lo, len_hi = buf[2:8]
        plen = len_lo | (len_hi << 8)
        total = 10 + plen
        if len(buf) < total:
            return
        payload = bytes(buf[8:8 + plen])
        crc = buf[8 + plen]
        end = buf[9 + plen]
        del buf[:total]
        if end != END:
            continue  # corrupt frame, already dropped
        if crc != _crc8(bytes([ftype, ver, _src, _seq, len_lo, len_hi]) + payload):
            continue
        yield ftype, ver, payload


class Link:
    """Thin stateful wrapper: tracks the latest known robot_state and
    accumulates comm_log lines seen while pumping the serial port."""

    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.buf = bytearray()
        self.seq = 0
        self.state = None
        self.logs: list[tuple[str, str]] = []
        self.seen_states: set[int] = set()  # every robot_state value observed since last clear

    def _send(self, payload: bytes):
        frame = build_frame(TYPE_COMMAND, 1, payload, self.seq)
        self.seq = (self.seq + 1) & 0xFF
        self.ser.write(frame)

    def set_mode(self, target: int):
        self._send(struct.pack("<BB", CMD_ID_SET_MODE, target))

    def param_get(self, param_id: int):
        self._send(struct.pack("<BH", CMD_ID_PARAM_GET, param_id))

    def param_set(self, param_id: int, value: float):
        self._send(struct.pack("<BHf", CMD_ID_PARAM_SET, param_id, value))

    def pump(self, duration_s: float, verbose: bool = True) -> list[tuple[int, float]]:
        """Read for duration_s, updating self.state/self.logs. Returns any
        (param_id, value) PARAM_REPORTs seen during the window."""
        reports = []
        deadline = time.time() + duration_s
        while time.time() < deadline:
            chunk = self.ser.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue
            self.buf.extend(chunk)
            for ftype, ver, payload in parse_frames(self.buf):
                if ftype == TYPE_LOG and len(payload) >= 1:
                    level = LOG_LEVEL_NAMES.get(payload[0], str(payload[0]))
                    msg = payload[1:].decode("utf-8", errors="replace")
                    if verbose:
                        print(f"    [{level}] {msg}")
                    self.logs.append((level, msg))
                elif ftype == TYPE_TELEM_A and ver == TELEM_VERSION:
                    self.state = decode_telem_a(payload)["robot_state"]
                    self.seen_states.add(self.state)
                elif ftype == TYPE_PARAM_REPORT and len(payload) >= 15:
                    param_id, value = struct.unpack_from("<Hf", payload)
                    reports.append((param_id, value))
        return reports

    def get_param(self, param_id: int, timeout: float = 1.0):
        self.param_get(param_id)
        for pid, val in self.pump(timeout, verbose=False):
            if pid == param_id:
                return val
        return None

    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, str(self.state))


def to_standby(link: Link):
    """Best-effort recovery to STANDBY from wherever we currently are.
    No software path disarms RUNNING directly (that's a real gap — see
    state_machine.cpp stateMachine_exit_manual(), which only fires from
    MANUAL/CALIBRATION) so this goes RUNNING -> ESTOP -> STANDBY, relying on
    FAULT_HUMAN_ESTOP being FAULT_SEVERITY_SOFT (soft-clear, no full reset
    needed)."""
    link.pump(0.05, verbose=False)
    if link.state == STATE_RUNNING:
        link.set_mode(STATE_ESTOP)
        link.pump(0.3, verbose=False)
    if link.state == STATE_ESTOP:
        link.set_mode(STATE_STANDBY)  # soft-clear path (on_command -> stateMachine_request_soft_clear)
        link.pump(0.3, verbose=False)
    if link.state == STATE_CMD_REJECT:
        link.pump(1.3, verbose=False)  # ~1s auto-return to STANDBY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None, help="COM port (auto-detected by VID:PID if omitted)")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    # comm_log messages are UTF-8 (some embed em-dashes); a console codepage
    # that can't represent a given character shouldn't crash a live hardware
    # run mid-test — replace instead of raising.
    sys.stdout.reconfigure(errors="replace")

    port = args.port or find_teensy_port()
    print(f"Opening {port} @ {BAUD}...")
    with serial.Serial(port, BAUD, timeout=0.1) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()
        link = Link(ser)

        # ── Safety gate: read live state before doing anything ──────────────
        print("Reading live motor-enable / bypass params...")
        checks = {
            "imu_enable":          PARAM_IMU_ENABLE,
            "hip_l_enable":        PARAM_HIP_L_ENABLE,
            "hip_r_enable":        PARAM_HIP_R_ENABLE,
            "wheel_l_enable":      PARAM_WHEEL_L_ENABLE,
            "wheel_r_enable":      PARAM_WHEEL_R_ENABLE,
            "calib_bypass_en":     PARAM_CALIB_BYPASS_EN,
            "run_wheel_bypass_en": PARAM_RUNNING_WHEEL_BYPASS_EN,
        }
        live = {name: link.get_param(pid) for name, pid in checks.items()}
        for name, pid in checks.items():
            print(f"  {name:22s} (0x{pid:04X}) = {live[name]}")

        motor_live = [n for n in ("hip_l_enable", "hip_r_enable", "wheel_l_enable", "wheel_r_enable")
                      if live.get(n) is not None and live[n] >= 0.5]
        if motor_live:
            print(f"\nABORT: {motor_live} are enabled live on the device — RUNNING would "
                  "command real torque. Disable them (Params tab) before running this test.")
            sys.exit(1)
        if live.get("imu_enable") is None or live["imu_enable"] < 0.5:
            print("\nABORT: imu_enable is off (or unreadable) — RUNNING would always be denied, "
                  "this test wouldn't prove anything. Enable it first.")
            sys.exit(1)
        if live.get("calib_bypass_en") is None or live["calib_bypass_en"] < 0.5:
            print("\nABORT: calib_bypass_en is off — RUNNING will be denied on the hip check "
                  "even with hips disabled. Turn it on (Params tab) and re-run; not doing this "
                  "automatically since it's a persisted flag, unlike run_wheel_bypass_en below.")
            sys.exit(1)

        if live.get("run_wheel_bypass_en") is None or live["run_wheel_bypass_en"] < 0.5:
            print("Turning on run_wheel_bypass_en for this test (not persisted — resets to 0 "
                  "on the next reboot regardless)...")
            link.param_set(PARAM_RUNNING_WHEEL_BYPASS_EN, 1.0)
            link.pump(0.3, verbose=False)

        print("\nAll 4 motors disabled live, IMU enabled, bypasses set — safe to proceed "
              "(no real torque should be possible in this configuration).")
        print("NOTE: physical RC channel injection isn't possible from software (CH5/CH9/CH10 "
              "are read-only telemetry mirrors of the real receiver, PARAM_FLAG_READONLY) — "
              "this test exercises the same state transitions via CMD_ID_SET_MODE instead. "
              "Profile switching (CH9) has no non-radio trigger and isn't covered.\n")

        to_standby(link)
        print(f"Starting state: {link.state_name()}\n")

        results: list[tuple[str, bool]] = []

        for i in range(1, args.rounds + 1):
            print(f"=== Round {i}/{args.rounds} ===")

            # 1. Fuzz: out-of-range SET_MODE target must be a safe no-op.
            junk = random.choice(FUZZ_TARGETS)
            print(f"  [fuzz] SET_MODE(target={junk}) — expect no state change, no crash")
            before = link.state
            link.set_mode(junk)
            link.pump(0.2)
            ok = link.state == before
            print(f"    -> state stayed {link.state_name()}: {'PASS' if ok else 'FAIL'}")
            results.append((f"round{i}.fuzz_target_{junk}", ok))

            # 2. Calibration request — must be denied (both hips disabled).
            to_standby(link)
            print("  [calibration] SET_MODE(CALIBRATION) — expect denial (hips disabled)")
            link.logs.clear()
            link.set_mode(STATE_CALIBRATION)
            link.pump(1.5)
            denied = any("Calibration denied" in m for _, m in link.logs)
            back_to_standby = link.state == STATE_STANDBY
            ok = denied and back_to_standby
            print(f"    -> denial logged: {denied}, back in STANDBY: {back_to_standby}: "
                  f"{'PASS' if ok else 'FAIL'}")
            results.append((f"round{i}.calibration_denied", ok))

            # 3. Arm into RUNNING. With no RC receiver connected (ibus_alive=0),
            # radio_update()'s level-based disarm check fires on the very next
            # tick regardless of entry path (armed = alive && ch10>1990, and
            # alive is always false here) — confirmed behavior, kept as-is by
            # design (RUNNING should only persist with a live radio link
            # corroborating "armed"). So the pass condition is: the state
            # machine DID transition through RUNNING (command worked), and
            # settled back in STANDBY afterward (interlock worked) — not that
            # it's still in RUNNING when we check.
            to_standby(link)
            print("  [arm] SET_MODE(RUNNING) — expect transition through RUNNING, "
                  "then radio-interlock disarm back to STANDBY (no RC receiver connected)")
            link.logs.clear()
            link.seen_states.clear()
            link.set_mode(STATE_RUNNING)
            link.pump(1.0)
            # on_running()'s entry action logs this exactly once on transition — a
            # guaranteed discrete event, unlike telemetry sampling (which can miss
            # a state that only persists for a single ~2ms tick before the radio
            # interlock reverts it; seen_states/TELEM_A alone under-reported this).
            entered_running = any("-> RUNNING (armed)" in m for _, m in link.logs)
            settled_standby = link.state == STATE_STANDBY
            ok = entered_running and settled_standby
            print(f"    -> entered RUNNING: {entered_running}, settled in {link.state_name()}: "
                  f"{'PASS' if ok else 'FAIL'}")
            results.append((f"round{i}.arm_running", ok))

            # 4. Recover to STANDBY before the next round.
            to_standby(link)
            ok = link.state == STATE_STANDBY
            print(f"  [recover] back to STANDBY: {'PASS' if ok else 'FAIL'}")
            results.append((f"round{i}.recover_standby", ok))
            print()

        # ── Cleanup: leave the throwaway bypass flag off ─────────────────────
        link.param_set(PARAM_RUNNING_WHEEL_BYPASS_EN, 0.0)
        link.pump(0.3, verbose=False)

        # ── Summary ───────────────────────────────────────────────────────────
        print("=== Summary ===")
        for name, ok in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        n_pass = sum(1 for _, ok in results if ok)
        print(f"\n{n_pass}/{len(results)} checks passed.")
        print(f"Final state: {link.state_name()}")
        if n_pass != len(results):
            sys.exit(1)


if __name__ == "__main__":
    main()
