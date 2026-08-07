"""telem_format.py — TelemetryPayload struct format + decoders (shared/comm_protocol.h).

Single source of truth for the TELEM_VERSION 12 wire layout, shared between
flash_monitor.py (live GUI decode, Qt) and wlog_to_csv.py / log_playback.py
(offline .wlog decode, no Qt). Keep in sync with the PROPAGATION CHECKLIST
in shared/comm_protocol.h whenever TelemetryPayload changes.
"""

import struct

TELEM_VERSION = 12  # must match TELEM_VERSION in shared/comm_protocol.h

# ── Frame checksum — CRC-8 (poly 0x07, init 0x00, MSB-first — CRC-8/SMBus) ────
# MIRROR: crc8_table()/crc8_step() in shared/CommLink/CommLink.cpp. Replaced the
# original 1-byte XOR on 2026-07-13 (flag-day — firmware and GUI must match).
_CRC8_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x07) & 0xFF if _c & 0x80 else (_c << 1) & 0xFF
    _CRC8_TABLE.append(_c)


def crc8(data: bytes) -> int:
    """CRC-8 over the given bytes (frame header fields + payload)."""
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc

from .generated_protocol import (
    STATE_NAMES as _STATE_NAMES, FAULT_NAMES as _FAULT_NAMES,
    FAULT_DESCRIPTIONS as _FAULT_DESCRIPTIONS,
)

# Struct formats for split telemetry (V11, packed, little-endian)
# TELEM_A: bytes 0-117 of TelemetryPayload
FMT_TELEM_A = "<I9fBB3f14HB6f2I3B"
# TELEM_B: bytes 118-246 of TelemetryPayload
# 4H tof_dist, 2f rates, 3f accel, 2f hip_vel, 10f hip_cmd, 5f ctrl, 2f ff, H health, BB diag, I loop, B profile,
# f pitch_trim, B esp32_link_ok, 3H esp32_status_age/uart_rx_drops/uart_seq_gaps,  ← V9
# f gain_sched_alpha,  ← V10
# B standup_state  ← V11
FMT_TELEM_B = "<4H2f3f2f10f5f2fHBBIBfBHHHfB"
assert struct.calcsize(FMT_TELEM_A) == 118, (
    f"FMT_TELEM_A size mismatch: got {struct.calcsize(FMT_TELEM_A)}, expected 118 — "
    "sync with TelemetryPayload in shared/comm_protocol.h"
)
assert struct.calcsize(FMT_TELEM_B) == 129, (
    f"FMT_TELEM_B size mismatch: got {struct.calcsize(FMT_TELEM_B)}, expected 129 — "
    "sync with TelemetryPayload in shared/comm_protocol.h"
)

# Full 247-byte TelemetryPayload in one shot (used for .wlog playback, where
# each LogRecord embeds the whole struct rather than the split A/B frames).
FMT_TELEM_FULL = "<" + FMT_TELEM_A[1:] + FMT_TELEM_B[1:]
assert struct.calcsize(FMT_TELEM_FULL) == 247, (
    f"FMT_TELEM_FULL size mismatch: got {struct.calcsize(FMT_TELEM_FULL)}, expected 247 — "
    "sync with TelemetryPayload in shared/comm_protocol.h"
)


def decode_telem_a(payload: bytes) -> dict:
    (ts,
     pitch, pitch_rate, wheel_vel, hip_l, hip_r, whl_tau_l, whl_tau_r, roll, yaw,
     state, fault,
     test_val, hip_l_trq, hip_r_trq,
     *ibus_and_alive,
     wm_l_vel, wm_r_vel, wm_l_pos, wm_r_pos, wm_l_vbus, wm_r_vbus,
     wm_l_err, wm_r_err,
     wm_l_st, wm_r_st, wm_mode) = struct.unpack(FMT_TELEM_A, payload)
    ibus_ch    = list(ibus_and_alive[:14])
    ibus_alive = bool(ibus_and_alive[14])
    return {
        "timestamp_ms":    ts,
        "pitch_rad":       pitch,
        "pitch_rate_rads": pitch_rate,
        "wheel_vel_avg":   wheel_vel,
        "hip_l_pos_rad":   hip_l,
        "hip_r_pos_rad":   hip_r,
        "whl_tau_l":       whl_tau_l,
        "whl_tau_r":       whl_tau_r,
        "roll_rad":        roll,
        "yaw_rad":         yaw,
        "robot_state":     state,
        "state_name":      _STATE_NAMES.get(state, str(state)),
        "fault_code":      fault,
        "fault_name":      _FAULT_NAMES.get(fault, f"0x{fault:02X}"),
        "fault_description": _FAULT_DESCRIPTIONS.get(fault, "Unknown fault"),
        "test_val":        test_val,
        "hip_l_torque_nm": hip_l_trq,
        "hip_r_torque_nm": hip_r_trq,
        "ibus_ch":         ibus_ch,
        "ibus_alive":      ibus_alive,
        "wm_l_vel_turns_s": wm_l_vel,
        "wm_r_vel_turns_s": wm_r_vel,
        "wm_l_pos_turns":   wm_l_pos,
        "wm_r_pos_turns":   wm_r_pos,
        "wm_l_vbus":        wm_l_vbus,
        "wm_r_vbus":        wm_r_vbus,
        "wm_l_error":       wm_l_err,
        "wm_r_error":       wm_r_err,
        "wm_l_state":       wm_l_st,
        "wm_r_state":       wm_r_st,
        "wm_mode":          wm_mode,
    }


def decode_telem_b(payload: bytes) -> dict:
    (tof0, tof1, tof2, tof3,
     roll_rate, yaw_rate,
     accel_x, accel_y, accel_z,
     hip_l_vel, hip_r_vel,
     hip_l_cmd_p, hip_r_cmd_p, hip_l_cmd_v, hip_r_cmd_v,
     hip_l_cmd_kp, hip_r_cmd_kp, hip_l_cmd_kd, hip_r_cmd_kd,
     hip_l_cmd_tff, hip_r_cmd_tff,
     theta_ref, v_ref, omega_cmd_rds, tau_sym, tau_yaw,
     ff1, ff2,
     health_flags, imu_loss_pct, jump_state, loop_count,
     active_profile, pitch_trim_rad,
     esp32_link_ok, esp32_status_age_ms, uart_rx_drops, uart_seq_gaps,
     gain_sched_alpha, standup_state) = struct.unpack(FMT_TELEM_B, payload)
    _NO_DATA = 0xFFFF
    tof_front = min((d for d in [tof0, tof1] if d != _NO_DATA), default=_NO_DATA)
    tof_rear  = min((d for d in [tof2, tof3] if d != _NO_DATA), default=_NO_DATA)
    return {
        "tof_dist_mm":         [tof0, tof1, tof2, tof3],
        "tof_front_min_mm":    tof_front,
        "tof_rear_min_mm":     tof_rear,
        "roll_rate_rads":      roll_rate,
        "yaw_rate_rads":       yaw_rate,
        "accel_x_ms2":         accel_x,
        "accel_y_ms2":         accel_y,
        "accel_z_ms2":         accel_z,
        "hip_l_vel_rads":      hip_l_vel,
        "hip_r_vel_rads":      hip_r_vel,
        "hip_l_cmd_pos_rad":   hip_l_cmd_p,
        "hip_r_cmd_pos_rad":   hip_r_cmd_p,
        "hip_l_cmd_vel_rads":  hip_l_cmd_v,
        "hip_r_cmd_vel_rads":  hip_r_cmd_v,
        "hip_l_cmd_kp":        hip_l_cmd_kp,
        "hip_r_cmd_kp":        hip_r_cmd_kp,
        "hip_l_cmd_kd":        hip_l_cmd_kd,
        "hip_r_cmd_kd":        hip_r_cmd_kd,
        "hip_l_cmd_tff":       hip_l_cmd_tff,
        "hip_r_cmd_tff":       hip_r_cmd_tff,
        "theta_ref":           theta_ref,
        "v_ref":               v_ref,
        "omega_cmd_rds":       omega_cmd_rds,
        "tau_sym":             tau_sym,
        "tau_yaw":             tau_yaw,
        "ff1_out":             ff1,
        "ff2_out":             ff2,
        "health_flags":        health_flags,
        "imu_packet_loss_pct": imu_loss_pct,
        "jump_state":          jump_state,
        "loop_count":          loop_count,
        "active_profile":      active_profile,
        "pitch_trim_rad":      pitch_trim_rad,
        "esp32_link_ok":       bool(esp32_link_ok),
        "esp32_status_age_ms": esp32_status_age_ms,
        "uart_rx_drops":       uart_rx_drops,
        "uart_seq_gaps":       uart_seq_gaps,
        "gain_sched_alpha":    gain_sched_alpha,
        "standup_state":       standup_state,
    }


def decode_telem_full(payload: bytes) -> dict:
    """Decode one full 247-byte TelemetryPayload blob (as embedded in a LogRecord)."""
    return {
        **decode_telem_a(payload[:118]),
        **decode_telem_b(payload[118:247]),
    }
