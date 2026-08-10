"""Write genuine WLOG_FORMAT_V1 records and matching .PARAMS sidecars."""

from __future__ import annotations

import csv
from pathlib import Path
import struct
import sys
from typing import Iterable, Mapping

from .params_control import PARAMS_BY_NAME


ROOT = Path(__file__).resolve().parents[4]
GUI_ROOT = ROOT / "software" / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))
from tabs.telem_format import (  # noqa: E402
    FMT_TELEM_A,
    FMT_TELEM_B,
    TELEM_VERSION,
    decode_telem_full,
)

WLOG_FORMAT_V1 = 1
WLOG_SAMPLE_HZ = 500
_HEADER = struct.Struct("<8sBBHHI14s")
_TELEM_SIZE = struct.calcsize("<" + FMT_TELEM_A[1:] + FMT_TELEM_B[1:])
_RECORD_SIZE = 4 + _TELEM_SIZE


def telemetry_defaults() -> dict:
    return {
        "timestamp_ms": 0, "pitch_rad": 0.0, "pitch_rate_rads": 0.0,
        "wheel_vel_avg": 0.0, "hip_l_pos_rad": 0.0, "hip_r_pos_rad": 0.0,
        "whl_tau_l": 0.0, "whl_tau_r": 0.0, "roll_rad": 0.0, "yaw_rad": 0.0,
        "robot_state": 3, "fault_code": 0, "test_val": 0.0,
        "hip_l_torque_nm": 0.0, "hip_r_torque_nm": 0.0,
        "ibus_ch": [1500] * 14, "ibus_alive": True,
        "wm_l_vel_turns_s": 0.0, "wm_r_vel_turns_s": 0.0,
        "wm_l_pos_turns": 0.0, "wm_r_pos_turns": 0.0,
        "wm_l_vbus": 24.0, "wm_r_vbus": 24.0,
        "wm_l_error": 0, "wm_r_error": 0,
        "wm_l_state": 8, "wm_r_state": 8, "wm_mode": 3,
        "tof_dist_mm": [0xFFFF] * 4,
        "roll_rate_rads": 0.0, "yaw_rate_rads": 0.0,
        "accel_x_ms2": 0.0, "accel_y_ms2": 0.0, "accel_z_ms2": 9.81,
        "hip_l_vel_rads": 0.0, "hip_r_vel_rads": 0.0,
        "hip_l_cmd_pos_rad": 0.0, "hip_r_cmd_pos_rad": 0.0,
        "hip_l_cmd_vel_rads": 0.0, "hip_r_cmd_vel_rads": 0.0,
        "hip_l_cmd_kp": 0.0, "hip_r_cmd_kp": 0.0,
        "hip_l_cmd_kd": 0.0, "hip_r_cmd_kd": 0.0,
        "hip_l_cmd_tff": 0.0, "hip_r_cmd_tff": 0.0,
        "theta_ref": 0.0, "v_ref": 0.0, "omega_cmd_rds": 0.0,
        "tau_sym": 0.0, "tau_yaw": 0.0, "ff1_out": 0.0, "ff2_out": 0.0,
        "health_flags": 0, "imu_packet_loss_pct": 0, "jump_state": 0,
        "loop_count": 0, "active_profile": 0, "pitch_trim_rad": 0.0,
        "esp32_link_ok": True, "esp32_status_age_ms": 0,
        "uart_rx_drops": 0, "uart_seq_gaps": 0,
        "gain_sched_alpha": 0.0, "standup_state": 0,
    }


def pack_telemetry(values: Mapping) -> bytes:
    t = telemetry_defaults()
    t.update(values)
    ibus = list(t["ibus_ch"])
    tof = list(t["tof_dist_mm"])
    if len(ibus) != 14 or len(tof) != 4:
        raise ValueError("ibus_ch must have 14 entries and tof_dist_mm must have 4")
    part_a = struct.pack(
        FMT_TELEM_A,
        int(t["timestamp_ms"]),
        *map(float, (t["pitch_rad"], t["pitch_rate_rads"], t["wheel_vel_avg"],
                     t["hip_l_pos_rad"], t["hip_r_pos_rad"], t["whl_tau_l"],
                     t["whl_tau_r"], t["roll_rad"], t["yaw_rad"])),
        int(t["robot_state"]), int(t["fault_code"]),
        float(t["test_val"]), float(t["hip_l_torque_nm"]), float(t["hip_r_torque_nm"]),
        *map(int, ibus), int(bool(t["ibus_alive"])),
        *map(float, (t["wm_l_vel_turns_s"], t["wm_r_vel_turns_s"],
                     t["wm_l_pos_turns"], t["wm_r_pos_turns"],
                     t["wm_l_vbus"], t["wm_r_vbus"])),
        int(t["wm_l_error"]), int(t["wm_r_error"]),
        int(t["wm_l_state"]), int(t["wm_r_state"]), int(t["wm_mode"]),
    )
    part_b = struct.pack(
        FMT_TELEM_B,
        *map(int, tof),
        *map(float, (t["roll_rate_rads"], t["yaw_rate_rads"],
                     t["accel_x_ms2"], t["accel_y_ms2"], t["accel_z_ms2"],
                     t["hip_l_vel_rads"], t["hip_r_vel_rads"],
                     t["hip_l_cmd_pos_rad"], t["hip_r_cmd_pos_rad"],
                     t["hip_l_cmd_vel_rads"], t["hip_r_cmd_vel_rads"],
                     t["hip_l_cmd_kp"], t["hip_r_cmd_kp"],
                     t["hip_l_cmd_kd"], t["hip_r_cmd_kd"],
                     t["hip_l_cmd_tff"], t["hip_r_cmd_tff"],
                     t["theta_ref"], t["v_ref"], t["omega_cmd_rds"],
                     t["tau_sym"], t["tau_yaw"], t["ff1_out"], t["ff2_out"])),
        int(t["health_flags"]), int(t["imu_packet_loss_pct"]), int(t["jump_state"]),
        int(t["loop_count"]), int(t["active_profile"]), float(t["pitch_trim_rad"]),
        int(bool(t["esp32_link_ok"])), int(t["esp32_status_age_ms"]),
        int(t["uart_rx_drops"]), int(t["uart_seq_gaps"]),
        float(t["gain_sched_alpha"]), int(t["standup_state"]),
    )
    return part_a + part_b


class WlogWriter:
    def __init__(self, path: Path, params: Mapping[str, float],
                 start_millis: int = 0, sample_rate_hz: int = WLOG_SAMPLE_HZ):
        self.path = Path(path)
        self.params_path = self.path.with_suffix(".PARAMS")
        self.sample_rate_hz = int(sample_rate_hz)
        self._handle = self.path.open("wb")
        self._handle.write(_HEADER.pack(
            b"WLRLOG\0", WLOG_FORMAT_V1, TELEM_VERSION, _RECORD_SIZE,
            self.sample_rate_hz, int(start_millis) & 0xFFFFFFFF, bytes(14)))
        self._params = {name: float(value) for name, value in params.items()}
        self._events: list[tuple[int, str, int, str, float]] = []
        for name, value in sorted(self._params.items(), key=lambda item: PARAMS_BY_NAME[item[0]].id):
            definition = PARAMS_BY_NAME[name]
            if "command" not in definition.flags:
                self._events.append((0, "DUMP", definition.id, name, value))
        self.count = 0

    def write(self, t_micros: int, telemetry: Mapping) -> None:
        self._handle.write(struct.pack("<I", int(t_micros) & 0xFFFFFFFF))
        self._handle.write(pack_telemetry(telemetry))
        self.count += 1

    def param_change(self, t_micros: int, name: str, value: float) -> None:
        definition = PARAMS_BY_NAME[name]
        number = float(value)
        if not definition.min <= number <= definition.max:
            raise ValueError(f"{name}={number} outside schema bounds")
        self._params[name] = number
        # Match firmware logging: command/shadow params are carried in TELEM,
        # not in the persistent-configuration sidecar.
        if "command" not in definition.flags:
            self._events.append((int(t_micros) & 0xFFFFFFFF, "CHANGE",
                                 definition.id, name, number))

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()
        with self.params_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("# t_micros,event,id,name,value\n")
            csv.writer(handle, lineterminator="\n").writerows(self._events)

    def __enter__(self) -> "WlogWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def decode_record(blob: bytes) -> dict:
    """Test/helper decoder using the GUI's canonical telemetry decoder."""
    if len(blob) != _TELEM_SIZE:
        raise ValueError(f"expected {_TELEM_SIZE} telemetry bytes, got {len(blob)}")
    return decode_telem_full(blob)
