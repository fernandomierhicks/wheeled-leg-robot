"""Compile production control_loop.cpp and compare it with the Python port."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SIL = Path(__file__).resolve().parent
FIRMWARE = ROOT / "firmware" / "robot_teensy" / "teensy"
sys.path.insert(0, str(ROOT / "simulation" / "mujoco"))


def _build(tmp_path: Path) -> Path:
    executable = tmp_path / "control_loop_harness.exe"
    rel = lambda path: str(Path(path).relative_to(ROOT))
    sources_and_includes = [
        rel(SIL / "control_loop_harness.cpp"),
        rel(FIRMWARE / "src" / "control_loop.cpp"),
        "/I" + rel(SIL / "stubs"), "/I" + rel(FIRMWARE / "src"),
        "/I" + rel(FIRMWARE / "lib" / "HipMotors"),
        "/I" + rel(FIRMWARE / "lib" / "WheelMotors"),
        "/I" + rel(FIRMWARE / "lib" / "ParamRegistry"),
        "/I" + rel(FIRMWARE / "lib" / "IMU"),
        "/I" + rel(FIRMWARE.parent / "shared"),
    ]
    candidates = sorted(Path("C:/Program Files (x86)/Microsoft Visual Studio").glob(
        "*/BuildTools/Common7/Tools/VsDevCmd.bat"), reverse=True)
    if not candidates:
        pytest.skip("Visual Studio C++ Build Tools are not installed")
    cl_command = ["cl.exe", "/nologo", "/std:c++20", "/EHsc", "/O2",
                  "/Fe:" + str(executable), "/Fo:" + str(tmp_path) + "\\",
                  *sources_and_includes]
    shell_command = f'call "{candidates[0]}" -arch=x64 >nul && ' + subprocess.list2cmdline(cl_command)
    result = subprocess.run(shell_command, cwd=ROOT, shell=True,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


def test_production_cpp_matches_python(tmp_path):
    from v4_twin_279mm_baseline.twin.firmware_control import ControlInput, FirmwareController

    executable = _build(tmp_path)
    vector_text = (SIL / "golden_vectors.csv").read_text(encoding="utf-8")
    native = subprocess.run([str(executable)], input=vector_text, text=True,
                            capture_output=True, check=True)
    actual = list(csv.DictReader(io.StringIO(native.stdout)))
    inputs = list(csv.DictReader(io.StringIO(vector_text)))
    controller = FirmwareController({
        "pitch_watchdog_en": 0.0, "roll_watchdog_en": 0.0,
        "hip_running_ramp_s": 0.0,
    })
    controller.reset(0.0, hip_alpha=0.0)
    for row, native_row in zip(inputs, actual, strict=True):
        time_s = float(row["time_ms"]) / 1000.0
        controller.params["v_cmd_ms"] = float(row["v_cmd"])
        controller.params["omega_cmd_rds"] = float(row["omega_cmd"])
        controller.params["radio_hip_cmd"] = float(row["hip_cmd"])
        result = controller.step(ControlInput(
            time_s=time_s, pitch_rad=float(row["pitch"]),
            pitch_rate_rads=float(row["pitch_rate"]),
            roll_rad=float(row["roll"]), roll_rate_rads=float(row["roll_rate"]),
            yaw_rate_rads=float(row["yaw_rate"]),
            wheel_l_turns_s=float(row["wheel_l_turns_s"]),
            wheel_r_turns_s=float(row["wheel_r_turns_s"]),
            hip_alpha=float(row["alpha"]),
            hip_l_torque_nm=float(row["hip_l_torque"]),
            hip_r_torque_nm=float(row["hip_r_torque"]), state="RUNNING",
        ))
        for name, expected in (
            ("tau_sym", result.tau_sym), ("tau_yaw", result.tau_yaw),
            ("theta_ref", result.theta_ref), ("tau_l", result.tau_l),
            ("tau_r", result.tau_r),
            ("alpha", result.gain_sched_alpha), ("pitch_trim", result.pitch_trim),
        ):
            assert math.isclose(float(native_row[name]), expected,
                                rel_tol=2e-5, abs_tol=2e-6), (row, name, native_row[name], expected)
