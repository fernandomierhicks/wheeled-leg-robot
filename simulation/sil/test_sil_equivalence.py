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
    # _USE_MATH_DEFINES: the Teensy toolchain provides M_PI from <math.h>; MSVC
    # only does when asked.
    cl_command = ["cl.exe", "/nologo", "/std:c++20", "/EHsc", "/O2",
                  "/D_USE_MATH_DEFINES",
                  "/Fe:" + str(executable), "/Fo:" + str(tmp_path) + "\\",
                  *sources_and_includes]
    shell_command = f'call "{candidates[0]}" -arch=x64 >nul && ' + subprocess.list2cmdline(cl_command)
    result = subprocess.run(shell_command, cwd=ROOT, shell=True,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


HEADER = ("time_ms,pitch,pitch_rate,roll,roll_rate,yaw_rate,wheel_l_turns_s,"
          "wheel_r_turns_s,alpha,hip_l_torque,hip_r_torque,v_cmd,omega_cmd,hip_cmd")


def _parked_trim_learn_vectors() -> str:
    """10 s parked at alpha 0.3, 4 mrad off the default trim: arms the learner."""
    rows = [HEADER]
    for tick in range(5000):
        rows.append(f"{2 * tick},-0.079100,0.000000,0.000000,0.000000,0.000000,"
                    "0.000000,0.000000,0.300000,0.000000,0.000000,0.000000,"
                    "0.000000,0.300000")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def harness(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("sil"))


@pytest.mark.parametrize("case", ["golden", "golden_encoder_yaw", "trim_learn"])
def test_production_cpp_matches_python(harness, case):
    from v4_twin_279mm_baseline.twin.firmware_control import ControlInput, FirmwareController
    from v4_twin_279mm_baseline.twin.params_control import PARAMS_BY_NAME

    overrides = {
        "golden": {},
        "golden_encoder_yaw": {"yaw_rate_src": 1.0},
        "trim_learn": {"trim_learn_en": 1.0},
    }[case]
    vector_text = (_parked_trim_learn_vectors() if case == "trim_learn"
                   else (SIL / "golden_vectors.csv").read_text(encoding="utf-8"))
    args = [f"{PARAMS_BY_NAME[name].id}={value}" for name, value in overrides.items()]
    native = subprocess.run([str(harness), *args], input=vector_text, text=True,
                            capture_output=True, check=True)
    actual = list(csv.DictReader(io.StringIO(native.stdout)))
    inputs = list(csv.DictReader(io.StringIO(vector_text)))
    controller = FirmwareController({
        "pitch_watchdog_en": 0.0, "roll_watchdog_en": 0.0,
        "hip_running_ramp_s": 0.0, **overrides,
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
    if case == "trim_learn":
        # Guard against a vacuous pass: the learner must actually have moved the trim.
        assert float(actual[-1]["pitch_trim"]) != pytest.approx(
            float(actual[0]["pitch_trim"]), abs=1e-4)
