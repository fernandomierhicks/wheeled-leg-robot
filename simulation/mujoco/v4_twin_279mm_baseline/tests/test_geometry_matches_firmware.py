"""Acceptance anchors shared by CAD, firmware, and the v4 digital twin."""

import math

import pytest

from v4_twin_279mm_baseline.params import RobotGeometry
from v4_twin_279mm_baseline.physics import (
    alpha_to_hip_q,
    firmware_hip_to_sim_q,
    hip_q_to_alpha,
    sim_q_to_firmware_hip,
    solve_ik,
)


@pytest.fixture(scope="module")
def robot() -> RobotGeometry:
    return RobotGeometry()


def _ik(robot: RobotGeometry, q: float) -> dict:
    pose = solve_ik(q, robot.as_dict())
    assert pose is not None
    return pose


def test_alpha_endpoints_match_firmware_l_eff(robot: RobotGeometry) -> None:
    ret = _ik(robot, alpha_to_hip_q(0.0, robot))
    ext = _ik(robot, alpha_to_hip_q(1.0, robot))
    assert abs(ret["W_z"]) == pytest.approx(0.098915, abs=5e-7)
    assert abs(ext["W_z"]) == pytest.approx(0.363396, abs=5e-7)


def test_hard_stop_stroke_and_ride_height(robot: RobotGeometry) -> None:
    ret = _ik(robot, robot.Q_RET)
    ext = _ik(robot, robot.Q_EXT)

    stroke = ret["W_z"] - ext["W_z"]
    ride_ret = robot.wheel_r - (ret["W_z"] - robot.A_Z)
    ride_ext = robot.wheel_r - (ext["W_z"] - robot.A_Z)

    assert stroke == pytest.approx(0.27662, abs=5e-6)
    assert ride_ret == pytest.approx(0.1193, abs=5e-5)
    assert ride_ext == pytest.approx(0.3959, abs=5e-5)


def test_alpha_uses_calibrated_span_not_hard_stop(robot: RobotGeometry) -> None:
    assert math.degrees(alpha_to_hip_q(0.0, robot)) == pytest.approx(23.0, abs=1e-5)
    assert math.degrees(alpha_to_hip_q(1.0, robot)) == pytest.approx(-57.0, abs=1e-8)
    assert hip_q_to_alpha(robot.Q_RET, robot) == 0.0
    assert hip_q_to_alpha(alpha_to_hip_q(0.5, robot), robot) == pytest.approx(0.5)


def test_named_firmware_sim_transform_round_trips() -> None:
    robot = RobotGeometry()
    for q in map(math.radians, (-57.0, -10.0, 0.0, 23.0, 28.0)):
        assert sim_q_to_firmware_hip(
            firmware_hip_to_sim_q(q, robot), robot) == pytest.approx(q)
    assert firmware_hip_to_sim_q(
        -robot.calib_backoff_rad, robot) == pytest.approx(robot.Q_ALPHA_RET)
