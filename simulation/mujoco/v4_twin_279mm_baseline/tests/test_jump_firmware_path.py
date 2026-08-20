from __future__ import annotations

import pytest

from v4_twin_279mm_baseline.controllers.jump import (
    JumpController, RobotMode, _retract_brake_duration, _retract_brake_sample,
)
from v4_twin_279mm_baseline.defaults import DEFAULT_PARAMS
from v4_twin_279mm_baseline.twin.params_control import default_values
from v4_twin_279mm_baseline.twin.tools import jump_study


def _jump_params() -> dict[str, float]:
    values = default_values()
    values.update({
        "jump_enable": 1.0,
        "jump_crouch_angle": 0.30,
        "jump_crouch_speed": 3.0,
        "jump_extend_angle": 1.20,
        "jump_retract_speed": 3.0,
        "jump_retract_angle": 0.65,
        "jump_land_min_air": 0.16,
        "jump_land_timeout": 0.8,
        "jmp_handoff_pitch": 0.08,
        "jmp_handoff_rate": 1.5,
        "jmp_handoff_hold_s": 0.02,
        "jmp_handoff_timeout": 0.5,
    })
    return values


def test_forward_nudge_only_occupies_final_crouch_window() -> None:
    controller = JumpController(DEFAULT_PARAMS.robot, _jump_params(), 0.002)
    controller.trigger()
    start = controller.update(
        0.0, DEFAULT_PARAMS.robot.Q_NOM, 0.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert start.mode == RobotMode.CROUCH
    assert start.velocity_offset_ms == 0.0

    nudging = controller.update(
        controller._crouch_duration - 0.05,
        DEFAULT_PARAMS.robot.Q_NOM, 0.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert nudging.mode == RobotMode.CROUCH
    assert nudging.velocity_offset_ms == pytest.approx(0.15)


def test_retract_brake_is_velocity_continuous_and_respects_hardstop_margin() -> None:
    duration = _retract_brake_duration(-1.37, -7.0, -1.4835, 0.0873)
    q0, dq0 = _retract_brake_sample(-1.37, -7.0, 0.0, duration)
    q1, dq1 = _retract_brake_sample(-1.37, -7.0, duration, duration)

    assert q0 == pytest.approx(-1.37)
    assert dq0 == pytest.approx(-7.0)
    assert duration < 0.015
    assert q1 == pytest.approx(-1.3962)
    assert dq1 == pytest.approx(0.0)


def test_landing_blanking_one_tick_phase_and_handoff_capture() -> None:
    robot = DEFAULT_PARAMS.robot
    controller = JumpController(robot, _jump_params(), 0.002)
    controller.trigger()
    controller.update(0.0, robot.Q_NOM, 0.0,
                      (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    extend_at = controller._crouch_duration
    extension = controller.update(
        extend_at, controller._crouch_q, 0.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert extension.mode == RobotMode.EXTEND

    retract_at = extend_at + 0.002
    retract = controller.update(
        retract_at, extension.q_hip_target, 0.0,
        (-10.0, 0.0, -10.0), (0.0, 0.0, 0.0))
    assert retract.mode == RobotMode.RETRACT
    assert retract.airborne_seen
    assert retract.hip_torque_limit_nm == pytest.approx(7.0)

    still_airborne = controller.update(
        retract_at + 0.11, extension.q_hip_target, 0.0,
        (0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    assert still_airborne.mode == RobotMode.RETRACT

    landing_at = retract_at + 0.17
    landing = controller.update(
        landing_at, extension.q_hip_target, 0.0,
        (0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    assert landing.mode == RobotMode.LANDING
    assert landing.landing_source == "accel rebound"

    handoff_at = landing_at + 0.002
    handoff = controller.update(
        handoff_at, extension.q_hip_target, 0.0,
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert handoff.mode == RobotMode.HANDOFF
    controller.post_control(
        handoff_at, pitch=0.0, pitch_rate=0.0,
        wheel_l_turns_s=0.0, wheel_r_turns_s=0.0,
        theta_ref=0.0, pitch_trim=0.0)
    controller.post_control(
        handoff_at + 0.022, pitch=0.0, pitch_rate=0.0,
        wheel_l_turns_s=0.0, wheel_r_turns_s=0.0,
        theta_ref=0.0, pitch_trim=0.0)
    assert controller.complete


def test_log_matched_jump_reproduces_launch_and_detects_true_contact() -> None:
    params, _values = jump_study._reference_params()
    case = jump_study.REFERENCE_CASES[0]
    result = jump_study.score_run(case, params, seed=20260813, keep_trace=True)

    assert not result["fell"]
    assert not result["fault"]
    assert result["peak_wheel_clearance_m"] > 0.025
    assert result["phase_s"]["EXTEND"] == pytest.approx(
        case.phase_s["EXTEND"], abs=0.01)
    assert result["phase_s"]["RETRACT"] == pytest.approx(
        case.phase_s["RETRACT"], abs=0.05)
    assert 0.0 <= result["landing_detection_error_s"] <= 0.025
    retract_rows = [row for row in result["trace"] if row["mode"] == "RETRACT"]
    assert retract_rows
    assert max(abs(row[side]) for row in retract_rows
               for side in ("tau_hip_L", "tau_hip_R")) <= 7.0 + 1e-9


def test_airborne_wheels_have_material_pitch_authority() -> None:
    params, reference = jump_study._reference_params()
    base, _fixed = jump_study._search_base(reference)
    values = dict(reference, jmp_handoff_torque=0.4)
    authority = jump_study.reaction_wheel_authority(
        jump_study._with_values(base, values))
    current_limit = authority["results"][0]
    assert current_limit["wheel_speed_limit_turns_s"] == 6.0
    assert abs(current_limit["body_pitch_change_deg_vs_zero"]) > 2.0
    assert current_limit["time_to_speed_limit_s"] < 0.05
