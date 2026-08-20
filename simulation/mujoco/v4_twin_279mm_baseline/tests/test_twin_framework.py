from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from v4_twin_279mm_baseline.defaults import DEFAULT_PARAMS
from v4_twin_279mm_baseline.optimizer.firmware_search_space import (
    FIRMWARE_SPACE_BY_STAGE,
)
from v4_twin_279mm_baseline.optimizer.robust_mujoco import (
    INTEGRATED_SPACE, SPACE_BY_STAGE,
)
from v4_twin_279mm_baseline.robot_match import (
    control_snapshot_sha256, load_latest_firmware_params, load_robot_match,
)
from v4_twin_279mm_baseline.sim_loop import (
    SimController, build_model_and_data, init_sim,
)
from v4_twin_279mm_baseline.twin.firmware_control import GAIN_ALLOWLIST
from v4_twin_279mm_baseline.twin.params_control import (
    PARAMS_BY_NAME, SCHEMA_SHA256,
)
from v4_twin_279mm_baseline.twin.runtime import run_scenario
from v4_twin_279mm_baseline.twin.scenario import Scenario, ScenarioEvent
from v4_twin_279mm_baseline.twin.tools.param_snapshot import (
    load_snapshot, snapshot_from_live, write_snapshot,
)
from v4_twin_279mm_baseline.twin.tools.push_params import planned_changes
from v4_twin_279mm_baseline.twin.tools.replay_wlog import replay
from v4_twin_279mm_baseline.twin.tools.validate_robot_match import simulate_case


ROOT = Path(__file__).resolve().parents[4]
GUI = ROOT / "software" / "gui"
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))
from analysis.wlog_metrics import decode_wlog  # noqa: E402


def test_generated_params_and_plant_id_contract():
    schema = json.loads((ROOT / "firmware/robot_teensy/protocol/schema.json").read_text())
    assert len(PARAMS_BY_NAME) == len(schema["parameters"])
    assert len(SCHEMA_SHA256) == 64
    for name in ("plant_id_en", "plant_id_amp", "plant_id_f0",
                 "plant_id_f1", "plant_id_dur"):
        assert name in PARAMS_BY_NAME
        assert "persistent" not in PARAMS_BY_NAME[name].flags


def test_optimizer_uses_only_schema_bounded_pushable_gains():
    rng = np.random.default_rng(3)
    for space in FIRMWARE_SPACE_BY_STAGE.values():
        candidate = space.random_init(rng)
        assert space.in_bounds(candidate)
        assert set(candidate) <= GAIN_ALLOWLIST
        for name, value in candidate.items():
            definition = PARAMS_BY_NAME[name]
            assert definition.min <= value <= definition.max


def test_robust_mujoco_search_is_schema_bounded_and_dry_run_pushable():
    protected = {
        "lqr_pitch_trim_ret", "lqr_pitch_trim_ext", "lqr_torque_limit",
        "pitch_watchdog_en", "pitch_watchdog_fwd", "pitch_watchdog_bwd",
        "roll_offset_max",
    }
    for space in SPACE_BY_STAGE.values():
        assert set(space.names) <= GAIN_ALLOWLIST
        assert set(space.names).isdisjoint(protected)
        for name, spec in space.params.items():
            definition = PARAMS_BY_NAME[name]
            assert definition.min <= spec.lo <= spec.hi <= definition.max

    candidate = load_snapshot(
        ROOT / "software" / "gui" / "parameter_exports"
        / "Robust_balance_candidate_2026-08-11.json")
    live = load_snapshot(
        ROOT / "software" / "gui" / "parameter_exports" / "Default gains.json")
    changes = planned_changes(candidate, live)
    assert {change["name"] for change in changes} == set(INTEGRATED_SPACE.names)
    assert len(candidate) == len(PARAMS_BY_NAME)
    assert control_snapshot_sha256(candidate) == (
        "9248a7c574614d31577a05aa98bd618369a60ab1919a76231ed9d510fee0fb55"
    )


def test_snapshot_round_trip_and_push_guard(tmp_path):
    live_items = [
        {"id": PARAMS_BY_NAME["lqr_k_pitch_ret"].id,
         "name": "lqr_k_pitch_ret", "value": -0.3},
        {"id": PARAMS_BY_NAME["wm_vel_limit"].id,
         "name": "wm_vel_limit", "value": PARAMS_BY_NAME["wm_vel_limit"].default},
    ]
    snapshot = snapshot_from_live(live_items, source="test")
    path = tmp_path / "params.json"
    write_snapshot(path, snapshot)
    values = load_snapshot(path)
    assert values["lqr_k_pitch_ret"] == -0.3
    changes = planned_changes({"lqr_k_pitch_ret": -0.4},
                              {"lqr_k_pitch_ret": -0.3})
    assert changes[0]["new"] == -0.4


def test_latest_gui_export_loads_despite_float32_endpoint_roundoff():
    values = load_snapshot(
        ROOT / "software" / "gui" / "parameter_exports" / "Default gains.json")
    assert values["standup_pitch_min"] == PARAMS_BY_NAME["standup_pitch_min"].min
    assert values["lqr_k_pitch_ret"] == -0.5
    assert values["lqr_torque_limit"] == 0.4


def test_default_profile_is_robot_matched_and_provenance_backed():
    firmware = dict(DEFAULT_PARAMS.firmware_params)
    wheel = DEFAULT_PARAMS.motors.wheel
    robot = DEFAULT_PARAMS.robot
    report = load_robot_match()
    assert len(firmware) == report["control_snapshot"]["parameter_count"]
    assert firmware["lqr_pitch_trim_ret"] == pytest.approx(-0.125, abs=1e-6)
    assert robot.calib_backoff_rad == pytest.approx(math.radians(1.0))
    assert robot.box_cg_x == 0.0
    assert robot.box_cg_z == 0.0
    assert robot.battery_cg_x == pytest.approx(0.010)
    assert robot.battery_cg_z == pytest.approx(-0.034)
    assert robot.total_mass_without_battery == pytest.approx(3.242)
    assert robot.total_mass == pytest.approx(3.518)
    assert wheel.current_limit == 10.0
    assert wheel.command_torque_scale == pytest.approx(3.41071436)
    assert wheel.torque_limit == pytest.approx(1.36428571)
    assert DEFAULT_PARAMS.motors.hip.torque_scale_ret == pytest.approx(1.25)
    assert DEFAULT_PARAMS.motors.hip.torque_scale_ext == pytest.approx(0.592736)
    assert report["control_snapshot"]["sha256"] == control_snapshot_sha256(firmware)
    fitted = [row for row in report["hip_drive"]["anchors"] if row["used_for_fit"]]
    assert len(fitted) == 2
    assert max(abs(row["sag_residual_rad"]) for row in fitted) < 0.003
    assert max(abs(row["torque_residual_nm"]) for row in fitted) < 0.1


def test_controller_snapshot_lock_rejects_export_drift(tmp_path):
    report = load_robot_match()
    report["control_snapshot"]["sha256"] = "0" * 64
    path = tmp_path / "stale_robot_match.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="controller snapshot changed"):
        load_latest_firmware_params(path)


def test_box_inertial_position_contains_fitted_box_cg_and_hip_motors():
    model, _ = build_model_and_data(DEFAULT_PARAMS)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    robot = DEFAULT_PARAMS.robot
    combined_mass = robot.m_box + 2.0 * robot.motor_mass
    expected_x = robot.m_box * robot.box_cg_x / combined_mass
    expected_z = (robot.m_box * robot.box_cg_z
                  + 2.0 * robot.motor_mass * robot.A_Z) / combined_mass
    assert model.body_ipos[box_id, 0] == pytest.approx(expected_x, abs=1e-6)
    assert model.body_ipos[box_id, 2] == pytest.approx(expected_z, abs=1e-6)
    battery_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "battery")
    assert model.body_mass[battery_id] == pytest.approx(robot.m_battery)
    assert model.body_pos[battery_id, 0] == pytest.approx(robot.battery_cg_x)
    assert model.body_pos[battery_id, 2] == pytest.approx(robot.battery_cg_z)
    assert model.body_subtreemass[box_id] == pytest.approx(robot.total_mass)


def test_robot_matched_mujoco_balances_at_logged_leg_height():
    result = simulate_case(0.72947, 5.0, duration_s=3.0, seed=9)
    assert result["stable"]
    assert result["firmware_fault"] is None


def test_wlog_gui_decode_and_replay(tmp_path):
    scenario = Scenario(
        name="smoke", duration_s=0.02, stage="lqr",
        initial_params={"pitch_watchdog_en": 0.0},
        initial_state={"pitch_rad": -0.14},
        events=(ScenarioEvent(0.01, {"v_cmd_ms": 0.05}),),
    )
    path = tmp_path / "smoke.WLOG"
    result = run_scenario(scenario, path, seed=4)
    decoded = decode_wlog(path)
    assert result["samples"] == decoded.count
    assert decoded.telem_version == 12
    replay_result = replay(path, mode="closed")
    assert replay_result["samples"] == decoded.count
    assert replay_result["plant"] == "mujoco"
    assert replay_result["controller_fit_allowed"] is False
    assert replay_result["coverage_fraction"] > 0.0
    assert np.isfinite(replay_result["worst_nrmse"])
    open_result = replay(path, mode="open")
    assert open_result["window_s"] == pytest.approx(0.1)
    assert open_result["coverage_fraction"] > 0.0


def test_mujoco_default_path_is_firmware_controller():
    model, data = build_model_and_data(DEFAULT_PARAMS)
    init_sim(model, data, DEFAULT_PARAMS)
    controller = SimController(model, data, DEFAULT_PARAMS, rng_seed=2)
    tick = controller.tick(model, data)
    assert tick["mode"] == "FIRMWARE_BALANCE"
    assert 0.0 <= tick["gain_sched_alpha"] <= 1.0
