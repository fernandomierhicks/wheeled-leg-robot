from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from v4_twin_279mm_baseline.defaults import DEFAULT_PARAMS
from v4_twin_279mm_baseline.optimizer.firmware_search_space import (
    FIRMWARE_SPACE_BY_STAGE,
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
    assert np.isfinite(replay_result["worst_nrmse"])


def test_mujoco_default_path_is_firmware_controller():
    model, data = build_model_and_data(DEFAULT_PARAMS)
    init_sim(model, data, DEFAULT_PARAMS)
    controller = SimController(model, data, DEFAULT_PARAMS, rng_seed=2)
    tick = controller.tick(model, data)
    assert tick["mode"] == "FIRMWARE_BALANCE"
    assert 0.0 <= tick["gain_sched_alpha"] <= 1.0
