"""test_sensor_delay.py — Unit tests for LatencyBuffer and delay depth calculation.

Run from repo root:
    python -m pytest simulation/mujoco/master_sim/tests/test_sensor_delay.py -v

No MuJoCo install required.
"""
import sys
import os
import math

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from master_sim_jump.models.latency import LatencyBuffer
from master_sim_jump.params import LatencyParams, SimTiming


# ── Test 1: n_steps=0 is a true pass-through ─────────────────────────────────

def test_passthrough_returns_immediately():
    buf = LatencyBuffer(n_steps=0, init_value=0.0)
    assert buf.push(1.0) == 1.0
    assert buf.push(2.0) == 2.0
    assert buf.push(-5.0) == -5.0


def test_passthrough_works_for_tuple():
    buf = LatencyBuffer(n_steps=0, init_value=(0.0, 0.0))
    val = (1.23, 4.56)
    assert buf.push(val) == val


# ── Test 2: correct delay depth and sequencing ───────────────────────────────

def test_delay_depth_3():
    """Output at step k must equal input from step k-3 (pre-filled with 0)."""
    buf = LatencyBuffer(n_steps=3, init_value=0.0)
    inputs  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    results = [buf.push(v) for v in inputs]
    assert results == expected, f"got {results}, want {expected}"


def test_delay_depth_1():
    buf = LatencyBuffer(n_steps=1, init_value=0.0)
    assert buf.push(99.0) == 0.0   # first push returns init
    assert buf.push(50.0) == 99.0  # second push returns first input


def test_init_value_fills_entire_buffer():
    """Buffer should return init_value for the first n_steps calls."""
    init = (7.0, 8.0, 9.0)
    buf = LatencyBuffer(n_steps=4, init_value=init)
    for _ in range(4):
        assert buf.push((0.0, 0.0, 0.0)) == init


def test_reset_refills_with_new_value():
    buf = LatencyBuffer(n_steps=2, init_value=0.0)
    buf.push(10.0)
    buf.push(20.0)          # buffer now holds [10, 20]
    buf.reset(init_value=99.0)
    # After reset both slots should be 99.0
    assert buf.push(0.0) == 99.0
    assert buf.push(0.0) == 99.0


# ── Test 3: buffer depth calculation matches LatencyParams / SimTiming ───────

def test_buffer_depth_formula_default():
    """n_sens and n_act must equal round(delay_s / dt_ctrl)."""
    timing = SimTiming()        # 0.5 ms physics, 1 kHz control → dt_ctrl = 1 ms
    dt_ctrl = timing.sim_timestep * timing.ctrl_steps   # = 0.001 s
    assert math.isclose(dt_ctrl, 0.001, rel_tol=1e-9)

    latency = LatencyParams()   # defaults: 1 ms sensor + 1 ms actuator
    n_sens_expected = round(latency.sensor_delay_s  / dt_ctrl)
    n_act_expected  = round(latency.actuator_delay_s / dt_ctrl)
    assert n_sens_expected == 1
    assert n_act_expected  == 1


def test_buffer_depth_formula_custom():
    """Formula holds for arbitrary delay values."""
    timing  = SimTiming()
    dt_ctrl = timing.sim_timestep * timing.ctrl_steps

    latency = LatencyParams(sensor_delay_s=0.005, actuator_delay_s=0.002)
    assert round(latency.sensor_delay_s  / dt_ctrl) == 5
    assert round(latency.actuator_delay_s / dt_ctrl) == 2


def test_zero_delay_gives_zero_depth():
    timing  = SimTiming()
    dt_ctrl = timing.sim_timestep * timing.ctrl_steps

    latency = LatencyParams(sensor_delay_s=0.0, actuator_delay_s=0.0)
    assert round(latency.sensor_delay_s  / dt_ctrl) == 0
    assert round(latency.actuator_delay_s / dt_ctrl) == 0
