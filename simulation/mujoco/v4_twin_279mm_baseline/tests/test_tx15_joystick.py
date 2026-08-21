"""Tests for the TX15 -> twin input path.

The decode half is pure and is the half that matters: it has to agree with
``radio_update()`` in firmware/robot_teensy/teensy/src/main.cpp, or a gain
tuned in sim is being tuned against a different machine than the one on the
bench, which is the one failure this whole feature exists to avoid.

No hardware needed; pygame is only touched by the device class, which these
tests do not construct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v4_twin_279mm_baseline.twin.tools.tx15_joystick import (  # noqa: E402
    AXIS_TO_CHANNEL, BUTTON_TO_CHANNEL, CH_HIGH_US, CH_LOW_US,
    ChannelDecoder, axis_to_us, button_to_us,
)

VEL_MAX, YAW_MAX, ROLL_MAX = 0.5, 1.0, 0.17


def dec() -> ChannelDecoder:
    return ChannelDecoder(VEL_MAX, YAW_MAX, ROLL_MAX)


# ── HID -> microseconds ──────────────────────────────────────────────────────

def test_axis_endpoints_reach_the_firmware_thresholds():
    # An axis at full deflection has to clear the same absolute microsecond
    # literals the firmware tests, or arming and both stick combos silently
    # stop working in sim while still working on the robot.
    assert axis_to_us(-1.0) == 1000
    assert axis_to_us(0.0) == 1500
    assert axis_to_us(1.0) == 2000
    assert axis_to_us(1.0) > CH_HIGH_US
    assert axis_to_us(-1.0) < CH_LOW_US


def test_axis_clamps_out_of_range():
    assert axis_to_us(-5.0) == 1000
    assert axis_to_us(5.0) == 2000


def test_button_is_only_ever_low_or_high():
    # EdgeTX maps channels 9-16 to buttons as "on if channel > 0", so a
    # three-position switch loses its middle position over USB. Pinning this
    # keeps the limitation visible rather than surprising.
    assert button_to_us(True) == 2000
    assert button_to_us(False) == 1000
    assert button_to_us(True) > CH_HIGH_US
    assert button_to_us(False) < CH_LOW_US


def test_edgetx_hid_layout():
    # radio/src/usb_joystick.cpp: X,Y,Z,Rx,Ry,Rz,S1,S2 -> channels 1..8;
    # buttons 1..8 -> channels 9..16.
    assert AXIS_TO_CHANNEL == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
    assert BUTTON_TO_CHANNEL[0] == 9
    assert BUTTON_TO_CHANNEL[7] == 16


# ── channels -> commands, mirroring radio_update() ───────────────────────────

def test_sticks_centred_command_nothing():
    c = dec().decode({1: 1500, 2: 1500, 3: 1000, 4: 1500})
    assert c.v_cmd_ms == pytest.approx(0.0)
    assert c.omega_cmd_rds == pytest.approx(0.0)
    assert c.roll_cmd_rad == pytest.approx(0.0)
    assert c.radio_hip_cmd == pytest.approx(0.0)


def test_velocity_scales_by_radio_vel_max():
    assert dec().decode({2: 2000}).v_cmd_ms == pytest.approx(VEL_MAX)
    assert dec().decode({2: 1000}).v_cmd_ms == pytest.approx(-VEL_MAX)


def test_yaw_is_inverted_so_stick_left_yaws_left():
    # main.cpp negates CH4: "inverted: stick left -> robot yaws left". Stick
    # left is a LOW channel, and yawing left is POSITIVE omega.
    assert dec().decode({4: 1000}).omega_cmd_rds == pytest.approx(YAW_MAX)
    assert dec().decode({4: 2000}).omega_cmd_rds == pytest.approx(-YAW_MAX)


def test_roll_is_not_inverted():
    # Stick right is a HIGH channel and positive roll. The asymmetry with yaw
    # is real and in the firmware; this test exists so nobody "fixes" it.
    assert dec().decode({1: 2000}).roll_cmd_rad == pytest.approx(ROLL_MAX)
    assert dec().decode({1: 1000}).roll_cmd_rad == pytest.approx(-ROLL_MAX)


def test_hip_is_unipolar():
    # Leg height is 0..1 across the whole stick, not bipolar about centre.
    assert dec().decode({3: 1000}).radio_hip_cmd == pytest.approx(0.0)
    assert dec().decode({3: 1500}).radio_hip_cmd == pytest.approx(0.5)
    assert dec().decode({3: 2000}).radio_hip_cmd == pytest.approx(1.0)


def test_arm_needs_the_full_endpoint():
    # armed = ch10 > 1990. A switch that lands a few ticks short does not arm,
    # which is the failure the plan warns about on real ELRS endpoints.
    assert dec().decode({10: 2000}).armed
    assert not dec().decode({10: 1990}).armed
    assert not dec().decode({10: 1500}).armed


def test_lean_button():
    assert dec().decode({15: 2000}).lean_enabled
    assert not dec().decode({15: 1000}).lean_enabled


# ── jump is an edge, not a level ─────────────────────────────────────────────

def test_jump_is_one_edge_per_press():
    d = dec()
    assert not d.decode({6: 1000}).jump_edge
    assert d.decode({6: 2000}).jump_edge, "press must produce exactly one edge"
    assert not d.decode({6: 2000}).jump_edge, "holding must not repeat"
    assert not d.decode({6: 1000}).jump_edge
    assert d.decode({6: 2000}).jump_edge, "release then press fires again"


# ── dead link ────────────────────────────────────────────────────────────────

def test_dead_link_commands_nothing_and_cannot_arm():
    # Mirrors the driver's FAILSAFE CONTRACT: channels read 0, which fails
    # every "> 1990" test. The sim should behave the same way the robot does
    # when the transmitter goes away mid-run.
    d = dec()
    live = {1: 2000, 2: 2000, 4: 2000, 6: 2000, 10: 2000, 15: 2000}
    assert d.decode(live).armed

    c = d.decode(live, link_alive=False)
    assert not c.armed
    assert not c.lean_enabled
    assert not c.jump_edge
    assert c.v_cmd_ms == pytest.approx(0.0)
    assert c.omega_cmd_rds == pytest.approx(0.0)
    assert not c.link_alive


def test_dead_link_clears_the_jump_edge_latch():
    # Coming back from a dropout with the button still held reads as a new
    # press. That is the safe direction: at worst a jump you asked for is
    # missed, never one you did not.
    d = dec()
    assert d.decode({6: 2000}).jump_edge
    assert not d.decode({6: 2000}).jump_edge
    d.decode({6: 2000}, link_alive=False)
    assert d.decode({6: 2000}).jump_edge


def test_absent_channel_is_neutral_not_full_reverse():
    # The bug this test was written to catch: a channel the transmitter never
    # sent arrives as 0, and (0 - 1500) / 500 clamps to -1.0 -- full reverse
    # velocity and full left roll, from a control that is simply not present.
    # Absent must mean neutral; only a genuinely dead link zeroes everything.
    c = dec().decode({10: 2000})          # only ARM present
    assert c.v_cmd_ms == pytest.approx(0.0)
    assert c.omega_cmd_rds == pytest.approx(0.0)
    assert c.roll_cmd_rad == pytest.approx(0.0)
    assert c.radio_hip_cmd == pytest.approx(0.0)   # leg height low, correct
    assert c.armed


def test_absent_switches_read_low():
    c = dec().decode({1: 1500, 2: 1500})
    assert not c.armed
    assert not c.lean_enabled
    assert not c.jump_edge


# ── end-to-end wiring ────────────────────────────────────────────────────────

def test_synthetic_pilot_drives_the_twin():
    """The integration test: a scripted 'pilot' flies the twin through the
    same seam the real transmitter uses.

    The value here is narrow and specific. The joystick binds to the
    scenario's profile callables, which sim_loop calls once per control tick.
    If that seam ever moves -- a renamed field, a changed signature, profiles
    stopping being callables -- nothing else in this file notices, and the
    failure would surface the first time the radio is plugged in, which is the
    worst moment to discover it.

    Short duration on purpose: this checks the wiring, not the controller.
    """
    import math

    from v4_twin_279mm_baseline import sim_loop
    from v4_twin_279mm_baseline.scenarios.base import ScenarioConfig, WorldConfig
    from v4_twin_279mm_baseline.defaults import DEFAULT_PARAMS
    from v4_twin_279mm_baseline.twin import params_control
    from v4_twin_279mm_baseline.twin.tools.tx15_joystick import axis_to_us

    limits = params_control.default_values()
    decoder = ChannelDecoder(limits["radio_vel_max"],
                             limits["radio_yaw_max"],
                             limits["radio_roll_max"])
    robot = DEFAULT_PARAMS.robot

    def channels(t):
        # Sit still, then ask for forward. Sticks centred elsewhere.
        return {1: axis_to_us(0.0), 2: axis_to_us(0.0 if t < 0.5 else 0.6),
                3: axis_to_us(0.0), 4: axis_to_us(0.0), 10: 2000}

    state = {"t": -1.0, "cmd": decoder.decode({})}

    def poll(t):
        if t != state["t"]:
            state["t"] = t
            state["cmd"] = decoder.decode(channels(t))
        return state["cmd"]

    scenario = ScenarioConfig(
        name="tx15_synthetic_pilot",
        display_name="TX15 synthetic pilot",
        duration=2.5,
        active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
        hip_mode="position",
        v_profile=lambda t: poll(t).v_cmd_ms,
        omega_profile=lambda t: poll(t).omega_cmd_rds,
        roll_profile=lambda t: poll(t).roll_cmd_rad,
        hip_profile=lambda t: robot.Q_RET
                              + poll(t).radio_hip_cmd * (robot.Q_EXT - robot.Q_RET),
        world=WorldConfig(), group="lqr", order=99.0,
    )

    metrics = sim_loop.run(DEFAULT_PARAMS, scenario)

    # It must have gone somewhere: a broken seam leaves v_cmd at zero and the
    # robot balancing on the spot, which is a silent, plausible-looking pass.
    travel = metrics.get("wheel_travel_m", 0.0)
    assert travel > 0.2, (
        "the synthetic pilot commanded forward velocity but the robot barely "
        "moved (%.3f m) -- the joystick-to-profile seam is probably broken"
        % travel)
    assert math.isfinite(travel)
