#!/usr/bin/env python3
"""Fly the digital twin with the TX15, live, in the interactive viewer.

    python twin/tools/fly_twin.py                 # fly it
    python twin/tools/fly_twin.py --check         # list HID devices, no sim
    python twin/tools/fly_twin.py --monitor       # print live channels, no sim

On the radio: plug in USB and choose **USB Joystick** when it asks -- not USB
Storage. The WLR ROBOT model already carries ``usbJoystickIfMode: JOYSTICK``.

Why bother: ``twin/params_control.py`` and the firmware's parameter table are
both generated from ``protocol/schema.json``, so a gain you find here is the
same named parameter on the robot. Driving it with the actual transmitter, on
the actual channel map, closes the last gap -- the sim stops being a model you
poke with scripted profiles and becomes a rehearsal of the thing you are about
to do on the floor.

What transfers, and what does not:

  transfers   stick feel, the channel map, command scaling, the arm/jump
              semantics, gain values through the shared schema
  does NOT    anything about the radio link itself. This is USB HID; there is
              no CRSF, no failsafe, no LQ. Link-loss behaviour has to be
              tested on the bench with a real receiver, by pulling its power.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
TWIN_ROOT = HERE.parents[2]          # v4_twin_279mm_baseline
sys.path.insert(0, str(TWIN_ROOT.parent))

from v4_twin_279mm_baseline.twin.tools.tx15_joystick import (  # noqa: E402
    ChannelDecoder, Tx15Joystick,
)

CHANNEL_NAMES = {
    1: "roll", 2: "vel", 3: "hip", 4: "yaw", 5: "log", 6: "jump",
    7: "tuneA", 8: "tuneB", 9: "prof", 10: "ARM", 11: "calib",
    12: "reset", 13: "tuneG", 14: "latch", 15: "lean", 16: "spare",
}


def list_devices() -> int:
    try:
        import pygame
    except ImportError:
        print("pygame is not installed:  pip install pygame")
        return 1
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("No HID joystick found.")
        print()
        print("On the radio: plug in USB, and when it asks, choose")
        print("'USB Joystick' -- NOT 'USB Storage'. If it did not ask, try")
        print("Radio settings -> SD-HC card / USB -> USB mode.")
        return 1
    for i in range(n):
        js = pygame.joystick.Joystick(i)
        js.init()
        print("[%d] %s -- %d axes, %d buttons"
              % (i, js.get_name(), js.get_numaxes(), js.get_numbuttons()))
        js.quit()
    pygame.quit()
    return 0


def monitor(joy: Tx15Joystick, decoder: ChannelDecoder) -> int:
    """Live channel dump. The fastest way to confirm the map before flying.

    This is the sim-side equivalent of the radio's Outputs screen, and it is
    worth a minute: it tells you whether every control lands where the model
    says it does, without a robot or a receiver anywhere near it.
    """
    print(joy.describe())
    print("\nMove every control. Ctrl-C to stop.\n")
    try:
        while True:
            ch = joy.poll()
            cmd = decoder.decode(ch)
            cols = "  ".join(
                "%s:%4d" % (CHANNEL_NAMES.get(n, str(n)), ch.get(n, 0))
                for n in sorted(ch))
            print("\r%s | v=%+.2f w=%+.2f hip=%.2f roll=%+.3f %s%s   "
                  % (cols, cmd.v_cmd_ms, cmd.omega_cmd_rds, cmd.radio_hip_cmd,
                     cmd.roll_cmd_rad,
                     "ARMED" if cmd.armed else "     ",
                     " LEAN" if cmd.lean_enabled else ""),
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")
    return 0


def fly(joy: Tx15Joystick, decoder: ChannelDecoder, duration: float) -> int:
    """Run the twin with the transmitter driving the command profiles.

    The seam is the scenario's profile callables. sim_loop already calls
    v_profile(t) / omega_profile(t) / roll_profile(t) / hip_profile(t) once per
    control tick, so binding those to a live joystick poll needs no change to
    the simulation core at all.
    """
    from v4_twin_279mm_baseline import sim_loop
    from v4_twin_279mm_baseline.scenarios.base import ScenarioConfig, WorldConfig
    from v4_twin_279mm_baseline.defaults import DEFAULT_PARAMS

    robot = DEFAULT_PARAMS.robot
    state = {"cmd": decoder.decode({})}

    def refresh(_t: float):
        # One poll per tick, shared by all four profile callables: polling
        # separately in each would sample the sticks at four slightly
        # different instants and shear the command set.
        state["cmd"] = decoder.decode(joy.poll())
        return state["cmd"]

    last_t = {"v": -1.0}

    def poll_once(t: float):
        if t != last_t["v"]:
            last_t["v"] = t
            refresh(t)
        return state["cmd"]

    def v_profile(t):
        return poll_once(t).v_cmd_ms

    def omega_profile(t):
        return poll_once(t).omega_cmd_rds

    def roll_profile(t):
        return poll_once(t).roll_cmd_rad

    def hip_profile(t):
        # radio_hip_cmd is normalised 0..1; the twin wants a hip angle.
        alpha = poll_once(t).radio_hip_cmd
        return robot.Q_RET + alpha * (robot.Q_EXT - robot.Q_RET)

    scenario = ScenarioConfig(
        name="tx15_live",
        display_name="TX15 live (joystick)",
        duration=duration,
        active_controllers=frozenset({"lqr", "velocity_pi", "yaw_pi"}),
        hip_mode="position",
        v_profile=v_profile,
        omega_profile=omega_profile,
        roll_profile=roll_profile,
        hip_profile=hip_profile,
        world=WorldConfig(),
        group="lqr",
        order=99.0,
    )

    print(joy.describe())
    print("\nFlying the twin. Sticks live. Ctrl-C to stop.\n")
    metrics = sim_loop.run(DEFAULT_PARAMS, scenario)
    print("\nDone.")
    for key in ("survived", "max_pitch", "max_roll", "wheel_travel_m"):
        if key in metrics:
            print("  %-16s %s" % (key, metrics[key]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="list HID devices and exit")
    ap.add_argument("--monitor", action="store_true",
                    help="print live channels without running the sim")
    ap.add_argument("--index", type=int, default=0, help="HID device index")
    ap.add_argument("--any", action="store_true",
                    help="use --index even if its name does not look like a TX15")
    ap.add_argument("--duration", type=float, default=600.0,
                    help="sim duration [s] (default 600)")
    args = ap.parse_args()

    if args.check:
        return list_devices()

    try:
        joy = Tx15Joystick(index=args.index,
                           name_hint="" if args.any else Tx15Joystick.DEFAULT_NAME_HINT)
    except RuntimeError as exc:
        print(exc)
        return 1
    except ImportError:
        print("pygame is not installed:  pip install pygame")
        return 1

    # Limits come from the shared schema, so sim and robot cannot drift apart
    # on what "full stick" means.
    try:
        from v4_twin_279mm_baseline.twin import params_control
        defaults = params_control.default_values()
        decoder = ChannelDecoder(
            vel_max=defaults["radio_vel_max"],
            yaw_max=defaults["radio_yaw_max"],
            roll_max=defaults["radio_roll_max"],
        )
    except Exception:
        # Falling back is fine for --monitor, but say so: silently using
        # different limits than the robot is exactly the drift this tool
        # exists to prevent.
        print("note: could not read radio_*_max from params_control; "
              "falling back to 0.5 / 1.0 / 0.1")
        decoder = ChannelDecoder(0.5, 1.0, 0.1)

    try:
        if args.monitor:
            return monitor(joy, decoder)
        return fly(joy, decoder, args.duration)
    finally:
        joy.close()


if __name__ == "__main__":
    sys.exit(main())
