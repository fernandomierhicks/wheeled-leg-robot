# Flying the twin with the TX15

    python twin/tools/fly_twin.py --check     # list HID devices
    python twin/tools/fly_twin.py --monitor   # live channel dump, no sim
    python twin/tools/fly_twin.py             # fly it

## Setup

1. Plug the TX15 in over USB.
2. When it asks, choose **USB Joystick** — *not* USB Storage. The WLR ROBOT
   model already carries `usbJoystickIfMode: JOYSTICK`.
3. `--check` should list a device whose name contains `TX15`.

If `--check` shows other devices but no TX15, the radio is almost certainly in
USB Storage mode. Unplug, replug, pick Joystick. The tool deliberately
**refuses to bind to a non-TX15 device** rather than silently grabbing index 0
— on a machine with a 3D mouse or a gamepad attached, that is otherwise a
confusing way to spend twenty minutes.

## Why this is worth having

`twin/params_control.py` and the firmware's parameter table are both generated
from `protocol/schema.json`, so a gain found here is the *same named
parameter* on the robot. Driving the sim with the actual transmitter, on the
actual channel map, closes the last gap: the twin stops being a model you poke
with scripted profiles and becomes a rehearsal of what you are about to do on
the floor.

Concretely, it lets you:

- practise jump-ladder runs and hard turns at zero risk
- A/B a gain change before it touches hardware
- build muscle memory for the panic actions on the real controls

## What transfers, and what does not

| Transfers | Does **not** transfer |
|---|---|
| Stick feel and the channel map | Anything about the radio link |
| Command scaling (`radio_*_max`, from the shared schema) | CRSF framing, failsafe, LQ |
| Arm and jump semantics | Link-loss behaviour |
| Gain values, via the shared parameter namespace | Real ELRS endpoint quirks |

Link-loss behaviour has to be tested on the bench with a real receiver by
pulling its power. This is USB HID — there is no link to lose.

## The one real limitation

EdgeTX's simple joystick mode (`radio/src/usb_joystick.cpp`) maps:

```
X Y Z Rx Ry Rz S1 S2  ->  channels 1..8   analog
buttons 1..8          ->  channels 9..16  "on if channel > 0"
```

So **channels 9–16 arrive as booleans**. Two consequences:

- **CH9 (speed profile)** is a 3-position switch, so over USB its middle
  position is indistinguishable from its up position.
- **CH13 (encoded live-tune group)** collapses entirely — every non-zero level
  reads as "on".

For stick work none of that matters: roll, velocity, hip height, yaw, jump and
both tune knobs are all on analog axes at full resolution. If you later need
faithful CH9/CH13, set `usbJoystickExtMode: 1` in the model and give them their
own axes.

## Checking the map before you fly

`--monitor` is the sim-side equivalent of the radio's Outputs screen. Move
every control and confirm each lands on the channel `radio/CHANNELS.md` says it
should. It takes a minute and needs no robot, no receiver, and no simulation.
