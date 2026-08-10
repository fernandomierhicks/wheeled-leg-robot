# Morning hardware handoff

The offline framework and firmware builds are ready. No robot command, flash,
motor motion, or live parameter write was performed overnight.

Work through the tests one at a time. Do not start a dynamic test until its
setup, physical restraints, ESTOP access, expected range, and abort condition
have been reviewed together.

## First test: T0.1 mass inventory

This is the safest starting point and needs no powered robot.

Purpose: replace the source-based `PlantParams.body_mass_kg=1.769488` prior and
the 520 g removable-wheel prior with as-built scale measurements. Both remain
explicitly provisional.

Equipment: a scale with at least 1 g resolution (or record its resolution if
coarser), a tray/tare container if loose parts cannot sit securely, and a way to
record each item.

Procedure:

1. Power the robot off and disconnect the battery.
2. Weigh the complete assembled robot with battery and all wiring in the exact
   intended driving configuration. Record the value and scale resolution.
3. If possible, separately weigh left wheel assembly, right wheel assembly,
   battery, body/electronics box, each femur, each tibia/dogleg, and each
   coupler. Do not disassemble anything merely to obtain a component mass.
4. Repeat the complete-robot measurement three times after lifting and
   replacing it on the scale.
5. Record the three totals and any component values in a message or CSV. Also
   note anything intentionally absent (covers, fasteners, payload, etc.).

Acceptance: the three complete totals should span no more than twice the scale
resolution. If they do, reposition the robot and repeat before changing the
twin.

After T0.1, the next hands-on test is T0.2 CG measurement. It should be planned
from the measured mass and actual available scales rather than improvised now.

## Build and offline status

- Protocol generation/check: pass.
- Generated twin params check: pass.
- Twin + SIL tests: 26 pass.
- Teensy 4.1 release build: pass.
- ESP32 release build: pass.
- GUI WLOG decoder and metrics on twin output: pass.
- Hardware flash/run: deliberately not done.

## Plant-ID firmware hook for later tests

The non-persistent params are `plant_id_en`, `plant_id_amp`, `plant_id_f0`,
`plant_id_f1`, and `plant_id_dur`. The chirp is symmetric wheel torque, enters
after the normal control law/barrier, and remains behind the normal torque
clamps and wheel governors. It auto-disarms at duration and whenever a
controlled state is exited. Do not arm it until a specific restrained test and
amplitude have been agreed.
