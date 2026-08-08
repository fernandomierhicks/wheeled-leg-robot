# HipMotors

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

AK45-10 hip motor driver — MIT Cheetah protocol over CAN2 (FlexCAN_T4).

## Wiring

| Signal | Teensy 4.1 pin |
|---|---|
| CAN2 TX | 1 |
| CAN2 RX | 0 |

CAN bus: 1 Mbps. Left motor CAN ID = 11, right = 12 (set in `config.h`).
Both motors share the same bus; the ID in the reply frame identifies which axis.

## API

```cpp
hip_motors_init();         // call once in setup()
hip_motors_enter_mit();    // enable MIT mode on both motors
// control loop:
hip_motors_poll();         // refresh hm_L/hm_R.ok, send this tick's MIT frame
hip_motors_send(pos_L, vel_L, kp_L, kd_L, trq_L,
                pos_R, vel_R, kp_R, kd_R, trq_R);
// shutdown:
hip_motors_exit_mit();
```

Feedback is interrupt-driven (`FlexCAN_T4::onReceive`). `hm_L` / `hm_R` globals hold the latest `{pos_rad, vel_rad_s, torque_nm, last_fb_ms, ok, mit_active}`.

## Parameter limits (hardware-enforced by driver)

Position, Kp and Kd are common to the whole AK family; **speed and torque are
per motor model**, from the table in §5.3 of the AK series driver manual
(v1.0.18 — its changelog notes "Corrected the AK45-10 motor parameters", so
older copies of that table are wrong for this motor).

| Field | Min | Max |
|---|---|---|
| pos | −12.5 rad | +12.5 rad |
| vel | −20 rad/s | +20 rad/s |
| kp | 0 N·m/rad | 500 N·m/rad |
| kd | 0 N·m·s/rad | 5 N·m·s/rad |
| torque | −8 N·m | +8 N·m |

## Gotchas

**Call `enter_mit()` explicitly before the first `hip_motors_send()`**, or the command is silently ignored — each motor's frame is skipped while its `mit_active == false`. Gating is per motor, so a single-leg bench setup still drives the active side.

**There is no periodic MIT re-entry, and there should not be one.** This README used to claim "the AK45-10 exits MIT mode after ~4 s without a re-entry frame", and `poll()` re-sent `enter_mit` every 3 s on that basis. Both are gone as of 2026-08-07. The claim came from a time when `poll()` sent *no* periodic frames at all (`f4e143c`); once the 500 Hz setpoint/ping stream was added (`6f1a515`) it was never re-tested. Measured on the bench with the re-entry disabled — 120 s of continuous loaded sine commanding, zero of 6015 samples below 0.05 N·m, longest command-moved-but-position-frozen span 40 ms (sine turning points + encoder quantisation). MIT survives on the 500 Hz stream alone.

Re-entry was not harmless: `enter_mit` is a mode transition, not an idempotent refresh, and it landed ~100 µs before the next stiff setpoint frame — a periodic disturbance in every state including `RUNNING`, audible as a repetitive hip jerk. If MIT ever does drop out, the right response is the existing feedback watchdog (`hm_*.ok` → `FAULT_HIP_FEEDBACK_LOST` → ESTOP), which surfaces the fault instead of hiding it behind a timer. See the block comment at the top of `hip_motors.cpp`.

**Inter-frame delay** — a 500 µs gap (`CAN_INTER_FRAME_US`) is inserted between every back-to-back pair of CAN TX frames. Removing it causes the second motor to miss the frame intermittently.

**Zeroing persists across power cycles** — `hip_motors_zero()` writes the zero to flash inside the motor. Call it only once at calibration, not in normal startup.

**The reply's third field is torque, not current** — the manual's byte table
labels it "current", but its own reference decoder scales it by the model's
*torque* range and names the result `torque`. It is shaft torque in N·m.
`hm_L/R.torque_nm` and the `hip_l/r_torque_nm` telemetry fields are that value.

**Getting the model constants wrong is silent** — the motor decodes commands
with its own constants and we decode replies with ours, so a mismatch scales
both directions with no error anywhere. Before 2026-07-31 this driver used
V ±65 / T ±18 (plus a bogus separate I ±20 for the reply), which reported hip
torque 2.5× high while delivering only 8/18 of every commanded feedforward
torque. Confirmed against bench log `20260728T053232`: reconstructing the MIT
impedance law from telemetry fit the reported value at 0.400 = 8/20, and
position-derivative versus reported velocity fit 0.308 = 20/65. If you ever
change motor model, re-check this table first.

**Feedback is only sent in response to a command** — `hm_L/R.ok` goes false if no `hip_motors_send()` calls are made (e.g. while MIT mode is still being established).
