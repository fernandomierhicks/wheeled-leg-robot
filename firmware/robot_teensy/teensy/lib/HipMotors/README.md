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
hip_motors_poll();         // refresh hm_L/hm_R.ok, re-enter MIT if due
hip_motors_send(pos_L, vel_L, kp_L, kd_L, trq_L,
                pos_R, vel_R, kp_R, kd_R, trq_R);
// shutdown:
hip_motors_exit_mit();
```

Feedback is interrupt-driven (`FlexCAN_T4::onReceive`). `hm_L` / `hm_R` globals hold the latest `{pos_rad, vel_rad_s, current_A, last_fb_ms, ok, mit_active}`.

## Parameter limits (hardware-enforced by driver)

| Field | Min | Max |
|---|---|---|
| pos | −12.5 rad | +12.5 rad |
| vel | −65 rad/s | +65 rad/s |
| kp | 0 N·m/rad | 500 N·m/rad |
| kd | 0 N·m·s/rad | 5 N·m·s/rad |
| torque | −18 N·m | +18 N·m |

## Gotchas

**MIT mode silently drops out** — the AK45-10 exits MIT mode after ~4 s without a re-entry frame. `hip_motors_poll()` re-sends `enter_mit` every 3 s automatically, but you must also call it explicitly on startup before the first `hip_motors_send()` or the command will be silently ignored (each motor's frame is skipped while its `mit_active == false` — gating is per motor, so a single-leg bench setup still drives the active side).

**Inter-frame delay** — a 500 µs gap (`CAN_INTER_FRAME_US`) is inserted between every back-to-back pair of CAN TX frames. Removing it causes the second motor to miss the frame intermittently.

**Zeroing persists across power cycles** — `hip_motors_zero()` writes the zero to flash inside the motor. Call it only once at calibration, not in normal startup.

**Feedback is only sent in response to a command** — `hm_L/R.ok` goes false if no `hip_motors_send()` calls are made (e.g. while MIT mode is still being established).
