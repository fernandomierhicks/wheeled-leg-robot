# Wheel Motor Flatsat Smoke Test

## Goals
1. Confirm motors handle real wheel mass (inertia) without issues
2. Verify brake resistor / regen path absorbs deceleration energy correctly

Direction and encoder feedback already confirmed working. No firmware changes needed.

## How It Works

`CMD_ID_WHEEL` with `WHEEL_SUB_SET_MODE` and `WHEEL_SUB_SEND` is already fully implemented, gated to `STATE_MANUAL`. Any incoming packet resets the 500 ms GUI watchdog, so the robot stays in MANUAL as long as the GUI is active.

## Pre-conditions

- Wheels physically mounted on motor shafts
- ODrive brake resistor wired and `config.brake_resistance` configured
- Both ODrive axes calibrated (motor + encoder)
- CAN bus connected Teensy ↔ ODrive
- Robot lying flat (flatsat), wheels in air

## Test Steps

1. Power on — confirm telemetry `wm_L.ok = true`, `wm_R.ok = true`
2. `CMD_ID_SET_MODE → STATE_MANUAL` — robot enters MANUAL (cyan LED breathe)
3. `CMD_ID_WHEEL / WHEEL_SUB_SET_MODE → VELOCITY` — wheels enter ODrive VELOCITY mode holding zero
4. `WHEEL_SUB_SEND (L=1.0, R=1.0 turns/s)` — both wheels spin; current should be near zero at steady state (no load)
5. Ramp up: `WHEEL_SUB_SEND (L=2.0, R=2.0)` — check for rattles, confirm no ODrive faults
6. **Regen test:** from 2 turns/s send `WHEEL_SUB_SEND (L=0, R=0)` — hard decel; watch `wm_l_vbus` / `wm_r_vbus` spike; brake resistor gets warm
7. Repeat step 6 a few times to confirm consistent regen
8. `WHEEL_SUB_SET_MODE → IDLE` — wheels coast to stop

**Abort anytime:** Stop GUI packets — robot auto-exits MANUAL after 500 ms, wheels go IDLE.
Or send `CMD_ID_SET_MODE → STATE_STANDBY` explicitly.

## What to Watch in Telemetry

| Signal | Expected |
|--------|----------|
| `wm_l_vel_turns_s`, `wm_r_vel_turns_s` | Tracks setpoint (closed-loop VELOCITY) |
| `wm_l_vbus`, `wm_r_vbus` | Spikes upward during decel — regen active |
| `wm_l_error`, `wm_r_error` | Stay 0 throughout |
| `wm_mode` | 1 = VELOCITY, 0 = IDLE |
| Brake resistor (physical) | Warm after 3+ regen cycles |

## Pass Criteria

- [ ] Wheels track velocity setpoint in telemetry
- [ ] `wm_l_vbus` / `wm_r_vbus` spike noticeably on step-down from 2.0 to 0 turns/s
- [ ] Brake resistor warm after repeated regen cycles
- [ ] No ODrive faults throughout (`wm_l_error`, `wm_r_error` = 0)
- [ ] `WHEEL_SUB_SET_MODE → IDLE` cleanly stops wheels
