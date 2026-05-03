# AK45-10 CAN Timing Stress Test

Hardware: Arduino UNO R4 WiFi — CANTX D4, CANRX D5, 1 Mbps

## Purpose

Find the reliable CAN command rate ceiling for two AK45-10 motors in MIT mode,
and confirm 500 Hz dual-motor control is feasible for the main balance controller.

## Key findings

| Setting | Value | Notes |
|---|---|---|
| Max reliable rate (both motors, per tick) | **1500 Hz** | 98% tick delivery |
| Inter-frame delay minimum (reliable) | **150 µs** | 100 µs dropped M2 frames |
| CAN burst cost per tick (both motors) | **~195 µs** | M1 + 150 µs gap + M2 |
| 500 Hz feasibility | **Yes, comfortably** | ~10% of tick budget used |

## Test configurations tested

### Config 1 — Alternating 500 Hz (HelloWorld baseline)
```
TARGET_HZ=500, BOTH_MOTORS_PER_TICK=false, INTER_FRAME_DELAY_US=1000
```
Each motor commanded at 250 Hz. Send ~17 µs/tick, period avg ~2020 µs.

### Config 2 — Both motors 1000 Hz
```
TARGET_HZ=1000, BOTH_MOTORS_PER_TICK=true, INTER_FRAME_DELAY_US=200
```
`send[246/247/250 us]  period[1000/1017/13958 us]  ticks=983`
Both motors working. Spike to 13958 µs from REENTER_MIT blocking delay(5).

### Config 3 — Both motors 2000 Hz (FAILED)
```
TARGET_HZ=2000, BOTH_MOTORS_PER_TICK=true, INTER_FRAME_DELAY_US=100, REENTER_MIT_MS=0
```
M2 stopped responding. 100 µs inter-frame is below the reliable threshold at 1 Mbps.

### Config 4 — Both motors 1500 Hz (current, working)
```
TARGET_HZ=1500, BOTH_MOTORS_PER_TICK=true, INTER_FRAME_DELAY_US=150, REENTER_MIT_MS=1000
```
`send[194/195/198 us]  rx[3/4/27 us]  period[669/680/13636 us]  ticks=1470`
Both motors commanding and replying correctly. 98% of target ticks delivered.

## Implications for balance controller (500 Hz)

At 500 Hz with `BOTH_MOTORS_PER_TICK=true`, each tick has a 2000 µs budget.
The dual-motor CAN burst costs ~195 µs — leaving ~1800 µs for IMU, LQR, odometry.
**500 Hz is not the bottleneck.**

## Known issue: REENTER_MIT blocking spike

`REENTER_DELAY_MS=5` causes a 5 ms blocking stall every `REENTER_MIT_MS`.
Visible as period max ~13–14 ms. In the real controller, either:
- Disable re-entry once motors are confirmed to stay in MIT mode reliably, or
- Make re-entry non-blocking (send the frame, skip the delay).

## Bug fixed during testing

Two consecutive `enter_mit` calls with no inter-frame gap dropped M2's frame —
M2 never re-entered MIT mode after the first REENTER interval and held position at 0.
Fix: add `delayMicroseconds(INTER_FRAME_DELAY_US)` between the two `enter_mit` calls
in the periodic re-enter block.
