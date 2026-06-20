# Wheel Motors GUI Tab — Implementation Handoff

## Context
`WheelMotorsTab` in `software/gui/main.py` is currently a placeholder stub. The ODrive wheel motor CAN driver (`firmware/robot_teensy/teensy/lib/WheelMotors/`) is fully implemented but: (1) `wheel_motors_init()` is never called in `setup()`, (2) no per-wheel telemetry fields exist in `TelemetryPayload`, and (3) no `CMD_ID_WHEEL` command exists. This plan wires up the full stack: firmware init → telemetry → GUI command/display.

---

## Data Flow

```
GUI (wheel_motors.py)
  → send_wheel_*() in comm_commands.py
    → CommLink serial frame → ESP32 → Teensy on_command()
      → CMD_ID_WHEEL handler → wheel_motors_set_mode() / wheel_motors_send()
        → CAN3 → ODrive (node 0 = L, node 1 = R)
          → encoder feedback → wm_L / wm_R structs
            → send_telemetry() → TelemetryBus → WheelMotorsTab._on_packet()
```

---

## Step 1 — `firmware/robot_teensy/shared/comm_protocol.h`

1. Add `#define CMD_ID_WHEEL  0x07` alongside `CMD_ID_HIP`.
2. Add wheel sub-command constants:
   ```c
   #define WHEEL_SUB_SET_MODE     0x01  // payload: uint8_t mode (WheelMode)
   #define WHEEL_SUB_SEND         0x02  // payload: float L, float R
   #define WHEEL_SUB_CLEAR_ERRORS 0x03  // no payload
   ```
3. Add `#define TELEM_PAYLOAD_V3  3`.
4. Append 11 new fields to `TelemetryPayload` (new size: **118 bytes**, old was 83):
   ```c
   float    wm_l_vel_turns_s;   // left wheel velocity  [turns/s]
   float    wm_r_vel_turns_s;   // right wheel velocity [turns/s]
   float    wm_l_pos_turns;     // left wheel position  [turns]
   float    wm_r_pos_turns;     // right wheel position [turns]
   float    wm_l_vbus;          // left ODrive bus voltage  [V]
   float    wm_r_vbus;          // right ODrive bus voltage [V]
   uint32_t wm_l_error;         // left ODrive Axis_Error word
   uint32_t wm_r_error;         // right ODrive Axis_Error word
   uint8_t  wm_l_state;         // left ODrive Axis_State  (1=IDLE, 8=CLOSED_LOOP)
   uint8_t  wm_r_state;         // right ODrive Axis_State
   uint8_t  wm_mode;            // current WheelMode (0=IDLE,1=VEL,2=POS,3=TRQ)
   ```

---

## Step 2 — `firmware/robot_teensy/teensy/src/main.cpp`

1. Add `#include "wheel_motors.h"` at the top.
2. In `setup()`, after `hip_motors_init()`:
   ```cpp
   wheel_motors_init();
   comm_log(LOG_LEVEL_INFO, "Wheel CAN init OK");
   ```
3. In `read_sensors()`, after hip poll:
   ```cpp
   wheel_motors_poll();
   wheel_motors_pet_watchdog();
   ```
4. In `send_telemetry()`, fill the new fields and update version to `TELEM_PAYLOAD_V3`:
   ```cpp
   telem.wm_l_vel_turns_s = wm_L.vel_turns_s;
   telem.wm_r_vel_turns_s = wm_R.vel_turns_s;
   telem.wm_l_pos_turns   = wm_L.pos_turns;
   telem.wm_r_pos_turns   = wm_R.pos_turns;
   telem.wm_l_vbus        = wm_L.vbus;
   telem.wm_r_vbus        = wm_R.vbus;
   telem.wm_l_error       = wm_L.error;
   telem.wm_r_error       = wm_R.error;
   telem.wm_l_state       = wm_L.axis_state;
   telem.wm_r_state       = wm_R.axis_state;
   telem.wm_mode          = (uint8_t)wm_mode;
   ```
5. In `on_command()`, add wheel handler after the hip block:
   ```cpp
   if (cmd_id == CMD_ID_WHEEL && len >= 2) {
       uint8_t sub = payload[1];
       if (sub == WHEEL_SUB_SET_MODE && len >= 3) {
           wheel_motors_set_mode((WheelMode)payload[2]);
       } else if (sub == WHEEL_SUB_SEND && len >= 10) {
           float L, R;
           memcpy(&L, payload + 2, 4);
           memcpy(&R, payload + 6, 4);
           wheel_motors_send(L, R);
       } else if (sub == WHEEL_SUB_CLEAR_ERRORS) {
           wheel_motors_clear_errors();
       }
       return;
   }
   ```

---

## Step 3 — `software/gui/comm_commands.py`

Add constants and three send helpers:
```python
CMD_ID_WHEEL           = 0x07
WHEEL_SUB_SET_MODE     = 0x01
WHEEL_SUB_SEND         = 0x02
WHEEL_SUB_CLEAR_ERRORS = 0x03

WHEEL_MODE_IDLE     = 0
WHEEL_MODE_VELOCITY = 1
WHEEL_MODE_POSITION = 2
WHEEL_MODE_TORQUE   = 3

def send_wheel_set_mode(mode: int):
    send_frame(build_frame(struct.pack("<BBB", CMD_ID_WHEEL, WHEEL_SUB_SET_MODE, mode)))

def send_wheel_setpoint(L: float, R: float):
    send_frame(build_frame(struct.pack("<BBff", CMD_ID_WHEEL, WHEEL_SUB_SEND, L, R)))

def send_wheel_clear_errors():
    send_frame(build_frame(struct.pack("<BB", CMD_ID_WHEEL, WHEEL_SUB_CLEAR_ERRORS)))
```

---

## Step 4 — `software/gui/flash_monitor.py` telemetry parser

After the existing v2 unpack block (around line 261), extend the dict when `length >= 118`:
```python
if length >= 118:
    wm_l_vel, wm_r_vel, wm_l_pos, wm_r_pos, wm_l_vbus, wm_r_vbus, \
    wm_l_err, wm_r_err, wm_l_st, wm_r_st, wm_mode_val = \
        _struct.unpack_from("<ffffffIIBBB", payload, 83)
    info.update({
        "wm_l_vel_turns_s": wm_l_vel,
        "wm_r_vel_turns_s": wm_r_vel,
        "wm_l_pos_turns":   wm_l_pos,
        "wm_r_pos_turns":   wm_r_pos,
        "wm_l_vbus":        wm_l_vbus,
        "wm_r_vbus":        wm_r_vbus,
        "wm_l_error":       wm_l_err,
        "wm_r_error":       wm_r_err,
        "wm_l_state":       wm_l_st,
        "wm_r_state":       wm_r_st,
        "wm_mode":          wm_mode_val,
    })
```

---

## Step 5 — New file `software/gui/wheel_motors.py`

Model closely on `hip_motors.py`. Key differences:

### `_WheelPanel` (per-motor widget)

**Readouts:**
- Velocity shown as both turns/s and RPM (`vel * 60`)
- Position in turns
- Vbus in V — green if > 20 V, orange otherwise
- ODrive state as string (1 → "IDLE", 8 → "CLOSED_LOOP", else numeric)
- Error word decoded to flag names (bitmask); red label if non-zero
- Freshness indicator dot derived from `wm_l_state != 0` or stale check

**Chart (pyqtgraph rolling, 750 samples ~15 s at 50 Hz):**
- Blue curve: velocity (turns/s)
- Orange dashed: commanded setpoint

**Control (enabled only in STATE_MANUAL):**
- Setpoint spinbox — range/units adapt to mode:
  - VELOCITY: ±20 turns/s
  - POSITION: ±100 turns
  - TORQUE:   ±5 N·m
- "Send" button → `send_wheel_setpoint(val, val)`
- Waveform: Amplitude + Frequency spinboxes, [Sine] [Square] checkable buttons (50 ms tick timer, same pattern as hip wave)
  - For VEL/TRQ, waveform center = 0; for POS, center = current position at start

**Buttons:**
- "Clear Errors" (always enabled) → `send_wheel_clear_errors()`

### `WheelMotorsTab` (main tab)

**Mode bar (same style as `HipMotorsTab`):**
- State label (mirrors robot state: STANDBY / MANUAL / ESTOP / ...)
- ODrive mode label (IDLE / VELOCITY / POSITION / TORQUE from `wm_mode`)
- Buttons: `Enter Manual` → `send_set_mode(STATE_MANUAL)`, `Exit Manual` → `send_set_mode(STATE_STANDBY)`

**Center column (between the two motor panels):**
- 4 mode buttons: IDLE / VEL / POS / TRQ → `send_wheel_set_mode(mode)` (enabled in MANUAL only)
- "Clear Both" button

**Differential drive helper (bonus):**
- A single velocity slider ± 5 turns/s that sends +val to R and −val to L (spin-in-place test)

**`_on_packet` handler:**
- Gate controls on `robot_state == STATE_MANUAL` (same as hip motors, reuse `STATE_MANUAL = 5`)
- Feed left/right panels with their respective telemetry fields
- Update mode bar labels

---

## Step 6 — `software/gui/main.py`

Replace:
```python
class WheelMotorsTab(_PlaceholderTab):
    def __init__(self): super().__init__("Wheel Motors")
```
With:
```python
from wheel_motors import WheelMotorsTab
```

---

## Verification Checklist

1. Flash firmware; check serial log for `"Wheel CAN init OK"`.
2. Power on ODrive; confirm heartbeat arrives (green freshness dots in GUI).
3. Enter MANUAL mode; select VELOCITY; set 0.5 turns/s → both wheels spin.
4. Enable Sine waveform at 0.2 Hz, amplitude 1.0 turns/s → smooth chart oscillation.
5. Trigger ODrive fault; confirm red error label with decoded name.
6. Press Clear Errors; label goes green.
7. Verify hip motors tab still functions (v2 parser path untouched).
