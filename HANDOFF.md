# Calibration discrete-event reporting (Teensy → ESP32 → GUI)

## Context

`firmware/robot_teensy/teensy/lib/Calibration/calibration.cpp` currently reports
hardstop discovery, computed software limits, homing completion, and faults via
raw `Serial.printf()` — visible only on a direct USB serial monitor, and these
raw text bytes are interleaved on the *same* USB port that `g_comm_usb`
(CommLink) uses for framed binary packets to the PC, which can corrupt the
GUI's packet stream.

The comm stack already has a precedent for event-driven (non-50Hz) messages:
`COMM_TYPE_LOG` (0x04), used by `check_imu_state()` in
`firmware/robot_teensy/teensy/src/main.cpp` for IMU state transitions. It's
variable-length, flows Teensy → ESP32 → GUI untouched, and is already displayed
in the Raw Data tab and logged to `logs/teensy.log`.

Goal: surface calibration hardstop/limit discovery in the GUI in two ways —
(1) human-readable log lines (reuse 0x04, replacing the printf calls), and
(2) a new structured event packet (0x05) that a Hip Motors tab widget can
consume directly to display the discovered L/R limits numerically, without
string parsing.

## 1. Make `comm_log` callable from `calibration.cpp`, add `comm_send_calib_event`

No new library — `calibration.cpp` is linked into the same firmware image as
`main.cpp`, so a plain non-static function with a declaration in a shared
header is enough.

- `firmware/robot_teensy/teensy/src/main.cpp:23-36`: drop the `static` from
  `comm_log()` (keep the implementation as-is).
- Add a second function in `main.cpp`, alongside `comm_log()`:
  ```cpp
  void comm_send_calib_event(uint8_t axis, uint8_t event,
                              float pos_rad, float min_rad, float max_rad);
  ```
  Builds a `CalibEventPayload` (see below) and sends it via `g_comm` and,
  if `Serial`, `g_comm_usb` — mirroring `comm_log()`'s send pattern.
- Declare both functions in `firmware/robot_teensy/shared/comm_protocol.h`
  (already reachable from `calibration.cpp` via its existing include chain —
  `hip_motors.h` → ... → `comm_protocol.h`; add `#include "comm_protocol.h"`
  to `calibration.cpp` directly if not already transitively included).

## 2. Protocol additions (`firmware/robot_teensy/shared/comm_protocol.h`)

```c
#define COMM_TYPE_CALIB_EVENT 0x05
#define CALIB_EVENT_PAYLOAD_V1 1

// Calibration event sub-types
#define CALIB_EVENT_START        0x01  // both axes: seek begins
#define CALIB_EVENT_BOTTOM_FOUND 0x02  // axis: bottom hardstop found & zeroed
#define CALIB_EVENT_LIMITS       0x03  // axis: top hardstop found, limits computed
#define CALIB_EVENT_DONE         0x04  // axis: returned home, holding
#define CALIB_EVENT_FAULT        0x05  // axis: hardstop not found within safety bound

typedef struct __attribute__((packed)) {
    uint8_t axis;     // HIP_MOTOR_BOTH/L/R
    uint8_t event;    // CALIB_EVENT_*
    float   pos_rad;  // measured position at the event
    float   min_rad;  // computed lower limit (LIMITS/DONE only, else 0)
    float   max_rad;  // computed upper limit (LIMITS/DONE only, else 0)
} CalibEventPayload;  // 14 bytes
```

Reuses existing `HIP_MOTOR_BOTH/L/R` IDs for the `axis` field.

## 3. `calibration.cpp` changes

- `#include "comm_protocol.h"` (for the `comm_log`/`comm_send_calib_event`
  declarations and the new `CALIB_EVENT_*`/`HIP_MOTOR_*` constants).
- `update_axis()` gains a `uint8_t axis_id` parameter (HIP_MOTOR_L/R), passed
  from the two call sites in `calibration_update()`.
- Replace each `Serial.printf("[Calib] ...")` with a `comm_log(LOG_LEVEL_*, ...)`
  call (same info, reworded without the `[Calib]` prefix since `log_level`
  already conveys severity) **and** a matching `comm_send_calib_event(...)`:
  - `calibration_start()`: `comm_log(INFO, "Calib: starting hardstop search")` +
    `comm_send_calib_event(HIP_MOTOR_BOTH, CALIB_EVENT_START, 0, 0, 0)`.
  - Bottom hardstop found (line ~78): `comm_log(INFO, "Calib %s: bottom hardstop found, zeroed", tag)` +
    `comm_send_calib_event(axis_id, CALIB_EVENT_BOTTOM_FOUND, hm.pos_rad, 0, 0)`.
  - Top hardstop / limits computed (line ~89): `comm_log(INFO, "Calib %s: limits [%.3f, %.3f] rad", tag, lim.min_rad, lim.max_rad)` +
    `comm_send_calib_event(axis_id, CALIB_EVENT_LIMITS, range, lim.min_rad, lim.max_rad)`.
  - Homing done (line ~106): `comm_log(INFO, "Calib %s: done, holding @ %.3f rad", tag, ax.ramp_target)` +
    `comm_send_calib_event(axis_id, CALIB_EVENT_DONE, ax.ramp_target, lim.min_rad, lim.max_rad)`.
  - Fault (line ~70): `comm_log(ERROR, "Calib %s: FAULT - hardstop not found within %.1f rad", tag, CALIB_SAFETY_BOUND_RAD)` +
    `comm_send_calib_event(axis_id, CALIB_EVENT_FAULT, ax.ramp_target, 0, 0)`.

## 4. GUI decoder (`software/gui/flash_monitor.py`)

- `_TYPE_NAMES`: add `0x05: "CALIB"`.
- In `PacketDecoder._parse()`, add a branch:
  ```python
  elif ptype == 0x05 and length >= 14:
      axis, event, pos, mn, mx = _struct.unpack_from("<BBfff", payload)
      info.update({
          "calib_axis": axis, "calib_event": event,
          "calib_pos_rad": pos, "calib_min_rad": mn, "calib_max_rad": mx,
      })
  ```
  (Mirrors the existing telemetry `_struct.unpack_from` pattern at
  `flash_monitor.py:237-260`.) This flows through `TelemetryBus` automatically.

## 5. Hip Motors tab widget (`software/gui/hip_motors.py`)

- `_MotorPanel.__init__`: add a "Limits" readout row via the existing
  `_readout(lay, "Limits", <color>)` helper (used for Position/Command/Current
  at lines 204-206), storing the label as `self._lbl_limits`.
- `HipMotorsTab._on_packet` (or a new handler subscribed alongside it): on
  `ptype == 0x05` with `calib_event in (CALIB_EVENT_LIMITS, CALIB_EVENT_DONE)`,
  route to `self._panel_L` or `self._panel_R` based on `calib_axis`
  (1=L, 2=R, matching `_HIP_MOTOR_L/R` already used for `_panel_L`/`_panel_R`
  construction at lines 493-494), and update `_lbl_limits` text to
  `"[min_rad, max_rad] rad"`.
- On `CALIB_EVENT_START`, clear both panels' `_lbl_limits` back to a
  placeholder (e.g. "—") so stale values from a previous run aren't shown.

## Verification

1. Build the Teensy firmware (`pio run` in `firmware/robot_teensy/teensy`) —
   confirms `comm_protocol.h` changes and the `calibration.cpp` ↔ `main.cpp`
   linkage compile.
2. Build the ESP32 firmware (`pio run` in `firmware/robot_teensy/esp32`) —
   confirm it still compiles (no changes expected there; `on_teensy_packet`
   already forwards unknown/new packet types transparently to USB/TCP).
3. Flash both, run calibration from the GUI's "Calibrate" button, and confirm:
   - Raw Data tab shows the new `CALIB`-prefixed log lines and `0x05 CALIB`
     packets.
   - Hip Motors tab's L and R panels populate a "Limits" readout with the
     discovered `[min_rad, max_rad]` once each axis finishes seeking.
   - `logs/teensy.log` contains the new log messages.
