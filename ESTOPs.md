# ESTOP Recovery Design

## Current Behavior

Only one exit path exists regardless of fault:
`ESTOP → (GUI sends STATE_STARTUP) → STARTUP → STANDBY`

This forces a full re-init (IMU + hip motor timeout check) even for trivial faults and gives
the operator no guidance on what action is actually required to recover safely.

---

## Fault Classification

| Code | Name | Recovery Tier | Rationale |
|------|------|--------------|-----------|
| 0x06 | FAULT_HUMAN_ESTOP | **Soft clear** | User-triggered, nothing actually broken |
| 0x09 | FAULT_WHEEL_RUNAWAY | **Soft clear** | Transient runaway; robot likely still upright |
| 0x08 | FAULT_PITCH_WATCHDOG | **Reposition** | Robot fell; must be stood up before reset |
| 0x05 | FAULT_CALIBRATION_TIMEOUT | **Reposition + recalibrate** | Hardstop not found; re-run calib after repositioning |
| 0x07 | FAULT_PARAM_OUT_OF_BOUNDS | **GUI fix** | Bad param value; fix in GUI before reset |
| 0x04 | FAULT_HIP_LARGE_POS_CMD | **GUI fix** | Commanded jump too large; check param |
| 0x03 | FAULT_HIP_FEEDBACK_LOST | **Reboot** | CAN dropout; power-cycle motors and reboot |
| 0x02 | FAULT_HIP_INIT_TIMEOUT | **Reboot** | Motors never responded at boot |
| 0x01 | FAULT_IMU_ERROR | **Reboot** | IMU hardware/wiring failure |

---

## Proposed Recovery Tiers

### Tier 1 — Soft Clear (radio TX or GUI)
**Faults:** `FAULT_HUMAN_ESTOP`, `FAULT_WHEEL_RUNAWAY`

- Skip STARTUP re-init; go directly ESTOP → STANDBY.
- **Radio path:** CH10 arm-switch rising edge, while in ESTOP with a Tier 1 fault, calls
  `stateMachine_request_soft_clear()` → goes to STANDBY directly.
- **GUI path:** Green "Clear ESTOP" button, no confirmation dialog needed.

### Tier 2 — Reposition Required (GUI only, with acknowledgement)
**Faults:** `FAULT_PITCH_WATCHDOG`

- GUI shows: *"Robot fell. Stand it upright, then click Confirm."*
- After user clicks Confirm: GUI sends reset → goes through STARTUP (re-checks IMU + hip).
- Radio TX **cannot** clear this tier.

### Tier 3 — Reposition + Recalibrate (GUI only)
**Faults:** `FAULT_CALIBRATION_TIMEOUT`

- GUI shows: *"Calibration failed. Reposition robot and restart calibration."*
- After GUI reset → STARTUP → STANDBY; user must re-trigger calibration via CH5 or GUI.

### Tier 4 — GUI Fix Required (GUI only)
**Faults:** `FAULT_PARAM_OUT_OF_BOUNDS`, `FAULT_HIP_LARGE_POS_CMD`

- GUI shows which parameter caused the fault (fault_code + param_id in telemetry).
- "Reset" button only enabled after the offending param has been changed.
- Goes through STARTUP on reset.

### Tier 5 — Reboot Required (display only, no reset path)
**Faults:** `FAULT_IMU_ERROR`, `FAULT_HIP_INIT_TIMEOUT`, `FAULT_HIP_FEEDBACK_LOST`

- GUI shows: *"Hardware fault — power cycle robot and reboot."*
- GUI "Reset" button is **disabled entirely**. Only physical reboot resolves.
- Existing reboot command (CMD) can still be sent from GUI for soft reboot.

---

## Implementation Plan

### Step 1 — Firmware: fault severity helper (`comm_protocol.h`)
Add `fault_severity_t` enum `{SOFT, REPOSITION, GUI_FIX, REBOOT}` and an inline
`fault_severity(fault_code_t)` function. Single source of truth, visible to firmware and GUI.

### Step 2 — Firmware: `stateMachine_request_soft_clear()` (`state_machine.cpp/.h`)
New request: if fault severity == SOFT, transition ESTOP → STANDBY directly, skipping STARTUP.
Otherwise log a warning and do nothing.

### Step 3 — Firmware: radio soft-clear path (`main.cpp`, `radio_update()`)
In the CH10 rising-edge handler: if in ESTOP AND fault is SOFT, call
`stateMachine_request_soft_clear()` instead of `stateMachine_request_running()`.

### Step 4 — Firmware: verify fault_code in telemetry
Confirm `fault_code` is included in every status packet so GUI always knows the active fault.

### Step 5 — GUI: tiered recovery panel (`robot_visualizer_tab.py` or `main.py`)
Replace the single "Reset" button with a dynamic panel driven by `fault_code` from telemetry:

| Tier | UI |
|------|----|
| SOFT | Green "Clear ESTOP" button (no dialog) |
| REPOSITION | Orange "Reset" + instruction label + "Robot is upright" checkbox |
| GUI_FIX | Yellow warning + param name; Reset greyed until param is changed |
| REBOOT | Red "Reboot required" label + "Send Reboot" button; Reset disabled |

---

## Files to Modify

| File | Change |
|------|--------|
| `firmware/robot_teensy/shared/comm_protocol.h` | Add `fault_severity_t` + `fault_severity()` |
| `firmware/robot_teensy/teensy/src/state_machine.h` | Declare `stateMachine_request_soft_clear()` |
| `firmware/robot_teensy/teensy/src/state_machine.cpp` | Implement soft-clear transition |
| `firmware/robot_teensy/teensy/src/main.cpp` | Radio CH10 soft-clear branch |
| `software/gui/robot_visualizer_tab.py` | Tiered recovery UI panel |

---

## Verification

1. Flash firmware, open GUI.
2. Trigger `FAULT_HUMAN_ESTOP` via GUI → verify CH10 radio flip clears to STANDBY without re-init.
3. Trigger `FAULT_PITCH_WATCHDOG` (tilt robot past 50°) → verify radio CH10 does NOT clear;
   GUI shows reposition dialog; after confirm, goes through STARTUP back to STANDBY.
4. Trigger `FAULT_IMU_ERROR` (disconnect IMU) → verify GUI shows "Reboot required", Reset disabled.
5. Send out-of-range param → verify GUI shows which param failed and blocks Reset until fixed.
