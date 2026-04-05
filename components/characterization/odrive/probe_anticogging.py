"""
probe_anticogging.py — Step-by-step anticogging diagnostic for ODrive 0.5.6
Prints full state at each stage so we can see exactly where it fails.
"""
import sys
import time
import odrive
from odrive.enums import (
    AXIS_STATE_IDLE,
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CONTROL_MODE_VELOCITY_CONTROL,
    CONTROL_MODE_POSITION_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

AXIS_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 0

AXIS_ERRORS = {
    0x00000001: "INVALID_STATE", 0x00000040: "MOTOR_FAILED",
    0x00000100: "ENCODER_FAILED", 0x00000200: "CONTROLLER_FAILED",
    0x00020000: "OVER_TEMP",     0x00040000: "UNKNOWN_POSITION",
}
MOTOR_ERRORS = {
    0x00000008: "DRV_FAULT",         0x00000020: "MODULATION_MAGNITUDE",
    0x00000100: "CURRENT_LIMIT_VIOLATION", 0x00010000: "BRAKE_RESISTOR_DISARMED",
    0x01000000: "UNKNOWN_VBUS_VOLTAGE",    0x04000000: "UNKNOWN_GAINS",
    0x08000000: "CONTROLLER_INITIALIZING",
}
ENCODER_ERRORS = {
    0x00000040: "ABS_SPI_TIMEOUT", 0x00000080: "ABS_SPI_COM_FAIL",
    0x00000100: "ABS_SPI_NOT_READY",
}
CONTROLLER_ERRORS = {
    0x00000001: "OVERSPEED", 0x00000020: "INVALID_ESTIMATE",
    0x00000080: "SPINOUT_DETECTED",
}

def decode(val, table):
    if val == 0: return "OK (0)"
    names = [n for b, n in table.items() if val & b]
    extra = val & ~sum(b for b in table if val & b)
    if extra: names.append(f"UNKNOWN(0x{extra:08X})")
    return ", ".join(names) if names else f"0x{val:08X}"

def dump(ax, label=""):
    if label: print(f"\n=== {label} ===")
    print(f"  state           : {ax.current_state}")
    print(f"  axis error      : {decode(ax.error, AXIS_ERRORS)}")
    print(f"  motor error     : {decode(ax.motor.error, MOTOR_ERRORS)}")
    print(f"  encoder error   : {decode(ax.encoder.error, ENCODER_ERRORS)}")
    print(f"  controller error: {decode(ax.controller.error, CONTROLLER_ERRORS)}")
    print(f"  motor.is_calibrated : {ax.motor.is_calibrated}")
    print(f"  encoder.is_ready    : {ax.encoder.is_ready}")
    print(f"  control_mode        : {ax.controller.config.control_mode}")
    print(f"  input_mode          : {ax.controller.config.input_mode}")
    print(f"  vel_limit           : {ax.controller.config.vel_limit}")
    print(f"  vel_gain            : {ax.controller.config.vel_gain}")
    print(f"  vel_integrator_gain : {ax.controller.config.vel_integrator_gain}")
    print(f"  pos_gain            : {ax.controller.config.pos_gain}")
    acog = ax.controller.config.anticogging
    print(f"  acog.anticogging_enabled : {acog.anticogging_enabled}")
    print(f"  acog.pre_calibrated      : {acog.pre_calibrated}")
    print(f"  acog.calib_pos_threshold : {acog.calib_pos_threshold}")
    print(f"  acog.calib_vel_threshold : {acog.calib_vel_threshold}")
    print(f"  acog.cogging_ratio       : {acog.cogging_ratio}")
    print(f"  acog.index               : {acog.index}")

def wait_idle(ax, timeout=30):
    t0 = time.monotonic()
    while ax.current_state != AXIS_STATE_IDLE:
        if time.monotonic() - t0 > timeout:
            print("  TIMEOUT waiting for IDLE")
            return False
        time.sleep(0.2)
    return True

# ── Connect ───────────────────────────────────────────────────────────────────
print("Connecting...")
odrv = odrive.find_any()
ax   = getattr(odrv, f"axis{AXIS_NUM}")
print(f"Connected  serial={hex(odrv.serial_number)}  fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
print(f"Vbus={odrv.vbus_voltage:.2f} V  brake_resistor={odrv.config.enable_brake_resistor}")

dump(ax, "INITIAL STATE")

# ── Clear errors ──────────────────────────────────────────────────────────────
print("\n--- Clearing errors ---")
odrv.clear_errors()
dump(ax, "AFTER CLEAR_ERRORS")

# ── Calibrate if needed ───────────────────────────────────────────────────────
if not ax.motor.is_calibrated or not ax.encoder.is_ready:
    print("\n--- Running FULL_CALIBRATION_SEQUENCE (motor will beep) ---")
    ax.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
    wait_idle(ax, timeout=60)
    dump(ax, "AFTER CALIBRATION")
    if ax.error or ax.motor.error or ax.encoder.error:
        print("ABORT: calibration failed")
        sys.exit(1)
else:
    print("\nMotor + encoder already calibrated — skipping calibration")

# ── Try VELOCITY control closed loop first (sanity check) ────────────────────
print("\n--- STEP 1: Enter VELOCITY closed loop at 0 vel ---")
ax.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
ax.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
ax.controller.input_vel           = 0.0
ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.5)
dump(ax, "VEL CLOSED LOOP @ 0")

if ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
    print("ABORT: could not enter closed loop in velocity mode")
    sys.exit(1)

print("\n--- STEP 2: Command 1 t/s for 2 s ---")
ax.controller.input_vel = 1.0
t0 = time.monotonic()
while time.monotonic() - t0 < 2.0:
    vel = ax.encoder.vel_estimate
    iq  = ax.motor.current_control.Iq_measured
    print(f"  vel={vel:+.3f} t/s  Iq={iq:.3f} A", end="\r")
    time.sleep(0.05)
print()
ax.controller.input_vel = 0.0
ax.requested_state = AXIS_STATE_IDLE
time.sleep(0.3)
dump(ax, "AFTER VEL TEST")

if ax.error or ax.motor.error or ax.encoder.error:
    print("ABORT: errors after velocity test")
    sys.exit(1)

# ── Try POSITION control closed loop ─────────────────────────────────────────
print("\n--- STEP 3: Enter POSITION closed loop ---")
odrv.clear_errors()
ax.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
ax.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.5)
dump(ax, "POS CLOSED LOOP")

if ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
    print("ABORT: could not enter closed loop in position mode")
    sys.exit(1)

# ── Start anticogging calibration ─────────────────────────────────────────────
print("\n--- STEP 4: Call start_anticogging_calibration() ---")
ax.controller.start_anticogging_calibration()
print("  Called. Monitoring for 10 s...")

t0 = time.monotonic()
prev_idx = -1
while time.monotonic() - t0 < 10.0:
    acog = ax.controller.config.anticogging
    idx  = acog.index
    vel  = ax.encoder.vel_estimate
    pos  = ax.encoder.pos_estimate
    state = ax.current_state
    ax_err = ax.error
    mot_err = ax.motor.error
    enc_err = ax.encoder.error
    if idx != prev_idx:
        print(f"  t={time.monotonic()-t0:.1f}s  idx={idx}  vel={vel:+.3f}  pos={pos:.3f}  state={state}  ax_err={ax_err}  mot_err={mot_err}  enc_err={enc_err}")
        prev_idx = idx
    else:
        print(f"  t={time.monotonic()-t0:.1f}s  idx={idx} (no change)  vel={vel:+.3f}  state={state}  ax_err={ax_err}", end="\r")
    if ax_err or mot_err or enc_err:
        print()
        dump(ax, "ERROR DURING ANTICOGGING")
        break
    time.sleep(0.2)

print()
dump(ax, "FINAL STATE")

ax.requested_state = AXIS_STATE_IDLE
