"""
vel_test.py — ODrive 0.5.6 velocity mode test + error diagnostics
Usage: python vel_test.py [axis]   (axis = 0 or 1, default 0)
"""
import sys
import time
import odrive
from odrive.enums import (
    AXIS_STATE_IDLE,
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

AXIS_NUM   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
VEL_LIMIT  = 10.0   # turns/s — raise from default 2.0
TEST_VEL   = 5.0    # turns/s command during test
RUN_SEC    = 5.0    # how long to spin

# --- Error decode tables (ODrive 0.5.6) ---
AXIS_ERRORS = {
    0x00000001: "INVALID_STATE",
    0x00000040: "MOTOR_FAILED",
    0x00000080: "SENSORLESS_ESTIMATOR_FAILED",
    0x00000100: "ENCODER_FAILED",
    0x00000200: "CONTROLLER_FAILED",
    0x00000800: "WATCHDOG_TIMER_EXPIRED",
    0x00001000: "MIN_ENDSTOP_PRESSED",
    0x00002000: "MAX_ENDSTOP_PRESSED",
    0x00004000: "ESTOP_REQUESTED",
    0x00010000: "HOMING_WITHOUT_ENDSTOP",
    0x00020000: "OVER_TEMP",
    0x00040000: "UNKNOWN_POSITION",
}
MOTOR_ERRORS = {
    0x00000001: "PHASE_RESISTANCE_OUT_OF_RANGE",
    0x00000002: "PHASE_INDUCTANCE_OUT_OF_RANGE",
    0x00000008: "DRV_FAULT",
    0x00000010: "CONTROL_DEADLINE_MISSED",
    0x00000020: "MODULATION_MAGNITUDE",
    0x00000040: "CURRENT_SENSE_SATURATION",
    0x00000100: "CURRENT_LIMIT_VIOLATION",
    0x00000200: "MODULATION_IS_NAN",
    0x00000400: "MOTOR_THERMISTOR_OVER_TEMP",
    0x00000800: "FET_THERMISTOR_OVER_TEMP",
    0x00001000: "TIMER_UPDATE_MISSED",
    0x00002000: "CURRENT_MEASUREMENT_UNAVAILABLE",
    0x00004000: "CONTROLLER_FAILED",
    0x00008000: "I_BUS_OUT_OF_RANGE",
    0x00010000: "BRAKE_RESISTOR_DISARMED",
    0x00020000: "SYSTEM_LEVEL",
    0x00040000: "BAD_TIMING",
    0x00080000: "UNKNOWN_PHASE_ESTIMATE",
    0x00100000: "UNKNOWN_PHASE_VEL",
    0x00200000: "UNKNOWN_TORQUE",
    0x00400000: "UNKNOWN_CURRENT_COMMAND",
    0x00800000: "UNKNOWN_CURRENT_MEASUREMENT",
    0x01000000: "UNKNOWN_VBUS_VOLTAGE",
    0x02000000: "UNKNOWN_VOLTAGE_COMMAND",
    0x04000000: "UNKNOWN_GAINS",
    0x08000000: "CONTROLLER_INITIALIZING",
    0x10000000: "UNBALANCED_PHASES",
}
ENCODER_ERRORS = {
    0x00000001: "UNSTABLE_GAIN",
    0x00000002: "CPR_POLEPAIRS_MISMATCH",
    0x00000004: "NO_RESPONSE",
    0x00000008: "UNSUPPORTED_ENCODER_MODE",
    0x00000010: "ILLEGAL_HALL_STATE",
    0x00000020: "INDEX_NOT_FOUND_YET",
    0x00000040: "ABS_SPI_TIMEOUT",
    0x00000080: "ABS_SPI_COM_FAIL",
    0x00000100: "ABS_SPI_NOT_READY",
    0x00000200: "HALL_NOT_CALIBRATED_YET",
}
CONTROLLER_ERRORS = {
    0x00000001: "OVERSPEED",
    0x00000002: "INVALID_INPUT_MODE",
    0x00000004: "UNSTABLE_GAIN",
    0x00000008: "INVALID_MIRROR_AXIS",
    0x00000010: "INVALID_LOAD_ENCODER",
    0x00000020: "INVALID_ESTIMATE",
    0x00000040: "INVALID_CIRCULAR_RANGE",
    0x00000080: "SPINOUT_DETECTED",
}

def decode(val, table):
    if val == 0:
        return "none"
    names = [name for bit, name in table.items() if val & bit]
    return ", ".join(names) if names else f"0x{val:08X}"

def print_errors(ax):
    print(f"  axis error      : {decode(ax.error, AXIS_ERRORS)}")
    print(f"  motor error     : {decode(ax.motor.error, MOTOR_ERRORS)}")
    print(f"  encoder error   : {decode(ax.encoder.error, ENCODER_ERRORS)}")
    print(f"  controller error: {decode(ax.controller.error, CONTROLLER_ERRORS)}")

# ── Connect ────────────────────────────────────────────────────────────────────
print("Connecting to ODrive...")
odrv = odrive.find_any()
ax   = getattr(odrv, f"axis{AXIS_NUM}")
print(f"Connected — serial {hex(odrv.serial_number)}  fw {odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
print(f"Vbus: {odrv.vbus_voltage:.2f} V")

# ── Initial state dump ─────────────────────────────────────────────────────────
print(f"\n=== axis{AXIS_NUM} state before test ===")
print(f"  current_state   : {ax.current_state}")
print(f"  encoder.is_ready: {ax.encoder.is_ready}")
print(f"  motor.is_calibrated: {ax.motor.is_calibrated}")
print(f"  vel_limit       : {ax.controller.config.vel_limit}")
print_errors(ax)

# ── Clear errors ───────────────────────────────────────────────────────────────
print("\nClearing errors...")
odrv.clear_errors()
# ax.clear_errors() not available on 0.5.6 — top-level clear_errors() covers all axes

# ── Raise vel_limit ────────────────────────────────────────────────────────────
print(f"Setting vel_limit -> {VEL_LIMIT} turns/s (was {ax.controller.config.vel_limit})")
ax.controller.config.vel_limit = VEL_LIMIT

# ── Calibrate if needed ────────────────────────────────────────────────────────
if not ax.motor.is_calibrated:
    print("\nMotor not calibrated — running full calibration sequence (motor will beep)...")
    ax.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
    while ax.current_state != AXIS_STATE_IDLE:
        time.sleep(0.2)
    print(f"  Calibration done. motor.is_calibrated = {ax.motor.is_calibrated}")
    print_errors(ax)
    if ax.error or ax.motor.error or ax.encoder.error:
        print("ERROR: calibration failed — aborting.")
        sys.exit(1)
else:
    print("Motor already calibrated — skipping calibration.")

# ── Set velocity control mode ─────────────────────────────────────────────────
print("\nConfiguring velocity control mode...")
ax.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
ax.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
ax.controller.input_vel           = 0.0

# ── Enter closed loop ─────────────────────────────────────────────────────────
print("Entering closed loop control...")
ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.2)
if ax.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
    print(f"  Failed to enter closed loop! state = {ax.current_state}")
    print_errors(ax)
    sys.exit(1)
print("  In closed loop.")

# ── Command velocity and monitor ──────────────────────────────────────────────
print(f"\nCommanding {TEST_VEL} turns/s for {RUN_SEC} s — monitoring errors...")
ax.controller.input_vel = TEST_VEL
t_start = time.monotonic()
error_detected = False
try:
    while time.monotonic() - t_start < RUN_SEC:
        t = time.monotonic() - t_start
        vel = ax.encoder.vel_estimate
        pos = ax.encoder.pos_estimate
        iq  = ax.motor.current_control.Iq_measured
        if ax.error or ax.motor.error or ax.encoder.error or ax.controller.error:
            print(f"\n[{t:.2f}s] ERROR DETECTED — stopping")
            print_errors(ax)
            error_detected = True
            break
        print(f"  t={t:.2f}s  vel={vel:+.2f} tr/s  pos={pos:.2f}  Iq={iq:.2f} A", end="\r")
        time.sleep(0.05)
finally:
    ax.controller.input_vel = 0.0
    ax.requested_state = AXIS_STATE_IDLE
    print()

# ── Final state ───────────────────────────────────────────────────────────────
print(f"\n=== axis{AXIS_NUM} state after test ===")
print_errors(ax)
if not error_detected:
    print("Test completed without errors.")
