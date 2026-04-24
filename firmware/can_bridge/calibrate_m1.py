"""calibrate_m1.py - Calibrate ODrive axis1 (M1) and save pre_calibrated flags."""
import sys, time
import odrive
from odrive.enums import *

print("Connecting to ODrive...")
odrv0 = odrive.find_any(timeout=10)
print("Connected: serial=%s  hw=%s  fw=%s" % (
    hex(odrv0.serial_number),
    "%d.%d-%d" % (odrv0.hw_version_major, odrv0.hw_version_minor, odrv0.hw_version_variant),
    "%d.%d.%d" % (odrv0.fw_version_major, odrv0.fw_version_minor, odrv0.fw_version_revision)))

ax = odrv0.axis1

print()
print("=== axis1 pre-calibration state ===")
print("  axis1.error             = 0x%08X" % ax.error)
print("  axis1.current_state     = %d" % ax.current_state)
print("  axis1.motor.error       = 0x%08X" % ax.motor.error)
print("  axis1.encoder.error     = 0x%08X" % ax.encoder.error)
print("  motor.is_calibrated     = %s" % ax.motor.is_calibrated)
print("  encoder.is_ready        = %s" % ax.encoder.is_ready)
print("  motor.pre_calibrated    = %s" % ax.motor.config.pre_calibrated)
print("  encoder.pre_calibrated  = %s" % ax.encoder.config.pre_calibrated)

# Clear any existing errors
print()
print("Clearing errors...")
odrv0.clear_errors()
time.sleep(0.5)
print("  axis1.error after clear = 0x%08X" % ax.error)

# Run full calibration sequence
print()
print("Starting FULL_CALIBRATION_SEQUENCE on axis1...")
print("(Motor will beep and spin briefly - do not touch)")
ax.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
time.sleep(1.0)

# Wait for calibration to finish (back to IDLE = state 1)
timeout = 60
t0 = time.time()
while ax.current_state != AXIS_STATE_IDLE:
    elapsed = time.time() - t0
    if elapsed > timeout:
        print("ERROR: Calibration timed out after %ds (stuck in state %d)" % (timeout, ax.current_state))
        print("  axis1.error         = 0x%08X" % ax.error)
        print("  axis1.motor.error   = 0x%08X" % ax.motor.error)
        print("  axis1.encoder.error = 0x%08X" % ax.encoder.error)
        sys.exit(1)
    print("  [%.1fs] state=%d  motor_err=0x%X  enc_err=0x%X" % (
        elapsed, ax.current_state, ax.motor.error, ax.encoder.error), end='\r')
    time.sleep(0.5)

print()
print("Calibration sequence finished.")

print()
print("=== axis1 post-calibration state ===")
print("  axis1.error             = 0x%08X" % ax.error)
print("  axis1.motor.error       = 0x%08X" % ax.motor.error)
print("  axis1.encoder.error     = 0x%08X" % ax.encoder.error)
print("  motor.is_calibrated     = %s" % ax.motor.is_calibrated)
print("  encoder.is_ready        = %s" % ax.encoder.is_ready)

if ax.error != 0:
    print()
    print("ERROR: axis1 has errors after calibration - cannot set pre_calibrated.")
    print("Check motor connections and encoder wiring on M1.")
    sys.exit(1)

# Set pre_calibrated so power-cycle skips re-calibration
print()
print("Setting pre_calibrated flags...")
ax.motor.config.pre_calibrated   = True
ax.encoder.config.pre_calibrated = True
print("  motor.pre_calibrated   = True")
print("  encoder.pre_calibrated = True")

print()
print("Saving configuration (ODrive will reboot)...")
odrv0.save_configuration()
# save_configuration() triggers a reboot so the script will lose connection
time.sleep(3.0)
print()
print("Done. ODrive has rebooted with saved calibration.")
print("Run diagnose_m1.py to confirm M1 can now enter closed-loop via CAN.")
