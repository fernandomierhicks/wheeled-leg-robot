"""
fix_and_spin.py — Disable anticogging, restore defaults, spin at 5 t/s
"""
import sys
import time
import odrive
from odrive.enums import *

AXIS_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print("Connecting to ODrive...")
odrv = odrive.find_any()
ax = getattr(odrv, f"axis{AXIS_NUM}")
print(f"Connected  fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}  Vbus={odrv.vbus_voltage:.1f}V")

# ── Check anticogging state ──────────────────────────────────────────────────
print("\n=== Anticogging state ===")
ac = ax.controller.config.anticogging
print(f"  anticogging_enabled : {ac.anticogging_enabled}")
print(f"  calib_anticogging   : {ac.calib_anticogging}")
print(f"  pre_calibrated      : {ac.pre_calibrated}")

# ── Disable anticogging ──────────────────────────────────────────────────────
print("\nDisabling anticogging...")
ac.anticogging_enabled = False
ac.pre_calibrated = False
# calib_anticogging is read-only runtime flag, clears on reboot

print(f"  anticogging_enabled = {ac.anticogging_enabled}")
print(f"  pre_calibrated      = {ac.pre_calibrated}")

# ── Save and reboot to clear any residual anticogging state ──────────────────
print("\nSaving config + reboot...")
try:
    odrv.save_configuration()
except:
    pass
time.sleep(4.0)

print("Reconnecting...")
odrv = odrive.find_any()
ax = getattr(odrv, f"axis{AXIS_NUM}")
print(f"Reconnected  Vbus={odrv.vbus_voltage:.1f}V")

# ── Clear errors, configure velocity mode ────────────────────────────────────
odrv.clear_errors()
ax.requested_state = AXIS_STATE_IDLE
time.sleep(0.3)

ax.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
ax.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
ax.controller.config.vel_limit = 20.0
ax.controller.input_vel = 0.0

print(f"\nGains: vel_gain={ax.controller.config.vel_gain:.4f}  "
      f"vel_integrator_gain={ax.controller.config.vel_integrator_gain:.4f}  "
      f"pos_gain={ax.controller.config.pos_gain:.1f}")

# ── Enter closed loop ────────────────────────────────────────────────────────
print("Entering closed loop...")
ax.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.3)
if ax.current_state != 8:
    print(f"FAILED state={ax.current_state} axis=0x{ax.error:08X} motor=0x{ax.motor.error:08X}")
    sys.exit(1)
print("In closed loop.")

# ── Command 5 t/s ────────────────────────────────────────────────────────────
print("Commanding 5 t/s for 5 seconds...")
ax.controller.input_vel = 5.0
t0 = time.monotonic()
try:
    while time.monotonic() - t0 < 5.0:
        vel = ax.encoder.vel_estimate
        iq = ax.motor.current_control.Iq_measured
        t = time.monotonic() - t0
        print(f"  t={t:.1f}s  vel={vel:+7.2f} t/s  Iq={iq:+.3f} A", end="\r")
        if ax.error or ax.motor.error:
            print(f"\n  ERROR: axis=0x{ax.error:08X} motor=0x{ax.motor.error:08X}")
            break
        time.sleep(0.05)
    print()
finally:
    ax.controller.input_vel = 0.0
    time.sleep(0.5)
    ax.requested_state = AXIS_STATE_IDLE
    print("Motor stopped, axis idle.")
