"""
calibrate.py — ODESC V4.2 full motor+encoder calibration
Firmware: modified v0.5.1 (no clear_errors, no .can sub-object)
Motor: Maytech MTO5065-70-HA-C  (7 pole pairs, ~0.05 ohm phase R)
Encoder: AS5047 SPI absolute (cpr=16384, CS on GPIO3)
"""
import sys
import time
import odrive
from odrive.enums import *

# v0.5.1 enum fallbacks (names vary across odrive package versions)
try:
    MOTOR_TYPE_HIGH_CURRENT
except NameError:
    MOTOR_TYPE_HIGH_CURRENT = 0
try:
    ENCODER_MODE_SPI_ABS_AMS
except NameError:
    ENCODER_MODE_SPI_ABS_AMS = 257
try:
    CONTROL_MODE_VELOCITY_CONTROL
except NameError:
    CONTROL_MODE_VELOCITY_CONTROL = 2
try:
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE
except NameError:
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE = 3
try:
    AXIS_STATE_IDLE
except NameError:
    AXIS_STATE_IDLE = 1

# ── Connect ─────────────────────────────────────────────────────────────────
print("Searching for ODESC...")
odrv = odrive.find_any(timeout=10)
ax = odrv.axis0
print(f"Connected  fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
print(f"Vbus = {odrv.vbus_voltage:.2f} V")

if odrv.vbus_voltage < 10:
    print("ERROR: Vbus too low — check PSU (need 24V, >=5A for calibration)")
    sys.exit(1)

# ── Pre-calibration config ──────────────────────────────────────────────────
print("\n=== Configuring motor ===")
m = ax.motor.config
m.motor_type = MOTOR_TYPE_HIGH_CURRENT  # 0
m.pole_pairs = 7
m.calibration_current = 10.0
m.resistance_calib_max_voltage = 4.0
m.current_lim = 15.0
m.torque_constant = 8.27 / 70  # Kv=70
print(f"  motor_type={m.motor_type}  pole_pairs={m.pole_pairs}")
print(f"  calibration_current={m.calibration_current}  resistance_calib_max_voltage={m.resistance_calib_max_voltage}")
print(f"  current_lim={m.current_lim}  torque_constant={m.torque_constant:.4f}")

print("\n=== Configuring encoder ===")
e = ax.encoder.config
e.mode = ENCODER_MODE_SPI_ABS_AMS  # 257
e.cpr = 16384
e.abs_spi_cs_gpio_pin = 3
print(f"  mode={e.mode}  cpr={e.cpr}  cs_pin={e.abs_spi_cs_gpio_pin}")

print("\n=== Configuring controller ===")
c = ax.controller.config
c.control_mode = CONTROL_MODE_VELOCITY_CONTROL
c.vel_limit = 5.0
print(f"  control_mode={c.control_mode}  vel_limit={c.vel_limit}")

# ── Print any pre-existing errors ───────────────────────────────────────────
print(f"\n=== Pre-cal errors ===")
print(f"  axis error  : 0x{ax.error:08X}")
print(f"  motor error : 0x{ax.motor.error:08X}")
print(f"  encoder error: 0x{ax.encoder.error:08X}")

# ── Run full calibration sequence ───────────────────────────────────────────
print("\n=== Starting FULL CALIBRATION SEQUENCE ===")
print("  (motor beep -> resistance/inductance -> encoder offset search)")
ax.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

# Poll until calibration finishes (returns to IDLE=1)
t0 = time.monotonic()
prev_state = None
while True:
    state = ax.current_state
    if state != prev_state:
        elapsed = time.monotonic() - t0
        print(f"  [{elapsed:5.1f}s] state = {state}")
        prev_state = state
    if state == AXIS_STATE_IDLE:
        if time.monotonic() - t0 > 2.0:  # give it time to leave IDLE first
            break
    if time.monotonic() - t0 > 30.0:
        print("  TIMEOUT — calibration took >30s, aborting")
        break
    time.sleep(0.2)

elapsed = time.monotonic() - t0
print(f"\nCalibration finished in {elapsed:.1f}s")

# ── Results ─────────────────────────────────────────────────────────────────
print(f"\n=== Post-cal errors ===")
print(f"  axis error  : 0x{ax.error:08X}")
print(f"  motor error : 0x{ax.motor.error:08X}")
print(f"  encoder error: 0x{ax.encoder.error:08X}")

print(f"\n=== Measured motor params ===")
print(f"  phase_resistance  = {ax.motor.config.phase_resistance:.6f} ohm")
print(f"  phase_inductance  = {ax.motor.config.phase_inductance:.6e} H")
print(f"  motor.is_calibrated = {ax.motor.is_calibrated}")

print(f"\n=== Encoder ===")
print(f"  encoder.is_ready = {ax.encoder.is_ready}")
try:
    print(f"  shadow_count     = {ax.encoder.shadow_count}")
    print(f"  pos_estimate     = {ax.encoder.pos_estimate:.4f}")
except:
    pass

# ── Check success ───────────────────────────────────────────────────────────
if ax.error != 0 or ax.motor.error != 0 or ax.encoder.error != 0:
    print("\n*** CALIBRATION FAILED — check errors above ***")
    # Decode common motor errors
    merr = ax.motor.error
    if merr & 0x01:  print("  -> PHASE_RESISTANCE_OUT_OF_RANGE")
    if merr & 0x02:  print("  -> PHASE_INDUCTANCE_OUT_OF_RANGE")
    if merr & 0x04:  print("  -> ADC_FAILED (DRV8301 issue?)")
    if merr & 0x10:  print("  -> DRV_FAULT")
    sys.exit(1)
else:
    print("\n*** CALIBRATION OK ***")
    print("Run odrv0.save_configuration() to persist, or re-run to verify repeatability.")
