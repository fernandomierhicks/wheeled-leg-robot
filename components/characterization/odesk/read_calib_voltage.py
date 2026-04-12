"""Quick check: what is resistance_calib_max_voltage actually set to on the board?"""
import odrive

print("Connecting...")
odrv = odrive.find_any(timeout=10)
print(f"Connected  fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
print(f"Vbus = {odrv.vbus_voltage:.2f} V")

m = odrv.axis0.motor.config
print(f"\nresistance_calib_max_voltage = {m.resistance_calib_max_voltage}")
print(f"calibration_current          = {m.calibration_current}")
print(f"phase_resistance             = {m.phase_resistance}")
print(f"phase_inductance             = {m.phase_inductance}")
print(f"motor_type                   = {m.motor_type}")
print(f"pole_pairs                   = {m.pole_pairs}")
print(f"current_lim                  = {m.current_lim}")

print(f"\naxis0.error       = 0x{odrv.axis0.error:08X}")
print(f"motor.error       = 0x{odrv.axis0.motor.error:08X}")
print(f"encoder.error     = 0x{odrv.axis0.encoder.error:08X}")
