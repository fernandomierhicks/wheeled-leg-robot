import odrive
print("Searching...")
odrv = odrive.find_any()
print(f"Connected! Serial: {hex(odrv.serial_number)}")
print(f"Vbus: {odrv.vbus_voltage:.2f} V")
