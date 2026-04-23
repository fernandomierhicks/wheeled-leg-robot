"""Direct ODrive encoder setup + diagnostic — bypasses GUI entirely.

Connects to the ODrive over USB, configures the SPI absolute encoder,
saves, reboots, reconnects, and then checks if the encoder is actually
communicating over SPI.

Usage:  python setup_encoder_direct.py
"""

import sys
import time
import odrive
from odrive.enums import *

AXIS = 0  # axis0
ENC_MODE = 256  # ENCODER_MODE_SPI_ABS_AMS
ENC_CPR = 16384  # AS5047P 14-bit
ENC_CS_PIN = 3  # GPIO3 for SPI chip-select


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def find_odrive():
    print("Searching for ODrive (timeout 10s)...")
    try:
        odrv = odrive.find_any(timeout=10)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    sn = hex(odrv.serial_number)
    fw = f"{odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}"
    hw = f"{odrv.hw_version_major}.{odrv.hw_version_minor}.{odrv.hw_version_variant}"
    print(f"Connected: serial={sn}  hw={hw}  fw={fw}")
    print(f"Vbus: {odrv.vbus_voltage:.1f} V")
    return odrv


def read_encoder_state(odrv, label=""):
    """Read and print all encoder-related state."""
    ax = odrv.axis0 if AXIS == 0 else odrv.axis1
    ec = ax.encoder.config
    enc = ax.encoder

    if label:
        print(f"\n--- {label} ---")

    print(f"  encoder.config.mode            = {ec.mode}")
    print(f"  encoder.config.cpr             = {ec.cpr}")
    print(f"  encoder.config.abs_spi_cs_gpio_pin = {ec.abs_spi_cs_gpio_pin}")
    print(f"  encoder.config.pre_calibrated  = {ec.pre_calibrated}")
    print(f"  encoder.config.bandwidth       = {ec.bandwidth}")
    print(f"  encoder.error                  = 0x{enc.error:04X}")
    print(f"  encoder.is_ready               = {enc.is_ready}")
    print(f"  encoder.pos_estimate           = {enc.pos_estimate:.4f}")
    print(f"  encoder.vel_estimate           = {enc.vel_estimate:.4f}")

    # Check shadow_count and count_in_cpr for SPI
    try:
        print(f"  encoder.shadow_count           = {enc.shadow_count}")
        print(f"  encoder.count_in_cpr           = {enc.count_in_cpr}")
    except Exception:
        pass

    # Axis-level info
    print(f"  axis.current_state             = {ax.current_state}")
    print(f"  axis.error                     = 0x{ax.error:04X}")
    print(f"  motor.error                    = 0x{ax.motor.error:04X}")

    return {
        "mode": ec.mode,
        "cpr": ec.cpr,
        "cs_pin": ec.abs_spi_cs_gpio_pin,
        "enc_error": enc.error,
        "is_ready": enc.is_ready,
        "pos_estimate": enc.pos_estimate,
        "axis_error": ax.error,
        "motor_error": ax.motor.error,
    }


def main():
    banner("STEP 1: Connect & read current state")
    odrv = find_odrive()
    state = read_encoder_state(odrv, "Current encoder state")

    # Check if config already matches
    needs_save = False
    if state["mode"] != ENC_MODE:
        print(f"\n  >> Encoder mode is {state['mode']}, need {ENC_MODE} — will fix")
        needs_save = True
    if state["cpr"] != ENC_CPR:
        print(f"  >> CPR is {state['cpr']}, need {ENC_CPR} — will fix")
        needs_save = True
    if state["cs_pin"] != ENC_CS_PIN:
        print(f"  >> CS pin is {state['cs_pin']}, need {ENC_CS_PIN} — will fix")
        needs_save = True

    banner("STEP 2: Write encoder config to RAM")
    ax = odrv.axis0 if AXIS == 0 else odrv.axis1
    ec = ax.encoder.config

    ec.mode = ENC_MODE
    ec.cpr = ENC_CPR
    ec.abs_spi_cs_gpio_pin = ENC_CS_PIN

    # ODrive v3.6 hardware SPI3 sits on GPIO 3/4/5. After a factory erase
    # those pins default to GPIO_MODE_ANALOG_IN (3), which disconnects the
    # STM32 digital buffer so the SPI peripheral cannot drive the bus.
    # Symptom: spi_error_rate ≈ 1.0 with one frozen pos_abs read at boot.
    # Force all three to DIGITAL (0).
    odrv.config.gpio3_mode = 0
    odrv.config.gpio4_mode = 0
    odrv.config.gpio5_mode = 0

    print(f"  Set mode = {ENC_MODE} (SPI_ABS_AMS)")
    print(f"  Set CPR  = {ENC_CPR}")
    print(f"  Set CS   = GPIO{ENC_CS_PIN}")
    print(f"  Set gpio3/4/5_mode = 0 (DIGITAL)")

    # Verify RAM write
    print(f"  Verify: mode={ec.mode}  cpr={ec.cpr}  cs={ec.abs_spi_cs_gpio_pin}")
    print(f"  Verify: gpio3={odrv.config.gpio3_mode}  gpio4={odrv.config.gpio4_mode}  gpio5={odrv.config.gpio5_mode}")

    banner("STEP 3: Save configuration (triggers reboot)")
    print("  Saving...")
    try:
        odrv.save_configuration()
    except Exception:
        pass  # reboot kills USB — expected
    print("  Save sent — ODrive is rebooting...")

    banner("STEP 4: Wait for reboot & reconnect")
    print("  Waiting 5 seconds for ODrive to reboot...")
    time.sleep(5)
    odrv = find_odrive()

    banner("STEP 5: Verify saved config")
    state = read_encoder_state(odrv, "Post-reboot encoder state")

    ok = True
    if state["mode"] != ENC_MODE:
        print(f"  FAIL: mode={state['mode']} (expected {ENC_MODE})")
        ok = False
    if state["cpr"] != ENC_CPR:
        print(f"  FAIL: cpr={state['cpr']} (expected {ENC_CPR})")
        ok = False
    if state["cs_pin"] != ENC_CS_PIN:
        print(f"  FAIL: cs_pin={state['cs_pin']} (expected {ENC_CS_PIN})")
        ok = False

    if ok:
        print("\n  CONFIG VERIFIED OK")
    else:
        print("\n  CONFIG VERIFICATION FAILED — check above")
        sys.exit(1)

    banner("STEP 6: SPI communication diagnostic")

    # Clear any errors first
    ax = odrv.axis0 if AXIS == 0 else odrv.axis1
    try:
        odrv.clear_errors()
    except Exception:
        pass
    time.sleep(0.5)

    enc = ax.encoder
    print(f"  After clear_errors:")
    print(f"    encoder.error      = 0x{enc.error:04X}")
    print(f"    encoder.is_ready   = {enc.is_ready}")
    print(f"    encoder.pos_estimate = {enc.pos_estimate:.4f}")

    # Sample position a few times to see if it's changing / valid
    print("\n  Sampling encoder position (rotate shaft if possible):")
    positions = []
    for i in range(5):
        pos = enc.pos_estimate
        shadow = enc.shadow_count
        cpr_val = enc.count_in_cpr
        err = enc.error
        print(f"    [{i}] pos={pos:10.4f}  shadow={shadow:8d}  count_in_cpr={cpr_val:6d}  err=0x{err:04X}")
        positions.append(pos)
        time.sleep(0.3)

    # Analysis
    banner("DIAGNOSIS")
    enc_err = enc.error
    if enc_err != 0:
        print(f"  ENCODER ERROR: 0x{enc_err:04X}")
        if enc_err & 0x0001:
            print("    - ERROR_UNSTABLE_GAIN")
        if enc_err & 0x0002:
            print("    - ERROR_CPR_POLEPAIRS_MISMATCH")
        if enc_err & 0x0004:
            print("    - ERROR_NO_RESPONSE")
        if enc_err & 0x0008:
            print("    - ERROR_UNSUPPORTED_ENCODER_MODE")
        if enc_err & 0x0010:
            print("    - ERROR_ILLEGAL_HALL_STATE")
        if enc_err & 0x0020:
            print("    - ERROR_INDEX_NOT_FOUND_YET")
        if enc_err & 0x0040:
            print("    - ERROR_ABS_SPI_TIMEOUT")
        if enc_err & 0x0080:
            print("    - ERROR_ABS_SPI_COM_FAIL")
        if enc_err & 0x0100:
            print("    - ERROR_ABS_SPI_NOT_READY")

        if enc_err & 0x0040:
            print("\n  >> SPI TIMEOUT: the encoder is not responding on SPI.")
            print("     Check wiring: MOSI, MISO, SCLK, and CS to GPIO3.")
            print("     On ODrive v3.6:")
            print("       GPIO1 = SPI SCK")
            print("       GPIO2 = SPI MISO")
            print("       GPIO3 = SPI CS (default)")
            print("       GPIO4 = SPI MOSI (if needed)")
            print("     Make sure encoder VDD=3.3V (NOT 5V from ODrive 5V rail)")
        elif enc_err & 0x0080:
            print("\n  >> SPI COM FAIL: SPI bus sees the encoder but gets bad data.")
            print("     Check for noise, long wires, or loose connections.")
        elif enc_err & 0x0100:
            print("\n  >> SPI NOT READY: encoder responded but is not ready.")
            print("     This can happen if the encoder needs time to warm up,")
            print("     or if the magnet is missing / too far from the IC.")
    else:
        all_same = len(set(f"{p:.4f}" for p in positions)) == 1
        if enc.is_ready:
            print("  ENCODER IS WORKING!")
            print(f"  Position: {enc.pos_estimate:.4f}")
            if all_same:
                print("  (Position stable — shaft not rotating, which is normal)")
            else:
                print("  (Position changing — shaft movement detected)")
        else:
            print("  No encoder error, but encoder.is_ready = False")
            print("  The encoder may need calibration (offset calibration).")
            print("  Try: odrv0.axis0.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION")

    print()


if __name__ == "__main__":
    main()
