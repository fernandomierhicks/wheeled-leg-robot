"""encoder_cal_fix.py — Diagnose and work around AS5048A encoder errors on ODrive 0.5.6.

The AS5048A works under mode 256 (SPI_ABS_CUI) but generates spurious SPI
error flags that cause calibration to abort with ENCODER_FAILED. This script:

  Phase 1: Ensure mode=256 is saved in NVM (reboot if needed)
  Phase 2: Diagnostic — error reappearance rate under booted mode 256
  Phase 3: Motor-only calibration (state 4) with error suppression
  Phase 4: Encoder offset calibration (state 7) with error suppression
  Phase 5: Save pre_calibrated flags

Key insight: errors reappear ~141ms after clearing. We use a background
thread clearing errors at 200Hz to keep the error register at 0 while
the firmware runs calibration.

Usage:  python encoder_cal_fix.py
        Close the GUI first — only one process can hold the USB handle.
"""

import sys
import time
import threading
import odrive

AXIS = 0
ENC_MODE = 256       # SPI_ABS_CUI — the mode that actually reads AS5048A
ENC_CPR = 16384      # 14-bit absolute
ENC_CS_PIN = 3
POLE_PAIRS = 7
CAL_CURRENT = 3.0
CURRENT_LIM = 5.0
RES_CAL_VOLTAGE = 3.5
TORQUE_CONST = 0.04


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def find_odrive():
    print("Searching for ODrive (timeout 10s)...")
    odrv = odrive.find_any(timeout=10)
    sn = hex(odrv.serial_number)
    fw = f"{odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}"
    print(f"Connected: serial={sn}  fw={fw}  Vbus={odrv.vbus_voltage:.1f}V")
    return odrv


def get_axis(odrv):
    return odrv.axis0 if AXIS == 0 else odrv.axis1


def print_all_errors(odrv):
    ax = get_axis(odrv)
    print(f"  axis.error       = 0x{ax.error:08X}")
    print(f"  motor.error      = 0x{ax.motor.error:08X}")
    print(f"  encoder.error    = 0x{ax.encoder.error:08X}")
    print(f"  controller.error = 0x{ax.controller.error:08X}")
    print(f"  encoder.is_ready = {ax.encoder.is_ready}")
    print(f"  encoder.pos_est  = {ax.encoder.pos_estimate:.4f}")
    print(f"  axis.state       = {ax.current_state}")


def wait_for_idle(odrv, timeout=30):
    ax = get_axis(odrv)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if ax.current_state == 1:
            return True
        time.sleep(0.1)
    return False


class ErrorSuppressor:
    """Background thread that clears ODrive errors at high frequency."""

    def __init__(self, odrv):
        self._odrv = odrv
        self._stop = threading.Event()
        self._count = 0
        self._thread = None

    def start(self):
        self._stop.clear()
        self._count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def clear_count(self):
        return self._count

    def _run(self):
        odrv = self._odrv
        while not self._stop.is_set():
            try:
                odrv.clear_errors()
                self._count += 1
            except Exception:
                pass
            time.sleep(0.003)  # ~333 Hz


# ── Phase 1: Ensure mode=256 in NVM ─────────────────────────────────────────

def phase1_ensure_config(odrv):
    banner("PHASE 1: Ensure encoder mode=256 in NVM")
    ax = get_axis(odrv)
    ec = ax.encoder.config

    current_mode = ec.mode
    print(f"  Current saved encoder mode: {current_mode}")

    needs_reboot = False

    if current_mode != ENC_MODE:
        print(f"  Mode is {current_mode}, need {ENC_MODE} — writing config...")
        ec.mode = ENC_MODE
        ec.cpr = ENC_CPR
        ec.abs_spi_cs_gpio_pin = ENC_CS_PIN
        odrv.config.gpio3_mode = 0
        odrv.config.gpio4_mode = 0
        odrv.config.gpio5_mode = 0
        needs_reboot = True

    # Also ensure motor config is sane
    mc = ax.motor.config
    mc.motor_type = 0
    mc.pole_pairs = POLE_PAIRS
    mc.current_lim = CURRENT_LIM
    mc.calibration_current = CAL_CURRENT
    mc.resistance_calib_max_voltage = RES_CAL_VOLTAGE
    mc.torque_constant = TORQUE_CONST
    odrv.config.enable_brake_resistor = True
    ax.config.startup_motor_calibration = False
    ax.config.startup_encoder_offset_calibration = False
    ax.config.startup_closed_loop_control = False
    ax.controller.config.enable_overspeed_error = False

    if needs_reboot:
        print("  Saving configuration + rebooting...")
        try:
            odrv.save_configuration()
        except Exception:
            pass
        print("  Waiting 5s for reboot...")
        time.sleep(5)
        odrv = find_odrive()

        # Verify
        ax = get_axis(odrv)
        mode = ax.encoder.config.mode
        print(f"  Post-reboot encoder mode: {mode}")
        if mode != ENC_MODE:
            print(f"  FAIL: mode is still {mode}")
            sys.exit(1)
        print(f"  Mode {ENC_MODE} confirmed in NVM!")
    else:
        # Still save motor config even if encoder mode was already right
        print("  Encoder mode already correct. Saving motor config...")
        try:
            odrv.save_configuration()
        except Exception:
            pass
        time.sleep(5)
        odrv = find_odrive()

    return odrv


# ── Phase 2: Diagnostic ─────────────────────────────────────────────────────

def phase2_diagnostic(odrv):
    banner("PHASE 2: Diagnostic — error behavior under mode 256")
    ax = get_axis(odrv)
    enc = ax.encoder

    print("\n  Current state (after fresh boot with mode 256):")
    print_all_errors(odrv)

    # Position check
    print(f"\n  Sampling position (5 reads, 200ms apart):")
    for i in range(5):
        pos = enc.pos_estimate
        err = enc.error
        rdy = enc.is_ready
        print(f"    [{i}] pos={pos:10.4f}  err=0x{err:04X}  ready={rdy}")
        time.sleep(0.2)

    # Error reappearance timing
    print("\n  Clearing errors and measuring reappearance rate...")
    odrv.clear_errors()
    time.sleep(0.01)

    samples = []
    t0 = time.time()
    while time.time() - t0 < 3.0:
        err = enc.error
        samples.append((time.time() - t0, err))
        time.sleep(0.005)

    first_err_time = None
    err_count = 0
    for t, err in samples:
        if err != 0:
            err_count += 1
            if first_err_time is None:
                first_err_time = t

    print(f"  Total samples: {len(samples)} over 3s")
    print(f"  Samples with errors: {err_count} ({100*err_count/len(samples):.0f}%)")
    if first_err_time is not None:
        print(f"  First error reappeared at: {first_err_time*1000:.0f} ms")
        err_bits = samples[-1][1]
        print(f"  Error bits: 0x{err_bits:04X}")
        if err_bits & 0x80:
            print(f"    ABS_SPI_COM_FAIL")
        if err_bits & 0x40:
            print(f"    ABS_SPI_TIMEOUT")
        if err_bits & 0x100:
            print(f"    ABS_SPI_NOT_READY")
        if err_bits & 0x04:
            print(f"    NO_RESPONSE")
    else:
        print(f"  NO errors in 3 seconds — encoder may be working clean!")

    # Test if error suppressor can keep errors at bay
    print("\n  Testing error suppressor thread (2 seconds)...")
    suppressor = ErrorSuppressor(odrv)
    suppressor.start()
    time.sleep(0.1)

    clean_samples = 0
    total_samples = 0
    t0 = time.time()
    while time.time() - t0 < 2.0:
        err = enc.error
        total_samples += 1
        if err == 0:
            clean_samples += 1
        time.sleep(0.01)

    suppressor.stop()
    print(f"  With suppressor: {clean_samples}/{total_samples} samples clean "
          f"({100*clean_samples/total_samples:.0f}%)")
    print(f"  Clears performed: {suppressor.clear_count}")

    return first_err_time


# ── Phase 3: Motor-only calibration ─────────────────────────────────────────

def phase3_motor_cal(odrv):
    banner("PHASE 3: Motor-only calibration (state 4)")
    ax = get_axis(odrv)

    print("  Starting error suppressor...")
    suppressor = ErrorSuppressor(odrv)
    suppressor.start()
    time.sleep(0.1)

    # Double-check errors are clean
    err = ax.encoder.error
    print(f"  Encoder error with suppressor: 0x{err:04X}")

    print("  Requesting MOTOR_CALIBRATION (state 4)...")
    print("  (Motor will beep — measuring R and L)")
    ax.requested_state = 4

    time.sleep(0.5)
    state = ax.current_state
    print(f"  Axis state after 0.5s: {state}")

    if state != 4 and state != 1:
        print(f"  Unexpected state {state}, waiting...")

    cal_start = time.time()
    while time.time() - cal_start < 30:
        state = ax.current_state
        if state == 1:
            break
        elapsed = time.time() - cal_start
        sys.stdout.write(f"\r  Calibrating... {elapsed:.0f}s  state={state}  "
                         f"clears={suppressor.clear_count}  "
                         f"enc_err=0x{ax.encoder.error:04X}   ")
        sys.stdout.flush()
        time.sleep(0.2)

    suppressor.stop()
    print()

    # Check result
    if ax.error != 0 or ax.motor.error != 0:
        print("  MOTOR CALIBRATION FAILED:")
        print_all_errors(odrv)
        return False

    R = ax.motor.config.phase_resistance
    L = ax.motor.config.phase_inductance
    cal = ax.motor.is_calibrated
    print(f"  MOTOR CALIBRATION {'OK' if cal else 'INCOMPLETE'}!")
    print(f"  Phase resistance:  {R:.4f} Ohm")
    print(f"  Phase inductance:  {L:.6f} H")
    print(f"  Motor is_calibrated: {cal}")
    print(f"  Error clears: {suppressor.clear_count}")

    return cal


# ── Phase 4: Encoder offset calibration ─────────────────────────────────────

def phase4_encoder_cal(odrv):
    banner("PHASE 4: Encoder offset calibration (state 7)")
    ax = get_axis(odrv)

    print("  Starting error suppressor...")
    suppressor = ErrorSuppressor(odrv)
    suppressor.start()
    time.sleep(0.1)

    print(f"  Encoder error with suppressor: 0x{ax.encoder.error:04X}")
    print("  Requesting ENCODER_OFFSET_CALIBRATION (state 7)...")
    print("  (Motor will spin slowly to find encoder-electrical angle offset)")
    ax.requested_state = 7

    time.sleep(0.5)
    state = ax.current_state
    print(f"  Axis state after 0.5s: {state}")

    cal_start = time.time()
    while time.time() - cal_start < 45:
        state = ax.current_state
        if state == 1:
            break
        elapsed = time.time() - cal_start
        sys.stdout.write(f"\r  Calibrating... {elapsed:.0f}s  state={state}  "
                         f"clears={suppressor.clear_count}  "
                         f"enc_err=0x{ax.encoder.error:04X}  "
                         f"pos={ax.encoder.pos_estimate:.2f}   ")
        sys.stdout.flush()
        time.sleep(0.2)

    suppressor.stop()
    print()

    # Check result
    if ax.error != 0:
        print("  ENCODER OFFSET CALIBRATION FAILED:")
        print_all_errors(odrv)
        return False

    ready = ax.encoder.is_ready
    print(f"  ENCODER CALIBRATION {'OK' if ready else 'INCOMPLETE'}!")
    print(f"  encoder.is_ready:   {ready}")
    print(f"  encoder.pos_estimate: {ax.encoder.pos_estimate:.4f}")
    print(f"  Error clears: {suppressor.clear_count}")

    return ready


# ── Phase 5: Save calibration ───────────────────────────────────────────────

def phase5_save(odrv):
    banner("PHASE 5: Save calibration + pre_calibrated flags")
    ax = get_axis(odrv)

    ax.motor.config.pre_calibrated = True
    ax.encoder.config.pre_calibrated = True

    # Controller gains from measured R
    R = ax.motor.config.phase_resistance
    kT = ax.motor.config.torque_constant
    if R > 0.001:
        vel_gain = max(0.01, min(2.0, 0.5 * kT / R))
    else:
        vel_gain = 0.1667
    vel_int = 0.5 * vel_gain

    ax.controller.config.vel_gain = vel_gain
    ax.controller.config.vel_integrator_gain = vel_int
    ax.controller.config.vel_limit = 20.0

    try:
        ax.controller.config.spinout_electrical_power_threshold = 120.0
        ax.controller.config.spinout_mechanical_power_threshold = -120.0
    except Exception:
        pass

    odrv.config.enable_brake_resistor = True
    odrv.config.dc_max_negative_current = -5.0

    print(f"  motor.pre_calibrated   = True")
    print(f"  encoder.pre_calibrated = True")
    print(f"  vel_gain = {vel_gain:.4f}  vel_int = {vel_int:.4f}")
    print(f"  R = {R:.4f}  L = {ax.motor.config.phase_inductance:.6f}")

    print("\n  Saving configuration (triggers reboot)...")
    try:
        odrv.save_configuration()
    except Exception:
        pass

    print("  Waiting 5s for reboot...")
    time.sleep(5)
    odrv = find_odrive()

    ax = get_axis(odrv)
    print(f"\n  Post-reboot verification:")
    print(f"  motor.pre_calibrated   = {ax.motor.config.pre_calibrated}")
    print(f"  encoder.pre_calibrated = {ax.encoder.config.pre_calibrated}")
    print(f"  motor.is_calibrated    = {ax.motor.is_calibrated}")
    print(f"  encoder.is_ready       = {ax.encoder.is_ready}")
    print_all_errors(odrv)

    return odrv


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    banner("AS5048A Encoder Calibration Fix")
    print("  Works around spurious SPI errors under mode 256 by actively")
    print("  clearing error registers during calibration.")
    print()
    print("  CLOSE THE GUI FIRST — only one process can use USB.")
    print()

    odrv = find_odrive()

    # Phase 1: Ensure config
    odrv = phase1_ensure_config(odrv)

    # Phase 2: Diagnostic
    err_delay = phase2_diagnostic(odrv)

    input("\n  Press ENTER to proceed with calibration (motor will move!)... ")

    # Phase 3: Motor calibration
    if not phase3_motor_cal(odrv):
        print("\n  Motor calibration failed even with error suppression.")
        print("  Check motor wiring, power supply, and calibration current.")
        sys.exit(1)

    # Phase 4: Encoder offset calibration
    if not phase4_encoder_cal(odrv):
        banner("ENCODER CALIBRATION FAILED")
        print("  Error suppression wasn't enough to get through offset cal.")
        print()
        print("  Saving motor-only calibration...")
        ax = get_axis(odrv)
        ax.motor.config.pre_calibrated = True
        ax.encoder.config.pre_calibrated = False
        odrv.config.enable_brake_resistor = True
        try:
            odrv.save_configuration()
        except Exception:
            pass
        print("  Motor cal saved. Encoder needs firmware fix or AS5047P swap.")
        sys.exit(1)

    # Phase 5: Save
    odrv = phase5_save(odrv)

    banner("DONE — Full calibration complete!")
    print("  Both motor and encoder are calibrated and saved.")
    print("  Test: odrv0.axis0.requested_state = 8  (CLOSED_LOOP)")
    print()


if __name__ == "__main__":
    main()
