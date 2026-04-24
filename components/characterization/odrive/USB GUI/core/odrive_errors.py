"""odrive_errors.py — Error decode tables for ODrive 0.5.6.

Merged from odrive_gui_v2.py (with hint text) and vel_test.py (extra flags).
"""


# ── Axis errors ───────────────────────────────────────────────────────────────

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

AXIS_ERROR_HINTS = {
    0x00000001: "Requested state is invalid from the current state. "
                "Clear errors and ensure axis is Idle before commanding.",
    0x00000040: "Motor subsystem failed. Check motor error register for details.",
    0x00000080: "Sensorless estimator failed. Check sensorless estimator config.",
    0x00000100: "Encoder subsystem failed. Check encoder error register for details.",
    0x00000200: "Controller subsystem failed. Check controller error register for details.",
    0x00000800: "Watchdog timer expired — no keepalive received in time.",
    0x00001000: "Min endstop triggered.",
    0x00002000: "Max endstop triggered.",
    0x00004000: "Emergency stop was requested.",
    0x00010000: "Homing attempted without endstop configured.",
    0x00020000: "Axis over-temperature. Let ODrive cool. Check airflow and "
                "FET thermistor limits.",
    0x00040000: "Position unknown — absolute encoder not ready or not configured. "
                "Check encoder is_ready.",
}


# ── Motor errors ──────────────────────────────────────────────────────────────

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

MOTOR_ERROR_HINTS = {
    0x00000001: "Phase resistance measured out of range during calibration.\n"
                "Fix: Check motor wiring for shorts/opens. Increase "
                "resistance_calib_max_voltage (try 4-8 V). Verify pole pairs.",
    0x00000002: "Phase inductance measured out of range during calibration.\n"
                "Fix: Check motor wiring. May need to increase calibration "
                "current for high-inductance motors.",
    0x00000008: "Gate driver chip (DRV8301/DRV8323) asserted FAULT.\n"
                "Causes: overcurrent trip, short circuit, gate driver "
                "undervoltage, overtemperature.\n"
                "Fix: Power off immediately. Check motor wiring for "
                "phase-to-phase shorts. Let ODrive cool. Reduce current_lim.",
    0x00000010: "Control loop deadline missed — MCU could not complete the "
                "control step in time.\n"
                "Fix: Reboot ODrive. If persistent, may indicate firmware "
                "or hardware issue.",
    0x00000020: "PWM modulation magnitude too high — back-EMF too high for "
                "bus voltage at this speed.\n"
                "Fix: Reduce vel_limit. Increase bus voltage.",
    0x00000040: "Current sense ADC saturated — actual current exceeded "
                "measurement range.\n"
                "Fix: Reduce load. Check for motor wiring shorts. "
                "Reduce current_lim.",
    0x00000100: "Motor current exceeded current_lim while running.\n"
                "Fix: Reduce velocity setpoint or load. Increase current_lim "
                "if motor can handle it. Check for mechanical binding.",
    0x00000200: "Modulation value became NaN — numerical instability.\n"
                "Fix: Reduce gains (vel_gain, vel_integrator_gain). "
                "Clear errors and recalibrate.",
    0x00000400: "Motor thermistor over-temperature.\n"
                "Fix: Let motor cool. Check thermistor wiring. "
                "Reduce current_lim.",
    0x00000800: "FET thermistor over-temperature (ODrive board too hot).\n"
                "Fix: Let ODrive cool. Improve airflow. Reduce current_lim.",
    0x00001000: "Timer update missed — control loop timing slip.\n"
                "Fix: Reboot ODrive.",
    0x00002000: "Current measurement unavailable — ADC data not ready.\n"
                "Fix: Reboot ODrive. Check for power supply noise.",
    0x00004000: "Motor controller subsystem failed. See controller error.\n"
                "Fix: Check controller error for root cause "
                "(e.g. OVERSPEED, SPINOUT_DETECTED).",
    0x00008000: "I_bus (DC bus current) measurement out of range.\n"
                "Fix: Check bus current draw. Reduce load or current_lim.",
    0x00010000: "Brake resistor disarmed.\n"
                "Fix: Set odrv0.config.enable_brake_resistor = True "
                "and save_configuration().",
    0x00020000: "System-level fault — cascading error from another subsystem.\n"
                "Most common cause: brake resistor not enabled.\n"
                "Fix: Enable brake resistor, clear errors, recalibrate.",
    0x00040000: "Bad control loop timing — phase estimates computed at "
                "wrong time.\nFix: Reboot ODrive.",
    0x00080000: "Phase angle estimate unavailable — encoder not ready or "
                "not calibrated.\n"
                "Fix: Ensure encoder is configured, calibrated, and "
                "is_ready = True before closed loop.",
    0x00100000: "Phase velocity estimate unavailable.\n"
                "Fix: Check encoder is_ready. Ensure encoder config is "
                "saved and pre_calibrated = True.",
    0x00200000: "Torque estimate unavailable — gains or motor constants "
                "not set.\nFix: Run full calibration. Ensure "
                "torque_constant and pole_pairs are set.",
    0x00400000: "Current command unavailable — controller could not produce "
                "a valid current setpoint.\n"
                "Fix: Check controller mode and input_mode.",
    0x00800000: "Current measurement unavailable at the time it was needed.\n"
                "Fix: Reboot ODrive. Check for power supply noise.",
    0x01000000: "Vbus voltage measurement unavailable.\n"
                "Fix: Check power supply. Enable brake resistor to "
                "prevent vbus spikes.",
    0x02000000: "Voltage command unavailable.\n"
                "Fix: Reboot ODrive. Check motor calibration.",
    0x04000000: "Gains unavailable — controller not initialised.\n"
                "Fix: Run calibration. Check motor and encoder config.",
    0x08000000: "Controller is still initializing.\n"
                "Fix: Wait for initialization to complete. Reboot if stuck.",
    0x10000000: "Unbalanced phases detected during calibration.\n"
                "Fix: Check motor wiring — one phase may be disconnected "
                "or shorted.",
}


# ── Encoder errors ────────────────────────────────────────────────────────────

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

ENCODER_ERROR_HINTS = {
    0x00000001: "Encoder gain is unstable — position estimate diverging.\n"
                "Fix: Check motor wiring. Verify CPR setting. "
                "Check encoder power supply.",
    0x00000002: "CPR does not match pole pairs.\n"
                "Fix: Verify CPR = encoder resolution (e.g. 8192 for AS5047). "
                "Verify pole_pairs matches your motor.",
    0x00000004: "No response from encoder — SPI/communication failure.\n"
                "Fix: Check encoder wiring (SPI: SCK, MISO, MOSI, CS). "
                "Verify 3.3V power to encoder.",
    0x00000008: "Encoder mode not supported by this firmware version.",
    0x00000010: "Illegal Hall sensor state (only for Hall-effect encoders).",
    0x00000020: "Index pulse not found yet (incremental encoders with index).\n"
                "Fix: Perform encoder index search first, or use an "
                "absolute encoder.",
    0x00000040: "SPI communication timed out waiting for ABS encoder.\n"
                "Fix: Check SPI wiring. Verify CS GPIO pin. "
                "Check encoder power supply.",
    0x00000080: "SPI communication with ABS encoder failed (bad data/CRC).\n"
                "Fix: Check SPI wiring for noise. Shorten cables. "
                "Verify encoder part number.",
    0x00000100: "ABS encoder not ready (still initialising).\n"
                "Fix: Wait a moment after power-on before commanding motion. "
                "Check encoder power supply ramp time.",
    0x00000200: "Hall sensor not yet calibrated.\n"
                "Fix: Run encoder offset calibration for Hall-effect encoder. "
                "Not relevant for SPI ABS.",
}


# ── Controller errors ─────────────────────────────────────────────────────────

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

CONTROLLER_ERROR_HINTS = {
    0x00000001: "Motor exceeded vel_limit. Reduce velocity setpoint or "
                "increase vel_limit.",
    0x00000002: "Invalid input_mode for the current control_mode.",
    0x00000004: "Control gains are unstable. Reduce vel_gain or "
                "vel_integrator_gain.",
    0x00000008: "Invalid mirror axis configuration.",
    0x00000010: "Invalid load encoder axis configuration.",
    0x00000020: "Position/velocity estimate is invalid. Check encoder "
                "is_ready and calibration.",
    0x00000040: "Invalid circular range configuration.",
    0x00000080: "Spinout detected — electrical and mechanical power disagree. "
                "Check motor wiring direction, encoder direction, or reduce "
                "spinout thresholds.",
}


# ── Decode helper ─────────────────────────────────────────────────────────────

def decode_errors(value, error_table):
    """Decode a bitfield error value into a list of (bit, name) tuples."""
    if value == 0:
        return []
    return [(bit, name) for bit, name in error_table.items() if value & bit]


def decode_errors_str(value, error_table):
    """Decode a bitfield error value into a human-readable string."""
    if value == 0:
        return "none"
    flags = decode_errors(value, error_table)
    if flags:
        return ", ".join(name for _, name in flags)
    return f"0x{value:08X} (unknown)"


def get_hint(bit, hint_table):
    """Get the hint text for a specific error bit, or None."""
    return hint_table.get(bit)


def format_all_errors(axis):
    """Return a multi-line string of all errors on an axis object.

    axis: an ODrive axis object (e.g. odrv0.axis0)
    """
    lines = []
    tables = [
        ("Axis",       "error",                    AXIS_ERRORS),
        ("Motor",      "motor.error",              MOTOR_ERRORS),
        ("Encoder",    "encoder.error",            ENCODER_ERRORS),
        ("Controller", "controller.error",         CONTROLLER_ERRORS),
    ]
    for label, attr_path, table in tables:
        obj = axis
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            val = 0
        else:
            val = obj if isinstance(obj, int) else 0
        lines.append(f"  {label:12s}: {decode_errors_str(val, table)}")
    return "\n".join(lines)
