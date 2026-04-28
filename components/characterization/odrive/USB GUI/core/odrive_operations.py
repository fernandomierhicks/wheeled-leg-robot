"""odrive_operations.py — High-level ODrive workflows.

Every ODrive operation (flash, calibrate, auto-tune, verify) is defined
exactly once here.  The GUI tabs and Claude command interface both call
these shared functions.
"""

import logging
from .constants import AXIS_STATES, MOTOR_TYPE_VALUES, ENCODER_TYPE_VALUES

log = logging.getLogger("odrive_gui")

# Startup flags that must always be forced False for safety
STARTUP_FLAGS = [
    "startup_motor_calibration",
    "startup_encoder_offset_calibration",
    "startup_encoder_index_search",
    "startup_closed_loop_control",
    "startup_homing",
]


# ── Startup flag safety ──────────────────────────────────────────────────────

def lock_startup_flags(axis):
    """Force all startup motion flags False."""
    for flag in STARTUP_FLAGS:
        try:
            setattr(axis.config, flag, False)
        except Exception:
            pass


def check_startup_flags(axis) -> bool:
    """Returns True if all startup flags are safely False."""
    for flag in STARTUP_FLAGS:
        if getattr(axis.config, flag, False):
            return False
    return True


# ── Read config ──────────────────────────────────────────────────────────────

def read_config(mgr, axis_idx: int) -> dict:
    """Read current motor/encoder/controller config from the ODrive.

    Returns a dict of parameter values, or raises on failure.
    """
    axis = mgr.get_axis(axis_idx)
    if axis is None:
        raise RuntimeError("Not connected")

    mc = axis.motor.config
    ec = axis.encoder.config
    cc = axis.controller.config
    odrv = mgr.odrv

    return {
        "motor_type":                   getattr(mc, "motor_type", 0),
        "pole_pairs":                   getattr(mc, "pole_pairs", 7),
        "current_lim":                  float(getattr(mc, "current_lim", 10)),
        "calibration_current":          float(getattr(mc, "calibration_current", 10)),
        "resistance_calib_max_voltage": float(getattr(mc, "resistance_calib_max_voltage", 2)),
        "torque_constant":              float(getattr(mc, "torque_constant", 0.04)),
        "phase_resistance":             float(getattr(mc, "phase_resistance", 0)),
        "phase_inductance":             float(getattr(mc, "phase_inductance", 0)),
        "vel_limit":                    float(getattr(cc, "vel_limit", 2)),
        "vel_gain":                     float(getattr(cc, "vel_gain", 0.1667)),
        "vel_integrator_gain":          float(getattr(cc, "vel_integrator_gain", 0.3333)),
        "encoder_mode":                 getattr(ec, "mode", 0),
        "abs_spi_cs_gpio_pin":          getattr(ec, "abs_spi_cs_gpio_pin", 3),
        "cpr":                          getattr(ec, "cpr", 8192),
        "encoder_pre_calibrated":       getattr(ec, "pre_calibrated", False),
        "motor_pre_calibrated":         getattr(mc, "pre_calibrated", False),
        "enable_brake_resistor":        getattr(odrv.config, "enable_brake_resistor", False),
        "brake_resistance":            float(getattr(odrv.config, "brake_resistance", 2.0)),
        "dc_max_negative_current":     float(getattr(odrv.config, "dc_max_negative_current", -0.01)),
        "watchdog_timeout":             float(getattr(axis.config, "watchdog_timeout", 0)),
        "enable_watchdog":              getattr(axis.config, "enable_watchdog", False),
        "dc_bus_undervoltage_trip_level": float(getattr(odrv.config, "dc_bus_undervoltage_trip_level", 8.0)),
        "dc_bus_overvoltage_trip_level":  float(getattr(odrv.config, "dc_bus_overvoltage_trip_level", 59.0)),
    }


# ── Flash config ─────────────────────────────────────────────────────────────

def flash_config(mgr, axis_idx: int, params: dict) -> dict:
    """Write config params to ODrive, lock safety flags, save_configuration.

    *params* is a dict with keys matching read_config output.
    Returns the snapshot taken before save (for later verify).
    Note: save_configuration() triggers a reboot.
    """
    axis = mgr.get_axis(axis_idx)
    odrv = mgr.odrv
    if axis is None or odrv is None:
        raise RuntimeError("Not connected")

    mc = axis.motor.config
    ec = axis.encoder.config
    cc = axis.controller.config

    # Motor
    mc.motor_type = params.get("motor_type", 0)
    mc.pole_pairs = params.get("pole_pairs", 7)
    mc.current_lim = params.get("current_lim", 10.0)
    mc.calibration_current = params.get("calibration_current", 10.0)
    try:
        mc.resistance_calib_max_voltage = params.get("resistance_calib_max_voltage", 2.0)
    except Exception:
        pass
    try:
        mc.torque_constant = params.get("torque_constant", 0.04)
    except Exception:
        pass

    # Controller
    cc.vel_limit = params.get("vel_limit", 2.0)

    # Encoder
    enc_mode = params.get("encoder_mode", 0)
    ec.mode = enc_mode
    ec.cpr = params.get("cpr", 8192)
    if enc_mode in (256, 257):  # SPI_ABS_CUI / SPI_ABS_AMS
        ec.abs_spi_cs_gpio_pin = params.get("abs_spi_cs_gpio_pin", 3)

    # Watchdog
    wdt = params.get("watchdog_timeout", 0.0)
    axis.config.watchdog_timeout = wdt
    axis.config.enable_watchdog = (wdt > 0)

    # Bus voltage protection
    odrv.config.dc_bus_undervoltage_trip_level = params.get("dc_bus_undervoltage_trip_level", 8.0)
    odrv.config.dc_bus_overvoltage_trip_level = params.get("dc_bus_overvoltage_trip_level", 59.0)

    # Safety: lock all startup flags
    lock_startup_flags(axis)
    cc.enable_overspeed_error = False

    # Brake resistor
    odrv.config.enable_brake_resistor = params.get("enable_brake_resistor", True)
    odrv.config.brake_resistance = params.get("brake_resistance", 2.0)
    odrv.config.dc_max_negative_current = params.get("dc_max_negative_current", -0.01)

    # Snapshot before save
    snapshot = snapshot_params(mgr, axis_idx)

    # Save triggers reboot
    try:
        odrv.save_configuration()
    except Exception:
        pass  # expected — ODrive reboots immediately

    log.info("flash_config: axis%d flashed + saved, rebooting", axis_idx)
    return snapshot


# ── Snapshot for verification ────────────────────────────────────────────────

def snapshot_params(mgr, axis_idx: int) -> dict:
    """Snapshot key params for post-reboot verification."""
    axis = mgr.get_axis(axis_idx)
    odrv = mgr.odrv
    if axis is None or odrv is None:
        return {}

    mc = axis.motor.config
    ec = axis.encoder.config
    cc = axis.controller.config

    snap = {
        "motor_type":                   getattr(mc, "motor_type", None),
        "pole_pairs":                   getattr(mc, "pole_pairs", None),
        "current_lim":                  getattr(mc, "current_lim", None),
        "calibration_current":          getattr(mc, "calibration_current", None),
        "resistance_calib_max_voltage": getattr(mc, "resistance_calib_max_voltage", None),
        "torque_constant":              getattr(mc, "torque_constant", None),
        "vel_limit":                    getattr(cc, "vel_limit", None),
        "enable_overspeed_error":       getattr(cc, "enable_overspeed_error", None),
        "enable_brake_resistor":        getattr(odrv.config, "enable_brake_resistor", None),
        "brake_resistance":            getattr(odrv.config, "brake_resistance", None),
        "dc_max_negative_current":     getattr(odrv.config, "dc_max_negative_current", None),
        "watchdog_timeout":             getattr(axis.config, "watchdog_timeout", None),
        "enable_watchdog":              getattr(axis.config, "enable_watchdog", None),
        "dc_bus_undervoltage_trip_level": getattr(odrv.config, "dc_bus_undervoltage_trip_level", None),
        "dc_bus_overvoltage_trip_level":  getattr(odrv.config, "dc_bus_overvoltage_trip_level", None),
        "encoder_mode":                 getattr(ec, "mode", None),
        "abs_spi_cs_gpio_pin":          getattr(ec, "abs_spi_cs_gpio_pin", None),
        "cpr":                          getattr(ec, "cpr", None),
    }
    for flag in STARTUP_FLAGS:
        snap[flag] = getattr(axis.config, flag, None)
    return snap


def verify_params(expected: dict, actual: dict) -> list[dict]:
    """Compare expected vs actual params. Returns list of {key, expected, actual, ok}."""
    results = []
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if isinstance(exp_val, float) and isinstance(act_val, float):
            ok = abs(exp_val - act_val) < 1e-5
        else:
            ok = (exp_val == act_val)
        results.append({"key": key, "expected": exp_val, "actual": act_val, "ok": ok})
    return results


# ── Post-calibration auto-tuning ─────────────────────────────────────────────

def auto_tune_after_cal(mgr, axis_idx: int) -> dict:
    """Set pre_calibrated flags, compute vel gains from phase_resistance,
    arm brake resistor, loosen spinout thresholds, save config.

    Returns dict of tuned values. save_configuration() triggers reboot.
    """
    axis = mgr.get_axis(axis_idx)
    odrv = mgr.odrv
    if axis is None or odrv is None:
        raise RuntimeError("Not connected")

    mc = axis.motor.config
    cc = axis.controller.config

    # Set pre-calibrated flags
    mc.pre_calibrated = True
    axis.encoder.config.pre_calibrated = True

    # Compute vel_gain from measured phase_resistance
    R = mc.phase_resistance
    kT = mc.torque_constant
    if R and R > 0.001:
        vel_gain = max(0.01, min(2.0, 0.5 * kT / R))
    else:
        vel_gain = cc.vel_gain
    vel_int = 0.5 * vel_gain

    cc.vel_gain = vel_gain
    cc.vel_integrator_gain = vel_int

    # Raise vel_limit if too low
    cc.vel_limit = max(cc.vel_limit, 20.0)

    # Loosen spinout thresholds (new in 0.5.6)
    try:
        cc.spinout_electrical_power_threshold = 120.0
        cc.spinout_mechanical_power_threshold = -120.0
    except Exception:
        pass

    tuned = {
        "vel_gain": vel_gain,
        "vel_integrator_gain": vel_int,
        "vel_limit": cc.vel_limit,
        "phase_resistance": R,
        "phase_inductance": mc.phase_inductance,
        "torque_constant": kT,
    }
    log.info("auto_tune: axis%d vel_gain=%.4f vel_int=%.4f R=%.4f kT=%.4f",
             axis_idx, vel_gain, vel_int, R, kT)

    # Save triggers reboot
    try:
        odrv.save_configuration()
    except Exception:
        pass

    return tuned


# ── Anticogging prerequisites ───────────────────────────────────────────────

def anticogging_prerequisites(mgr, axis_idx: int) -> dict:
    """Check prerequisites for anticogging calibration.

    Returns dict with 'ok' bool and 'checks' list of {name, ok, msg}.
    """
    axis = mgr.get_axis(axis_idx)
    if axis is None:
        return {"ok": False, "checks": [
            {"name": "Connection", "ok": False, "msg": "Not connected"}]}

    checks = []

    motor_cal = getattr(axis.motor, "is_calibrated", False)
    checks.append({"name": "Motor calibrated", "ok": motor_cal,
                    "msg": "YES" if motor_cal else "Run calibration from Setup tab"})

    enc_ready = getattr(axis.encoder, "is_ready", False)
    checks.append({"name": "Encoder ready", "ok": enc_ready,
                    "msg": "YES" if enc_ready else "Run calibration from Setup tab"})

    errs = mgr.axis_errors(axis_idx)
    any_err = any(v != 0 for v in errs.values())
    checks.append({"name": "No errors", "ok": not any_err,
                    "msg": "Clear" if not any_err else "Clear errors first"})

    vg = float(getattr(axis.controller.config, "vel_gain", 0))
    checks.append({"name": "vel_gain > 0", "ok": vg > 0.001,
                    "msg": f"{vg:.4f}" if vg > 0.001 else "Configure gains first"})

    state = getattr(axis, "current_state", 0)
    checks.append({"name": "Axis idle", "ok": state == 1,
                    "msg": "IDLE" if state == 1 else AXIS_STATES.get(state, f"State {state}")})

    vbus = mgr.vbus_voltage()
    v_ok = vbus is not None and vbus > 10.0
    checks.append({"name": "Vbus > 10V", "ok": v_ok,
                    "msg": f"{vbus:.1f}V" if vbus else "?"})

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
