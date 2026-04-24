"""odrive_presets.py -- Save/load full ODrive config as JSON, factory presets.

Walks all .config. subtrees on the device, reads every scalar value,
and serialises to a JSON file.  Loading writes each value back and
optionally saves + reboots.

Factory preset for Maytech MTO5065-70-HA-C + AS5048A (SPI_ABS_CUI mode 256).
"""

import json
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("odrive_gui")


# ---- Config tree walk -------------------------------------------------------

# Prefixes to walk for a full config snapshot.  We only care about
# .config. subtrees because those are the user-settable values that
# persist across reboots (after save_configuration).
_CONFIG_ROOTS = [
    ("config",                             "odrv"),
    ("axis0.config",                       "axis0"),
    ("axis0.motor.config",                 "axis0.motor"),
    ("axis0.encoder.config",               "axis0.encoder"),
    ("axis0.controller.config",            "axis0.controller"),
    ("axis0.trap_traj.config",             "axis0.trap_traj"),
    ("axis1.config",                       "axis1"),
    ("axis1.motor.config",                 "axis1.motor"),
    ("axis1.encoder.config",               "axis1.encoder"),
    ("axis1.controller.config",            "axis1.controller"),
    ("axis1.trap_traj.config",             "axis1.trap_traj"),
]


def _read_config_obj(obj) -> dict:
    """Read all scalar attributes from a .config object into a dict."""
    result = {}
    try:
        attrs = sorted(a for a in dir(obj) if not a.startswith("_"))
    except Exception:
        return result

    for attr in attrs:
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(val, (int, float, bool)):
            result[attr] = val
        # skip compound objects, callables, strings, bytes
    return result


def read_full_config(odrv) -> dict:
    """Walk all .config subtrees and return a nested dict.

    Top-level keys are dotted paths like "axis0.motor.config".
    Each maps to a flat dict of {param: value}.
    Also stores metadata (serial, fw version, timestamp).
    """
    snapshot = {"_meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "serial": hex(odrv.serial_number),
        "fw": f"{odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}",
        "hw": f"{odrv.hw_version_major}.{odrv.hw_version_minor}.{odrv.hw_version_variant}",
    }}

    for attr_path, _label in _CONFIG_ROOTS:
        try:
            obj = odrv
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            snapshot[attr_path] = _read_config_obj(obj)
        except Exception as e:
            log.warning("presets: failed to read %s: %s", attr_path, e)

    return snapshot


def write_full_config(odrv, snapshot: dict) -> list[dict]:
    """Write a snapshot dict back to the device.  Does NOT save or reboot.

    Returns a list of {path, param, value, ok, error} dicts.
    """
    results = []
    for attr_path, _label in _CONFIG_ROOTS:
        params = snapshot.get(attr_path, {})
        if not params:
            continue
        try:
            obj = odrv
            for part in attr_path.split("."):
                obj = getattr(obj, part)
        except Exception as e:
            for param in params:
                results.append({"path": attr_path, "param": param,
                                "value": params[param], "ok": False,
                                "error": str(e)})
            continue

        for param, value in params.items():
            try:
                setattr(obj, param, value)
                results.append({"path": attr_path, "param": param,
                                "value": value, "ok": True, "error": ""})
            except Exception as e:
                results.append({"path": attr_path, "param": param,
                                "value": value, "ok": False,
                                "error": str(e)})
    return results


# ---- JSON file I/O ----------------------------------------------------------

def save_config_to_file(odrv, filepath: str) -> dict:
    """Read full config from device and write to JSON file.

    Returns the snapshot dict.
    """
    snapshot = read_full_config(odrv)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    log.info("presets: saved config to %s (%d sections)",
             filepath, len(snapshot) - 1)
    return snapshot


def load_config_from_file(filepath: str) -> dict:
    """Read a config snapshot from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- Factory Presets --------------------------------------------------------

# Maytech MTO5065-70-HA-C (KV70, 7 pole pairs) + AS5048A via SPI
# (mode 256 = SPI_ABS_CUI, CPR 16384, CS pin 3)
# Values sourced from COMPONENTS.md and setup_encoder_direct.py
FACTORY_PRESET_MAYTECH_AS5048 = {
    "_meta": {
        "preset_name": "Maytech MTO5065-70-HA-C + AS5048A",
        "description": "Factory preset for Maytech 5065 KV70 wheel motor with "
                       "AS5048A absolute encoder over SPI (mode 256).",
    },
    "config": {
        "enable_brake_resistor": True,
        "dc_max_negative_current": -5.0,
    },
    "axis0.config": {
        "startup_motor_calibration": False,
        "startup_encoder_offset_calibration": False,
        "startup_encoder_index_search": False,
        "startup_closed_loop_control": False,
        "startup_homing": False,
        "watchdog_timeout": 0.0,
    },
    "axis0.motor.config": {
        "motor_type": 0,                        # HIGH_CURRENT
        "pole_pairs": 7,
        "current_lim": 10.0,
        "calibration_current": 10.0,
        "resistance_calib_max_voltage": 4.0,
        "torque_constant": 0.1364,               # from COMPONENTS.md Kt
        "current_control_bandwidth": 1000.0,
    },
    "axis0.encoder.config": {
        "mode": 256,                             # SPI_ABS_CUI (AS5048A)
        "cpr": 16384,                            # 14-bit
        "abs_spi_cs_gpio_pin": 3,
        "bandwidth": 1000.0,
    },
    "axis0.controller.config": {
        "control_mode": 2,                       # VELOCITY
        "input_mode": 1,                         # PASSTHROUGH
        "vel_limit": 20.0,
        "vel_gain": 0.16,
        "vel_integrator_gain": 0.32,
        "pos_gain": 20.0,
        "enable_overspeed_error": False,
    },
}


def get_factory_presets() -> dict[str, dict]:
    """Return all available factory presets as {name: snapshot}."""
    return {
        "Maytech MTO5065 + AS5048A": FACTORY_PRESET_MAYTECH_AS5048,
    }
