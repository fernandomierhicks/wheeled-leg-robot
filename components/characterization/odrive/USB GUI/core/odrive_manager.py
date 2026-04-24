"""odrive_manager.py — Thread-safe ODrive device controller.

Central class that owns the USB connection and serialises all access
through a threading.Lock so that the poll timer, tabs, and command
interface never collide.
"""

import logging
import threading

import odrive

from .constants import AXIS_STATES

log = logging.getLogger("odrive_gui")


class ODriveManager:
    """Thread-safe wrapper around a single ODrive USB connection."""

    def __init__(self):
        self._odrv = None
        self._lock = threading.Lock()
        self._expecting_reboot = False

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._odrv is not None

    @property
    def odrv(self):
        """Raw device handle — use safe_get/safe_set instead where possible."""
        return self._odrv

    def connect(self):
        """Blocking connect — call from a worker thread, not the GUI thread.

        Returns the odrv object on success, raises on failure.
        """
        with self._lock:
            if self._odrv is not None:
                return self._odrv
        odrv = odrive.find_any()
        with self._lock:
            self._odrv = odrv
        log.info("Connected: serial=%s fw=%s.%s.%s",
                 hex(odrv.serial_number),
                 odrv.fw_version_major,
                 odrv.fw_version_minor,
                 odrv.fw_version_revision)
        return odrv

    def disconnect(self):
        """Release the device handle."""
        with self._lock:
            self._odrv = None
        log.info("Disconnected")

    def set_device(self, odrv):
        """Replace the device handle (used after reconnect)."""
        with self._lock:
            self._odrv = odrv

    # ── Safe property access ──────────────────────────────────────────────────

    def safe_get(self, attr_path: str, default=None):
        """Read a dotted attribute path like 'axis0.motor.error'.

        Returns default if device is disconnected or the read fails.
        """
        with self._lock:
            odrv = self._odrv
        if odrv is None:
            return default
        try:
            obj = odrv
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return obj
        except Exception:
            return default

    def safe_set(self, attr_path: str, value) -> bool:
        """Write a dotted attribute path like 'axis0.controller.config.vel_limit'.

        Returns True on success, False on failure.
        """
        with self._lock:
            odrv = self._odrv
        if odrv is None:
            return False
        try:
            parts = attr_path.split(".")
            obj = odrv
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            log.info("SET %s = %s", attr_path, value)
            return True
        except Exception as exc:
            log.error("SET %s failed: %s", attr_path, exc)
            return False

    def execute(self, func_path: str, *args, **kwargs):
        """Call a device function like 'save_configuration' or 'clear_errors'.

        Returns the function result, or raises on failure.
        """
        with self._lock:
            odrv = self._odrv
        if odrv is None:
            raise RuntimeError("Not connected")
        parts = func_path.split(".")
        obj = odrv
        for part in parts:
            obj = getattr(obj, part)
        result = obj(*args, **kwargs)
        log.info("EXEC %s(%s %s) -> %s", func_path, args, kwargs, result)
        return result

    # ── Axis helper ───────────────────────────────────────────────────────────

    def get_axis(self, idx: int):
        """Return axis0 or axis1, or None if disconnected."""
        with self._lock:
            odrv = self._odrv
        if odrv is None:
            return None
        return odrv.axis0 if idx == 0 else odrv.axis1

    # ── Status helpers ────────────────────────────────────────────────────────

    def device_info(self) -> dict | None:
        """Return hw/fw version dict, or None if disconnected."""
        with self._lock:
            odrv = self._odrv
        if odrv is None:
            return None
        try:
            return {
                "serial": hex(odrv.serial_number),
                "hw": f"{odrv.hw_version_major}.{odrv.hw_version_minor}.{odrv.hw_version_variant}",
                "fw": f"{odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}",
                "fw_unreleased": getattr(odrv, "fw_version_unreleased", 1),
            }
        except Exception:
            return None

    def vbus_voltage(self) -> float | None:
        """Read bus voltage, or None if disconnected."""
        return self.safe_get("vbus_voltage")

    def axis_state_str(self, idx: int) -> str:
        """Human-readable axis state string."""
        val = self.safe_get(f"axis{idx}.current_state")
        if val is None:
            return "—"
        return AXIS_STATES.get(val, f"Unknown ({val})")

    def axis_errors(self, idx: int) -> dict:
        """Return dict of error register values for the axis."""
        prefix = f"axis{idx}"
        return {
            "axis": self.safe_get(f"{prefix}.error", 0),
            "motor": self.safe_get(f"{prefix}.motor.error", 0),
            "encoder": self.safe_get(f"{prefix}.encoder.error", 0),
            "controller": self.safe_get(f"{prefix}.controller.error", 0),
        }

    # ── Reboot support ────────────────────────────────────────────────────────

    @property
    def expecting_reboot(self) -> bool:
        return self._expecting_reboot

    @expecting_reboot.setter
    def expecting_reboot(self, val: bool):
        self._expecting_reboot = val
