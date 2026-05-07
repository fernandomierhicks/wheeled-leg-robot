import threading
import serial
import serial.tools.list_ports
from PyQt6.QtCore import QObject, pyqtSignal

# VID/PID fingerprints — add variants as needed
SIGNATURES: dict[str, list[tuple[int, int]]] = {
    "teensy": [(0x16C0, 0x0483)],
    "esp32":  [(0x2886, 0x0056), (0x303A, 0x1001), (0x10C4, 0xEA60)],
}


class SerialPortManager(QObject):
    """Singleton. Owns every open serial port so Flash and Monitor never collide."""

    port_released = pyqtSignal(str)   # emitted with device name after release
    port_opened   = pyqtSignal(str)   # emitted with device name after open

    _instance: "SerialPortManager | None" = None

    @classmethod
    def instance(cls) -> "SerialPortManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._open: dict[str, serial.Serial] = {}

    # ── Discovery ─────────────────────────────────────────────────────────────

    def scan(self) -> dict[str, str]:
        """Return {device: port_path} for auto-detected devices."""
        found: dict[str, str] = {}
        for info in serial.tools.list_ports.comports():
            for device, sigs in SIGNATURES.items():
                if device not in found:
                    for vid, pid in sigs:
                        if info.vid == vid and info.pid == pid:
                            found[device] = info.device
        return found

    def all_ports(self) -> list[str]:
        return sorted(p.device for p in serial.tools.list_ports.comports())

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def acquire(self, device: str, port: str, baud: int) -> "serial.Serial | None":
        with self._lock:
            if device in self._open:
                return self._open[device]
            try:
                s = serial.Serial(port, baud, timeout=0)
                self._open[device] = s
            except Exception:
                return None
        self.port_opened.emit(device)
        return self._open[device]

    def release(self, device: str):
        with self._lock:
            s = self._open.pop(device, None)
        if s and s.is_open:
            try:
                s.close()
            except Exception:
                pass
        self.port_released.emit(device)

    def release_all(self):
        """Call from Flash before running pio — takes precedence over monitor."""
        with self._lock:
            items = list(self._open.items())
            self._open.clear()
        for device, s in items:
            if s.is_open:
                try:
                    s.close()
                except Exception:
                    pass
            self.port_released.emit(device)

    def is_open(self, device: str) -> bool:
        with self._lock:
            return device in self._open
