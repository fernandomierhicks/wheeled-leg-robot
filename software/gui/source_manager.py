from PyQt6.QtCore import QObject, pyqtSignal
from port_manager import SerialPortManager

# Priority order — first entry wins when multiple sources are available.
# "wifi" is reserved for future UDP/WiFi transport.
PRIORITY: list[str] = ["esp32", "teensy", "wifi"]

TRANSPORT_LABEL: dict[str, str] = {
    "esp32":  "USB Serial",
    "teensy": "USB Serial",
    "wifi":   "WiFi",
}


class SourceManager(QObject):
    """Singleton. Tracks which devices are connected and picks the active telemetry source."""

    source_changed = pyqtSignal(str)   # device name, or "" for none

    _instance: "SourceManager | None" = None

    @classmethod
    def instance(cls) -> "SourceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._connected: set[str] = set()
        self._active: str = ""
        pm = SerialPortManager.instance()
        pm.port_opened.connect(self._on_opened)
        pm.port_released.connect(self._on_released)

    def _on_opened(self, device: str):
        self._connected.add(device)
        self._update()

    def _on_released(self, device: str):
        self._connected.discard(device)
        self._update()

    def _update(self):
        new_active = next((d for d in PRIORITY if d in self._connected), "")
        if new_active != self._active:
            self._active = new_active
            self.source_changed.emit(new_active)

    @property
    def active(self) -> str:
        return self._active

    def is_active(self, device: str) -> bool:
        return self._active == device
