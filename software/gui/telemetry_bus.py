from PyQt6.QtCore import QObject, pyqtSignal


class TelemetryBus(QObject):
    """App-wide singleton — any decoded telemetry packet is emitted here."""
    packet = pyqtSignal(dict)

    _instance: "TelemetryBus | None" = None

    @classmethod
    def instance(cls) -> "TelemetryBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
