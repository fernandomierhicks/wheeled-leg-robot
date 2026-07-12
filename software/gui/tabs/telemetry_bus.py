from PyQt6.QtCore import QObject, pyqtSignal


class TelemetryBus(QObject):
    """App-wide singleton — any decoded telemetry packet is emitted here."""
    packet = pyqtSignal(dict)

    # True while the Logs tab is replaying a .wlog file — live sources stop
    # emitting onto `packet` (see flash_monitor.py PacketDecoder._parse())
    # so playback isn't fought over by a connected robot.
    playback_active: bool = False
    playback_state_changed = pyqtSignal(bool)

    _instance: "TelemetryBus | None" = None

    @classmethod
    def instance(cls) -> "TelemetryBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_playback_active(self, active: bool):
        if self.playback_active == active:
            return
        self.playback_active = active
        self.playback_state_changed.emit(active)
