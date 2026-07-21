import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


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

    def __init__(self):
        super().__init__()
        self._latest_live: dict | None = None
        self._latest_lock = threading.Lock()
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(50)  # smooth 20 Hz UI without event-queue backlog
        self._live_timer.timeout.connect(self._flush_live)
        self._live_timer.start()

    def publish(self, info: dict):
        """Publish decoded data, coalescing only high-rate telemetry.

        Parameter reports, diagnostics, logs, and command evidence remain
        immediate; only ptype 0x01 is latest-sample wins.
        """
        if info.get("ptype") != 0x01:
            self.packet.emit(info)
            return
        with self._latest_lock:
            self._latest_live = info

    def _flush_live(self):
        with self._latest_lock:
            info = self._latest_live
            self._latest_live = None
        if info is not None and not self.playback_active:
            self.packet.emit(info)
