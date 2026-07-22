"""host_logger.py — GUI-side capture of every TelemetryBus packet to a
timestamped .jsonl file, independent of SD logging (log_transfer.py).

Where SD logging only records the fixed TELEM struct on the robot, this
captures whatever the GUI actually receives — TELEM, robot LOG messages,
CALIB events, WIFI_DIAG — each packet dict as one JSON line with a host-side
timestamp, so nothing needs a matching fixed-size record layout. Manual
start/stop only (see Logs tab); does not run automatically.
"""

import json
import time
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .log_transfer import LOG_DIR
from .telemetry_bus import TelemetryBus


class HostLogger(QObject):
    """App-wide singleton. Subscribes to TelemetryBus.packet directly (not
    through source_manager filtering) so it sees exactly what every other
    live tab sees, including during multi-transport arbitration."""

    logging_state_changed = pyqtSignal(bool)
    record_count_changed  = pyqtSignal(int)

    _instance: "HostLogger | None" = None

    @classmethod
    def instance(cls) -> "HostLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file  = None
        self._path: Path | None = None
        self._count = 0
        TelemetryBus.instance().packet.connect(self._on_packet)

    def is_logging(self) -> bool:
        return self._file is not None

    def path(self) -> "Path | None":
        return self._path

    def record_count(self) -> int:
        return self._count

    def start(self):
        if self._file is not None:
            return
        self._path = LOG_DIR / f"HOSTLOG_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")
        self._count = 0
        self.logging_state_changed.emit(True)
        self.record_count_changed.emit(0)

    def stop(self):
        if self._file is None:
            return
        self._file.close()
        self._file = None
        self.logging_state_changed.emit(False)

    def _on_packet(self, info: dict):
        # Skip .wlog playback (Logs tab replays onto the same bus) — that's
        # historical data being re-viewed, not something newly received.
        if self._file is None or TelemetryBus.instance().playback_active:
            return
        record = {"host_ts": time.time(), **info}
        self._file.write(json.dumps(record, default=str) + "\n")
        # Flushed every record, not just buffered: the whole point is to
        # survive a GUI/firmware crash mid-session, so data must hit disk
        # as it arrives rather than waiting on Python's buffer to fill.
        self._file.flush()
        self._count += 1
        self.record_count_changed.emit(self._count)
