"""host_logger.py — GUI-side capture of every TelemetryBus packet to a
timestamped .jsonl file, automatically synchronized by log_transfer.py when
Teensy SD logging changes state.

Where SD logging only records the fixed TELEM struct on the robot, this
captures whatever the GUI actually receives — TELEM, robot LOG messages,
CALIB events, WIFI_DIAG — each packet dict as one JSON line with a host-side
timestamp, so nothing needs a matching fixed-size record layout. It can still
be controlled manually, and LogTransferManager also starts/stops it whenever
Teensy SD logging starts/stops from any observed trigger.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .comm_commands import send_param_get_all
from .log_paths import new_run_dir
from .telem_format import TELEM_VERSION
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
        TelemetryBus.instance().live_packet.connect(self._on_packet)

    def is_logging(self) -> bool:
        return self._file is not None

    def path(self) -> "Path | None":
        return self._path

    def record_count(self) -> int:
        return self._count

    def start(self):
        if self._file is not None:
            return
        run_dir = new_run_dir("HOST")
        self._path = run_dir / "host.jsonl"
        started_utc = datetime.now(timezone.utc).isoformat()
        manifest = {
            "capture_schema": 1,
            "source_kind": "host",
            "started_utc": started_utc,
            "telem_version": TELEM_VERSION,
            "files": {"host": self._path.name},
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        self._file = open(self._path, "x", encoding="utf-8")
        self._file.write(json.dumps({"record_type": "capture_header", **manifest}) + "\n")
        self._file.flush()
        self._count = 0
        self.logging_state_changed.emit(True)
        self.record_count_changed.emit(0)
        # PARAM_REPORT responses travel through live_packet too, giving the
        # analyzer a complete starting snapshot followed by later set echoes.
        send_param_get_all()

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
        record = {
            "host_ts": time.time(),
            "host_utc_ns": time.time_ns(),
            "host_monotonic_ns": time.monotonic_ns(),
            **info,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        # Flush Python's buffer every record so a process crash loses at most
        # a partial final JSON line (the decoder explicitly tolerates that).
        self._file.flush()
        self._count += 1
        self.record_count_changed.emit(self._count)
