"""log_transfer.py — SD-log directory listing + reliable file download.

Subscribes to LogPacketBus, which every PacketDecoder instance (USB serial
for "teensy"/"esp32", and WiFi) feeds identically — so LIST/GET/DELETE work
the same whether the active link is USB or WiFi, with no transport-specific
code here at all.

CRC32 note: the Teensy only accumulates crc32 over the bytes it streams in
one GET call (see sd_logger.cpp sd_logger_begin_get docstring). To keep
verification meaningful, a stalled/incomplete/corrupt transfer is retried
as a full restart from chunk 0 rather than resuming from the gap, so the
reported crc32 always covers the exact bytes assembled here.
"""

import time
import zlib
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .comm_commands import send_log_get, send_log_list, send_log_start, send_log_stop
from .log_bus import LogPacketBus

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_INFO_ENTRY      = 0x01
LOG_INFO_LIST_END   = 0x02
LOG_INFO_XFER_BEGIN = 0x03
LOG_INFO_XFER_END   = 0x04
LOG_INFO_STATUS     = 0x05

CHUNK_TIMEOUT_S = 2.0   # re-request if no chunk arrives for this long
MAX_RETRIES     = 3


class LogTransferManager(QObject):
    """App-wide singleton — GUI-facing SD log directory + download manager."""

    directory_updated = pyqtSignal(list)           # list[{"index": int, "size": int}]
    transfer_progress  = pyqtSignal(int, int, int)  # file_index, got_chunks, total_chunks
    transfer_complete  = pyqtSignal(int, str, bool) # file_index, local_path ("" on failure), crc_ok
    status_ack         = pyqtSignal(int, int)       # file_index, status (0=ok) — START/STOP/DELETE
    logging_state_changed = pyqtSignal(bool)        # True while a log is actively recording

    _instance: "LogTransferManager | None" = None

    @classmethod
    def instance(cls) -> "LogTransferManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._directory: list[dict] = []
        self._listing: list[dict] = []

        self._xfer_active = False
        self._xfer_index  = -1
        self._xfer_total  = 0
        self._xfer_crc    = 0
        self._chunks: dict[int, bytes] = {}
        self._last_chunk_ms = 0.0
        self._retries = 0

        self._logging       = False
        self._logging_token = 0

        LogPacketBus.instance().packet.connect(self._on_packet)

        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._check_timeout)
        self._watchdog.start()

    # ── Logging start/stop (single source of truth for all GUI controls) ──────
    # The firmware's LOG_INFO STATUS ack doesn't distinguish START/STOP/DELETE,
    # so button state here is driven by the click + a local mirror of the
    # firmware's own auto-stop deadline, not by waiting for that ack.

    def is_logging(self) -> bool:
        return self._logging

    def start_logging(self, duration_ms: int = 0):
        if self._logging:
            return
        self._logging = True
        self._logging_token += 1
        token = self._logging_token
        send_log_start(duration_ms)
        self.logging_state_changed.emit(True)
        if duration_ms > 0:
            QTimer.singleShot(duration_ms + 500, lambda: self._auto_stop_fire(token))

    def stop_logging(self):
        if not self._logging:
            return
        self._logging_token += 1  # invalidate any pending auto-stop callback
        send_log_stop()
        self._logging = False
        self.logging_state_changed.emit(False)

    def _auto_stop_fire(self, token: int):
        if token != self._logging_token or not self._logging:
            return
        self._logging = False
        self.logging_state_changed.emit(False)

    # ── Public API ───────────────────────────────────────────────────────────

    def refresh(self):
        """Request a fresh directory listing. Result arrives via directory_updated."""
        self._listing = []
        send_log_list()

    def download(self, file_index: int):
        """Start (or restart) downloading one .wlog file. Result arrives via
        transfer_progress (repeatedly) then transfer_complete (once)."""
        self._xfer_active = True
        self._xfer_index  = file_index
        self._xfer_total  = 0
        self._xfer_crc    = 0
        self._chunks      = {}
        self._retries     = 0
        self._last_chunk_ms = time.monotonic()
        send_log_get(file_index, 0)

    def is_transferring(self) -> bool:
        return self._xfer_active

    # ── LogPacketBus handler ────────────────────────────────────────────────

    def _on_packet(self, info: dict):
        ptype = info.get("ptype")
        if ptype == 0x12:
            self._on_log_info(info)
        elif ptype == 0x13:
            self._on_log_data(info)

    def _on_log_info(self, info: dict):
        sub = info.get("log_info_type")
        idx = info.get("log_file_index")
        if sub == LOG_INFO_ENTRY:
            self._listing.append({"index": idx, "size": info.get("log_file_size", 0)})
        elif sub == LOG_INFO_LIST_END:
            self._directory = list(self._listing)
            self.directory_updated.emit(self._directory)
        elif sub == LOG_INFO_XFER_BEGIN:
            if not self._xfer_active or idx != self._xfer_index:
                return
            self._xfer_total = info.get("log_total_chunks", 0)
            self._last_chunk_ms = time.monotonic()
        elif sub == LOG_INFO_XFER_END:
            if not self._xfer_active or idx != self._xfer_index:
                return
            self._xfer_crc = info.get("log_crc32", 0)
            self._finish_transfer()
        elif sub == LOG_INFO_STATUS:
            self.status_ack.emit(idx, info.get("log_status", 1))

    def _on_log_data(self, info: dict):
        idx = info.get("log_file_index")
        if not self._xfer_active or idx != self._xfer_index:
            return
        self._chunks[info.get("log_chunk_index")] = info.get("log_data", b"")
        self._last_chunk_ms = time.monotonic()
        if self._xfer_total:
            self.transfer_progress.emit(self._xfer_index, len(self._chunks), self._xfer_total)

    # ── Retry / assembly ─────────────────────────────────────────────────────

    def _first_missing_chunk(self):
        for i in range(self._xfer_total):
            if i not in self._chunks:
                return i
        return None

    def _restart_transfer(self) -> bool:
        """Restart the current transfer from chunk 0. Returns False (and gives
        up, emitting a failed transfer_complete) once MAX_RETRIES is exceeded."""
        self._retries += 1
        if self._retries > MAX_RETRIES:
            self._xfer_active = False
            self.transfer_complete.emit(self._xfer_index, "", False)
            return False
        self._chunks = {}
        self._xfer_total = 0
        self._last_chunk_ms = time.monotonic()
        send_log_get(self._xfer_index, 0)
        return True

    def _check_timeout(self):
        if not self._xfer_active or self._xfer_total == 0:
            return
        if time.monotonic() - self._last_chunk_ms < CHUNK_TIMEOUT_S:
            return
        self._restart_transfer()

    def _finish_transfer(self):
        if self._first_missing_chunk() is not None:
            self._restart_transfer()
            return

        data = b"".join(self._chunks[i] for i in range(self._xfer_total))
        crc_ok = (zlib.crc32(data) & 0xFFFFFFFF) == self._xfer_crc
        if not crc_ok:
            self._restart_transfer()
            return

        out_path = LOG_DIR / f"LOG{self._xfer_index:04d}.WLOG"
        out_path.write_bytes(data)
        self._xfer_active = False
        self.transfer_complete.emit(self._xfer_index, str(out_path), True)
