"""log_transfer.py — SD-log directory listing + reliable file download.

Subscribes to LogPacketBus, which every PacketDecoder instance (USB serial
for "teensy"/"esp32", and WiFi) feeds identically — so LIST/GET/DELETE work
the same whether the active link is USB or WiFi, with no transport-specific
code here at all.

CRC32 note: the Teensy accumulates crc32 over the bytes streamed by one GET.
Repairs therefore remember the requested start chunk and validate that exact
suffix. The CRC from the initial start_chunk=0 request is retained as the
whole-file CRC, so the final assembled file still gets end-to-end validation.
"""

import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .comm_commands import (
    LOG_FILE_KIND_PARAMS, LOG_FILE_KIND_WLOG, send_log_get, send_log_list,
    send_log_start, send_log_stop,
)
from .log_bus import LogPacketBus
from .log_paths import RUNS_DIR

LOG_DIR = RUNS_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_INFO_ENTRY      = 0x01
LOG_INFO_LIST_END   = 0x02
LOG_INFO_XFER_BEGIN = 0x03
LOG_INFO_XFER_END   = 0x04
LOG_INFO_STATUS     = 0x05
LOG_INFO_STARTED    = 0x06
LOG_INFO_STOPPED    = 0x07

CHUNK_TIMEOUT_S = 3.0   # re-request if no metadata/chunk arrives for this long
MAX_RETRIES     = 12    # inexpensive suffix repairs, not whole-file restarts


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
        self._xfer_kind   = LOG_FILE_KIND_WLOG
        self._xfer_total  = 0
        self._xfer_crc    = 0
        self._full_crc: int | None = None
        self._request_start = 0
        self._awaiting_begin = False
        self._chunks: dict[int, bytes] = {}
        # Timestamp of the last *new* valid chunk, rather than merely the
        # last packet.  The Teensy retransmits an unacknowledged chunk every
        # 100 ms; treating those duplicates as activity used to leave the GUI
        # waiting forever if the ESP32->GUI hop was stuck after that chunk.
        self._last_chunk_ms = 0.0
        self._retries = 0
        self._failure_reason = "transfer failed"
        self._pending_params_index: int | None = None  # see download_with_params()
        self._output_dir: Path | None = None

        self._logging       = False
        self._logging_token = 0

        LogPacketBus.instance().packet.connect(self._on_packet)

        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._check_timeout)
        self._watchdog.start()

    # ── Logging start/stop (single source of truth for all GUI controls) ──────
    # The firmware's LOG_INFO STATUS ack doesn't distinguish START/STOP/DELETE,
    # so a GUI-initiated click drives button state immediately (+ a local
    # mirror of the firmware's own auto-stop deadline) rather than waiting on
    # that ack. A trigger with no GUI command of its own — e.g. the CH6 radio
    # switch (see teensy/src/main.cpp radio_update()) — has no click to drive
    # off of, so LOG_INFO_STARTED/STOPPED (_on_log_info below) is the fallback:
    # authoritative, firmware-sourced, and a no-op whenever the click path
    # already has us in the right state (so it never fights that path's own
    # token/timer bookkeeping).

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

    def download(self, file_index: int, kind: int = LOG_FILE_KIND_WLOG):
        """Start downloading one file (.wlog by default, or its .PARAMS sidecar
        via kind=LOG_FILE_KIND_PARAMS), or attach to an already-running
        download of the same file+kind. Result arrives via transfer_progress
        (repeatedly) then transfer_complete (once).

        A caller that gives up waiting and calls this again for the same file
        (e.g. a client-side timeout on a large, still-in-progress transfer)
        must not discard the accumulated chunks — that would restart the
        whole transfer from scratch and could prevent it from ever finishing.
        """
        if self._xfer_active and self._xfer_index == file_index and self._xfer_kind == kind:
            return
        if kind == LOG_FILE_KIND_WLOG or self._output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            self._output_dir = LOG_DIR / f"{stamp}_SD_LOG{file_index:04d}"
            self._output_dir.mkdir(parents=True, exist_ok=False)
            (self._output_dir / "manifest.json").write_text(
                json.dumps({
                    "capture_schema": 1,
                    "source_kind": "sd",
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                    "robot_log_index": file_index,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        self._xfer_active = True
        self._xfer_index  = file_index
        self._xfer_kind   = kind
        self._xfer_total  = 0
        self._xfer_crc    = 0
        self._full_crc    = None
        self._request_start = 0
        self._awaiting_begin = True
        self._chunks      = {}
        self._retries     = 0
        self._failure_reason = "transfer failed"
        self._last_chunk_ms = time.monotonic()
        send_log_get(file_index, 0, kind)

    def download_with_params(self, file_index: int):
        """Download a .wlog and, once that finishes successfully, its paired
        .PARAMS sidecar — the single place that knows about the two-file
        pairing so callers (e.g. the Logs tab's Download button) don't need to."""
        # Older recordings do not have a sidecar.  Do not convert a perfectly
        # successful .WLOG download into a long, futile .PARAMS retry in that
        # case.  A missing listing is treated conservatively as WLOG-only.
        entry = next((f for f in self._directory if f["index"] == file_index), None)
        self._pending_params_index = file_index if entry and entry.get("has_params") else None
        self.download(file_index, LOG_FILE_KIND_WLOG)

    def is_transferring(self) -> bool:
        return self._xfer_active

    def failure_reason(self) -> str:
        return self._failure_reason

    def known_size(self, file_index: int) -> int | None:
        """Cached file size (bytes) from the last directory listing, if any."""
        return next((f["size"] for f in self._directory if f["index"] == file_index), None)

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
            self._listing.append({
                "index": idx,
                "size": info.get("log_file_size", 0),
                "has_params": bool(info.get("log_status", 0)),
            })
        elif sub == LOG_INFO_LIST_END:
            self._directory = list(self._listing)
            self.directory_updated.emit(self._directory)
        elif sub == LOG_INFO_XFER_BEGIN:
            if not self._xfer_active or idx != self._xfer_index:
                return
            self._xfer_total = info.get("log_total_chunks", 0)
            self._awaiting_begin = False
            self._last_chunk_ms = time.monotonic()
        elif sub == LOG_INFO_XFER_END:
            if (not self._xfer_active or idx != self._xfer_index
                    or self._awaiting_begin):
                return
            self._xfer_crc = info.get("log_crc32", 0)
            if self._request_start == 0:
                # Even if one of this attempt's chunks was lost after the
                # Teensy, XFER_END still carries the CRC of the complete file.
                # Retain it while repairing only the missing suffix.
                self._full_crc = self._xfer_crc
            self._finish_transfer()
        elif sub == LOG_INFO_STATUS:
            status = info.get("log_status", 1)
            self.status_ack.emit(idx, status)
            # A failed GET is reported as STATUS, not XFER_BEGIN.  Fail it
            # immediately; previously this looked exactly like silence and
            # cost 36+ seconds of timeout retries.
            if status and self._xfer_active and idx == self._xfer_index:
                self._xfer_active = False
                self._pending_params_index = None
                self._output_dir = None
                self._failure_reason = "log GET was rejected by the robot"
                self.transfer_complete.emit(self._xfer_index, "", False)
        elif sub == LOG_INFO_STARTED:
            if not self._logging:
                self._logging = True
                self._logging_token += 1
                self.logging_state_changed.emit(True)
        elif sub == LOG_INFO_STOPPED:
            if self._logging:
                self._logging_token += 1
                self._logging = False
                self.logging_state_changed.emit(False)

    def _on_log_data(self, info: dict):
        idx = info.get("log_file_index")
        if (not self._xfer_active or idx != self._xfer_index
                or self._awaiting_begin):
            return
        chunk_index = info.get("log_chunk_index")
        data = info.get("log_data", b"")
        if not isinstance(chunk_index, int) or not isinstance(data, bytes):
            return
        # Do not let a malformed or stale frame outside this GET's advertised
        # range affect progress or postpone recovery.
        if chunk_index < self._request_start or (
                self._xfer_total and chunk_index >= self._xfer_total):
            return
        previous = self._chunks.get(chunk_index)
        self._chunks[chunk_index] = data
        if previous != data:
            self._last_chunk_ms = time.monotonic()
        if self._xfer_total:
            self.transfer_progress.emit(self._xfer_index, len(self._chunks), self._xfer_total)

    # ── Retry / assembly ─────────────────────────────────────────────────────

    def _first_missing_chunk(self):
        for i in range(self._xfer_total):
            if i not in self._chunks:
                return i
        return None

    def _restart_transfer(self, start_chunk: int | None = None,
                          reason: str = "transfer timeout") -> bool:
        """Repair from *start_chunk*, preserving the contiguous prefix.

        The Teensy CRC in XFER_END covers only this requested suffix. Returns
        False and emits a failed transfer once MAX_RETRIES is exceeded.
        """
        self._retries += 1
        if self._retries > MAX_RETRIES:
            self._xfer_active = False
            self._output_dir = None
            self._failure_reason = f"{reason} after {MAX_RETRIES} retries"
            self.transfer_complete.emit(self._xfer_index, "", False)
            return False
        if start_chunk is None:
            start_chunk = self._first_missing_chunk()
            if start_chunk is None:
                start_chunk = self._request_start
        start_chunk = max(0, int(start_chunk))

        # Bytes at/after the repair point must all come from the same GET so
        # its suffix CRC is meaningful. Earlier validated CommLink frames are
        # retained instead of retransmitting a multi-megabyte prefix.
        self._chunks = {i: data for i, data in self._chunks.items() if i < start_chunk}
        self._request_start = start_chunk
        self._awaiting_begin = True
        self._xfer_crc = 0
        self._last_chunk_ms = time.monotonic()
        send_log_get(self._xfer_index, start_chunk, self._xfer_kind)
        return True

    def _check_timeout(self):
        if not self._xfer_active:
            return
        if time.monotonic() - self._last_chunk_ms < CHUNK_TIMEOUT_S:
            return
        # If XFER_BEGIN itself was lost, repeat the current request. Otherwise
        # continue at the first observable hole. Teensy treats GET as an
        # idempotent restart, so this is safe even if its old stream is active.
        missing = self._first_missing_chunk() if self._xfer_total else self._request_start
        self._restart_transfer(missing, "transfer timeout")

    def _finish_transfer(self):
        missing = self._first_missing_chunk()
        if missing is not None:
            self._restart_transfer(missing, "missing chunks")
            return

        suffix = b"".join(self._chunks[i] for i in range(self._request_start, self._xfer_total))
        suffix_crc_ok = (zlib.crc32(suffix) & 0xFFFFFFFF) == self._xfer_crc
        if not suffix_crc_ok:
            self._restart_transfer(self._request_start, "suffix CRC mismatch")
            return

        data = b"".join(self._chunks[i] for i in range(self._xfer_total))
        if self._full_crc is not None and (zlib.crc32(data) & 0xFFFFFFFF) != self._full_crc:
            # A repaired suffix passed its own CRC but the complete assembly
            # disagrees with the original whole-file CRC. Restart from zero;
            # this is corruption, not an ordinary missing-frame repair.
            self._full_crc = None
            self._restart_transfer(0, "full-file CRC mismatch")
            return

        ext = "PARAMS" if self._xfer_kind == LOG_FILE_KIND_PARAMS else "WLOG"
        if self._output_dir is None:
            raise RuntimeError("log transfer has no output bundle")
        out_path = self._output_dir / f"LOG{self._xfer_index:04d}.{ext}"
        out_path.write_bytes(data)
        finished_index = self._xfer_index
        finished_kind = self._xfer_kind
        self._xfer_active = False
        self.transfer_complete.emit(finished_index, str(out_path), True)

        if (finished_kind == LOG_FILE_KIND_WLOG
                and self._pending_params_index == finished_index):
            self._pending_params_index = None
            self.download(finished_index, LOG_FILE_KIND_PARAMS)
        else:
            self._output_dir = None
