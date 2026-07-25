"""log_playback.py — "Logs" tab: SD-log retrieval + rosbag-style playback.

Retrieve panel: start/stop/timed logging, refresh directory, download,
delete — all via LogTransferManager, which works identically over USB serial
or WiFi (see log_transfer.py / log_bus.py).

Playback panel: pick a .wlog already downloaded to the default logs/ folder
(or browse elsewhere), play/pause/scrub/speed. Playback is driven by
LogPlaybackController, a singleton, so the status-bar mini transport (see
main.py) can control/reflect the same session from any tab. Each record is
replayed onto TelemetryBus.packet exactly as a live TELEM packet, so every
other tab (3D visualizer, IMU charts, Raw Data) animates unchanged.
"""

import struct
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from .comm_commands import send_log_delete
from .host_logger import HostLogger
from .log_paths import RUNS_DIR
from .log_transfer import LogTransferManager
from .telem_format import TELEM_VERSION, decode_telem_full
from .telemetry_bus import TelemetryBus
from .theme import BG, BLUE, BORDER, DIM, GREEN, RED, SURFACE, TEXT

_FMT_HEADER = "<8sBBHHI14s"
_T_MICROS_SIZE = 4

SPEED_OPTIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
SPEED_DEFAULT_INDEX = SPEED_OPTIONS.index(1.0)


class LogReader:
    """Random-access reader for a .wlog file. read(idx) returns one dict
    shaped like a live TELEM packet (ptype=0x01), so TelemetryBus consumers
    can't tell replay apart from a connected robot."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        header = self._f.read(32)
        if len(header) < 32:
            raise ValueError(f"{path}: too short for WlogHeader")
        (magic, self.format_version, self.telem_version, self.record_size,
         self.sample_rate_hz, self.start_millis, _reserved) = struct.unpack(_FMT_HEADER, header)
        magic = magic.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        if magic != "WLRLOG":
            raise ValueError(f"{path}: bad magic {magic!r} — not a .wlog file")
        self._data_start = 32
        data_bytes = self.path.stat().st_size - self._data_start
        self.count = max(0, data_bytes // self.record_size)

    def read(self, idx: int) -> dict | None:
        if idx < 0 or idx >= self.count:
            return None
        self._f.seek(self._data_start + idx * self.record_size)
        rec = self._f.read(self.record_size)
        if len(rec) < self.record_size:
            return None
        t_micros = struct.unpack_from("<I", rec, 0)[0]
        telem_size = self.record_size - _T_MICROS_SIZE
        info = decode_telem_full(rec[_T_MICROS_SIZE:_T_MICROS_SIZE + telem_size])
        info["ptype"] = 0x01
        info["type_name"] = "TELEM"
        info["t_micros"] = t_micros
        return info

    def close(self):
        self._f.close()


class LogPlaybackController(QObject):
    """App-wide singleton — owns the open LogReader + play/pause/seek state,
    so the Logs tab and the status-bar mini transport both drive/reflect the
    same session regardless of which tab is currently visible."""

    file_opened      = pyqtSignal(str, int, int)  # path, count, sample_rate_hz
    open_failed      = pyqtSignal(str)
    playing_changed  = pyqtSignal(bool)
    position_changed = pyqtSignal(int, int)        # idx, total

    _instance: "LogPlaybackController | None" = None

    @classmethod
    def instance(cls) -> "LogPlaybackController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reader: LogReader | None = None
        self._playing = False
        self._idx = 0
        self._speed = 1.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    # ── File ─────────────────────────────────────────────────────────────────

    def open(self, path: Path) -> bool:
        self.pause()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        try:
            reader = LogReader(path)
        except Exception as e:
            self.open_failed.emit(str(e))
            return False
        self._reader = reader
        self._idx = 0
        self.file_opened.emit(str(path), reader.count, reader.sample_rate_hz)
        self.position_changed.emit(0, reader.count)
        return True

    def is_open(self) -> bool:
        return self._reader is not None

    def reader(self) -> "LogReader | None":
        return self._reader

    # ── Transport ────────────────────────────────────────────────────────────

    def is_playing(self) -> bool:
        return self._playing

    def set_speed(self, speed: float):
        self._speed = max(0.01, speed)
        if self._playing:
            self._restart_timer()

    def _restart_timer(self):
        self._timer.start(max(1, int(1000 / self._reader.sample_rate_hz / self._speed)))

    def play(self):
        if self._reader is None or self._reader.count == 0 or self._playing:
            return
        if self._idx >= self._reader.count:
            self._idx = 0
        self._playing = True
        TelemetryBus.instance().set_playback_active(True)
        self._restart_timer()
        self.playing_changed.emit(True)

    def pause(self):
        if not self._playing:
            return
        self._timer.stop()
        self._playing = False
        self.playing_changed.emit(False)
        # Deliberately leaves TelemetryBus.playback_active True — the whole
        # point is to keep the "playback mode" indicator up while paused, so
        # a mid-review pause never gets mistaken for a live feed resuming.

    def toggle(self):
        self.pause() if self._playing else self.play()

    def stop(self):
        """Full exit from playback mode: rewind, and let live sources resume."""
        self._timer.stop()
        was_playing = self._playing
        self._playing = False
        self._idx = 0
        TelemetryBus.instance().set_playback_active(False)
        if was_playing:
            self.playing_changed.emit(False)
        if self._reader is not None:
            self.position_changed.emit(0, self._reader.count)

    def seek(self, idx: int):
        if self._reader is None:
            return
        idx = max(0, min(idx, self._reader.count - 1))
        self._idx = idx
        self._emit_current()
        self.position_changed.emit(self._idx, self._reader.count)

    def _emit_current(self):
        if self._reader is None:
            return
        rec = self._reader.read(self._idx)
        if rec is not None:
            TelemetryBus.instance().packet.emit(rec)

    def _on_tick(self):
        if self._reader is None or self._idx >= self._reader.count:
            self.pause()  # freeze on last frame rather than silently going live
            return
        self._emit_current()
        self.position_changed.emit(self._idx, self._reader.count)
        self._idx += 1


def _btn_style(color: str) -> str:
    return (
        f"QPushButton {{ background: {SURFACE}; color: {color}; border: 1px solid {color}; "
        f"border-radius: 3px; padding: 4px 12px; }}"
        f"QPushButton:hover {{ background: {BORDER}; }}"
        f"QPushButton:disabled {{ color: {DIM}; border: 1px solid {BORDER}; }}"
    )


class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._xfer = LogTransferManager.instance()
        self._pb   = LogPlaybackController.instance()
        self._host = HostLogger.instance()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.addWidget(self._build_retrieve_panel())
        outer.addWidget(self._build_host_capture_panel())
        outer.addWidget(self._build_playback_panel())
        outer.addStretch()

        self._xfer.directory_updated.connect(self._on_directory)
        self._xfer.transfer_progress.connect(self._on_progress)
        self._xfer.transfer_complete.connect(self._on_transfer_complete)
        self._xfer.status_ack.connect(self._on_status_ack)
        self._xfer.logging_state_changed.connect(self._on_logging_state)

        self._host.logging_state_changed.connect(self._on_host_logging_state)
        self._host.record_count_changed.connect(self._on_host_record_count)

        self._pb.file_opened.connect(self._on_file_opened)
        self._pb.open_failed.connect(self._on_open_failed)
        self._pb.playing_changed.connect(self._on_playing_changed)
        self._pb.position_changed.connect(self._on_position_changed)

        self._refresh_local_files()

    # ── Retrieve panel ───────────────────────────────────────────────────────

    def _build_retrieve_panel(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; }}")
        lay = QVBoxLayout(frame)

        title = QLabel("SD Log Retrieval")
        title.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 13px;")
        lay.addWidget(title)

        ctrl_row = QHBoxLayout()
        self._btn_start = QPushButton("Start")
        self._btn_start.setStyleSheet(_btn_style(GREEN))
        self._btn_start.clicked.connect(self._on_start)
        ctrl_row.addWidget(self._btn_start)

        self._dur_spin = QSpinBox()
        self._dur_spin.setRange(0, 3600)
        self._dur_spin.setValue(30)
        self._dur_spin.setSuffix(" s (0 = until Stop)")
        ctrl_row.addWidget(self._dur_spin)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setStyleSheet(_btn_style(RED))
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._xfer.stop_logging)
        ctrl_row.addWidget(self._btn_stop)

        ctrl_row.addSpacing(16)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setStyleSheet(_btn_style(BLUE))
        btn_refresh.clicked.connect(self._xfer.refresh)
        ctrl_row.addWidget(btn_refresh)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._file_list = QListWidget()
        self._file_list.setStyleSheet(
            f"QListWidget {{ background: {BG}; color: {TEXT}; border: 1px solid {BORDER}; }}"
        )
        self._file_list.setFixedHeight(120)
        lay.addWidget(self._file_list)

        file_row = QHBoxLayout()
        btn_download = QPushButton("Download Selected")
        btn_download.setStyleSheet(_btn_style(BLUE))
        btn_download.clicked.connect(self._on_download)
        file_row.addWidget(btn_download)

        btn_delete = QPushButton("Delete Selected")
        btn_delete.setStyleSheet(_btn_style(RED))
        btn_delete.clicked.connect(self._on_delete)
        file_row.addWidget(btn_delete)
        file_row.addStretch()
        lay.addLayout(file_row)

        self._progress = QProgressBar()
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: {BG}; color: {TEXT}; border: 1px solid {BORDER}; }}"
            f"QProgressBar::chunk {{ background: {GREEN}; }}"
        )
        lay.addWidget(self._progress)

        self._lbl_status = QLabel("No directory loaded — click Refresh")
        self._lbl_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(self._lbl_status)

        return frame

    def _on_start(self):
        self._xfer.start_logging(self._dur_spin.value() * 1000)

    def _on_logging_state(self, active: bool):
        self._btn_start.setEnabled(not active)
        self._btn_stop.setEnabled(active)

    def _on_directory(self, entries: list):
        self._file_list.clear()
        for e in entries:
            size_kb = e["size"] / 1024.0
            tag = "  [+params]" if e.get("has_params") else ""
            item = QListWidgetItem(f"LOG{e['index']:04d}.WLOG   ({size_kb:.1f} KB){tag}")
            item.setData(Qt.ItemDataRole.UserRole, e["index"])
            self._file_list.addItem(item)
        self._lbl_status.setText(f"{len(entries)} log(s) on card")

    def _selected_index(self):
        item = self._file_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _on_download(self):
        idx = self._selected_index()
        if idx is None:
            return
        self._progress.setValue(0)
        self._lbl_status.setText(f"Downloading LOG{idx:04d}.WLOG...")
        self._xfer.download_with_params(idx)

    def _on_delete(self):
        idx = self._selected_index()
        if idx is None:
            return
        send_log_delete(idx)

    def _on_progress(self, idx, got, total):
        pct = int(100 * got / total) if total else 0
        self._progress.setValue(pct)
        self._lbl_status.setText(f"LOG{idx:04d}: {got}/{total} chunks ({pct}%)")

    def _on_transfer_complete(self, idx, path, crc_ok):
        # Fires once per file — download_with_params() chains a second
        # (.PARAMS) transfer after the .wlog succeeds, so this can run twice
        # per Download click. Name comes from the path, not a hardcoded
        # extension, so the status line reflects whichever file just finished.
        name = Path(path).name if path else f"LOG{idx:04d}"
        if crc_ok:
            self._lbl_status.setText(f"{name} downloaded OK -> {path}")
        else:
            self._lbl_status.setText(
                f"LOG{idx:04d} download FAILED ({self._xfer.failure_reason()})"
            )
        self._xfer.refresh()
        self._refresh_local_files()

    def _on_status_ack(self, idx, status):
        ok = "OK" if status == 0 else f"ERROR {status}"
        self._lbl_status.setText(f"LOG{idx:04d}.WLOG command ack: {ok}")
        if status == 0:
            self._xfer.refresh()

    # ── Host capture panel ───────────────────────────────────────────────────

    def _build_host_capture_panel(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; }}")
        lay = QVBoxLayout(frame)

        title = QLabel("Host Capture (GUI-side)")
        title.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 13px;")
        lay.addWidget(title)

        note = QLabel(
            "Records every packet from the selected live source before UI coalescing — telemetry, robot log "
            "messages, calibration events, WiFi diagnostics — to a timestamped "
            ".jsonl file in data/logs/runs/, at whatever rate it arrives. Independent "
            "of SD logging; starts/stops only when you click below."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(note)

        ctrl_row = QHBoxLayout()
        self._btn_host_start = QPushButton("Start")
        self._btn_host_start.setStyleSheet(_btn_style(GREEN))
        self._btn_host_start.clicked.connect(self._host.start)
        ctrl_row.addWidget(self._btn_host_start)

        self._btn_host_stop = QPushButton("Stop")
        self._btn_host_stop.setStyleSheet(_btn_style(RED))
        self._btn_host_stop.setEnabled(False)
        self._btn_host_stop.clicked.connect(self._host.stop)
        ctrl_row.addWidget(self._btn_host_stop)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._lbl_host_status = QLabel("Not capturing")
        self._lbl_host_status.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(self._lbl_host_status)

        return frame

    def _on_host_logging_state(self, active: bool):
        self._btn_host_start.setEnabled(not active)
        self._btn_host_stop.setEnabled(active)
        if active:
            self._lbl_host_status.setText(f"Capturing -> {self._host.path().name}")
        elif self._host.path() is not None:
            self._lbl_host_status.setText(
                f"Stopped. Wrote {self._host.record_count()} records -> {self._host.path()}"
            )

    def _on_host_record_count(self, count: int):
        if self._host.is_logging():
            self._lbl_host_status.setText(f"Capturing -> {self._host.path().name}  ({count} records)")

    # ── Playback panel ───────────────────────────────────────────────────────

    def _build_playback_panel(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; }}")
        lay = QVBoxLayout(frame)

        title = QLabel("Playback")
        title.setStyleSheet(f"color: {TEXT}; font-weight: bold; font-size: 13px;")
        lay.addWidget(title)

        lbl_local = QLabel(f"Saved locally in {RUNS_DIR}:")
        lbl_local.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(lbl_local)

        local_row = QHBoxLayout()
        self._local_list = QListWidget()
        self._local_list.setStyleSheet(
            f"QListWidget {{ background: {BG}; color: {TEXT}; border: 1px solid {BORDER}; }}"
        )
        self._local_list.setFixedHeight(100)
        self._local_list.itemDoubleClicked.connect(self._on_local_double_clicked)
        local_row.addWidget(self._local_list)

        local_btns = QVBoxLayout()
        btn_open_selected = QPushButton("Open Selected")
        btn_open_selected.setStyleSheet(_btn_style(BLUE))
        btn_open_selected.clicked.connect(self._on_open_local_selected)
        local_btns.addWidget(btn_open_selected)

        btn_browse = QPushButton("Browse elsewhere...")
        btn_browse.setStyleSheet(_btn_style(BLUE))
        btn_browse.clicked.connect(self._on_browse)
        local_btns.addWidget(btn_browse)

        btn_local_refresh = QPushButton("Refresh list")
        btn_local_refresh.setStyleSheet(_btn_style(DIM))
        btn_local_refresh.clicked.connect(self._refresh_local_files)
        local_btns.addWidget(btn_local_refresh)
        local_btns.addStretch()
        local_row.addLayout(local_btns)
        lay.addLayout(local_row)

        row1 = QHBoxLayout()
        self._lbl_file = QLabel("No file open")
        self._lbl_file.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        row1.addWidget(self._lbl_file)
        row1.addStretch()

        btn_export = QPushButton("Export CSV")
        btn_export.setStyleSheet(_btn_style(GREEN))
        btn_export.clicked.connect(self._on_export_csv)
        row1.addWidget(btn_export)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_play.setStyleSheet(_btn_style(GREEN))
        self._btn_play.clicked.connect(self._pb.toggle)
        row2.addWidget(self._btn_play)

        btn_stop = QPushButton("Stop")
        btn_stop.setStyleSheet(_btn_style(RED))
        btn_stop.clicked.connect(self._pb.stop)
        row2.addWidget(btn_stop)

        row2.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems([f"{s:g}x" for s in SPEED_OPTIONS])
        self._speed_combo.setCurrentIndex(SPEED_DEFAULT_INDEX)
        self._speed_combo.currentIndexChanged.connect(
            lambda i: self._pb.set_speed(SPEED_OPTIONS[i])
        )
        self._speed_combo.setStyleSheet(
            f"QComboBox {{ background: {BG}; color: {TEXT}; border: 1px solid {BORDER}; padding: 2px 6px; }}"
        )
        row2.addWidget(self._speed_combo)
        row2.addStretch()
        lay.addLayout(row2)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self._pb.seek)
        lay.addWidget(self._slider)

        self._lbl_pos = QLabel("0 / 0")
        self._lbl_pos.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        lay.addWidget(self._lbl_pos)

        return frame

    def _refresh_local_files(self):
        self._local_list.clear()
        if not RUNS_DIR.exists():
            return
        for p in sorted(RUNS_DIR.rglob("*.[Ww][Ll][Oo][Gg]")):
            size_kb = p.stat().st_size / 1024.0
            item = QListWidgetItem(f"{p.name}   ({size_kb:.1f} KB)")
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self._local_list.addItem(item)

    def _on_local_double_clicked(self, item: QListWidgetItem):
        self._pb.open(Path(item.data(Qt.ItemDataRole.UserRole)))

    def _on_open_local_selected(self):
        item = self._local_list.currentItem()
        if item is None:
            return
        self._pb.open(Path(item.data(Qt.ItemDataRole.UserRole)))

    def _on_browse(self):
        start_dir = str(RUNS_DIR) if RUNS_DIR.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open .wlog", start_dir, "WLOG files (*.WLOG *.wlog)")
        if path:
            self._pb.open(Path(path))

    def _on_file_opened(self, path: str, count: int, sample_rate_hz: int):
        reader = self._pb.reader()
        self._slider.setRange(0, max(0, count - 1))
        self._slider.setValue(0)
        telem_v = reader.telem_version if reader else "?"
        self._lbl_file.setText(f"{Path(path).name}  ({count} records @ {sample_rate_hz} Hz, telem_v{telem_v})")
        if reader is not None and reader.telem_version != TELEM_VERSION:
            QMessageBox.warning(
                self, "Version mismatch",
                f"Log was captured with TELEM_VERSION {reader.telem_version}, "
                f"this GUI decodes TELEM_VERSION {TELEM_VERSION}. Fields may be misread.",
            )

    def _on_open_failed(self, msg: str):
        QMessageBox.warning(self, "Open failed", msg)

    def _on_playing_changed(self, playing: bool):
        self._btn_play.setText("Pause" if playing else "Play")

    def _on_position_changed(self, idx: int, total: int):
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._lbl_pos.setText(f"{idx} / {total}")

    def _on_export_csv(self):
        reader = self._pb.reader()
        if reader is None:
            return
        tools_dir = Path(__file__).parent.parent / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from wlog_to_csv import read_wlog
        import pandas as pd
        out_path = reader.path.with_suffix(".csv")
        rows = list(read_wlog(reader.path))
        pd.DataFrame(rows).to_csv(out_path, index=False)
        QMessageBox.information(self, "Export complete", f"Wrote {len(rows)} rows -> {out_path}")
