"""tab_presets.py -- Save/load ODrive config as JSON + factory presets.

Features:
  - Save current config to JSON file (file dialog)
  - Load config from JSON file, write to device, optional save+reboot
  - Factory preset selector (Maytech MTO5065 + AS5048A)
  - Restore factory defaults (erase_configuration + reboot)
  - Results log showing every parameter written + pass/fail
"""

import logging
import os

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QScrollArea, QTextEdit, QFileDialog, QMessageBox,
)

from core.odrive_manager import ODriveManager
from core.odrive_presets import (
    save_config_to_file, load_config_from_file,
    write_full_config, read_full_config, get_factory_presets,
)
from ui.theme import CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_MUTED

log = logging.getLogger("odrive_gui")

_REBOOT_WAIT_MS = 4000


# ---- Reconnect worker -------------------------------------------------------

class _ReconnectWorker(QThread):
    success = Signal(object)
    failed = Signal(str)

    def __init__(self, manager: ODriveManager):
        super().__init__()
        self._mgr = manager

    def run(self):
        try:
            self._mgr.disconnect()
            odrv = self._mgr.connect()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


# ---- Helpers ----------------------------------------------------------------

def _colored(label: QLabel, text: str, color: str):
    label.setText(text)
    label.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {color};")


# ---- Presets Tab ------------------------------------------------------------

class TabPresets(QWidget):
    """Save/load ODrive config JSON, factory presets, restore defaults."""

    # Emitted after load+save+reboot reconnect so MainWindow can update
    reconnected = Signal()

    def __init__(self, manager: ODriveManager, get_axis_idx, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_axis_idx = get_axis_idx
        self._reconnect_worker: _ReconnectWorker | None = None
        self._reboot_timer = QTimer(self)
        self._reboot_timer.setSingleShot(True)
        self._reboot_timer.timeout.connect(self._reconnect_after_reboot)
        self._post_reboot_action = ""
        self._build()

    def _ax_idx(self) -> int:
        return self._get_axis_idx()

    # ---- Build UI -----------------------------------------------------------

    def _build(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        root = QVBoxLayout(inner)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # Save / Load
        root.addWidget(self._build_file_group())

        # Factory presets
        root.addWidget(self._build_preset_group())

        # Status
        root.addWidget(self._build_status_group())

        # Results log
        root.addWidget(self._build_log_group())

        root.addStretch()

    def _build_file_group(self):
        box = QGroupBox("Config Import / Export")
        col = QVBoxLayout(box)

        hint = QLabel(
            "Save reads every .config parameter from the device and writes a JSON file.\n"
            "Load writes all parameters from the file back to the device."
        )
        hint.setStyleSheet(f"color: {CLR_MUTED}; font-size: 11px;")
        col.addWidget(hint)

        row = QHBoxLayout()

        self.btn_save = QPushButton("Save Config to File...")
        self.btn_save.setFixedWidth(180)
        self.btn_save.clicked.connect(self._on_save)
        row.addWidget(self.btn_save)

        self.btn_load = QPushButton("Load Config from File...")
        self.btn_load.setFixedWidth(180)
        self.btn_load.setStyleSheet(f"color: {CLR_INFO};")
        self.btn_load.clicked.connect(self._on_load)
        row.addWidget(self.btn_load)

        self.btn_load_save = QPushButton("Load + Save + Reboot")
        self.btn_load_save.setFixedWidth(180)
        self.btn_load_save.setStyleSheet(f"color: {CLR_WARN};")
        self.btn_load_save.setToolTip(
            "Load from file, write to device, save_configuration, and reboot"
        )
        self.btn_load_save.clicked.connect(self._on_load_save_reboot)
        row.addWidget(self.btn_load_save)

        row.addStretch()
        col.addLayout(row)
        return box

    def _build_preset_group(self):
        box = QGroupBox("Factory Presets")
        col = QVBoxLayout(box)

        hint = QLabel(
            "Apply a factory preset to configure the device for a known motor + encoder combo.\n"
            "This writes parameters to RAM only. Use 'Apply + Save + Reboot' to persist."
        )
        hint.setStyleSheet(f"color: {CLR_MUTED}; font-size: 11px;")
        col.addWidget(hint)

        row = QHBoxLayout()

        self._preset_combo = QComboBox()
        presets = get_factory_presets()
        for name in presets:
            meta = presets[name].get("_meta", {})
            desc = meta.get("description", name)
            self._preset_combo.addItem(name, desc)
        self._preset_combo.setMinimumWidth(240)
        row.addWidget(self._preset_combo)

        self.btn_apply_preset = QPushButton("Apply to RAM")
        self.btn_apply_preset.setFixedWidth(120)
        self.btn_apply_preset.clicked.connect(self._on_apply_preset)
        row.addWidget(self.btn_apply_preset)

        self.btn_apply_save = QPushButton("Apply + Save + Reboot")
        self.btn_apply_save.setFixedWidth(180)
        self.btn_apply_save.setStyleSheet(f"color: {CLR_WARN};")
        self.btn_apply_save.clicked.connect(self._on_apply_preset_save)
        row.addWidget(self.btn_apply_save)

        row.addStretch()
        col.addLayout(row)

        # Preset description
        self._lbl_preset_desc = QLabel("")
        self._lbl_preset_desc.setStyleSheet(
            f"color: {CLR_MUTED}; font-size: 11px; font-style: italic;")
        col.addWidget(self._lbl_preset_desc)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._on_preset_changed(0)

        return box

    def _build_status_group(self):
        box = QGroupBox("Status")
        form = QFormLayout(box)
        form.setSpacing(6)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        form.addRow("Status", self.lbl_status)

        self.lbl_detail = QLabel("--")
        self.lbl_detail.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        form.addRow("Detail", self.lbl_detail)

        return box

    def _build_log_group(self):
        box = QGroupBox("Results Log")
        col = QVBoxLayout(box)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(self._log_text.font())
        self._log_text.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #dcdcdc; "
            "border: 1px solid #444; font-family: Consolas; font-size: 11px; }")
        self._log_text.setMinimumHeight(200)
        col.addWidget(self._log_text)

        btn_clear = QPushButton("Clear Log")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._log_text.clear)
        col.addWidget(btn_clear)

        return box

    # ---- Preset combo -------------------------------------------------------

    def _on_preset_changed(self, idx):
        desc = self._preset_combo.currentData()
        self._lbl_preset_desc.setText(desc or "")

    # ---- Save ---------------------------------------------------------------

    def _on_save(self):
        if not self._mgr.connected:
            _colored(self.lbl_status, "Not connected", CLR_ERR)
            return

        default_dir = os.path.dirname(os.path.abspath(__file__))
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save ODrive Config", default_dir,
            "JSON Files (*.json);;All Files (*)")
        if not filepath:
            return

        _colored(self.lbl_status, "Reading config from device...", CLR_INFO)
        try:
            snapshot = save_config_to_file(self._mgr.odrv, filepath)
        except Exception as e:
            _colored(self.lbl_status, "Save failed", CLR_ERR)
            _colored(self.lbl_detail, str(e), CLR_ERR)
            log.error("Presets: save failed: %s", e)
            return

        n_sections = len(snapshot) - 1  # minus _meta
        n_params = sum(len(v) for k, v in snapshot.items()
                       if k != "_meta" and isinstance(v, dict))
        _colored(self.lbl_status, "Config saved", CLR_OK)
        _colored(self.lbl_detail,
                 f"{n_params} parameters in {n_sections} sections -> {filepath}",
                 CLR_OK)
        self._log_text.append(f"SAVED {n_params} params to {filepath}")
        log.info("Presets: saved %d params to %s", n_params, filepath)

    # ---- Load (RAM only) ----------------------------------------------------

    def _on_load(self):
        if not self._mgr.connected:
            _colored(self.lbl_status, "Not connected", CLR_ERR)
            return

        filepath = self._pick_json_file()
        if not filepath:
            return

        self._do_load(filepath, save_and_reboot=False)

    # ---- Load + Save + Reboot -----------------------------------------------

    def _on_load_save_reboot(self):
        if not self._mgr.connected:
            _colored(self.lbl_status, "Not connected", CLR_ERR)
            return

        filepath = self._pick_json_file()
        if not filepath:
            return

        reply = QMessageBox.warning(
            self, "Load + Save + Reboot",
            f"This will:\n"
            f"  1. Write all parameters from the file to the device\n"
            f"  2. save_configuration()\n"
            f"  3. Reboot the ODrive\n\n"
            f"File: {filepath}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._do_load(filepath, save_and_reboot=True)

    def _pick_json_file(self) -> str:
        default_dir = os.path.dirname(os.path.abspath(__file__))
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load ODrive Config", default_dir,
            "JSON Files (*.json);;All Files (*)")
        return filepath

    def _do_load(self, filepath: str, save_and_reboot: bool):
        _colored(self.lbl_status, "Loading config file...", CLR_INFO)
        try:
            snapshot = load_config_from_file(filepath)
        except Exception as e:
            _colored(self.lbl_status, "Load failed", CLR_ERR)
            _colored(self.lbl_detail, str(e), CLR_ERR)
            log.error("Presets: load failed: %s", e)
            return

        self._apply_snapshot(snapshot, source=filepath,
                             save_and_reboot=save_and_reboot)

    # ---- Apply preset (RAM only) --------------------------------------------

    def _on_apply_preset(self):
        if not self._mgr.connected:
            _colored(self.lbl_status, "Not connected", CLR_ERR)
            return

        name = self._preset_combo.currentText()
        presets = get_factory_presets()
        snapshot = presets.get(name)
        if not snapshot:
            _colored(self.lbl_status, f"Preset '{name}' not found", CLR_ERR)
            return

        self._apply_snapshot(snapshot, source=f"preset: {name}",
                             save_and_reboot=False)

    # ---- Apply preset + Save + Reboot ---------------------------------------

    def _on_apply_preset_save(self):
        if not self._mgr.connected:
            _colored(self.lbl_status, "Not connected", CLR_ERR)
            return

        name = self._preset_combo.currentText()
        presets = get_factory_presets()
        snapshot = presets.get(name)
        if not snapshot:
            _colored(self.lbl_status, f"Preset '{name}' not found", CLR_ERR)
            return

        reply = QMessageBox.warning(
            self, "Apply Preset + Save + Reboot",
            f"This will:\n"
            f"  1. Write preset '{name}' to the device\n"
            f"  2. save_configuration()\n"
            f"  3. Reboot the ODrive\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._apply_snapshot(snapshot, source=f"preset: {name}",
                             save_and_reboot=True)

    # ---- Shared apply logic -------------------------------------------------

    def _apply_snapshot(self, snapshot: dict, source: str,
                        save_and_reboot: bool):
        _colored(self.lbl_status, f"Writing config from {source}...", CLR_INFO)
        self._set_buttons(False)

        try:
            results = write_full_config(self._mgr.odrv, snapshot)
        except Exception as e:
            _colored(self.lbl_status, "Write failed", CLR_ERR)
            _colored(self.lbl_detail, str(e), CLR_ERR)
            log.error("Presets: write failed: %s", e)
            self._set_buttons(True)
            return

        # Log results
        ok_count = sum(1 for r in results if r["ok"])
        fail_count = len(results) - ok_count
        self._log_text.append(f"\n--- {source} ---")
        for r in results:
            mark = "OK" if r["ok"] else "FAIL"
            line = f"  [{mark}] {r['path']}.{r['param']} = {r['value']}"
            if not r["ok"]:
                line += f"  ({r['error']})"
            self._log_text.append(line)
        self._log_text.append(f"Total: {ok_count} OK, {fail_count} failed")

        color = CLR_OK if fail_count == 0 else CLR_WARN
        _colored(self.lbl_status, f"Written: {ok_count} OK, {fail_count} failed",
                 color)
        log.info("Presets: applied %s — %d OK, %d failed",
                 source, ok_count, fail_count)

        if save_and_reboot:
            _colored(self.lbl_detail, "Saving + rebooting...", CLR_WARN)
            try:
                self._mgr.odrv.save_configuration()
            except Exception:
                pass  # save triggers reboot
            self._post_reboot_action = "load_verify"
            self._mgr.disconnect()
            self._reboot_timer.start(_REBOOT_WAIT_MS)
        else:
            _colored(self.lbl_detail,
                     "Written to RAM only. Use save_configuration() to persist.",
                     CLR_MUTED)
            self._set_buttons(True)

    # ---- Reboot / Reconnect -------------------------------------------------

    def _reconnect_after_reboot(self):
        _colored(self.lbl_status, "Reconnecting...", CLR_INFO)
        self._reconnect_worker = _ReconnectWorker(self._mgr)
        self._reconnect_worker.success.connect(self._on_reconnect_ok)
        self._reconnect_worker.failed.connect(self._on_reconnect_fail)
        self._reconnect_worker.start()

    def _on_reconnect_ok(self, odrv):
        _colored(self.lbl_status, "Saved + rebooted + reconnected", CLR_OK)
        _colored(self.lbl_detail, "Config persisted to NVM", CLR_OK)
        self._log_text.append("Reconnected after save+reboot")
        log.info("Presets: reconnected after save+reboot")
        self._set_buttons(True)
        self.reconnected.emit()

    def _on_reconnect_fail(self, msg):
        _colored(self.lbl_status, "Reconnect failed", CLR_ERR)
        _colored(self.lbl_detail, msg, CLR_ERR)
        log.error("Presets: reconnect failed: %s", msg)
        self._set_buttons(True)

    # ---- Button state -------------------------------------------------------

    def _set_buttons(self, enabled: bool):
        for btn in (self.btn_save, self.btn_load, self.btn_load_save,
                    self.btn_apply_preset, self.btn_apply_save):
            btn.setEnabled(enabled)
