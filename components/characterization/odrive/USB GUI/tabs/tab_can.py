"""tab_can.py — CAN bus setup for robot integration.

Configures the ODrive so it streams encoder feedback and accepts
velocity/position/torque commands from the Arduino over CAN.

Required settings (applied by "Apply & Save"):
  odrv0.can.config.baud_rate               = 1_000_000  # 1 Mbps
  odrv0.axis0.config.can.node_id           = 0          # matches ODESC_NODE_L
  odrv0.axis1.config.can.node_id           = 1          # matches ODESC_NODE_R
  odrv0.axis0.config.can.encoder_rate_ms   = 10         # 100 Hz encoder stream
  odrv0.axis1.config.can.encoder_rate_ms   = 10
  odrv0.axis0.config.can.heartbeat_rate_ms = 100        # 10 Hz heartbeat
  odrv0.axis1.config.can.heartbeat_rate_ms = 100
  odrv0.save_configuration()   → ODrive reboots
"""

import logging

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox,
)

from ui.theme import CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_MUTED

log = logging.getLogger("odrive_gui")

_REBOOT_WAIT_MS = 4500   # ms to wait for ODrive to come back after save


class _ApplyWorker(QThread):
    done   = Signal(str, bool)   # (message, success)

    def __init__(self, mgr, baud, node_id, enc_rate_ms, hb_rate_ms):
        super().__init__()
        self._mgr        = mgr
        self._baud       = baud
        self._node_id    = node_id
        self._enc_rate   = enc_rate_ms
        self._hb_rate    = hb_rate_ms

    def run(self):
        try:
            odrv = self._mgr.odrv
            if odrv is None:
                self.done.emit("Not connected", False)
                return

            odrv.can.config.baud_rate = self._baud
            # Configure both axes — firmware expects node_id 0=left, 1=right
            odrv.axis0.config.can.node_id = self._node_id
            odrv.axis0.config.can.encoder_rate_ms = self._enc_rate
            odrv.axis0.config.can.heartbeat_rate_ms = self._hb_rate
            odrv.axis1.config.can.node_id = self._node_id + 1
            odrv.axis1.config.can.encoder_rate_ms = self._enc_rate
            odrv.axis1.config.can.heartbeat_rate_ms = self._hb_rate

            log.info(
                "CAN config applied: baud=%d node=%d enc_rate=%d hb_rate=%d",
                self._baud, self._node_id, self._enc_rate, self._hb_rate,
            )
            self.done.emit("Config written — saving…", True)
        except Exception as e:
            self.done.emit(f"Error: {e}", False)


class _SaveWorker(QThread):
    done = Signal(str, bool)

    def __init__(self, mgr):
        super().__init__()
        self._mgr = mgr

    def run(self):
        try:
            self._mgr.odrv.save_configuration()
        except Exception:
            pass   # save_configuration() triggers a reboot → USB exception expected
        self.done.emit("Saved — ODrive rebooting…", True)


class _ReadWorker(QThread):
    done = Signal(object)   # dict of values, or None on error

    def __init__(self, mgr):
        super().__init__()
        self._mgr = mgr

    def run(self):
        try:
            odrv = self._mgr.odrv
            if odrv is None:
                self.done.emit(None)
                return
            self.done.emit({
                "baud":        odrv.can.config.baud_rate,
                "node_id":     odrv.axis0.config.can.node_id,
                "node_id_r":   odrv.axis1.config.can.node_id,
                "enc_rate":    odrv.axis0.config.can.encoder_rate_ms,
                "hb_rate":     odrv.axis0.config.can.heartbeat_rate_ms,
            })
        except Exception as e:
            log.warning("CAN tab read error: %s", e)
            self.done.emit(None)


class TabCan(QWidget):
    """CAN setup tab — configure ODrive for robot CAN integration."""

    # Emitted after save + reboot so main window can reconnect
    reconnected = Signal()

    def __init__(self, mgr, ax_idx_fn):
        super().__init__()
        self._mgr    = mgr
        self._ax_idx = ax_idx_fn

        self._apply_worker: _ApplyWorker | None = None
        self._save_worker:  _SaveWorker  | None = None
        self._read_worker:  _ReadWorker  | None = None
        self._reboot_timer = QTimer(self)
        self._reboot_timer.setSingleShot(True)
        self._reboot_timer.timeout.connect(self._after_reboot)

        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Current config readout ────────────────────────────────────────
        grp_read = QGroupBox("Current CAN config (from ODrive)")
        fl_read = QFormLayout(grp_read)
        fl_read.setSpacing(6)

        self._lbl_baud    = QLabel("—")
        self._lbl_node    = QLabel("—")
        self._lbl_node_r  = QLabel("—")
        self._lbl_enc     = QLabel("—")
        self._lbl_hb      = QLabel("—")

        for lbl, widget in [
            ("Baud rate",           self._lbl_baud),
            ("Axis 0 node ID (L)",  self._lbl_node),
            ("Axis 1 node ID (R)",  self._lbl_node_r),
            ("Encoder rate (ms)",   self._lbl_enc),
            ("Heartbeat rate (ms)", self._lbl_hb),
        ]:
            widget.setStyleSheet(f"font-family: monospace; color: {CLR_INFO};")
            fl_read.addRow(lbl, widget)

        btn_read = QPushButton("Read from ODrive")
        btn_read.clicked.connect(self._do_read)
        fl_read.addRow("", btn_read)
        root.addWidget(grp_read)

        # ── Target config ────────────────────────────────────────────────
        grp_set = QGroupBox("Target CAN config")
        fl_set = QFormLayout(grp_set)
        fl_set.setSpacing(6)

        note = QLabel(
            "These values must match config.h in the Arduino firmware.\n"
            "Defaults below are correct for the wheeled-leg robot."
        )
        note.setStyleSheet(f"color: {CLR_MUTED}; font-size: 11px;")
        note.setWordWrap(True)
        fl_set.addRow(note)

        self._sp_baud = QSpinBox()
        self._sp_baud.setRange(125_000, 1_000_000)
        self._sp_baud.setSingleStep(125_000)
        self._sp_baud.setValue(1_000_000)
        self._sp_baud.setSuffix(" bps")
        fl_set.addRow("Baud rate", self._sp_baud)

        self._sp_node = QSpinBox()
        self._sp_node.setRange(0, 63)
        self._sp_node.setValue(0)
        fl_set.addRow("Axis 0 node ID (L)", self._sp_node)

        self._sp_enc = QSpinBox()
        self._sp_enc.setRange(0, 10_000)
        self._sp_enc.setSingleStep(5)
        self._sp_enc.setValue(10)
        self._sp_enc.setSuffix(" ms  (0 = off)")
        fl_set.addRow("Encoder stream rate", self._sp_enc)

        self._sp_hb = QSpinBox()
        self._sp_hb.setRange(0, 10_000)
        self._sp_hb.setSingleStep(50)
        self._sp_hb.setValue(100)
        self._sp_hb.setSuffix(" ms  (0 = off)")
        fl_set.addRow("Heartbeat rate", self._sp_hb)

        root.addWidget(grp_set)

        # ── Commands generated ───────────────────────────────────────────
        grp_cmd = QGroupBox("Commands that will be sent")
        cmd_vb = QVBoxLayout(grp_cmd)
        self._lbl_cmds = QLabel()
        self._lbl_cmds.setStyleSheet(
            f"font-family: monospace; font-size: 11px; color: {CLR_MUTED};")
        self._lbl_cmds.setWordWrap(True)
        cmd_vb.addWidget(self._lbl_cmds)
        root.addWidget(grp_cmd)
        self._update_cmd_preview()

        # Update preview whenever spinboxes change
        for sp in (self._sp_baud, self._sp_node, self._sp_enc, self._sp_hb):
            sp.valueChanged.connect(self._update_cmd_preview)

        # ── Apply / Save buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._btn_apply = QPushButton("Apply (no save)")
        self._btn_apply.setToolTip("Write settings to ODrive RAM — lost on reboot")
        self._btn_apply.clicked.connect(self._do_apply)
        btn_row.addWidget(self._btn_apply)

        self._btn_save = QPushButton("Apply && Save")
        self._btn_save.setToolTip(
            "Write settings then call save_configuration().\n"
            "ODrive will reboot (~4 s) and reconnect automatically."
        )
        self._btn_save.setStyleSheet(
            "QPushButton { background: #3a5a3a; } QPushButton:hover { background: #4a7a4a; }"
        )
        self._btn_save.clicked.connect(self._do_apply_and_save)
        btn_row.addWidget(self._btn_save)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # ── Status ──────────────────────────────────────────────────────
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(f"font-family: monospace; color: {CLR_MUTED};")
        root.addWidget(self._lbl_status)

        root.addStretch()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_cmd_preview(self):
        b = self._sp_baud.value()
        n = self._sp_node.value()
        e = self._sp_enc.value()
        h = self._sp_hb.value()
        self._lbl_cmds.setText(
            f"odrv0.can.config.baud_rate = {b}\n"
            f"odrv0.axis0.config.can.node_id = {n}\n"
            f"odrv0.axis1.config.can.node_id = {n + 1}\n"
            f"odrv0.axis0.config.can.encoder_rate_ms = {e}\n"
            f"odrv0.axis1.config.can.encoder_rate_ms = {e}\n"
            f"odrv0.axis0.config.can.heartbeat_rate_ms = {h}\n"
            f"odrv0.axis1.config.can.heartbeat_rate_ms = {h}\n"
            f"odrv0.save_configuration()   ← only with 'Apply & Save'"
        )

    def _set_status(self, msg: str, color: str = CLR_MUTED):
        self._lbl_status.setText(msg)
        self._lbl_status.setStyleSheet(f"font-family: monospace; color: {color};")

    def _set_buttons_enabled(self, enabled: bool):
        self._btn_apply.setEnabled(enabled)
        self._btn_save.setEnabled(enabled)

    # ── Read ─────────────────────────────────────────────────────────────────

    def _do_read(self):
        if not self._mgr.connected:
            self._set_status("Not connected", CLR_ERR)
            return
        self._set_status("Reading…", CLR_INFO)
        self._read_worker = _ReadWorker(self._mgr)
        self._read_worker.done.connect(self._on_read_done)
        self._read_worker.start()

    def _on_read_done(self, result):
        if result is None:
            self._set_status("Read failed", CLR_ERR)
            return
        self._lbl_baud.setText(f"{result['baud']:,} bps")
        self._lbl_node.setText(str(result['node_id']))
        self._lbl_node_r.setText(str(result['node_id_r']))
        self._lbl_enc.setText(
            f"{result['enc_rate']} ms"
            + (f"  ({1000 // result['enc_rate']} Hz)" if result['enc_rate'] > 0 else "  (off)")
        )
        self._lbl_hb.setText(
            f"{result['hb_rate']} ms"
            + (f"  ({1000 // result['hb_rate']} Hz)" if result['hb_rate'] > 0 else "  (off)")
        )
        self._set_status("Read OK", CLR_OK)

        # Pre-fill target spinboxes with current values
        self._sp_baud.setValue(result['baud'])
        self._sp_node.setValue(result['node_id'])
        self._sp_enc.setValue(result['enc_rate'])
        self._sp_hb.setValue(result['hb_rate'])

    # ── Apply ─────────────────────────────────────────────────────────────────

    def _do_apply(self, then_save: bool = False):
        if not self._mgr.connected:
            self._set_status("Not connected — connect to ODrive first", CLR_ERR)
            return
        self._set_status("Applying…", CLR_INFO)
        self._set_buttons_enabled(False)
        self._apply_worker = _ApplyWorker(
            self._mgr,
            self._sp_baud.value(),
            self._sp_node.value(),
            self._sp_enc.value(),
            self._sp_hb.value(),
        )
        if then_save:
            self._apply_worker.done.connect(self._on_apply_done_then_save)
        else:
            self._apply_worker.done.connect(self._on_apply_done)
        self._apply_worker.start()

    def _on_apply_done(self, msg: str, ok: bool):
        self._set_status(msg, CLR_OK if ok else CLR_ERR)
        self._set_buttons_enabled(True)

    def _do_apply_and_save(self):
        self._do_apply(then_save=True)

    def _on_apply_done_then_save(self, msg: str, ok: bool):
        if not ok:
            self._set_status(msg, CLR_ERR)
            self._set_buttons_enabled(True)
            return
        self._set_status("Config written — saving…", CLR_INFO)
        self._save_worker = _SaveWorker(self._mgr)
        self._save_worker.done.connect(self._on_save_done)
        self._save_worker.start()

    def _on_save_done(self, msg: str, ok: bool):
        self._set_status(f"{msg}  (reconnecting in {_REBOOT_WAIT_MS // 1000} s…)", CLR_WARN)
        self._mgr.disconnect()
        self._reboot_timer.start(_REBOOT_WAIT_MS)

    def _after_reboot(self):
        self._set_status("Reconnecting…", CLR_INFO)
        try:
            self._mgr.connect()
            self._set_status("Reconnected — CAN config saved ✓", CLR_OK)
            self._do_read()   # refresh readout
            self.reconnected.emit()
        except Exception as e:
            self._set_status(f"Reconnect failed: {e}", CLR_ERR)
        self._set_buttons_enabled(True)

    # ── Called by main window on connect ─────────────────────────────────────

    def on_connected(self):
        """Auto-read config when ODrive connects."""
        self._do_read()

    def poll_update(self):
        pass  # nothing to poll continuously
