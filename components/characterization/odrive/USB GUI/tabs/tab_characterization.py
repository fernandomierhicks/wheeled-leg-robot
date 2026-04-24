"""tab_characterization.py — Motor characterization tab.

Two workflows:
  1. Scale-based Kt — lock the motor shaft with a lever arm resting on a
     scale, apply a commanded Iq in torque control, read the force from the
     scale, compute Kt = (F · r) / Iq_measured. Average across rows and
     optionally write + save to NVM.
  2. No-load friction sweep — step through a range of velocities and record
     mean Iq at each step to produce an Iq-vs-velocity friction curve.

Both operations run in QThread workers so the GUI stays responsive.
"""

import logging
import time

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QSplitter, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QButtonGroup, QRadioButton,
    QMessageBox,
)

import pyqtgraph as pg

from ui.theme import (
    CLR_OK, CLR_WARN, CLR_ERR, CLR_INFO, CLR_MUTED, CLR_PANEL,
)

log = logging.getLogger("odrive_gui")

# ODrive 0.5.6 enums (values hardcoded to avoid import-order issues)
AXIS_STATE_IDLE = 1
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
CONTROL_MODE_TORQUE_CONTROL = 1
CONTROL_MODE_VELOCITY_CONTROL = 2
INPUT_MODE_PASSTHROUGH = 1

GRAVITY = 9.80665
REBOOT_WAIT_MS = 4000


def _colored(label: QLabel, text: str, color: str):
    label.setText(text)
    label.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {color};")


def _dspin(lo: float, hi: float, val: float, dec: int = 2,
           step: float = 0.1, suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(dec)
    sb.setSingleStep(step)
    sb.setValue(val)
    if suffix:
        sb.setSuffix(f" {suffix}")
    sb.setFixedWidth(110)
    return sb


# ══════════════════════════════════════════════════════════════════════════════
#  Stall Hold Worker — command a constant Iq and sample Iq_measured until stop
# ══════════════════════════════════════════════════════════════════════════════
class StallHoldWorker(QThread):
    progress = Signal(str)
    sample = Signal(float, float)  # t_rel, Iq_measured
    finished_mean = Signal(float, int)  # mean Iq_measured, n_samples
    error = Signal(str)

    def __init__(self, manager, axis_idx, iq_target):
        super().__init__()
        self._mgr = manager
        self._axis_idx = axis_idx
        self._iq_target = float(iq_target)
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        axis = self._mgr.get_axis(self._axis_idx)
        if axis is None:
            self.error.emit("Not connected")
            return
        try:
            self.progress.emit("Returning to Idle...")
            axis.requested_state = AXIS_STATE_IDLE
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:
                if getattr(axis, "current_state", 0) == AXIS_STATE_IDLE:
                    break
                time.sleep(0.05)

            for obj in (axis, axis.motor, axis.encoder, axis.controller):
                try:
                    obj.error = 0
                except Exception:
                    pass

            # Read current torque_constant so we can command a specific Iq via
            # input_torque. Iq_cmd = input_torque / torque_constant, so we
            # command input_torque = Iq_target * Kt_stored.
            try:
                kt_stored = float(axis.motor.config.torque_constant)
            except Exception:
                kt_stored = 0.0
            if kt_stored <= 1e-9:
                self.error.emit(
                    "motor.config.torque_constant is zero — cannot command Iq. "
                    "Write a seed value first (e.g. 0.04)."
                )
                return

            self.progress.emit("Setting torque control...")
            axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
            axis.controller.input_torque = 0.0

            self.progress.emit("Entering closed-loop...")
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                if getattr(axis, "current_state", 0) == AXIS_STATE_CLOSED_LOOP_CONTROL:
                    break
                time.sleep(0.05)
            else:
                self.error.emit(
                    f"Timed out entering closed-loop. "
                    f"axis.error=0x{getattr(axis, 'error', 0):08X} "
                    f"motor.error=0x{getattr(axis.motor, 'error', 0):08X}"
                )
                return

            self.progress.emit(
                f"Holding Iq = {self._iq_target:.3f} A. Read scale, then Stop."
            )
            axis.controller.input_torque = self._iq_target * kt_stored

            # Settle before averaging.
            time.sleep(0.4)

            iq_samples = []
            t_start = time.monotonic()
            while not self._stop_flag:
                if getattr(axis, "error", 0) != 0:
                    self.error.emit(f"Axis error while holding: 0x{axis.error:08X}")
                    break
                try:
                    iq = float(axis.motor.current_control.Iq_measured)
                except Exception:
                    iq = 0.0
                t_rel = time.monotonic() - t_start
                self.sample.emit(t_rel, iq)
                iq_samples.append(iq)
                time.sleep(0.05)

            axis.controller.input_torque = 0.0
            time.sleep(0.2)
            axis.requested_state = AXIS_STATE_IDLE

            if iq_samples:
                mean_iq = sum(iq_samples) / len(iq_samples)
                self.finished_mean.emit(mean_iq, len(iq_samples))
            else:
                self.finished_mean.emit(0.0, 0)

        except Exception as e:
            self.error.emit(f"Stall hold error: {e}")
            try:
                axis.controller.input_torque = 0.0
                axis.requested_state = AXIS_STATE_IDLE
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  Reconnect Worker — used after save_configuration() reboots the board
# ══════════════════════════════════════════════════════════════════════════════
class _ReconnectWorker(QThread):
    success = Signal(object)
    failed = Signal(str)

    def __init__(self, manager):
        super().__init__()
        self._mgr = manager

    def run(self):
        try:
            self._mgr.disconnect()
            odrv = self._mgr.connect()
            self.success.emit(odrv)
        except Exception as e:
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  Friction Sweep Worker
# ══════════════════════════════════════════════════════════════════════════════
class FrictionSweepWorker(QThread):
    progress = Signal(str)
    point = Signal(float, float)  # vel, Iq_mean
    done = Signal()
    error = Signal(str)

    def __init__(self, manager, axis_idx, vel_min, vel_max, steps, dwell_s):
        super().__init__()
        self._mgr = manager
        self._axis_idx = axis_idx
        self._vel_min = vel_min
        self._vel_max = vel_max
        self._steps = steps
        self._dwell_s = dwell_s

    def run(self):
        axis = self._mgr.get_axis(self._axis_idx)
        if axis is None:
            self.error.emit("Not connected")
            return
        try:
            for obj in (axis, axis.motor, axis.encoder, axis.controller):
                try:
                    obj.error = 0
                except Exception:
                    pass

            try:
                if axis.controller.config.vel_limit < self._vel_max * 1.2:
                    axis.controller.config.vel_limit = self._vel_max * 1.5
            except Exception:
                pass

            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
            axis.controller.input_vel = self._vel_min

            self.progress.emit("Entering closed-loop...")
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                if getattr(axis, "current_state", 0) == AXIS_STATE_CLOSED_LOOP_CONTROL:
                    break
                time.sleep(0.05)
            else:
                self.error.emit("Timed out waiting for closed-loop.")
                return

            n = max(2, self._steps)
            vels = [
                self._vel_min + (self._vel_max - self._vel_min) * i / (n - 1)
                for i in range(n)
            ]

            for vel in vels:
                if getattr(axis, "error", 0) != 0:
                    self.error.emit(f"Axis error during sweep: 0x{axis.error:08X}")
                    break
                axis.controller.input_vel = vel
                self.progress.emit(f"vel = {vel:.2f} t/s...")
                time.sleep(self._dwell_s)

                iq_vals = []
                for _ in range(10):
                    try:
                        iq_vals.append(float(axis.motor.current_control.Iq_measured))
                    except Exception:
                        iq_vals.append(0.0)
                    time.sleep(0.05)

                self.point.emit(vel, sum(iq_vals) / len(iq_vals))

            axis.controller.input_vel = 0
            time.sleep(0.3)
            axis.requested_state = AXIS_STATE_IDLE
            self.done.emit()

        except Exception as e:
            self.error.emit(f"Friction sweep error: {e}")
            try:
                axis.requested_state = AXIS_STATE_IDLE
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  Characterization Tab
# ══════════════════════════════════════════════════════════════════════════════
class TabCharacterization(QWidget):
    """Scale-based Kt measurement + no-load friction sweep."""

    # Emitted after save+reboot+reconnect so MainWindow can refresh status bar.
    reconnected = Signal()

    KT_COL_IQ_CMD = 0
    KT_COL_IQ_MEAS = 1
    KT_COL_FORCE = 2
    KT_COL_KT = 3

    def __init__(self, manager, get_axis_idx, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_axis_idx = get_axis_idx

        self._stall_worker: StallHoldWorker | None = None
        self._sweep_worker: FrictionSweepWorker | None = None
        self._reconnect_worker: _ReconnectWorker | None = None
        self._sweep_points: list[tuple[float, float]] = []

        self._hold_t: list[float] = []
        self._hold_iq: list[float] = []
        self._active_row: int = -1
        self._kt_table_loading: bool = False

        self._reboot_timer = QTimer(self)
        self._reboot_timer.setSingleShot(True)
        self._reboot_timer.timeout.connect(self._reconnect_after_reboot)

        self._build()

    # ── UI build ─────────────────────────────────────────────────────────────
    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        inner = QWidget()
        scroll.setWidget(inner)
        left = QVBoxLayout(inner)
        left.setSpacing(10)
        left.setContentsMargins(8, 8, 8, 8)
        left.addWidget(self._build_stall_kt_group())
        left.addWidget(self._build_sweep_group())
        left.addStretch()
        splitter.addWidget(scroll)

        splitter.addWidget(self._build_chart_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_stall_kt_group(self) -> QGroupBox:
        box = QGroupBox("Scale-Based Kt (lever arm + scale)")
        root = QVBoxLayout(box)

        form = QFormLayout()
        form.setSpacing(4)
        self._lever_mm = _dspin(1.0, 500.0, 50.0, dec=1, step=1.0, suffix="mm")
        self._lever_mm.valueChanged.connect(self._recompute_all_kt)
        form.addRow("Lever arm r", self._lever_mm)

        unit_row = QHBoxLayout()
        self._rb_grams = QRadioButton("grams")
        self._rb_newtons = QRadioButton("N")
        self._rb_grams.setChecked(True)
        self._unit_grp = QButtonGroup(self)
        self._unit_grp.addButton(self._rb_grams, 0)
        self._unit_grp.addButton(self._rb_newtons, 1)
        self._unit_grp.buttonClicked.connect(self._on_unit_changed)
        unit_row.addWidget(QLabel("Force unit:"))
        unit_row.addWidget(self._rb_grams)
        unit_row.addWidget(self._rb_newtons)
        unit_row.addStretch()
        form.addRow(unit_row)
        root.addLayout(form)

        self._kt_table = QTableWidget(0, 4)
        self._kt_table.setHorizontalHeaderLabels(
            ["Iq cmd (A)", "Iq meas (A)", "Force (g)", "Kt (Nm/A)"]
        )
        self._kt_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._kt_table.setMaximumHeight(180)
        self._kt_table.itemChanged.connect(self._on_kt_item_changed)
        root.addWidget(self._kt_table)

        row_btns = QHBoxLayout()
        btn_add = QPushButton("+ Row")
        btn_add.clicked.connect(self._kt_add_row)
        btn_del = QPushButton("− Row")
        btn_del.clicked.connect(self._kt_remove_row)
        row_btns.addWidget(btn_add)
        row_btns.addWidget(btn_del)
        row_btns.addStretch()
        root.addLayout(row_btns)

        hold_row = QHBoxLayout()
        self._btn_hold = QPushButton("Apply selected row")
        self._btn_hold.clicked.connect(self._start_stall_hold)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_stall_hold)
        hold_row.addWidget(self._btn_hold)
        hold_row.addWidget(self._btn_stop)
        root.addLayout(hold_row)

        self._lbl_stall_status = QLabel("Idle")
        self._lbl_stall_status.setStyleSheet(
            f"color: {CLR_MUTED}; font-family: monospace;")
        root.addWidget(self._lbl_stall_status)

        self._lbl_kt_mean = QLabel("Mean Kt: —")
        self._lbl_kt_mean.setStyleSheet(
            f"font-family: monospace; font-size: 13px; color: {CLR_INFO};")
        root.addWidget(self._lbl_kt_mean)

        self._btn_save_kt = QPushButton("Write & Save Kt → ODrive (NVM)")
        self._btn_save_kt.setEnabled(False)
        self._btn_save_kt.clicked.connect(self._on_save_kt)
        root.addWidget(self._btn_save_kt)

        self._kt_add_row()
        return box

    def _build_sweep_group(self) -> QGroupBox:
        box = QGroupBox("No-Load Friction Sweep (Iq vs velocity)")
        root = QVBoxLayout(box)

        form = QFormLayout()
        form.setSpacing(4)
        self._sw_vel_min = _dspin(0.1, 10.0, 0.5, dec=1, step=0.5, suffix="t/s")
        self._sw_vel_max = _dspin(1.0, 30.0, 10.0, dec=1, step=1.0, suffix="t/s")
        self._sw_steps = QSpinBox()
        self._sw_steps.setRange(2, 30)
        self._sw_steps.setValue(10)
        self._sw_steps.setFixedWidth(110)
        self._sw_dwell = _dspin(0.5, 10.0, 1.5, dec=1, step=0.5, suffix="s")
        form.addRow("Vel min", self._sw_vel_min)
        form.addRow("Vel max", self._sw_vel_max)
        form.addRow("Steps", self._sw_steps)
        form.addRow("Dwell/step", self._sw_dwell)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_sweep_run = QPushButton("Run Friction Sweep")
        self._btn_sweep_run.clicked.connect(self._start_friction_sweep)
        self._lbl_sweep_status = QLabel("Idle")
        self._lbl_sweep_status.setStyleSheet(
            f"color: {CLR_MUTED}; font-family: monospace;")
        btn_row.addWidget(self._btn_sweep_run)
        btn_row.addWidget(self._lbl_sweep_status, stretch=1)
        root.addLayout(btn_row)

        self._sweep_table = QTableWidget(0, 2)
        self._sweep_table.setHorizontalHeaderLabels(["Vel (t/s)", "Iq_mean (A)"])
        self._sweep_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._sweep_table.setMaximumHeight(160)
        self._sweep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self._sweep_table)

        return box

    def _build_chart_panel(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(4)

        pg.setConfigOptions(antialias=False)

        self._gfx = pg.GraphicsLayoutWidget()
        self._gfx.setBackground(CLR_PANEL)
        col.addWidget(self._gfx, stretch=1)

        self._plot_hold = self._gfx.addPlot()
        self._plot_hold.setLabel("left", "Iq measured (A)")
        self._plot_hold.setLabel("bottom", "t (s) — stall hold")
        self._plot_hold.showGrid(x=True, y=True, alpha=0.15)
        self._curve_hold_iq = self._plot_hold.plot(
            pen=pg.mkPen(color=CLR_INFO, width=2))

        self._gfx.nextRow()
        self._plot_sweep = self._gfx.addPlot()
        self._plot_sweep.setLabel("left", "Iq (A)")
        self._plot_sweep.setLabel("bottom", "vel (t/s)")
        self._plot_sweep.showGrid(x=True, y=True, alpha=0.15)
        self._curve_friction = self._plot_sweep.plot(
            pen=pg.mkPen(color=CLR_ERR, width=2),
            symbol="o", symbolBrush=CLR_ERR, symbolSize=6,
        )

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_clear = QPushButton("Clear plots")
        btn_clear.setFixedWidth(110)
        btn_clear.clicked.connect(self._clear_plots)
        btn_row.addWidget(btn_clear)
        col.addLayout(btn_row)

        return panel

    def _clear_plots(self):
        self._hold_t.clear()
        self._hold_iq.clear()
        self._curve_hold_iq.setData([], [])
        self._curve_friction.setData([], [])

    # ── Kt table helpers ─────────────────────────────────────────────────────
    def _kt_add_row(self):
        self._kt_table_loading = True
        row = self._kt_table.rowCount()
        self._kt_table.insertRow(row)
        # Seed defaults: 1 A, blank meas, blank force, blank Kt
        default_iq = 1.0 * (row + 1)
        self._kt_table.setItem(
            row, self.KT_COL_IQ_CMD, QTableWidgetItem(f"{default_iq:.2f}"))

        meas_item = QTableWidgetItem("")
        meas_item.setFlags(meas_item.flags() & ~Qt.ItemIsEditable)
        self._kt_table.setItem(row, self.KT_COL_IQ_MEAS, meas_item)

        self._kt_table.setItem(row, self.KT_COL_FORCE, QTableWidgetItem(""))

        kt_item = QTableWidgetItem("")
        kt_item.setFlags(kt_item.flags() & ~Qt.ItemIsEditable)
        self._kt_table.setItem(row, self.KT_COL_KT, kt_item)

        self._kt_table_loading = False
        self._kt_table.selectRow(row)

    def _kt_remove_row(self):
        row = self._kt_table.currentRow()
        if row < 0:
            row = self._kt_table.rowCount() - 1
        if row < 0:
            return
        self._kt_table.removeRow(row)
        self._recompute_mean_kt()

    def _on_unit_changed(self, *_):
        header = "Force (g)" if self._rb_grams.isChecked() else "Force (N)"
        self._kt_table.setHorizontalHeaderItem(
            self.KT_COL_FORCE, QTableWidgetItem(header))
        self._recompute_all_kt()

    def _on_kt_item_changed(self, item: QTableWidgetItem):
        if self._kt_table_loading:
            return
        if item.column() in (self.KT_COL_IQ_CMD, self.KT_COL_FORCE):
            self._recompute_row_kt(item.row())
            self._recompute_mean_kt()

    def _row_value(self, row: int, col: int) -> float | None:
        it = self._kt_table.item(row, col)
        if it is None:
            return None
        txt = it.text().strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            return None

    def _row_kt(self, row: int) -> float | None:
        iq_meas = self._row_value(row, self.KT_COL_IQ_MEAS)
        iq_cmd = self._row_value(row, self.KT_COL_IQ_CMD)
        iq = iq_meas if iq_meas is not None else iq_cmd
        force = self._row_value(row, self.KT_COL_FORCE)
        if iq is None or force is None or abs(iq) < 1e-6:
            return None
        # Convert force → Newtons
        if self._rb_grams.isChecked():
            f_n = force * 1e-3 * GRAVITY
        else:
            f_n = force
        r_m = self._lever_mm.value() * 1e-3
        torque = f_n * r_m
        return torque / iq

    def _recompute_row_kt(self, row: int):
        kt = self._row_kt(row)
        self._kt_table_loading = True
        item = self._kt_table.item(row, self.KT_COL_KT)
        if item is None:
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._kt_table.setItem(row, self.KT_COL_KT, item)
        item.setText(f"{kt:.5f}" if kt is not None else "")
        self._kt_table_loading = False

    def _recompute_all_kt(self):
        for row in range(self._kt_table.rowCount()):
            self._recompute_row_kt(row)
        self._recompute_mean_kt()

    def _collect_kts(self) -> list[float]:
        kts = []
        for row in range(self._kt_table.rowCount()):
            kt = self._row_kt(row)
            if kt is not None and kt > 0:
                kts.append(kt)
        return kts

    def _recompute_mean_kt(self):
        kts = self._collect_kts()
        if not kts:
            self._lbl_kt_mean.setText("Mean Kt: —")
            self._btn_save_kt.setEnabled(False)
            return
        mean_kt = sum(kts) / len(kts)
        std_kt = float(np.std(kts)) if len(kts) > 1 else 0.0
        self._lbl_kt_mean.setText(
            f"Mean Kt: {mean_kt:.5f} Nm/A  (n={len(kts)}, σ={std_kt:.5f})"
        )
        self._btn_save_kt.setEnabled(self._mgr.connected)

    # ── Stall hold ───────────────────────────────────────────────────────────
    def _start_stall_hold(self):
        if not self._mgr.connected:
            _colored(self._lbl_stall_status, "Not connected", CLR_WARN)
            return
        if self._stall_worker and self._stall_worker.isRunning():
            return

        row = self._kt_table.currentRow()
        if row < 0:
            _colored(self._lbl_stall_status, "Select a row first", CLR_WARN)
            return
        iq_cmd = self._row_value(row, self.KT_COL_IQ_CMD)
        if iq_cmd is None or abs(iq_cmd) < 1e-6:
            _colored(self._lbl_stall_status, "Row has no Iq cmd", CLR_WARN)
            return

        self._active_row = row
        self._hold_t.clear()
        self._hold_iq.clear()
        self._curve_hold_iq.setData([], [])

        self._btn_hold.setEnabled(False)
        self._btn_stop.setEnabled(True)
        _colored(self._lbl_stall_status, "Running...", CLR_INFO)

        self._stall_worker = StallHoldWorker(
            self._mgr, self._get_axis_idx(), iq_cmd)
        self._stall_worker.progress.connect(
            lambda m: _colored(self._lbl_stall_status, m, CLR_INFO))
        self._stall_worker.sample.connect(self._on_hold_sample)
        self._stall_worker.finished_mean.connect(self._on_hold_finished)
        self._stall_worker.error.connect(self._on_hold_error)
        self._stall_worker.start()
        log.info("Characterization: stall hold started row=%d Iq_cmd=%.3f",
                 row, iq_cmd)

    def _stop_stall_hold(self):
        if self._stall_worker and self._stall_worker.isRunning():
            self._stall_worker.stop()
            _colored(self._lbl_stall_status, "Stopping...", CLR_WARN)

    def _on_hold_sample(self, t_rel: float, iq: float):
        self._hold_t.append(t_rel)
        self._hold_iq.append(iq)
        self._curve_hold_iq.setData(self._hold_t, self._hold_iq)

    def _on_hold_finished(self, mean_iq: float, n: int):
        self._btn_hold.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if self._active_row >= 0 and n > 0:
            self._kt_table_loading = True
            meas_item = self._kt_table.item(self._active_row, self.KT_COL_IQ_MEAS)
            if meas_item is None:
                meas_item = QTableWidgetItem("")
                meas_item.setFlags(meas_item.flags() & ~Qt.ItemIsEditable)
                self._kt_table.setItem(
                    self._active_row, self.KT_COL_IQ_MEAS, meas_item)
            meas_item.setText(f"{mean_iq:.4f}")
            self._kt_table_loading = False
            self._recompute_row_kt(self._active_row)
            self._recompute_mean_kt()
            _colored(
                self._lbl_stall_status,
                f"Done — mean Iq_meas = {mean_iq:.4f} A (n={n}). Enter force in row.",
                CLR_OK,
            )
            log.info(
                "Characterization: stall hold row=%d mean Iq_meas=%.4f (n=%d)",
                self._active_row, mean_iq, n,
            )
        else:
            _colored(self._lbl_stall_status, "Stopped — no samples", CLR_WARN)
        self._active_row = -1

    def _on_hold_error(self, msg: str):
        self._btn_hold.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._active_row = -1
        _colored(self._lbl_stall_status, msg, CLR_ERR)
        log.error("Characterization: stall hold error: %s", msg)

    # ── Save Kt to NVM ───────────────────────────────────────────────────────
    def _on_save_kt(self):
        if not self._mgr.connected:
            _colored(self._lbl_stall_status, "Not connected", CLR_WARN)
            return
        kts = self._collect_kts()
        if not kts:
            _colored(self._lbl_stall_status, "No valid rows to average", CLR_WARN)
            return
        mean_kt = sum(kts) / len(kts)
        ax = self._get_axis_idx()

        reply = QMessageBox.question(
            self,
            "Write & Save Kt",
            f"Write Kt = {mean_kt:.5f} Nm/A to axis{ax}.motor.config.torque_constant\n"
            f"and save_configuration() (ODrive will reboot).\n\n"
            f"Averaged over n={len(kts)} rows. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        ok = self._mgr.safe_set(
            f"axis{ax}.motor.config.torque_constant", mean_kt)
        if not ok:
            _colored(self._lbl_stall_status, "Kt write failed", CLR_ERR)
            return

        _colored(self._lbl_stall_status, "Saving + rebooting...", CLR_WARN)
        self._btn_save_kt.setEnabled(False)
        self._btn_hold.setEnabled(False)
        try:
            self._mgr.odrv.save_configuration()
        except Exception:
            pass  # save_configuration triggers reboot — USB exception expected
        self._mgr.disconnect()
        self._reboot_timer.start(REBOOT_WAIT_MS)
        log.info("Characterization: wrote Kt=%.5f to axis%d and saved",
                 mean_kt, ax)

    def _reconnect_after_reboot(self):
        _colored(self._lbl_stall_status, "Reconnecting after reboot...", CLR_INFO)
        self._reconnect_worker = _ReconnectWorker(self._mgr)
        self._reconnect_worker.success.connect(self._on_reconnect_ok)
        self._reconnect_worker.failed.connect(self._on_reconnect_fail)
        self._reconnect_worker.start()

    def _on_reconnect_ok(self, _odrv):
        _colored(self._lbl_stall_status, "Kt saved to NVM + reconnected", CLR_OK)
        self._btn_hold.setEnabled(True)
        self._recompute_mean_kt()
        log.info("Characterization: reconnected after save+reboot")
        self.reconnected.emit()

    def _on_reconnect_fail(self, msg: str):
        _colored(self._lbl_stall_status, f"Reconnect failed: {msg}", CLR_ERR)
        self._btn_hold.setEnabled(True)
        log.error("Characterization: reconnect after save failed: %s", msg)

    # ── Friction sweep ───────────────────────────────────────────────────────
    def _start_friction_sweep(self):
        if not self._mgr.connected:
            _colored(self._lbl_sweep_status, "Not connected", CLR_WARN)
            return
        if self._sweep_worker and self._sweep_worker.isRunning():
            return
        if self._sw_vel_max.value() <= self._sw_vel_min.value():
            _colored(self._lbl_sweep_status, "vel_max must exceed vel_min", CLR_ERR)
            return

        self._sweep_points.clear()
        self._sweep_table.setRowCount(0)
        self._curve_friction.setData([], [])

        self._btn_sweep_run.setEnabled(False)
        _colored(self._lbl_sweep_status, "Running...", CLR_INFO)

        self._sweep_worker = FrictionSweepWorker(
            self._mgr, self._get_axis_idx(),
            self._sw_vel_min.value(), self._sw_vel_max.value(),
            self._sw_steps.value(), self._sw_dwell.value(),
        )
        self._sweep_worker.progress.connect(
            lambda m: _colored(self._lbl_sweep_status, m, CLR_INFO))
        self._sweep_worker.point.connect(self._on_sweep_point)
        self._sweep_worker.done.connect(self._on_sweep_done)
        self._sweep_worker.error.connect(self._on_sweep_error)
        self._sweep_worker.start()
        log.info("Characterization: friction sweep started")

    def _on_sweep_point(self, vel: float, iq_mean: float):
        self._sweep_points.append((vel, iq_mean))
        row = self._sweep_table.rowCount()
        self._sweep_table.insertRow(row)
        self._sweep_table.setItem(row, 0, QTableWidgetItem(f"{vel:.3f}"))
        self._sweep_table.setItem(row, 1, QTableWidgetItem(f"{iq_mean:.4f}"))
        vels = np.array([p[0] for p in self._sweep_points])
        iqs = np.array([p[1] for p in self._sweep_points])
        self._curve_friction.setData(vels, iqs)

    def _on_sweep_done(self):
        self._btn_sweep_run.setEnabled(True)
        _colored(self._lbl_sweep_status, "Done", CLR_OK)
        log.info(
            "Characterization: friction sweep done (%d points)",
            len(self._sweep_points),
        )

    def _on_sweep_error(self, msg: str):
        self._btn_sweep_run.setEnabled(True)
        _colored(self._lbl_sweep_status, "Error", CLR_ERR)
        log.error("Characterization: sweep error: %s", msg)

    # ── Poll hook (no-op; workers push data themselves) ──────────────────────
    def poll_update(self):
        pass
