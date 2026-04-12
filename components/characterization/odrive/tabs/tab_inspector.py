"""tab_inspector.py — Recursive property tree browser for ODrive 3.6 fw 0.5.6.

Features:
  - Probe button walks the full device attribute tree in a background thread
  - QTreeWidget with Path / Value / Type columns
  - Search bar filters the tree in real-time
  - Double-click a value cell to edit writable properties
  - Firmware version + serial displayed at top
"""

import logging
import threading

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTreeWidget, QTreeWidgetItem, QProgressBar,
    QMessageBox, QHeaderView,
)

from core.odrive_manager import ODriveManager
from ui.theme import CLR_OK, CLR_ERR, CLR_INFO, CLR_MUTED, CLR_WARN

log = logging.getLogger("odrive_gui")

# Column indices
COL_PATH = 0
COL_VALUE = 1
COL_TYPE = 2

# Colors for tree items
_CLR_LEAF = QColor("#dcdcdc")       # normal value
_CLR_CALLABLE = QColor("#888888")   # functions
_CLR_ERROR = QColor("#ff6464")      # read errors
_CLR_WRITABLE = QColor("#64c8ff")   # editable values (config paths)


# ── Background tree-walk worker ──────────────────────────────────────────────

class _ProbeWorker(QThread):
    """Walk the ODrive attribute tree in background, emitting nodes as found."""

    progress = Signal(int)           # approximate node count so far
    node_found = Signal(dict)        # {"path": str, "value": str, "type": str,
                                     #  "is_leaf": bool, "parent_path": str,
                                     #  "writable": bool, "callable": bool}
    finished = Signal(int)           # total node count
    error = Signal(str)

    MAX_DEPTH = 8

    def __init__(self, manager: ODriveManager):
        super().__init__()
        self._mgr = manager
        self._count = 0
        self._visited = set()

    def run(self):
        odrv = self._mgr.odrv
        if odrv is None:
            self.error.emit("Not connected")
            return
        try:
            self._walk(odrv, "odrv0", "", depth=0)
            self.finished.emit(self._count)
        except Exception as e:
            self.error.emit(str(e))

    def _walk(self, obj, path: str, parent_path: str, depth: int):
        obj_id = id(obj)
        if obj_id in self._visited:
            return
        self._visited.add(obj_id)

        try:
            attrs = sorted(a for a in dir(obj) if not a.startswith("_"))
        except Exception:
            return

        for attr_name in attrs:
            full_path = f"{path}.{attr_name}"
            try:
                val = getattr(obj, attr_name)
            except Exception as e:
                self._emit_leaf(full_path, path, f"ERR({e})", "error",
                                writable=False, callable_=False)
                continue

            if isinstance(val, (int, float, str, bool, bytes, type(None))):
                type_name = type(val).__name__
                writable = ".config." in full_path
                self._emit_leaf(full_path, path, repr(val), type_name,
                                writable=writable, callable_=False)
                continue

            # Check for sub-attributes (compound object)
            try:
                sub_attrs = [x for x in dir(val) if not x.startswith("_")]
            except Exception:
                sub_attrs = []

            is_compound = bool(sub_attrs) and not isinstance(val, type)

            if callable(val) and not is_compound:
                self._emit_leaf(full_path, path, "()", "callable",
                                writable=False, callable_=True)
            elif is_compound and depth < self.MAX_DEPTH:
                # Emit the branch node
                self.node_found.emit({
                    "path": full_path,
                    "value": "",
                    "type": f"object ({len(sub_attrs)} attrs)",
                    "is_leaf": False,
                    "parent_path": path,
                    "writable": False,
                    "callable": callable(val),
                })
                self._count += 1
                if self._count % 20 == 0:
                    self.progress.emit(self._count)
                self._walk(val, full_path, path, depth + 1)
            else:
                self._emit_leaf(full_path, path, repr(val), type(val).__name__,
                                writable=False, callable_=False)

    def _emit_leaf(self, path, parent_path, value_str, type_name,
                   writable, callable_):
        self.node_found.emit({
            "path": path,
            "value": value_str,
            "type": type_name,
            "is_leaf": True,
            "parent_path": parent_path,
            "writable": writable,
            "callable": callable_,
        })
        self._count += 1
        if self._count % 20 == 0:
            self.progress.emit(self._count)


# ── Inspector Tab ────────────────────────────────────────────────────────────

class TabInspector(QWidget):
    """Property tree browser with search, inline editing, and firmware display."""

    def __init__(self, manager: ODriveManager, ax_idx_fn, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._ax_idx = ax_idx_fn
        self._worker: _ProbeWorker | None = None
        self._tree_items: dict[str, QTreeWidgetItem] = {}  # path -> item
        self._all_paths: list[str] = []  # for search filtering
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Firmware info row ─────────────────────────────────────────────────
        fw_row = QHBoxLayout()

        self._lbl_fw = QLabel("Firmware: —")
        self._lbl_fw.setStyleSheet(
            f"font-family: monospace; font-size: 12px; color: {CLR_MUTED};")
        fw_row.addWidget(self._lbl_fw)

        fw_row.addStretch()

        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet(
            f"font-family: monospace; font-size: 11px; color: {CLR_MUTED};")
        fw_row.addWidget(self._lbl_count)

        layout.addLayout(fw_row)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()

        self._btn_probe = QPushButton("Probe Device")
        self._btn_probe.setFixedWidth(120)
        self._btn_probe.clicked.connect(self._start_probe)
        ctrl_row.addWidget(self._btn_probe)

        self._btn_expand = QPushButton("Expand All")
        self._btn_expand.setFixedWidth(90)
        self._btn_expand.clicked.connect(lambda: self._tree.expandAll())
        ctrl_row.addWidget(self._btn_expand)

        self._btn_collapse = QPushButton("Collapse All")
        self._btn_collapse.setFixedWidth(90)
        self._btn_collapse.clicked.connect(lambda: self._tree.collapseAll())
        ctrl_row.addWidget(self._btn_collapse)

        ctrl_row.addSpacing(20)

        ctrl_row.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("e.g. vel_gain, motor.config, encoder.error")
        self._search.textChanged.connect(self._apply_filter)
        ctrl_row.addWidget(self._search)

        layout.addLayout(ctrl_row)

        # ── Progress bar ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Probing... %v properties found")
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        layout.addWidget(self._progress)

        # ── Tree widget ───────────────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Path", "Value", "Type"])
        self._tree.setColumnCount(3)
        self._tree.setFont(QFont("Consolas", 10))
        self._tree.setAlternatingRowColors(True)
        self._tree.setStyleSheet(
            "QTreeWidget { background: #1e1e1e; alternate-background-color: #252526; "
            "color: #dcdcdc; border: 1px solid #444; }"
            "QTreeWidget::item:selected { background: #264f78; }"
            "QHeaderView::section { background: #3c3c3c; color: #ccc; "
            "border: 1px solid #444; padding: 3px; }"
        )
        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._tree.setColumnWidth(0, 360)
        self._tree.setColumnWidth(1, 220)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._tree)

        # ── Hint ──────────────────────────────────────────────────────────────
        hint = QLabel("Double-click a value to edit writable properties (config paths).  "
                      "Blue = writable, gray = callable/function.")
        hint.setStyleSheet(f"color: {CLR_MUTED}; font-size: 11px;")
        layout.addWidget(hint)

    # ── Probe ─────────────────────────────────────────────────────────────────

    def _start_probe(self):
        if not self._mgr.connected:
            QMessageBox.warning(self, "Not Connected",
                                "Connect to an ODrive first.")
            return

        if self._worker is not None and self._worker.isRunning():
            return

        # Show firmware info
        info = self._mgr.device_info()
        if info:
            fw_tag = f"fw {info['fw']}"
            if info["fw_unreleased"]:
                fw_tag += " (dev)"
            self._lbl_fw.setText(
                f"Firmware: hw {info['hw']}  {fw_tag}  serial {info['serial']}")
            self._lbl_fw.setStyleSheet(
                f"font-family: monospace; font-size: 12px; color: {CLR_OK};")

        # Clear previous
        self._tree.clear()
        self._tree_items.clear()
        self._all_paths.clear()
        self._search.clear()
        self._lbl_count.setText("")
        self._progress.setValue(0)
        self._progress.show()
        self._btn_probe.setEnabled(False)
        self._btn_probe.setText("Probing...")

        log.info("INSPECTOR: starting property tree probe")

        self._worker = _ProbeWorker(self._mgr)
        self._worker.progress.connect(self._on_progress)
        self._worker.node_found.connect(self._on_node_found)
        self._worker.finished.connect(self._on_probe_done)
        self._worker.error.connect(self._on_probe_error)
        self._worker.start()

    def _on_progress(self, count: int):
        self._progress.setValue(count)

    def _on_node_found(self, node: dict):
        path = node["path"]
        parent_path = node["parent_path"]
        self._all_paths.append(path)

        # Determine display name (last segment of path)
        name = path.rsplit(".", 1)[-1] if "." in path else path

        item = QTreeWidgetItem()
        item.setText(COL_PATH, name)
        item.setData(COL_PATH, Qt.ItemDataRole.UserRole, path)  # full path
        item.setText(COL_VALUE, node["value"])
        item.setText(COL_TYPE, node["type"])
        item.setData(COL_VALUE, Qt.ItemDataRole.UserRole, node["writable"])

        # Color coding
        if node["type"] == "error":
            item.setForeground(COL_VALUE, QBrush(_CLR_ERROR))
        elif node["callable"]:
            item.setForeground(COL_VALUE, QBrush(_CLR_CALLABLE))
            item.setForeground(COL_PATH, QBrush(_CLR_CALLABLE))
        elif node["writable"]:
            item.setForeground(COL_VALUE, QBrush(_CLR_WRITABLE))

        item.setToolTip(COL_PATH, path)

        # Insert into tree
        parent_item = self._tree_items.get(parent_path)
        if parent_item is not None:
            parent_item.addChild(item)
        else:
            self._tree.addTopLevelItem(item)

        self._tree_items[path] = item

    def _on_probe_done(self, total: int):
        self._progress.hide()
        self._btn_probe.setEnabled(True)
        self._btn_probe.setText("Probe Device")
        self._lbl_count.setText(f"{total} properties found")
        log.info("INSPECTOR: probe complete — %d properties", total)

        # Auto-expand first two levels for usability
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            top.setExpanded(True)
            for j in range(top.childCount()):
                top.child(j).setExpanded(True)

    def _on_probe_error(self, msg: str):
        self._progress.hide()
        self._btn_probe.setEnabled(True)
        self._btn_probe.setText("Probe Device")
        self._lbl_count.setText(f"Probe failed: {msg}")
        self._lbl_count.setStyleSheet(
            f"font-family: monospace; font-size: 11px; color: {CLR_ERR};")
        log.error("INSPECTOR: probe failed — %s", msg)

    # ── Search / Filter ───────────────────────────────────────────────────────

    def _apply_filter(self, text: str):
        """Show/hide tree items based on search text."""
        query = text.strip().lower()
        if not query:
            # Show everything
            self._set_all_visible(True)
            return

        # First hide everything, then show matches + their ancestors
        self._set_all_visible(False)

        for path, item in self._tree_items.items():
            if query in path.lower():
                # Show this item and all ancestors
                self._show_with_ancestors(item)

    def _set_all_visible(self, visible: bool):
        for item in self._tree_items.values():
            item.setHidden(not visible)

    def _show_with_ancestors(self, item: QTreeWidgetItem):
        """Unhide the item and all its parent items."""
        item.setHidden(False)
        parent = item.parent()
        while parent is not None:
            parent.setHidden(False)
            parent.setExpanded(True)
            parent = parent.parent()

    # ── Inline Editing ────────────────────────────────────────────────────────

    def _on_double_click(self, item: QTreeWidgetItem, column: int):
        if column != COL_VALUE:
            return

        full_path = item.data(COL_PATH, Qt.ItemDataRole.UserRole)
        writable = item.data(COL_VALUE, Qt.ItemDataRole.UserRole)
        current_value = item.text(COL_VALUE)

        if not full_path or not current_value:
            return

        # Strip the "odrv0." prefix for safe_set
        if full_path.startswith("odrv0."):
            attr_path = full_path[len("odrv0."):]
        else:
            return

        if not writable:
            log.info("INSPECTOR: %s is read-only", full_path)
            return

        # Make the cell editable temporarily
        old_flags = item.flags()
        item.setFlags(old_flags | Qt.ItemFlag.ItemIsEditable)
        self._tree.editItem(item, COL_VALUE)

        # Connect to catch when editing finishes
        # We use a one-shot connection via itemChanged
        self._editing_item = item
        self._editing_path = attr_path
        self._editing_old = current_value
        self._editing_old_flags = old_flags
        self._tree.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        # Disconnect immediately to avoid re-entry
        try:
            self._tree.itemChanged.disconnect(self._on_item_changed)
        except RuntimeError:
            return

        if item is not self._editing_item or column != COL_VALUE:
            item.setFlags(self._editing_old_flags)
            return

        new_text = item.text(COL_VALUE).strip()
        attr_path = self._editing_path
        old_text = self._editing_old

        # Restore flags
        item.setFlags(self._editing_old_flags)

        if new_text == old_text:
            return

        # Parse the new value
        try:
            value = _parse_value(new_text)
        except ValueError as e:
            item.setText(COL_VALUE, old_text)
            log.warning("INSPECTOR: invalid value for %s: %s", attr_path, e)
            return

        # Try to write
        ok = self._mgr.safe_set(attr_path, value)
        if ok:
            # Read back to confirm
            readback = self._mgr.safe_get(attr_path)
            item.setText(COL_VALUE, repr(readback))
            item.setForeground(COL_VALUE, QBrush(QColor(CLR_OK)))
            log.info("INSPECTOR: SET %s = %s (readback: %s)",
                     attr_path, value, readback)
        else:
            item.setText(COL_VALUE, old_text)
            item.setForeground(COL_VALUE, QBrush(_CLR_ERROR))
            log.error("INSPECTOR: failed to set %s = %s", attr_path, new_text)


def _parse_value(text: str):
    """Parse a user-entered string into a Python value."""
    # Strip repr quotes if present
    stripped = text.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() == "none":
        return None
    # Try int
    try:
        return int(stripped, 0)  # supports hex 0x...
    except (ValueError, TypeError):
        pass
    # Try float
    try:
        return float(stripped)
    except (ValueError, TypeError):
        pass
    # Return as string (strip quotes if user typed them)
    if len(stripped) >= 2 and stripped[0] in ('"', "'") and stripped[-1] == stripped[0]:
        return stripped[1:-1]
    raise ValueError(f"Cannot parse: {text}")
