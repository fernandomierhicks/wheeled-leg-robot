"""Deterministic, local-only inspection and control of the robot GUI.

Robot state should normally be consumed through the semantic commands exposed by
``RemoteControlServer``.  This bridge is the completeness fallback: every
operator-relevant Qt widget receives a stable ID, can be inspected, and (when it
is an input) can be invoked without screen-coordinate automation.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTextEdit,
    QTreeWidget,
    QWidget,
)


_RELEVANT_TYPES = (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTextEdit,
    QTreeWidget,
)

_RISK_WORDS = re.compile(
    r"\b(arm|running|run|jump|stand|manual|calibrat|flash|upload|reboot|reset|"
    r"delete|erase|estop|e-stop|motor|torque)\b",
    re.IGNORECASE,
)


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "unnamed"


def _short_text(widget: QWidget) -> str:
    if isinstance(widget, QAbstractButton):
        return widget.text()
    if isinstance(widget, QLabel):
        return widget.text()
    if isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit)):
        return widget.placeholderText() if isinstance(widget, QLineEdit) else ""
    if isinstance(widget, QComboBox):
        return widget.currentText()
    return ""


class GuiOperatorBridge:
    """Inspect and invoke one ``MainWindow`` from its Qt GUI thread."""

    def __init__(self, main_window: QMainWindow):
        self._window = main_window

    def _registry(self) -> dict[str, tuple[QWidget, str]]:
        registry: dict[str, tuple[QWidget, str]] = {}
        claimed: set[int] = set()

        tab_widgets = getattr(self._window, "_tab_widgets", {})
        roots: list[tuple[str, QWidget]] = [
            (f"tab/{_slug(title)}", widget) for title, widget in tab_widgets.items()
        ]
        roots.append(("window", self._window))

        for prefix, root in roots:
            widgets = [root, *root.findChildren(QWidget)]
            class_counts: dict[str, int] = {}
            for widget in widgets:
                identity = id(widget)
                is_custom_widget = not type(widget).__module__.startswith("PyQt6")
                if identity in claimed or (not isinstance(widget, _RELEVANT_TYPES) and not is_custom_widget):
                    continue
                claimed.add(identity)
                class_name = type(widget).__name__
                index = class_counts.get(class_name, 0)
                class_counts[class_name] = index + 1
                object_name = widget.objectName().strip()
                text_hint = _slug(_short_text(widget))[:40]
                suffix = f"{class_name.lower()}-{index}"
                if object_name:
                    suffix += f"-{_slug(object_name)}"
                elif text_hint and not isinstance(widget, QLabel):
                    suffix += f"-{text_hint}"
                operator_id = f"{prefix}/{suffix}"
                widget.setProperty("operatorId", operator_id)
                if not widget.accessibleName():
                    widget.setAccessibleName(operator_id)
                registry[operator_id] = (widget, prefix)
        return registry

    @staticmethod
    def _actions(widget: QWidget) -> list[str]:
        actions: list[str] = []
        if isinstance(widget, QAbstractButton):
            actions.append("click")
            if widget.isCheckable():
                actions.append("set_checked")
        if isinstance(widget, QComboBox):
            actions.extend(("select_index", "select_text"))
        if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            actions.append("set_value")
        if isinstance(widget, QLineEdit):
            actions.extend(("set_text", "submit"))
        if isinstance(widget, QTabWidget):
            actions.extend(("select_index", "select_text"))
        return actions

    @staticmethod
    def _snapshot_widget(operator_id: str, widget: QWidget, group: str) -> dict:
        item: dict = {
            "id": operator_id,
            "group": group,
            "class": type(widget).__name__,
            "object_name": widget.objectName(),
            "accessible_name": widget.accessibleName(),
            "enabled": widget.isEnabled(),
            "visible": widget.isVisible(),
            "actions": GuiOperatorBridge._actions(widget),
            "tooltip": widget.toolTip(),
        }

        if isinstance(widget, QAbstractButton):
            item.update(text=widget.text(), checkable=widget.isCheckable())
            if widget.isCheckable():
                item["checked"] = widget.isChecked()
        elif isinstance(widget, QLabel):
            item["text"] = widget.text()
        elif isinstance(widget, QComboBox):
            item.update(
                value=widget.currentText(),
                index=widget.currentIndex(),
                options=[widget.itemText(i) for i in range(widget.count())],
            )
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            item.update(value=widget.value(), minimum=widget.minimum(), maximum=widget.maximum())
        elif isinstance(widget, QProgressBar):
            item.update(value=widget.value(), minimum=widget.minimum(), maximum=widget.maximum())
        elif isinstance(widget, QLineEdit):
            item.update(value=widget.text(), placeholder=widget.placeholderText())
        elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
            text = widget.toPlainText()
            item.update(read_only=widget.isReadOnly(), text=text[-4000:], text_truncated=len(text) > 4000)
        elif isinstance(widget, QTabWidget):
            item.update(
                index=widget.currentIndex(),
                value=widget.tabText(widget.currentIndex()) if widget.currentIndex() >= 0 else "",
                options=[widget.tabText(i) for i in range(widget.count())],
            )
        elif isinstance(widget, QListWidget):
            item.update(row_count=widget.count(), column_count=1)
        elif isinstance(widget, QTreeWidget):
            item.update(row_count=widget.topLevelItemCount(), column_count=widget.columnCount())
        elif isinstance(widget, QTableWidget):
            item.update(row_count=widget.rowCount(), column_count=widget.columnCount())

        risk_text = " ".join(
            str(item.get(key, "")) for key in ("id", "text", "tooltip", "accessible_name")
        )
        item["safety"] = "operator_acknowledgement" if _RISK_WORDS.search(risk_text) else "routine"
        return item

    def snapshot(self, query: str | None = None) -> list[dict]:
        query_lower = (query or "").strip().lower()
        result = [
            self._snapshot_widget(operator_id, widget, group)
            for operator_id, (widget, group) in self._registry().items()
        ]
        if query_lower:
            result = [item for item in result if query_lower in str(item).lower()]
        return result

    def manifest(self) -> dict:
        widgets = self.snapshot()
        return {
            "schema_version": 1,
            "generated_monotonic_ms": int(time.monotonic() * 1000),
            "widget_count": len(widgets),
            "actionable_count": sum(bool(item["actions"]) for item in widgets),
            "groups": sorted({item["group"] for item in widgets}),
            "widgets": widgets,
        }

    def parity_report(self, semantic_commands: list[str] | None = None) -> dict:
        widgets = self.snapshot()
        missing_ids = [item for item in widgets if not item["id"]]
        missing_accessibility = [item["id"] for item in widgets if not item["accessible_name"]]
        actionable = [item for item in widgets if item["actions"]]
        return {
            "schema_version": 1,
            "coverage_percent": 100.0 if not missing_ids and not missing_accessibility else 0.0,
            "widget_count": len(widgets),
            "actionable_count": len(actionable),
            "semantic_commands": sorted(semantic_commands or []),
            "semantic_command_count": len(semantic_commands or []),
            "fallback_mapped_count": len(widgets),
            "missing_ids": [item.get("class") for item in missing_ids],
            "missing_accessibility": missing_accessibility,
            "unmapped": [],
            "policy": (
                "Semantic API is preferred. Every inventoried widget also has a deterministic "
                "operator ID, accessibility name, readable snapshot, and guarded invocation when actionable."
            ),
        }

    def invoke(self, operator_id: str, action: str, value=None, *, acknowledge_risk=False) -> dict:
        entry = self._registry().get(operator_id)
        if entry is None:
            return {"ok": False, "error": f"unknown widget id {operator_id!r}"}
        widget, group = entry
        before = self._snapshot_widget(operator_id, widget, group)
        if not widget.isEnabled():
            return {"ok": False, "error": "widget is disabled", "widget": before}
        if action not in before["actions"]:
            return {"ok": False, "error": f"action {action!r} is not supported", "widget": before}
        if before["safety"] != "routine" and not acknowledge_risk:
            return {
                "ok": False,
                "error": "operator acknowledgement required",
                "required": "acknowledge_risk=true",
                "widget": before,
            }

        if action == "click" and isinstance(widget, QAbstractButton):
            widget.click()
        elif action == "set_checked" and isinstance(widget, QAbstractButton):
            desired = bool(value)
            if widget.isChecked() != desired:
                widget.click()
        elif action == "select_index" and isinstance(widget, (QComboBox, QTabWidget)):
            widget.setCurrentIndex(int(value))
        elif action == "select_text" and isinstance(widget, QComboBox):
            index = widget.findText(str(value))
            if index < 0:
                return {"ok": False, "error": f"option {value!r} not found", "widget": before}
            widget.setCurrentIndex(index)
        elif action == "select_text" and isinstance(widget, QTabWidget):
            options = [widget.tabText(i) for i in range(widget.count())]
            if str(value) not in options:
                return {"ok": False, "error": f"tab {value!r} not found", "widget": before}
            widget.setCurrentIndex(options.index(str(value)))
        elif action == "set_value" and isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
            widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox) else int(value))
        elif action == "set_text" and isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif action == "submit" and isinstance(widget, QLineEdit):
            if value is not None:
                widget.setText(str(value))
            widget.returnPressed.emit()
        else:
            return {"ok": False, "error": "unsupported widget/action combination", "widget": before}

        QApplication.processEvents()
        return {
            "ok": True,
            "before": before,
            "after": self._snapshot_widget(operator_id, widget, group),
        }

    def select_tab(self, title: str) -> dict:
        widget = getattr(self._window, "_tab_widgets", {}).get(title)
        if widget is None:
            return {"ok": False, "error": f"unknown tab {title!r}"}
        for pane_name in ("_left_pane", "_right_pane"):
            pane = getattr(self._window, pane_name, None)
            if pane is not None and pane.indexOf(widget) >= 0:
                pane.setCurrentWidget(widget)
                self._window.raise_()
                self._window.activateWindow()
                QApplication.processEvents()
                return {"ok": True, "tab": title, "location": pane_name.removeprefix("_")}
        floating = getattr(self._window, "_floating", {}).get(widget)
        if floating is not None:
            floating.raise_()
            floating.activateWindow()
            return {"ok": True, "tab": title, "location": "floating"}
        return {"ok": False, "error": f"tab {title!r} has no active container"}

    def screenshot(self) -> dict:
        screen = self._window.screen() or QApplication.primaryScreen()
        if screen is None:
            return {"ok": False, "error": "no Qt screen is available"}
        out_dir = Path(tempfile.gettempdir()) / "wheeled-leg-robot-operator"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gui-{int(time.time() * 1000)}.png"
        pixmap = screen.grabWindow(int(self._window.winId()))
        if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
            return {"ok": False, "error": "failed to capture GUI screenshot"}
        return {
            "ok": True,
            "path": str(path),
            "width": pixmap.width(),
            "height": pixmap.height(),
        }
