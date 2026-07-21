import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tabs.operator_bridge import GuiOperatorBridge

_APP = QApplication.instance() or QApplication([])


def _window():
    app = _APP
    window = QMainWindow()
    pane = QTabWidget(window)
    page = QWidget()
    layout = QVBoxLayout(page)
    status = QLabel("Connected")
    arm = QPushButton("Arm")
    routine = QPushButton("Refresh")
    combo = QComboBox()
    combo.addItems(["Auto", "WiFi"])
    line = QLineEdit()
    spin = QSpinBox()
    spin.setRange(0, 10)
    for widget in (status, arm, routine, combo, line, spin):
        layout.addWidget(widget)
    pane.addTab(page, "Dashboard")
    window.setCentralWidget(pane)
    window._left_pane = pane
    window._right_pane = None
    window._floating = {}
    window._tab_widgets = {"Dashboard": page}
    window.show()
    app.processEvents()
    return app, window, arm, routine, combo, line, spin


def _find(snapshot, *, text=None, class_name=None):
    for item in snapshot:
        if text is not None and item.get("text") != text:
            continue
        if class_name is not None and item.get("class") != class_name:
            continue
        return item
    raise AssertionError(f"widget not found: text={text!r}, class={class_name!r}")


class OperatorBridgeTests(unittest.TestCase):
    def test_manifest_is_stable_and_covers_inputs_and_outputs(self):
        app, window, *_ = _window()
        bridge = GuiOperatorBridge(window)
        first = bridge.manifest()
        second = bridge.manifest()

        self.assertGreaterEqual(first["widget_count"], 7)
        self.assertGreaterEqual(first["actionable_count"], 5)
        self.assertEqual(
            [item["id"] for item in first["widgets"]],
            [item["id"] for item in second["widgets"]],
        )
        self.assertEqual(_find(first["widgets"], text="Connected")["actions"], [])
        self.assertEqual(_find(first["widgets"], text="Arm")["safety"], "operator_acknowledgement")
        parity = bridge.parity_report(["health", "ui_snapshot"])
        self.assertEqual(parity["coverage_percent"], 100.0)
        self.assertEqual(parity["unmapped"], [])

        window.close()
        app.processEvents()

    def test_invoke_requires_risk_ack_and_updates_supported_controls(self):
        app, window, arm, routine, combo, line, spin = _window()
        bridge = GuiOperatorBridge(window)
        snapshot = bridge.snapshot()
        arm_id = _find(snapshot, text="Arm")["id"]
        routine_id = _find(snapshot, text="Refresh")["id"]
        combo_id = _find(snapshot, class_name="QComboBox")["id"]
        line_id = _find(snapshot, class_name="QLineEdit")["id"]
        spin_id = _find(snapshot, class_name="QSpinBox")["id"]

        hits = []
        arm.clicked.connect(lambda: hits.append("arm"))
        routine.clicked.connect(lambda: hits.append("refresh"))

        denied = bridge.invoke(arm_id, "click")
        self.assertFalse(denied["ok"])
        self.assertEqual(hits, [])
        self.assertTrue(bridge.invoke(arm_id, "click", acknowledge_risk=True)["ok"])
        self.assertTrue(bridge.invoke(routine_id, "click")["ok"])
        self.assertEqual(hits, ["arm", "refresh"])

        self.assertTrue(bridge.invoke(combo_id, "select_text", "WiFi")["ok"])
        self.assertEqual(combo.currentText(), "WiFi")
        self.assertTrue(bridge.invoke(line_id, "set_text", "hello")["ok"])
        self.assertEqual(line.text(), "hello")
        self.assertTrue(bridge.invoke(spin_id, "set_value", 7)["ok"])
        self.assertEqual(spin.value(), 7)

        window.close()
        app.processEvents()

    def test_select_tab_and_query_snapshot(self):
        app, window, *_ = _window()
        bridge = GuiOperatorBridge(window)
        self.assertTrue(bridge.select_tab("Dashboard")["ok"])
        matches = bridge.snapshot("connected")
        self.assertTrue(any(item.get("text") == "Connected" for item in matches))
        self.assertFalse(bridge.select_tab("Missing")["ok"])

        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
