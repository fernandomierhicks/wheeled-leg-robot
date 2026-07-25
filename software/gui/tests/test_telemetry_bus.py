import unittest

from PyQt6.QtCore import QCoreApplication

from tabs.telemetry_bus import TelemetryBus


class TelemetryBusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        TelemetryBus._instance = None
        self.bus = TelemetryBus.instance()
        self.bus._live_timer.stop()

    def tearDown(self):
        self.bus._live_timer.stop()
        TelemetryBus._instance = None

    def test_live_capture_sees_every_telem_before_ui_coalescing(self):
        live = []
        ui = []
        self.bus.live_packet.connect(live.append)
        self.bus.packet.connect(ui.append)

        self.bus.publish({"ptype": 0x01, "loop_count": 1})
        self.bus.publish({"ptype": 0x01, "loop_count": 2})
        self.assertEqual([item["loop_count"] for item in live], [1, 2])
        self.assertEqual(ui, [])

        self.bus._flush_live()
        self.assertEqual([item["loop_count"] for item in ui], [2])


if __name__ == "__main__":
    unittest.main()
