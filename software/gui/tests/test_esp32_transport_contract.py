"""Static ownership guards for the ESP32 relay architecture.

These assertions intentionally fail if a future edit reintroduces multiple
encoders/writers for one physical stream or unbounded parser work.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "firmware" / "robot_teensy" / "esp32" / "src" / "main.cpp"
CONFIG = ROOT / "firmware" / "robot_teensy" / "esp32" / "src" / "config.h"


class Esp32TransportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.config = CONFIG.read_text(encoding="utf-8")

    def test_one_encoder_per_physical_uplink(self):
        self.assertNotIn("g_usb_diag", self.source)
        self.assertNotIn("g_wifi_diag", self.source)
        self.assertNotIn("g_udp_diag_stream", self.source)
        self.assertEqual(2, len(re.findall(r"g_usb\.send\(", self.source)))
        self.assertEqual(3, len(re.findall(r"g_telem_udp\.send\(", self.source)))

    def test_one_uplink_writer_task(self):
        self.assertEqual(1, len(re.findall(r"xTaskCreatePinnedToCore\(uplink_task", self.source)))
        self.assertNotIn("log_uplink_task", self.source)
        self.assertNotIn("g_usb_mutex", self.source)

    def test_uart_parser_and_tcp_write_are_bounded(self):
        self.assertIn("g_teensy.update(UART_PARSE_BUDGET_BYTES)", self.source)
        self.assertIn("g_usb.update(HOST_PARSE_BUDGET_BYTES)", self.source)
        self.assertIn("g_comm_tcp->update(HOST_PARSE_BUDGET_BYTES)", self.source)
        self.assertIn("MSG_DONTWAIT", self.source)
        self.assertIn("UART_PARSE_BUDGET_BYTES", self.config)
        self.assertIn("HOST_PARSE_BUDGET_BYTES", self.config)

    def test_control_and_bulk_queues_precede_telemetry(self):
        control = self.source.index("xQueueReceive(g_control_uplink_q")
        log = self.source.index("xQueueReceive(g_log_uplink_q", control)
        telemetry = self.source.index("xQueueReceive(g_uplink_q", log)
        self.assertLess(control, log)
        self.assertLess(log, telemetry)


if __name__ == "__main__":
    unittest.main()
