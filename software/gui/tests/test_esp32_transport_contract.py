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
WIFI = ROOT / "software" / "gui" / "tabs" / "wifi_transport.py"


class Esp32TransportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.config = CONFIG.read_text(encoding="utf-8")
        cls.wifi = WIFI.read_text(encoding="utf-8")

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

    def test_wifi_uses_combined_leased_unicast(self):
        self.assertRegex(self.config, r"#define WIFI_TELEM_COMBINED\s+1")
        self.assertIn("WLR_CLAIM_V1", self.source)
        self.assertIn("WLR_ACK_V1", self.source)
        self.assertIn("WLR_BUSY_V1", self.source)
        self.assertIn("WLR_CLAIM_V1", self.wifi)
        self.assertIn("WLR_ACK_V1", self.wifi)

    def test_tcp_does_not_duplicate_telemetry(self):
        self.assertIn("type != COMM_TYPE_TELEM_A && type != COMM_TYPE_TELEM_B", self.source)
        self.assertIn("TCP_NODELAY", self.wifi)

    def test_udp_timeout_does_not_close_tcp(self):
        marker = "time.monotonic() - self._last_udp_time > TELEMETRY_TIMEOUT_S"
        timeout_start = self.wifi.index(marker)
        timeout_block = self.wifi[timeout_start:self.wifi.index("udp.close()", timeout_start)]
        self.assertNotIn("_close_tcp", timeout_block)


if __name__ == "__main__":
    unittest.main()
