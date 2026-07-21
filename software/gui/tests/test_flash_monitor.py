import unittest

from tabs.flash_monitor import _decode_printable_serial_line


class PrintableSerialLineTests(unittest.TestCase):
    def test_accepts_ascii_diagnostic(self):
        self.assertEqual(
            _decode_printable_serial_line(b"[ESP32] client connected\r"),
            "[ESP32] client connected",
        )

    def test_rejects_binary_frame(self):
        self.assertIsNone(
            _decode_printable_serial_line(b"\xaa\x55\x13\x00LOG_DATA\x00\xff")
        )

    def test_rejects_unbounded_line(self):
        self.assertIsNone(_decode_printable_serial_line(b"x" * 513))


if __name__ == "__main__":
    unittest.main()
