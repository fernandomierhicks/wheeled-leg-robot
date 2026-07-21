import json
import unittest
from pathlib import Path

from tabs import comm_commands
from tabs.flash_monitor import PacketDecoder
from tabs.telem_format import crc8


VECTORS_PATH = (
    Path(__file__).resolve().parents[3]
    / "firmware"
    / "robot_teensy"
    / "shared"
    / "protocol_golden_vectors.json"
)


def _load_vectors():
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]


class ProtocolGoldenVectorTests(unittest.TestCase):
    def test_frozen_frame_bytes_match_python_crc(self):
        for vector in _load_vectors():
            with self.subTest(vector=vector["name"]):
                payload = bytes.fromhex(vector["payload_hex"])
                header = bytes([
                    vector["type"],
                    vector["version"],
                    vector["source"],
                    vector["sequence"],
                    len(payload) & 0xFF,
                    len(payload) >> 8,
                ])
                expected = b"\xAA\x55" + header + payload + bytes([crc8(header + payload), 0xEF])
                self.assertEqual(expected.hex(), vector["frame_hex"])

    def test_python_command_encoder_matches_command_vectors(self):
        for vector in _load_vectors():
            if vector["type"] != comm_commands.COMM_TYPE_CMD:
                continue
            with self.subTest(vector=vector["name"]):
                comm_commands._seq[0] = vector["sequence"]
                actual = comm_commands.build_frame(bytes.fromhex(vector["payload_hex"]))
                self.assertEqual(actual.hex(), vector["frame_hex"])

    def test_gui_decoder_accepts_every_vector_exactly_once(self):
        for vector in _load_vectors():
            with self.subTest(vector=vector["name"]):
                decoded = []
                decoder = PacketDecoder("golden")
                decoder.packet_decoded.connect(decoded.append)
                decoder.feed(bytes.fromhex(vector["frame_hex"]))
                self.assertEqual(len(decoded), 1)
                self.assertEqual(decoded[0]["ptype"], vector["type"])
                self.assertEqual(decoded[0]["version"], vector["version"])
                self.assertEqual(decoded[0]["source"], vector["source"])
                self.assertEqual(decoded[0]["seq"], vector["sequence"])
                self.assertEqual(decoded[0]["length"], len(bytes.fromhex(vector["payload_hex"])))
                self.assertEqual(decoded[0]["link_crc_drops"], 0)


if __name__ == "__main__":
    unittest.main()
