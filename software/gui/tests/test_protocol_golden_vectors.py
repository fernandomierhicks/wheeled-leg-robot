import json
import struct
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
                actual = comm_commands.build_frame(bytes.fromhex(vector["payload_hex"]), version=1)
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

    def test_v2_command_envelope_and_correlated_result_decode(self):
        request_id = 0x12345678
        frame = comm_commands.build_frame(
            bytes([comm_commands.CMD_ID_SET_MODE, 3]),
            request_id=request_id,
        )
        self.assertEqual(frame[3], comm_commands.CMD_PAYLOAD_V2)
        self.assertEqual(struct.unpack_from("<I", frame, 8)[0], request_id)

        payload = struct.pack("<IBBBB", request_id, comm_commands.CMD_ID_SET_MODE, 1, 0, 2)
        header = bytes([0x16, 1, 1, 7, len(payload), 0])
        result_frame = b"\xAA\x55" + header + payload + bytes([crc8(header + payload), 0xEF])
        decoded = []
        decoder = PacketDecoder("result-test")
        decoder.packet_decoded.connect(decoded.append)
        decoder.feed(result_frame)
        self.assertEqual(decoded[0]["request_id"], request_id)
        self.assertTrue(decoded[0]["command_accepted"])
        self.assertEqual(decoded[0]["command_state"], 2)


if __name__ == "__main__":
    unittest.main()
