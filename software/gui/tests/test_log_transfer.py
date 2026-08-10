import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

from PyQt6.QtCore import QCoreApplication

from tabs import log_transfer


class LogTransferRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._old_log_dir = log_transfer.LOG_DIR
        self._old_send_log_get = log_transfer.send_log_get
        log_transfer.LOG_DIR = Path(self._temp_dir.name)
        self.requests = []
        log_transfer.send_log_get = lambda index, start=0, kind=0: self.requests.append((index, start, kind))
        self.manager = log_transfer.LogTransferManager()
        self.manager._watchdog.stop()
        self.completed = []
        self.manager.transfer_complete.connect(
            lambda index, path, ok: self.completed.append((index, path, ok))
        )

    def tearDown(self):
        self.manager._watchdog.stop()
        log_transfer.LogPacketBus.instance().packet.disconnect(self.manager._on_packet)
        self.manager.deleteLater()
        log_transfer.LOG_DIR = self._old_log_dir
        log_transfer.send_log_get = self._old_send_log_get
        self._temp_dir.cleanup()

    def _begin(self, index, total):
        self.manager._on_log_info({
            "log_info_type": log_transfer.LOG_INFO_XFER_BEGIN,
            "log_file_index": index,
            "log_total_chunks": total,
        })

    def _chunk(self, index, chunk_index, data):
        self.manager._on_log_data({
            "log_file_index": index,
            "log_chunk_index": chunk_index,
            "log_data": data,
        })

    def _end(self, index, data):
        self.manager._on_log_info({
            "log_info_type": log_transfer.LOG_INFO_XFER_END,
            "log_file_index": index,
            "log_crc32": zlib.crc32(data) & 0xFFFFFFFF,
        })

    def test_missing_chunk_repairs_suffix_and_preserves_full_crc(self):
        chunks = [b"header", b"one", b"two", b"tail"]
        whole = b"".join(chunks)

        self.manager.download(7)
        self.assertEqual(self.requests, [(7, 0, 0)])
        self._begin(7, len(chunks))
        self._chunk(7, 0, chunks[0])
        self._chunk(7, 2, chunks[2])
        self._chunk(7, 3, chunks[3])
        self._end(7, whole)

        self.assertEqual(self.requests[-1], (7, 1, 0))
        self.assertEqual(self.manager._chunks, {0: chunks[0]})

        self._begin(7, len(chunks))
        for i in range(1, len(chunks)):
            self._chunk(7, i, chunks[i])
        self._end(7, b"".join(chunks[1:]))

        self.assertEqual(len(self.completed), 1)
        _, path, ok = self.completed[0]
        self.assertTrue(ok)
        self.assertEqual(Path(path).read_bytes(), whole)

    def test_bad_suffix_crc_retries_same_suffix(self):
        chunks = [b"zero", b"one", b"two"]

        self.manager.download(4)
        self._begin(4, len(chunks))
        self._chunk(4, 0, chunks[0])
        self._chunk(4, 2, chunks[2])
        self._end(4, b"".join(chunks))
        self.assertEqual(self.requests[-1], (4, 1, 0))

        self._begin(4, len(chunks))
        self._chunk(4, 1, chunks[1])
        self._chunk(4, 2, b"corrupt")
        self._end(4, b"".join(chunks[1:]))

        self.assertEqual(self.requests[-1], (4, 1, 0))
        self.assertEqual(self.manager._chunks, {0: chunks[0]})
        self.assertEqual(self.completed, [])

    def test_timeout_before_xfer_begin_repeats_request(self):
        self.manager.download(9)
        self.manager._last_chunk_ms -= log_transfer.CHUNK_TIMEOUT_S + 1
        self.manager._check_timeout()
        self.assertEqual(self.requests, [(9, 0, 0), (9, 0, 0)])

    def test_duplicate_chunk_does_not_mask_a_stalled_relay(self):
        self.manager.download(9)
        self._begin(9, 3)
        self._chunk(9, 0, b"zero")
        self.manager._last_chunk_ms -= log_transfer.CHUNK_TIMEOUT_S + 1

        # This is what arrives when the ESP32 received a retransmission but
        # its acknowledgement cannot advance the Teensy stream.  It is not
        # progress and must not reset the GUI repair watchdog.
        self._chunk(9, 0, b"zero")
        self.manager._check_timeout()

        self.assertEqual(self.requests[-1], (9, 1, 0))

    def test_get_status_error_fails_without_waiting_for_timeouts(self):
        self.manager.download(6)
        self.manager._on_log_info({
            "log_info_type": log_transfer.LOG_INFO_STATUS,
            "log_file_index": 6,
            "log_status": 1,
        })

        self.assertFalse(self.manager.is_transferring())
        self.assertEqual(self.manager.failure_reason(), "log GET was rejected by the robot")
        self.assertEqual(self.completed, [(6, "", False)])

    def test_stale_frames_are_ignored_until_repair_begin(self):
        chunks = [b"zero", b"one", b"two"]

        self.manager.download(5)
        self._begin(5, len(chunks))
        self._chunk(5, 0, chunks[0])
        self._chunk(5, 2, chunks[2])
        self._end(5, b"".join(chunks))
        self.assertEqual(self.requests[-1], (5, 1, 0))

        # These can already be in the ESP32/OS transmit path when the repair
        # request is issued. They belong to the superseded attempt.
        self._chunk(5, 1, b"stale")
        self._end(5, b"".join(chunks))
        self.assertEqual(self.manager._chunks, {0: chunks[0]})
        self.assertEqual(self.requests, [(5, 0, 0), (5, 1, 0)])

        self._begin(5, len(chunks))
        self._chunk(5, 1, chunks[1])
        self._chunk(5, 2, chunks[2])
        self._end(5, b"".join(chunks[1:]))
        self.assertEqual(len(self.completed), 1)
        self.assertTrue(self.completed[0][2])

    def test_gui_sd_commands_start_and_stop_host_capture(self):
        self.manager._host = Mock()
        with (
            patch.object(log_transfer, "send_log_start") as send_start,
            patch.object(log_transfer, "send_log_stop") as send_stop,
        ):
            self.manager.start_logging(2500)
            self.manager.stop_logging()

        send_start.assert_called_once_with(2500)
        send_stop.assert_called_once_with()
        self.manager._host.start.assert_called_once_with()
        self.manager._host.stop.assert_called_once_with()

    def test_onboard_sd_events_start_and_stop_host_capture(self):
        self.manager._host = Mock()
        self.manager._on_log_info({
            "log_info_type": log_transfer.LOG_INFO_STARTED,
            "log_file_index": 12,
        })
        self.manager._on_log_info({
            "log_info_type": log_transfer.LOG_INFO_STOPPED,
            "log_file_index": 12,
        })

        self.manager._host.start.assert_called_once_with()
        self.manager._host.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
