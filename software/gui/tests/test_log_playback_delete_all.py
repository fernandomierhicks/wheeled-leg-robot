import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QListWidgetItem, QMessageBox

from tabs.log_playback import LogsTab


_APP = QApplication.instance() or QApplication([])


class LogsTabDeleteAllTests(unittest.TestCase):
    def setUp(self):
        self.tab = LogsTab()
        for idx in (2, 7, 11):
            item = QListWidgetItem(f"LOG{idx:04d}.WLOG")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.tab._file_list.addItem(item)
        self.tab._update_delete_buttons()

    def tearDown(self):
        self.tab.deleteLater()

    def test_cancel_keeps_every_teensy_log(self):
        with (
            patch.object(self.tab._xfer, "is_logging", return_value=False),
            patch.object(self.tab._xfer, "is_transferring", return_value=False),
            patch("tabs.log_playback.QMessageBox.question",
                  return_value=QMessageBox.StandardButton.No) as question,
            patch("tabs.log_playback.send_log_delete") as send_delete,
        ):
            self.tab._on_delete_all()

        self.assertEqual(question.call_args.args[1], "Delete All Teensy Logs?")
        self.assertIn("cannot be undone", question.call_args.args[2].lower())
        send_delete.assert_not_called()

    def test_confirm_deletes_each_listed_log_and_refreshes(self):
        with (
            patch.object(self.tab._xfer, "is_logging", return_value=False),
            patch.object(self.tab._xfer, "is_transferring", return_value=False),
            patch.object(self.tab._xfer, "refresh") as refresh,
            patch("tabs.log_playback.QMessageBox.question",
                  return_value=QMessageBox.StandardButton.Yes),
            patch("tabs.log_playback.QTimer.singleShot",
                  side_effect=lambda _ms, callback: callback()),
            patch("tabs.log_playback.send_log_delete") as send_delete,
        ):
            self.tab._on_delete_all()
            self.tab._on_status_ack(2, 0)
            self.tab._on_status_ack(7, 0)
            self.tab._on_status_ack(11, 0)

        self.assertEqual(
            [call.args[0] for call in send_delete.call_args_list],
            [2, 7, 11],
        )
        refresh.assert_called_once_with()
        self.assertEqual(self.tab._lbl_status.text(), "Deleted all 3 Teensy logs")


if __name__ == "__main__":
    unittest.main()
