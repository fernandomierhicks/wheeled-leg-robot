"""tab_terminal.py — Interactive REPL terminal + Claude command inbox display."""

import logging

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel,
)

from ui.theme import CLR_OK, CLR_ERR, CLR_INFO, CLR_MUTED

log = logging.getLogger("odrive_gui")


class TabTerminal(QWidget):
    """REPL tab: type ODrive commands, see results, and watch Claude inbox traffic."""

    def __init__(self, cmd_interface, parent=None):
        super().__init__(parent)
        self._cmd = cmd_interface
        self._history: list[str] = []
        self._hist_idx = -1
        self._build()

        # Show inbox commands in the terminal too
        self._cmd.result_ready.connect(self._on_inbox_result)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Help label
        hint = QLabel(
            "Namespace: odrv (odrv0), ax0, ax1  |  "
            "Meta: __CLEAR_ERRORS__, __IDLE__, __STOP__, __STATUS__"
        )
        hint.setStyleSheet(f"color: {CLR_MUTED}; font-size: 11px;")
        layout.addWidget(hint)

        # Output area
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 10))
        self._output.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #dcdcdc; border: 1px solid #444; }"
        )
        layout.addWidget(self._output)

        # Input row
        row = QHBoxLayout()
        row.addWidget(QLabel(">>>"))

        self._input = QLineEdit()
        self._input.setFont(QFont("Consolas", 10))
        self._input.setPlaceholderText("odrv.vbus_voltage")
        self._input.returnPressed.connect(self._on_enter)
        self._input.installEventFilter(self)
        row.addWidget(self._input)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(60)
        btn_clear.clicked.connect(self._output.clear)
        row.addWidget(btn_clear)

        layout.addLayout(row)

    # ── Event filter for command history (Up / Down) ──────────────────────────

    def eventFilter(self, obj, event):
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Up:
                self._navigate_history(-1)
                return True
            if event.key() == Qt.Key.Key_Down:
                self._navigate_history(1)
                return True
        return super().eventFilter(obj, event)

    def _navigate_history(self, direction: int):
        if not self._history:
            return
        self._hist_idx = max(0, min(len(self._history) - 1,
                                     self._hist_idx + direction))
        self._input.setText(self._history[self._hist_idx])

    # ── Execute on Enter ──────────────────────────────────────────────────────

    def _on_enter(self):
        cmd = self._input.text().strip()
        if not cmd:
            return
        self._history.append(cmd)
        self._hist_idx = len(self._history)
        self._input.clear()

        self._append(f">>> {cmd}", CLR_INFO)
        result_text, is_error = self._cmd.execute(cmd)
        color = CLR_ERR if is_error else CLR_OK
        self._append(result_text, color)

        # Also log to cmd files so Claude can see interactive commands
        self._cmd._log_result(cmd, result_text, is_error)
        log.info("TERMINAL: %s -> %s", cmd, result_text)

    # ── Inbox result display ──────────────────────────────────────────────────

    def _on_inbox_result(self, cmd: str, result: str, is_error: bool):
        self._append(f"[inbox] {cmd}", CLR_MUTED)
        color = CLR_ERR if is_error else CLR_OK
        self._append(result, color)

    # ── Output helpers ────────────────────────────────────────────────────────

    def _append(self, text: str, color: str):
        self._output.moveCursor(QTextCursor.MoveOperation.End)
        self._output.insertHtml(
            f'<span style="color:{color}; white-space:pre;">'
            f'{_escape_html(text)}</span><br>'
        )
        self._output.moveCursor(QTextCursor.MoveOperation.End)


def _escape_html(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>"))
