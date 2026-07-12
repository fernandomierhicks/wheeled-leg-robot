BG      = "#12121e"
SURFACE = "#1a1a2e"
BORDER  = "#2a2a4a"
TEXT    = "#d8d8d8"
DIM     = "#888899"
GREEN   = "#00e676"
ORANGE  = "#ff9800"
RED     = "#f44336"
BLUE    = "#448aff"
YELLOW  = "#ffe57f"
WHITE   = "#ffffff"
MONO    = "Consolas, 'Courier New', monospace"

APP_STYLE = f"""
QMainWindow, QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG};
}}
QTabBar::tab {{
    background: {SURFACE};
    color: {DIM};
    padding: 8px 20px;
    border: 1px solid {BORDER};
    border-bottom: none;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {BG};
    color: {TEXT};
    border-bottom: 2px solid {BLUE};
}}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    padding: 2px 8px;
}}
QStatusBar QLabel {{ padding: 0 10px; color: {DIM}; }}
QPushButton {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 12px;
    border-radius: 3px;
}}
QPushButton:hover  {{ border-color: {BLUE}; color: {BLUE}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:checked {{ background: {BLUE}; color: #fff; border-color: {BLUE}; }}
QPushButton:disabled {{ color: {DIM}; }}
QComboBox {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 3px 8px;
    border-radius: 3px;
}}
QComboBox QAbstractItemView {{ background: {SURFACE}; color: {TEXT}; }}
QLineEdit {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 3px 8px;
    border-radius: 3px;
}}
QCheckBox {{ color: {DIM}; }}
QCheckBox::indicator:checked {{ background: {BLUE}; border: 1px solid {BLUE}; }}
QFrame[frameShape="5"] {{ color: {BORDER}; }}
QScrollBar:vertical {{
    background: {SURFACE}; width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 20px;
}}
"""
