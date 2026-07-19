import os

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

# ── Icon assets for QSS-styled control subcontrols (checkbox check mark,
# spinbox up/down arrows) — QSS url() needs forward slashes even on Windows.
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def _asset_url(name: str) -> str:
    return os.path.join(_ASSETS_DIR, name).replace("\\", "/")


_ICON_CHECK         = _asset_url("check.svg")
_ICON_CHEVRON_UP    = _asset_url("chevron_up.svg")
_ICON_CHEVRON_DOWN  = _asset_url("chevron_down.svg")

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
QCheckBox {{ color: {DIM}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border: 1px solid {BLUE}; }}
QCheckBox::indicator:checked {{
    background: {GREEN};
    border: 1px solid {GREEN};
    image: url({_ICON_CHECK});
}}
QCheckBox::indicator:disabled {{ background: {BG}; border: 1px solid {BORDER}; }}
QSpinBox, QDoubleSpinBox {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px 4px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {BLUE}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled {{ background: {BG}; color: {DIM}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 14px;
    border-left: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    background: {SURFACE};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 14px;
    border-left: 1px solid {BORDER};
    background: {SURFACE};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {BORDER}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({_ICON_CHEVRON_UP});
    width: 8px;
    height: 5px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({_ICON_CHEVRON_DOWN});
    width: 8px;
    height: 5px;
}}
QFrame[frameShape="5"] {{ color: {BORDER}; }}
QScrollBar:vertical {{
    background: {SURFACE}; width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 20px;
}}
"""
