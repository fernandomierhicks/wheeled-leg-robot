"""theme.py — Dark theme colors and stylesheet for the ODrive GUI."""

# Named colors
CLR_BG    = "#2d2d2d"
CLR_PANEL = "#252526"
CLR_OK    = "#64ffb4"
CLR_WARN  = "#ffa03c"
CLR_ERR   = "#ff6464"
CLR_INFO  = "#64c8ff"
CLR_CAL   = "#ffff64"
CLR_LABEL = "#cccccc"
CLR_MUTED = "#888888"

DARK_STYLE = f"""
QMainWindow, QWidget {{ background: {CLR_BG}; color: {CLR_LABEL}; }}
QGroupBox {{
    border: 1px solid #444; border-radius: 4px; margin-top: 10px;
    padding-top: 6px; color: {CLR_LABEL};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: #aaa; }}
QPushButton {{
    background: #3c3c3c; border: 1px solid #555; border-radius: 3px;
    padding: 4px 10px; color: {CLR_LABEL};
}}
QPushButton:hover {{ background: #4a4a4a; }}
QPushButton:disabled {{ color: #555; border-color: #444; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #3c3c3c; border: 1px solid #555; border-radius: 3px;
    padding: 2px 4px; color: {CLR_LABEL};
}}
QLabel {{ color: {CLR_LABEL}; }}
QTextEdit {{ background: {CLR_PANEL}; color: {CLR_LABEL}; border: 1px solid #444; }}
QRadioButton {{ color: {CLR_LABEL}; }}
QCheckBox {{ color: {CLR_LABEL}; }}
QTabWidget::pane {{ border: 1px solid #444; }}
QTabBar::tab {{
    background: #3c3c3c; border: 1px solid #444; border-bottom: none;
    padding: 5px 14px; color: #aaa;
}}
QTabBar::tab:selected {{ background: {CLR_BG}; color: {CLR_LABEL}; }}
"""
