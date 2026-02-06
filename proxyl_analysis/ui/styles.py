"""
Styling and theming for ProxylFit Qt UI components.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication


PROXYLFIT_STYLE = """
QMainWindow, QDialog {
    background-color: #f5f5f5;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #cccccc;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QPushButton {
    background-color: #e0e0e0;
    border: 1px solid #b0b0b0;
    border-radius: 4px;
    padding: 8px 16px;
    min-width: 80px;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #d0d0d0;
}

QPushButton:pressed {
    background-color: #c0c0c0;
}

QPushButton#acceptButton {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}

QPushButton#acceptButton:hover {
    background-color: #45a049;
}

QPushButton#cancelButton {
    background-color: #f44336;
    color: white;
}

QPushButton#cancelButton:hover {
    background-color: #da190b;
}

QPushButton#exportButton {
    background-color: #2196F3;
    color: white;
}

QPushButton#exportButton:hover {
    background-color: #1976D2;
}

QLabel#titleLabel {
    font-size: 16px;
    font-weight: bold;
    color: #333333;
    padding: 5px;
}

QLabel#instructionLabel {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    border-radius: 4px;
    padding: 8px;
    color: #2e7d32;
}

QLabel#infoLabel {
    background-color: #fff3e0;
    border: 1px solid #ffe0b2;
    border-radius: 4px;
    padding: 8px;
}

QStatusBar {
    background-color: #e0e0e0;
    border-top: 1px solid #b0b0b0;
}

QSlider::groove:horizontal {
    border: 1px solid #999999;
    height: 8px;
    background: #e0e0e0;
    margin: 2px 0;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #4CAF50;
    border: 1px solid #388E3C;
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}
"""


def get_logo_path() -> Optional[Path]:
    """Get path to ProxylFit logo if it exists."""
    logo_path = Path(__file__).parent.parent.parent / "proxylfit.png"
    if logo_path.exists():
        return logo_path
    logo_path = Path("proxylfit.png")
    if logo_path.exists():
        return logo_path
    return None


def init_qt_app() -> QApplication:
    """Initialize Qt application with ProxylFit styling."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("ProxylFit")
    app.setStyle("Fusion")
    app.setStyleSheet(PROXYLFIT_STYLE)
    app.setQuitOnLastWindowClosed(False)
    return app
