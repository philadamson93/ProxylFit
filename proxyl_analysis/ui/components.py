"""
Reusable UI components for ProxylFit.
"""

from typing import Callable

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import get_logo_path


class MatplotlibCanvas(FigureCanvas):
    """A matplotlib canvas that can be embedded in Qt widgets."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('white')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def add_subplot(self, *args, **kwargs):
        """Add a subplot to the figure."""
        return self.fig.add_subplot(*args, **kwargs)

    def clear(self):
        """Clear the figure."""
        self.fig.clear()
        self.draw()


class LogoWidget(QLabel):
    """Widget displaying the ProxylFit logo."""

    def __init__(self, parent=None, max_height=60):
        super().__init__(parent)
        logo_path = get_logo_path()
        if logo_path:
            pixmap = QPixmap(str(logo_path))
            scaled = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)
            self.setPixmap(scaled)
        else:
            self.setText("ProxylFit")
            self.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")


class HeaderWidget(QWidget):
    """Standard header widget with title and logo."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title section
        title_layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("color: #666; font-size: 11px;")
            title_layout.addWidget(subtitle_label)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Logo
        logo = LogoWidget(self)
        layout.addWidget(logo)


class InstructionWidget(QLabel):
    """Widget for displaying instructions to the user."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("instructionLabel")
        self.setWordWrap(True)


class InfoWidget(QLabel):
    """Widget for displaying information/statistics."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("infoLabel")
        self.setWordWrap(True)

    def update_info(self, text: str):
        """Update the displayed information."""
        self.setText(text)


class ButtonBar(QWidget):
    """Standard button bar with configurable buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.addStretch()
        self.buttons = {}

    def add_button(self, name: str, text: str, callback: Callable,
                   button_type: str = "default") -> QPushButton:
        """Add a button to the bar."""
        btn = QPushButton(text)

        if button_type == "accept":
            btn.setObjectName("acceptButton")
        elif button_type == "cancel":
            btn.setObjectName("cancelButton")
        elif button_type == "export":
            btn.setObjectName("exportButton")

        btn.clicked.connect(callback)
        self.layout.addWidget(btn)
        self.buttons[name] = btn
        return btn

    def add_stretch(self):
        """Add stretch between buttons."""
        self.layout.addStretch()
