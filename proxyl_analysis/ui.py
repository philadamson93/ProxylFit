"""
Qt-based UI module for ProxylFit.

Provides modern, responsive UI components using PySide6 with proper layout management.
Embeds matplotlib figures using FigureCanvasQTAgg for scientific visualization.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable

# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QFrame, QSplitter, QStatusBar,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog,
    QMessageBox, QSizePolicy, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QIcon, QPixmap, QPalette, QColor

# Matplotlib Qt backend
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath


# ============================================================================
# Styling and Theming
# ============================================================================

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
    logo_path = Path(__file__).parent.parent / "proxylfit.png"
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
    return app


# ============================================================================
# Base Components
# ============================================================================

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


# ============================================================================
# ROI Selection Dialogs
# ============================================================================

class ROISelectorDialog(QDialog):
    """Qt dialog for interactive rectangle ROI selection."""

    roi_selected = Signal(np.ndarray)  # Emits the ROI mask when selected

    def __init__(self, image_slice: np.ndarray, title: str = "Select ROI", parent=None):
        super().__init__(parent)
        self.image_slice = image_slice
        self.roi_coords = None
        self.mask = None
        self._title = title

        self.setWindowTitle(f"ProxylFit - {title}")
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)

        self._setup_ui()
        self._setup_plot()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = HeaderWidget("ROI Selection", "Click and drag to select rectangular region")
        layout.addWidget(header)

        # Instructions
        instructions = InstructionWidget(
            "Instructions:\n"
            "1. Click and drag on the image to select a rectangular ROI\n"
            "2. The selection can be adjusted by dragging the corners/edges\n"
            "3. Click 'Accept ROI' when satisfied with the selection"
        )
        layout.addWidget(instructions)

        # Main content area
        content_layout = QHBoxLayout()

        # Canvas with toolbar
        canvas_layout = QVBoxLayout()
        self.canvas = MatplotlibCanvas(self, width=8, height=6)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        content_layout.addLayout(canvas_layout, stretch=3)

        # Info panel
        info_layout = QVBoxLayout()
        self.info_widget = InfoWidget("ROI not selected yet")
        info_layout.addWidget(self.info_widget)
        info_layout.addStretch()
        content_layout.addLayout(info_layout, stretch=1)

        layout.addLayout(content_layout)

        # Button bar
        button_bar = ButtonBar()
        button_bar.add_button("cancel", "Cancel", self.reject, "cancel")
        button_bar.add_stretch()
        button_bar.add_button("accept", "Accept ROI", self._accept_roi, "accept")
        layout.addWidget(button_bar)

        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Drag to select ROI")

    def _setup_plot(self):
        """Set up the matplotlib plot with rectangle selector."""
        self.ax = self.canvas.add_subplot(111)

        # Display image
        self.im = self.ax.imshow(self.image_slice.T, cmap='gray', origin='lower')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(self._title)

        # Add colorbar
        self.canvas.fig.colorbar(self.im, ax=self.ax, shrink=0.8)

        # Initialize rectangle selector
        self.rectangle_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        self.canvas.draw()

    def _on_rectangle_select(self, eclick, erelease):
        """Callback for rectangle selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Ensure coordinates are in correct order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Store ROI coordinates
        self.roi_coords = (x_min, x_max, y_min, y_max)

        # Create mask
        self.mask = np.zeros(self.image_slice.shape, dtype=bool)
        self.mask[x_min:x_max, y_min:y_max] = True

        # Update info display
        roi_size = (x_max - x_min) * (y_max - y_min)
        roi_pixels = self.image_slice[self.mask]

        info_text = (
            f"ROI Coordinates:\n"
            f"  X: {x_min} to {x_max}\n"
            f"  Y: {y_min} to {y_max}\n\n"
            f"Statistics:\n"
            f"  Pixels: {roi_size}\n"
            f"  Mean: {np.mean(roi_pixels):.2f}\n"
            f"  Std: {np.std(roi_pixels):.2f}\n"
            f"  Min: {np.min(roi_pixels):.2f}\n"
            f"  Max: {np.max(roi_pixels):.2f}"
        )
        self.info_widget.update_info(info_text)

        self.status_bar.showMessage(f"ROI selected: {roi_size} pixels")

    def _accept_roi(self):
        """Accept the current ROI selection."""
        if self.mask is None or not np.any(self.mask):
            QMessageBox.warning(self, "No ROI", "Please select an ROI first.")
            return

        self.roi_selected.emit(self.mask)
        self.accept()

    def get_mask(self) -> Optional[np.ndarray]:
        """Get the ROI mask after dialog closes."""
        return self.mask

    def get_stats(self) -> dict:
        """Get statistics about the selected ROI."""
        if self.mask is None:
            return {}

        roi_pixels = self.image_slice[self.mask]
        return {
            'num_pixels': np.sum(self.mask),
            'mean_intensity': np.mean(roi_pixels),
            'std_intensity': np.std(roi_pixels),
            'min_intensity': np.min(roi_pixels),
            'max_intensity': np.max(roi_pixels),
            'coordinates': self.roi_coords
        }


class ManualContourDialog(QDialog):
    """Qt dialog for manual contour ROI drawing."""

    roi_selected = Signal(np.ndarray)

    def __init__(self, image_4d: np.ndarray, z_index: int = 0,
                 title: str = "Draw ROI Contour", parent=None):
        super().__init__(parent)
        self.image_4d = image_4d
        self.z_index = z_index
        self.max_z = image_4d.shape[2] - 1
        self.image_slice = image_4d[:, :, z_index, 0]
        self.mask = None
        self._title = title

        # Drawing state
        self.contour_points = []
        self.drawing = False
        self.current_path = []
        self.path_plots = []

        self.setWindowTitle(f"ProxylFit - {title}")
        self.setMinimumSize(1000, 750)
        self.resize(1100, 850)

        self._setup_ui()
        self._setup_plot()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = HeaderWidget("Manual Contour ROI",
                             f"Z-slice: {self.z_index}/{self.max_z}")
        layout.addWidget(header)

        # Instructions
        instructions = InstructionWidget(
            "Instructions:\n"
            "- Drag to draw contour around ROI\n"
            "- Press 'C' or click 'Close Contour' to complete the shape\n"
            "- Press 'R' or click 'Reset' to start over\n"
            "- Use slider or arrow keys to change Z-slice"
        )
        layout.addWidget(instructions)

        # Main content
        content_layout = QHBoxLayout()

        # Canvas with toolbar
        canvas_layout = QVBoxLayout()
        self.canvas = MatplotlibCanvas(self, width=9, height=7)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        content_layout.addLayout(canvas_layout, stretch=3)

        # Control panel
        control_layout = QVBoxLayout()

        # Z-slice control
        z_group = QGroupBox("Z-Slice Navigation")
        z_layout = QVBoxLayout(z_group)

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(self.max_z)
        self.z_slider.setValue(self.z_index)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_slider)

        self.z_label = QLabel(f"Slice: {self.z_index}/{self.max_z}")
        z_layout.addWidget(self.z_label)

        control_layout.addWidget(z_group)

        # Drawing controls
        draw_group = QGroupBox("Drawing Controls")
        draw_layout = QVBoxLayout(draw_group)

        close_btn = QPushButton("Close Contour (C)")
        close_btn.clicked.connect(self._close_contour)
        draw_layout.addWidget(close_btn)

        reset_btn = QPushButton("Reset (R)")
        reset_btn.clicked.connect(self._reset_contour)
        draw_layout.addWidget(reset_btn)

        control_layout.addWidget(draw_group)

        # Info panel
        self.info_widget = InfoWidget("Draw a contour to see statistics")
        control_layout.addWidget(self.info_widget)

        control_layout.addStretch()
        content_layout.addLayout(control_layout, stretch=1)

        layout.addLayout(content_layout)

        # Button bar
        button_bar = ButtonBar()
        button_bar.add_button("cancel", "Cancel", self.reject, "cancel")
        button_bar.add_stretch()
        button_bar.add_button("accept", "Accept ROI", self._accept_roi, "accept")
        layout.addWidget(button_bar)

        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Draw contour by clicking and dragging")

    def _setup_plot(self):
        """Set up the matplotlib plot."""
        self.ax = self.canvas.add_subplot(111)
        self._update_image()

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Enable keyboard focus
        self.canvas.setFocusPolicy(Qt.StrongFocus)

    def _update_image(self):
        """Update the displayed image."""
        self.ax.clear()
        self.image_slice = self.image_4d[:, :, self.z_index, 0]
        self.im = self.ax.imshow(self.image_slice.T, cmap='gray', origin='lower')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f"{self._title} (Z={self.z_index})")
        self.canvas.draw()

    def _on_z_changed(self, value):
        """Handle Z-slice slider change."""
        self.z_index = value
        self.z_label.setText(f"Slice: {self.z_index}/{self.max_z}")
        self._reset_contour()
        self._update_image()

    def _on_press(self, event):
        """Start drawing when mouse is pressed."""
        if event.inaxes != self.ax or event.button != 1:
            return

        self.drawing = True
        self.current_path = [(event.xdata, event.ydata)]
        self.status_bar.showMessage("Drawing...")

    def _on_release(self, event):
        """Stop drawing when mouse is released."""
        if not self.drawing:
            return

        self.drawing = False
        if len(self.current_path) > 2:
            self.contour_points.extend(self.current_path)
            self.status_bar.showMessage(f"Added {len(self.current_path)} points to contour")
        self.current_path = []

    def _on_motion(self, event):
        """Add points while dragging."""
        if not self.drawing or event.inaxes != self.ax:
            return

        if len(self.current_path) > 0:
            last_x, last_y = self.current_path[-1]
            dist = ((event.xdata - last_x)**2 + (event.ydata - last_y)**2)**0.5
            if dist > 2:
                self.current_path.append((event.xdata, event.ydata))

                if len(self.current_path) > 1:
                    prev_x, prev_y = self.current_path[-2]
                    line = self.ax.plot([prev_x, event.xdata],
                                       [prev_y, event.ydata], 'r-', linewidth=2)[0]
                    self.path_plots.append(line)
                    self.canvas.draw()

    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'r':
            self._reset_contour()
        elif event.key == 'c':
            self._close_contour()
        elif event.key == 'up':
            if self.z_index < self.max_z:
                self.z_slider.setValue(self.z_index + 1)
        elif event.key == 'down':
            if self.z_index > 0:
                self.z_slider.setValue(self.z_index - 1)

    def _reset_contour(self):
        """Reset the contour."""
        self.contour_points = []
        self.current_path = []
        self.mask = None

        for plot in self.path_plots:
            try:
                plot.remove()
            except:
                pass
        self.path_plots = []

        self.canvas.draw()
        self.info_widget.update_info("Draw a contour to see statistics")
        self.status_bar.showMessage("Contour reset")

    def _close_contour(self):
        """Close the contour and create mask."""
        if len(self.contour_points) < 3:
            QMessageBox.warning(self, "Not Enough Points",
                              "Draw at least 3 points to create a contour.")
            return

        # Draw closing line
        start = self.contour_points[0]
        end = self.contour_points[-1]
        closing_line = self.ax.plot([end[0], start[0]],
                                   [end[1], start[1]], 'g-', linewidth=3)[0]
        self.path_plots.append(closing_line)

        # Create mask
        self._create_mask_from_contour()
        self.canvas.draw()

    def _create_mask_from_contour(self):
        """Create binary mask from contour points."""
        if len(self.contour_points) < 3:
            return

        path = MplPath(self.contour_points)

        height, width = self.image_slice.shape[1], self.image_slice.shape[0]
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

        mask_flat = path.contains_points(points)
        mask_display = mask_flat.reshape((height, width))
        self.mask = mask_display.T

        # Show overlay
        mask_overlay = np.ma.masked_where(~mask_display, np.ones_like(mask_display))
        overlay = self.ax.imshow(mask_overlay, cmap='Reds', alpha=0.4,
                                origin='lower', extent=self.im.get_extent())
        self.path_plots.append(overlay)

        # Update info
        num_pixels = np.sum(self.mask)
        roi_pixels = self.image_slice[self.mask]

        info_text = (
            f"Contour Statistics:\n"
            f"  Pixels: {num_pixels}\n"
            f"  Points: {len(self.contour_points)}\n\n"
            f"Intensity:\n"
            f"  Mean: {np.mean(roi_pixels):.2f}\n"
            f"  Std: {np.std(roi_pixels):.2f}\n"
            f"  Min: {np.min(roi_pixels):.2f}\n"
            f"  Max: {np.max(roi_pixels):.2f}"
        )
        self.info_widget.update_info(info_text)
        self.status_bar.showMessage(f"Contour closed: {num_pixels} pixels selected")

    def _accept_roi(self):
        """Accept the ROI."""
        if self.mask is None or not np.any(self.mask):
            QMessageBox.warning(self, "No ROI",
                              "Please draw and close a contour first.")
            return

        self.roi_selected.emit(self.mask)
        self.accept()

    def get_mask(self) -> Optional[np.ndarray]:
        """Get the ROI mask."""
        return self.mask

    def get_z_index(self) -> int:
        """Get the final Z-index."""
        return self.z_index


class InjectionTimeSelectorDialog(QDialog):
    """Qt dialog for selecting injection time from signal data."""

    time_selected = Signal(int)  # Emits the selected time index

    def __init__(self, time: np.ndarray, signal: np.ndarray,
                 time_units: str = 'minutes', output_dir: str = './output',
                 parent=None):
        super().__init__(parent)
        self.time = time
        self.signal = signal
        self.time_units = time_units
        self.output_dir = output_dir
        self.injection_index = 0

        self.setWindowTitle("ProxylFit - Injection Time Selection")
        self.setMinimumSize(1000, 650)
        self.resize(1100, 700)

        self._setup_ui()
        self._setup_plot()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = HeaderWidget("Injection Time Selection",
                             "Click on the time point when contrast was injected")
        layout.addWidget(header)

        # Instructions
        instructions = InstructionWidget(
            "Instructions:\n"
            "- Click on the plot to select the injection time point\n"
            "- The red vertical line shows the current selection\n"
            "- Use 'Export CSV' to save the timecourse data"
        )
        layout.addWidget(instructions)

        # Main content
        content_layout = QHBoxLayout()

        # Canvas with toolbar
        canvas_layout = QVBoxLayout()
        self.canvas = MatplotlibCanvas(self, width=10, height=5)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        content_layout.addLayout(canvas_layout, stretch=3)

        # Info panel
        info_layout = QVBoxLayout()

        self.info_widget = InfoWidget(
            f"Selected Time: {self.time[0]:.2f} {self.time_units}\n"
            f"Index: 0\n"
            f"Signal: {self.signal[0]:.2f}"
        )
        info_layout.addWidget(self.info_widget)

        # Statistics
        stats_group = QGroupBox("Signal Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_text = (
            f"Time points: {len(self.time)}\n"
            f"Time range: {self.time[0]:.1f} - {self.time[-1]:.1f} {self.time_units}\n"
            f"Signal range: {np.min(self.signal):.1f} - {np.max(self.signal):.1f}"
        )
        stats_label = QLabel(stats_text)
        stats_layout.addWidget(stats_label)
        info_layout.addWidget(stats_group)

        info_layout.addStretch()
        content_layout.addLayout(info_layout, stretch=1)

        layout.addLayout(content_layout)

        # Button bar
        button_bar = ButtonBar()
        button_bar.add_button("cancel", "Cancel", self.reject, "cancel")
        button_bar.add_stretch()
        button_bar.add_button("export", "Export CSV", self._export_csv, "export")
        button_bar.add_button("accept", "Set Injection Time", self._accept_time, "accept")
        layout.addWidget(button_bar)

        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Click on plot to select injection time")

    def _setup_plot(self):
        """Set up the matplotlib plot."""
        self.ax = self.canvas.add_subplot(111)

        # Plot signal
        self.line, = self.ax.plot(self.time, self.signal, 'b-o',
                                  linewidth=2, markersize=4, label='Signal')

        self.ax.set_xlabel(f'Time ({self.time_units})')
        self.ax.set_ylabel('Signal Intensity')
        self.ax.set_title('Select Injection Time Point')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        # Set y-axis to data range
        y_min, y_max = np.min(self.signal), np.max(self.signal)
        y_range = y_max - y_min
        self.ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        # Initial injection marker
        self.injection_marker = self.ax.axvline(x=self.time[0], color='red',
                                                linewidth=3, label='Injection time')

        # Connect click event
        self.canvas.mpl_connect('button_press_event', self._on_click)

        self.canvas.draw()

    def _on_click(self, event):
        """Handle click to select injection time."""
        if event.inaxes != self.ax or event.button != 1:
            return

        # Find closest time point
        closest_idx = np.argmin(np.abs(self.time - event.xdata))
        self.injection_index = closest_idx

        # Update marker
        self.injection_marker.remove()
        self.injection_marker = self.ax.axvline(x=self.time[closest_idx],
                                                color='red', linewidth=3)
        self.canvas.draw()

        # Update info
        self.info_widget.update_info(
            f"Selected Time: {self.time[closest_idx]:.2f} {self.time_units}\n"
            f"Index: {closest_idx}\n"
            f"Signal: {self.signal[closest_idx]:.2f}"
        )

        self.status_bar.showMessage(
            f"Selected: {self.time[closest_idx]:.2f} {self.time_units} (index {closest_idx})"
        )

    def _export_csv(self):
        """Export timecourse data to CSV."""
        import csv

        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        csv_file = Path(self.output_dir) / "timecourse_data.csv"

        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'Time ({self.time_units})', 'Mean Intensity'])
                for t, s in zip(self.time, self.signal):
                    writer.writerow([f'{t:.3f}', f'{s:.6f}'])

            self.status_bar.showMessage(f"Exported to: {csv_file}")
            QMessageBox.information(self, "Export Complete",
                                  f"Data exported to:\n{csv_file}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    def _accept_time(self):
        """Accept the selected injection time."""
        self.time_selected.emit(self.injection_index)
        self.accept()

    def get_injection_index(self) -> int:
        """Get the selected injection index."""
        return self.injection_index


class FitResultsDialog(QDialog):
    """Qt dialog for displaying fit results."""

    def __init__(self, time: np.ndarray, signal: np.ndarray,
                 fitted_signal: np.ndarray, fit_results: Dict,
                 save_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.time = time
        self.signal = signal
        self.fitted_signal = fitted_signal
        self.fit_results = fit_results
        self.save_path = save_path

        self.setWindowTitle("ProxylFit - Kinetic Model Fitting Results")
        self.setMinimumSize(1000, 750)
        self.resize(1100, 850)

        self._setup_ui()
        self._setup_plot()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = HeaderWidget("Kinetic Model Fitting Results",
                             "Extended Proxyl kinetic model fit")
        layout.addWidget(header)

        # Main content
        content_layout = QHBoxLayout()

        # Canvas with toolbar
        canvas_layout = QVBoxLayout()
        self.canvas = MatplotlibCanvas(self, width=9, height=7)
        self.toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        content_layout.addLayout(canvas_layout, stretch=3)

        # Parameters panel
        params_layout = QVBoxLayout()

        # Fitted parameters
        params_group = QGroupBox("Fitted Parameters")
        params_grid = QGridLayout(params_group)

        params = [
            ("kb (buildup)", f"{self.fit_results['kb']:.4f} ± {self.fit_results['kb_error']:.4f}"),
            ("kd (decay)", f"{self.fit_results['kd']:.4f} ± {self.fit_results['kd_error']:.4f}"),
            ("knt (non-tracer)", f"{self.fit_results['knt']:.4f} ± {self.fit_results['knt_error']:.4f}"),
            ("A0 (baseline)", f"{self.fit_results['A0']:.3f} ± {self.fit_results['A0_error']:.3f}"),
            ("A1 (amplitude)", f"{self.fit_results['A1']:.3f} ± {self.fit_results['A1_error']:.3f}"),
            ("t0 (onset)", f"{self.fit_results['t0']:.2f} ± {self.fit_results['t0_error']:.2f}"),
        ]

        for i, (name, value) in enumerate(params):
            params_grid.addWidget(QLabel(f"{name}:"), i, 0)
            params_grid.addWidget(QLabel(value), i, 1)

        params_layout.addWidget(params_group)

        # Fit quality
        quality_group = QGroupBox("Fit Quality")
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.addWidget(QLabel(f"R²: {self.fit_results['r_squared']:.4f}"))
        quality_layout.addWidget(QLabel(f"RMSE: {self.fit_results['rmse']:.3f}"))
        params_layout.addWidget(quality_group)

        params_layout.addStretch()
        content_layout.addLayout(params_layout, stretch=1)

        layout.addLayout(content_layout)

        # Button bar
        button_bar = ButtonBar()
        button_bar.add_button("save", "Save Plot", self._save_plot, "export")
        button_bar.add_stretch()
        button_bar.add_button("close", "Close", self.accept, "default")
        layout.addWidget(button_bar)

    def _setup_plot(self):
        """Set up the fit results plot."""
        # Create subplots
        gs = self.canvas.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax1 = self.canvas.fig.add_subplot(gs[0])
        self.ax2 = self.canvas.fig.add_subplot(gs[1])

        # Main plot: signal and fit
        self.ax1.plot(self.time, self.signal, 'bo-', markersize=4,
                     linewidth=2, label='Data')
        self.ax1.plot(self.time, self.fitted_signal, 'r-',
                     linewidth=2, label='Fitted Model')
        self.ax1.set_ylabel('Signal Intensity')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Extended Proxyl Kinetic Model Fit')

        # Residuals plot
        residuals = self.signal - self.fitted_signal
        self.ax2.plot(self.time, residuals, 'go-', markersize=3)
        self.ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        self.ax2.set_xlabel(f'Time ({self.fit_results.get("time_units", "minutes")})')
        self.ax2.set_ylabel('Residuals')
        self.ax2.set_title('Fit Residuals')
        self.ax2.grid(True, alpha=0.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _save_plot(self):
        """Save the plot to file."""
        if self.save_path:
            save_path = self.save_path
        else:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", "kinetic_fit.png",
                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
            )

        if save_path:
            self.canvas.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Save Complete",
                                  f"Plot saved to:\n{save_path}")


# ============================================================================
# Convenience Functions (Drop-in replacements for matplotlib-based functions)
# ============================================================================

def select_rectangle_roi_qt(image_4d: np.ndarray, z_index: int) -> np.ndarray:
    """
    Qt-based interactive selection of rectangular ROI.

    Drop-in replacement for select_rectangle_roi().
    """
    app = init_qt_app()

    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")

    image_slice = image_4d[:, :, z_index, 0]
    title = f"ROI Selection - Slice {z_index} (Timepoint 0)"

    dialog = ROISelectorDialog(image_slice, title)
    result = dialog.exec()

    if result == QDialog.Accepted:
        mask = dialog.get_mask()
        stats = dialog.get_stats()
        if stats:
            print(f"\nROI Statistics:")
            print(f"  Number of pixels: {stats['num_pixels']}")
            print(f"  Mean intensity: {stats['mean_intensity']:.2f}")
            print(f"  Std intensity: {stats['std_intensity']:.2f}")
        return mask
    else:
        print("ROI selection cancelled")
        return np.zeros(image_slice.shape, dtype=bool)


def select_manual_contour_roi_qt(image_4d: np.ndarray, z_index: int) -> np.ndarray:
    """
    Qt-based interactive manual contour ROI selection.

    Drop-in replacement for select_manual_contour_roi().
    """
    app = init_qt_app()

    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")

    dialog = ManualContourDialog(image_4d, z_index)
    result = dialog.exec()

    if result == QDialog.Accepted:
        mask = dialog.get_mask()
        final_z = dialog.get_z_index()

        if mask is not None and np.any(mask):
            num_pixels = np.sum(mask)
            print(f"\nManual Contour ROI Statistics (z-slice {final_z}):")
            print(f"  Number of pixels: {num_pixels}")
        return mask if mask is not None else np.zeros(image_4d[:, :, z_index, 0].shape, dtype=bool)
    else:
        print("ROI selection cancelled")
        return np.zeros(image_4d[:, :, z_index, 0].shape, dtype=bool)


def select_injection_time_qt(time: np.ndarray, signal: np.ndarray,
                            time_units: str = 'minutes',
                            output_dir: str = './output') -> int:
    """
    Qt-based interactive injection time selection.

    Drop-in replacement for select_injection_time().
    """
    app = init_qt_app()

    dialog = InjectionTimeSelectorDialog(time, signal, time_units, output_dir)
    result = dialog.exec()

    injection_index = dialog.get_injection_index()

    if result == QDialog.Accepted:
        print(f"Injection time set: {time[injection_index]:.1f} {time_units}")
    else:
        print(f"Selection cancelled, using default: {time[injection_index]:.1f} {time_units}")

    return injection_index


def plot_fit_results_qt(time: np.ndarray, signal: np.ndarray,
                       fitted_signal: np.ndarray, fit_results: Dict,
                       save_path: Optional[str] = None) -> None:
    """
    Qt-based fit results visualization.

    Drop-in replacement for plot_fit_results().
    """
    app = init_qt_app()

    dialog = FitResultsDialog(time, signal, fitted_signal, fit_results, save_path)

    # Auto-save if path provided
    if save_path:
        dialog.canvas.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    dialog.exec()
