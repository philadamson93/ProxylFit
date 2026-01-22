"""
ROI selection dialogs for ProxylFit.
"""

from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox,
    QStatusBar, QSlider, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath

from .styles import init_qt_app
from .components import MatplotlibCanvas, HeaderWidget, InstructionWidget, InfoWidget, ButtonBar


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
