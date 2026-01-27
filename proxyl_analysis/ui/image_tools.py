"""
Image tools dialog for averaged and difference images (T002/T003).

Enhanced with:
- T012: DICOM export for derived images
- T013: ROI contour overlay and metrics display
"""

import csv
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox,
    QSpinBox, QSlider, QMessageBox, QWidget, QCheckBox, QFileDialog,
    QComboBox
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import init_qt_app
from .components import HeaderWidget


def compute_averaged_image(image_4d: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """
    Compute averaged image over specified frame range.

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    start_idx : int
        Starting frame index (inclusive)
    end_idx : int
        Ending frame index (inclusive)

    Returns
    -------
    np.ndarray
        Averaged 3D image [x, y, z]
    """
    return np.mean(image_4d[:, :, :, start_idx:end_idx + 1], axis=3)


def compute_difference_image(image_4d: np.ndarray,
                              region_a: Tuple[int, int],
                              region_b: Tuple[int, int]) -> np.ndarray:
    """
    Compute difference image: mean(region_b) - mean(region_a).

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    region_a : tuple
        (start_idx, end_idx) for Region A (inclusive)
    region_b : tuple
        (start_idx, end_idx) for Region B (inclusive)

    Returns
    -------
    np.ndarray
        Difference image [x, y, z]
    """
    avg_a = np.mean(image_4d[:, :, :, region_a[0]:region_a[1] + 1], axis=3)
    avg_b = np.mean(image_4d[:, :, :, region_b[0]:region_b[1] + 1], axis=3)
    return avg_b - avg_a


class ImageToolsDialog(QDialog):
    """
    Dialog for creating averaged and difference images from time series data.

    T002: Single region selection -> averaged image
    T003: Two region selection -> difference image (B - A)
    T012: DICOM export with descriptive filenames
    T013: ROI overlay and metrics display
    """

    def __init__(self,
                 image_4d: np.ndarray,
                 time_array: np.ndarray,
                 roi_signal: np.ndarray,
                 time_units: str = 'minutes',
                 output_dir: str = './output',
                 initial_mode: str = 'average',
                 roi_mask: Optional[np.ndarray] = None,
                 spacing: Optional[Tuple[float, float, float]] = None,
                 source_dicom: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.image_4d = image_4d
        self.time_array = time_array
        self.roi_signal = roi_signal
        self.time_units = time_units
        self.output_dir = output_dir
        self.num_timepoints = len(time_array)
        self.num_slices = image_4d.shape[2]

        # ROI overlay state (T013)
        self.roi_mask = roi_mask
        self.spacing = spacing or (1.0, 1.0, 1.0)
        self.source_dicom = source_dicom
        self.show_roi_overlay = roi_mask is not None

        # Metrics cache
        self.metrics = None

        # State
        self.mode = initial_mode  # 'average' or 'difference'
        self.current_z = self.num_slices // 2

        # Region selection state
        self.region_a_start = None
        self.region_a_end = None
        self.region_b_start = None
        self.region_b_end = None
        self.selecting_region = None  # 'a_start', 'a_end', 'b_start', 'b_end', or None

        # Preview image cache
        self.preview_image = None

        self.setWindowTitle("ProxylFit - Image Tools")
        self.setMinimumSize(1000, 700)
        self.resize(1100, 750)

        self._setup_ui()
        self._update_mode_ui()

        # Initialize time labels and auto-start selection for region A
        self._update_time_labels()
        self._start_selection('a')

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Header
        header = HeaderWidget("Image Tools", "Create averaged or difference images from time series")
        layout.addWidget(header)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))

        self.avg_mode_btn = QPushButton("Averaged Image")
        self.avg_mode_btn.setCheckable(True)
        self.avg_mode_btn.setChecked(self.mode == 'average')
        self.avg_mode_btn.clicked.connect(lambda: self._set_mode('average'))
        mode_layout.addWidget(self.avg_mode_btn)

        self.diff_mode_btn = QPushButton("Difference Image")
        self.diff_mode_btn.setCheckable(True)
        self.diff_mode_btn.setChecked(self.mode == 'difference')
        self.diff_mode_btn.clicked.connect(lambda: self._set_mode('difference'))
        mode_layout.addWidget(self.diff_mode_btn)

        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Main content: left (curve + controls) and right (preview)
        content_layout = QHBoxLayout()

        # Left panel: curve and region controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Time series plot
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout(pad=2)
        left_layout.addWidget(self.canvas)

        # Connect click events
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)

        # Region A controls
        self.region_a_group = QGroupBox("Region A (Blue)")
        region_a_layout = QHBoxLayout(self.region_a_group)

        region_a_layout.addWidget(QLabel("Start:"))
        self.region_a_start_spin = QSpinBox()
        self.region_a_start_spin.setRange(0, self.num_timepoints - 1)
        self.region_a_start_spin.valueChanged.connect(self._on_region_a_start_changed)
        region_a_layout.addWidget(self.region_a_start_spin)

        self.region_a_start_time_label = QLabel("(-- min)")
        self.region_a_start_time_label.setStyleSheet("color: #666; min-width: 60px;")
        region_a_layout.addWidget(self.region_a_start_time_label)

        region_a_layout.addWidget(QLabel("End:"))
        self.region_a_end_spin = QSpinBox()
        self.region_a_end_spin.setRange(0, self.num_timepoints - 1)
        self.region_a_end_spin.valueChanged.connect(self._on_region_a_end_changed)
        region_a_layout.addWidget(self.region_a_end_spin)

        self.region_a_end_time_label = QLabel("(-- min)")
        self.region_a_end_time_label.setStyleSheet("color: #666; min-width: 60px;")
        region_a_layout.addWidget(self.region_a_end_time_label)

        self.select_a_btn = QPushButton("Select on Plot")
        self.select_a_btn.clicked.connect(lambda: self._start_selection('a'))
        region_a_layout.addWidget(self.select_a_btn)

        self.clear_a_btn = QPushButton("Clear")
        self.clear_a_btn.clicked.connect(self._clear_region_a)
        region_a_layout.addWidget(self.clear_a_btn)

        region_a_layout.addStretch()
        left_layout.addWidget(self.region_a_group)

        # Region B controls (only visible in difference mode)
        self.region_b_group = QGroupBox("Region B (Red)")
        region_b_layout = QHBoxLayout(self.region_b_group)

        region_b_layout.addWidget(QLabel("Start:"))
        self.region_b_start_spin = QSpinBox()
        self.region_b_start_spin.setRange(0, self.num_timepoints - 1)
        self.region_b_start_spin.valueChanged.connect(self._on_region_b_start_changed)
        region_b_layout.addWidget(self.region_b_start_spin)

        self.region_b_start_time_label = QLabel("(-- min)")
        self.region_b_start_time_label.setStyleSheet("color: #666; min-width: 60px;")
        region_b_layout.addWidget(self.region_b_start_time_label)

        region_b_layout.addWidget(QLabel("End:"))
        self.region_b_end_spin = QSpinBox()
        self.region_b_end_spin.setRange(0, self.num_timepoints - 1)
        self.region_b_end_spin.valueChanged.connect(self._on_region_b_end_changed)
        region_b_layout.addWidget(self.region_b_end_spin)

        self.region_b_end_time_label = QLabel("(-- min)")
        self.region_b_end_time_label.setStyleSheet("color: #666; min-width: 60px;")
        region_b_layout.addWidget(self.region_b_end_time_label)

        self.select_b_btn = QPushButton("Select on Plot")
        self.select_b_btn.clicked.connect(lambda: self._start_selection('b'))
        region_b_layout.addWidget(self.select_b_btn)

        self.clear_b_btn = QPushButton("Clear")
        self.clear_b_btn.clicked.connect(self._clear_region_b)
        region_b_layout.addWidget(self.clear_b_btn)

        region_b_layout.addStretch()
        left_layout.addWidget(self.region_b_group)

        # Instructions
        self.instructions_label = QLabel("")
        self.instructions_label.setStyleSheet("color: #666; font-style: italic;")
        self.instructions_label.setWordWrap(True)
        left_layout.addWidget(self.instructions_label)

        left_layout.addStretch()
        content_layout.addWidget(left_panel, stretch=1)

        # Right panel: image preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Preview:"))

        # Preview figure
        self.preview_figure = Figure(figsize=(4, 4), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.preview_ax = self.preview_figure.add_subplot(111)
        self.preview_figure.tight_layout(pad=1)
        right_layout.addWidget(self.preview_canvas)

        # Z-slice slider
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-slice:"))

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, self.num_slices - 1)
        self.z_slider.setValue(self.current_z)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_slider)

        self.z_label = QLabel(f"{self.current_z}/{self.num_slices - 1}")
        z_layout.addWidget(self.z_label)

        right_layout.addLayout(z_layout)

        # Preview button
        self.preview_btn = QPushButton("Refresh Preview")
        self.preview_btn.clicked.connect(self._update_preview)
        self.preview_btn.setEnabled(False)
        right_layout.addWidget(self.preview_btn)

        # ROI overlay checkbox (T013)
        roi_layout = QHBoxLayout()
        self.roi_checkbox = QCheckBox("Show ROI contour")
        self.roi_checkbox.setChecked(self.show_roi_overlay)
        self.roi_checkbox.toggled.connect(self._on_roi_toggle)
        self.roi_checkbox.setEnabled(self.roi_mask is not None)
        roi_layout.addWidget(self.roi_checkbox)
        roi_layout.addStretch()
        right_layout.addLayout(roi_layout)

        # Metrics panel (T013)
        self.metrics_group = QGroupBox("ROI Metrics")
        metrics_layout = QVBoxLayout(self.metrics_group)
        self.metrics_label = QLabel("Select regions to see metrics")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        metrics_layout.addWidget(self.metrics_label)

        # Export metrics button
        self.export_metrics_btn = QPushButton("Export Metrics (CSV)")
        self.export_metrics_btn.clicked.connect(self._export_metrics)
        self.export_metrics_btn.setEnabled(False)
        metrics_layout.addWidget(self.export_metrics_btn)

        right_layout.addWidget(self.metrics_group)

        right_layout.addStretch()
        content_layout.addWidget(right_panel, stretch=1)

        layout.addLayout(content_layout)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_btn = QPushButton("Save Image")
        self.save_btn.setMinimumSize(120, 40)
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.save_btn.clicked.connect(self._save_image)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        button_layout.addSpacing(15)

        exit_btn = QPushButton("Return to Menu")
        exit_btn.setMinimumSize(120, 40)
        exit_btn.clicked.connect(self.accept)
        button_layout.addWidget(exit_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Initial plot
        self._update_plot()

    def _set_mode(self, mode: str):
        """Switch between average and difference modes."""
        self.mode = mode
        self._update_mode_ui()
        self._update_plot()
        self._check_can_preview()

    def _update_mode_ui(self):
        """Update UI based on current mode."""
        self.avg_mode_btn.setChecked(self.mode == 'average')
        self.diff_mode_btn.setChecked(self.mode == 'difference')

        # Show/hide region B controls
        self.region_b_group.setVisible(self.mode == 'difference')

        # Update region A label
        if self.mode == 'average':
            self.region_a_group.setTitle("Region (frames to average)")
            self.instructions_label.setText(
                "Click 'Select on Plot' then click twice on the curve to set start and end points. "
                "Or edit the values directly in the spinboxes."
            )
        else:
            self.region_a_group.setTitle("Region A (Blue) - subtracted from B")
            self.instructions_label.setText(
                "Select two regions. Result = mean(Region B) - mean(Region A). "
                "Click 'Select on Plot' for each region, then click twice on the curve."
            )

    def _start_selection(self, region: str):
        """Start interactive selection for a region."""
        if region == 'a':
            self.selecting_region = 'a_start'
            self.select_a_btn.setText("Click START...")
            self.select_a_btn.setStyleSheet("background-color: #2196F3; color: white;")
        else:
            self.selecting_region = 'b_start'
            self.select_b_btn.setText("Click START...")
            self.select_b_btn.setStyleSheet("background-color: #F44336; color: white;")

    def _on_plot_click(self, event):
        """Handle click on the time series plot."""
        if event.inaxes != self.ax or self.selecting_region is None:
            return

        # Find nearest time index
        click_x = event.xdata
        idx = np.argmin(np.abs(self.time_array - click_x))

        if self.selecting_region == 'a_start':
            self.region_a_start = idx
            self.region_a_start_spin.setValue(idx)
            self.selecting_region = 'a_end'
            self.select_a_btn.setText("Click END...")
        elif self.selecting_region == 'a_end':
            self.region_a_end = idx
            # Ensure start <= end
            if self.region_a_end < self.region_a_start:
                self.region_a_start, self.region_a_end = self.region_a_end, self.region_a_start
            self.region_a_start_spin.setValue(self.region_a_start)
            self.region_a_end_spin.setValue(self.region_a_end)
            self.selecting_region = None
            self.select_a_btn.setText("Select on Plot")
            self.select_a_btn.setStyleSheet("")
        elif self.selecting_region == 'b_start':
            self.region_b_start = idx
            self.region_b_start_spin.setValue(idx)
            self.selecting_region = 'b_end'
            self.select_b_btn.setText("Click END...")
        elif self.selecting_region == 'b_end':
            self.region_b_end = idx
            # Ensure start <= end
            if self.region_b_end < self.region_b_start:
                self.region_b_start, self.region_b_end = self.region_b_end, self.region_b_start
            self.region_b_start_spin.setValue(self.region_b_start)
            self.region_b_end_spin.setValue(self.region_b_end)
            self.selecting_region = None
            self.select_b_btn.setText("Select on Plot")
            self.select_b_btn.setStyleSheet("")

        self._update_plot()
        self._check_can_preview()

    def _format_time(self, index: int) -> str:
        """Format time value for display."""
        if index is None or index < 0 or index >= len(self.time_array):
            return "(-- min)"
        time_val = self.time_array[index]
        return f"({time_val:.1f} {self.time_units})"

    def _update_time_labels(self):
        """Update all time labels based on current spinbox values."""
        self.region_a_start_time_label.setText(self._format_time(self.region_a_start_spin.value()))
        self.region_a_end_time_label.setText(self._format_time(self.region_a_end_spin.value()))
        self.region_b_start_time_label.setText(self._format_time(self.region_b_start_spin.value()))
        self.region_b_end_time_label.setText(self._format_time(self.region_b_end_spin.value()))

    def _on_region_a_start_changed(self, value):
        """Handle region A start spinbox change."""
        self.region_a_start = value
        self.region_a_start_time_label.setText(self._format_time(value))
        self._update_plot()
        self._check_can_preview()

    def _on_region_a_end_changed(self, value):
        """Handle region A end spinbox change."""
        self.region_a_end = value
        self.region_a_end_time_label.setText(self._format_time(value))
        self._update_plot()
        self._check_can_preview()

    def _on_region_b_start_changed(self, value):
        """Handle region B start spinbox change."""
        self.region_b_start = value
        self.region_b_start_time_label.setText(self._format_time(value))
        self._update_plot()
        self._check_can_preview()

    def _on_region_b_end_changed(self, value):
        """Handle region B end spinbox change."""
        self.region_b_end = value
        self.region_b_end_time_label.setText(self._format_time(value))
        self._update_plot()
        self._check_can_preview()

    def _clear_region_a(self):
        """Clear region A selection."""
        self.region_a_start = None
        self.region_a_end = None
        self.selecting_region = None
        self.select_a_btn.setText("Select on Plot")
        self.select_a_btn.setStyleSheet("")
        self._update_plot()
        self._check_can_preview()

    def _clear_region_b(self):
        """Clear region B selection."""
        self.region_b_start = None
        self.region_b_end = None
        self.selecting_region = None
        self.select_b_btn.setText("Select on Plot")
        self.select_b_btn.setStyleSheet("")
        self._update_plot()
        self._check_can_preview()

    def _check_can_preview(self):
        """Check if we have enough data to preview, and auto-show if ready."""
        if self.mode == 'average':
            can_preview = (self.region_a_start is not None and
                          self.region_a_end is not None)
        else:
            can_preview = (self.region_a_start is not None and
                          self.region_a_end is not None and
                          self.region_b_start is not None and
                          self.region_b_end is not None)

        self.preview_btn.setEnabled(can_preview)
        self.save_btn.setEnabled(can_preview)

        # Auto-update preview when selection is complete
        if can_preview:
            self._update_preview()

    def _update_plot(self):
        """Update the time series plot with region highlights."""
        self.ax.clear()

        # Plot signal
        self.ax.plot(self.time_array, self.roi_signal, 'k-', linewidth=1.5)

        # Show start marker while selecting (before end is set) - Region A
        if self.region_a_start is not None and self.region_a_end is None:
            self.ax.axvline(self.time_array[self.region_a_start], color='blue',
                           linestyle='-', linewidth=2, alpha=0.8)
            self.ax.plot(self.time_array[self.region_a_start],
                        self.roi_signal[self.region_a_start],
                        'bo', markersize=10, label='Region A start')

        # Highlight region A (blue) - when both start and end are set
        if self.region_a_start is not None and self.region_a_end is not None:
            start_idx = min(self.region_a_start, self.region_a_end)
            end_idx = max(self.region_a_start, self.region_a_end)
            self.ax.axvspan(
                self.time_array[start_idx],
                self.time_array[end_idx],
                alpha=0.3, color='blue', label='Region A'
            )
            # Mark boundaries
            self.ax.axvline(self.time_array[start_idx], color='blue', linestyle='--', alpha=0.7)
            self.ax.axvline(self.time_array[end_idx], color='blue', linestyle='--', alpha=0.7)

        # Highlight region B (red) - only in difference mode
        if self.mode == 'difference':
            # Show start marker while selecting (before end is set) - Region B
            if self.region_b_start is not None and self.region_b_end is None:
                self.ax.axvline(self.time_array[self.region_b_start], color='red',
                               linestyle='-', linewidth=2, alpha=0.8)
                self.ax.plot(self.time_array[self.region_b_start],
                            self.roi_signal[self.region_b_start],
                            'ro', markersize=10, label='Region B start')

            # Highlight region B (red) - when both start and end are set
            if self.region_b_start is not None and self.region_b_end is not None:
                start_idx = min(self.region_b_start, self.region_b_end)
                end_idx = max(self.region_b_start, self.region_b_end)
                self.ax.axvspan(
                    self.time_array[start_idx],
                    self.time_array[end_idx],
                    alpha=0.3, color='red', label='Region B'
                )
                # Mark boundaries
                self.ax.axvline(self.time_array[start_idx], color='red', linestyle='--', alpha=0.7)
                self.ax.axvline(self.time_array[end_idx], color='red', linestyle='--', alpha=0.7)

        self.ax.set_xlabel(f'Time ({self.time_units})')
        self.ax.set_ylabel('Signal (ROI mean)')
        self.ax.set_title('ROI Time Series - Click to select region boundaries')

        if self.mode == 'difference' and (self.region_a_start is not None or self.region_b_start is not None):
            self.ax.legend(loc='upper right')

        self.figure.tight_layout(pad=2)
        self.canvas.draw()

    def _on_z_changed(self, value):
        """Handle z-slice slider change."""
        self.current_z = value
        self.z_label.setText(f"{value}/{self.num_slices - 1}")
        if self.preview_image is not None:
            self._show_preview()

    def _update_preview(self):
        """Compute and show preview image."""
        if self.mode == 'average':
            if self.region_a_start is None or self.region_a_end is None:
                return
            start_idx = min(self.region_a_start, self.region_a_end)
            end_idx = max(self.region_a_start, self.region_a_end)
            self.preview_image = compute_averaged_image(self.image_4d, start_idx, end_idx)
        else:
            if (self.region_a_start is None or self.region_a_end is None or
                self.region_b_start is None or self.region_b_end is None):
                return
            region_a = (min(self.region_a_start, self.region_a_end),
                       max(self.region_a_start, self.region_a_end))
            region_b = (min(self.region_b_start, self.region_b_end),
                       max(self.region_b_start, self.region_b_end))
            self.preview_image = compute_difference_image(self.image_4d, region_a, region_b)

        self._show_preview()

    def _show_preview(self):
        """Display the preview image with optional ROI overlay."""
        if self.preview_image is None:
            return

        self.preview_ax.clear()

        slice_data = self.preview_image[:, :, self.current_z].T

        if self.mode == 'average':
            im = self.preview_ax.imshow(slice_data, cmap='gray', aspect='equal', origin='lower')
            self.preview_ax.set_title(f'Averaged Image (z={self.current_z})')
        else:
            # Diverging colormap centered at 0
            vmax = np.max(np.abs(slice_data))
            vmin = -vmax
            im = self.preview_ax.imshow(slice_data, cmap='RdBu_r', aspect='equal',
                                        vmin=vmin, vmax=vmax, origin='lower')
            self.preview_ax.set_title(f'Difference B-A (z={self.current_z})')

        # Add ROI contour overlay (T013)
        if self.show_roi_overlay and self.roi_mask is not None:
            # Get the ROI mask for current z-slice
            if self.roi_mask.ndim == 3 and self.current_z < self.roi_mask.shape[2]:
                roi_slice = self.roi_mask[:, :, self.current_z].T
            elif self.roi_mask.ndim == 2:
                roi_slice = self.roi_mask.T
            else:
                roi_slice = None

            if roi_slice is not None and np.any(roi_slice):
                self.preview_ax.contour(roi_slice, levels=[0.5], colors='cyan', linewidths=2)

        self.preview_ax.axis('off')
        self.preview_figure.tight_layout(pad=1)
        self.preview_canvas.draw()

        # Update metrics
        self._update_metrics()

    def _save_image(self):
        """Save the computed image to disk with format selection (T012)."""
        if self.preview_image is None:
            self._update_preview()
            if self.preview_image is None:
                QMessageBox.warning(self, "No Image", "Please select regions first.")
                return

        # Build operation params and filename
        if self.mode == 'average':
            start_idx = min(self.region_a_start, self.region_a_end)
            end_idx = max(self.region_a_start, self.region_a_end)
            base_filename = f"avg_t{start_idx}-t{end_idx}_{end_idx - start_idx + 1}frames"
            operation_params = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_frames': end_idx - start_idx + 1,
                'time_range': (float(self.time_array[start_idx]), float(self.time_array[end_idx])),
                'time_units': self.time_units
            }
        else:
            a_start = min(self.region_a_start, self.region_a_end)
            a_end = max(self.region_a_start, self.region_a_end)
            b_start = min(self.region_b_start, self.region_b_end)
            b_end = max(self.region_b_start, self.region_b_end)
            base_filename = f"diff_t{a_start}-t{a_end}_minus_t{b_start}-t{b_end}"
            operation_params = {
                'region_a_start': a_start,
                'region_a_end': a_end,
                'region_b_start': b_start,
                'region_b_end': b_end,
                'description': f'frames {b_start}-{b_end} minus frames {a_start}-{a_end}',
                'time_units': self.time_units
            }

        # Use derived_images subdirectory per T011
        from ..io import get_dataset_path
        output_path = get_dataset_path(self.output_dir, 'derived_images')

        # Ask user for format and filename
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Derived Image",
            str(output_path / base_filename),
            "DICOM Files (*.dcm);;NumPy Files (*.npz);;NIfTI Files (*.nii.gz)"
        )

        if not filename:
            return

        filepath = Path(filename)

        # Save based on format
        if filepath.suffix == '.dcm':
            # DICOM export (T012) - saves one 2D file per z-slice
            from ..io import save_derived_image_as_dicom
            saved_files = save_derived_image_as_dicom(
                self.preview_image,
                str(filepath),
                'averaged' if self.mode == 'average' else 'difference',
                operation_params,
                self.spacing,
                self.source_dicom
            )
            # Show appropriate message based on number of files
            if len(saved_files) == 1:
                msg = f"Image saved to:\n{saved_files[0]}"
            else:
                msg = f"Saved {len(saved_files)} DICOM slices to:\n{Path(saved_files[0]).parent}"
            QMessageBox.information(self, "Saved", msg)
            return
        elif filepath.suffix == '.gz' or str(filepath).endswith('.nii.gz'):
            # NIfTI export
            try:
                import nibabel as nib
                affine = np.diag([self.spacing[0], self.spacing[1], self.spacing[2], 1.0])
                nifti_img = nib.Nifti1Image(self.preview_image, affine)
                nib.save(nifti_img, str(filepath))
            except ImportError:
                QMessageBox.warning(self, "Missing Package",
                                   "NIfTI export requires nibabel. Install with: pip install nibabel")
                return
        else:
            # Default: NPZ format
            np.savez(filepath, image=self.preview_image, **operation_params)

        QMessageBox.information(
            self, "Saved",
            f"Image saved to:\n{filepath}"
        )

    def _on_roi_toggle(self, checked: bool):
        """Handle ROI overlay checkbox toggle."""
        self.show_roi_overlay = checked
        if self.preview_image is not None:
            self._show_preview()

    def _calculate_metrics(self) -> Optional[Dict[str, Any]]:
        """Calculate metrics within ROI for current preview image (T013)."""
        if self.preview_image is None or self.roi_mask is None:
            return None

        # Get ROI values from the 3D preview image
        if self.roi_mask.ndim == 3:
            roi_values = self.preview_image[self.roi_mask > 0]
        elif self.roi_mask.ndim == 2:
            # 2D mask - apply to current z-slice only
            slice_data = self.preview_image[:, :, self.current_z]
            roi_values = slice_data[self.roi_mask > 0]
        else:
            return None

        if len(roi_values) == 0:
            return None

        metrics = {
            'n_pixels': len(roi_values),
            'mean': float(np.mean(roi_values)),
            'std': float(np.std(roi_values)),
            'min': float(np.min(roi_values)),
            'max': float(np.max(roi_values)),
            'z_slice': self.current_z if self.roi_mask.ndim == 2 else 'all'
        }

        # For difference images, also calculate metrics for each input region
        if self.mode == 'difference':
            a_start = min(self.region_a_start, self.region_a_end)
            a_end = max(self.region_a_start, self.region_a_end)
            b_start = min(self.region_b_start, self.region_b_end)
            b_end = max(self.region_b_start, self.region_b_end)

            avg_a = np.mean(self.image_4d[:, :, :, a_start:a_end + 1], axis=3)
            avg_b = np.mean(self.image_4d[:, :, :, b_start:b_end + 1], axis=3)

            if self.roi_mask.ndim == 3:
                region_a_vals = avg_a[self.roi_mask > 0]
                region_b_vals = avg_b[self.roi_mask > 0]
            else:
                region_a_vals = avg_a[:, :, self.current_z][self.roi_mask > 0]
                region_b_vals = avg_b[:, :, self.current_z][self.roi_mask > 0]

            metrics['region_a'] = {
                'timepoints': f't{a_start}-t{a_end}',
                'mean': float(np.mean(region_a_vals)),
                'std': float(np.std(region_a_vals))
            }
            metrics['region_b'] = {
                'timepoints': f't{b_start}-t{b_end}',
                'mean': float(np.mean(region_b_vals)),
                'std': float(np.std(region_b_vals))
            }

            # Percent change
            if metrics['region_a']['mean'] != 0:
                metrics['percent_change'] = (
                    (metrics['region_b']['mean'] - metrics['region_a']['mean'])
                    / metrics['region_a']['mean'] * 100
                )

        return metrics

    def _update_metrics(self):
        """Update the metrics display panel (T013)."""
        self.metrics = self._calculate_metrics()

        if self.metrics is None:
            if self.roi_mask is None:
                self.metrics_label.setText("No ROI available.\nDraw ROI in main workflow to see metrics.")
            else:
                self.metrics_label.setText("Select regions to see metrics")
            self.export_metrics_btn.setEnabled(False)
            return

        # Format metrics text
        lines = []
        lines.append(f"ROI: {self.metrics['n_pixels']} pixels")
        if self.metrics['z_slice'] != 'all':
            lines.append(f"Z-slice: {self.metrics['z_slice']}")
        lines.append("")

        if self.mode == 'average':
            lines.append(f"Mean:  {self.metrics['mean']:.2f} +/- {self.metrics['std']:.2f}")
            lines.append(f"Range: [{self.metrics['min']:.2f}, {self.metrics['max']:.2f}]")
        else:
            # Difference mode - show both regions
            ra = self.metrics.get('region_a', {})
            rb = self.metrics.get('region_b', {})

            lines.append(f"Region A ({ra.get('timepoints', '')}):")
            lines.append(f"  Mean: {ra.get('mean', 0):.2f} +/- {ra.get('std', 0):.2f}")
            lines.append("")
            lines.append(f"Region B ({rb.get('timepoints', '')}):")
            lines.append(f"  Mean: {rb.get('mean', 0):.2f} +/- {rb.get('std', 0):.2f}")
            lines.append("")
            lines.append(f"Difference (B - A):")
            lines.append(f"  Mean: {self.metrics['mean']:.2f} +/- {self.metrics['std']:.2f}")
            lines.append(f"  Range: [{self.metrics['min']:.2f}, {self.metrics['max']:.2f}]")

            if 'percent_change' in self.metrics:
                pct = self.metrics['percent_change']
                sign = '+' if pct > 0 else ''
                lines.append(f"  Change: {sign}{pct:.1f}%")

        self.metrics_label.setText('\n'.join(lines))
        self.export_metrics_btn.setEnabled(True)

    def _export_metrics(self):
        """Export metrics to CSV file (T013)."""
        if self.metrics is None:
            QMessageBox.warning(self, "No Metrics", "No metrics available to export.")
            return

        # Use derived_images subdirectory
        from ..io import get_dataset_path
        output_path = get_dataset_path(self.output_dir, 'derived_images')

        # Build default filename
        if self.mode == 'average':
            start_idx = min(self.region_a_start, self.region_a_end)
            end_idx = max(self.region_a_start, self.region_a_end)
            base_filename = f"avg_t{start_idx}-t{end_idx}_metrics.csv"
        else:
            a_start = min(self.region_a_start, self.region_a_end)
            a_end = max(self.region_a_start, self.region_a_end)
            b_start = min(self.region_b_start, self.region_b_end)
            b_end = max(self.region_b_start, self.region_b_end)
            base_filename = f"diff_t{a_start}-t{a_end}_minus_t{b_start}-t{b_end}_metrics.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics",
            str(output_path / base_filename),
            "CSV Files (*.csv)"
        )

        if not filepath:
            return

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'region', 'value', 'std', 'min', 'max'])

            if self.mode == 'average':
                writer.writerow(['mean_intensity', 'averaged', self.metrics['mean'],
                               self.metrics['std'], self.metrics['min'], self.metrics['max']])
            else:
                ra = self.metrics.get('region_a', {})
                rb = self.metrics.get('region_b', {})
                writer.writerow(['mean_intensity', 'region_a', ra.get('mean', ''), ra.get('std', ''), '', ''])
                writer.writerow(['mean_intensity', 'region_b', rb.get('mean', ''), rb.get('std', ''), '', ''])
                writer.writerow(['mean_intensity', 'difference', self.metrics['mean'],
                               self.metrics['std'], self.metrics['min'], self.metrics['max']])
                if 'percent_change' in self.metrics:
                    writer.writerow(['percent_change', 'difference', self.metrics['percent_change'], '', '', ''])

            writer.writerow(['n_pixels', 'roi', self.metrics['n_pixels'], '', '', ''])
            writer.writerow(['z_slice', 'roi', self.metrics['z_slice'], '', '', ''])
            writer.writerow(['operation', 'info', self.mode, '', '', ''])
            writer.writerow(['time_units', 'info', self.time_units, '', '', ''])

        QMessageBox.information(self, "Exported", f"Metrics exported to:\n{filepath}")


def show_image_tools_dialog(image_4d: np.ndarray,
                            time_array: np.ndarray,
                            roi_signal: np.ndarray,
                            time_units: str = 'minutes',
                            output_dir: str = './output',
                            initial_mode: str = 'average',
                            roi_mask: Optional[np.ndarray] = None,
                            spacing: Optional[Tuple[float, float, float]] = None,
                            source_dicom: Optional[str] = None) -> None:
    """
    Show the image tools dialog for creating averaged/difference images.

    Parameters
    ----------
    image_4d : np.ndarray
        4D image data [x, y, z, t]
    time_array : np.ndarray
        Time points
    roi_signal : np.ndarray
        ROI signal time series (for display)
    time_units : str
        Time units for display
    output_dir : str
        Directory to save images
    initial_mode : str
        'average' or 'difference'
    roi_mask : np.ndarray, optional
        ROI mask for overlay and metrics calculation (T013)
    spacing : tuple, optional
        Voxel spacing (x, y, z) for DICOM export (T012)
    source_dicom : str, optional
        Path to source DICOM for metadata in exports (T012)
    """
    app = init_qt_app()

    dialog = ImageToolsDialog(
        image_4d=image_4d,
        time_array=time_array,
        roi_signal=roi_signal,
        time_units=time_units,
        output_dir=output_dir,
        initial_mode=initial_mode,
        roi_mask=roi_mask,
        spacing=spacing,
        source_dicom=source_dicom
    )

    dialog.exec()
