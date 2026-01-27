"""
Parameter map options dialog (T014).

Provides options for:
- Single-slice mode vs. all slices
- ROI-only processing (reuse or redraw)
- Kernel configuration
- Results viewer with metrics
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox,
    QSpinBox, QSlider, QMessageBox, QWidget, QCheckBox, QFileDialog,
    QComboBox, QRadioButton, QButtonGroup, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import init_qt_app
from .components import HeaderWidget


class ParameterMappingWorker(QThread):
    """Background worker for running parameter mapping without blocking the UI."""

    progress = Signal(float, int, int)  # progress_pct, current, total
    finished = Signal(object)  # param_maps dict
    error = Signal(str)  # error message

    def __init__(self, registered_4d, time_array, options, roi_mask=None,
                 injection_idx=None, time_units='minutes', parent=None):
        super().__init__(parent)
        self.registered_4d = registered_4d
        self.time_array = time_array
        self.options = options
        self.roi_mask = roi_mask
        self.injection_idx = injection_idx
        self.time_units = time_units
        self._is_cancelled = False

    def run(self):
        """Run parameter mapping in background thread."""
        try:
            from ..parameter_mapping import create_parameter_maps

            param_maps = create_parameter_maps(
                registered_4d=self.registered_4d,
                time_array=self.time_array,
                window_size=self.options['window_size'],
                z_slice=self.options['z_slice'],
                time_units=self.time_units,
                progress_callback=self._emit_progress,
                roi_mask=self.roi_mask,
                kernel_type=self.options['kernel_type'],
                injection_time_index=self.injection_idx
            )

            if not self._is_cancelled:
                self.finished.emit(param_maps)
        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))

    def _emit_progress(self, progress_pct, current, total):
        """Emit progress signal."""
        self.progress.emit(progress_pct, current, total)

    def cancel(self):
        """Request cancellation."""
        self._is_cancelled = True


class ParameterMappingProgressDialog(QDialog):
    """Progress dialog shown during parameter mapping."""

    def __init__(self, registered_4d, time_array, options, roi_mask=None,
                 injection_idx=None, time_units='minutes', parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProxylFit - Parameter Mapping Progress")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)

        # Results stored here after completion
        self.param_maps = None
        self.error_message = None

        self._setup_ui()
        self._start_worker(registered_4d, time_array, options, roi_mask,
                          injection_idx, time_units)

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Creating Parameter Maps")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # Status message
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% complete")
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Position count
        self.position_label = QLabel("")
        self.position_label.setStyleSheet("color: #666;")
        layout.addWidget(self.position_label)

        layout.addStretch()

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumSize(100, 35)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _start_worker(self, registered_4d, time_array, options, roi_mask,
                     injection_idx, time_units):
        """Start the background worker."""
        self.worker = ParameterMappingWorker(
            registered_4d, time_array, options, roi_mask,
            injection_idx, time_units, self
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, progress_pct, current, total):
        """Handle progress update."""
        self.progress_bar.setValue(int(progress_pct))
        self.status_label.setText(f"Fitting kinetic model at each position...")
        self.position_label.setText(f"Position {current} of {total}")

    def _on_finished(self, param_maps):
        """Handle successful completion."""
        self.param_maps = param_maps
        self.accept()

    def _on_error(self, error_msg):
        """Handle error."""
        self.error_message = error_msg
        QMessageBox.critical(self, "Error", f"Parameter mapping failed:\n{error_msg}")
        self.reject()

    def _on_cancel(self):
        """Handle cancel button."""
        if hasattr(self, 'worker'):
            self.worker.cancel()
        self.reject()

    def closeEvent(self, event):
        """Handle dialog close."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().closeEvent(event)


class ParameterMapOptionsDialog(QDialog):
    """
    Dialog for configuring parameter map generation options.

    Features:
    - Single-slice vs. all slices mode
    - ROI-only processing with reuse/redraw option
    - Kernel type and size configuration
    """

    def __init__(self,
                 max_z: int = 8,
                 current_z: int = 4,
                 existing_roi: Optional[np.ndarray] = None,
                 existing_injection_idx: Optional[int] = None,
                 default_window_size: Tuple[int, int, int] = (15, 15, 3),
                 parent=None):
        super().__init__(parent)
        self.max_z = max_z
        self.current_z = current_z
        self.existing_roi = existing_roi
        self.existing_injection_idx = existing_injection_idx
        self.default_window_size = default_window_size

        self.result = None

        self.setWindowTitle("Parameter Map Options")
        self.setMinimumSize(500, 550)
        self.resize(550, 600)

        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header = HeaderWidget("Parameter Map Options", "Configure spatial parameter mapping")
        layout.addWidget(header)

        # Slice selection
        self._create_slice_section(layout)

        # ROI options
        self._create_roi_section(layout)

        # Kernel configuration
        self._create_kernel_section(layout)

        # Injection time
        self._create_injection_section(layout)

        layout.addStretch()

        # Buttons
        self._create_buttons(layout)

    def _create_slice_section(self, parent_layout):
        """Create slice selection section."""
        group = QGroupBox("Slice Selection")
        layout = QVBoxLayout(group)

        # All slices vs single slice
        self.all_slices_radio = QRadioButton("Process all slices (slower, full 3D maps)")
        self.single_slice_radio = QRadioButton("Single slice mode (faster)")
        self.all_slices_radio.setChecked(True)

        layout.addWidget(self.all_slices_radio)
        layout.addWidget(self.single_slice_radio)

        # Z-slice selection for single slice mode
        z_layout = QHBoxLayout()
        z_layout.addSpacing(20)
        z_layout.addWidget(QLabel("Z-slice:"))

        self.z_spinbox = QSpinBox()
        self.z_spinbox.setMinimum(0)
        self.z_spinbox.setMaximum(self.max_z)
        self.z_spinbox.setValue(self.current_z)
        self.z_spinbox.setEnabled(False)
        z_layout.addWidget(self.z_spinbox)

        self.z_max_label = QLabel(f"/ {self.max_z}")
        z_layout.addWidget(self.z_max_label)
        z_layout.addStretch()

        layout.addLayout(z_layout)

        # Connect radio buttons
        self.single_slice_radio.toggled.connect(self._on_slice_mode_changed)

        parent_layout.addWidget(group)

    def _create_roi_section(self, parent_layout):
        """Create ROI options section."""
        group = QGroupBox("ROI Processing")
        layout = QVBoxLayout(group)

        # Whole image vs ROI only
        self.whole_image_radio = QRadioButton("Process whole image")
        self.roi_only_radio = QRadioButton("Process within ROI only (faster)")
        self.whole_image_radio.setChecked(True)

        layout.addWidget(self.whole_image_radio)
        layout.addWidget(self.roi_only_radio)

        # ROI options (enabled when ROI only is selected)
        roi_options_layout = QHBoxLayout()
        roi_options_layout.addSpacing(20)

        self.reuse_roi_radio = QRadioButton("Reuse existing ROI")
        self.redraw_roi_radio = QRadioButton("Draw new ROI")

        # Enable reuse only if we have an existing ROI
        self.reuse_roi_radio.setEnabled(self.existing_roi is not None)
        if self.existing_roi is not None:
            self.reuse_roi_radio.setChecked(True)
            num_pixels = int(np.sum(self.existing_roi))
            self.reuse_roi_radio.setText(f"Reuse existing ROI ({num_pixels} pixels)")
        else:
            self.redraw_roi_radio.setChecked(True)

        # Group these together
        self.roi_action_group = QButtonGroup(self)
        self.roi_action_group.addButton(self.reuse_roi_radio)
        self.roi_action_group.addButton(self.redraw_roi_radio)

        roi_options_layout.addWidget(self.reuse_roi_radio)
        roi_options_layout.addWidget(self.redraw_roi_radio)
        roi_options_layout.addStretch()

        self.roi_options_widget = QWidget()
        self.roi_options_widget.setLayout(roi_options_layout)
        self.roi_options_widget.setEnabled(False)
        layout.addWidget(self.roi_options_widget)

        # Connect signals
        self.roi_only_radio.toggled.connect(self._on_roi_mode_changed)

        parent_layout.addWidget(group)

    def _create_kernel_section(self, parent_layout):
        """Create kernel configuration section."""
        group = QGroupBox("Kernel Configuration")
        layout = QVBoxLayout(group)

        # Kernel type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Kernel type:"))

        self.kernel_type_combo = QComboBox()
        self.kernel_type_combo.addItems(["sliding_window", "gaussian", "uniform", "box"])
        self.kernel_type_combo.setCurrentText("sliding_window")
        type_layout.addWidget(self.kernel_type_combo)
        type_layout.addStretch()

        layout.addLayout(type_layout)

        # Kernel size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Window size:"))

        self.window_x_spin = QSpinBox()
        self.window_x_spin.setRange(3, 31)
        self.window_x_spin.setValue(self.default_window_size[0])
        self.window_x_spin.setSingleStep(2)
        size_layout.addWidget(self.window_x_spin)

        size_layout.addWidget(QLabel("x"))

        self.window_y_spin = QSpinBox()
        self.window_y_spin.setRange(3, 31)
        self.window_y_spin.setValue(self.default_window_size[1])
        self.window_y_spin.setSingleStep(2)
        size_layout.addWidget(self.window_y_spin)

        size_layout.addWidget(QLabel("x"))

        self.window_z_spin = QSpinBox()
        self.window_z_spin.setRange(1, 9)
        self.window_z_spin.setValue(self.default_window_size[2])
        size_layout.addWidget(self.window_z_spin)

        size_layout.addWidget(QLabel("voxels"))
        size_layout.addStretch()

        layout.addLayout(size_layout)

        parent_layout.addWidget(group)

    def _create_injection_section(self, parent_layout):
        """Create injection time section."""
        group = QGroupBox("Injection Time")
        layout = QVBoxLayout(group)

        # Reuse existing vs select new
        self.reuse_injection_radio = QRadioButton("Reuse existing injection time")
        self.select_injection_radio = QRadioButton("Select injection time interactively")

        # Enable reuse only if we have an existing injection time
        if self.existing_injection_idx is not None:
            self.reuse_injection_radio.setEnabled(True)
            self.reuse_injection_radio.setText(f"Reuse existing injection time (index {self.existing_injection_idx})")
            self.reuse_injection_radio.setChecked(True)
        else:
            self.reuse_injection_radio.setEnabled(False)
            self.select_injection_radio.setChecked(True)

        layout.addWidget(self.reuse_injection_radio)
        layout.addWidget(self.select_injection_radio)

        parent_layout.addWidget(group)

    def _create_buttons(self, parent_layout):
        """Create action buttons."""
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumSize(100, 35)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        button_layout.addSpacing(15)

        # Run button
        run_btn = QPushButton("Run Parameter Mapping")
        run_btn.setMinimumSize(180, 40)
        run_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        run_btn.clicked.connect(self._on_run)
        button_layout.addWidget(run_btn)

        button_layout.addStretch()
        parent_layout.addLayout(button_layout)

    def _on_slice_mode_changed(self, checked: bool):
        """Handle slice mode radio button change."""
        self.z_spinbox.setEnabled(checked)

    def _on_roi_mode_changed(self, checked: bool):
        """Handle ROI mode radio button change."""
        self.roi_options_widget.setEnabled(checked)

    def _on_run(self):
        """Handle run button click."""
        self.result = {
            # Slice options
            'single_slice': self.single_slice_radio.isChecked(),
            'z_slice': self.z_spinbox.value() if self.single_slice_radio.isChecked() else None,

            # ROI options
            'roi_only': self.roi_only_radio.isChecked(),
            'reuse_roi': self.reuse_roi_radio.isChecked() if self.roi_only_radio.isChecked() else False,
            'redraw_roi': self.redraw_roi_radio.isChecked() if self.roi_only_radio.isChecked() else False,

            # Kernel options
            'kernel_type': self.kernel_type_combo.currentText(),
            'window_size': (
                self.window_x_spin.value(),
                self.window_y_spin.value(),
                self.window_z_spin.value()
            ),

            # Injection time
            'reuse_injection': self.reuse_injection_radio.isChecked(),
            'select_injection': self.select_injection_radio.isChecked()
        }
        self.accept()

    def get_result(self) -> Optional[dict]:
        """Get the dialog result."""
        return self.result


class ParameterMapResultsDialog(QDialog):
    """
    Dialog for viewing parameter map results with ROI metrics (T014).

    Features:
    - Parameter map visualization
    - ROI overlay
    - Overlay on anatomical image with opacity control
    - Metrics display (mean +/- std within ROI)
    - Export options
    """

    def __init__(self,
                 param_maps: Dict[str, np.ndarray],
                 spacing: Tuple[float, float, float],
                 roi_mask: Optional[np.ndarray] = None,
                 output_dir: str = './output',
                 reference_image: Optional[np.ndarray] = None,
                 source_dicom: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.param_maps = param_maps
        self.spacing = spacing
        self.roi_mask = roi_mask
        self.output_dir = output_dir
        self.reference_image = reference_image or param_maps.get('reference_slice')
        self.source_dicom = source_dicom

        # Get map dimensions
        self.kb_map = param_maps.get('kb_map', np.array([]))
        self.num_slices = self.kb_map.shape[2] if self.kb_map.ndim == 3 else 1
        self.current_z = self.num_slices // 2 if self.num_slices > 1 else 0

        # Current displayed map
        self.current_map = 'kb_map'

        # Colorbar reference (to remove when switching maps)
        self.colorbar = None

        # Overlay settings
        self.overlay_mode = False
        self.overlay_opacity = 0.7

        self.setWindowTitle("Parameter Map Results")
        self.setMinimumSize(900, 700)
        self.resize(950, 750)

        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Header
        header = HeaderWidget("Parameter Map Results", "View and export parameter maps")
        layout.addWidget(header)

        # Main content: left (map view) and right (metrics)
        content_layout = QHBoxLayout()

        # Left panel: map visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Map selection
        map_layout = QHBoxLayout()
        map_layout.addWidget(QLabel("Display:"))

        self.map_combo = QComboBox()
        self.map_combo.addItems([
            "kb (buildup rate)",
            "kd (decay rate)",
            "knt (non-tracer rate)",
            "R-squared (fit quality)",
            "A1 (tracer amplitude)",
            "A2 (non-tracer amplitude)"
        ])
        self.map_combo.currentIndexChanged.connect(self._on_map_changed)
        map_layout.addWidget(self.map_combo)
        map_layout.addStretch()

        left_layout.addLayout(map_layout)

        # Map figure
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        left_layout.addWidget(self.canvas)

        # Z-slice slider
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-slice:"))

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, max(0, self.num_slices - 1))
        self.z_slider.setValue(self.current_z)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_slider)

        self.z_label = QLabel(f"{self.current_z}/{max(0, self.num_slices - 1)}")
        z_layout.addWidget(self.z_label)

        left_layout.addLayout(z_layout)

        # ROI overlay checkbox
        roi_layout = QHBoxLayout()
        self.roi_checkbox = QCheckBox("Show ROI contour")
        self.roi_checkbox.setChecked(self.roi_mask is not None)
        self.roi_checkbox.setEnabled(self.roi_mask is not None)
        self.roi_checkbox.toggled.connect(self._update_display)
        roi_layout.addWidget(self.roi_checkbox)
        roi_layout.addStretch()

        left_layout.addLayout(roi_layout)

        # Overlay on anatomical image
        overlay_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Overlay on image")
        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.setEnabled(self.reference_image is not None)
        self.overlay_checkbox.toggled.connect(self._on_overlay_toggled)
        overlay_layout.addWidget(self.overlay_checkbox)

        overlay_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.setMaximumWidth(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_slider.setEnabled(False)
        overlay_layout.addWidget(self.opacity_slider)

        self.opacity_label = QLabel("70%")
        self.opacity_label.setMinimumWidth(35)
        overlay_layout.addWidget(self.opacity_label)
        overlay_layout.addStretch()

        left_layout.addLayout(overlay_layout)

        content_layout.addWidget(left_panel, stretch=2)

        # Right panel: metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # Metrics group
        metrics_group = QGroupBox("ROI Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_label = QLabel("Loading metrics...")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        metrics_layout.addWidget(self.metrics_label)

        right_layout.addWidget(metrics_group)

        # Processing info
        info_group = QGroupBox("Processing Info")
        info_layout = QVBoxLayout(info_group)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 10px; color: #666;")
        info_layout.addWidget(self.info_label)

        right_layout.addWidget(info_group)

        right_layout.addStretch()

        # Export buttons
        export_layout = QVBoxLayout()
        export_layout.addWidget(QLabel("Export:"))

        export_dicom_btn = QPushButton("Save as DICOM")
        export_dicom_btn.clicked.connect(lambda: self._export('dicom'))
        export_layout.addWidget(export_dicom_btn)

        export_npz_btn = QPushButton("Save as NPZ")
        export_npz_btn.clicked.connect(lambda: self._export('npz'))
        export_layout.addWidget(export_npz_btn)

        export_nifti_btn = QPushButton("Save as NIfTI")
        export_nifti_btn.clicked.connect(lambda: self._export('nifti'))
        export_layout.addWidget(export_nifti_btn)

        export_csv_btn = QPushButton("Export Metrics (CSV)")
        export_csv_btn.clicked.connect(self._export_metrics)
        export_layout.addWidget(export_csv_btn)

        right_layout.addLayout(export_layout)

        content_layout.addWidget(right_panel, stretch=1)

        layout.addLayout(content_layout)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumSize(100, 35)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Update info
        self._update_info()

    def _on_map_changed(self, index: int):
        """Handle map selection change."""
        map_keys = ['kb_map', 'kd_map', 'knt_map', 'r_squared_map', 'a1_amplitude_map', 'a2_amplitude_map']
        self.current_map = map_keys[index] if index < len(map_keys) else 'kb_map'
        self._update_display()

    def _on_z_changed(self, value: int):
        """Handle z-slice slider change."""
        self.current_z = value
        self.z_label.setText(f"{value}/{max(0, self.num_slices - 1)}")
        self._update_display()

    def _on_overlay_toggled(self, checked: bool):
        """Handle overlay checkbox toggle."""
        self.overlay_mode = checked
        self.opacity_slider.setEnabled(checked)
        self._update_display()

    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change."""
        self.overlay_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self._update_display()

    def _update_display(self):
        """Update the parameter map display."""
        # Remove old colorbar if it exists
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

        self.ax.clear()

        # Get the current map
        map_data = self.param_maps.get(self.current_map)
        mask = self.param_maps.get('mask')

        if map_data is None or mask is None:
            self.ax.set_title("No data available")
            self.canvas.draw()
            return

        # Get the slice
        if map_data.ndim == 3 and self.current_z < map_data.shape[2]:
            map_slice = map_data[:, :, self.current_z].T
            mask_slice = mask[:, :, self.current_z].T
        else:
            map_slice = map_data[:, :, 0].T if map_data.ndim == 3 else map_data.T
            mask_slice = mask[:, :, 0].T if mask.ndim == 3 else mask.T

        # Mask invalid values
        display_data = np.where(mask_slice, map_slice, np.nan)

        # Choose colormap
        if 'r_squared' in self.current_map:
            cmap = 'RdYlBu_r'
            vmin, vmax = 0, 1
        else:
            cmap = 'plasma'
            vmin, vmax = None, None

        # Show anatomical image as background if overlay mode is enabled
        if self.overlay_mode and self.reference_image is not None:
            # Get reference slice
            if self.reference_image.ndim == 3:
                ref_slice = self.reference_image[:, :, self.current_z].T
            else:
                ref_slice = self.reference_image.T

            # Show grayscale anatomical image
            self.ax.imshow(ref_slice, cmap='gray', origin='lower')

            # Overlay parameter map with transparency
            im = self.ax.imshow(display_data, cmap=cmap, origin='lower',
                               vmin=vmin, vmax=vmax, alpha=self.overlay_opacity)
        else:
            im = self.ax.imshow(display_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        # Add ROI contour if enabled
        if self.roi_checkbox.isChecked() and self.roi_mask is not None:
            if self.roi_mask.ndim == 2:
                roi_slice = self.roi_mask.T
            elif self.current_z < self.roi_mask.shape[2]:
                roi_slice = self.roi_mask[:, :, self.current_z].T
            else:
                roi_slice = None

            if roi_slice is not None and np.any(roi_slice):
                self.ax.contour(roi_slice, levels=[0.5], colors='cyan', linewidths=2)

        # Colorbar - store reference so we can remove it later
        self.colorbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046)

        # Title
        title_map = {
            'kb_map': 'kb (buildup rate)',
            'kd_map': 'kd (decay rate)',
            'knt_map': 'knt (non-tracer rate)',
            'r_squared_map': 'R-squared',
            'a1_amplitude_map': 'A1 (tracer amplitude)',
            'a2_amplitude_map': 'A2 (non-tracer amplitude)'
        }
        self.ax.set_title(f"{title_map.get(self.current_map, self.current_map)} (z={self.current_z})")
        self.ax.axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

        # Update metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update the metrics display."""
        if self.roi_mask is None:
            self.metrics_label.setText("No ROI available.\nDraw ROI to see metrics.")
            return

        # Get values within ROI for each parameter
        lines = []

        # Get ROI slice
        if self.roi_mask.ndim == 2:
            roi_slice = self.roi_mask
        elif self.current_z < self.roi_mask.shape[2]:
            roi_slice = self.roi_mask[:, :, self.current_z]
        else:
            self.metrics_label.setText("ROI not available for this slice")
            return

        mask = self.param_maps.get('mask')
        if mask is None:
            return

        # Combined mask: ROI and valid fits
        if mask.ndim == 3:
            combined_mask = roi_slice & mask[:, :, self.current_z]
        else:
            combined_mask = roi_slice & mask[:, :, 0]

        n_pixels = int(np.sum(combined_mask))
        lines.append(f"ROI + fitted: {n_pixels} pixels")
        lines.append(f"Z-slice: {self.current_z}")
        lines.append("")

        # Calculate metrics for each map
        param_names = [
            ('kb_map', 'kb (buildup)'),
            ('kd_map', 'kd (decay)'),
            ('knt_map', 'knt (non-tracer)'),
            ('r_squared_map', 'R-squared')
        ]

        for key, name in param_names:
            map_data = self.param_maps.get(key)
            if map_data is None:
                continue

            if map_data.ndim == 3:
                slice_data = map_data[:, :, self.current_z]
            else:
                slice_data = map_data[:, :, 0]

            roi_values = slice_data[combined_mask]
            if len(roi_values) > 0:
                mean_val = np.nanmean(roi_values)
                std_val = np.nanstd(roi_values)
                lines.append(f"{name}:")
                lines.append(f"  {mean_val:.4f} +/- {std_val:.4f}")

        self.metrics_label.setText('\n'.join(lines))

    def _update_info(self):
        """Update the processing info display."""
        metadata = self.param_maps.get('metadata', {})

        lines = []
        if 'kernel_type' in metadata:
            lines.append(f"Kernel: {metadata['kernel_type']}")
        if 'window_x' in metadata:
            lines.append(f"Window: {metadata['window_x']}x{metadata['window_y']}x{metadata['window_z']}")
        if 'success_rate' in metadata:
            lines.append(f"Success rate: {metadata['success_rate']:.1f}%")
        if 'processing_time' in metadata:
            lines.append(f"Time: {metadata['processing_time']:.1f}s")
        if 'total_positions' in metadata:
            lines.append(f"Positions: {metadata['total_positions']}")

        self.info_label.setText('\n'.join(lines))

    def _export(self, format_type: str):
        """Export parameter maps."""
        from ..io import get_dataset_path
        output_path = get_dataset_path(self.output_dir, 'parameter_maps')

        if format_type == 'dicom':
            from ..io import save_parameter_map_as_dicom

            folder = QFileDialog.getExistingDirectory(
                self, "Select Export Folder",
                str(output_path)
            )
            if folder:
                folder = Path(folder)
                metadata = self.param_maps.get('metadata', {})
                total_files = 0

                for key in ['kb_map', 'kd_map', 'knt_map', 'r_squared_map']:
                    data = self.param_maps.get(key)
                    if data is not None:
                        saved = save_parameter_map_as_dicom(
                            data, key, str(folder), self.spacing,
                            self.source_dicom, metadata
                        )
                        total_files += len(saved)

                QMessageBox.information(
                    self, "Exported",
                    f"Saved {total_files} DICOM files to:\n{folder}"
                )

        elif format_type == 'npz':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Parameter Maps",
                str(output_path / "parameter_maps.npz"),
                "NumPy Files (*.npz)"
            )
            if filepath:
                np.savez_compressed(filepath, **self.param_maps)
                QMessageBox.information(self, "Exported", f"Saved to:\n{filepath}")

        elif format_type == 'nifti':
            try:
                import nibabel as nib

                folder = QFileDialog.getExistingDirectory(
                    self, "Select Export Folder",
                    str(output_path)
                )
                if folder:
                    folder = Path(folder)
                    affine = np.diag([self.spacing[0], self.spacing[1], self.spacing[2], 1.0])

                    for key in ['kb_map', 'kd_map', 'knt_map', 'r_squared_map']:
                        data = self.param_maps.get(key)
                        if data is not None:
                            img = nib.Nifti1Image(data.astype(np.float32), affine)
                            nib.save(img, folder / f"{key}.nii.gz")

                    QMessageBox.information(self, "Exported", f"NIfTI files saved to:\n{folder}")

            except ImportError:
                QMessageBox.warning(self, "Missing Package",
                                   "NIfTI export requires nibabel. Install with: pip install nibabel")

    def _export_metrics(self):
        """Export metrics to CSV."""
        import csv

        from ..io import get_dataset_path
        output_path = get_dataset_path(self.output_dir, 'parameter_maps')

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics",
            str(output_path / "parameter_map_metrics.csv"),
            "CSV Files (*.csv)"
        )

        if not filepath:
            return

        # Calculate metrics for all slices
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['z_slice', 'parameter', 'n_pixels', 'mean', 'std', 'min', 'max'])

            for z in range(self.num_slices):
                # Get ROI for this slice
                if self.roi_mask is None:
                    continue

                if self.roi_mask.ndim == 2:
                    roi_slice = self.roi_mask
                else:
                    roi_slice = self.roi_mask[:, :, z]

                mask = self.param_maps.get('mask')
                if mask is None:
                    continue

                if mask.ndim == 3:
                    combined_mask = roi_slice & mask[:, :, z]
                else:
                    combined_mask = roi_slice & mask[:, :, 0]

                n_pixels = int(np.sum(combined_mask))
                if n_pixels == 0:
                    continue

                for key in ['kb_map', 'kd_map', 'knt_map', 'r_squared_map']:
                    map_data = self.param_maps.get(key)
                    if map_data is None:
                        continue

                    if map_data.ndim == 3:
                        slice_data = map_data[:, :, z]
                    else:
                        slice_data = map_data[:, :, 0]

                    roi_values = slice_data[combined_mask]
                    writer.writerow([
                        z, key.replace('_map', ''),
                        n_pixels,
                        np.nanmean(roi_values),
                        np.nanstd(roi_values),
                        np.nanmin(roi_values),
                        np.nanmax(roi_values)
                    ])

        QMessageBox.information(self, "Exported", f"Metrics exported to:\n{filepath}")


def show_parameter_map_options(max_z: int = 8,
                                current_z: int = 4,
                                existing_roi: Optional[np.ndarray] = None,
                                existing_injection_idx: Optional[int] = None,
                                default_window_size: Tuple[int, int, int] = (15, 15, 3)) -> Optional[dict]:
    """
    Show the parameter map options dialog.

    Parameters
    ----------
    max_z : int
        Maximum z-slice index
    current_z : int
        Current z-slice
    existing_roi : np.ndarray, optional
        Existing ROI mask from previous workflow
    existing_injection_idx : int, optional
        Existing injection time index
    default_window_size : tuple
        Default window size (x, y, z)

    Returns
    -------
    dict or None
        User's options, or None if cancelled
    """
    app = init_qt_app()

    dialog = ParameterMapOptionsDialog(
        max_z=max_z,
        current_z=current_z,
        existing_roi=existing_roi,
        existing_injection_idx=existing_injection_idx,
        default_window_size=default_window_size
    )

    result = dialog.exec()

    if result == QDialog.Accepted:
        return dialog.get_result()
    return None


def show_parameter_map_results(param_maps: Dict[str, np.ndarray],
                                spacing: Tuple[float, float, float],
                                roi_mask: Optional[np.ndarray] = None,
                                output_dir: str = './output',
                                source_dicom: Optional[str] = None) -> None:
    """
    Show the parameter map results viewer.

    Parameters
    ----------
    param_maps : dict
        Parameter maps from create_parameter_maps()
    spacing : tuple
        Voxel spacing (x, y, z)
    roi_mask : np.ndarray, optional
        ROI mask for overlay and metrics
    output_dir : str
        Output directory for exports
    source_dicom : str, optional
        Path to source DICOM for metadata in exports
    """
    app = init_qt_app()

    dialog = ParameterMapResultsDialog(
        param_maps=param_maps,
        spacing=spacing,
        roi_mask=roi_mask,
        output_dir=output_dir,
        source_dicom=source_dicom
    )

    dialog.exec()
