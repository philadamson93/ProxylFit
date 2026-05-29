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
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QSlider, QMessageBox, QWidget, QCheckBox,
    QFileDialog, QComboBox, QRadioButton, QButtonGroup, QFrame,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import init_qt_app
from .components import HeaderWidget


# Percentile used to compute display ranges for the %Enhancement and %NTE
# colormaps. The 99th percentile gives a robust auto-scale: outlier voxels
# from poor fits (e.g., where A0 ≈ 0 makes the percent shoot up) saturate at
# the LUT extremes instead of squashing the typical-signal voxels into a
# narrow band. Edit if you want tighter or looser auto-range.
PERCENT_RANGE_PERCENTILE = 99.0

# Hard cap on the %NTE / %NTE_est LUT extent. Used for both maps so they
# display on the same color scale. Bumped from 15% → 20% after pinning A0
# to the pre-injection baseline — without A0 absorbing kinetic-term mismatch
# the recovered %NTE values are larger in magnitude and a 15% cap clipped
# real signal at the extremes. Voxels outside ±20% still saturate at the
# green / magenta ends. Set to None to disable the cap entirely; symmetric
# so the black midpoint of the diverging LUT stays exactly on 0.
NTE_RANGE_MAX = 20.0

# Display range for the kd (decay rate) map. kd is non-negative; values
# above the upper bound saturate at the brightest LUT color. Adjust if your
# data sits in a different regime.
KD_DISPLAY_MIN = 0.0
KD_DISPLAY_MAX = 0.15


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

            # Translate user-marked exclusions (FULL-array indices —
            # what the user clicked on the injection-time plot) to
            # post-injection-array space for create_parameter_maps,
            # which slices off the pre-injection portion internally.
            full_excl = self.options.get('excluded_indices') or []
            if self.injection_idx is not None and full_excl:
                excl_post = [
                    int(i) - int(self.injection_idx)
                    for i in full_excl
                    if int(i) >= int(self.injection_idx)
                ]
            else:
                excl_post = list(full_excl)

            param_maps = create_parameter_maps(
                registered_4d=self.registered_4d,
                time_array=self.time_array,
                window_size=self.options['window_size'],
                z_slice=self.options['z_slice'],
                time_units=self.time_units,
                progress_callback=self._emit_progress,
                roi_mask=self.roi_mask,
                kernel_type=self.options['kernel_type'],
                injection_time_index=self.injection_idx,
                stride=self.options.get('stride', 1),
                steady_state_time=self.options.get('steady_state_time'),
                excluded_indices=excl_post,
            )

            # Tag the metadata with the FULL-array indices too so the
            # measurement-ROI fit can re-use the same exclusion list
            # without having to re-translate from post space.
            if param_maps is not None:
                meta = param_maps.setdefault('metadata', {})
                meta['excluded_indices_full'] = list(full_excl)

            if not self._is_cancelled:
                self.finished.emit(param_maps)
        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))

    def _emit_progress(self, progress_pct, current, total):
        """Emit progress signal. Returns False to request cancellation."""
        if self._is_cancelled:
            return False
        self.progress.emit(progress_pct, current, total)
        return True

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
        self._cleanup_worker()
        self.reject()

    def _cleanup_worker(self):
        """Stop and wait for worker thread to finish."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("Cancelling... please wait")
            self.cancel_btn.setEnabled(False)
            # Disconnect signals to prevent callbacks after dialog closes
            try:
                self.worker.progress.disconnect()
                self.worker.finished.disconnect()
                self.worker.error.disconnect()
            except RuntimeError:
                pass  # Already disconnected
            # Wait for thread to finish (with timeout to prevent infinite hang)
            if not self.worker.wait(10000):  # 10 second timeout
                # Force terminate if still running (last resort)
                self.worker.terminate()
                self.worker.wait(1000)

    def closeEvent(self, event):
        """Handle dialog close."""
        self._cleanup_worker()
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
                 default_window_size: Tuple[int, int, int] = (15, 15, 1),
                 default_steady_state_time: float = 100.0,
                 time_units: str = 'minutes',
                 parent=None):
        super().__init__(parent)
        self.max_z = max_z
        self.current_z = current_z
        self.existing_roi = existing_roi
        self.existing_injection_idx = existing_injection_idx
        self.default_window_size = default_window_size
        # Default value (in time_units) for the NTE steady-state-time
        # spinbox. Mirrors the same control on the injection time
        # dialog so the user's prior choice carries forward.
        self._default_steady_state_time = float(default_steady_state_time)
        self._time_units = time_units

        self.result = None

        self.setWindowTitle("Parameter Map Options")
        # Bumped from 500×550 / 550×600 — adding the Fit Options group
        # plus the kernel section's three rows (type / size / stride)
        # made the original size cramped, especially on macOS where
        # QGroupBox titles eat ~14 px of vertical space per group.
        self.setMinimumSize(600, 720)
        self.resize(640, 780)

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

        # Fit options (separate group so model-fit toggles don't crowd
        # the kernel-geometry section)
        self._create_fit_options_section(layout)

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
        """Create kernel configuration section.

        Uses QFormLayout so the label column ("Kernel type:" / "Window
        size:" / "Pixel stride:") and the value widgets line up
        automatically — Qt handles label/value alignment, vertical
        spacing, and label-column width without any manual fiddling.
        Multi-widget rows (the W×H×D window-size triplet and the
        stride row with its hint) are wrapped in a small QHBoxLayout
        and added with addRow(label, layout).
        """
        group = QGroupBox("Kernel Configuration")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(12, 14, 12, 12)
        outer.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        # Kernel type
        self.kernel_type_combo = QComboBox()
        self.kernel_type_combo.addItems(["sliding_window", "gaussian", "uniform", "box"])
        self.kernel_type_combo.setCurrentText("sliding_window")
        self.kernel_type_combo.setMinimumWidth(160)
        form.addRow("Kernel type:", self.kernel_type_combo)

        # Window size — three spinboxes joined by × characters, all in a
        # single horizontal sub-row. Wrapping them in a QWidget gives
        # QFormLayout one widget to align against the label.
        size_widget = QWidget()
        size_row = QHBoxLayout(size_widget)
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(6)

        self.window_x_spin = QSpinBox()
        self.window_x_spin.setRange(3, 31)
        self.window_x_spin.setValue(self.default_window_size[0])
        self.window_x_spin.setSingleStep(2)
        self.window_x_spin.setFixedWidth(70)
        size_row.addWidget(self.window_x_spin)
        size_row.addWidget(QLabel("×"))

        self.window_y_spin = QSpinBox()
        self.window_y_spin.setRange(3, 31)
        self.window_y_spin.setValue(self.default_window_size[1])
        self.window_y_spin.setSingleStep(2)
        self.window_y_spin.setFixedWidth(70)
        size_row.addWidget(self.window_y_spin)
        size_row.addWidget(QLabel("×"))

        self.window_z_spin = QSpinBox()
        self.window_z_spin.setRange(1, 9)
        self.window_z_spin.setValue(self.default_window_size[2])
        self.window_z_spin.setFixedWidth(70)
        size_row.addWidget(self.window_z_spin)
        size_row.addSpacing(4)
        size_row.addWidget(QLabel("voxels"))
        size_row.addStretch()

        form.addRow("Window size:", size_widget)

        # Stride (downsampling step) — spinbox + inline hint.
        stride_widget = QWidget()
        stride_row = QHBoxLayout(stride_widget)
        stride_row.setContentsMargins(0, 0, 0, 0)
        stride_row.setSpacing(8)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 32)
        self.stride_spin.setValue(1)
        self.stride_spin.setSingleStep(1)
        self.stride_spin.setFixedWidth(70)
        self.stride_spin.setToolTip(
            "Skip pixels to trade resolution for speed.\n"
            "Stride=1: fit every pixel (full resolution)\n"
            "Stride=8: fit every 8th pixel (~64× faster)"
        )
        stride_row.addWidget(self.stride_spin)

        stride_hint = QLabel("(1 = full resolution, higher = faster / coarser)")
        stride_hint.setStyleSheet("color: #666; font-size: 11px;")
        stride_row.addWidget(stride_hint)
        stride_row.addStretch()

        form.addRow("Pixel stride:", stride_widget)

        outer.addLayout(form)
        parent_layout.addWidget(group)

    def _create_fit_options_section(self, parent_layout):
        """Create fit options section (model-fitting constraints).

        Kept separate from Kernel Configuration so per-pixel kernel
        geometry (size, stride, type) doesn't crowd the model-fit
        controls. Currently one input: NTE steady-state time. Sets
        the lower bound on the non-tracer rate constant knt so the
        fitted (1−exp(−knt·t)) term reaches ~95% of A2 within the
        user-specified window. Without this, knt can drift to ~0
        and inflate the reported A2 to absorb residuals — even
        though the tail of the signal is nowhere near saturation.
        The same value flows to the per-voxel parameter map fits
        and to the "Kinetic Fit on this ROI" button in the
        param-map dialog so all three views stay consistent.
        """
        group = QGroupBox("Fit Options")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(12, 14, 12, 12)
        outer.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        # NTE steady-state time spinbox. Range 10–500 in time_units;
        # default 100 (typical PROXYL upper bound for steady-state).
        # Bounds knt from below at ln(20)/t_steady so a 100-min
        # setting forces knt ≥ ~0.030/min.
        ss_widget = QWidget()
        ss_row = QHBoxLayout(ss_widget)
        ss_row.setContentsMargins(0, 0, 0, 0)
        ss_row.setSpacing(8)

        self.steady_state_spin = QSpinBox()
        self.steady_state_spin.setRange(10, 500)
        self.steady_state_spin.setValue(int(round(self._default_steady_state_time)))
        self.steady_state_spin.setSingleStep(5)
        self.steady_state_spin.setFixedWidth(110)
        self.steady_state_spin.setSuffix(f" {self._time_units}")
        self.steady_state_spin.setToolTip(
            "Maximum time after the signal peak at which the non-tracer\n"
            "effect should reach steady state (within ~5% of A2). Sets\n"
            "the lower bound on knt: knt ≥ ln(20)/t_steady. Typical\n"
            "values for in-vivo PROXYL data: 70–100 minutes. Without\n"
            "this constraint, knt can drift toward 0, inflating A2 to\n"
            "absorb residuals even when the tail isn't saturating."
        )
        ss_row.addWidget(self.steady_state_spin)

        ss_hint = QLabel("(knt ≥ ln(20)/t_steady — 95% of A2 by t_steady)")
        ss_hint.setStyleSheet("color: #666; font-size: 11px;")
        ss_row.addWidget(ss_hint)
        ss_row.addStretch()

        form.addRow("NTE steady-state time:", ss_widget)

        outer.addLayout(form)
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

            # Stride (downsampling)
            'stride': self.stride_spin.value(),

            # Fit options
            'steady_state_time': float(self.steady_state_spin.value()),

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
                 registered_4d: Optional[np.ndarray] = None,
                 registered_t2: Optional[np.ndarray] = None,
                 time_array: Optional[np.ndarray] = None,
                 dataset_dir: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.param_maps = param_maps
        self.spacing = spacing
        self.roi_mask = roi_mask
        self.output_dir = output_dir
        self.reference_image = reference_image or param_maps.get('reference_slice')
        self.source_dicom = source_dicom
        # Anatomical sources kept around so the Save-as-DICOM export can
        # optionally write the corresponding T1 baseline and T2 volumes
        # alongside the parameter maps. Either may be None if not available.
        self.registered_4d = registered_4d
        self.registered_t2 = registered_t2
        # Time array + dataset directory let the Measurement ROI panel's
        # "Kinetic Fit" button extract a per-ROI signal from registered_4d
        # and run fit_proxyl_kinetics in-place — same dialog the main
        # menu's kinetic_fit action opens. Both optional: the button is
        # only enabled when registered_4d AND time_array are present.
        self.time_array = time_array
        self.dataset_dir = dataset_dir

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

        # Optional second ROI drawn directly on a parameter map for
        # measurement-only purposes (separate from the fitting ROI used
        # during parameter mapping). 2D mask in (x, y) shape, applies to
        # whichever z-slice is currently displayed. None when no
        # measurement is active.
        self.measurement_roi_mask = None
        # z-slice index where the measurement ROI was originally drawn,
        # so the metrics panel can flag "drawn on z=3, viewing z=5" when
        # the user scrolls to a different slice.
        self.measurement_roi_drawn_z = None
        # Shared ROI counter N allocated when the measurement ROI is
        # drawn. Used by both the kinetic-fit bundle and the metrics
        # CSV so they share a per-ROI number. None when no ROI active.
        self._measurement_roi_n = None
        self._measurement_lasso = None

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
            "%Enhancement (A1/A0)",
            "%NTE (A2/A0)",
            "%NTE_est (A2_est/A0_est)",
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

        # Connect mouse hover for pixel readout
        self.canvas.mpl_connect('motion_notify_event', self._on_pixel_hover)

        # Pixel value readout label
        self.pixel_label = QLabel("Pixel: \u2014")
        self.pixel_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        left_layout.addWidget(self.pixel_label)

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

        # Measurement ROI tool — lets the user draw a lasso ROI on any
        # currently displayed parameter map (or anatomical underlay) and
        # read out mean ± std for every available map within it. This is
        # separate from the fitting ROI used during parameter mapping;
        # both are shown simultaneously when both exist.
        measure_group = QGroupBox("Measurement ROI")
        measure_layout = QVBoxLayout(measure_group)

        self.measure_status_label = QLabel(
            "Click 'Draw' then drag-and-release on the map."
        )
        self.measure_status_label.setStyleSheet("font-size: 10px; color: #666;")
        self.measure_status_label.setWordWrap(True)
        measure_layout.addWidget(self.measure_status_label)

        measure_btn_row = QHBoxLayout()
        self.draw_measure_btn = QPushButton("Draw")
        self.draw_measure_btn.clicked.connect(self._start_drawing_measurement_roi)
        measure_btn_row.addWidget(self.draw_measure_btn)

        self.clear_measure_btn = QPushButton("Clear")
        self.clear_measure_btn.clicked.connect(self._clear_measurement_roi)
        self.clear_measure_btn.setEnabled(False)
        measure_btn_row.addWidget(self.clear_measure_btn)
        measure_layout.addLayout(measure_btn_row)

        # Kinetic Fit button — runs fit_proxyl_kinetics on the signal
        # extracted from the drawn measurement ROI and opens the fit
        # results dialog modally. Disabled until both an ROI is drawn
        # AND the dialog was given the data needed to do the fit
        # (registered_4d + time_array).
        self.kinetic_fit_btn = QPushButton("Kinetic Fit on this ROI")
        self.kinetic_fit_btn.clicked.connect(self._run_kinetic_fit_on_measurement_roi)
        self.kinetic_fit_btn.setEnabled(False)
        measure_layout.addWidget(self.kinetic_fit_btn)

        right_layout.addWidget(measure_group)

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

        export_png_btn = QPushButton("Save as PNG")
        export_png_btn.clicked.connect(lambda: self._export('png'))
        export_layout.addWidget(export_png_btn)

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
        map_keys = [
            'kb_map', 'kd_map', 'knt_map', 'r_squared_map',
            'a1_percent_map', 'a2_percent_map', 'a2_percent_est_map',
        ]
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

    def _percent_display_range(self, key: str, mode: str,
                               max_limit: Optional[float] = None):
        """
        Compute (vmin, vmax) for a percent map, robust to outliers, and cache
        the result so cross-slice scrolling keeps a stable color scale.

        Parameters
        ----------
        key : str
            'a1_percent_map' or 'a2_percent_map'.
        mode : str
            'positive' for %Enhancement (vmin=0, vmax=p99), or 'symmetric'
            for %NTE (vmin=-p99(|data|), vmax=+p99(|data|)) with black at zero.
        max_limit : float, optional
            Hard cap on the absolute extent of the range. Useful for the
            %NTE map where even the percentile-clipped extent can be wider
            than the band of interest — capping keeps the colormap focused
            on that band and saturates outliers at the LUT extremes.

        Returns
        -------
        (vmin, vmax) : tuple of (float | None)
            Falls back to (None, None) — letting matplotlib auto-scale —
            when the map is empty or entirely NaN.
        """
        cache_attr = f'_range_cache_{key}'
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        data = self.param_maps.get(key)
        if data is None:
            result = (None, None)
        else:
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                result = (None, None)
            elif mode == 'symmetric':
                limit = float(np.percentile(np.abs(finite), PERCENT_RANGE_PERCENTILE))
                if max_limit is not None:
                    limit = min(limit, max_limit)
                result = (-limit, limit) if limit > 0 else (None, None)
            elif mode == 'positive':
                limit = float(np.percentile(finite, PERCENT_RANGE_PERCENTILE))
                if max_limit is not None:
                    limit = min(limit, max_limit)
                result = (0.0, limit) if limit > 0 else (None, None)
            else:
                raise ValueError(f"unknown range mode: {mode!r}")

        setattr(self, cache_attr, result)
        return result

    def _on_pixel_hover(self, event):
        """Handle mouse hover over parameter map canvas."""
        if event.inaxes != self.ax:
            self.pixel_label.setText("Pixel: \u2014")
            return

        map_data = self.param_maps.get(self.current_map)
        if map_data is None:
            self.pixel_label.setText("Pixel: \u2014")
            return

        # matplotlib can return None for xdata/ydata at axes edges
        if event.xdata is None or event.ydata is None:
            self.pixel_label.setText("Pixel: \u2014")
            return

        # Canvas xdata/ydata map to array indices after .T and origin='lower'
        # Display: map_data[:, :, z].T with origin='lower'
        # So event.xdata → x-index, event.ydata → y-index in the original array
        xi = int(round(event.xdata))
        yi = int(round(event.ydata))

        # Bounds check
        if xi < 0 or yi < 0 or xi >= map_data.shape[0] or yi >= map_data.shape[1]:
            self.pixel_label.setText("Pixel: \u2014")
            return

        # Look up value
        if map_data.ndim == 3 and self.current_z < map_data.shape[2]:
            value = map_data[xi, yi, self.current_z]
        elif map_data.ndim == 3:
            value = map_data[xi, yi, 0]
        else:
            value = map_data[xi, yi]

        # Format
        map_short = self.current_map.replace('_map', '')
        if np.isnan(value):
            self.pixel_label.setText(f"Pixel ({xi}, {yi}): {map_short} = \u2014")
        else:
            self.pixel_label.setText(f"Pixel ({xi}, {yi}): {map_short} = {value:.4f}")

    # ------------------------------------------------------------------
    # Measurement ROI (drawn on any displayed map)
    # ------------------------------------------------------------------

    def _start_drawing_measurement_roi(self):
        """Activate the lasso selector on the parameter map canvas.

        The user drag-and-releases to draw a freeform polygon. On release,
        ``_on_measurement_lasso_done`` converts the vertices into a 2D
        mask in (x, y) shape that matches every parameter map in
        ``self.param_maps``. Drawing replaces any previous measurement
        ROI; the fitting ROI from parameter mapping is unaffected.
        """
        from matplotlib.widgets import LassoSelector

        # If a previous lasso is still active (e.g., user clicked Draw
        # twice without finishing), drop it first so we don't stack
        # listeners.
        if self._measurement_lasso is not None:
            try:
                self._measurement_lasso.disconnect_events()
            except Exception:
                pass
            self._measurement_lasso = None

        self.measure_status_label.setText(
            "Drawing\u2026 drag a freeform shape on the map and release."
        )
        self.draw_measure_btn.setEnabled(False)

        self._measurement_lasso = LassoSelector(
            self.ax,
            onselect=self._on_measurement_lasso_done,
            useblit=True,
        )
        self.canvas.draw_idle()

    def _on_measurement_lasso_done(self, vertices):
        """Convert the lasso polygon into a (x, y) boolean mask."""
        from matplotlib.path import Path

        # Always tear down the lasso first so the user can re-arm Draw
        # even if this draw produced an empty mask.
        if self._measurement_lasso is not None:
            try:
                self._measurement_lasso.disconnect_events()
            except Exception:
                pass
            self._measurement_lasso = None
        self.draw_measure_btn.setEnabled(True)

        if not vertices or len(vertices) < 3:
            self.measure_status_label.setText(
                "Need at least 3 points \u2014 click 'Draw' to try again."
            )
            return

        # Param maps are indexed (x, y, z); event.xdata is the x-index
        # and event.ydata is the y-index after the .T transpose used in
        # _update_display. Build a (nx, ny) mask the same shape so it
        # plugs straight into the existing slice_data[mask] indexing.
        nx, ny = self.kb_map.shape[0], self.kb_map.shape[1]
        xs, ys = np.mgrid[0:nx, 0:ny]
        points = np.column_stack([xs.ravel(), ys.ravel()])
        path = Path(vertices)
        inside = path.contains_points(points).reshape(nx, ny)

        if not inside.any():
            self.measure_status_label.setText(
                "Polygon enclosed no whole pixels \u2014 click 'Draw' to retry."
            )
            return

        self.measurement_roi_mask = inside
        self.measurement_roi_drawn_z = self.current_z
        # Allocate one shared ROI counter N for this measurement ROI,
        # used by both the upcoming "Kinetic Fit on this ROI" save
        # bundle (kinetic_fit_results_<N>.csv) and "Export Metrics CSV"
        # (parameter_map_metric_<N>.csv). Pick the max free N across
        # both target dirs so neither output collides with anything
        # already on disk. Computed once per ROI draw, reset on Clear.
        self._measurement_roi_n = self._allocate_shared_roi_n()
        self.clear_measure_btn.setEnabled(True)
        # Only enable the kinetic-fit button when we actually have the
        # data needed to do the fit (registered_4d for signal extraction,
        # time_array for the time axis). Otherwise the click would just
        # error out, so keep it disabled.
        self.kinetic_fit_btn.setEnabled(
            self.registered_4d is not None
            and self.time_array is not None
        )
        n_pixels = int(inside.sum())
        self.measure_status_label.setText(
            f"ROI active: {n_pixels} pixels. Stats below; "
            f"click 'Clear' to remove."
        )

        # Redraws the contour and recomputes metrics.
        self._update_display()

    def _clear_measurement_roi(self):
        """Drop the measurement ROI and refresh the display + metrics."""
        self.measurement_roi_mask = None
        self.measurement_roi_drawn_z = None
        # Release the shared ROI counter so the next Draw allocates
        # a fresh N (and so Export Metrics, if accidentally invoked
        # before a new draw, doesn't reuse a stale number).
        self._measurement_roi_n = None
        self.clear_measure_btn.setEnabled(False)
        self.kinetic_fit_btn.setEnabled(False)
        self.measure_status_label.setText(
            "Click 'Draw' then drag-and-release on the map."
        )
        self._update_display()

    def _allocate_shared_roi_n(self):
        """Pick the next free ROI counter N for the measurement ROI.

        Used as the index for both the kinetic-fit bundle and the
        parameter-map metrics CSV the user can produce from this ROI.
        Returns the max next-free N across both target directories so
        the chosen N doesn't collide with anything already on disk in
        either folder.
        """
        from pathlib import Path
        from ..io import next_indexed_path
        import re

        if not self.dataset_dir:
            return 1

        kinetic_dir = Path(self.dataset_dir) / "kinetic_fits"
        metrics_dir = (Path(self.dataset_dir) / "parameter_maps"
                       / "parameter_map_metrics")
        candidates = []
        for parent, prefix, suffix in [
            (kinetic_dir, "kinetic_fit_results", ".csv"),
            (metrics_dir, "parameter_map_metric", ".csv"),
        ]:
            try:
                p = next_indexed_path(parent, prefix, suffix)
                m = re.match(rf'{prefix}_(\d+){re.escape(suffix)}$', p.name)
                if m:
                    candidates.append(int(m.group(1)))
            except Exception:
                continue
        return max(candidates) if candidates else 1

    def _run_kinetic_fit_on_measurement_roi(self):
        """Run kinetic fit on the drawn measurement ROI's signal.

        Pulls the signal time-series from registered_4d using the same
        single-slice-mean convention as the menu's kinetic_fit action,
        applies the same A0-pinning + pre-injection trimming, then
        opens FitResultsDialog modally. The dialog inherits the dataset
        directory so its Save button writes the per-ROI bundle into the
        same kinetic_fits/ subfolder as fits launched from the menu.
        """
        from PySide6.QtWidgets import QMessageBox
        from ..roi_selection import compute_roi_timeseries
        from ..model import fit_proxyl_kinetics
        from .fitting import plot_fit_results_qt

        if self.measurement_roi_mask is None:
            return  # button shouldn't be clickable, but defensive

        # Need the injection time index from when the parameter maps
        # were computed. It's stored in metadata; if absent the maps
        # were fit on the full curve and pinning A0 doesn't apply
        # cleanly. Fall back to fitting without pre-injection in that
        # case.
        metadata = self.param_maps.get('metadata', {}) or {}
        injection_idx = metadata.get('injection_time_index')
        time_units = metadata.get('time_units', 'minutes')
        steady_state_time = metadata.get('steady_state_time')
        # FULL-array exclusion list, mirrored from the parameter-map
        # options dialog. Translated below to post-injection space
        # before being handed to fit_proxyl_kinetics.
        excluded_full = metadata.get('excluded_indices_full') or []

        z = self.measurement_roi_drawn_z
        if z is None:
            z = self.current_z

        # Extract the time series for the drawn ROI.
        try:
            roi_signal = compute_roi_timeseries(
                self.registered_4d, self.measurement_roi_mask, z_slice=z,
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Kinetic fit failed",
                f"Could not extract ROI signal:\n{e}",
            )
            return

        # Trim to post-injection for the fit; keep pre-injection slice
        # for A0 pinning + display.
        if injection_idx is not None and injection_idx > 0:
            time_fit = self.time_array[injection_idx:]
            signal_fit = roi_signal[injection_idx:]
            pre_time = self.time_array[:injection_idx]
            pre_signal = roi_signal[:injection_idx]
        else:
            time_fit = self.time_array
            signal_fit = roi_signal
            pre_time = None
            pre_signal = None

        if len(signal_fit) < 8:
            QMessageBox.warning(
                self, "Kinetic fit failed",
                "Need at least 8 timepoints after injection for the fit.",
            )
            return

        # Run the fit.
        try:
            # Translate FULL-array exclusion indices to the
            # post-injection-array space the kinetic fit sees.
            if injection_idx is not None and excluded_full:
                excluded_post = [
                    int(i) - int(injection_idx)
                    for i in excluded_full
                    if int(i) >= int(injection_idx)
                ]
            else:
                excluded_post = list(excluded_full)
            kb, kd, knt, fitted_signal, fit_results = fit_proxyl_kinetics(
                time_fit, signal_fit, time_units,
                pre_injection_signal=pre_signal,
                steady_state_time=steady_state_time,
                excluded_indices=excluded_post,
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Kinetic fit failed",
                f"fit_proxyl_kinetics raised:\n{e}",
            )
            return

        # Open the kinetic fit results dialog (modal). Pass param_maps
        # + forced_n so _save_all (a) labels its output bundle with the
        # same N as the matching parameter_map_metric_<N>.csv export
        # for this measurement ROI, and (b) enriches the composite-
        # summary row with per-voxel _pm mean/std for each map.
        plot_fit_results_qt(
            time_fit, signal_fit, fitted_signal, fit_results,
            roi_mask=self.measurement_roi_mask,
            reference_image=(self.registered_4d[:, :, :, 0]
                             if self.registered_4d is not None else None),
            roi_z_slice=z,
            dataset_dir=self.dataset_dir,
            pre_injection_time=pre_time,
            pre_injection_signal=pre_signal,
            param_maps=self.param_maps,
            forced_n=self._measurement_roi_n,
            # Snapshot of the live parameter-map dialog figure so
            # FitResultsDialog._save_all can drop a matching
            # parameter_map_metric_roi_<N>.png alongside the kinetic
            # bundle when the user clicks Save.
            param_map_figure=self.figure,
        )
        self.measure_status_label.setText(
            "Click 'Draw' then drag-and-release on the map."
        )
        self._update_display()

    # Single-source-of-truth for which colormap and value range each
    # parameter map uses. Used by both the live single-slice display
    # and the multi-map stitched PNG exporter so the two stay in sync —
    # update the LUT in one place and the stitched grid follows.
    _MAP_TITLE_LABELS = {
        'kb_map': 'kb (buildup rate)',
        'kd_map': 'kd (decay rate)',
        'knt_map': 'knt (non-tracer rate)',
        'r_squared_map': 'R-squared',
        'a1_amplitude_map': 'A1 (tracer amplitude)',
        'a2_amplitude_map': 'A2 (non-tracer amplitude)',
        'a0_est_map': 'A0_est (baseline initial estimate)',
        'a2_est_map': 'A2_est (non-tracer initial estimate)',
        'a1_percent_map': '%Enhancement (A1/A0)',
        'a2_percent_map': '%NTE (A2/A0)',
        'a2_percent_est_map': '%NTE_est (A2_est/A0_est)',
        'baseline_map': 'A0 (baseline)',
        't0_map': 't0 (tracer onset)',
        'tmax_map': 'tmax (NTE onset)',
    }

    def _resolve_cmap_for_map(self, map_key):
        """Pick a (cmap, vmin, vmax) tuple for a given parameter map key.

        Centralised so both the interactive viewer (_update_display) and
        the stitched-grid PNG exporter (_export_stitched_png) stay in
        sync — kb/kd/knt share an ImageJ 16_color LUT, %NTE/%NTE_est
        share a diverging LUT with ±NTE_RANGE_MAX cap, R² uses
        RdYlBu_r on [0, 1], and so on.
        """
        from .colormaps import imagej_16_colors, nte_diverging  # noqa: F401
        if 'r_squared' in map_key:
            return 'RdYlBu_r', 0, 1
        if map_key == 'kb_map':
            # kb (buildup rate) shares the ImageJ 16_color LUT with
            # kd/knt for visual consistency, but auto-ranges instead
            # of using the fixed 0–0.15 cap — kb upper bound in the
            # fit model is 1.0 (one order of magnitude higher than
            # kd/knt), so a fixed cap would saturate most voxels at
            # the bright end of the LUT.
            return imagej_16_colors, None, None
        if map_key in ('kd_map', 'knt_map'):
            # Fixed 0–0.15 range so the discrete bands stay comparable
            # across datasets and between the two rate parameters.
            return imagej_16_colors, KD_DISPLAY_MIN, KD_DISPLAY_MAX
        if map_key == 'a1_percent_map':
            return (imagej_16_colors,
                    *self._percent_display_range(
                        'a1_percent_map', mode='positive'))
        if map_key == 'a2_percent_map':
            return (nte_diverging,
                    *self._percent_display_range(
                        'a2_percent_map', mode='symmetric',
                        max_limit=NTE_RANGE_MAX))
        if map_key == 'a2_percent_est_map':
            # %NTE_est shares the LUT, range, and cap so it can be
            # visually compared slice-for-slice against fitted %NTE.
            return (nte_diverging,
                    *self._percent_display_range(
                        'a2_percent_est_map', mode='symmetric',
                        max_limit=NTE_RANGE_MAX))
        return 'plasma', None, None

    def _export_stitched_grid_png(self, folder, selected_maps,
                                  include_t1, include_t2, include_roi,
                                  variants):
        """Build a stitched PNG grid (rows = maps, cols = z-slices).

        Renders one composite PNG per (overlay variant, ROI on/off)
        combination so the user can pick the version they want without
        re-running the export. Each row gets its own colorbar+label —
        kb/kd/knt/percent maps don't share a single LUT, so a per-row
        colorbar is necessary.

        Saved at the export-folder root as:
          stitched_all_maps.png             (plain, no ROI)
          stitched_all_maps_overlay.png     (overlay variant)
          stitched_all_maps_with_roi.png    (plain, ROI contour)
          stitched_all_maps_overlay_with_roi.png (both)

        Skipped when num_slices <= 1 (caller checks).
        """
        from pathlib import Path
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        folder = Path(folder)
        saved = []

        nz = int(self.num_slices)

        # Build the row list. Anatomical references go on top so the
        # parameter maps below them sit in a consistent column-aligned
        # context. Each row spec carries everything _draw_grid_row
        # needs to render itself — keeps that loop tidy.
        rows = []
        if include_t1 and self.registered_4d is not None:
            rows.append({
                'kind': 'anatomical',
                'label': 'T1',
                'data': self.registered_4d[:, :, :, 0],
                'cmap': 'gray',
                'vmin': None,
                'vmax': None,
            })
        if include_t2 and self.registered_t2 is not None:
            rows.append({
                'kind': 'anatomical',
                'label': 'T2',
                'data': self.registered_t2,
                'cmap': 'gray',
                'vmin': None,
                'vmax': None,
            })
        for key in selected_maps:
            cmap, vmin, vmax = self._resolve_cmap_for_map(key)
            if isinstance(cmap, str):
                cmap_obj = mpl.colormaps[cmap].copy()
            else:
                cmap_obj = cmap.copy()
            label = self._MAP_TITLE_LABELS.get(key, key.replace('_map', ''))
            rows.append({
                'kind': 'param',
                'key': key,
                'label': label,
                'cmap': cmap_obj,
                'vmin': vmin,
                'vmax': vmax,
            })

        if not rows:
            return saved

        nrows = len(rows)

        # Auto-vmin/vmax for any row that came back with None — we want
        # a single LUT range per row so all slices in that row share the
        # same colorbar. Compute across the full 3D volume and the
        # row's mask.
        param_mask = self.param_maps.get('mask')
        for row in rows:
            if row['kind'] != 'param':
                continue
            if row['vmin'] is not None and row['vmax'] is not None:
                continue
            data3d = self.param_maps.get(row['key'])
            if data3d is None:
                row['vmin'], row['vmax'] = 0, 1
                continue
            if param_mask is not None and data3d.shape == param_mask.shape:
                masked = data3d[param_mask]
            else:
                masked = data3d.flatten()
            masked = masked[np.isfinite(masked)]
            if masked.size == 0:
                row['vmin'], row['vmax'] = 0, 1
                continue
            row['vmin'] = float(np.percentile(masked, 1))
            row['vmax'] = float(np.percentile(masked, 99))
            if row['vmax'] <= row['vmin']:
                row['vmax'] = row['vmin'] + 1e-6

        roi_overlay_modes = [(False, '')]
        if include_roi:
            roi_overlay_modes = [(True, '_with_roi')]
            # Plain (no ROI) is still useful even when ROI was requested,
            # so emit both — keeps the file with the cleaner reference
            # available alongside the annotated one.
            roi_overlay_modes.insert(0, (False, ''))

        # One PNG per (overlay variant) × (with/without ROI) combo.
        for overlay_on, variant_suffix in variants:
            for show_roi, roi_suffix in roi_overlay_modes:
                fname = "stitched_all_maps"
                if variant_suffix:
                    fname += variant_suffix
                if roi_suffix:
                    fname += roi_suffix
                fname += ".png"
                save_path = folder / fname

                # Layout: extra column on the right reserved for the
                # per-row colorbars. width_ratios biases space toward
                # the data columns.
                fig_w = max(2.0 * nz + 2.0, 6.0)
                fig_h = max(2.0 * nrows + 0.4, 3.0)
                fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
                gs = fig.add_gridspec(
                    nrows=nrows,
                    ncols=nz + 1,
                    width_ratios=[1] * nz + [0.05],
                    wspace=0.05,
                    hspace=0.18,
                    left=0.06, right=0.95,
                    top=0.94, bottom=0.04,
                )

                for r_idx, row in enumerate(rows):
                    cbar_ax = fig.add_subplot(gs[r_idx, nz])
                    last_im = None
                    for z in range(nz):
                        ax = fig.add_subplot(gs[r_idx, z])
                        last_im = self._draw_grid_cell(
                            ax, row, z,
                            overlay_on=overlay_on and row['kind'] == 'param',
                            show_roi=show_roi,
                        )
                        if z == 0:
                            ax.set_ylabel(row['label'], fontsize=10,
                                          rotation=0, ha='right',
                                          va='center', labelpad=10)
                        if r_idx == 0:
                            ax.set_title(f"z{z:02d}", fontsize=9)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    if last_im is not None and row['kind'] == 'param':
                        fig.colorbar(last_im, cax=cbar_ax)
                        cbar_ax.tick_params(labelsize=7)
                    else:
                        cbar_ax.axis('off')

                fig.suptitle(
                    f"Parameter map grid — {nz} slices × {nrows} rows",
                    fontsize=12, y=0.99,
                )
                fig.savefig(
                    str(save_path), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                )
                plt.close(fig)
                saved.append(str(save_path))

        return saved

    def _draw_grid_cell(self, ax, row, z, overlay_on, show_roi):
        """Render a single grid cell for the stitched-grid exporter.

        Mirrors the logic in _update_display but writes into a caller-
        provided Axes rather than the live canvas. Returns the imshow
        handle so the caller can attach a row colorbar.
        """
        # Anatomical row: just show the grayscale slice and bail — no
        # parameter map underneath, no overlay opacity.
        if row['kind'] == 'anatomical':
            data = row['data']
            if data.ndim == 3 and z < data.shape[2]:
                ref = data[:, :, z].T
            elif data.ndim == 3:
                ref = data[:, :, 0].T
            else:
                ref = data.T
            return ax.imshow(ref, cmap='gray', origin='lower',
                             vmin=row['vmin'], vmax=row['vmax'])

        # Parameter map row.
        map_data = self.param_maps.get(row['key'])
        mask = self.param_maps.get('mask')
        cmap = row['cmap']

        if map_data is None or mask is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            return None

        if map_data.ndim == 3 and z < map_data.shape[2]:
            map_slice = map_data[:, :, z].T
            mask_slice = mask[:, :, z].T
        else:
            map_slice = map_data[:, :, 0].T if map_data.ndim == 3 else map_data.T
            mask_slice = mask[:, :, 0].T if mask.ndim == 3 else mask.T

        display_data = np.where(mask_slice, map_slice, np.nan)

        if not overlay_on:
            cmap.set_bad('black')

        if overlay_on and self.reference_image is not None:
            if self.reference_image.ndim == 3 and z < self.reference_image.shape[2]:
                ref_slice = self.reference_image[:, :, z].T
            elif self.reference_image.ndim == 3:
                ref_slice = self.reference_image[:, :, 0].T
            else:
                ref_slice = self.reference_image.T
            ax.imshow(ref_slice, cmap='gray', origin='lower')
            im = ax.imshow(
                display_data, cmap=cmap, origin='lower',
                vmin=row['vmin'], vmax=row['vmax'],
                alpha=self.overlay_opacity,
            )
        else:
            im = ax.imshow(
                display_data, cmap=cmap, origin='lower',
                vmin=row['vmin'], vmax=row['vmax'],
            )

        # ROI contour (cyan = fitting ROI, yellow = measurement ROI).
        if show_roi and self.roi_mask is not None:
            if self.roi_mask.ndim == 2:
                roi_slice = self.roi_mask.T
            elif z < self.roi_mask.shape[2]:
                roi_slice = self.roi_mask[:, :, z].T
            else:
                roi_slice = None
            if roi_slice is not None and np.any(roi_slice):
                ax.contour(roi_slice, levels=[0.5], colors='cyan', linewidths=1.0)
        if show_roi and self.measurement_roi_mask is not None:
            ax.contour(
                self.measurement_roi_mask.T,
                levels=[0.5], colors='yellow', linewidths=1.0,
            )

        return im

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

        # Choose colormap. R² uses a 0–1 diverging palette; %Enhancement uses
        # the ImageJ 16_color discrete LUT (positive-only); %NTE uses a custom
        # diverging LUT with black at zero so the sign of non-tracer
        # enhancement reads at a glance. Other maps stay on plasma.
        #
        # Display ranges for the two percent maps are auto-scaled per dataset
        # from the 99th percentile of the full 3D volume (see
        # _percent_display_range), so outlier voxels saturate at the LUT
        # extremes instead of compressing the rest of the map. The cached
        # range is shared across all z-slices so visual comparison between
        # slices stays consistent.
        cmap, vmin, vmax = self._resolve_cmap_for_map(self.current_map)

        # Render NaN / undefined voxels (mask=False, divide-by-zero in
        # %Enhancement / %NTE, etc.) as black in the standalone view —
        # otherwise the matplotlib default (transparent → white axes
        # background) makes unfit voxels disappear into the page. Resolve
        # string colormap names to a Colormap object first, and copy so
        # we don't mutate the global instance shared with other plots.
        # In overlay mode we keep the default (transparent) so the
        # grayscale anatomical underneath stays visible through unfit
        # voxels — black there would just darken the anatomy uselessly.
        import matplotlib as mpl
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap].copy()
        else:
            cmap = cmap.copy()
        overlay_active = self.overlay_mode and self.reference_image is not None
        if not overlay_active:
            cmap.set_bad('black')

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

        # Add ROI contour if enabled (cyan = the fitting ROI used during
        # parameter mapping)
        if self.roi_checkbox.isChecked() and self.roi_mask is not None:
            if self.roi_mask.ndim == 2:
                roi_slice = self.roi_mask.T
            elif self.current_z < self.roi_mask.shape[2]:
                roi_slice = self.roi_mask[:, :, self.current_z].T
            else:
                roi_slice = None

            if roi_slice is not None and np.any(roi_slice):
                self.ax.contour(roi_slice, levels=[0.5], colors='cyan', linewidths=2)

        # Measurement ROI contour (yellow), shown whenever set so it's
        # visually distinct from the cyan fitting ROI. The mask is 2D
        # in (x, y) and applies to whichever z-slice is shown.
        if self.measurement_roi_mask is not None:
            self.ax.contour(
                self.measurement_roi_mask.T,
                levels=[0.5], colors='yellow', linewidths=2,
            )

        # Colorbar - store reference so we can remove it later
        self.colorbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046)

        # Title — shared label dict so the stitched export and the
        # single-slice viewer agree on the human-readable name.
        title = self._MAP_TITLE_LABELS.get(self.current_map, self.current_map)
        self.ax.set_title(f"{title} (z={self.current_z})")
        self.ax.axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

        # Update metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update the metrics display.

        Shows up to two sections:
        - 'Fitting ROI' — the ROI used during parameter mapping (if any).
          Stats are intersected with the fit_mask so only converged
          voxels contribute.
        - 'Measurement ROI' — a freeform ROI drawn directly on the
          displayed map (if any). Stats use just NaN-filtering so the
          user gets exactly what they outlined.
        """
        # Param names shared across both ROI sections
        param_names = [
            ('kb_map', 'kb (buildup)'),
            ('kd_map', 'kd (decay)'),
            ('knt_map', 'knt (non-tracer)'),
            ('r_squared_map', 'R-squared'),
            ('a1_percent_map', '%Enhancement'),
            ('a2_percent_map', '%NTE'),
            ('a2_percent_est_map', '%NTE_est'),
        ]

        lines = []

        # ----- Fitting ROI block (existing behavior) -----
        if self.roi_mask is not None:
            if self.roi_mask.ndim == 2:
                roi_slice = self.roi_mask
            elif self.current_z < self.roi_mask.shape[2]:
                roi_slice = self.roi_mask[:, :, self.current_z]
            else:
                roi_slice = None

            mask = self.param_maps.get('mask')
            if roi_slice is not None and mask is not None:
                if mask.ndim == 3:
                    combined_mask = roi_slice & mask[:, :, self.current_z]
                else:
                    combined_mask = roi_slice & mask[:, :, 0]

                n_pixels = int(np.sum(combined_mask))
                lines.append("─── Fitting ROI (cyan) ───")
                lines.append(f"ROI + fitted: {n_pixels} pixels")
                lines.append(f"Z-slice: {self.current_z}")
                lines.append("")

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
                        lines.append(f"{name}:")
                        lines.append(
                            f"  {np.nanmean(roi_values):.4f} "
                            f"+/- {np.nanstd(roi_values):.4f}"
                        )

        # ----- Measurement ROI block -----
        if self.measurement_roi_mask is not None:
            if lines:
                lines.append("")
            lines.append("─── Measurement ROI (yellow) ───")
            n_drawn = int(self.measurement_roi_mask.sum())
            drawn_z = self.measurement_roi_drawn_z
            # If viewing a different slice from where the ROI was drawn,
            # flag it explicitly — the same 2D mask is being applied to
            # whatever slice is shown, but feature alignment may have
            # drifted, so the user should usually re-draw on the new
            # slice for a clean measurement.
            if drawn_z is not None and drawn_z != self.current_z:
                lines.append(
                    f"Drawn: {n_drawn} pixels  "
                    f"(drawn on z={drawn_z}, viewing z={self.current_z})"
                )
            else:
                lines.append(
                    f"Drawn: {n_drawn} pixels  (z-slice {self.current_z})"
                )
            lines.append("")

            for key, name in param_names:
                map_data = self.param_maps.get(key)
                if map_data is None:
                    continue
                if map_data.ndim == 3:
                    slice_data = map_data[:, :, self.current_z]
                else:
                    slice_data = map_data[:, :, 0]
                # Don't intersect with fit_mask here — the user drew this
                # ROI to measure exactly what they outlined. NaN voxels
                # are filtered out by nanmean/nanstd, so unfit pixels
                # don't poison the stats.
                roi_values = slice_data[self.measurement_roi_mask]
                valid = roi_values[~np.isnan(roi_values)]
                if len(valid) > 0:
                    lines.append(f"{name}:")
                    lines.append(
                        f"  {np.nanmean(valid):.4f} "
                        f"+/- {np.nanstd(valid):.4f}  (n={len(valid)})"
                    )
                else:
                    lines.append(f"{name}: — (no fitted voxels in ROI)")

        if not lines:
            self.metrics_label.setText(
                "No ROI available.\n"
                "Draw a measurement ROI on the map to see stats."
            )
            return

        self.metrics_label.setText('\n'.join(lines))

    def _update_info(self):
        """Update the processing info display."""
        metadata = self.param_maps.get('metadata', {})

        lines = []
        if 'kernel_type' in metadata:
            lines.append(f"Kernel: {metadata['kernel_type']}")
        if all(k in metadata for k in ('window_x', 'window_y', 'window_z')):
            lines.append(f"Window: {metadata['window_x']}x{metadata['window_y']}x{metadata['window_z']}")
        if 'success_rate' in metadata:
            lines.append(f"Success rate: {metadata['success_rate']:.1f}%")
        if 'processing_time' in metadata:
            lines.append(f"Time: {metadata['processing_time']:.1f}s")
        if 'total_positions' in metadata:
            lines.append(f"Positions: {metadata['total_positions']}")

        self.info_label.setText('\n'.join(lines))

    def _show_map_selection_dialog(self, format_type: str):
        """Show dialog to select which parameter maps to export."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select Maps to Export ({format_type.upper()})")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Select parameter maps to export:"))

        # Map display names
        map_names = {
            'kb_map': 'kb (buildup rate)',
            'kd_map': 'kd (decay rate)',
            'knt_map': 'knt (non-tracer rate)',
            'r_squared_map': 'R² (fit quality)',
            'a1_percent_map': '%Enhancement (A1/A0)',
            'a2_percent_map': '%NTE (A2/A0)',
            'a2_percent_est_map': '%NTE_est (A2_est/A0_est)',
            'a1_amplitude_map': 'A1 (amplitude)',
            'a2_amplitude_map': 'A2 (non-tracer amplitude)',
        }

        checkboxes = {}
        for key, name in map_names.items():
            if self.param_maps.get(key) is not None:
                cb = QCheckBox(name)
                cb.setChecked(key == self.current_map)  # Pre-select current map
                checkboxes[key] = cb
                layout.addWidget(cb)

        # Reset the captured anatomical-export flags — these are populated
        # below from checkbox state *before* the dialog goes out of scope.
        # We can't keep the QCheckBox widgets around past dialog.exec()
        # because Qt deletes child widgets when the parent dialog is
        # destroyed, leaving stale Python wrappers that raise RuntimeError
        # on .isChecked(). So we read the values now, store booleans.
        self._include_t1 = False
        self._include_t2 = False

        # Local checkbox handles (don't store on self — they die with the dialog).
        include_t1_cb = None
        include_t2_cb = None

        # T1/T2 anatomical reference checkboxes — apply to BOTH DICOM and
        # PNG export formats now. For DICOM the destination subfolders are
        # T1_baseline_map/ and T2_map/ (mirroring the parameter-map naming
        # where the "_map" suffix marks the DICOM tree). For PNG they go
        # to T1_baseline/ and T2/ (no suffix, mirroring the parameter-map
        # PNG layout).
        anat_label_text = (
            "\nInclude anatomical reference (DICOM and PNG variants):"
        )
        anat_label = QLabel(anat_label_text)
        anat_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(anat_label)

        if self.registered_4d is not None:
            include_t1_cb = QCheckBox("T1 baseline (timepoint 0)")
            # Default off — parameter maps are the primary export, T1/T2
            # are opt-in extras.
            include_t1_cb.setChecked(False)
            layout.addWidget(include_t1_cb)
        if self.registered_t2 is not None:
            include_t2_cb = QCheckBox("T2 anatomical")
            include_t2_cb.setChecked(False)
            layout.addWidget(include_t2_cb)
        if include_t1_cb is None and include_t2_cb is None:
            disabled = QLabel("(no T1/T2 source available for this dataset)")
            disabled.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(disabled)

        # PNG-only: anatomical-overlay variant checkboxes. Captured as
        # booleans on self before the dialog goes out of scope.
        self._png_plain = True
        self._png_overlay = False
        png_plain_cb = None
        png_overlay_cb = None

        if format_type == 'png':
            layout.addWidget(QLabel(""))  # Spacer

            self._include_roi_cb = QCheckBox("Include ROI overlay")
            self._include_roi_cb.setChecked(self.roi_checkbox.isChecked())
            layout.addWidget(self._include_roi_cb)

            anat_label = QLabel("\nAnatomical underlay:")
            anat_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(anat_label)

            png_plain_cb = QCheckBox("Save without anatomical overlay")
            png_plain_cb.setChecked(True)
            layout.addWidget(png_plain_cb)

            png_overlay_cb = QCheckBox("Save with anatomical overlay")
            # Only enabled when there's a reference image to underlay on.
            has_reference = self.reference_image is not None
            png_overlay_cb.setEnabled(has_reference)
            png_overlay_cb.setChecked(has_reference)
            layout.addWidget(png_overlay_cb)
            if not has_reference:
                hint = QLabel("(no anatomical reference loaded for this dataset)")
                hint.setStyleSheet("color: #888; font-style: italic;")
                layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            selected = [key for key, cb in checkboxes.items() if cb.isChecked()]
            include_roi = self._include_roi_cb.isChecked() if format_type == 'png' else False
            # Capture anatomical flags as plain booleans while the dialog
            # (and its child widgets) are still alive.
            self._include_t1 = bool(include_t1_cb and include_t1_cb.isChecked())
            self._include_t2 = bool(include_t2_cb and include_t2_cb.isChecked())
            # Same for PNG variant flags.
            if format_type == 'png':
                self._png_plain = bool(png_plain_cb and png_plain_cb.isChecked())
                self._png_overlay = bool(png_overlay_cb and png_overlay_cb.isChecked())
                # Default to at least the plain variant if user accidentally
                # unchecked both — exporting nothing is rarely the intent.
                if not (self._png_plain or self._png_overlay):
                    self._png_plain = True
            # Allow OK with zero parameter maps but at least one anatomical
            # checked — useful when the user just wants T1/T2 alongside an
            # existing parameter-map export.
            if not selected and not (self._include_t1 or self._include_t2):
                return None, False
            return selected, include_roi
        return None, False

    def _export(self, format_type: str):
        """Export parameter maps."""
        from ..io import get_dataset_path
        output_path = get_dataset_path(self.output_dir, 'parameter_maps')

        # Show map selection dialog
        selected_maps, include_roi = self._show_map_selection_dialog(format_type)

        # Anatomical-export flags are captured as booleans (not widget
        # references) inside the dialog method — see _show_map_selection_dialog
        # — so they're safe to read here even though the dialog has gone
        # out of scope and Qt has destroyed the underlying QCheckBox widgets.
        # T1/T2 checkboxes apply to both DICOM and PNG formats now.
        include_t1 = bool(getattr(self, '_include_t1', False))
        include_t2 = bool(getattr(self, '_include_t2', False))

        # Allow proceeding when at least one of: parameter maps, T1, or T2
        # was checked. The dialog enforces this too, but guard here in case
        # of a code path that bypasses it.
        if not selected_maps and not include_t1 and not include_t2:
            return
        # selected_maps may be an empty list if the user only wants T1/T2.
        if selected_maps is None:
            selected_maps = []

        if format_type == 'dicom':
            from ..io import save_parameter_map_as_dicom, save_volume_as_dicom_series

            folder = QFileDialog.getExistingDirectory(
                self, "Select Export Folder",
                str(output_path)
            )
            if folder:
                folder = Path(folder)
                metadata = self.param_maps.get('metadata', {})
                total_files = 0

                for key in selected_maps:
                    data = self.param_maps.get(key)
                    if data is not None:
                        saved = save_parameter_map_as_dicom(
                            data, key, str(folder), self.spacing,
                            self.source_dicom, metadata
                        )
                        total_files += len(saved)

                # Optional: T1 baseline DICOM. Subfolder T1_baseline_map/
                # (the "_map" suffix marks DICOM trees, mirroring the
                # parameter-map naming convention). Series offset 6000
                # is well clear of the parameter-map offsets (4000–5300).
                if include_t1 and self.registered_4d is not None:
                    t1_baseline = self.registered_4d[:, :, :, 0]
                    saved = save_volume_as_dicom_series(
                        t1_baseline, "T1_baseline_map",
                        str(folder), self.spacing,
                        source_dicom=self.source_dicom,
                        series_description="T1 baseline (registered, t=0)",
                        series_offset=6000,
                    )
                    total_files += len(saved)

                # Optional: T2 anatomical DICOM. Subfolder T2_map/.
                if include_t2 and self.registered_t2 is not None:
                    saved = save_volume_as_dicom_series(
                        self.registered_t2, "T2_map",
                        str(folder), self.spacing,
                        source_dicom=self.source_dicom,
                        series_description="T2 anatomical (registered)",
                        series_offset=6100,
                    )
                    total_files += len(saved)

                QMessageBox.information(
                    self, "Exported",
                    f"Saved {total_files} DICOM files to:\n{folder}"
                )

        elif format_type == 'png':
            folder = QFileDialog.getExistingDirectory(
                self, "Select Export Folder",
                str(output_path)
            )
            if folder:
                import shutil

                folder = Path(folder)
                saved_files = []

                # Save current display settings so we can restore the
                # viewer to where the user left it after the export run.
                original_map = self.current_map
                original_z = self.current_z
                original_roi = self.roi_checkbox.isChecked()
                original_overlay = self.overlay_mode

                # Set ROI display
                self.roi_checkbox.setChecked(include_roi)

                # Build the list of anatomical-overlay variants the user
                # asked for. Each entry is (overlay_on?, filename_suffix).
                # When both are checked we emit two PNGs per slice with
                # different suffixes so they can coexist in the same
                # subfolder without overwriting each other.
                variants = []
                if getattr(self, '_png_plain', True):
                    variants.append((False, ''))
                if getattr(self, '_png_overlay', False):
                    variants.append((True, '_overlay'))
                if not variants:
                    variants = [(False, '')]  # safety net

                # Iterate over every selected map AND every z-slice so the
                # export gives you a complete stack instead of just whatever
                # slice was visible at the moment Save was clicked. Each map
                # gets its own subfolder mirroring the DICOM export layout
                # (<folder>/<map_name>/<map_name>_zNN.png), and stale files
                # from previous exports are wiped before the new run so a
                # re-export with different settings doesn't leave orphans.
                for key in selected_maps:
                    self.current_map = key
                    map_name = key.replace('_map', '')

                    map_dir = folder / map_name
                    shutil.rmtree(map_dir, ignore_errors=True)
                    map_dir.mkdir(parents=True, exist_ok=True)

                    for z in range(self.num_slices):
                        self.current_z = z
                        for overlay_on, variant_suffix in variants:
                            self.overlay_mode = overlay_on
                            self._update_display()

                            filename = f"{map_name}_z{z:02d}"
                            if variant_suffix:
                                filename += variant_suffix
                            if include_roi:
                                filename += "_with_roi"
                            filename += ".png"

                            filepath = map_dir / filename
                            self.figure.savefig(
                                str(filepath), dpi=150, bbox_inches='tight',
                                facecolor='white', edgecolor='none',
                            )
                            saved_files.append(str(filepath))

                # Optional anatomical PNG series alongside the parameter
                # maps. Subfolders T1_baseline/ and T2/ (no "_map" suffix
                # — the suffix marks the DICOM tree, not PNG).
                if include_t1 and self.registered_4d is not None:
                    from ..io import save_volume_as_png_series
                    saved = save_volume_as_png_series(
                        self.registered_4d[:, :, :, 0],
                        "T1_baseline",
                        str(folder),
                    )
                    saved_files.extend(saved)
                if include_t2 and self.registered_t2 is not None:
                    from ..io import save_volume_as_png_series
                    saved = save_volume_as_png_series(
                        self.registered_t2,
                        "T2",
                        str(folder),
                    )
                    saved_files.extend(saved)

                # Stitched grid PNGs — only meaningful when there's
                # more than one z-slice. One PNG per overlay variant
                # so plain and overlay versions stay easy to grab
                # separately. Always emits the plain version (rows =
                # maps, columns = slices), and the with-roi variant
                # when include_roi is on.
                if self.num_slices > 1:
                    try:
                        stitched_paths = self._export_stitched_grid_png(
                            folder=folder,
                            selected_maps=selected_maps,
                            include_t1=include_t1,
                            include_t2=include_t2,
                            include_roi=include_roi,
                            variants=variants,
                        )
                        saved_files.extend(stitched_paths)
                    except Exception as e:
                        # Stitched export is best-effort — failure
                        # shouldn't blow up the per-slice export the
                        # user actually asked for.
                        print(f"Stitched grid PNG failed: {e}")

                # Restore original settings
                self.current_map = original_map
                self.current_z = original_z
                self.roi_checkbox.setChecked(original_roi)
                self.overlay_mode = original_overlay
                self._update_display()

                QMessageBox.information(
                    self, "Exported",
                    f"Saved {len(saved_files)} PNG files to:\n{folder}"
                )

    def _export_metrics(self):
        """Export metrics to CSV with a companion ROI overlay PNG.

        CSV schema: roi_type, z_slice, parameter, n_pixels, mean, std,
        min, max. Writes a row per (ROI × parameter × z-slice). Both
        the fitting ROI (per slice, intersected with the fit mask) and
        the measurement ROI (single slice — wherever it was drawn —
        with NaN filtering only) are included so the file captures
        everything the user can see in the metrics panel.

        Default save location is
        ``<dataset>/parameter_maps/parameter_map_metrics/parameter_map_metric_<N>.csv``
        with the next available N auto-detected so successive saves
        don't overwrite each other (one full bundle per measurement
        ROI). The companion PNG uses the same N and the suffix
        ``parameter_map_metric_roi_<N>.png``.
        """
        import csv

        from ..io import get_dataset_path, next_indexed_path, index_from_filename

        # Default save location: the per-dataset metrics subfolder.
        metrics_dir = Path(get_dataset_path(
            self.output_dir,
            'parameter_maps/parameter_map_metrics',
        ))
        # When a measurement ROI is active, reuse the shared ROI
        # counter so this metrics CSV lands on the same N as the
        # matching kinetic_fit_results_<N>.csv from the "Kinetic Fit
        # on this ROI" button. Otherwise pick the next free N as
        # before.
        if (self.measurement_roi_mask is not None
                and getattr(self, '_measurement_roi_n', None) is not None):
            metrics_dir.mkdir(parents=True, exist_ok=True)
            default_path = (metrics_dir
                            / f"parameter_map_metric_{int(self._measurement_roi_n)}.csv")
        else:
            default_path = next_indexed_path(
                metrics_dir, "parameter_map_metric", ".csv"
            )

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics",
            str(default_path),
            "CSV Files (*.csv)"
        )

        if not filepath:
            return

        # Build the unified-format rows for the measurement ROI. The
        # CSV/PNG schema mirrors the kinetic-fit results bundle
        # (parameter, description, value, std, units) — the column
        # order is identical so a Combined Metrics PNG can stack the
        # two tables seamlessly. roi_type / z_slice / n_pixels are
        # surfaced in the title instead of as columns now: the CSV
        # only ever describes one ROI on one z, so per-row repetition
        # was redundant. Parameter names get the _pm suffix so they
        # don't collide with kinetic-fit names in the composite
        # summary CSV (kb vs kb_pm, etc.).
        param_specs = self._build_pm_metric_specs()

        n_pixels = 0
        z_for_title = None
        unified_rows = []

        if self.measurement_roi_mask is not None:
            z = self.measurement_roi_drawn_z
            if z is None:
                z = self.current_z
            z_for_title = z
            for spec in param_specs:
                map_data = self.param_maps.get(spec['key'])
                if map_data is None:
                    continue
                if map_data.ndim == 3:
                    slice_data = map_data[:, :, z]
                else:
                    slice_data = map_data[:, :, 0]
                values = slice_data[self.measurement_roi_mask]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                if not n_pixels:
                    n_pixels = int(values.size)
                unified_rows.append((
                    spec['name_pm'],
                    spec['description'],   # already includes "(mean)"
                    float(np.nanmean(values)),
                    float(np.nanstd(values)),
                    spec['units'],
                ))

        if not unified_rows and self.roi_mask is not None:
            # Fallback when the user hasn't drawn a measurement ROI:
            # use the fitting ROI restricted to the currently shown
            # z-slice. Title context still describes one ROI / one z.
            fit_mask = self.param_maps.get('mask')
            z = self.current_z
            z_for_title = z
            roi_slice = (self.roi_mask if self.roi_mask.ndim == 2
                         else self.roi_mask[:, :, z])
            if fit_mask is not None:
                fit_slice = (fit_mask[:, :, z]
                             if fit_mask.ndim == 3
                             else fit_mask[:, :, 0])
                combined = roi_slice & fit_slice
            else:
                combined = roi_slice
            if combined.any():
                for spec in param_specs:
                    map_data = self.param_maps.get(spec['key'])
                    if map_data is None:
                        continue
                    if map_data.ndim == 3:
                        slice_data = map_data[:, :, z]
                    else:
                        slice_data = map_data[:, :, 0]
                    values = slice_data[combined]
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue
                    if not n_pixels:
                        n_pixels = int(values.size)
                    unified_rows.append((
                        spec['name_pm'],
                        spec['description'],
                        float(np.nanmean(values)),
                        float(np.nanstd(values)),
                        spec['units'],
                    ))

        unified_header = ['parameter', 'description', 'value', 'std', 'units']
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(unified_header)
            for row in unified_rows:
                writer.writerow(row)

        # Companion files share the CSV's index N. Two PNGs are saved:
        #
        #   parameter_map_metric_roi_<N>.png   — snapshot of the dialog
        #         figure with active ROI contour overlay (map context).
        #   parameter_map_metric_<N>.png       — table render of the
        #         numeric data so the metrics are presentation-ready.
        csv_path = Path(filepath)
        n = index_from_filename(csv_path, "parameter_map_metric", ".csv")
        if n is not None:
            roi_png_path = csv_path.parent / f"parameter_map_metric_roi_{n}.png"
            table_png_path = csv_path.parent / f"parameter_map_metric_{n}.png"
        else:
            roi_png_path = next_indexed_path(
                csv_path.parent, "parameter_map_metric_roi", ".png"
            )
            # When the CSV got renamed off-pattern, fall back to next
            # free pattern slot for the table PNG too.
            table_png_path = next_indexed_path(
                csv_path.parent, "parameter_map_metric", ".png"
            )

        # 1) ROI overlay PNG (figure snapshot)
        try:
            self.figure.savefig(
                str(roi_png_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none',
            )
            roi_png_msg = f"\nROI overlay PNG: {roi_png_path}"
        except Exception as e:
            roi_png_msg = f"\n(ROI overlay PNG failed: {e})"

        # 2) Metrics table PNG (numeric data rendered as a table) using
        #    the same unified header as the kinetic fit results table.
        from ..roi_selection import save_table_as_png

        def _fmt(v, fmt='.4f'):
            if v == '' or v is None:
                return ''
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(v)

        png_rows = [[
            row[0], row[1], _fmt(row[2]), _fmt(row[3]), row[4],
        ] for row in unified_rows]

        title_bits = []
        if n is not None:
            title_bits.append(f"ROI #{n}")
        if z_for_title is not None:
            title_bits.append(f"z={z_for_title}")
        if n_pixels:
            title_bits.append(f"n={n_pixels} pixels")
        title_suffix = (" (" + ", ".join(title_bits) + ")") if title_bits else ""
        try:
            save_table_as_png(
                png_rows, unified_header, str(table_png_path),
                title=f"Parameter map metrics{title_suffix}",
                # Cap description so one verbose row doesn't blow up
                # the column. Same cap as the kinetic-fit table so the
                # two render with comparable proportions.
                max_col_chars=[None, 22, None, None, None],
            )
            table_png_msg = f"\nMetrics table PNG: {table_png_path}"
        except Exception as e:
            table_png_msg = f"\n(metrics table PNG failed: {e})"

        QMessageBox.information(
            self, "Exported",
            f"Metrics CSV: {filepath}{roi_png_msg}{table_png_msg}",
        )

    # Single source of truth for the parameter-map metric rows so the
    # _export_metrics path and the Combined Metrics PNG render the
    # same parameters in the same order with the same _pm naming and
    # description text. Description includes "(mean)" so the table
    # reads as a per-voxel average in the ROI rather than a fitted
    # parameter.
    def _build_pm_metric_specs(self):
        time_units = (self.param_maps.get('metadata', {}) or {}).get(
            'time_units', 'minutes',
        )
        return [
            {'key': 'baseline_map', 'name_pm': 'A0_pm',
             'description': 'baseline signal (mean)', 'units': ''},
            {'key': 'a1_amplitude_map', 'name_pm': 'A1_pm',
             'description': 'tracer amplitude (mean)', 'units': ''},
            {'key': 'a2_amplitude_map', 'name_pm': 'A2_pm',
             'description': 'non-tracer amplitude (mean)', 'units': ''},
            {'key': 'kb_map', 'name_pm': 'kb_pm',
             'description': 'buildup rate (mean)',
             'units': f'1/{time_units}'},
            {'key': 'kd_map', 'name_pm': 'kd_pm',
             'description': 'decay rate (mean)',
             'units': f'1/{time_units}'},
            {'key': 'knt_map', 'name_pm': 'knt_pm',
             'description': 'non-tracer rate (mean)',
             'units': f'1/{time_units}'},
            {'key': 't0_map', 'name_pm': 't0_pm',
             'description': 'tracer onset (mean)', 'units': time_units},
            {'key': 'tmax_map', 'name_pm': 'tmax_pm',
             'description': 'NTE onset (mean)', 'units': time_units},
            {'key': 'a1_percent_map', 'name_pm': 'pct_enhancement_pm',
             'description': '%Enhancement (mean)', 'units': '%'},
            {'key': 'a2_percent_map', 'name_pm': 'pct_nte_pm',
             'description': '%NTE (mean)', 'units': '%'},
            {'key': 'a2_percent_est_map', 'name_pm': 'pct_nte_est_pm',
             'description': '%NTE_est (mean)', 'units': '%'},
            {'key': 'r_squared_map', 'name_pm': 'R_squared_pm',
             'description': 'goodness of fit (mean)', 'units': ''},
        ]


def show_parameter_map_options(max_z: int = 8,
                                current_z: int = 4,
                                existing_roi: Optional[np.ndarray] = None,
                                existing_injection_idx: Optional[int] = None,
                                default_window_size: Tuple[int, int, int] = (15, 15, 1),
                                default_steady_state_time: float = 100.0,
                                time_units: str = 'minutes') -> Optional[dict]:
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
    default_steady_state_time : float
        Pre-fill for the NTE steady-state-time spinbox (in time_units).
    time_units : str
        Unit suffix for the NTE steady-state-time spinbox.

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
        default_window_size=default_window_size,
        default_steady_state_time=default_steady_state_time,
        time_units=time_units,
    )

    result = dialog.exec()

    if result == QDialog.Accepted:
        return dialog.get_result()
    return None


def show_parameter_map_results(param_maps: Dict[str, np.ndarray],
                                spacing: Tuple[float, float, float],
                                roi_mask: Optional[np.ndarray] = None,
                                output_dir: str = './output',
                                source_dicom: Optional[str] = None,
                                registered_4d: Optional[np.ndarray] = None,
                                registered_t2: Optional[np.ndarray] = None,
                                time_array: Optional[np.ndarray] = None,
                                dataset_dir: Optional[str] = None) -> None:
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
    registered_4d : np.ndarray, optional
        Registered T1 4D volume [x, y, z, t]. When provided, the Save-as-
        DICOM export gains a "Include T1 baseline" checkbox that writes
        registered_4d[:, :, :, 0] alongside the parameter maps.
    registered_t2 : np.ndarray, optional
        Registered T2 3D volume. When provided, the Save-as-DICOM export
        gains a "Include T2 anatomical" checkbox.
    """
    app = init_qt_app()

    dialog = ParameterMapResultsDialog(
        param_maps=param_maps,
        spacing=spacing,
        roi_mask=roi_mask,
        output_dir=output_dir,
        source_dicom=source_dicom,
        registered_4d=registered_4d,
        registered_t2=registered_t2,
        time_array=time_array,
        dataset_dir=dataset_dir,
    )

    dialog.exec()
