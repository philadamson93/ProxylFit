"""
Registration-related UI components for ProxylFit.
"""

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QSpinBox, QMessageBox, QSplitter,
    QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import PROXYLFIT_STYLE, init_qt_app


class RegistrationWorker(QThread):
    """Background worker for running registration without blocking the UI."""

    progress = Signal(int, int, str, object)  # current, total, message, metrics_info
    finished = Signal(object, object)  # registered_4d, metrics
    error = Signal(str)  # error message

    def __init__(self, image_4d, spacing, output_dir, dicom_path, parent=None):
        super().__init__(parent)
        self.image_4d = image_4d
        self.spacing = spacing
        self.output_dir = output_dir
        self.dicom_path = dicom_path
        self._is_cancelled = False

    def run(self):
        """Run registration in background thread."""
        try:
            # Import here to avoid circular imports
            from ..registration import register_timeseries

            registered_4d, metrics = register_timeseries(
                self.image_4d,
                self.spacing,
                output_dir=self.output_dir,
                show_quality_window=False,  # We'll show after dialog closes
                dicom_path=self.dicom_path,
                progress_callback=self._emit_progress
            )
            if not self._is_cancelled:
                self.finished.emit(registered_4d, metrics)
        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))

    def _emit_progress(self, current, total, message, metrics_info):
        """Emit progress signal (called from registration loop)."""
        self.progress.emit(current, total, message, metrics_info)

    def cancel(self):
        """Request cancellation (note: registration cannot be interrupted mid-timepoint)."""
        self._is_cancelled = True


class RegistrationProgressDialog(QDialog):
    """Progress dialog shown during registration with live metrics plotting."""

    def __init__(self, image_4d, spacing, output_dir, dicom_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProxylFit - Registration Progress")
        self.setModal(True)
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)

        # Store output dir for display
        self.output_dir = output_dir

        # Results stored here after completion
        self.registered_4d = None
        self.metrics = None

        # Data for live plotting
        self.timepoints = []
        self.translations = []
        self.mse_values = []

        self._setup_ui()
        self._start_worker(image_4d, spacing, output_dir, dicom_path)

    def _setup_ui(self):
        """Set up the dialog UI with live plotting."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Registering Time Series")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # Status and progress row
        status_layout = QHBoxLayout()

        # Status message
        self.status_label = QLabel("Initializing registration...")
        self.status_label.setFont(QFont("Arial", 11))
        status_layout.addWidget(self.status_label, 1)

        # Current metrics (compact)
        self.current_metrics_label = QLabel("")
        self.current_metrics_label.setStyleSheet("color: #666666;")
        status_layout.addWidget(self.current_metrics_label)

        layout.addLayout(status_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Live plot area
        plot_group = QGroupBox("Registration Metrics")
        plot_layout = QVBoxLayout(plot_group)

        # Create matplotlib figure with two subplots
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.figure.set_facecolor('#f5f5f5')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)

        # Create subplots
        self.ax_trans = self.figure.add_subplot(121)
        self.ax_mse = self.figure.add_subplot(122)

        self._setup_plots()
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_group)

        # Detail label
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.detail_label)

        # Save location label (hidden until complete)
        self.save_label = QLabel("")
        self.save_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self.save_label.setWordWrap(True)
        self.save_label.hide()
        layout.addWidget(self.save_label)

        # Note about not closing
        self.note_label = QLabel("Please wait... This may take several minutes for large datasets.")
        self.note_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self.note_label)

        self.setStyleSheet(PROXYLFIT_STYLE)

    def _setup_plots(self):
        """Initialize the live plots."""
        # Translation plot
        self.ax_trans.set_xlabel('Timepoint', fontsize=9)
        self.ax_trans.set_ylabel('Translation (mm)', fontsize=9)
        self.ax_trans.set_title('Translation Magnitude', fontsize=10, fontweight='bold')
        self.ax_trans.grid(True, alpha=0.3)
        self.trans_line, = self.ax_trans.plot([], [], 'b-o', markersize=3, linewidth=1)

        # MSE plot
        self.ax_mse.set_xlabel('Timepoint', fontsize=9)
        self.ax_mse.set_ylabel('MSE', fontsize=9)
        self.ax_mse.set_title('Mean Squared Error', fontsize=10, fontweight='bold')
        self.ax_mse.grid(True, alpha=0.3)
        self.mse_line, = self.ax_mse.plot([], [], 'r-o', markersize=3, linewidth=1)

        self.figure.tight_layout()

    def _start_worker(self, image_4d, spacing, output_dir, dicom_path):
        """Start the background registration worker."""
        self.worker = RegistrationWorker(image_4d, spacing, output_dir, dicom_path, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, current, total, message, metrics_info):
        """Handle progress updates from worker."""
        self.status_label.setText(message)
        percent = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.detail_label.setText(f"Completed {current} of {total} timepoints")

        # Update metrics display and plot if we have metrics
        if metrics_info is not None:
            # Update compact metrics label
            self.current_metrics_label.setText(
                f"Trans: {metrics_info['translation_mm']:.2f}mm | "
                f"MSE: {metrics_info['mse']:.0f} | "
                f"Time: {metrics_info['time']:.1f}s"
            )

            # Add to plot data
            self.timepoints.append(current)
            self.translations.append(metrics_info['translation_mm'])
            self.mse_values.append(metrics_info['mse'])

            # Update plots
            self._update_plots(total)

    def _update_plots(self, total):
        """Update the live plots with new data."""
        # Update translation plot
        self.trans_line.set_data(self.timepoints, self.translations)
        self.ax_trans.set_xlim(0, total + 1)
        if self.translations:
            max_trans = max(self.translations) * 1.1
            self.ax_trans.set_ylim(0, max(max_trans, 0.1))

        # Update MSE plot
        self.mse_line.set_data(self.timepoints, self.mse_values)
        self.ax_mse.set_xlim(0, total + 1)
        if self.mse_values:
            min_mse = min(self.mse_values) * 0.9
            max_mse = max(self.mse_values) * 1.1
            self.ax_mse.set_ylim(min_mse, max_mse)

        # Redraw
        self.canvas.draw_idle()

    def _on_finished(self, registered_4d, metrics):
        """Handle successful completion."""
        self.registered_4d = registered_4d
        self.metrics = metrics

        # Show save location
        self.status_label.setText("Registration complete!")
        self.progress_bar.setValue(100)
        self.note_label.hide()
        self.save_label.setText(f"Data saved to: {self.output_dir}")
        self.save_label.show()

        self.accept()

    def _on_error(self, error_msg):
        """Handle registration error."""
        QMessageBox.critical(self, "Registration Error", f"Registration failed:\n\n{error_msg}")
        self.reject()

    def closeEvent(self, event):
        """Handle dialog close (user clicked X)."""
        if self.worker.isRunning():
            # Warn user that registration is in progress
            reply = QMessageBox.question(
                self,
                "Cancel Registration?",
                "Registration is still in progress. Are you sure you want to cancel?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(5000)  # Wait up to 5 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class RegistrationReviewDialog(QDialog):
    """Qt-based registration quality review dialog."""

    def __init__(self, original_4d, registered_4d, metrics, z_slice=None, output_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProxylFit - Registration Review")
        self.setMinimumSize(1200, 800)

        self.original_4d = original_4d
        self.registered_4d = registered_4d
        self.metrics = metrics
        self.output_dir = output_dir

        # State
        self.current_timepoint = 1
        self.max_timepoint = original_4d.shape[3] - 1
        self.z_slice = z_slice if z_slice is not None else original_4d.shape[2] // 2
        self.max_z_slice = original_4d.shape[2] - 1
        self.accepted = False

        self._setup_ui()
        self._plot_metrics()
        self._update_display()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Image Registration Review")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Main content - images on top, plots below
        content_splitter = QSplitter(Qt.Vertical)

        # === Image Row ===
        image_widget = QWidget()
        image_layout = QHBoxLayout(image_widget)
        image_layout.setSpacing(10)

        # Create 4 image canvases
        self.fig_images = Figure(figsize=(12, 3), dpi=100)
        self.fig_images.set_facecolor('#f5f5f5')
        self.canvas_images = FigureCanvas(self.fig_images)

        self.ax_ref = self.fig_images.add_subplot(141)
        self.ax_registered = self.fig_images.add_subplot(142)
        self.ax_diff = self.fig_images.add_subplot(143)
        self.ax_unregistered = self.fig_images.add_subplot(144)

        # Initialize image objects with placeholder data (will be updated in _update_display)
        # This avoids clearing/recreating axes on every update which causes size changes
        placeholder = np.zeros((10, 10))
        self.im_ref = self.ax_ref.imshow(placeholder, cmap='gray', origin='lower')
        self.im_registered = self.ax_registered.imshow(placeholder, cmap='gray', origin='lower')
        self.im_diff = self.ax_diff.imshow(placeholder, cmap='hot', origin='lower')
        self.im_unregistered = self.ax_unregistered.imshow(placeholder, cmap='gray', origin='lower')

        # Set up axes titles and styling once
        self.ax_ref.set_title('Reference (t=0)', fontsize=10)
        self.ax_ref.axis('off')
        self.ax_registered.set_title('Registered t=1', fontsize=10)
        self.ax_registered.axis('off')
        self.ax_diff.set_title('|Reference - Registered|', fontsize=10)
        self.ax_diff.axis('off')
        self.ax_unregistered.set_title('Original t=1 (unregistered)', fontsize=10)
        self.ax_unregistered.axis('off')

        self.fig_images.tight_layout()

        image_layout.addWidget(self.canvas_images)
        content_splitter.addWidget(image_widget)

        # === Metrics Row ===
        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)

        # Create 3 metric plot canvases
        self.fig_metrics = Figure(figsize=(12, 3), dpi=100)
        self.fig_metrics.set_facecolor('#f5f5f5')
        self.canvas_metrics = FigureCanvas(self.fig_metrics)

        self.ax_translation = self.fig_metrics.add_subplot(131)
        self.ax_rotation = self.fig_metrics.add_subplot(132)
        self.ax_mse = self.fig_metrics.add_subplot(133)

        # Enable click navigation on plots
        self.canvas_metrics.mpl_connect('button_press_event', self._on_plot_click)

        metrics_layout.addWidget(self.canvas_metrics)

        # Info panel
        info_group = QGroupBox("Current Timepoint")
        info_layout = QVBoxLayout(info_group)
        info_group.setFixedWidth(200)

        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Arial", 10))
        info_layout.addWidget(self.info_label)

        metrics_layout.addWidget(info_group)
        content_splitter.addWidget(metrics_widget)

        layout.addWidget(content_splitter, 1)

        # === Controls Row ===
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        # Z-slice control
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-slice:"))
        self.z_spinbox = QSpinBox()
        self.z_spinbox.setRange(0, self.max_z_slice)
        self.z_spinbox.setValue(self.z_slice)
        self.z_spinbox.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_spinbox)
        controls_layout.addLayout(z_layout)

        controls_layout.addStretch()

        # Navigation buttons
        self.btn_first = QPushButton("First")
        self.btn_first.clicked.connect(self._first_timepoint)
        controls_layout.addWidget(self.btn_first)

        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self._prev_timepoint)
        controls_layout.addWidget(self.btn_prev)

        self.timepoint_label = QLabel("")
        self.timepoint_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.timepoint_label.setMinimumWidth(120)
        self.timepoint_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(self.timepoint_label)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._next_timepoint)
        controls_layout.addWidget(self.btn_next)

        self.btn_last = QPushButton("Last")
        self.btn_last.clicked.connect(self._last_timepoint)
        controls_layout.addWidget(self.btn_last)

        controls_layout.addStretch()

        # Accept button
        self.btn_accept = QPushButton("Accept Registration")
        self.btn_accept.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_accept.clicked.connect(self._accept_registration)
        controls_layout.addWidget(self.btn_accept)

        layout.addLayout(controls_layout)

        # Save location info
        if self.output_dir:
            save_info = QLabel(f"Registration data saved to: {self.output_dir}")
            save_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
            save_info.setWordWrap(True)
            layout.addWidget(save_info)

        # Instructions
        instructions = QLabel("Use ←→ keys to navigate timepoints, ↑↓ for z-slices. Click on plots to jump to timepoint.")
        instructions.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(instructions)

        self.setStyleSheet(PROXYLFIT_STYLE)

        # Colorbar reference
        self.diff_colorbar = None

    def _plot_metrics(self):
        """Plot the metrics charts."""
        timepoints = [m.timepoint for m in self.metrics]

        # Translation plot
        trans_mag = [np.linalg.norm(m.translation) for m in self.metrics]
        self.ax_translation.plot(timepoints, trans_mag, 'b-o', markersize=4)
        self.ax_translation.set_xlabel('Timepoint', fontsize=9)
        self.ax_translation.set_ylabel('Translation (mm)', fontsize=9)
        self.ax_translation.set_title('Translation Magnitude', fontsize=10, fontweight='bold')
        self.ax_translation.grid(True, alpha=0.3)
        self.trans_marker, = self.ax_translation.plot([], [], 'ro', markersize=10)

        # Rotation plot
        rot_mag = [np.linalg.norm(m.rotation) * 180 / np.pi for m in self.metrics]
        self.ax_rotation.plot(timepoints, rot_mag, 'm-o', markersize=4)
        self.ax_rotation.set_xlabel('Timepoint', fontsize=9)
        self.ax_rotation.set_ylabel('Rotation (degrees)', fontsize=9)
        self.ax_rotation.set_title('Rotation Magnitude', fontsize=10, fontweight='bold')
        self.ax_rotation.grid(True, alpha=0.3)
        self.rot_marker, = self.ax_rotation.plot([], [], 'ro', markersize=10)

        # MSE plot
        mse_values = [m.mean_squared_error for m in self.metrics]
        self.ax_mse.plot(timepoints, mse_values, 'g-o', markersize=4)
        self.ax_mse.set_xlabel('Timepoint', fontsize=9)
        self.ax_mse.set_ylabel('MSE', fontsize=9)
        self.ax_mse.set_title('Mean Squared Error', fontsize=10, fontweight='bold')
        self.ax_mse.grid(True, alpha=0.3)
        self.mse_marker, = self.ax_mse.plot([], [], 'ro', markersize=10)

        self.fig_metrics.tight_layout()

    def _update_display(self):
        """Update the image display for current timepoint."""
        t = self.current_timepoint
        z = self.z_slice

        # Get slices
        ref_slice = self.original_4d[:, :, z, 0].T
        orig_slice = self.original_4d[:, :, z, t].T
        reg_slice = self.registered_4d[:, :, z, t].T
        diff_slice = np.abs(ref_slice - reg_slice)

        # Update image data using set_data() - avoids layout recalculation
        self.im_ref.set_data(ref_slice)
        self.im_ref.set_clim(vmin=ref_slice.min(), vmax=ref_slice.max())

        self.im_registered.set_data(reg_slice)
        self.im_registered.set_clim(vmin=reg_slice.min(), vmax=reg_slice.max())

        self.im_diff.set_data(diff_slice)
        self.im_diff.set_clim(vmin=diff_slice.min(), vmax=diff_slice.max())

        self.im_unregistered.set_data(orig_slice)
        self.im_unregistered.set_clim(vmin=orig_slice.min(), vmax=orig_slice.max())

        # Update titles that change with timepoint
        self.ax_registered.set_title(f'Registered t={t}', fontsize=10)
        self.ax_unregistered.set_title(f'Original t={t} (unregistered)', fontsize=10)

        self.canvas_images.draw()

        # Update metric markers
        m = self.metrics[t]
        self.trans_marker.set_data([m.timepoint], [np.linalg.norm(m.translation)])
        self.rot_marker.set_data([m.timepoint], [np.linalg.norm(m.rotation) * 180 / np.pi])
        self.mse_marker.set_data([m.timepoint], [m.mean_squared_error])
        self.canvas_metrics.draw()

        # Update info label
        tx, ty, tz = m.translation
        trans_mag = np.linalg.norm(m.translation)
        rot_mag = np.linalg.norm(m.rotation) * 180 / np.pi

        self.info_label.setText(
            f"Timepoint: {t}/{self.max_timepoint}\n"
            f"Z-slice: {z}/{self.max_z_slice}\n\n"
            f"MSE: {m.mean_squared_error:.2f}\n\n"
            f"Translation (mm):\n"
            f"  X: {tx:.3f}\n"
            f"  Y: {ty:.3f}\n"
            f"  Z: {tz:.3f}\n"
            f"  Mag: {trans_mag:.3f}\n\n"
            f"Rotation: {rot_mag:.3f}°\n"
            f"Time: {m.registration_time:.1f}s"
        )

        self.timepoint_label.setText(f"Timepoint {t}/{self.max_timepoint}")

    def _on_plot_click(self, event):
        """Handle click on metrics plots to jump to timepoint."""
        if event.inaxes in [self.ax_translation, self.ax_rotation, self.ax_mse]:
            clicked_t = round(event.xdata) if event.xdata else None
            if clicked_t and 1 <= clicked_t <= self.max_timepoint:
                self.current_timepoint = clicked_t
                self._update_display()

    def _on_z_changed(self, value):
        """Handle z-slice spinbox change."""
        self.z_slice = value
        self._update_display()

    def _first_timepoint(self):
        self.current_timepoint = 1
        self._update_display()

    def _prev_timepoint(self):
        if self.current_timepoint > 1:
            self.current_timepoint -= 1
            self._update_display()

    def _next_timepoint(self):
        if self.current_timepoint < self.max_timepoint:
            self.current_timepoint += 1
            self._update_display()

    def _last_timepoint(self):
        self.current_timepoint = self.max_timepoint
        self._update_display()

    def _accept_registration(self):
        self.accepted = True
        self.accept()

    def keyPressEvent(self, event):
        """Handle keyboard navigation."""
        if event.key() == Qt.Key_Left:
            self._prev_timepoint()
        elif event.key() == Qt.Key_Right:
            self._next_timepoint()
        elif event.key() == Qt.Key_Up:
            if self.z_slice < self.max_z_slice:
                self.z_spinbox.setValue(self.z_slice + 1)
        elif event.key() == Qt.Key_Down:
            if self.z_slice > 0:
                self.z_spinbox.setValue(self.z_slice - 1)
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._accept_registration()
        else:
            super().keyPressEvent(event)


class RegistrationSplashWindow(QMainWindow):
    """Simple window shown behind the registration progress dialog."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProxylFit")
        self.setMinimumSize(600, 400)

        # Central widget with info
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignCenter)

        # Logo/title
        title = QLabel("ProxylFit")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("PROXYL MRI Analysis")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666666;")
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setStyleSheet(PROXYLFIT_STYLE)

    def set_status(self, text):
        self.status_label.setText(text)


def run_registration_with_progress(image_4d, spacing, output_dir, dicom_path, parent=None):
    """
    Run registration with a progress dialog.

    Parameters
    ----------
    image_4d : np.ndarray
        4D image data to register
    spacing : tuple
        Voxel spacing (x, y, z)
    output_dir : str
        Directory to save registration results
    dicom_path : str
        Path to source DICOM file
    parent : QWidget, optional
        Parent widget for the dialog

    Returns
    -------
    tuple
        (registered_4d, metrics) on success, (None, None) on cancel/error
    """
    # Show a splash window behind the progress dialog if no parent provided
    splash = None
    if parent is None:
        splash = RegistrationSplashWindow()
        splash.set_status("Registration in progress...")
        splash.show()
        # Process events to ensure window is displayed
        QApplication.processEvents()
        parent = splash

    dialog = RegistrationProgressDialog(image_4d, spacing, output_dir, dicom_path, parent)
    result = dialog.exec()

    # Close splash if we created one
    if splash is not None:
        splash.close()

    if result == QDialog.Accepted:
        return dialog.registered_4d, dialog.metrics
    return None, None


def show_registration_review_qt(original_4d, registered_4d, metrics, z_slice=None, output_dir=None, parent=None):
    """
    Show the Qt-based registration review dialog.

    Parameters
    ----------
    original_4d : np.ndarray
        Original 4D data
    registered_4d : np.ndarray
        Registered 4D data
    metrics : list
        List of RegistrationMetrics
    z_slice : int, optional
        Initial z-slice to display
    output_dir : str, optional
        Path where registration data was saved (displayed to user)
    parent : QWidget, optional
        Parent widget

    Returns
    -------
    bool
        True if user accepted, False otherwise
    """
    dialog = RegistrationReviewDialog(original_4d, registered_4d, metrics, z_slice, output_dir, parent)
    result = dialog.exec()
    return dialog.accepted
