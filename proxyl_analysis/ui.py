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
    QMessageBox, QSizePolicy, QSlider, QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QSize, QThread
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
# Registration Progress Components
# ============================================================================

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
            from .registration import register_timeseries

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

        # Update images
        self.ax_ref.clear()
        self.ax_ref.imshow(ref_slice, cmap='gray', origin='lower')
        self.ax_ref.set_title(f'Reference (t=0)', fontsize=10)
        self.ax_ref.axis('off')

        self.ax_registered.clear()
        self.ax_registered.imshow(reg_slice, cmap='gray', origin='lower')
        self.ax_registered.set_title(f'Registered t={t}', fontsize=10)
        self.ax_registered.axis('off')

        self.ax_diff.clear()
        im_diff = self.ax_diff.imshow(diff_slice, cmap='hot', origin='lower')
        self.ax_diff.set_title(f'|Reference - Registered|', fontsize=10)
        self.ax_diff.axis('off')

        self.ax_unregistered.clear()
        self.ax_unregistered.imshow(orig_slice, cmap='gray', origin='lower')
        self.ax_unregistered.set_title(f'Original t={t} (unregistered)', fontsize=10)
        self.ax_unregistered.axis('off')

        self.fig_images.tight_layout()
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


# ============================================================================
# Main Workflow Menu
# ============================================================================

class MainMenuDialog(QDialog):
    """
    Main workflow menu shown after registration completes.

    Provides access to:
    - Load new experiment / Load previous session
    - ROI Analysis (with T2 as default source)
    - Parameter Maps (sliding window, pixel-level)
    - Image Tools (averaged, difference images)
    - Export options
    """

    # Signals for workflow actions
    roi_analysis_requested = Signal(dict)  # Emits ROI analysis settings
    parameter_maps_requested = Signal(dict)  # Emits parameter map settings
    export_requested = Signal(str)  # Emits export type

    def __init__(self,
                 registered_4d: Optional[np.ndarray] = None,
                 spacing: Optional[Tuple] = None,
                 time_array: Optional[np.ndarray] = None,
                 dicom_path: str = "",
                 output_dir: str = './output',
                 registered_t2: Optional[np.ndarray] = None,
                 roi_state: Optional[dict] = None,
                 parent=None):
        super().__init__(parent)
        self.registered_4d = registered_4d
        self.spacing = spacing
        self.time_array = time_array
        self.dicom_path = dicom_path
        self.output_dir = output_dir
        self.registered_t2 = registered_t2

        # State - persisted across menu returns via roi_state dict
        if roi_state:
            self.roi_mask = roi_state.get('roi_mask')
            self.roi_signal = roi_state.get('roi_signal')
            self.injection_idx = roi_state.get('injection_idx')
            self.injection_time = roi_state.get('injection_time')
        else:
            self.roi_mask = None
            self.roi_signal = None
            self.injection_idx = None
            self.injection_time = None
        self.result = None  # Stores the user's action

        # Determine max z-slice
        if registered_4d is not None:
            self.max_z = registered_4d.shape[2] - 1
        else:
            self.max_z = 8  # Default

        self.setWindowTitle("ProxylFit - Analysis Menu")
        self.setMinimumSize(700, 800)
        self.resize(750, 850)

        self._setup_ui()
        self._update_data_status()

    def _setup_ui(self):
        """Build the menu UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area to handle overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = HeaderWidget("ProxylFit Analysis Menu", "Select an analysis workflow")
        layout.addWidget(header)

        # Experiment section
        self._create_experiment_section(layout)

        # Data status
        self._create_data_status_section(layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # ROI Analysis section
        self._create_roi_section(layout)

        # Parameter Maps section
        self._create_param_maps_section(layout)

        # Image Tools section
        self._create_image_tools_section(layout)

        # Export section
        self._create_export_section(layout)

        # Spacer at bottom of scrollable content
        layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Footer with Exit button (outside scroll area, always visible)
        footer = ButtonBar()
        footer.add_button("exit", "Exit", self._on_exit, "cancel")
        main_layout.addWidget(footer)

    def _create_experiment_section(self, parent_layout):
        """Create the Experiment section for loading data."""
        group = QGroupBox("Experiment")
        layout = QVBoxLayout(group)

        # Buttons row
        btn_layout = QHBoxLayout()

        load_new_btn = QPushButton("Load New T1 DICOM...")
        load_new_btn.clicked.connect(self._load_new_experiment)
        btn_layout.addWidget(load_new_btn)

        load_prev_btn = QPushButton("Load Previous Session...")
        load_prev_btn.clicked.connect(self._load_previous_session)
        btn_layout.addWidget(load_prev_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Current experiment info
        self.experiment_info_label = QLabel("No data loaded")
        self.experiment_info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.experiment_info_label)

        parent_layout.addWidget(group)

    def _create_data_status_section(self, parent_layout):
        """Create the data status display."""
        status_layout = QHBoxLayout()

        # T1 status
        self.t1_status_label = QLabel("T1 Data: Not loaded")
        status_layout.addWidget(self.t1_status_label)

        status_layout.addSpacing(20)

        # Registration status
        self.reg_status_label = QLabel("Registration: —")
        status_layout.addWidget(self.reg_status_label)

        status_layout.addStretch()

        # T2 status and load button
        self.t2_status_label = QLabel("T2 Data: Not loaded")
        status_layout.addWidget(self.t2_status_label)

        self.load_t2_btn = QPushButton("Load T2 Volume...")
        self.load_t2_btn.clicked.connect(self._load_t2_volume)
        self.load_t2_btn.setEnabled(self.registered_4d is not None)
        status_layout.addWidget(self.load_t2_btn)

        parent_layout.addLayout(status_layout)

    def _create_roi_section(self, parent_layout):
        """Create the ROI Analysis section."""
        group = QGroupBox("ROI Analysis")
        layout = QVBoxLayout(group)

        description = QLabel("Draw ROI to extract time series and set injection time")
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

        # ROI Source
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("ROI Source:"))

        self.t2_source_radio = QRadioButton("T2")
        self.t1_source_radio = QRadioButton("T1")
        self.t2_source_radio.setChecked(True)  # T2 is default

        # Disable T2 option if not loaded
        self.t2_source_radio.setEnabled(self.registered_t2 is not None)
        if self.registered_t2 is None:
            self.t1_source_radio.setChecked(True)

        source_layout.addWidget(self.t2_source_radio)
        source_layout.addWidget(self.t1_source_radio)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # ROI Method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("ROI Method:"))

        self.rect_radio = QRadioButton("Rectangle")
        self.contour_radio = QRadioButton("Manual Contour")
        self.segment_radio = QRadioButton("Segment")
        self.contour_radio.setChecked(True)  # Default

        method_layout.addWidget(self.rect_radio)
        method_layout.addWidget(self.contour_radio)
        method_layout.addWidget(self.segment_radio)
        method_layout.addStretch()
        layout.addLayout(method_layout)

        # Z-slice
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-slice:"))

        self.z_spinbox = QSpinBox()
        self.z_spinbox.setMinimum(0)
        self.z_spinbox.setMaximum(self.max_z)
        self.z_spinbox.setValue(min(4, self.max_z))
        z_layout.addWidget(self.z_spinbox)

        self.z_max_label = QLabel(f"/ {self.max_z}")
        z_layout.addWidget(self.z_max_label)

        z_layout.addStretch()
        layout.addLayout(z_layout)

        # ROI status line
        self.roi_status_label = QLabel("ROI: Not drawn")
        self.roi_status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.roi_status_label)

        # Buttons row - Draw ROI and Run Kinetic Fit
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # Draw ROI button - green
        self.start_roi_btn = QPushButton("Draw ROI")
        self.start_roi_btn.setMinimumSize(120, 40)
        self.start_roi_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.start_roi_btn.clicked.connect(self._draw_roi)
        self.start_roi_btn.setEnabled(self.registered_4d is not None)
        btn_layout.addWidget(self.start_roi_btn)

        btn_layout.addSpacing(15)

        # Run Kinetic Fit button - orange, requires ROI + injection time
        self.kinetic_fit_btn = QPushButton("Run Kinetic Fit")
        self.kinetic_fit_btn.setMinimumSize(140, 40)
        self.kinetic_fit_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #F57C00; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.kinetic_fit_btn.clicked.connect(self._run_kinetic_fit)
        self.kinetic_fit_btn.setEnabled(False)  # Enabled when ROI + injection time set
        self.kinetic_fit_btn.setToolTip("Draw ROI and select injection time first")
        btn_layout.addWidget(self.kinetic_fit_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        parent_layout.addWidget(group)

    def _create_param_maps_section(self, parent_layout):
        """Create the Parameter Maps section."""
        group = QGroupBox("Parameter Maps")
        layout = QVBoxLayout(group)

        description = QLabel("Generate spatial parameter maps across the image")
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

        # Map type selection
        type_layout = QHBoxLayout()

        self.sliding_window_radio = QRadioButton("Sliding Window")
        self.pixel_level_radio = QRadioButton("Pixel-Level (slower, full resolution)")
        self.sliding_window_radio.setChecked(True)

        # Pixel-level not yet implemented
        self.pixel_level_radio.setEnabled(False)
        self.pixel_level_radio.setToolTip("Coming soon (T005)")

        type_layout.addWidget(self.sliding_window_radio)
        type_layout.addWidget(self.pixel_level_radio)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # Window size (for sliding window)
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))

        self.window_x_spin = QSpinBox()
        self.window_x_spin.setRange(3, 31)
        self.window_x_spin.setValue(15)
        self.window_x_spin.setSingleStep(2)
        window_layout.addWidget(self.window_x_spin)

        window_layout.addWidget(QLabel("x"))

        self.window_y_spin = QSpinBox()
        self.window_y_spin.setRange(3, 31)
        self.window_y_spin.setValue(15)
        self.window_y_spin.setSingleStep(2)
        window_layout.addWidget(self.window_y_spin)

        window_layout.addWidget(QLabel("x"))

        self.window_z_spin = QSpinBox()
        self.window_z_spin.setRange(1, 9)
        self.window_z_spin.setValue(3)
        window_layout.addWidget(self.window_z_spin)

        window_layout.addWidget(QLabel("voxels"))
        window_layout.addStretch()
        layout.addLayout(window_layout)

        # Scope
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))

        self.whole_image_radio = QRadioButton("Whole image")
        self.within_roi_radio = QRadioButton("Within ROI (select first)")
        self.whole_image_radio.setChecked(True)

        scope_layout.addWidget(self.whole_image_radio)
        scope_layout.addWidget(self.within_roi_radio)
        scope_layout.addStretch()
        layout.addLayout(scope_layout)

        # Create button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.create_maps_btn = QPushButton("Create Parameter Maps")
        self.create_maps_btn.setMinimumSize(180, 40)
        self.create_maps_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.create_maps_btn.clicked.connect(self._create_parameter_maps)
        self.create_maps_btn.setEnabled(self.registered_4d is not None)
        btn_layout.addWidget(self.create_maps_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        parent_layout.addWidget(group)

    def _create_image_tools_section(self, parent_layout):
        """Create the Image Tools section."""
        group = QGroupBox("Image Tools")
        layout = QVBoxLayout(group)

        description = QLabel("Select time ranges on signal curve to generate processed images. Requires ROI + injection time.")
        description.setStyleSheet("color: #666;")
        description.setWordWrap(True)
        layout.addWidget(description)

        # Buttons
        btn_layout = QHBoxLayout()

        self.averaged_btn = QPushButton("Averaged Image")
        self.averaged_btn.clicked.connect(self._create_averaged_image)
        self.averaged_btn.setEnabled(False)  # Enabled after ROI + injection time
        self.averaged_btn.setToolTip("Draw ROI and select injection time first")
        btn_layout.addWidget(self.averaged_btn)

        self.difference_btn = QPushButton("Difference Image")
        self.difference_btn.clicked.connect(self._create_difference_image)
        self.difference_btn.setEnabled(False)  # Enabled after ROI + injection time
        self.difference_btn.setToolTip("Draw ROI and select injection time first")
        btn_layout.addWidget(self.difference_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        parent_layout.addWidget(group)

    def _create_export_section(self, parent_layout):
        """Create the Export section."""
        group = QGroupBox("Export")
        layout = QHBoxLayout(group)

        self.export_data_btn = QPushButton("Registered 4D Data")
        self.export_data_btn.clicked.connect(lambda: self._export("registered_data"))
        self.export_data_btn.setEnabled(self.registered_4d is not None)
        layout.addWidget(self.export_data_btn)

        self.export_report_btn = QPushButton("Registration Report")
        self.export_report_btn.clicked.connect(lambda: self._export("registration_report"))
        self.export_report_btn.setEnabled(self.registered_4d is not None)
        layout.addWidget(self.export_report_btn)

        self.export_timeseries_btn = QPushButton("Time Series CSV")
        self.export_timeseries_btn.clicked.connect(lambda: self._export("timeseries"))
        self.export_timeseries_btn.setEnabled(self.roi_signal is not None)
        layout.addWidget(self.export_timeseries_btn)

        layout.addStretch()

        parent_layout.addWidget(group)

    def _update_data_status(self):
        """Update all data status displays."""
        # T1 status
        if self.registered_4d is not None:
            shape = self.registered_4d.shape
            self.t1_status_label.setText(f"T1 Data: {shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}")
            self.reg_status_label.setText("Registration: Complete")
            self.reg_status_label.setStyleSheet("color: green;")

            # Update experiment info
            dicom_name = Path(self.dicom_path).name if self.dicom_path else "Unknown"
            self.experiment_info_label.setText(f"Current: {dicom_name}\nOutput: {self.output_dir}")
        else:
            self.t1_status_label.setText("T1 Data: Not loaded")
            self.reg_status_label.setText("Registration: —")
            self.reg_status_label.setStyleSheet("")
            self.experiment_info_label.setText("No data loaded")

        # T2 status
        if self.registered_t2 is not None:
            self.t2_status_label.setText("T2 Data: Loaded")
            self.t2_status_label.setStyleSheet("color: green;")
            self.t2_source_radio.setEnabled(True)
            self.t2_source_radio.setChecked(True)
        else:
            self.t2_status_label.setText("T2 Data: Not loaded")
            self.t2_status_label.setStyleSheet("")
            self.t2_source_radio.setEnabled(False)
            self.t1_source_radio.setChecked(True)

        # Update button states
        has_data = self.registered_4d is not None
        self.load_t2_btn.setEnabled(has_data)
        self.start_roi_btn.setEnabled(has_data)
        self.create_maps_btn.setEnabled(has_data)
        self.export_data_btn.setEnabled(has_data)

        # Update ROI status (enables kinetic fit, image tools if ROI exists)
        self._update_roi_status()
        self.export_report_btn.setEnabled(has_data)

        # Update z-slice bounds
        if has_data:
            self.max_z = self.registered_4d.shape[2] - 1
            self.z_spinbox.setMaximum(self.max_z)
            self.z_max_label.setText(f"/ {self.max_z}")

    def _load_new_experiment(self):
        """Load a new T1 DICOM and run registration."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load T1 DICOM", "",
            "DICOM Files (*.dcm);;All Files (*)"
        )
        if not file_path:
            return

        # Store result for caller to handle
        self.result = {
            'action': 'load_new',
            'dicom_path': file_path
        }
        self.accept()

    def _load_previous_session(self):
        """Load a previous session from saved registration data."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Previous Session Folder",
            str(Path(self.output_dir).parent) if self.output_dir else "."
        )
        if not folder_path:
            return

        # Check if valid registration data exists
        reg_data_file = Path(folder_path) / "registered_4d_data.npz"
        if not reg_data_file.exists():
            QMessageBox.warning(
                self, "Invalid Session",
                f"No registration data found in:\n{folder_path}\n\n"
                "Expected file: registered_4d_data.npz"
            )
            return

        # Store result for caller to handle
        self.result = {
            'action': 'load_previous',
            'session_path': folder_path
        }
        self.accept()

    def _load_t2_volume(self):
        """Open file dialog and load T2 volume."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load T2 DICOM", "",
            "DICOM Files (*.dcm);;All Files (*)"
        )
        if not file_path:
            return

        # Store result for caller to handle T2 registration
        self.result = {
            'action': 'load_t2',
            't2_path': file_path
        }
        self.accept()

    def _draw_roi(self):
        """Launch ROI drawing workflow (just ROI + injection time, no fitting)."""
        # Gather settings
        if self.t2_source_radio.isChecked() and self.registered_t2 is not None:
            roi_source = 't2'
        else:
            roi_source = 't1'

        if self.rect_radio.isChecked():
            roi_mode = 'rectangle'
        elif self.segment_radio.isChecked():
            roi_mode = 'segment'
        else:
            roi_mode = 'contour'

        self.result = {
            'action': 'draw_roi',
            'roi_source': roi_source,
            'roi_mode': roi_mode,
            'z_slice': self.z_spinbox.value()
        }
        self.accept()

    def _run_kinetic_fit(self):
        """Launch kinetic fitting on existing ROI data."""
        if self.roi_mask is None or self.roi_signal is None or self.injection_idx is None:
            QMessageBox.warning(
                self, "Missing Data",
                "Please draw an ROI and select injection time first."
            )
            return

        self.result = {
            'action': 'kinetic_fit',
            'roi_mask': self.roi_mask,
            'roi_signal': self.roi_signal,
            'injection_idx': self.injection_idx,
            'injection_time': self.injection_time
        }
        self.accept()

    def set_roi_data(self, roi_mask: np.ndarray, roi_signal: np.ndarray,
                     injection_idx: int, injection_time: float):
        """Set ROI data after drawing (called by run_analysis.py)."""
        self.roi_mask = roi_mask
        self.roi_signal = roi_signal
        self.injection_idx = injection_idx
        self.injection_time = injection_time
        self._update_roi_status()

    def _update_roi_status(self):
        """Update ROI status display and button states."""
        if self.roi_mask is not None and self.injection_idx is not None:
            num_pixels = int(np.sum(self.roi_mask))
            self.roi_status_label.setText(
                f"ROI: {num_pixels} pixels | Injection: t={self.injection_idx}"
            )
            self.roi_status_label.setStyleSheet("color: green; font-weight: bold;")
            # Enable kinetic fit button
            self.kinetic_fit_btn.setEnabled(True)
            self.kinetic_fit_btn.setToolTip("")
            # Enable image tools
            self.averaged_btn.setEnabled(True)
            self.averaged_btn.setToolTip("")
            self.difference_btn.setEnabled(True)
            self.difference_btn.setToolTip("")
            # Enable time series export
            self.export_timeseries_btn.setEnabled(True)
        elif self.roi_mask is not None:
            num_pixels = int(np.sum(self.roi_mask))
            self.roi_status_label.setText(f"ROI: {num_pixels} pixels | Injection: Not set")
            self.roi_status_label.setStyleSheet("color: #FF9800;")
        else:
            self.roi_status_label.setText("ROI: Not drawn")
            self.roi_status_label.setStyleSheet("color: #666; font-style: italic;")

    def _create_parameter_maps(self):
        """Launch parameter mapping workflow."""
        self.result = {
            'action': 'parameter_maps',
            'pixel_level': self.pixel_level_radio.isChecked(),
            'window_size': (
                self.window_x_spin.value(),
                self.window_y_spin.value(),
                self.window_z_spin.value()
            ),
            'within_roi': self.within_roi_radio.isChecked()
        }
        self.accept()

    def _create_averaged_image(self):
        """Launch averaged image tool (T002)."""
        if self.roi_signal is None:
            QMessageBox.warning(
                self, "Missing Data",
                "Please draw an ROI and select injection time first."
            )
            return

        self.result = {
            'action': 'image_tools',
            'mode': 'average',
            'roi_signal': self.roi_signal
        }
        self.accept()

    def _create_difference_image(self):
        """Launch difference image tool (T003)."""
        if self.roi_signal is None:
            QMessageBox.warning(
                self, "Missing Data",
                "Please draw an ROI and select injection time first."
            )
            return

        self.result = {
            'action': 'image_tools',
            'mode': 'difference',
            'roi_signal': self.roi_signal
        }
        self.accept()

    def _export(self, export_type: str):
        """Handle export requests."""
        self.result = {
            'action': 'export',
            'export_type': export_type
        }
        self.accept()

    def _on_exit(self):
        """Handle exit button."""
        self.result = {'action': 'exit'}
        self.reject()

    def get_result(self) -> Optional[dict]:
        """Get the result after dialog closes."""
        return self.result


def show_main_menu(registered_4d: Optional[np.ndarray] = None,
                   spacing: Optional[Tuple] = None,
                   time_array: Optional[np.ndarray] = None,
                   dicom_path: str = "",
                   output_dir: str = './output',
                   registered_t2: Optional[np.ndarray] = None,
                   roi_state: Optional[dict] = None) -> Optional[dict]:
    """
    Show the main workflow menu.

    Parameters
    ----------
    registered_4d : np.ndarray, optional
        Registered 4D image data [x, y, z, t]
    spacing : tuple, optional
        Voxel spacing (x, y, z)
    time_array : np.ndarray, optional
        Time array for the data
    dicom_path : str
        Path to the source DICOM file
    output_dir : str
        Output directory path
    registered_t2 : np.ndarray, optional
        Registered T2 volume (if loaded)
    roi_state : dict, optional
        Preserved ROI state with keys: roi_mask, roi_signal, injection_idx, injection_time

    Returns
    -------
    dict or None
        User's action and settings, or None if cancelled
    """
    app = init_qt_app()

    dialog = MainMenuDialog(
        registered_4d=registered_4d,
        spacing=spacing,
        time_array=time_array,
        dicom_path=dicom_path,
        output_dir=output_dir,
        registered_t2=registered_t2,
        roi_state=roi_state
    )

    result = dialog.exec()

    if result == QDialog.Accepted:
        return dialog.get_result()
    else:
        return dialog.get_result()  # May contain 'exit' action


# =============================================================================
# T002/T003: Image Tools Dialog (Averaged and Difference Images)
# =============================================================================

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
    """

    def __init__(self,
                 image_4d: np.ndarray,
                 time_array: np.ndarray,
                 roi_signal: np.ndarray,
                 time_units: str = 'minutes',
                 output_dir: str = './output',
                 initial_mode: str = 'average',
                 parent=None):
        super().__init__(parent)
        self.image_4d = image_4d
        self.time_array = time_array
        self.roi_signal = roi_signal
        self.time_units = time_units
        self.output_dir = output_dir
        self.num_timepoints = len(time_array)
        self.num_slices = image_4d.shape[2]

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

        region_a_layout.addWidget(QLabel("End:"))
        self.region_a_end_spin = QSpinBox()
        self.region_a_end_spin.setRange(0, self.num_timepoints - 1)
        self.region_a_end_spin.valueChanged.connect(self._on_region_a_end_changed)
        region_a_layout.addWidget(self.region_a_end_spin)

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

        region_b_layout.addWidget(QLabel("End:"))
        self.region_b_end_spin = QSpinBox()
        self.region_b_end_spin.setRange(0, self.num_timepoints - 1)
        self.region_b_end_spin.valueChanged.connect(self._on_region_b_end_changed)
        region_b_layout.addWidget(self.region_b_end_spin)

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
        self.preview_btn = QPushButton("Update Preview")
        self.preview_btn.clicked.connect(self._update_preview)
        self.preview_btn.setEnabled(False)
        right_layout.addWidget(self.preview_btn)

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

    def _on_region_a_start_changed(self, value):
        """Handle region A start spinbox change."""
        self.region_a_start = value
        self._update_plot()
        self._check_can_preview()

    def _on_region_a_end_changed(self, value):
        """Handle region A end spinbox change."""
        self.region_a_end = value
        self._update_plot()
        self._check_can_preview()

    def _on_region_b_start_changed(self, value):
        """Handle region B start spinbox change."""
        self.region_b_start = value
        self._update_plot()
        self._check_can_preview()

    def _on_region_b_end_changed(self, value):
        """Handle region B end spinbox change."""
        self.region_b_end = value
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
        """Check if we have enough data to preview."""
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

    def _update_plot(self):
        """Update the time series plot with region highlights."""
        self.ax.clear()

        # Plot signal
        self.ax.plot(self.time_array, self.roi_signal, 'k-', linewidth=1.5)

        # Highlight region A (blue)
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
        """Display the preview image."""
        if self.preview_image is None:
            return

        self.preview_ax.clear()

        slice_data = self.preview_image[:, :, self.current_z].T

        if self.mode == 'average':
            im = self.preview_ax.imshow(slice_data, cmap='gray', aspect='equal')
            self.preview_ax.set_title(f'Averaged Image (z={self.current_z})')
        else:
            # Diverging colormap centered at 0
            vmax = np.max(np.abs(slice_data))
            vmin = -vmax
            im = self.preview_ax.imshow(slice_data, cmap='RdBu_r', aspect='equal',
                                        vmin=vmin, vmax=vmax)
            self.preview_ax.set_title(f'Difference B-A (z={self.current_z})')

        self.preview_ax.axis('off')
        self.preview_figure.tight_layout(pad=1)
        self.preview_canvas.draw()

    def _save_image(self):
        """Save the computed image to disk."""
        if self.preview_image is None:
            self._update_preview()
            if self.preview_image is None:
                QMessageBox.warning(self, "No Image", "Please select regions first.")
                return

        # Generate filename
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.mode == 'average':
            start_idx = min(self.region_a_start, self.region_a_end)
            end_idx = max(self.region_a_start, self.region_a_end)
            filename = f"averaged_image_frames{start_idx}-{end_idx}.npz"

            metadata = {
                'type': 'averaged_image',
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
            filename = f"difference_image_A{a_start}-{a_end}_B{b_start}-{b_end}.npz"

            metadata = {
                'type': 'difference_image',
                'region_a': {'start_idx': a_start, 'end_idx': a_end},
                'region_b': {'start_idx': b_start, 'end_idx': b_end},
                'description': f'frames {b_start}-{b_end} minus frames {a_start}-{a_end}',
                'time_units': self.time_units
            }

        filepath = output_path / filename
        np.savez(filepath, image=self.preview_image, **metadata)

        QMessageBox.information(
            self, "Saved",
            f"Image saved to:\n{filepath}"
        )


def show_image_tools_dialog(image_4d: np.ndarray,
                            time_array: np.ndarray,
                            roi_signal: np.ndarray,
                            time_units: str = 'minutes',
                            output_dir: str = './output',
                            initial_mode: str = 'average') -> None:
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
    """
    app = init_qt_app()

    dialog = ImageToolsDialog(
        image_4d=image_4d,
        time_array=time_array,
        roi_signal=roi_signal,
        time_units=time_units,
        output_dir=output_dir,
        initial_mode=initial_mode
    )

    dialog.exec()
