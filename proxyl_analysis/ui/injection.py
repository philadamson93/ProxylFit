"""
Injection time selection dialog for ProxylFit.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QStatusBar,
    QMessageBox, QFileDialog,
)
from PySide6.QtCore import Signal

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .styles import init_qt_app
from .components import MatplotlibCanvas, HeaderWidget, InstructionWidget, InfoWidget, ButtonBar


class InjectionTimeSelectorDialog(QDialog):
    """Qt dialog for selecting injection time from signal data."""

    time_selected = Signal(int)  # Emits the selected time index

    def __init__(self, time: np.ndarray, signal: np.ndarray,
                 time_units: str = 'minutes', output_dir: str = './output',
                 roi_mask: Optional[np.ndarray] = None,
                 reference_image: Optional[np.ndarray] = None,
                 roi_z_slice: Optional[int] = None,
                 parent=None):
        super().__init__(parent)
        self.time = time
        self.signal = signal
        self.time_units = time_units
        self.output_dir = output_dir
        self.injection_index = 0
        # Optional ROI context for the Export CSV companion PNG. When
        # roi_mask + reference_image are both supplied, the timecourse
        # CSV is accompanied by a same-basename .png showing the
        # anatomical slice with the ROI contour drawn on it.
        self.roi_mask = roi_mask
        self.reference_image = reference_image
        self.roi_z_slice = roi_z_slice

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
            "- Timecourse data is saved later from the kinetic fit "
            "page so the per-ROI bundle stays consistent."
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

        # Button bar — no Export CSV here. Timecourse data is saved
        # from the kinetic fit page (Save button) along with the fit
        # results CSV, plot PNG, and ROI overlay PNG, all sharing one
        # auto-incremented index N. Splitting the timecourse export
        # across two pages used to let N drift, making per-ROI bundles
        # hard to reassemble.
        button_bar = ButtonBar()
        button_bar.add_button("cancel", "Cancel", self.reject, "cancel")
        button_bar.add_stretch()
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
        """Export timecourse data to CSV with a companion ROI overlay PNG.

        Default location: ``<output_dir>/kinetic_fits/timecourse_data_<N>.csv``
        with N auto-incremented per ROI so successive ROIs in a session
        don't overwrite each other. The matching ROI overlay companion
        PNG is ``kinetic_fit_roi_<N>.png`` in the same directory (same N).
        """
        import csv
        from ..io import next_indexed_path, index_from_filename

        # Default to the dataset's kinetic_fits/ subfolder with the
        # next free index — keeps per-ROI bundles together.
        kinetic_dir = Path(self.output_dir) / "kinetic_fits"
        default_path = next_indexed_path(
            kinetic_dir, "timecourse_data", ".csv"
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Timecourse CSV", str(default_path),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not save_path:
            return

        csv_file = Path(save_path)

        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'Time ({self.time_units})', 'Mean Intensity'])
                for t, s in zip(self.time, self.signal):
                    writer.writerow([f'{t:.3f}', f'{s:.6f}'])

            # Companion PNG when ROI context was provided. Filename
            # uses the matching index (kinetic_fit_roi_<N>.png) so the
            # bundle for one ROI shares one N. Falls back to the next
            # free roi_<N>.png if the user renamed the CSV.
            png_msg = ""
            if (self.roi_mask is not None
                    and self.reference_image is not None):
                from ..roi_selection import save_roi_overlay_png
                n = index_from_filename(
                    csv_file, "timecourse_data", ".csv"
                )
                if n is not None:
                    png_path = csv_file.parent / f"kinetic_fit_roi_{n}.png"
                else:
                    png_path = next_indexed_path(
                        csv_file.parent, "kinetic_fit_roi", ".png"
                    )
                try:
                    save_roi_overlay_png(
                        reference_image=self.reference_image,
                        roi_mask=self.roi_mask,
                        z_slice=self.roi_z_slice,
                        output_path=str(png_path),
                        title=(f"ROI on T1 baseline (z={self.roi_z_slice})"
                               if self.roi_z_slice is not None
                               else "ROI on T1 baseline"),
                    )
                    png_msg = f"\nROI overlay PNG: {png_path}"
                except Exception as e:
                    png_msg = f"\n(PNG companion failed: {e})"

            self.status_bar.showMessage(f"Exported to: {csv_file}")
            QMessageBox.information(self, "Export Complete",
                                  f"Data exported to:\n{csv_file}{png_msg}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    def _accept_time(self):
        """Accept the selected injection time."""
        self.time_selected.emit(self.injection_index)
        self.accept()

    def get_injection_index(self) -> int:
        """Get the selected injection index."""
        return self.injection_index


def select_injection_time_qt(time: np.ndarray, signal: np.ndarray,
                            time_units: str = 'minutes',
                            output_dir: str = './output',
                            roi_mask: Optional[np.ndarray] = None,
                            reference_image: Optional[np.ndarray] = None,
                            roi_z_slice: Optional[int] = None) -> int:
    """
    Qt-based interactive injection time selection.

    Drop-in replacement for select_injection_time(). The optional roi_mask /
    reference_image / roi_z_slice arguments are forwarded to the dialog so
    its Export CSV button can drop a companion ROI overlay PNG next to the
    saved timecourse.
    """
    app = init_qt_app()

    dialog = InjectionTimeSelectorDialog(
        time, signal, time_units, output_dir,
        roi_mask=roi_mask,
        reference_image=reference_image,
        roi_z_slice=roi_z_slice,
    )
    result = dialog.exec()

    injection_index = dialog.get_injection_index()

    if result == QDialog.Accepted:
        print(f"Injection time set: {time[injection_index]:.1f} {time_units}")
    else:
        print(f"Selection cancelled, using default: {time[injection_index]:.1f} {time_units}")

    return injection_index
