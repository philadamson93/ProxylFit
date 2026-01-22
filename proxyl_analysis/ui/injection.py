"""
Injection time selection dialog for ProxylFit.
"""

from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QStatusBar, QMessageBox
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
