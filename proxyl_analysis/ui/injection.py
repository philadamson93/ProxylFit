"""
Injection time selection dialog for ProxylFit.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QStatusBar,
    QMessageBox, QFileDialog, QSpinBox, QPushButton,
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
                 steady_state_default: float = 100.0,
                 excluded_default: Optional[set] = None,
                 parent=None):
        super().__init__(parent)
        self.time = time
        self.signal = signal
        self.time_units = time_units
        self.output_dir = output_dir
        self.injection_index = 0
        # Default value (in time_units) for the NTE steady-state-time
        # spinbox. Mirrors the same control on the parameter map
        # options dialog so users can pick once on the injection
        # page and have the kinetic fit honour it.
        self._steady_state_default = float(steady_state_default)
        # Indices flagged as excluded from the kinetic fit. Right-
        # clicking a data point toggles its membership; the upcoming
        # fit ignores these points (typically bolus-passage indices
        # 6–7 that aren't well described by the kb/kd/knt processes).
        self.excluded_indices = set(int(i) for i in (excluded_default or ()))
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
            "- Left-click on the plot to set the injection time point\n"
            "- Right-click on a data point to exclude it from the kinetic "
            "fit (e.g. bolus-passage points 6–7); right-click again to "
            "restore. Excluded points are shown with a red ×.\n"
            "- The red vertical line shows the current injection selection."
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

        # Fit options — NTE steady-state time. Sets the lower bound
        # on knt so the non-tracer term reaches ~95% of A2 within the
        # user-specified window. Default 100 (in time_units); mirrors
        # the same spinbox on the parameter map options dialog so the
        # user can pick once and have both the kinetic fit and any
        # subsequent parameter map honour it.
        fit_options_group = QGroupBox("Fit Options")
        fit_options_layout = QVBoxLayout(fit_options_group)

        ss_row = QHBoxLayout()
        ss_label = QLabel("NTE steady-state time:")
        ss_row.addWidget(ss_label)
        self.steady_state_spin = QSpinBox()
        self.steady_state_spin.setRange(10, 500)
        self.steady_state_spin.setValue(int(round(self._steady_state_default)))
        self.steady_state_spin.setSingleStep(5)
        self.steady_state_spin.setSuffix(f" {self.time_units}")
        self.steady_state_spin.setToolTip(
            "Maximum time after the signal peak at which the non-tracer\n"
            "effect should reach steady state (within ~5% of A2). Sets\n"
            "the lower bound on knt: knt ≥ ln(20)/t_steady. Typical\n"
            "values for in-vivo PROXYL data: 70–100 minutes. Without\n"
            "this constraint, knt can drift toward 0 and inflate A2\n"
            "to absorb residuals even when the tail isn't saturating."
        )
        ss_row.addWidget(self.steady_state_spin)
        ss_row.addStretch()
        fit_options_layout.addLayout(ss_row)

        ss_hint = QLabel("knt lower bound = ln(20)/t_steady")
        ss_hint.setStyleSheet("color: #666; font-size: 11px;")
        fit_options_layout.addWidget(ss_hint)

        info_layout.addWidget(fit_options_group)

        # Excluded-points panel — live readout of which indices are
        # currently flagged for exclusion plus a one-click reset. The
        # actual toggling happens on the plot via right-click.
        excl_group = QGroupBox("Excluded points (right-click on plot)")
        excl_layout = QVBoxLayout(excl_group)
        self.excluded_label = QLabel(self._format_excluded_text())
        self.excluded_label.setWordWrap(True)
        self.excluded_label.setStyleSheet("padding: 4px;")
        excl_layout.addWidget(self.excluded_label)
        clear_btn = QPushButton("Clear all exclusions")
        clear_btn.clicked.connect(self._clear_exclusions)
        excl_layout.addWidget(clear_btn)
        info_layout.addWidget(excl_group)

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

        # Excluded-points overlay (red ×). Updated whenever the user
        # right-clicks to toggle a point. Empty (x=[], y=[]) until
        # the first toggle so it's never visible until needed.
        self.excluded_scatter = self.ax.scatter(
            [], [], marker='x', color='red', s=80, linewidths=2.5,
            zorder=5, label='Excluded'
        )
        self._refresh_excluded_marks()

        # Connect mouse-click events. Left = injection time,
        # right = toggle exclude.
        self.canvas.mpl_connect('button_press_event', self._on_click)

        self.canvas.draw()

    def _on_click(self, event):
        """Handle click events on the plot.

        Left-click sets the injection time; right-click toggles the
        nearest data point's exclusion from the kinetic fit.
        """
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            self._on_left_click(event)
        elif event.button == 3:
            self._on_right_click(event)

    def _on_left_click(self, event):
        """Pick injection time from the closest time point."""
        closest_idx = int(np.argmin(np.abs(self.time - event.xdata)))
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

    def _on_right_click(self, event):
        """Toggle the nearest data point's exclusion flag.

        Distance is measured in display (pixel) coordinates so the
        click feels right regardless of axis aspect ratio. A small
        radius cap prevents accidental toggles when the user clicks
        far from any actual point.
        """
        if event.x is None or event.y is None:
            return

        # Convert each (time, signal) pair to display coords and find
        # the closest one to the click.
        xdata = np.asarray(self.time)
        ydata = np.asarray(self.signal)
        xy_pix = self.ax.transData.transform(np.column_stack([xdata, ydata]))
        dx = xy_pix[:, 0] - event.x
        dy = xy_pix[:, 1] - event.y
        dist = np.hypot(dx, dy)
        nearest = int(np.argmin(dist))

        # Only toggle when the click is reasonably close to a point.
        # 14 px ≈ marker size + a little slack.
        if dist[nearest] > 14:
            self.status_bar.showMessage(
                "Right-click closer to a data point to toggle exclude."
            )
            return

        if nearest in self.excluded_indices:
            self.excluded_indices.discard(nearest)
            self.status_bar.showMessage(
                f"Restored point at index {nearest} (will be fit)."
            )
        else:
            self.excluded_indices.add(nearest)
            self.status_bar.showMessage(
                f"Excluded point at index {nearest} from fit."
            )

        self._refresh_excluded_marks()
        self.excluded_label.setText(self._format_excluded_text())
        self.canvas.draw()

    def _refresh_excluded_marks(self):
        """Update the red-× scatter overlay to match excluded_indices."""
        if not self.excluded_indices:
            self.excluded_scatter.set_offsets(np.empty((0, 2)))
            return
        idx = sorted(self.excluded_indices)
        pts = np.column_stack([self.time[idx], self.signal[idx]])
        self.excluded_scatter.set_offsets(pts)

    def _format_excluded_text(self) -> str:
        """Render the excluded-indices summary for the side panel."""
        if not self.excluded_indices:
            return "(none — right-click data points on the plot to exclude)"
        idx_str = ", ".join(str(i) for i in sorted(self.excluded_indices))
        return f"Indices: {idx_str}\nCount: {len(self.excluded_indices)}"

    def _clear_exclusions(self):
        """Reset the excluded-indices set and redraw the plot."""
        if not self.excluded_indices:
            return
        self.excluded_indices.clear()
        self._refresh_excluded_marks()
        self.excluded_label.setText(self._format_excluded_text())
        self.canvas.draw()
        self.status_bar.showMessage("Cleared all point exclusions.")

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

    def get_steady_state_time(self) -> float:
        """Return the user-set NTE steady-state time (in time_units)."""
        try:
            return float(self.steady_state_spin.value())
        except RuntimeError:
            # Widget already destroyed (e.g. dialog closed) — fall
            # back to the default that was passed in.
            return float(self._steady_state_default)

    def get_excluded_indices(self) -> list:
        """Return the sorted list of indices flagged for exclusion."""
        return sorted(int(i) for i in self.excluded_indices)


def select_injection_time_qt(time: np.ndarray, signal: np.ndarray,
                            time_units: str = 'minutes',
                            output_dir: str = './output',
                            roi_mask: Optional[np.ndarray] = None,
                            reference_image: Optional[np.ndarray] = None,
                            roi_z_slice: Optional[int] = None,
                            steady_state_default: float = 100.0,
                            excluded_default: Optional[set] = None,
                            return_steady_state: bool = False):
    """
    Qt-based interactive injection time selection.

    Drop-in replacement for select_injection_time(). The optional roi_mask /
    reference_image / roi_z_slice arguments are forwarded to the dialog so
    its Export CSV button can drop a companion ROI overlay PNG next to the
    saved timecourse.

    The dialog also carries an NTE steady-state-time spinbox (default 100
    in time_units) that the upcoming kinetic fit uses to bound knt from
    below: ``knt_lower = ln(20) / t_steady``. Right-clicking points on
    the plot toggles them as excluded from the fit (typically bolus-
    passage points 6–7).

    Pass ``return_steady_state=True`` to receive a
    ``(injection_index, steady_state_time, excluded_indices)`` triple
    instead of just the index.
    """
    app = init_qt_app()

    dialog = InjectionTimeSelectorDialog(
        time, signal, time_units, output_dir,
        roi_mask=roi_mask,
        reference_image=reference_image,
        roi_z_slice=roi_z_slice,
        steady_state_default=steady_state_default,
        excluded_default=excluded_default,
    )
    result = dialog.exec()

    injection_index = dialog.get_injection_index()
    steady_state = dialog.get_steady_state_time()
    excluded = dialog.get_excluded_indices()

    if result == QDialog.Accepted:
        print(f"Injection time set: {time[injection_index]:.1f} {time_units}")
        print(
            f"Fit option: NTE steady-state time = {steady_state:.0f} {time_units} "
            f"(knt ≥ {np.log(20.0) / steady_state:.4f}/{time_units})"
        )
        if excluded:
            print(
                f"Fit option: {len(excluded)} point(s) excluded from fit "
                f"(indices {excluded})"
            )
    else:
        print(f"Selection cancelled, using default: {time[injection_index]:.1f} {time_units}")

    if return_steady_state:
        return injection_index, steady_state, excluded
    return injection_index
