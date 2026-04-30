"""
Fit results dialog for ProxylFit.
"""

import csv
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QFileDialog, QMessageBox
)

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .styles import init_qt_app
from .components import MatplotlibCanvas, HeaderWidget, ButtonBar


class FitResultsDialog(QDialog):
    """Qt dialog for displaying fit results."""

    def __init__(self, time: np.ndarray, signal: np.ndarray,
                 fitted_signal: np.ndarray, fit_results: Dict,
                 save_path: Optional[str] = None,
                 roi_mask: Optional[np.ndarray] = None,
                 reference_image: Optional[np.ndarray] = None,
                 roi_z_slice: Optional[int] = None,
                 parent=None):
        super().__init__(parent)
        self.time = time
        self.signal = signal
        self.fitted_signal = fitted_signal
        self.fit_results = fit_results
        self.save_path = save_path
        # Optional ROI context for the Save Results Table companion PNG.
        # When all three are present, saving the CSV also writes a
        # same-basename .png showing the reference image with the ROI
        # contour overlaid, so the kinetic data is self-documenting.
        self.roi_mask = roi_mask
        self.reference_image = reference_image
        self.roi_z_slice = roi_z_slice

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

        # Initial estimates (may be absent on legacy sessions)
        a0_est = self.fit_results.get('A0_est')
        a2_est = self.fit_results.get('A2_est')
        a0_init = f"  (init: {a0_est:.3f})" if a0_est is not None else ""
        a2_init = f"  (init: {a2_est:.3f})" if a2_est is not None else ""

        params = [
            ("kb (buildup)", f"{self.fit_results['kb']:.4f} ± {self.fit_results['kb_error']:.4f}"),
            ("kd (decay)", f"{self.fit_results['kd']:.4f} ± {self.fit_results['kd_error']:.4f}"),
            ("knt (non-tracer)", f"{self.fit_results['knt']:.4f} ± {self.fit_results['knt_error']:.4f}"),
            ("A0 (baseline)", f"{self.fit_results['A0']:.3f} ± {self.fit_results['A0_error']:.3f}{a0_init}"),
            ("A1 (amplitude)", f"{self.fit_results['A1']:.3f} ± {self.fit_results['A1_error']:.3f}"),
            ("A2 (non-tracer)", f"{self.fit_results['A2']:.3f} ± {self.fit_results['A2_error']:.3f}{a2_init}"),
            ("t0 (onset)", f"{self.fit_results['t0']:.2f} ± {self.fit_results['t0_error']:.2f}"),
            ("tmax (NTE onset)", f"{self.fit_results['tmax']:.2f} ± {self.fit_results['tmax_error']:.2f}"),
        ]

        for i, (name, value) in enumerate(params):
            params_grid.addWidget(QLabel(f"{name}:"), i, 0)
            params_grid.addWidget(QLabel(value), i, 1)

        params_layout.addWidget(params_group)

        # Derived parameters
        derived_group = QGroupBox("Derived Parameters")
        derived_grid = QGridLayout(derived_group)

        A0 = self.fit_results['A0']
        A1 = self.fit_results['A1']
        A2 = self.fit_results['A2']

        if A0 != 0:
            self.pct_enhancement = (A1 / A0) * 100
            self.pct_nte = (A2 / A0) * 100
        else:
            self.pct_enhancement = float('nan')
            self.pct_nte = float('nan')

        # Initial-estimate version of %NTE (may be absent on legacy sessions)
        if a0_est is not None and a2_est is not None and a0_est != 0:
            self.pct_nte_est = (a2_est / a0_est) * 100
        else:
            self.pct_nte_est = float('nan')

        derived_grid.addWidget(QLabel("%Enhancement (A1/A0):"), 0, 0)
        derived_grid.addWidget(QLabel(f"{self.pct_enhancement:.1f}%"), 0, 1)
        derived_grid.addWidget(QLabel("%NTE (A2/A0):"), 1, 0)
        derived_grid.addWidget(QLabel(f"{self.pct_nte:.1f}%"), 1, 1)
        derived_grid.addWidget(QLabel("%NTE_est (A2_est/A0_est):"), 2, 0)
        nte_est_text = f"{self.pct_nte_est:.1f}%" if not np.isnan(self.pct_nte_est) else "—"
        derived_grid.addWidget(QLabel(nte_est_text), 2, 1)

        params_layout.addWidget(derived_group)

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
        button_bar.add_button("save_table", "Save Results Table", self._save_results_table, "export")
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

    def _save_results_table(self):
        """Save fit results as a CSV table."""
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results Table", "kinetic_fit_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not save_path:
            return

        r = self.fit_results
        time_units = r.get('time_units', 'minutes')

        # Initial-estimate values (may be absent on legacy sessions). Render
        # missing values as empty cells so the column lines up.
        a0_est = r.get('A0_est')
        a2_est = r.get('A2_est')
        a0_est_cell = a0_est if a0_est is not None else ''
        a2_est_cell = a2_est if a2_est is not None else ''
        pct_nte_est_cell = (
            self.pct_nte_est if not np.isnan(self.pct_nte_est) else ''
        )

        rows = [
            ('A0', 'baseline signal', r['A0'], r['A0_error'], ''),
            ('A0_est', 'baseline signal (initial estimate)', a0_est_cell, '', ''),
            ('A1', 'tracer amplitude', r['A1'], r['A1_error'], ''),
            ('A2', 'non-tracer amplitude', r['A2'], r['A2_error'], ''),
            ('A2_est', 'non-tracer amplitude (initial estimate)', a2_est_cell, '', ''),
            ('kb', 'buildup rate', r['kb'], r['kb_error'], f'1/{time_units}'),
            ('kd', 'decay rate', r['kd'], r['kd_error'], f'1/{time_units}'),
            ('knt', 'non-tracer rate', r['knt'], r['knt_error'], f'1/{time_units}'),
            ('t0', 'tracer onset', r['t0'], r['t0_error'], time_units),
            ('tmax', 'NTE onset', r['tmax'], r['tmax_error'], time_units),
            ('%Enhancement', 'A1/A0 * 100', self.pct_enhancement, '', '%'),
            ('%NTE', 'A2/A0 * 100', self.pct_nte, '', '%'),
            ('%NTE_est', 'A2_est/A0_est * 100', pct_nte_est_cell, '', '%'),
            ('R_squared', 'goodness of fit', r['r_squared'], '', ''),
            ('RMSE', 'root mean square error', r['rmse'], '', ''),
        ]

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'description', 'value', 'error', 'units'])
            for row in rows:
                writer.writerow(row)

        # Companion PNG: reference image (T1 baseline) on the slice the
        # ROI was drawn on, with ROI contour overlaid. Lives next to the
        # CSV so the fit data is traceable back to the source region.
        # Only attempt if the caller plumbed in the ROI context.
        png_msg = ""
        if (self.roi_mask is not None
                and self.reference_image is not None):
            from pathlib import Path
            from ..roi_selection import save_roi_overlay_png
            png_path = Path(save_path).with_suffix('.png')
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

        QMessageBox.information(self, "Save Complete",
                              f"Results table saved to:\n{save_path}{png_msg}")

    def _save_plot(self):
        """Save the plot to file.

        Always prompt the user — even when the GUI workflow seeded a default
        path — so several ROIs in a row can be saved under different names
        instead of overwriting the same file.
        """
        default_path = self.save_path or "kinetic_fit.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", default_path,
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if save_path:
            self.canvas.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Save Complete",
                                  f"Plot saved to:\n{save_path}")


def plot_fit_results_qt(time: np.ndarray, signal: np.ndarray,
                       fitted_signal: np.ndarray, fit_results: Dict,
                       save_path: Optional[str] = None,
                       roi_mask: Optional[np.ndarray] = None,
                       reference_image: Optional[np.ndarray] = None,
                       roi_z_slice: Optional[int] = None) -> None:
    """
    Qt-based fit results visualization.

    Drop-in replacement for plot_fit_results(). The optional roi_mask /
    reference_image / roi_z_slice arguments are forwarded to the dialog
    so its Save Results Table button can drop a companion PNG showing
    the ROI on the anatomical reference next to the saved CSV.
    """
    app = init_qt_app()

    dialog = FitResultsDialog(
        time, signal, fitted_signal, fit_results, save_path,
        roi_mask=roi_mask,
        reference_image=reference_image,
        roi_z_slice=roi_z_slice,
    )

    # Auto-save if path provided
    if save_path:
        dialog.canvas.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    dialog.exec()
