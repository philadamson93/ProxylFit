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
            ("A2 (non-tracer)", f"{self.fit_results['A2']:.3f} ± {self.fit_results['A2_error']:.3f}"),
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

        derived_grid.addWidget(QLabel("%Enhancement (A1/A0):"), 0, 0)
        derived_grid.addWidget(QLabel(f"{self.pct_enhancement:.1f}%"), 0, 1)
        derived_grid.addWidget(QLabel("%NTE (A2/A0):"), 1, 0)
        derived_grid.addWidget(QLabel(f"{self.pct_nte:.1f}%"), 1, 1)

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

        rows = [
            ('A0', 'baseline signal', r['A0'], r['A0_error'], ''),
            ('A1', 'tracer amplitude', r['A1'], r['A1_error'], ''),
            ('A2', 'non-tracer amplitude', r['A2'], r['A2_error'], ''),
            ('kb', 'buildup rate', r['kb'], r['kb_error'], f'1/{time_units}'),
            ('kd', 'decay rate', r['kd'], r['kd_error'], f'1/{time_units}'),
            ('knt', 'non-tracer rate', r['knt'], r['knt_error'], f'1/{time_units}'),
            ('t0', 'tracer onset', r['t0'], r['t0_error'], time_units),
            ('tmax', 'NTE onset', r['tmax'], r['tmax_error'], time_units),
            ('%Enhancement', 'A1/A0 * 100', self.pct_enhancement, '', '%'),
            ('%NTE', 'A2/A0 * 100', self.pct_nte, '', '%'),
            ('R_squared', 'goodness of fit', r['r_squared'], '', ''),
            ('RMSE', 'root mean square error', r['rmse'], '', ''),
        ]

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'description', 'value', 'error', 'units'])
            for row in rows:
                writer.writerow(row)

        QMessageBox.information(self, "Save Complete",
                              f"Results table saved to:\n{save_path}")

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
