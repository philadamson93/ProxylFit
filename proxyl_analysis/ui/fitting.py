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
                 dataset_dir: Optional[str] = None,
                 pre_injection_time: Optional[np.ndarray] = None,
                 pre_injection_signal: Optional[np.ndarray] = None,
                 parent=None):
        super().__init__(parent)
        self.time = time
        self.signal = signal
        self.fitted_signal = fitted_signal
        self.fit_results = fit_results
        self.save_path = save_path
        # Optional pre-injection data — the slice of the time series
        # that was excluded from the fit because it happens before the
        # user's chosen injection time. When provided, the plot shows
        # these points alongside the post-injection data, with the fit
        # curve drawn as A0 over the pre-injection range (real signal
        # there is just baseline noise, not a kinetic prediction). The
        # timecourse CSV save also includes these points so the file
        # captures the full ROI signal, not just the fit window.
        self.pre_injection_time = pre_injection_time
        self.pre_injection_signal = pre_injection_signal
        # Optional ROI context for the Save Results Table companion PNG.
        # When all three are present, saving the CSV also writes a
        # same-N .png showing the reference image with the ROI contour
        # overlaid, so the kinetic data is self-documenting.
        self.roi_mask = roi_mask
        self.reference_image = reference_image
        self.roi_z_slice = roi_z_slice
        # Per-dataset directory ("study number") used as the parent for
        # the kinetic_fits/ subdir where Save Plot / Save Results Table
        # default their auto-incremented filenames. When None, the
        # dialogs fall back to a reasonable cwd-relative default.
        self.dataset_dir = dataset_dir

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
            ("tmax (NTE onset)", f"{self.fit_results['tmax']:.2f}  [fixed = argmax(signal)]"),
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

        # Button bar — one consolidated Save button that writes all four
        # per-ROI files with a shared index N: timecourse_data_N.csv,
        # kinetic_fit_results_N.csv, kinetic_fit_N.png, and
        # kinetic_fit_roi_N.png. Splitting Plot vs Results Table into
        # separate buttons used to allow N to drift between them, which
        # made the bundle hard to reassemble after the fact.
        button_bar = ButtonBar()
        button_bar.add_button("save", "Save", self._save_all, "export")
        button_bar.add_stretch()
        button_bar.add_button("close", "Close", self.accept, "default")
        layout.addWidget(button_bar)

    def _setup_plot(self):
        """Set up the fit results plot.

        Shows all available data points — pre-injection (when provided)
        and post-injection — but draws the fit curve only as physically
        meaningful values:
            * For t < t0 (pre-injection): fit = A0 (constant baseline).
              Real signal there is baseline noise, not a kinetic curve.
            * For t >= t0 (post-injection, from the trimmed window the
              fit was actually run on): fit = the curve_fit output.

        Residuals are shown for both regions: pre-injection residuals
        are simply (signal - A0).
        """
        # Create subplots
        gs = self.canvas.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax1 = self.canvas.fig.add_subplot(gs[0])
        self.ax2 = self.canvas.fig.add_subplot(gs[1])

        a0 = float(self.fit_results.get('A0', 0.0))
        time_units = self.fit_results.get('time_units', 'minutes')

        has_pre = (self.pre_injection_time is not None
                   and self.pre_injection_signal is not None
                   and len(self.pre_injection_time) > 0)

        # ---- Top plot: data + fit ----
        if has_pre:
            # Pre-injection points styled the same as post-injection
            # points (single blue 'Data' series visually) but plotted
            # separately because their fit value is A0, not the curve.
            self.ax1.plot(self.pre_injection_time, self.pre_injection_signal,
                          'bo-', markersize=4, linewidth=2)
            # Pre-injection fit = constant A0. Drawn dashed to flag
            # that this region wasn't fit; the optimizer never saw it.
            self.ax1.plot(self.pre_injection_time,
                          np.full_like(self.pre_injection_time, a0),
                          'r--', linewidth=2, alpha=0.7,
                          label=f'A0 (pre-injection baseline)')
            # Vertical injection-time marker between the two regions —
            # midpoint of (last pre-injection timepoint, first post-
            # injection timepoint) so it sits cleanly between the two
            # sample sets even when the spacing is uneven.
            t_inj = (float(self.pre_injection_time[-1])
                     + float(self.time[0])) / 2.0
            self.ax1.axvline(t_inj, color='gray', linestyle=':',
                             linewidth=1.5, alpha=0.7,
                             label='Injection time')

        self.ax1.plot(self.time, self.signal, 'bo-', markersize=4,
                     linewidth=2, label='Data')
        self.ax1.plot(self.time, self.fitted_signal, 'r-',
                     linewidth=2, label='Fitted Model')

        # Excluded points: overlay red × marks at any post-injection
        # indices the optimizer skipped. fit_results carries the
        # post-injection-space indices as fed into fit_proxyl_kinetics.
        excluded_post = self.fit_results.get('excluded_indices') or []
        excluded_post = [int(i) for i in excluded_post
                         if 0 <= int(i) < len(self.time)]
        if excluded_post:
            self.ax1.scatter(
                np.asarray(self.time)[excluded_post],
                np.asarray(self.signal)[excluded_post],
                marker='x', color='red', s=80, linewidths=2.5,
                zorder=5, label='Excluded from fit',
            )

        self.ax1.set_ylabel('Signal Intensity')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Extended Proxyl Kinetic Model Fit')

        # ---- Bottom plot: residuals ----
        residuals = self.signal - self.fitted_signal
        if has_pre:
            pre_residuals = self.pre_injection_signal - a0
            self.ax2.plot(self.pre_injection_time, pre_residuals,
                          'go-', markersize=3, alpha=0.6)
        self.ax2.plot(self.time, residuals, 'go-', markersize=3)
        self.ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        self.ax2.set_xlabel(f'Time ({time_units})')
        self.ax2.set_ylabel('Residuals')
        self.ax2.set_title('Fit Residuals')
        self.ax2.grid(True, alpha=0.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _save_all(self):
        """Save the full per-ROI bundle with a single shared index N.

        Writes four files into ``<dataset>/kinetic_fits/`` with the next
        free N (auto-incremented across the dataset's history so prior
        ROIs aren't overwritten):

        - ``timecourse_data_<N>.csv``      — time and mean signal arrays
        - ``kinetic_fit_results_<N>.csv``  — fitted parameters + errors
        - ``kinetic_fit_<N>.png``          — fit curve plot snapshot
        - ``kinetic_fit_roi_<N>.png``      — anatomical with ROI contour
                                             (only when ROI context was
                                             provided to the dialog)

        Splitting Plot vs Results Table into separate buttons used to
        let N drift between them — e.g., the plot landed at _3 while
        the table landed at _5 because two new files appeared in the
        directory between clicks. Bundling them under one button with
        one N pick guarantees the four files for a given ROI are
        always findable as a set.
        """
        from pathlib import Path
        from ..io import next_indexed_path
        from ..roi_selection import save_roi_overlay_png

        # Pick a single N for all four files in this bundle. Use the
        # results CSV's slot to determine N — the helper scans for any
        # existing files of that pattern, so we get a value that doesn't
        # collide with prior bundles.
        if self.dataset_dir:
            kinetic_dir = Path(self.dataset_dir) / "kinetic_fits"
        else:
            kinetic_dir = Path.cwd() / "kinetic_fits"
        kinetic_dir.mkdir(parents=True, exist_ok=True)

        default_results_csv = next_indexed_path(
            kinetic_dir, "kinetic_fit_results", ".csv"
        )
        # Extract N from the auto-suggested name.
        import re
        m = re.match(r'kinetic_fit_results_(\d+)\.csv$', default_results_csv.name)
        n_default = int(m.group(1)) if m else 1

        # Let the user confirm/redirect by selecting the results CSV
        # path. Whatever directory and filename they choose drives the
        # other three filenames (we infer N from the CSV name; if they
        # rename it off-pattern, we just take the next free N in the
        # chosen directory for each output).
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save", str(default_results_csv),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not save_path:
            return

        results_csv = Path(save_path)
        target_dir = results_csv.parent

        # Resolve the shared N for the bundle. If the user kept the
        # auto-suggested filename pattern, use that N; otherwise pick
        # max+1 across the four pattern slots in the target dir so we
        # never collide with anything already there.
        m2 = re.match(r'kinetic_fit_results_(\d+)\.csv$', results_csv.name)
        if m2:
            n = int(m2.group(1))
        else:
            # Fall back: take the highest N across all four slots and
            # add 1, so the bundle stays contiguous.
            from ..io import next_indexed_path as _next
            candidates = [
                _next(target_dir, "kinetic_fit_results", ".csv"),
                _next(target_dir, "timecourse_data", ".csv"),
                _next(target_dir, "kinetic_fit", ".png"),
                _next(target_dir, "kinetic_fit_roi", ".png"),
            ]
            n = max(int(re.match(r'.*_(\d+)\.\w+$', p.name).group(1))
                    for p in candidates)

        timecourse_csv = target_dir / f"timecourse_data_{n}.csv"
        plot_png       = target_dir / f"kinetic_fit_{n}.png"
        roi_png        = target_dir / f"kinetic_fit_roi_{n}.png"
        results_png    = target_dir / f"kinetic_fit_results_{n}.png"

        # ---- 1. Results CSV ----
        r = self.fit_results
        time_units = r.get('time_units', 'minutes')

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
            ('tmax', 'NTE onset (fixed = argmax(signal))', r['tmax'], '', time_units),
            ('%Enhancement', 'A1/A0 * 100', self.pct_enhancement, '', '%'),
            ('%NTE', 'A2/A0 * 100', self.pct_nte, '', '%'),
            ('%NTE_est', 'A2_est/A0_est * 100', pct_nte_est_cell, '', '%'),
            ('R_squared', 'goodness of fit', r['r_squared'], '', ''),
            ('RMSE', 'root mean square error', r['rmse'], '', ''),
        ]
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'description', 'value', 'error', 'units'])
            for row in rows:
                writer.writerow(row)

        # ---- 1b. Results table as a presentation-ready PNG ----
        # Same content as results CSV but rendered as a table image so
        # the user can drop it into a slide / report without needing
        # to import the CSV into a spreadsheet first. Floats formatted
        # to a sensible number of significant digits per parameter.
        from ..roi_selection import save_table_as_png

        def _fmt_value(v, fmt='.4f'):
            if v == '' or v is None:
                return ''
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(v)

        png_rows = []
        for (pname, pdesc, pval, perr, punits) in rows:
            fmt = '.4f' if pname in ('kb', 'kd', 'knt', 'R_squared') else '.3f'
            png_rows.append([
                pname,
                pdesc,
                _fmt_value(pval, fmt),
                _fmt_value(perr, fmt) if perr != '' else '',
                punits,
            ])
        try:
            save_table_as_png(
                png_rows,
                ['parameter', 'description', 'value', 'error', 'units'],
                str(results_png),
                title=f"Kinetic fit results (N={n})",
            )
            results_png_msg = f"\n  {results_png.name}"
        except Exception as e:
            results_png_msg = f"\n  (results table PNG failed: {e})"

        # ---- 2. Timecourse CSV (raw signal arrays) ----
        # Includes pre-injection points when the dialog received them,
        # so the saved CSV matches the full time series the user sees
        # in the dialog plot rather than just the trimmed fit window.
        with open(timecourse_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'Time ({time_units})', 'Mean Intensity'])
            if (self.pre_injection_time is not None
                    and self.pre_injection_signal is not None):
                for t, s in zip(self.pre_injection_time,
                                self.pre_injection_signal):
                    writer.writerow([f'{t:.3f}', f'{s:.6f}'])
            for t, s in zip(self.time, self.signal):
                writer.writerow([f'{t:.3f}', f'{s:.6f}'])

        # ---- 3. Fit curve plot ----
        self.canvas.fig.savefig(
            str(plot_png), dpi=300, bbox_inches='tight',
        )

        # ---- 4. ROI overlay companion (when ROI context provided) ----
        roi_msg = ""
        if (self.roi_mask is not None
                and self.reference_image is not None):
            try:
                save_roi_overlay_png(
                    reference_image=self.reference_image,
                    roi_mask=self.roi_mask,
                    z_slice=self.roi_z_slice,
                    output_path=str(roi_png),
                    title=(f"ROI on T1 baseline (z={self.roi_z_slice})"
                           if self.roi_z_slice is not None
                           else "ROI on T1 baseline"),
                )
                roi_msg = f"\n  {roi_png.name}"
            except Exception as e:
                roi_msg = f"\n  (ROI overlay PNG failed: {e})"
        else:
            roi_msg = "\n  (no ROI context — overlay PNG skipped)"

        QMessageBox.information(
            self, "Save Complete",
            f"Saved per-ROI bundle (N={n}) to:\n{target_dir}\n"
            f"  {results_csv.name}"
            f"{results_png_msg}\n"
            f"  {timecourse_csv.name}\n"
            f"  {plot_png.name}"
            f"{roi_msg}",
        )

    # ------------------------------------------------------------------
    # Legacy single-file save methods. Kept callable for tests and any
    # external programmatic users. The UI no longer exposes these — the
    # Save button calls _save_all so the per-ROI bundle is written
    # together with one shared N. Splitting plot vs. results table into
    # separate buttons used to let N drift between them.
    # ------------------------------------------------------------------

    def _save_results_table(self):
        """Single-file: write only the fit-parameters CSV.

        UI no longer wires this; kept for tests and programmatic callers.
        """
        from pathlib import Path
        from ..io import next_indexed_path

        if self.dataset_dir:
            kinetic_dir = Path(self.dataset_dir) / "kinetic_fits"
            default_path = next_indexed_path(
                kinetic_dir, "kinetic_fit_results", ".csv"
            )
        else:
            default_path = Path("kinetic_fit_results.csv")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results Table", str(default_path),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not save_path:
            return

        r = self.fit_results
        time_units = r.get('time_units', 'minutes')
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
            ('tmax', 'NTE onset (fixed = argmax(signal))', r['tmax'], '', time_units),
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
        QMessageBox.information(self, "Save Complete",
                              f"Results table saved to:\n{save_path}")

    def _save_plot(self):
        """Single-file: write only the fit-curve plot PNG.

        UI no longer wires this; kept for tests and programmatic callers.
        """
        from pathlib import Path
        from ..io import next_indexed_path

        if self.dataset_dir:
            kinetic_dir = Path(self.dataset_dir) / "kinetic_fits"
            default_path = next_indexed_path(
                kinetic_dir, "kinetic_fit", ".png"
            )
        else:
            default_path = Path(self.save_path or "kinetic_fit.png")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", str(default_path),
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
                       roi_z_slice: Optional[int] = None,
                       dataset_dir: Optional[str] = None,
                       pre_injection_time: Optional[np.ndarray] = None,
                       pre_injection_signal: Optional[np.ndarray] = None) -> None:
    """
    Qt-based fit results visualization.

    Drop-in replacement for plot_fit_results(). ``time`` / ``signal``
    are the post-injection arrays the fit was run on;
    ``pre_injection_time`` / ``pre_injection_signal`` are the slice
    that was excluded from the fit. When the latter are provided, the
    plot shows the full curve (pre + post) with the fit line drawn at
    A0 over the pre-injection range and the kinetic model over the
    post-injection range.
    """
    app = init_qt_app()

    dialog = FitResultsDialog(
        time, signal, fitted_signal, fit_results, save_path,
        roi_mask=roi_mask,
        reference_image=reference_image,
        roi_z_slice=roi_z_slice,
        dataset_dir=dataset_dir,
        pre_injection_time=pre_injection_time,
        pre_injection_signal=pre_injection_signal,
    )

    # Auto-save if path provided
    if save_path:
        dialog.canvas.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    dialog.exec()
