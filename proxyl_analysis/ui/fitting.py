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
                 param_maps: Optional[Dict] = None,
                 forced_n: Optional[int] = None,
                 param_map_figure: Optional[object] = None,
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
        # Parameter maps + forced N flow in only when this dialog was
        # launched from the parameter-map measurement-ROI button. They
        # let _save_all enrich the composite-summary row with per-voxel
        # _pm mean/std for each map masked to the ROI, and reuse the
        # same N as the matching parameter_map_metric_<N>.csv export
        # so the two outputs share an ROI counter.
        self.param_maps = param_maps
        self.forced_n = forced_n
        # Matplotlib Figure of the parameter-map dialog (when launched
        # from the measurement-ROI button). Snapshotted in _save_all
        # to produce parameter_map_metric_roi_<N>.png so the PM-side
        # ROI overlay shares the bundle's N.
        self.param_map_figure = param_map_figure

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

        # When the dialog was launched from the measurement-ROI button
        # the caller pre-allocated an N to share with the matching
        # parameter_map_metric_<N>.csv export. Use it to drive the
        # default filename so the auto-suggested CSV name lands on
        # that exact N.
        import re
        if self.forced_n is not None:
            default_results_csv = (
                kinetic_dir / f"kinetic_fit_results_{int(self.forced_n)}.csv"
            )
            n_default = int(self.forced_n)
        else:
            default_results_csv = next_indexed_path(
                kinetic_dir, "kinetic_fit_results", ".csv"
            )
            m = re.match(r'kinetic_fit_results_(\d+)\.csv$',
                         default_results_csv.name)
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
            ('tmax', 'NTE onset', r['tmax'], r['tmax_error'], time_units),
            ('%Enhancement', 'A1/A0 * 100', self.pct_enhancement, '', '%'),
            ('%NTE', 'A2/A0 * 100', self.pct_nte, '', '%'),
            ('%NTE_est', 'A2_est/A0_est * 100', pct_nte_est_cell, '', '%'),
            ('R_squared', 'goodness of fit', r['r_squared'], '', ''),
            ('RMSE', 'root mean square error', r['rmse'], '', ''),
        ]
        # Unified column header: 'std' instead of 'error' so this CSV
        # shares its schema with the parameter_map_metric_<N>.csv
        # written from the same ROI. Both render to PNG with the
        # same column layout so a Combined Metrics PNG can stack
        # them seamlessly.
        unified_header = ['parameter', 'description', 'value', 'std', 'units']
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(unified_header)
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
        # Cap the description column at ~22 chars so the one or two
        # really long rows ("non-tracer amplitude (initial estimate)")
        # don't force the column wide enough to leave every other row
        # swimming in blank space. The longer rows simply extend a
        # touch past the cell — still readable, and the rest of the
        # table reads tighter.
        unified_col_caps = [None, 22, None, None, None]
        try:
            save_table_as_png(
                png_rows,
                unified_header,
                str(results_png),
                title=f"Kinetic fit results (N={n})",
                max_col_chars=unified_col_caps,
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

        # ---- 5. Composite per-ROI summary CSV ----
        # One row per saved ROI fit, accumulated across all bundles
        # written to this dataset's kinetic_fits/ folder. Rows keyed
        # by roi_N so re-saving an existing N updates that row in
        # place; new N appends. Lets the operator review every fit
        # this session has produced as a single tabular summary.
        composite_msg = ""
        try:
            composite_path = target_dir / "composite_summary.csv"
            n_pixels = (int(np.sum(self.roi_mask))
                        if self.roi_mask is not None else '')

            def _pct_std(amp, amp_std, base, base_std):
                """σ(amp/base · 100) via error propagation. When base is
                pinned (base_std == 0, e.g. A0 from pre-injection mean)
                this reduces to (amp_std / base) · 100 — the dominant
                term. Returns '' when base is zero."""
                try:
                    base_f = float(base)
                    if base_f == 0:
                        return ''
                    amp_f = float(amp)
                    amp_std_f = float(amp_std) if amp_std not in ('', None) else 0.0
                    base_std_f = float(base_std) if base_std not in ('', None) else 0.0
                    if base_std_f == 0:
                        return abs(amp_std_f) / abs(base_f) * 100.0
                    if amp_f == 0:
                        return abs(base_std_f / base_f) * 100.0
                    rel = ((amp_std_f / amp_f) ** 2
                           + (base_std_f / base_f) ** 2) ** 0.5
                    return abs(amp_f / base_f) * rel * 100.0
                except (TypeError, ValueError, ZeroDivisionError):
                    return ''

            pct_enh_std = _pct_std(
                r['A1'], r['A1_error'], r['A0'], r['A0_error']
            )
            pct_nte_std = _pct_std(
                r['A2'], r['A2_error'], r['A0'], r['A0_error']
            )

            new_row = {
                'roi_N': n,
                'z_slice': (self.roi_z_slice if self.roi_z_slice is not None else ''),
                'n_pixels': n_pixels,
                'A0': r['A0'],
                'A1': r['A1'],
                'A1_std': r['A1_error'],
                'A2': r['A2'],
                'A2_std': r['A2_error'],
                'kb': r['kb'],
                'kb_std': r['kb_error'],
                'kd': r['kd'],
                'kd_std': r['kd_error'],
                'knt': r['knt'],
                'knt_std': r['knt_error'],
                't0': r['t0'],
                't0_std': r['t0_error'],
                'tmax': r['tmax'],
                'tmax_std': r['tmax_error'],
                'pct_enhancement': self.pct_enhancement,
                'pct_enh_std': pct_enh_std,
                'pct_nte': self.pct_nte,
                'pct_nte_std': pct_nte_std,
                'pct_nte_est': (self.pct_nte_est
                                if not np.isnan(self.pct_nte_est) else ''),
                'R_squared': r['r_squared'],
            }

            # Per-voxel parameter-map statistics, computed by masking
            # each map with the measurement ROI at the relevant z-slice.
            # Suffix '_pm' distinguishes these from the ROI-mean curve
            # fit values above (e.g. kb_pm vs kb). Only populated when
            # this dialog was launched from the parameter-map measurement
            # ROI button — regular main-menu kinetic fits leave the
            # columns blank.
            pm_columns = [
                ('A0_pm', 'A0_pm_std', 'baseline_map'),
                ('A1_pm', 'A1_pm_std', 'a1_amplitude_map'),
                ('A2_pm', 'A2_pm_std', 'a2_amplitude_map'),
                ('kb_pm', 'kb_pm_std', 'kb_map'),
                ('kd_pm', 'kd_pm_std', 'kd_map'),
                ('knt_pm', 'knt_pm_std', 'knt_map'),
                ('t0_pm', 't0_pm_std', 't0_map'),
                ('tmax_pm', 'tmax_pm_std', 'tmax_map'),
                ('pct_enhancement_pm', 'pct_enh_pm_std', 'a1_percent_map'),
                ('pct_nte_pm', 'pct_nte_pm_std', 'a2_percent_map'),
                ('pct_nte_est_pm', 'pct_nte_est_pm_std', 'a2_percent_est_map'),
                ('R_squared_pm', None, 'r_squared_map'),
            ]

            def _pm_stats(map_key):
                """Return (mean, std) of a parameter map's voxels inside
                the ROI at roi_z_slice. Only counts voxels the param-map
                fit successfully converged on (mask=True). Returns
                ('', '') when context is missing or no voxel converges.
                """
                if not self.param_maps or self.roi_mask is None:
                    return '', ''
                map3d = self.param_maps.get(map_key)
                fit_mask = self.param_maps.get('mask')
                if map3d is None:
                    return '', ''
                z = self.roi_z_slice
                if z is None or z >= map3d.shape[2]:
                    return '', ''
                slice_data = map3d[:, :, z]
                slice_mask = (fit_mask[:, :, z]
                              if fit_mask is not None
                              else np.ones_like(slice_data, dtype=bool))
                if self.roi_mask.ndim == 2:
                    roi_slice = self.roi_mask
                elif z < self.roi_mask.shape[2]:
                    roi_slice = self.roi_mask[:, :, z]
                else:
                    return '', ''
                combined = slice_mask & roi_slice
                if not np.any(combined):
                    return '', ''
                values = slice_data[combined]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    return '', ''
                return float(np.mean(values)), float(np.std(values))

            for mean_col, std_col, map_key in pm_columns:
                mean_val, std_val = _pm_stats(map_key)
                new_row[mean_col] = mean_val
                if std_col is not None:
                    new_row[std_col] = std_val

            header = list(new_row.keys())

            # Read any existing rows so we can update-by-roi_N rather
            # than blindly appending duplicates.
            existing_rows = {}
            if composite_path.exists():
                try:
                    with open(composite_path, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for prev in reader:
                            try:
                                key = int(prev.get('roi_N', ''))
                            except (TypeError, ValueError):
                                continue
                            existing_rows[key] = prev
                except Exception:
                    # If the composite is malformed, drop it and start
                    # fresh — better than erroring out a save.
                    existing_rows = {}

            existing_rows[int(n)] = {k: new_row[k] for k in header}

            # Format floats consistently when writing back out so the
            # CSV stays human-readable. Unknown types pass through.
            def _fmt(v):
                if v == '' or v is None:
                    return ''
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return v
                if fv == int(fv) and abs(fv) < 1e9 and 'std' not in 'forced':
                    pass  # let numerical formatting decide below
                # Use a generous fixed format; the consumer can re-format.
                return f"{fv:.6g}"

            with open(composite_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for key in sorted(existing_rows.keys()):
                    row = existing_rows[key]
                    writer.writerow([_fmt(row.get(col, '')) for col in header])

            composite_msg = (
                f"\n  composite_summary.csv "
                f"({len(existing_rows)} ROI{'s' if len(existing_rows) != 1 else ''})"
            )
        except Exception as e:
            composite_msg = f"\n  (composite summary update failed: {e})"

        # ---- 6. Parameter-map-metric companion bundle ----
        # Only when this dialog was launched from the parameter-map
        # measurement-ROI button (param_maps + param_map_figure are
        # set). Writes the PM-side artefacts into
        # <dataset>/parameter_maps/parameter_map_metrics/ with the same
        # shared ROI counter N as the kinetic bundle:
        #
        #   parameter_map_metric_<N>.csv          unified-format table
        #   parameter_map_metric_<N>.png          rendered table image
        #   parameter_map_metric_roi_<N>.png      param-map figure snapshot
        #   summary_strip_<N>.png                 2-row layout: top = 3
        #                                         images; bottom = 2 tables
        pm_msg = ""
        if (self.param_maps is not None
                and self.dataset_dir is not None):
            try:
                pm_msg = self._save_pm_metric_bundle(
                    n=n,
                    kinetic_results_png=results_png,
                    kinetic_plot_png=plot_png,
                    kinetic_roi_png=roi_png,
                )
            except Exception as e:
                pm_msg = f"\n  (PM metric bundle failed: {e})"

        QMessageBox.information(
            self, "Save Complete",
            f"Saved per-ROI bundle (N={n}) to:\n{target_dir}\n"
            f"  {results_csv.name}"
            f"{results_png_msg}\n"
            f"  {timecourse_csv.name}\n"
            f"  {plot_png.name}"
            f"{roi_msg}"
            f"{composite_msg}"
            f"{pm_msg}",
        )

    def _save_pm_metric_bundle(self, n, kinetic_results_png,
                               kinetic_plot_png, kinetic_roi_png):
        """Write the parameter-map-metric companion artefacts.

        Driven from _save_all when this dialog was launched from the
        parameter-map measurement-ROI button. Produces, in
        ``<dataset>/parameter_maps/parameter_map_metrics/`` with the
        same N as the kinetic bundle:

        - ``parameter_map_metric_<N>.csv``      unified columns
        - ``parameter_map_metric_<N>.png``      rendered table
        - ``parameter_map_metric_roi_<N>.png``  PM dialog figure snapshot
        - ``summary_strip_<N>.png``             2-row composite:
              top = 3 images (kinetic_fit_roi, parameter_map_metric_roi,
              kinetic_fit); bottom = 2 tables (kinetic_fit_results,
              parameter_map_metric).
        """
        from pathlib import Path
        from ..roi_selection import save_table_as_png
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        metrics_dir = (Path(self.dataset_dir)
                       / "parameter_maps" / "parameter_map_metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        pm_csv  = metrics_dir / f"parameter_map_metric_{n}.csv"
        pm_png  = metrics_dir / f"parameter_map_metric_{n}.png"
        pm_roi_png  = metrics_dir / f"parameter_map_metric_roi_{n}.png"
        strip_png    = metrics_dir / f"summary_strip_{n}.png"

        # Build PM-metric rows. Single source of truth — these specs
        # mirror ParameterMapResultsDialog._build_pm_metric_specs so
        # the two paths render identical content.
        time_units = (self.param_maps.get('metadata', {}) or {}).get(
            'time_units', 'minutes',
        )
        specs = [
            ('baseline_map',        'A0_pm',
             'baseline signal (mean)', ''),
            ('a1_amplitude_map',    'A1_pm',
             'tracer amplitude (mean)', ''),
            ('a2_amplitude_map',    'A2_pm',
             'non-tracer amplitude (mean)', ''),
            ('kb_map',              'kb_pm',
             'buildup rate (mean)', f'1/{time_units}'),
            ('kd_map',              'kd_pm',
             'decay rate (mean)', f'1/{time_units}'),
            ('knt_map',             'knt_pm',
             'non-tracer rate (mean)', f'1/{time_units}'),
            ('t0_map',              't0_pm',
             'tracer onset (mean)', time_units),
            ('tmax_map',            'tmax_pm',
             'NTE onset (mean)', time_units),
            ('a1_percent_map',      'pct_enhancement_pm',
             '%Enhancement (mean)', '%'),
            ('a2_percent_map',      'pct_nte_pm',
             '%NTE (mean)', '%'),
            ('a2_percent_est_map',  'pct_nte_est_pm',
             '%NTE_est (mean)', '%'),
            ('r_squared_map',       'R_squared_pm',
             'goodness of fit (mean)', ''),
        ]

        z = self.roi_z_slice
        if z is None:
            z = 0
        fit_mask = self.param_maps.get('mask')
        if self.roi_mask is None:
            return "\n  (PM metric bundle skipped: no ROI mask)"

        if self.roi_mask.ndim == 2:
            roi_slice = self.roi_mask
        elif z < self.roi_mask.shape[2]:
            roi_slice = self.roi_mask[:, :, z]
        else:
            return "\n  (PM metric bundle skipped: ROI z out of range)"
        if fit_mask is not None and fit_mask.ndim == 3 and z < fit_mask.shape[2]:
            slice_mask = roi_slice & fit_mask[:, :, z]
        else:
            slice_mask = roi_slice
        n_pixels = int(np.sum(slice_mask))

        unified_rows = []
        for key, name_pm, desc, units in specs:
            map3d = self.param_maps.get(key)
            if map3d is None or map3d.ndim != 3 or z >= map3d.shape[2]:
                continue
            values = map3d[:, :, z][slice_mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            unified_rows.append((
                name_pm, desc,
                float(np.nanmean(values)),
                float(np.nanstd(values)),
                units,
            ))

        # 6a) PM CSV (unified header)
        unified_header = ['parameter', 'description', 'value', 'std', 'units']
        with open(pm_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(unified_header)
            for row in unified_rows:
                writer.writerow(row)

        # Format helper for the table renderer.
        def _fmt(v, fmt='.4f'):
            if v == '' or v is None:
                return ''
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(v)

        pm_png_rows = [[
            r[0], r[1], _fmt(r[2]), _fmt(r[3]), r[4],
        ] for r in unified_rows]

        title_suffix = f"ROI #{n}"
        title_suffix += f", z={z}, n={n_pixels} pixels"

        # 6b) PM table PNG. Cap description column the same way the
        # kinetic-fit table does so the two render with comparable
        # proportions when laid out side-by-side in the strip.
        save_table_as_png(
            pm_png_rows, unified_header, str(pm_png),
            title=f"Parameter map metrics ({title_suffix})",
            max_col_chars=[None, 22, None, None, None],
        )

        # 6c) PM figure snapshot (param-map dialog)
        if self.param_map_figure is not None:
            try:
                self.param_map_figure.savefig(
                    str(pm_roi_png), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                )
            except Exception:
                pm_roi_png = None
        else:
            pm_roi_png = None

        # 6d) Summary strip — 2-row layout for slide readability:
        #
        #   Top row   :  kinetic_fit_roi  |  parameter_map_metric_roi  |  kinetic_fit
        #                (3 panels, each spanning 2 of 6 grid columns)
        #   Bottom row:        kinetic_fit_results table  |  parameter_map_metric table
        #                (2 panels, each spanning 3 of 6 grid columns)
        #
        # Combined-metrics PNG is gone — the two tables on the bottom
        # row deliver the same content with the per-table widths the
        # user actually wants.
        from matplotlib.gridspec import GridSpec

        top_panels = [
            ('Kinetic fit ROI', kinetic_roi_png),
            ('Parameter map ROI', pm_roi_png),
            ('Kinetic fit', kinetic_plot_png),
        ]
        bottom_panels = [
            ('Kinetic fit results', kinetic_results_png),
            ('Parameter map metrics', pm_png),
        ]

        # Filter out missing sources (best-effort — a missing PM ROI
        # snapshot, for instance, just leaves an empty placeholder
        # rather than blowing up the whole strip).
        def _present(p):
            return p is not None and Path(p).exists()

        any_present = (
            any(_present(p) for _, p in top_panels)
            or any(_present(p) for _, p in bottom_panels)
        )

        if any_present:
            fig = plt.figure(figsize=(18.0, 11.0), dpi=120)
            gs = GridSpec(
                nrows=2, ncols=6, figure=fig,
                hspace=0.18, wspace=0.08,
                left=0.02, right=0.98, top=0.93, bottom=0.03,
                height_ratios=[1.0, 1.1],
            )

            # Top row: 3 image panels, 2 grid cols each.
            for i, (label, src) in enumerate(top_panels):
                ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
                if _present(src):
                    try:
                        img = mpimg.imread(str(src))
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, '(load failed)',
                                transform=ax.transAxes, ha='center')
                else:
                    ax.text(0.5, 0.5, '(not generated)',
                            transform=ax.transAxes, ha='center',
                            color='#888')
                ax.set_title(label, fontsize=11)
                ax.axis('off')

            # Bottom row: 2 table panels, 3 grid cols each.
            for i, (label, src) in enumerate(bottom_panels):
                ax = fig.add_subplot(gs[1, i*3:(i+1)*3])
                if _present(src):
                    try:
                        img = mpimg.imread(str(src))
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, '(load failed)',
                                transform=ax.transAxes, ha='center')
                else:
                    ax.text(0.5, 0.5, '(not generated)',
                            transform=ax.transAxes, ha='center',
                            color='#888')
                ax.set_title(label, fontsize=11)
                ax.axis('off')

            # Prefix with the dataset folder name (e.g. study number
            # "35352258") so the strip is self-identifying when
            # dropped into a slide deck without the surrounding path.
            dataset_label = ''
            if self.dataset_dir:
                dataset_label = Path(self.dataset_dir).name
            strip_title = f"Per-ROI summary ({title_suffix})"
            if dataset_label:
                strip_title = f"{dataset_label} — {strip_title}"
            fig.suptitle(strip_title, fontsize=13, y=0.98)
            fig.savefig(
                str(strip_png), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none',
            )
            plt.close(fig)

        msg_parts = [
            f"\n  parameter_maps/parameter_map_metrics/",
            f"\n    {pm_csv.name}",
            f"\n    {pm_png.name}",
        ]
        if pm_roi_png is not None:
            msg_parts.append(f"\n    {Path(pm_roi_png).name}")
        if any_present:
            msg_parts.append(f"\n    {strip_png.name}")
        return ''.join(msg_parts)

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
            ('tmax', 'NTE onset', r['tmax'], r['tmax_error'], time_units),
            ('%Enhancement', 'A1/A0 * 100', self.pct_enhancement, '', '%'),
            ('%NTE', 'A2/A0 * 100', self.pct_nte, '', '%'),
            ('%NTE_est', 'A2_est/A0_est * 100', pct_nte_est_cell, '', '%'),
            ('R_squared', 'goodness of fit', r['r_squared'], '', ''),
            ('RMSE', 'root mean square error', r['rmse'], '', ''),
        ]
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'description', 'value', 'std', 'units'])
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
                       pre_injection_signal: Optional[np.ndarray] = None,
                       param_maps: Optional[Dict] = None,
                       forced_n: Optional[int] = None,
                       param_map_figure: Optional[object] = None) -> None:
    """
    Qt-based fit results visualization.

    Drop-in replacement for plot_fit_results(). ``time`` / ``signal``
    are the post-injection arrays the fit was run on;
    ``pre_injection_time`` / ``pre_injection_signal`` are the slice
    that was excluded from the fit. When the latter are provided, the
    plot shows the full curve (pre + post) with the fit line drawn at
    A0 over the pre-injection range and the kinetic model over the
    post-injection range.

    ``param_maps`` and ``forced_n`` flow in only when launched from
    the parameter-map dialog's measurement-ROI button — see
    FitResultsDialog.__init__ for their effect on _save_all.
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
        param_maps=param_maps,
        forced_n=forced_n,
        param_map_figure=param_map_figure,
    )

    # Auto-save if path provided
    if save_path:
        dialog.canvas.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    dialog.exec()
