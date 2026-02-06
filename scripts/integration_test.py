#!/usr/bin/env python3
"""
Integration test: exercises the full ProxylFit pipeline with real data.

Loads a real registered session, creates ROI, runs kinetic fitting,
exports CSV, exercises image tools and parameter map export, and
verifies all user-feedback fixes (A1–A3, C1–C3, D1) work end-to-end.

Usage:
    uv run python scripts/integration_test.py
"""

import csv
import json
import shutil
import sys
import tempfile
import time as time_mod
from pathlib import Path

import numpy as np

# Ensure project is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the first available registered session
SESSION_CANDIDATES = [
    project_root / "output" / "35354281",
    project_root / "output" / "35354296",
    project_root / "output" / "35354333",
]

RESULTS_DIR = project_root / "output" / "integration_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestResult:
    """Track pass/fail for each test."""

    def __init__(self):
        self.results = []

    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        self.results.append((name, status, detail))
        icon = "  ✓" if condition else "  ✗"
        print(f"{icon} {name}" + (f"  ({detail})" if detail else ""))
        return condition

    def summary(self):
        passed = sum(1 for _, s, _ in self.results if s == "PASS")
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed")
        if passed < total:
            print("\nFailed tests:")
            for name, status, detail in self.results:
                if status == "FAIL":
                    print(f"  ✗ {name}: {detail}")
        print(f"{'='*60}")
        return passed == total


def find_session():
    """Find a valid registered session directory."""
    for candidate in SESSION_CANDIDATES:
        dicom_dir = candidate / "registered" / "dicoms"
        if dicom_dir.exists() and any(dicom_dir.glob("z*_t*.dcm")):
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Test stages
# ---------------------------------------------------------------------------

def test_load_session(T, session_dir):
    """Stage 1: Load registered 4D data from a real session."""
    print("\n--- Stage 1: Load Session ---")

    from proxyl_analysis.registration import load_registration_data

    registered_4d, spacing, metrics = load_registration_data(session_dir)

    T.check("4D data loaded", registered_4d is not None)
    T.check("4D shape has 4 dims", registered_4d.ndim == 4,
            f"shape={registered_4d.shape}")
    T.check("Spacing is 3-tuple", len(spacing) == 3,
            f"spacing={spacing}")
    T.check("Data is float", np.issubdtype(registered_4d.dtype, np.floating),
            f"dtype={registered_4d.dtype}")

    nx, ny, nz, nt = registered_4d.shape
    T.check("Has multiple z-slices", nz >= 2, f"nz={nz}")
    T.check("Has multiple timepoints", nt >= 10, f"nt={nt}")

    return registered_4d, spacing, metrics


def test_roi_and_signal(T, registered_4d):
    """Stage 2: Create ROI mask and compute timeseries."""
    print("\n--- Stage 2: ROI & Signal Extraction ---")

    from proxyl_analysis.roi_selection import compute_roi_timeseries

    nx, ny, nz, nt = registered_4d.shape

    # Create a central rectangular ROI mask (2D, applied across all z)
    roi_mask = np.zeros((nx, ny), dtype=bool)
    cx, cy = nx // 2, ny // 2
    roi_mask[cx - 5:cx + 5, cy - 5:cy + 5] = True

    T.check("ROI mask created", np.sum(roi_mask) > 0,
            f"pixels={np.sum(roi_mask)}")

    signal = compute_roi_timeseries(registered_4d, roi_mask)

    T.check("Signal is 1D", signal.ndim == 1)
    T.check("Signal length == nt", len(signal) == nt,
            f"len={len(signal)}, nt={nt}")
    T.check("Signal has variance", np.std(signal) > 0,
            f"std={np.std(signal):.1f}")
    T.check("Signal mean > 0", np.mean(signal) > 0,
            f"mean={np.mean(signal):.1f}")

    return roi_mask, signal


def test_kinetic_fit(T, signal, nt):
    """Stage 3: Run kinetic model fitting."""
    print("\n--- Stage 3: Kinetic Fitting ---")

    from proxyl_analysis.model import fit_proxyl_kinetics
    from proxyl_analysis.run_analysis import create_time_array

    time_array = create_time_array(nt, time_units='minutes')

    T.check("Time array created", len(time_array) == nt,
            f"len={len(time_array)}, range=[{time_array[0]:.1f}, {time_array[-1]:.1f}]")

    kb, kd, knt, fitted_signal, fit_results = fit_proxyl_kinetics(
        time_array, signal, time_units='minutes'
    )

    T.check("kb is finite", np.isfinite(kb), f"kb={kb:.4f}")
    T.check("kd is finite", np.isfinite(kd), f"kd={kd:.4f}")
    T.check("Fitted signal shape", fitted_signal.shape == signal.shape)
    T.check("fit_results has R²", 'r_squared' in fit_results,
            f"R²={fit_results.get('r_squared', '?')}")
    T.check("fit_results has A2", 'A2' in fit_results,
            f"A2={fit_results.get('A2', '?')}")
    T.check("fit_results has tmax", 'tmax' in fit_results,
            f"tmax={fit_results.get('tmax', '?')}")

    return time_array, fitted_signal, fit_results


def test_fit_results_dialog_and_csv_export(T, time_array, signal, fitted_signal,
                                           fit_results, output_dir):
    """Stage 4 (C3): FitResultsDialog CSV export with %Enhancement and %NTE."""
    print("\n--- Stage 4: Fit Results Dialog & CSV Export (C3) ---")

    from unittest.mock import patch
    from proxyl_analysis.ui.fitting import FitResultsDialog
    from proxyl_analysis.ui.styles import init_qt_app

    app = init_qt_app()

    dialog = FitResultsDialog(time_array, signal, fitted_signal, fit_results)

    T.check("Dialog created", dialog is not None)
    T.check("Has pct_enhancement attr", hasattr(dialog, 'pct_enhancement'))
    T.check("Has pct_nte attr", hasattr(dialog, 'pct_nte'))
    T.check("pct_enhancement is finite",
            np.isfinite(dialog.pct_enhancement),
            f"val={dialog.pct_enhancement:.2f}%")
    T.check("pct_nte is finite",
            np.isfinite(dialog.pct_nte),
            f"val={dialog.pct_nte:.2f}%")

    # Export CSV
    csv_path = str(Path(output_dir) / "fit_results.csv")
    with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
               return_value=(csv_path, 'CSV Files (*.csv)')):
        with patch('PySide6.QtWidgets.QMessageBox.information'):
            dialog._save_results_table()

    T.check("CSV file created", Path(csv_path).exists())

    if Path(csv_path).exists():
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = {row[0]: row for row in reader}

        T.check("CSV has correct header",
                header == ['parameter', 'description', 'value', 'error', 'units'])
        T.check("CSV has kb", 'kb' in rows)
        T.check("CSV has A2", 'A2' in rows)
        T.check("CSV has tmax", 'tmax' in rows)
        T.check("CSV has %Enhancement", '%Enhancement' in rows)
        T.check("CSV has %NTE", '%NTE' in rows)
        T.check("CSV has R_squared", 'R_squared' in rows)

        # Verify values match dialog
        if '%Enhancement' in rows:
            csv_enh = float(rows['%Enhancement'][2])
            T.check("CSV %Enhancement matches dialog",
                    abs(csv_enh - dialog.pct_enhancement) < 0.01,
                    f"csv={csv_enh:.2f}, dialog={dialog.pct_enhancement:.2f}")

    dialog.close()
    app.processEvents()


def test_image_tools_dialog(T, registered_4d, time_array, signal, roi_mask, output_dir):
    """Stage 5 (C1, C2): ImageToolsDialog labels and colormap."""
    print("\n--- Stage 5: Image Tools Dialog (C1, C2) ---")

    from proxyl_analysis.ui.image_tools import ImageToolsDialog
    from proxyl_analysis.ui.styles import init_qt_app
    from PySide6.QtWidgets import QApplication

    app = init_qt_app()

    # Create 3D roi mask for image tools (it expects [x, y, z])
    nx, ny, nz, nt = registered_4d.shape
    roi_mask_3d = np.zeros((nx, ny, nz), dtype=bool)
    for z in range(nz):
        roi_mask_3d[:, :, z] = roi_mask

    # Test difference mode
    dialog = ImageToolsDialog(
        image_4d=registered_4d,
        time_array=time_array,
        roi_signal=signal,
        time_units='minutes',
        output_dir=output_dir,
        initial_mode='difference',
        roi_mask=roi_mask_3d,
    )

    dialog.show()
    for _ in range(5):
        app.processEvents()
        time_mod.sleep(0.05)

    # C1: Check simplified labels
    region_a_title = dialog.region_a_group.title()
    T.check("C1: Region A label simplified",
            region_a_title == "Region A",
            f"actual='{region_a_title}'")

    if hasattr(dialog, 'region_b_group'):
        region_b_title = dialog.region_b_group.title()
        T.check("C1: Region B label simplified",
                region_b_title == "Region B",
                f"actual='{region_b_title}'")

    # C1: Check instruction text mentions formula
    if hasattr(dialog, 'instruction_label'):
        instr_text = dialog.instruction_label.text()
        T.check("C1: Instructions mention mean(Region B)",
                "mean(Region B)" in instr_text,
                f"text='{instr_text[:80]}...'")

    # C2: Programmatically set regions and trigger preview to check cmap
    dialog.region_a_start_spin.setValue(0)
    dialog.region_a_end_spin.setValue(3)
    if hasattr(dialog, 'region_b_start_spin'):
        dialog.region_b_start_spin.setValue(5)
        dialog.region_b_end_spin.setValue(8)

    # Capture screenshot for visual inspection
    pixmap = dialog.grab()
    screenshot_path = Path(output_dir) / "integration_image_tools_diff.png"
    pixmap.save(str(screenshot_path))
    T.check("Image tools screenshot saved", screenshot_path.exists())

    dialog.close()
    app.processEvents()

    # Also test average mode
    dialog_avg = ImageToolsDialog(
        image_4d=registered_4d,
        time_array=time_array,
        roi_signal=signal,
        time_units='minutes',
        output_dir=output_dir,
        initial_mode='average',
        roi_mask=roi_mask_3d,
    )
    dialog_avg.show()
    for _ in range(5):
        app.processEvents()
        time_mod.sleep(0.05)

    pixmap_avg = dialog_avg.grab()
    screenshot_avg = Path(output_dir) / "integration_image_tools_avg.png"
    pixmap_avg.save(str(screenshot_avg))
    T.check("Avg mode screenshot saved", screenshot_avg.exists())

    dialog_avg.close()
    app.processEvents()


def test_parameter_map_export(T, registered_4d, spacing, roi_mask, output_dir):
    """Stage 6 (A1, A2): Parameter map results and DICOM export."""
    print("\n--- Stage 6: Parameter Map Export (A1 roi_checkbox, A2 numpy JSON) ---")

    from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog
    from proxyl_analysis.ui.styles import init_qt_app
    from PySide6.QtWidgets import QApplication

    app = init_qt_app()

    nx, ny, nz, _ = registered_4d.shape

    # Build synthetic parameter maps from the real data shape
    np.random.seed(42)
    mask = np.random.random((nx, ny, nz)) > 0.5

    param_maps = {
        'kb_map': np.random.random((nx, ny, nz)) * 0.5 * mask,
        'kd_map': np.random.random((nx, ny, nz)) * 0.1 * mask,
        'knt_map': np.random.random((nx, ny, nz)) * 0.01 * mask,
        'r_squared_map': (np.random.random((nx, ny, nz)) * 0.3 + 0.6) * mask,
        'a1_amplitude_map': np.random.random((nx, ny, nz)) * 3000 * mask,
        'a2_amplitude_map': (np.random.random((nx, ny, nz)) * 500 - 250) * mask,
        'mask': mask,
        'metadata': {
            # A2 fix: these numpy types must serialize to JSON without crashing
            'kernel_type': 'sliding_window',
            'window_x': np.int64(15),
            'window_y': np.int64(15),
            'window_z': np.int64(3),
            'success_rate': np.float64(75.2),
            'processing_time': np.float64(42.5),
            'total_positions': np.int64(nx * ny * nz),
        }
    }

    # Create 3D ROI mask
    roi_mask_3d = np.zeros((nx, ny, nz), dtype=bool)
    for z in range(nz):
        roi_mask_3d[:, :, z] = roi_mask

    export_dir = str(Path(output_dir) / "param_maps")
    Path(export_dir).mkdir(parents=True, exist_ok=True)

    dialog = ParameterMapResultsDialog(
        param_maps=param_maps,
        spacing=spacing,
        roi_mask=roi_mask_3d,
        output_dir=export_dir,
    )

    dialog.show()
    for _ in range(8):
        app.processEvents()
        time_mod.sleep(0.05)

    # A1: The dialog should have created without AttributeError on roi_checkbox
    T.check("A1: ParameterMapResultsDialog created (no roi_checkbox crash)",
            dialog is not None)
    T.check("A1: has roi_checkbox attribute",
            hasattr(dialog, 'roi_checkbox'))

    # Capture screenshot
    pixmap = dialog.grab()
    screenshot_path = Path(output_dir) / "integration_param_map.png"
    pixmap.save(str(screenshot_path))
    T.check("Parameter map screenshot saved", screenshot_path.exists())

    # A2: Test DICOM export with numpy metadata
    from proxyl_analysis.io import save_parameter_map_as_dicom

    dicom_export_dir = str(Path(output_dir) / "dicom_export")
    saved_files = save_parameter_map_as_dicom(
        param_maps['kb_map'], 'kb_map', dicom_export_dir, spacing,
        source_dicom=None, metadata=param_maps['metadata']
    )

    T.check("A2: DICOM export succeeded (numpy int64 metadata)",
            len(saved_files) > 0, f"files={len(saved_files)}")

    # Verify the metadata was written as valid JSON
    if saved_files:
        import pydicom
        ds = pydicom.dcmread(saved_files[0])
        parsed = json.loads(ds.ImageComments)
        T.check("A2: metadata window_x is int",
                isinstance(parsed['window_x'], int),
                f"type={type(parsed['window_x']).__name__}")
        T.check("A2: metadata total_positions serialized",
                'total_positions' in parsed)

    dialog.close()
    app.processEvents()


def test_quit_on_last_window(T):
    """Stage 7 (A3): Closing dialogs should not exit Python."""
    print("\n--- Stage 7: QuitOnLastWindowClosed (A3) ---")

    from proxyl_analysis.ui.styles import init_qt_app
    from PySide6.QtWidgets import QApplication

    app = init_qt_app()

    T.check("A3: quitOnLastWindowClosed is False",
            app.quitOnLastWindowClosed() is False)

    # Create and close a dialog — Python should survive
    from proxyl_analysis.ui.fitting import FitResultsDialog

    time = np.linspace(0, 70, 50)
    signal = np.ones(50) * 5000
    fitted = np.ones(50) * 5000
    fr = {
        'kb': 0.5, 'kd': 0.02, 'knt': 0.005,
        'A0': 5000.0, 'A1': 2000.0, 'A2': -200.0,
        't0': 3.0, 'tmax': 20.0,
        'kb_error': 0.1, 'kd_error': 0.005, 'knt_error': 0.001,
        'A0_error': 50.0, 'A1_error': 100.0, 'A2_error': 50.0,
        't0_error': 0.5, 'tmax_error': 2.0,
        'r_squared': 0.9, 'rmse': 100.0,
        'residuals': np.zeros(50),
        'time_units': 'minutes',
    }
    dialog = FitResultsDialog(time, signal, fitted, fr)
    dialog.show()
    app.processEvents()
    dialog.close()
    app.processEvents()

    # If we reach here, Python didn't exit
    T.check("A3: Python survived dialog close", True)


def test_session_loading_ux(T, session_dir):
    """Stage 8 (D1): Session loading error messages."""
    print("\n--- Stage 8: Session Loading UX (D1) ---")

    from unittest.mock import patch
    from proxyl_analysis.ui.main_menu import MainMenuDialog
    from proxyl_analysis.ui.styles import init_qt_app

    app = init_qt_app()

    # Test 1: Loading a valid session should succeed
    dialog = MainMenuDialog()
    with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
               return_value=session_dir):
        dialog._load_previous_session()

    T.check("D1: Valid session loads successfully",
            dialog.result is not None and dialog.result.get('action') == 'load_previous',
            f"result={dialog.result}")
    dialog.close()
    app.processEvents()

    # Test 2: Loading an empty directory should show warning
    tmp_dir = tempfile.mkdtemp(prefix="proxylfit_integ_")
    try:
        dialog2 = MainMenuDialog()
        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=tmp_dir):
            with patch('PySide6.QtWidgets.QMessageBox.warning') as mock_warn:
                dialog2._load_previous_session()
                T.check("D1: Empty folder triggers warning",
                        mock_warn.called)
        dialog2.close()
        app.processEvents()

        # Test 3: Loading parent of a valid session should hint at child
        child = Path(tmp_dir) / "my_session"
        dicoms = child / "registered" / "dicoms"
        dicoms.mkdir(parents=True)
        (dicoms / "z00_t000.dcm").touch()

        dialog3 = MainMenuDialog()
        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=tmp_dir):
            with patch('PySide6.QtWidgets.QMessageBox.warning') as mock_warn:
                dialog3._load_previous_session()
                if mock_warn.called:
                    msg = mock_warn.call_args[0][2]
                    T.check("D1: Child session hint in message",
                            "my_session" in msg,
                            f"msg='{msg[:100]}...'")
                else:
                    T.check("D1: Child session hint in message", False,
                            "warning not called")
        dialog3.close()
        app.processEvents()
    finally:
        shutil.rmtree(tmp_dir)


def test_parameter_mapping_real_data(T, registered_4d, time_array, roi_mask, output_dir):
    """Stage 9: Run actual parameter mapping on a small ROI of real data."""
    print("\n--- Stage 9: Parameter Mapping on Real Data ---")

    from proxyl_analysis.parameter_mapping import create_parameter_maps

    nx, ny, nz, nt = registered_4d.shape

    # Create 3D ROI mask — use a small area for speed
    roi_mask_3d = np.zeros((nx, ny, nz), dtype=bool)
    cx, cy = nx // 2, ny // 2
    # Small 6x6 area in center slice only
    center_z = nz // 2
    roi_mask_3d[cx-3:cx+3, cy-3:cy+3, center_z] = True

    n_roi_pixels = np.sum(roi_mask_3d)
    T.check("Small ROI for mapping", n_roi_pixels > 0,
            f"pixels={n_roi_pixels}")

    print(f"  Running parameter mapping on {n_roi_pixels} voxels "
          f"(z={center_z}, window=5x5x1)...")

    param_maps = create_parameter_maps(
        registered_4d,
        time_array,
        window_size=(5, 5, 1),
        z_slice=center_z,
        time_units='minutes',
        roi_mask=roi_mask_3d[:, :, center_z],  # 2D mask for single slice
        kernel_type='sliding_window',
    )

    T.check("Parameter maps returned", param_maps is not None)
    T.check("Has kb_map", 'kb_map' in param_maps)
    T.check("Has kd_map", 'kd_map' in param_maps)
    T.check("Has r_squared_map", 'r_squared_map' in param_maps)
    T.check("Has mask", 'mask' in param_maps)
    T.check("Has metadata", 'metadata' in param_maps)

    if 'mask' in param_maps:
        n_successful = np.sum(param_maps['mask'])
        T.check("Some fits succeeded", n_successful > 0,
                f"successful={n_successful}/{n_roi_pixels}")

    if 'kb_map' in param_maps:
        valid_kb = param_maps['kb_map'][param_maps['mask']]
        if len(valid_kb) > 0:
            T.check("kb values are finite",
                    np.all(np.isfinite(valid_kb)),
                    f"range=[{np.min(valid_kb):.4f}, {np.max(valid_kb):.4f}]")

    # Export these maps as DICOM (tests A2 numpy serialization with real metadata)
    from proxyl_analysis.io import save_parameter_map_as_dicom

    export_dir = str(Path(output_dir) / "real_param_dicom")
    saved = save_parameter_map_as_dicom(
        param_maps['kb_map'], 'kb_map', export_dir,
        spacing=(1.0, 1.0, 1.0),
        metadata=param_maps.get('metadata', {})
    )
    T.check("Real param map DICOM export", len(saved) > 0,
            f"files={len(saved)}")

    return param_maps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ProxylFit Integration Test")
    print("=" * 60)

    T = TestResult()

    # Find a valid session
    session_dir = find_session()
    if session_dir is None:
        print("ERROR: No registered session found in output/. Cannot run integration test.")
        print(f"  Looked in: {[str(p) for p in SESSION_CANDIDATES]}")
        sys.exit(1)

    print(f"Using session: {session_dir}")

    # Set up output dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {RESULTS_DIR}")

    # Stage 1: Load
    registered_4d, spacing, metrics = test_load_session(T, session_dir)

    # Stage 2: ROI
    roi_mask, signal = test_roi_and_signal(T, registered_4d)

    # Stage 3: Fit
    nt = registered_4d.shape[3]
    time_array, fitted_signal, fit_results = test_kinetic_fit(T, signal, nt)

    # Stage 4: Fit Results dialog + CSV export (C3)
    test_fit_results_dialog_and_csv_export(
        T, time_array, signal, fitted_signal, fit_results, str(RESULTS_DIR)
    )

    # Stage 5: Image Tools (C1, C2)
    test_image_tools_dialog(
        T, registered_4d, time_array, signal, roi_mask, str(RESULTS_DIR)
    )

    # Stage 6: Parameter Map export (A1, A2)
    test_parameter_map_export(
        T, registered_4d, spacing, roi_mask, str(RESULTS_DIR)
    )

    # Stage 7: QuitOnLastWindowClosed (A3)
    test_quit_on_last_window(T)

    # Stage 8: Session loading UX (D1)
    test_session_loading_ux(T, session_dir)

    # Stage 9: Actual parameter mapping on real data
    test_parameter_mapping_real_data(
        T, registered_4d, time_array, roi_mask, str(RESULTS_DIR)
    )

    # Summary
    all_passed = T.summary()

    print(f"\nOutput files in: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(RESULTS_DIR)}  ({size:,} bytes)")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
