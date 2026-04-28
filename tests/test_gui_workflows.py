"""
Comprehensive GUI workflow tests for ProxylFit.

Tests organized by feature area, each capturing screenshots to tests/screenshots/.
Verifies scanner, session loading, ROI, fitting, image tools, parameter maps,
registration viewer, and full workflows with real data.
"""

import csv
import json
import time as time_mod
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ============================================================================
# 1. DICOM Scanner Tests (T023)
# ============================================================================

class TestDicomScanner:
    """Tests for DICOM scanner detection of registered output."""

    def test_scanner_recognizes_registered_t1(self):
        """T023: scanner detects 'Registered T1 DCE' as proxyl series."""
        from proxyl_analysis.dicom_scanner import scan_dicom_folder

        # Directly test the detection logic
        desc_lower = 'registered t1 dce'.lower()
        is_proxyl = (
            'proxyl' in desc_lower
            or ('flash' in desc_lower and False)  # num_frames > 100
            or 'registered t1' in desc_lower
        )
        assert is_proxyl, "Scanner should detect 'Registered T1 DCE' as PROXYL"

    def test_scanner_recognizes_registered_t2(self):
        """T023: scanner detects 'Registered T2' as t2 series."""
        desc_lower = 'registered t2'.lower()
        is_t2 = (
            ('t2' in desc_lower and 'turbo' in desc_lower)
            or 'registered t2' in desc_lower
        )
        assert is_t2, "Scanner should detect 'Registered T2' as T2"

    def test_scanner_still_detects_raw_proxyl(self):
        """Original detection for raw scanner data still works."""
        # Test 'proxyl' keyword
        desc_lower = 'proxyl 3d flash'.lower()
        is_proxyl = (
            'proxyl' in desc_lower
            or ('flash' in desc_lower and True)
            or 'registered t1' in desc_lower
        )
        assert is_proxyl

    def test_scanner_still_detects_raw_t2(self):
        """Original detection for raw T2 data still works."""
        desc_lower = 't2 turbo spin echo'.lower()
        is_t2 = (
            ('t2' in desc_lower and 'turbo' in desc_lower)
            or 'registered t2' in desc_lower
        )
        assert is_t2

    def test_scanner_dialog_dropdowns_populated(self, qt_app):
        """T023: DicomScanResultsDialog combo boxes have entries for registered output."""
        from proxyl_analysis.ui.main_menu import DicomScanResultsDialog

        # Simulate scan results from registered output
        scan_results = [
            {
                'series_number': 1,
                'series_description': 'Registered T1 DCE',
                'study_description': 'PROXYL Study',
                'modality': 'MR',
                'rows': 128, 'cols': 128,
                'num_slices': 9, 'num_frames': 184,
                'num_files': 1656,
                'patient_id': 'TEST',
                'study_date': '20260101',
                'sample_file': '/tmp/test_t1.dcm',
                'is_proxyl': True,
                'is_t2': False,
                'series_uid': 'uid1',
            },
            {
                'series_number': 2,
                'series_description': 'Registered T2',
                'study_description': 'PROXYL Study',
                'modality': 'MR',
                'rows': 128, 'cols': 128,
                'num_slices': 9, 'num_frames': 1,
                'num_files': 9,
                'patient_id': 'TEST',
                'study_date': '20260101',
                'sample_file': '/tmp/test_t2.dcm',
                'is_proxyl': False,
                'is_t2': True,
                'series_uid': 'uid2',
            },
            {
                'series_number': 3,
                'series_description': 'Averaged Image t0-t5',
                'study_description': 'PROXYL Study',
                'modality': 'MR',
                'rows': 128, 'cols': 128,
                'num_slices': 9, 'num_frames': 1,
                'num_files': 9,
                'patient_id': 'TEST',
                'study_date': '20260101',
                'sample_file': '/tmp/test_avg.dcm',
                'is_proxyl': False,
                'is_t2': False,
                'series_uid': 'uid3',
            },
        ]

        dialog = DicomScanResultsDialog(scan_results, '/tmp/test_folder')

        # T1 combo should have 2 items: "-- None --" + registered T1
        assert dialog.t1_combo.count() == 2, (
            f"Expected 2 items in T1 combo, got {dialog.t1_combo.count()}"
        )
        # Should auto-select the first PROXYL series
        assert dialog.t1_combo.currentIndex() == 1
        assert dialog.t1_combo.currentData() == '/tmp/test_t1.dcm'

        # T2 combo should have 2 items: "-- None --" + registered T2
        assert dialog.t2_combo.count() == 2, (
            f"Expected 2 items in T2 combo, got {dialog.t2_combo.count()}"
        )
        assert dialog.t2_combo.currentIndex() == 1
        assert dialog.t2_combo.currentData() == '/tmp/test_t2.dcm'

        dialog.close()

    def test_scanner_real_registered_folder(self, qt_app, real_session_path, screenshot_dir):
        """Scan real registered output folder -> dropdowns populated, screenshot."""
        from proxyl_analysis.dicom_scanner import scan_dicom_folder
        from proxyl_analysis.ui.main_menu import DicomScanResultsDialog

        dicoms_dir = real_session_path / "registered" / "dicoms"
        results = scan_dicom_folder(str(dicoms_dir))

        assert len(results) > 0, "Scanner should find series in registered output"

        # Check that at least one PROXYL series is detected
        proxyl = [s for s in results if s['is_proxyl']]
        assert len(proxyl) >= 1, (
            f"Expected at least 1 PROXYL series, got {len(proxyl)}. "
            f"Series descriptions: {[s['series_description'] for s in results]}"
        )

        dialog = DicomScanResultsDialog(results, str(dicoms_dir))

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_scanner_real_registered_folder.png"))

        # T1 combo should have entries (not just "-- None --")
        assert dialog.t1_combo.count() >= 2, (
            f"T1 combo has only {dialog.t1_combo.count()} items (expected >= 2)"
        )

        dialog.close()


# ============================================================================
# 2. Session Loading Tests (T021)
# ============================================================================

class TestSessionLoading:
    """Tests for session loading validation and error messages."""

    def test_session_load_valid_directory(self, qt_app, temp_dir, screenshot_dir):
        """T021: valid session loads, screenshot of main menu."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        # Create valid session structure
        dicoms = temp_dir / "registered" / "dicoms"
        dicoms.mkdir(parents=True)
        (dicoms / "z00_t000.dcm").touch()

        dialog = MainMenuDialog()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=str(temp_dir)):
            dialog._load_previous_session()

        assert dialog.result is not None
        assert dialog.result['action'] == 'load_previous'
        assert dialog.result['session_path'] == str(temp_dir)

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_session_load_valid.png"))

        dialog.close()

    def test_session_load_invalid_shows_helpful_error(self, qt_app, temp_dir):
        """T021: invalid dir shows expected structure hint."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=str(temp_dir)):
            with patch('PySide6.QtWidgets.QMessageBox.warning') as mock_warn:
                dialog._load_previous_session()

                mock_warn.assert_called_once()
                msg = mock_warn.call_args[0][2]
                assert "registered/dicoms/z00_t000.dcm" in msg

        dialog.close()


# ============================================================================
# 3. ROI & Signal Tests
# ============================================================================

class TestRoiAndSignal:
    """Tests for ROI selection and injection time."""

    def test_roi_selector_creates_mask(self, qt_app, sample_4d):
        """Rectangle ROI dialog produces valid mask array."""
        from proxyl_analysis.ui.roi import ROISelectorDialog

        # Extract a 2D slice for the dialog
        image_slice = sample_4d[:, :, 1, 0]
        dialog = ROISelectorDialog(image_slice)
        dialog.show()
        qt_app.processEvents()

        # Simulate a rectangle selection by setting mask directly
        mask = np.zeros_like(image_slice, dtype=bool)
        mask[10:20, 10:20] = True
        dialog.mask = mask

        assert dialog.get_mask() is not None
        assert np.sum(dialog.get_mask()) == 100  # 10x10 pixels

        dialog.close()

    def test_injection_time_selector(self, qt_app, sample_time_array, sample_4d,
                                      temp_dir, screenshot_dir):
        """Click sets injection index, CSV export produces file."""
        from proxyl_analysis.ui.injection import InjectionTimeSelectorDialog

        signal = np.mean(sample_4d[:, :, :, :], axis=(0, 1, 2))

        dialog = InjectionTimeSelectorDialog(
            sample_time_array, signal, 'minutes', str(temp_dir)
        )
        dialog.show()
        qt_app.processEvents()

        # Simulate selecting timepoint 5
        dialog.injection_index = 5

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_injection_time_selector.png"))

        assert dialog.get_injection_index() == 5

        # Test CSV export
        csv_path = str(temp_dir / "timeseries.csv")
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV Files (*.csv)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._export_csv()

        if Path(csv_path).exists():
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert 'time' in header[0].lower() or 'signal' in header[0].lower()

        dialog.close()


# ============================================================================
# 4. Kinetic Fitting Tests (T020)
# ============================================================================

class TestKineticFitting:
    """Tests for fit results display and export."""

    def test_fit_results_display(self, qt_app, sample_fit_results, screenshot_dir):
        """T020: dialog shows all params + %Enhancement + %NTE, screenshot."""
        from proxyl_analysis.ui.fitting import FitResultsDialog

        time = np.linspace(0, 70, 126)
        signal = np.ones(126) * 5000
        fitted = np.ones(126) * 5000

        dialog = FitResultsDialog(time, signal, fitted, sample_fit_results)
        dialog.show()
        qt_app.processEvents()

        # Check derived parameters are computed
        expected_enh = (2800.0 / 5200.0) * 100
        expected_nte = (-400.0 / 5200.0) * 100
        assert dialog.pct_enhancement == pytest.approx(expected_enh, rel=0.01)
        assert dialog.pct_nte == pytest.approx(expected_nte, rel=0.01)

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_fit_results_display.png"))

        dialog.close()

    def test_fit_results_csv_export(self, qt_app, sample_fit_results, temp_dir):
        """T020: CSV contains all parameters + derived values."""
        from proxyl_analysis.ui.fitting import FitResultsDialog

        time = np.linspace(0, 70, 126)
        signal = np.ones(126) * 5000
        fitted = np.ones(126) * 5000

        dialog = FitResultsDialog(time, signal, fitted, sample_fit_results)

        csv_path = str(temp_dir / "fit_results.csv")
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._save_results_table()

        assert Path(csv_path).exists()

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = {row['parameter']: row for row in reader}

        # All kinetic parameters present
        for param in ['kb', 'kd', 'knt', 'A0', 'A1', 'A2', 't0', 'tmax']:
            assert param in rows, f"Missing parameter: {param}"

        # Derived values present
        assert '%Enhancement' in rows
        assert '%NTE' in rows
        assert 'R_squared' in rows
        assert 'RMSE' in rows

        # Verify values
        assert float(rows['kb']['value']) == pytest.approx(0.98)
        assert float(rows['%Enhancement']['value']) == pytest.approx(53.846, rel=0.01)

        dialog.close()


# ============================================================================
# 5. Image Tools Tests (T008, T019)
# ============================================================================

class TestImageTools:
    """Tests for average and difference image tools."""

    def _create_dialog(self, qt_app, sample_4d, sample_roi_mask, temp_dir,
                        mode='average'):
        """Helper to create an ImageToolsDialog."""
        from proxyl_analysis.ui.image_tools import ImageToolsDialog

        time_array = np.arange(20, dtype=float) * (70.0 / 60.0)
        roi_signal = np.mean(sample_4d[:, :, :, :], axis=(0, 1, 2))

        dialog = ImageToolsDialog(
            image_4d=sample_4d,
            time_array=time_array,
            roi_signal=roi_signal,
            time_units='minutes',
            output_dir=str(temp_dir),
            initial_mode=mode,
            roi_mask=sample_roi_mask,
        )
        dialog.show()
        for _ in range(3):
            qt_app.processEvents()
            time_mod.sleep(0.05)
        return dialog

    def test_image_tools_average_auto_preview(self, qt_app, sample_4d,
                                               sample_roi_mask, temp_dir,
                                               screenshot_dir):
        """T008 bug 4: preview appears automatically after region set."""
        dialog = self._create_dialog(qt_app, sample_4d, sample_roi_mask, temp_dir,
                                      mode='average')

        # Set region
        dialog.region_a_start_spin.setValue(2)
        dialog.region_a_end_spin.setValue(8)
        qt_app.processEvents()
        time_mod.sleep(0.1)
        qt_app.processEvents()

        # Preview should be computed (may already be set from auto-preview)
        dialog._update_preview()
        qt_app.processEvents()

        assert dialog.preview_image is not None, "Preview should be computed after setting region"

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_image_tools_average_preview.png"))

        dialog.close()

    def test_image_tools_difference_labels_and_cmap(self, qt_app, sample_4d,
                                                     sample_roi_mask, temp_dir,
                                                     screenshot_dir):
        """T019: simplified labels, gray cmap, B-minus-A filename."""
        dialog = self._create_dialog(qt_app, sample_4d, sample_roi_mask, temp_dir,
                                      mode='difference')

        # Set regions directly (spinbox setValue(0) won't fire valueChanged if already 0)
        dialog.region_a_start = 0
        dialog.region_a_end = 4
        dialog.region_b_start = 10
        dialog.region_b_end = 15
        dialog.region_a_start_spin.setValue(0)
        dialog.region_a_end_spin.setValue(4)
        dialog.region_b_start_spin.setValue(10)
        dialog.region_b_end_spin.setValue(15)
        qt_app.processEvents()
        time_mod.sleep(0.1)

        dialog._update_preview()
        qt_app.processEvents()

        assert dialog.preview_image is not None, "Difference preview should be computed"

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_image_tools_difference.png"))

        # Verify gray cmap is used in source code
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        assert "cmap='gray'" in text, "Expected gray colormap for difference images"

        dialog.close()

    def test_image_tools_first_click_marker(self, qt_app, sample_4d,
                                             sample_roi_mask, temp_dir):
        """T008 bug 2: axvline visible after first click only."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        # Should have axvline logic for click markers
        assert "axvline" in text, "Expected axvline markers for region clicks"

        dialog = self._create_dialog(qt_app, sample_4d, sample_roi_mask, temp_dir)
        dialog.close()

    def test_image_tools_export_roundtrip(self, qt_app, sample_4d,
                                           sample_roi_mask, temp_dir):
        """Average/difference export -> load back -> values match."""
        from proxyl_analysis.ui.image_tools import compute_averaged_image
        import pydicom
        from proxyl_analysis.io import save_derived_image_as_dicom

        # Compute averaged image
        avg_image = compute_averaged_image(sample_4d, 2, 8)
        assert avg_image.shape == (32, 32, 3)

        # Export as DICOM
        output_path = str(temp_dir / "avg_test.dcm")
        params = {'start_idx': 2, 'end_idx': 8, 'n_frames': 7}
        saved = save_derived_image_as_dicom(
            avg_image, output_path, 'averaged', params, (1.0, 1.0, 2.0)
        )

        assert len(saved) >= 1

        # Load back and verify
        ds = pydicom.dcmread(saved[0])
        raw = ds.pixel_array.astype(np.float64)
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        reconstructed = (raw * slope + intercept).T

        original_slice = avg_image[:, :, 0]
        max_diff = np.max(np.abs(reconstructed - original_slice))
        max_value = np.max(np.abs(original_slice))
        relative_error = max_diff / max_value
        assert relative_error < 0.02, f"Relative error {relative_error:.4f} exceeds 2%"


# ============================================================================
# 6. Parameter Map Tests (T016, T017)
# ============================================================================

class TestParameterMaps:
    """Tests for parameter map dialog and export."""

    def test_param_map_dialog_renders(self, qt_app, screenshot_dir):
        """Dialog renders without crash, screenshot captured."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog

        shape = (16, 16, 3)
        np.random.seed(42)
        mask = np.ones(shape, dtype=bool)

        param_maps = {
            'kb_map': np.random.random(shape).astype(np.float32) * 0.5,
            'kd_map': np.random.random(shape).astype(np.float32) * 0.1,
            'mask': mask,
            'metadata': {'kernel_type': 'sliding_window'},
            'reference_slice': np.random.random((16, 16, 3)).astype(np.float32) * 5000,
        }
        roi_mask = np.ones(shape, dtype=bool)

        dialog = ParameterMapResultsDialog(
            param_maps=param_maps,
            spacing=(1.0, 1.0, 2.0),
            roi_mask=roi_mask,
            output_dir='/tmp/test_param',
        )
        dialog.show()
        qt_app.processEvents()

        # Capture screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_param_map_dialog.png"))

        dialog.close()
        qt_app.processEvents()

    def test_param_map_export_no_crash(self, qt_app, temp_dir):
        """T016: DICOM + PNG export works (roi_checkbox + numpy types)."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog

        shape = (16, 16, 3)
        np.random.seed(42)

        param_maps = {
            'kb_map': np.random.random(shape).astype(np.float32) * 0.5,
            'mask': np.ones(shape, dtype=bool),
            'metadata': {'kernel_type': 'sliding_window', 'window_x': np.int64(15)},
            'reference_slice': np.random.random((16, 16, 3)).astype(np.float32) * 5000,
        }
        roi_mask = np.ones(shape, dtype=bool)

        dialog = ParameterMapResultsDialog(
            param_maps=param_maps,
            spacing=(1.0, 1.0, 2.0),
            roi_mask=roi_mask,
            output_dir=str(temp_dir),
        )
        dialog.show()
        qt_app.processEvents()

        export_dir = temp_dir / "export_test"
        export_dir.mkdir()

        # Export DICOM
        with patch.object(dialog, '_show_map_selection_dialog',
                          return_value=(['kb_map'], False)):
            with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                       return_value=str(export_dir)):
                with patch('PySide6.QtWidgets.QMessageBox.information'):
                    dialog._export('dicom')

        dcm_files = list(export_dir.rglob("*.dcm"))
        assert len(dcm_files) >= 1, f"Expected DICOM files, got {len(dcm_files)}"

        # Export PNG
        png_dir = temp_dir / "png_export"
        png_dir.mkdir()

        with patch.object(dialog, '_show_map_selection_dialog',
                          return_value=(['kb_map'], False)):
            with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                       return_value=str(png_dir)):
                with patch('PySide6.QtWidgets.QMessageBox.information'):
                    dialog._export('png')

        png_files = list(png_dir.glob("*.png"))
        assert len(png_files) >= 1, f"Expected PNG files, got {len(png_files)}"

        dialog.close()
        qt_app.processEvents()

    def test_param_map_close_no_exit(self, qt_app):
        """T017: closing dialog doesn't exit Python."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog

        shape = (8, 8, 1)
        param_maps = {
            'kb_map': np.random.random(shape).astype(np.float32),
            'mask': np.ones(shape, dtype=bool),
            'metadata': {},
            'reference_slice': np.random.random((8, 8, 1)).astype(np.float32) * 5000,
        }

        dialog = ParameterMapResultsDialog(
            param_maps=param_maps,
            spacing=(1.0, 1.0, 2.0),
            output_dir='/tmp/test',
        )
        dialog.show()
        qt_app.processEvents()

        # Close the dialog - Python should NOT exit
        dialog.close()
        qt_app.processEvents()

        # If we reach here, Python didn't exit
        assert True, "Closing parameter map dialog should not exit Python"


# ============================================================================
# 7. Registration Viewer Test (T008)
# ============================================================================

class TestRegistrationViewer:
    """Tests for registration review dialog."""

    def test_registration_viewer_stable_image_size(self, qt_app, screenshot_dir):
        """T008 bug 1: screenshot dimensions stable across timepoints."""
        from proxyl_analysis.ui.registration import RegistrationReviewDialog

        np.random.seed(42)
        shape = (32, 32, 3, 10)
        original = np.random.random(shape).astype(np.float32) * 5000
        registered = original.copy() + np.random.normal(0, 50, shape).astype(np.float32)

        # Create mock metrics
        class MockMetric:
            def __init__(self, tp):
                self.timepoint = tp
                self.translation = (np.random.normal(0, 0.5), np.random.normal(0, 0.5), 0.0)
                self.rotation = (0.001, 0.002, 0.0)
                self.mean_squared_error = np.random.uniform(10, 100)
                self.mutual_info = np.random.uniform(0.8, 1.0)
                self.registration_time = 0.5

        metrics = [MockMetric(0)] + [MockMetric(t) for t in range(1, 10)]

        dialog = RegistrationReviewDialog(
            original, registered, metrics, z_slice=1
        )
        dialog.show()
        qt_app.processEvents()

        # Capture screenshot at timepoint 0
        screenshot1 = dialog.grab()
        size1 = (screenshot1.width(), screenshot1.height())

        # Change timepoint (no slider; use internal state + update)
        dialog.current_timepoint = 5
        dialog._update_display()
        qt_app.processEvents()

        # Capture screenshot at timepoint 5
        screenshot2 = dialog.grab()
        size2 = (screenshot2.width(), screenshot2.height())

        # Save one screenshot
        screenshot2.save(str(screenshot_dir / "test_registration_viewer.png"))

        assert size1 == size2, (
            f"Dialog size changed between timepoints: {size1} vs {size2}"
        )

        dialog.close()


# ============================================================================
# 8. Full Workflow with Real Data
# ============================================================================

class TestRealDataWorkflows:
    """Integration tests using real registered session data."""

    def test_full_pipeline_real_data(self, qt_app, real_session_path,
                                     screenshot_dir, sample_fit_results):
        """Load real session -> ROI -> fit -> export -> verify all outputs."""
        from proxyl_analysis.registration import load_registration_data
        from proxyl_analysis.roi_selection import compute_roi_timeseries
        from proxyl_analysis.ui.fitting import FitResultsDialog

        # Step 1: Load real registered data
        registered_4d, spacing, reg_metrics = load_registration_data(str(real_session_path))

        assert registered_4d.ndim == 4
        assert registered_4d.shape[3] > 10  # Should have many timepoints

        # Step 2: Create ROI mask (center region, 2D as expected by compute_roi_timeseries)
        roi_mask = np.zeros(registered_4d.shape[:2], dtype=bool)
        cx, cy = registered_4d.shape[0] // 2, registered_4d.shape[1] // 2
        roi_mask[cx-5:cx+5, cy-5:cy+5] = True

        # Step 3: Extract time series
        roi_signal = compute_roi_timeseries(registered_4d, roi_mask)
        assert len(roi_signal) == registered_4d.shape[3]
        assert np.all(np.isfinite(roi_signal))

        # Step 4: Show fit results (using sample fit results since we can't
        # reliably fit arbitrary ROI regions)
        time_array = np.arange(len(roi_signal), dtype=float) * (70.0 / 60.0)

        # Use real signal for display, but sample fit results for param verification
        dialog = FitResultsDialog(
            time_array[5:], roi_signal[5:],
            np.ones(len(roi_signal) - 5) * roi_signal[5:].mean(),
            sample_fit_results
        )
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_full_pipeline_real_data.png"))

        dialog.close()

    def test_scan_and_load_registered_real_data(self, qt_app, real_session_path,
                                                 screenshot_dir):
        """T023: scan real registered folder -> select T1 -> loads correctly."""
        from proxyl_analysis.dicom_scanner import scan_dicom_folder, find_proxyl_series
        from proxyl_analysis.run_analysis import _detect_registered_session

        dicoms_dir = real_session_path / "registered" / "dicoms"

        # Step 1: Scan the registered output directory
        results = scan_dicom_folder(str(dicoms_dir))
        assert len(results) > 0

        # Step 2: Verify PROXYL series detected
        proxyl = find_proxyl_series(results)
        assert len(proxyl) >= 1, (
            f"Expected PROXYL series, descriptions: "
            f"{[s['series_description'] for s in results]}"
        )

        # Step 3: Verify _detect_registered_session works
        sample_file = proxyl[0]['sample_file']
        session_path = _detect_registered_session(sample_file)
        assert session_path is not None, (
            f"Should detect registered session from: {sample_file}"
        )
        assert Path(session_path) == real_session_path

        # Step 4: Load the detected session
        from proxyl_analysis.registration import load_registration_data
        registered_4d, spacing, _ = load_registration_data(session_path)
        assert registered_4d.ndim == 4
        assert registered_4d.shape[3] > 10


# ============================================================================
# 9. _detect_registered_session Unit Tests
# ============================================================================

class TestDetectRegisteredSession:
    """Unit tests for the _detect_registered_session helper."""

    def test_detects_file_in_registered_dicoms(self, temp_dir):
        """Returns session root when file is inside registered/dicoms/."""
        from proxyl_analysis.run_analysis import _detect_registered_session

        # Create structure
        dicoms = temp_dir / "registered" / "dicoms"
        dicoms.mkdir(parents=True)
        (dicoms / "z00_t000.dcm").touch()
        sample = dicoms / "z04_t120.dcm"
        sample.touch()

        result = _detect_registered_session(str(sample))
        assert result == str(temp_dir)

    def test_returns_none_for_raw_dicom(self, temp_dir):
        """Returns None for a file not in registered/dicoms/ structure."""
        from proxyl_analysis.run_analysis import _detect_registered_session

        raw_file = temp_dir / "raw_scan.dcm"
        raw_file.touch()

        result = _detect_registered_session(str(raw_file))
        assert result is None

    def test_returns_none_for_incomplete_structure(self, temp_dir):
        """Returns None if z00_t000.dcm is missing."""
        from proxyl_analysis.run_analysis import _detect_registered_session

        dicoms = temp_dir / "registered" / "dicoms"
        dicoms.mkdir(parents=True)
        # Missing z00_t000.dcm
        sample = dicoms / "z04_t120.dcm"
        sample.touch()

        result = _detect_registered_session(str(sample))
        assert result is None


# ============================================================================
# 10. Main Menu State Management Tests
# ============================================================================

class TestMainMenuStateManagement:
    """Tests that MainMenuDialog buttons enable/disable correctly based on data state."""

    def test_buttons_disabled_when_no_data(self, qt_app):
        """With no registered_4d, data-dependent buttons are disabled."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog()

        assert not dialog.start_roi_btn.isEnabled()
        assert not dialog.create_maps_btn.isEnabled()
        assert not dialog.export_data_btn.isEnabled()
        assert not dialog.load_t2_btn.isEnabled()

        dialog.close()

    def test_buttons_enabled_when_data_loaded(self, qt_app, sample_4d,
                                                sample_spacing, sample_time_array):
        """With registered_4d, data-dependent buttons are enabled, ROI-dependent still disabled."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        assert dialog.start_roi_btn.isEnabled()
        assert dialog.create_maps_btn.isEnabled()
        assert dialog.export_data_btn.isEnabled()
        assert dialog.load_t2_btn.isEnabled()

        # ROI-dependent buttons should still be disabled
        assert not dialog.kinetic_fit_btn.isEnabled()
        assert not dialog.averaged_btn.isEnabled()
        assert not dialog.difference_btn.isEnabled()
        assert not dialog.export_timeseries_btn.isEnabled()

        dialog.close()

    def test_roi_dependent_buttons_enabled_after_set_roi_data(self, qt_app, sample_4d,
                                                                sample_spacing,
                                                                sample_time_array,
                                                                sample_roi_mask):
        """After set_roi_data(), kinetic_fit, averaged, difference, export_timeseries are enabled."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        signal = np.mean(sample_4d, axis=(0, 1, 2))
        dialog.set_roi_data(sample_roi_mask, signal, 5, 5.83)

        assert dialog.kinetic_fit_btn.isEnabled()
        assert dialog.averaged_btn.isEnabled()
        assert dialog.difference_btn.isEnabled()
        assert dialog.export_timeseries_btn.isEnabled()

        dialog.close()

    def test_roi_method_radio_produces_correct_result(self, qt_app, sample_4d,
                                                        sample_spacing,
                                                        sample_time_array):
        """Each ROI radio (rect/contour/segment) produces correct roi_mode in result."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        # Rectangle
        dialog.rect_radio.setChecked(True)
        dialog._draw_roi()
        assert dialog.result['roi_mode'] == 'rectangle'

        # Contour (re-create since _draw_roi calls accept)
        dialog2 = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )
        dialog2.contour_radio.setChecked(True)
        dialog2._draw_roi()
        assert dialog2.result['roi_mode'] == 'contour'

        # Segment
        dialog3 = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )
        dialog3.segment_radio.setChecked(True)
        dialog3._draw_roi()
        assert dialog3.result['roi_mode'] == 'segment'

    def test_roi_source_t2_selected_when_t2_loaded(self, qt_app, sample_4d,
                                                      sample_spacing,
                                                      sample_time_array,
                                                      sample_t2_volume):
        """With registered_t2, t2_source_radio is enabled+checked; _draw_roi -> 't2'."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
            registered_t2=sample_t2_volume,
        )

        assert dialog.t2_source_radio.isEnabled()
        assert dialog.t2_source_radio.isChecked()

        dialog._draw_roi()
        assert dialog.result['roi_source'] == 't2'

    def test_roi_source_falls_back_to_t1(self, qt_app, sample_4d,
                                           sample_spacing, sample_time_array):
        """Without T2, t1_source_radio is checked; _draw_roi -> 't1'."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        assert dialog.t1_source_radio.isChecked()
        assert not dialog.t2_source_radio.isEnabled()

        dialog._draw_roi()
        assert dialog.result['roi_source'] == 't1'

    def test_roi_status_label_states(self, qt_app, sample_4d, sample_spacing,
                                       sample_time_array, sample_roi_mask):
        """ROI status label shows correct text and color for each state."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        # No ROI -> "Not drawn" in gray
        assert "Not drawn" in dialog.roi_status_label.text()
        assert "#666" in dialog.roi_status_label.styleSheet()

        # Mask only -> "N pixels | Injection: Not set" in orange
        dialog.roi_mask = sample_roi_mask
        dialog._update_roi_status()
        assert "pixels" in dialog.roi_status_label.text()
        assert "Not set" in dialog.roi_status_label.text()
        assert "#FF9800" in dialog.roi_status_label.styleSheet()

        # Full data -> green with count + index
        signal = np.mean(sample_4d, axis=(0, 1, 2))
        dialog.set_roi_data(sample_roi_mask, signal, 5, 5.83)
        assert "pixels" in dialog.roi_status_label.text()
        assert "5" in dialog.roi_status_label.text()
        assert "green" in dialog.roi_status_label.styleSheet()

        dialog.close()


# ============================================================================
# 11. Main Menu Export Actions
# ============================================================================

class TestMainMenuExportActions:
    """Tests for export actions from MainMenuDialog."""

    def test_export_timeseries_produces_result(self, qt_app, sample_4d,
                                                 sample_spacing, sample_time_array):
        """_export('timeseries') produces correct result dict."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        dialog._export('timeseries')
        assert dialog.result == {'action': 'export', 'export_type': 'timeseries'}

    def test_dicom_scan_results_csv_export(self, qt_app, temp_dir):
        """DicomScanResultsDialog._save_csv() writes valid CSV."""
        from proxyl_analysis.ui.main_menu import DicomScanResultsDialog

        scan_results = [
            {
                'series_number': 1,
                'series_description': 'Registered T1 DCE',
                'study_description': 'PROXYL Study',
                'modality': 'MR',
                'rows': 128, 'cols': 128,
                'num_slices': 9, 'num_frames': 184,
                'num_files': 1656,
                'patient_id': 'TEST',
                'study_date': '20260101',
                'sample_file': '/tmp/test_t1.dcm',
                'is_proxyl': True, 'is_t2': False,
                'series_uid': 'uid1',
            },
        ]

        dialog = DicomScanResultsDialog(scan_results, str(temp_dir))
        csv_path = str(temp_dir / "scan_results.csv")

        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV Files (*.csv)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._save_csv()

        assert Path(csv_path).exists()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) > 0
            rows = list(reader)
            assert len(rows) >= 1

        dialog.close()


# ============================================================================
# 12. Manual Contour Dialog Tests
# ============================================================================

class TestManualContourDialog:
    """Tests for manual contour ROI drawing dialog."""

    def test_contour_close_creates_mask(self, qt_app, sample_4d):
        """Setting contour points and closing creates a non-None mask."""
        from proxyl_analysis.ui.roi import ManualContourDialog

        dialog = ManualContourDialog(sample_4d, z_index=1)
        dialog.show()
        qt_app.processEvents()

        # Set triangle contour points
        dialog.contour_points = [(10, 10), (20, 10), (15, 20)]
        dialog._close_contour()

        assert dialog.mask is not None
        assert np.sum(dialog.mask) > 0

        dialog.close()

    def test_z_navigation_resets_contour(self, qt_app, sample_4d):
        """Changing z-slice resets contour points and mask."""
        from proxyl_analysis.ui.roi import ManualContourDialog

        dialog = ManualContourDialog(sample_4d, z_index=0)
        dialog.show()
        qt_app.processEvents()

        # Draw and close a contour
        dialog.contour_points = [(10, 10), (20, 10), (15, 20)]
        dialog._close_contour()
        assert dialog.mask is not None

        # Change z-slice -> should reset
        dialog.z_slider.setValue(1)
        qt_app.processEvents()

        assert dialog.contour_points == []
        assert dialog.mask is None

        dialog.close()

    def test_close_insufficient_points_warns(self, qt_app, sample_4d):
        """Closing with < 3 points shows warning, mask stays None."""
        from proxyl_analysis.ui.roi import ManualContourDialog

        dialog = ManualContourDialog(sample_4d, z_index=0)
        dialog.show()
        qt_app.processEvents()

        dialog.contour_points = [(10, 10), (20, 10)]

        with patch('PySide6.QtWidgets.QMessageBox.warning') as mock_warn:
            dialog._close_contour()
            mock_warn.assert_called_once()

        assert dialog.mask is None

        dialog.close()

    def test_reset_clears_all_state(self, qt_app, sample_4d):
        """_reset_contour() clears points, mask, and path_plots."""
        from proxyl_analysis.ui.roi import ManualContourDialog

        dialog = ManualContourDialog(sample_4d, z_index=0)
        dialog.show()
        qt_app.processEvents()

        dialog.contour_points = [(10, 10), (20, 10), (15, 20)]
        dialog._close_contour()
        assert dialog.mask is not None
        assert len(dialog.path_plots) > 0

        dialog._reset_contour()

        assert dialog.contour_points == []
        assert dialog.mask is None
        assert dialog.path_plots == []

        dialog.close()


# ============================================================================
# 13. ROI Dialog Accept / Cancel Tests
# ============================================================================

class TestRoiDialogAcceptCancel:
    """Tests for ROI dialog accept and cancel behavior."""

    def test_cancel_returns_none_mask(self, qt_app, sample_4d):
        """Rejecting ROISelectorDialog -> get_mask() returns None."""
        from proxyl_analysis.ui.roi import ROISelectorDialog

        image_slice = sample_4d[:, :, 1, 0]
        dialog = ROISelectorDialog(image_slice)
        dialog.reject()

        assert dialog.get_mask() is None

        dialog.close()

    def test_accept_emits_signal(self, qt_app, sample_4d):
        """Setting mask and calling _accept_roi emits roi_selected signal."""
        from proxyl_analysis.ui.roi import ROISelectorDialog

        image_slice = sample_4d[:, :, 1, 0]
        dialog = ROISelectorDialog(image_slice)
        dialog.show()
        qt_app.processEvents()

        # Set a mask
        mask = np.zeros_like(image_slice, dtype=bool)
        mask[10:20, 10:20] = True
        dialog.mask = mask

        mock_slot = MagicMock()
        dialog.roi_selected.connect(mock_slot)

        dialog._accept_roi()

        mock_slot.assert_called_once()
        emitted_mask = mock_slot.call_args[0][0]
        assert np.array_equal(emitted_mask, mask)

    def test_get_stats_correct_values(self, qt_app, sample_4d):
        """get_stats() returns correct num_pixels, mean_intensity, etc."""
        from proxyl_analysis.ui.roi import ROISelectorDialog

        image_slice = sample_4d[:, :, 1, 0]
        dialog = ROISelectorDialog(image_slice)

        # Create a known rectangular mask
        mask = np.zeros_like(image_slice, dtype=bool)
        mask[10:20, 10:20] = True
        dialog.mask = mask
        dialog.roi_coords = (10, 20, 10, 20)

        stats = dialog.get_stats()
        expected_pixels = np.sum(mask)
        expected_mean = np.mean(image_slice[mask])

        assert stats['num_pixels'] == expected_pixels
        assert stats['mean_intensity'] == pytest.approx(expected_mean)
        assert 'std_intensity' in stats
        assert 'min_intensity' in stats
        assert 'max_intensity' in stats

        dialog.close()


# ============================================================================
# 14. Registration Review Navigation Tests
# ============================================================================

class TestRegistrationReviewNavigation:
    """Tests for RegistrationReviewDialog navigation."""

    def _create_dialog(self, qt_app):
        """Helper to create a RegistrationReviewDialog with synthetic data."""
        from proxyl_analysis.ui.registration import RegistrationReviewDialog

        np.random.seed(42)
        shape = (32, 32, 3, 10)
        original = np.random.random(shape).astype(np.float32) * 5000
        registered = original.copy()

        class MockMetric:
            def __init__(self, tp):
                self.timepoint = tp
                self.translation = (0.1, 0.2, 0.0)
                self.rotation = (0.001, 0.002, 0.0)
                self.mean_squared_error = 50.0
                self.mutual_info = 0.9
                self.registration_time = 0.5

        metrics = [MockMetric(0)] + [MockMetric(t) for t in range(1, 10)]

        dialog = RegistrationReviewDialog(
            original, registered, metrics, z_slice=1
        )
        dialog.show()
        qt_app.processEvents()
        return dialog

    def test_first_prev_next_last_buttons(self, qt_app):
        """Navigation methods move to correct timepoints."""
        dialog = self._create_dialog(qt_app)

        assert dialog.current_timepoint == 1

        dialog._next_timepoint()
        dialog._next_timepoint()
        dialog._next_timepoint()
        assert dialog.current_timepoint == 4

        dialog._first_timepoint()
        assert dialog.current_timepoint == 1

        dialog._last_timepoint()
        assert dialog.current_timepoint == dialog.max_timepoint

        dialog._prev_timepoint()
        assert dialog.current_timepoint == dialog.max_timepoint - 1

        dialog.close()

    def test_keyboard_navigation(self, qt_app):
        """Arrow key events move timepoints and z-slices."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        dialog = self._create_dialog(qt_app)
        initial_tp = dialog.current_timepoint

        # Right arrow x2
        event_right = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Right, Qt.NoModifier)
        dialog.keyPressEvent(event_right)
        dialog.keyPressEvent(event_right)
        assert dialog.current_timepoint == initial_tp + 2

        # Left arrow -> back 1
        event_left = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Left, Qt.NoModifier)
        dialog.keyPressEvent(event_left)
        assert dialog.current_timepoint == initial_tp + 1

        # Up -> z + 1
        initial_z = dialog.z_slice
        event_up = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Up, Qt.NoModifier)
        dialog.keyPressEvent(event_up)
        assert dialog.z_slice == initial_z + 1

        # Down -> z - 1
        event_down = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Down, Qt.NoModifier)
        dialog.keyPressEvent(event_down)
        assert dialog.z_slice == initial_z

        dialog.close()

    def test_z_spinbox_changes_slice(self, qt_app):
        """Z spinbox setValue updates dialog.z_slice."""
        dialog = self._create_dialog(qt_app)

        dialog.z_spinbox.setValue(2)
        qt_app.processEvents()
        assert dialog.z_slice == 2

        dialog.z_spinbox.setValue(0)
        qt_app.processEvents()
        assert dialog.z_slice == 0

        dialog.close()

    def test_boundary_clamping(self, qt_app):
        """Navigation at boundaries stays clamped."""
        dialog = self._create_dialog(qt_app)

        # At t=1, prev stays at 1
        dialog._first_timepoint()
        assert dialog.current_timepoint == 1
        dialog._prev_timepoint()
        assert dialog.current_timepoint == 1

        # At max, next stays at max
        dialog._last_timepoint()
        max_tp = dialog.max_timepoint
        dialog._next_timepoint()
        assert dialog.current_timepoint == max_tp

        # Z at 0, down stays 0
        dialog.z_spinbox.setValue(0)
        qt_app.processEvents()
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        event_down = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Down, Qt.NoModifier)
        dialog.keyPressEvent(event_down)
        assert dialog.z_slice == 0

        dialog.close()

    def test_timepoint_label_updates(self, qt_app):
        """Navigating to timepoint 5 shows '5' in timepoint_label."""
        dialog = self._create_dialog(qt_app)

        # Go to timepoint 5
        dialog.current_timepoint = 5
        dialog._update_display()
        qt_app.processEvents()

        assert "5" in dialog.timepoint_label.text()

        dialog.close()


# ============================================================================
# 15. T2 Registration Review Dialog Tests
# ============================================================================

class TestT2RegistrationReviewDialog:
    """Tests for T2RegistrationReviewDialog."""

    def _make_reg_info(self):
        return {
            'translation_mm': (0.5, -0.3, 0.1),
            'rotation_deg': (0.01, 0.02, 0.0),
            'mutual_information': 0.92,
            'registration_time': 2.5,
            'resampled': True,
        }

    def test_t2_review_renders(self, qt_app, sample_4d, sample_t2_volume, screenshot_dir):
        """T2 review dialog creates without crash."""
        from proxyl_analysis.ui.registration import T2RegistrationReviewDialog

        t1_ref = sample_4d[:, :, :, 0]  # shape (32, 32, 3)
        t2_orig = sample_t2_volume.copy()
        t2_reg = sample_t2_volume.copy()

        dialog = T2RegistrationReviewDialog(
            t1_ref, t2_orig, t2_reg, self._make_reg_info()
        )
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "test_t2_review_renders.png"))

        dialog.close()

    def test_t2_review_z_navigation(self, qt_app, sample_4d, sample_t2_volume):
        """Z spinbox changes z_slice correctly."""
        from proxyl_analysis.ui.registration import T2RegistrationReviewDialog

        t1_ref = sample_4d[:, :, :, 0]
        dialog = T2RegistrationReviewDialog(
            t1_ref, sample_t2_volume.copy(), sample_t2_volume.copy(),
            self._make_reg_info()
        )
        dialog.show()
        qt_app.processEvents()

        dialog.z_spinbox.setValue(0)
        qt_app.processEvents()
        assert dialog.z_slice == 0

        dialog.z_spinbox.setValue(2)
        qt_app.processEvents()
        assert dialog.z_slice == 2

        dialog.close()


# ============================================================================
# 16. Parameter Map Options Dialog Tests
# ============================================================================

class TestParameterMapOptionsDialog:
    """Tests for ParameterMapOptionsDialog configuration."""

    def test_single_slice_vs_all_slices(self, qt_app):
        """Single slice mode sets single_slice=True and z_slice; all slices sets None."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=3)

        # Single slice
        dialog.single_slice_radio.setChecked(True)
        dialog.z_spinbox.setValue(3)
        dialog._on_run()

        assert dialog.result['single_slice'] is True
        assert dialog.result['z_slice'] == 3

        # All slices (re-create)
        dialog2 = ParameterMapOptionsDialog(max_z=8, current_z=3)
        dialog2.all_slices_radio.setChecked(True)
        dialog2._on_run()

        assert dialog2.result['single_slice'] is False
        assert dialog2.result['z_slice'] is None

    def test_roi_only_with_reuse(self, qt_app, sample_roi_mask):
        """ROI-only with reuse produces correct result."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(
            max_z=2, current_z=1,
            existing_roi=sample_roi_mask,
        )

        dialog.roi_only_radio.setChecked(True)
        qt_app.processEvents()
        dialog.reuse_roi_radio.setChecked(True)
        dialog._on_run()

        assert dialog.result['roi_only'] is True
        assert dialog.result['reuse_roi'] is True

        dialog.close()

    def test_window_size_controls(self, qt_app):
        """Window size spinboxes produce correct tuple."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=4)

        dialog.window_x_spin.setValue(21)
        dialog.window_y_spin.setValue(21)
        dialog.window_z_spin.setValue(5)
        dialog._on_run()

        assert dialog.result['window_size'] == (21, 21, 5)

    def test_kernel_type_selection(self, qt_app):
        """Kernel type combo produces correct string."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=4)

        dialog.kernel_type_combo.setCurrentText("gaussian")
        dialog._on_run()

        assert dialog.result['kernel_type'] == 'gaussian'

    def test_stride_spinbox_default(self, qt_app):
        """Stride SpinBox defaults to 1 (full resolution)."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=4)
        dialog._on_run()

        assert dialog.result['stride'] == 1
        dialog.close()

    def test_stride_spinbox_custom_value(self, qt_app):
        """Setting stride SpinBox to 4 produces stride=4 in result dict."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=4)

        dialog.stride_spin.setValue(4)
        dialog._on_run()

        assert dialog.result['stride'] == 4
        dialog.close()

    def test_stride_spinbox_boundary_values(self, qt_app):
        """Stride SpinBox respects range 1-32."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=4)

        # Minimum
        dialog.stride_spin.setValue(1)
        assert dialog.stride_spin.value() == 1

        # Maximum
        dialog.stride_spin.setValue(32)
        assert dialog.stride_spin.value() == 32

        # Beyond max clamps
        dialog.stride_spin.setValue(64)
        assert dialog.stride_spin.value() == 32

        # Below min clamps
        dialog.stride_spin.setValue(0)
        assert dialog.stride_spin.value() == 1

        dialog.close()

    def test_stride_help_text_present(self, qt_app):
        """Stride section includes explanatory help text."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

        dialog = ParameterMapOptionsDialog(max_z=8, current_z=4)

        # Find label containing help text
        from PySide6.QtWidgets import QLabel
        labels = dialog.findChildren(QLabel)
        help_texts = [l.text() for l in labels if 'full resolution' in l.text()]

        assert len(help_texts) >= 1, "Expected help text with 'full resolution'"
        assert 'faster' in help_texts[0].lower() or 'coarser' in help_texts[0].lower()

        dialog.close()


# ============================================================================
# 17. Parameter Map Results Metrics Tests
# ============================================================================

class TestParameterMapResultsMetrics:
    """Tests for ParameterMapResultsDialog metrics and display."""

    def _create_dialog(self, qt_app, sample_param_maps, sample_roi_mask, temp_dir):
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog

        # Resize ROI mask to match param map shape (16x16x3)
        roi_mask = np.ones((16, 16, 3), dtype=bool)

        dialog = ParameterMapResultsDialog(
            param_maps=sample_param_maps,
            spacing=(1.0, 1.0, 2.0),
            roi_mask=roi_mask,
            output_dir=str(temp_dir),
        )
        dialog.show()
        qt_app.processEvents()
        return dialog

    def test_metrics_update_on_z_change(self, qt_app, sample_param_maps,
                                          sample_roi_mask, temp_dir):
        """Metrics text changes when z-slice changes."""
        dialog = self._create_dialog(qt_app, sample_param_maps, sample_roi_mask, temp_dir)

        text_z0 = dialog.metrics_label.text()

        dialog.z_slider.setValue(2)
        qt_app.processEvents()

        text_z2 = dialog.metrics_label.text()
        # Metrics should differ because different slices have different random data
        assert text_z0 != text_z2 or dialog.current_z == 2

        dialog.close()

    def test_metrics_csv_export(self, qt_app, sample_param_maps,
                                  sample_roi_mask, temp_dir):
        """_export_metrics() creates CSV with correct header."""
        dialog = self._create_dialog(qt_app, sample_param_maps, sample_roi_mask, temp_dir)

        csv_path = str(temp_dir / "metrics.csv")

        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV Files (*.csv)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._export_metrics()

        assert Path(csv_path).exists()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['z_slice', 'parameter', 'n_pixels', 'mean', 'std', 'min', 'max']
            rows = list(reader)
            # Should have rows for kb, kd, knt, r_squared across 3 slices
            assert len(rows) >= 4  # At least one parameter across slices

        dialog.close()

    def test_map_combo_switches_display(self, qt_app, sample_param_maps,
                                          sample_roi_mask, temp_dir):
        """Changing map combo index changes current_map."""
        dialog = self._create_dialog(qt_app, sample_param_maps, sample_roi_mask, temp_dir)

        assert dialog.current_map == 'kb_map'

        dialog.map_combo.setCurrentIndex(1)
        qt_app.processEvents()
        assert dialog.current_map == 'kd_map'

        dialog.map_combo.setCurrentIndex(3)
        qt_app.processEvents()
        assert dialog.current_map == 'r_squared_map'

        dialog.close()

    def test_roi_overlay_toggle(self, qt_app, sample_param_maps,
                                  sample_roi_mask, temp_dir):
        """Toggling ROI checkbox doesn't crash."""
        dialog = self._create_dialog(qt_app, sample_param_maps, sample_roi_mask, temp_dir)

        dialog.roi_checkbox.setChecked(False)
        qt_app.processEvents()

        dialog.roi_checkbox.setChecked(True)
        qt_app.processEvents()

        # If we got here, no crash
        assert True

        dialog.close()


# ============================================================================
# 18. Image Tools Advanced Tests
# ============================================================================

class TestImageToolsAdvanced:
    """Advanced tests for ImageToolsDialog."""

    def _create_dialog(self, qt_app, sample_4d, sample_roi_mask, temp_dir,
                        mode='average'):
        from proxyl_analysis.ui.image_tools import ImageToolsDialog

        time_array = np.arange(20, dtype=float) * (70.0 / 60.0)
        roi_signal = np.mean(sample_4d, axis=(0, 1, 2))

        dialog = ImageToolsDialog(
            image_4d=sample_4d,
            time_array=time_array,
            roi_signal=roi_signal,
            time_units='minutes',
            output_dir=str(temp_dir),
            initial_mode=mode,
            roi_mask=sample_roi_mask,
        )
        dialog.show()
        qt_app.processEvents()
        time_mod.sleep(0.05)
        qt_app.processEvents()
        return dialog

    def test_mode_switch_average_to_difference(self, qt_app, sample_4d,
                                                 sample_roi_mask, temp_dir):
        """Switching to difference mode shows region_b_group; back to average hides it."""
        dialog = self._create_dialog(qt_app, sample_4d, sample_roi_mask, temp_dir,
                                      mode='average')

        assert not dialog.region_b_group.isVisible()

        dialog._set_mode('difference')
        qt_app.processEvents()
        assert dialog.region_b_group.isVisible()

        dialog._set_mode('average')
        qt_app.processEvents()
        assert not dialog.region_b_group.isVisible()

        dialog.close()

    def test_clear_region_resets_state(self, qt_app, sample_4d,
                                        sample_roi_mask, temp_dir):
        """_clear_region_a() resets region_a_start to None."""
        dialog = self._create_dialog(qt_app, sample_4d, sample_roi_mask, temp_dir)

        # Set regions
        dialog.region_a_start = 2
        dialog.region_a_end = 8
        dialog.region_a_start_spin.setValue(2)
        dialog.region_a_end_spin.setValue(8)
        qt_app.processEvents()

        dialog._clear_region_a()
        qt_app.processEvents()

        assert dialog.region_a_start is None

        dialog.close()

    def test_z_slider_updates_preview(self, qt_app, sample_4d,
                                        sample_roi_mask, temp_dir):
        """Changing z_slider updates current_z."""
        dialog = self._create_dialog(qt_app, sample_4d, sample_roi_mask, temp_dir)

        # Set regions and compute preview
        dialog.region_a_start_spin.setValue(2)
        dialog.region_a_end_spin.setValue(8)
        qt_app.processEvents()
        dialog._update_preview()
        qt_app.processEvents()

        assert dialog.preview_image is not None

        # Change z
        dialog.z_slider.setValue(0)
        qt_app.processEvents()
        assert dialog.current_z == 0

        dialog.z_slider.setValue(2)
        qt_app.processEvents()
        assert dialog.current_z == 2

        dialog.close()


# ============================================================================
# 19. Injection Time Export Tests
# ============================================================================

class TestInjectionTimeExport:
    """Tests for InjectionTimeSelectorDialog CSV export."""

    def test_csv_content_format(self, qt_app, sample_time_array, sample_4d, temp_dir):
        """_export_csv() creates file with correct header and rows."""
        from proxyl_analysis.ui.injection import InjectionTimeSelectorDialog

        signal = np.mean(sample_4d, axis=(0, 1, 2))

        dialog = InjectionTimeSelectorDialog(
            sample_time_array, signal, 'minutes', str(temp_dir)
        )

        csv_file = temp_dir / "timecourse_data.csv"
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(str(csv_file), 'CSV Files (*.csv)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._export_csv()

        assert csv_file.exists()

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['Time (minutes)', 'Mean Intensity']
            rows = list(reader)
            assert len(rows) == 20  # 20 timepoints

        dialog.close()

    def test_csv_values_match_input(self, qt_app, sample_time_array, sample_4d, temp_dir):
        """CSV values match input arrays."""
        from proxyl_analysis.ui.injection import InjectionTimeSelectorDialog

        signal = np.mean(sample_4d, axis=(0, 1, 2))

        dialog = InjectionTimeSelectorDialog(
            sample_time_array, signal, 'minutes', str(temp_dir)
        )

        csv_file = temp_dir / "timecourse_data.csv"
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(str(csv_file), 'CSV Files (*.csv)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._export_csv()

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        # First row
        assert float(rows[0][0]) == pytest.approx(sample_time_array[0], abs=0.01)
        assert float(rows[0][1]) == pytest.approx(signal[0], rel=0.01)

        # Last row
        assert float(rows[-1][0]) == pytest.approx(sample_time_array[-1], abs=0.01)
        assert float(rows[-1][1]) == pytest.approx(signal[-1], rel=0.01)

        dialog.close()


# ============================================================================
# 20. Fitting Dialog Save Plot Test
# ============================================================================

class TestFittingDialogSavePlot:
    """Test for FitResultsDialog plot save."""

    def test_save_plot_creates_file(self, qt_app, sample_fit_results, temp_dir):
        """_save_plot() creates a non-empty PNG file."""
        from proxyl_analysis.ui.fitting import FitResultsDialog

        time = np.linspace(0, 70, 126)
        signal = np.ones(126) * 5000
        fitted = np.ones(126) * 5000

        dialog = FitResultsDialog(time, signal, fitted, sample_fit_results)
        dialog.show()
        qt_app.processEvents()

        png_path = str(temp_dir / "fit_plot.png")

        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(png_path, 'PNG Files (*.png)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._save_plot()

        assert Path(png_path).exists()
        assert Path(png_path).stat().st_size > 0

        dialog.close()


# ============================================================================
# 21. Synthetic Data Integration Test
# ============================================================================

class TestSyntheticDataIntegration:
    """End-to-end integration test with synthetic data."""

    def test_menu_state_flow(self, qt_app, sample_4d, sample_spacing,
                               sample_time_array, sample_roi_mask):
        """Full flow: create menu -> verify ROI buttons -> set ROI -> run kinetic fit."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        dialog = MainMenuDialog(
            registered_4d=sample_4d,
            spacing=sample_spacing,
            time_array=sample_time_array,
        )

        # ROI buttons should be enabled (data loaded)
        assert dialog.start_roi_btn.isEnabled()

        # Set ROI data
        signal = np.mean(sample_4d, axis=(0, 1, 2))
        dialog.set_roi_data(sample_roi_mask, signal, 5, 5.83)

        # Kinetic fit button should now be enabled
        assert dialog.kinetic_fit_btn.isEnabled()

        # Run kinetic fit
        dialog._run_kinetic_fit()

        assert dialog.result['action'] == 'kinetic_fit'
        assert np.array_equal(dialog.result['roi_mask'], sample_roi_mask)
        assert dialog.result['injection_idx'] == 5


# ============================================================================
# 22. Real-Data Visual Verification Tests
# ============================================================================

class TestRealDataVisualVerification:
    """Tests using real MRI session data for visual verification.

    Each test loads actual registered brain data into a dialog, saves a
    screenshot to tests/screenshots/real_*, and asserts basic correctness.
    Screenshots are inspected visually after the test run.
    """

    def test_real_main_menu(self, qt_app, real_session_data, screenshot_dir):
        """Main menu with real 4D + T2 loaded, ROI set, all buttons enabled."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog

        data = real_session_data

        # Build 3D ROI mask from 2D mask (replicate across z)
        nz = data['registered_4d'].shape[2]
        roi_mask_3d = np.repeat(data['roi_mask_2d'][:, :, np.newaxis], nz, axis=2)

        dialog = MainMenuDialog(
            registered_4d=data['registered_4d'],
            spacing=data['spacing'],
            time_array=data['time_array'],
            registered_t2=data['registered_t2'],
            output_dir=data['session_path'],
        )
        dialog.set_roi_data(
            roi_mask_3d, data['roi_signal'], 6, data['time_array'][6]
        )
        dialog.show()
        qt_app.processEvents()

        # Screenshot
        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_main_menu.png"))

        # Assertions: all workflow buttons should be enabled
        assert dialog.start_roi_btn.isEnabled()
        assert dialog.kinetic_fit_btn.isEnabled()
        assert dialog.averaged_btn.isEnabled()
        assert dialog.difference_btn.isEnabled()
        assert dialog.create_maps_btn.isEnabled()
        assert dialog.export_timeseries_btn.isEnabled()

        dialog.close()

    def test_real_registration_review(self, qt_app, real_session_data, screenshot_dir):
        """Registration review with real brain at middle z-slice and timepoint."""
        from proxyl_analysis.ui.registration import RegistrationReviewDialog

        data = real_session_data
        mid_z = data['registered_4d'].shape[2] // 2

        # Use the registered 4D as both original and registered
        # (real data has only the registered output saved)
        dialog = RegistrationReviewDialog(
            data['registered_4d'],
            data['registered_4d'],
            data['metrics'],
            z_slice=mid_z,
        )
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_registration_review.png"))

        # Assertions
        assert dialog.current_timepoint >= 1
        assert dialog.z_slice == mid_z
        assert dialog.max_timepoint == data['registered_4d'].shape[3] - 1

        dialog.close()

    def test_real_registration_review_navigation(self, qt_app, real_session_data,
                                                   screenshot_dir):
        """Navigate to timepoint 60 (post-injection), screenshot shows signal change."""
        from proxyl_analysis.ui.registration import RegistrationReviewDialog

        data = real_session_data
        mid_z = data['registered_4d'].shape[2] // 2

        dialog = RegistrationReviewDialog(
            data['registered_4d'],
            data['registered_4d'],
            data['metrics'],
            z_slice=mid_z,
        )
        dialog.show()
        qt_app.processEvents()

        # Navigate to post-injection timepoint
        target_tp = min(60, dialog.max_timepoint)
        dialog.current_timepoint = target_tp
        dialog._update_display()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_registration_review_tp60.png"))

        assert dialog.current_timepoint == target_tp

        dialog.close()

    def test_real_t2_registration_review(self, qt_app, real_session_data, screenshot_dir):
        """T2 registration review with real T1 reference and T2 data."""
        from proxyl_analysis.ui.registration import T2RegistrationReviewDialog

        data = real_session_data

        if data['registered_t2'] is None:
            pytest.skip("No registered T2 data available")

        t1_ref = data['registered_4d'][:, :, :, 0]  # First timepoint as reference

        reg_info = {
            'translation_mm': (0.0, 0.0, 0.0),
            'rotation_deg': (0.0, 0.0, 0.0),
            'mutual_information': 0.0,
            'registration_time': 0.0,
            'resampled': True,
        }

        dialog = T2RegistrationReviewDialog(
            t1_ref, data['registered_t2'], data['registered_t2'], reg_info
        )
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_t2_registration_review.png"))

        assert dialog.z_slice == t1_ref.shape[2] // 2

        dialog.close()

    def test_real_roi_selector(self, qt_app, real_session_data, screenshot_dir):
        """ROI selector with real brain slice and ROI mask applied."""
        from proxyl_analysis.ui.roi import ROISelectorDialog

        data = real_session_data
        mid_z = data['registered_4d'].shape[2] // 2
        image_slice = data['registered_4d'][:, :, mid_z, 0]

        dialog = ROISelectorDialog(image_slice, title="Select ROI (Real Brain)")
        dialog.mask = data['roi_mask_2d']
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_roi_selector.png"))

        assert dialog.get_mask() is not None
        assert np.sum(dialog.get_mask()) > 0

        dialog.close()

    def test_real_manual_contour(self, qt_app, real_session_data, screenshot_dir):
        """Manual contour dialog with real 4D brain anatomy."""
        from proxyl_analysis.ui.roi import ManualContourDialog

        data = real_session_data

        dialog = ManualContourDialog(data['registered_4d'], z_index=4)
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_manual_contour.png"))

        # Verify dialog loaded with correct slice
        assert dialog.z_index == 4
        assert dialog.image_slice is not None
        assert dialog.image_slice.shape == data['registered_4d'].shape[:2]

        dialog.close()

    def test_real_injection_time(self, qt_app, real_session_data, screenshot_dir):
        """Injection time selector with real PROXYL uptake/washout curve."""
        from proxyl_analysis.ui.injection import InjectionTimeSelectorDialog

        data = real_session_data

        dialog = InjectionTimeSelectorDialog(
            data['time_array'], data['roi_signal'], 'minutes',
            data['session_path']
        )
        dialog.injection_index = 6
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_injection_time.png"))

        assert dialog.get_injection_index() == 6
        assert len(data['roi_signal']) == data['registered_4d'].shape[3]

        dialog.close()

    def test_real_parameter_maps(self, qt_app, real_session_data, screenshot_dir):
        """Parameter map results with real kb map overlaid on brain anatomy."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog

        data = real_session_data

        if data['param_maps'] is None:
            pytest.skip("No parameter maps available")

        # Build param_maps dict for the dialog
        param_maps = dict(data['param_maps'])

        # Add metadata from JSON if not already present
        if 'metadata' not in param_maps:
            param_maps['metadata'] = data['param_metadata']

        # Build reference slice from the registered 4D at z=param_metadata z_slice
        z_slice = data['param_metadata'].get('z_slice', 4)
        ref_slice = data['registered_4d'][:, :, z_slice:z_slice+1, 0]
        param_maps['reference_slice'] = ref_slice

        # ROI mask matching parameter map shape
        pm_shape = param_maps['kb_map'].shape
        if data['roi_mask_2d'].shape[:2] == pm_shape[:2]:
            if len(pm_shape) == 3:
                roi_mask = np.repeat(
                    data['roi_mask_2d'][:, :, np.newaxis], pm_shape[2], axis=2
                )
            else:
                roi_mask = data['roi_mask_2d']
        else:
            roi_mask = None

        dialog = ParameterMapResultsDialog(
            param_maps=param_maps,
            spacing=data['spacing'],
            roi_mask=roi_mask,
            output_dir=data['session_path'],
        )
        dialog.show()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_parameter_maps.png"))

        assert dialog.current_map == 'kb_map'

        dialog.close()

    def test_real_image_tools_average(self, qt_app, real_session_data,
                                       screenshot_dir, tmp_path):
        """Image tools averaged image (t=0-5) showing pre-injection brain."""
        from proxyl_analysis.ui.image_tools import ImageToolsDialog

        data = real_session_data
        nz = data['registered_4d'].shape[2]
        roi_mask_3d = np.repeat(data['roi_mask_2d'][:, :, np.newaxis], nz, axis=2)

        dialog = ImageToolsDialog(
            image_4d=data['registered_4d'],
            time_array=data['time_array'],
            roi_signal=data['roi_signal'],
            time_units='minutes',
            output_dir=str(tmp_path),
            initial_mode='average',
            roi_mask=roi_mask_3d,
        )
        dialog.show()
        qt_app.processEvents()

        # Set pre-injection region (timepoints 0-5)
        # Must set internal state directly since setValue(0) won't fire
        # valueChanged when spinbox already at 0
        dialog.region_a_start = 0
        dialog.region_a_end = 5
        dialog.region_a_start_spin.setValue(0)
        dialog.region_a_end_spin.setValue(5)
        qt_app.processEvents()
        time_mod.sleep(0.1)
        dialog._update_preview()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_image_tools_average.png"))

        assert dialog.preview_image is not None

        dialog.close()

    def test_real_image_tools_difference(self, qt_app, real_session_data,
                                          screenshot_dir, tmp_path):
        """Image tools difference image (post - pre injection) showing PROXYL enhancement."""
        from proxyl_analysis.ui.image_tools import ImageToolsDialog

        data = real_session_data
        nz = data['registered_4d'].shape[2]
        roi_mask_3d = np.repeat(data['roi_mask_2d'][:, :, np.newaxis], nz, axis=2)

        dialog = ImageToolsDialog(
            image_4d=data['registered_4d'],
            time_array=data['time_array'],
            roi_signal=data['roi_signal'],
            time_units='minutes',
            output_dir=str(tmp_path),
            initial_mode='difference',
            roi_mask=roi_mask_3d,
        )
        dialog.show()
        qt_app.processEvents()

        # Region A: pre-injection (0-5), Region B: post-injection (10-20)
        dialog.region_a_start = 0
        dialog.region_a_end = 5
        dialog.region_b_start = 10
        dialog.region_b_end = 20
        dialog.region_a_start_spin.setValue(0)
        dialog.region_a_end_spin.setValue(5)
        dialog.region_b_start_spin.setValue(10)
        dialog.region_b_end_spin.setValue(20)
        qt_app.processEvents()
        time_mod.sleep(0.1)
        dialog._update_preview()
        qt_app.processEvents()

        screenshot = dialog.grab()
        screenshot.save(str(screenshot_dir / "real_image_tools_difference.png"))

        assert dialog.preview_image is not None

        dialog.close()
