"""
Tests for user feedback fixes.

Validates:
- A1: show_roi_cb → roi_checkbox fix in parameter map export
- A2: numpy int64 JSON serialization in DICOM metadata
- A3: setQuitOnLastWindowClosed(False) in init_qt_app
- C1: Simplified difference image labels and filenames
- C2: Grayscale colormap for difference images
- C3: Fit results table export with %Enhancement and %NTE
- D1: Session loading error messages
- Item 1: Time axis — temporal resolution from DICOM metadata
- Item 7: Pixel value readout on parameter maps (hover)
- Item 8: Stride-based downsampling for parameter maps
"""

import csv
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="proxylfit_test_")
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_fit_results():
    """Realistic fit_results dict as produced by fit_proxyl_kinetics."""
    return {
        'kb': 0.98, 'kd': 0.035, 'knt': 0.008,
        'A0': 5200.0, 'A1': 2800.0, 'A2': -400.0,
        't0': 4.3, 'tmax': 30.0,
        'kb_error': 0.697, 'kd_error': 0.010, 'knt_error': 0.003,
        'A0_error': 48.5, 'A1_error': 185.3, 'A2_error': 95.0,
        't0_error': 1.52, 'tmax_error': 4.8,
        'r_squared': 0.76, 'rmse': 163.5,
        'residuals': np.random.normal(0, 120, 126),
        'time_units': 'minutes',
    }


# ===========================================================================
# A1: show_roi_cb → roi_checkbox
# ===========================================================================

class TestA1_RoiCheckboxAttribute:
    """A1: Verify parameter_map_options.py uses roi_checkbox, not show_roi_cb."""

    def test_no_show_roi_cb_references(self):
        """The string 'show_roi_cb' should not appear anywhere in the file."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "parameter_map_options.py"
        text = src.read_text()
        assert "show_roi_cb" not in text, (
            "Found 'show_roi_cb' in parameter_map_options.py — "
            "should be 'roi_checkbox'"
        )

    def test_roi_checkbox_used_in_export(self):
        """roi_checkbox should appear in the _export and _show_map_selection_dialog methods."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "parameter_map_options.py"
        text = src.read_text()
        assert text.count("self.roi_checkbox") >= 5, (
            "Expected at least 5 occurrences of self.roi_checkbox "
            "(definition + 4 export references)"
        )


# ===========================================================================
# A2: numpy int64 JSON serialization
# ===========================================================================

class TestA2_NumpyJsonSerialization:
    """A2: DICOM metadata with numpy types should serialize without error."""

    def test_numpy_int64_in_metadata(self, temp_dir):
        """save_parameter_map_as_dicom should handle numpy int64 in metadata."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        data = np.random.random((8, 8, 1)).astype(np.float32)
        metadata = {
            'window_x': np.int64(15),
            'window_y': np.int64(15),
            'window_z': np.int64(3),
            'success_rate': np.float64(98.5),
            'total_positions': np.int64(2282),
            'nested': {'value': np.int64(42)},
        }

        saved = save_parameter_map_as_dicom(
            data, 'kb_map', temp_dir, (1.0, 1.0, 2.0),
            source_dicom=None, metadata=metadata
        )
        assert len(saved) == 1

        # Verify the metadata was written and is valid JSON
        import pydicom
        ds = pydicom.dcmread(saved[0])
        parsed = json.loads(ds.ImageComments)
        assert parsed['window_x'] == 15
        assert isinstance(parsed['window_x'], int)
        assert parsed['nested']['value'] == 42

    def test_numpy_array_in_metadata(self, temp_dir):
        """numpy arrays in metadata should be converted to lists."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        data = np.random.random((8, 8, 1)).astype(np.float32)
        metadata = {
            'spacing': np.array([1.0, 1.0, 2.0]),
        }

        saved = save_parameter_map_as_dicom(
            data, 'kb_map', temp_dir, (1.0, 1.0, 2.0),
            source_dicom=None, metadata=metadata
        )

        import pydicom
        ds = pydicom.dcmread(saved[0])
        parsed = json.loads(ds.ImageComments)
        assert parsed['spacing'] == [1.0, 1.0, 2.0]

    def test_empty_metadata_no_crash(self, temp_dir):
        """Empty metadata dict should not crash."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        data = np.random.random((8, 8, 1)).astype(np.float32)
        saved = save_parameter_map_as_dicom(
            data, 'kb_map', temp_dir, (1.0, 1.0, 2.0),
            source_dicom=None, metadata={}
        )
        assert len(saved) == 1


# ===========================================================================
# A3: setQuitOnLastWindowClosed(False)
# ===========================================================================

class TestA3_QuitOnLastWindowClosed:
    """A3: init_qt_app should set quitOnLastWindowClosed to False."""

    def test_quit_on_last_window_closed_is_false(self):
        from proxyl_analysis.ui.styles import init_qt_app
        app = init_qt_app()
        assert app.quitOnLastWindowClosed() is False

    def test_init_qt_app_source_contains_setting(self):
        """Source code should contain setQuitOnLastWindowClosed(False)."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "styles.py"
        text = src.read_text()
        assert "setQuitOnLastWindowClosed(False)" in text


# ===========================================================================
# C1: Difference image labels and filenames
# ===========================================================================

class TestC1_DifferenceImageLabels:
    """C1: Simplified labels and B-minus-A filename ordering."""

    def test_region_labels_simplified(self):
        """Region group boxes should not contain color descriptions."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        # Old labels had "(Blue)" and "(Red)" in GroupBox titles
        assert 'Region A (Blue)' not in text
        assert 'Region B (Red)' not in text
        assert '"subtracted from B"' not in text

    def test_filename_b_minus_a_order(self):
        """Difference filename should put B range first (matches computation)."""
        from proxyl_analysis.ui.image_tools import ImageToolsDialog
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        # The filename template should be diff_t{b}...minus_t{a}...
        assert "f\"diff_t{b_start}-t{b_end}_minus_t{a_start}-t{a_end}\"" in text

    def test_instruction_text_formula(self):
        """Instructions should state Result = mean(Region B) - mean(Region A)."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        assert "mean(Region B)" in text
        assert "mean(Region A)" in text


# ===========================================================================
# C2: Grayscale colormap for difference images
# ===========================================================================

class TestC2_GrayscaleColormap:
    """C2: Difference images should use grayscale by default."""

    def test_no_rdbu_colormap(self):
        """RdBu_r should not be used for difference images."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        assert "RdBu_r" not in text

    def test_gray_colormap_for_difference(self):
        """_show_preview should use cmap='gray' for difference mode."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "image_tools.py"
        text = src.read_text()
        # Both average and difference modes should use gray
        # Count 'gray' occurrences in imshow calls
        assert text.count("cmap='gray'") >= 2, (
            "Expected cmap='gray' for both averaged and difference imshow"
        )


# ===========================================================================
# C3: Fit results table export
# ===========================================================================

class TestC3_FitResultsExport:
    """C3: %Enhancement, %NTE computation and CSV export."""

    def test_percent_enhancement_calculation(self, sample_fit_results):
        """(A1 / A0) * 100 should equal expected value."""
        A0, A1 = sample_fit_results['A0'], sample_fit_results['A1']
        expected = (A1 / A0) * 100  # (2800 / 5200) * 100 ≈ 53.85
        assert abs(expected - 53.846) < 0.01

    def test_percent_nte_calculation(self, sample_fit_results):
        """(A2 / A0) * 100 should equal expected value."""
        A0, A2 = sample_fit_results['A0'], sample_fit_results['A2']
        expected = (A2 / A0) * 100  # (-400 / 5200) * 100 ≈ -7.69
        assert abs(expected - (-7.692)) < 0.01

    def test_percent_enhancement_zero_baseline(self):
        """When A0=0, derived values should be NaN, not crash."""
        from proxyl_analysis.ui.fitting import FitResultsDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()
        time = np.linspace(0, 10, 20)
        signal = np.ones(20)
        fitted = np.ones(20)
        fr = {
            'kb': 0.1, 'kd': 0.01, 'knt': 0.001,
            'A0': 0.0, 'A1': 1.0, 'A2': 0.5,
            't0': 1.0, 'tmax': 5.0,
            'kb_error': 0.01, 'kd_error': 0.001, 'knt_error': 0.0001,
            'A0_error': 0.1, 'A1_error': 0.1, 'A2_error': 0.05,
            't0_error': 0.1, 'tmax_error': 0.5,
            'r_squared': 0.99, 'rmse': 0.01,
            'residuals': np.zeros(20),
            'time_units': 'minutes',
        }
        dialog = FitResultsDialog(time, signal, fitted, fr)
        assert np.isnan(dialog.pct_enhancement)
        assert np.isnan(dialog.pct_nte)
        dialog.close()

    def test_save_results_table_csv(self, temp_dir, sample_fit_results):
        """Save Results Table should produce valid CSV with all parameters."""
        from proxyl_analysis.ui.fitting import FitResultsDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()
        time = np.linspace(0, 70, 126)
        signal = np.ones(126) * 5000
        fitted = np.ones(126) * 5000

        dialog = FitResultsDialog(time, signal, fitted, sample_fit_results)

        csv_path = str(Path(temp_dir) / "results.csv")

        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV Files (*.csv)')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._save_results_table()

        # Verify CSV
        assert Path(csv_path).exists()
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['parameter', 'description', 'value', 'error', 'units']

            rows = {row[0]: row for row in reader}

        # Check key parameters present
        assert 'kb' in rows
        assert 'A0' in rows
        assert 'A2' in rows
        assert 'tmax' in rows
        assert '%Enhancement' in rows
        assert '%NTE' in rows
        assert 'R_squared' in rows
        assert 'RMSE' in rows

        # Verify values
        assert float(rows['kb'][2]) == pytest.approx(0.98)
        assert float(rows['A2'][2]) == pytest.approx(-400.0)
        assert float(rows['%Enhancement'][2]) == pytest.approx(53.846, rel=0.01)
        assert float(rows['%NTE'][2]) == pytest.approx(-7.692, rel=0.01)

        dialog.close()

    def test_fitting_dialog_shows_a2_and_tmax(self):
        """Params panel should include A2, tmax, and derived params."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "fitting.py"
        text = src.read_text()
        assert "A2 (non-tracer)" in text
        assert "tmax (NTE onset)" in text
        assert "%Enhancement" in text
        assert "%NTE" in text
        assert "Save Results Table" in text


# ===========================================================================
# D1: Session loading error messages
# ===========================================================================

class TestD1_SessionLoadingFeedback:
    """D1: Better error messages when loading invalid sessions."""

    def test_invalid_session_message_includes_tip(self):
        """Error message should include helpful tip about output folder."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "main_menu.py"
        text = src.read_text()
        assert "output folder" in text.lower() or "originally ran registration" in text.lower()

    def test_load_previous_detects_child_sessions(self, temp_dir):
        """When user selects parent of a valid session, hint about it."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()

        # Create a valid session structure inside a child folder
        child = Path(temp_dir) / "my_session"
        dicoms = child / "registered" / "dicoms"
        dicoms.mkdir(parents=True)
        (dicoms / "z00_t000.dcm").touch()

        dialog = MainMenuDialog()

        # Mock the file dialog to return the parent (not the child)
        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=temp_dir):
            with patch('PySide6.QtWidgets.QMessageBox.warning') as mock_warn:
                dialog._load_previous_session()

                # Should show warning with hint about child folder
                mock_warn.assert_called_once()
                msg = mock_warn.call_args[0][2]  # 3rd positional arg is the message text
                assert "my_session" in msg

        dialog.close()

    def test_load_valid_session_accepts(self, temp_dir):
        """When user selects a valid session folder, dialog should accept."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()

        # Create valid session structure
        dicoms = Path(temp_dir) / "registered" / "dicoms"
        dicoms.mkdir(parents=True)
        (dicoms / "z00_t000.dcm").touch()

        dialog = MainMenuDialog()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=temp_dir):
            dialog._load_previous_session()

        assert dialog.result is not None
        assert dialog.result['action'] == 'load_previous'
        assert dialog.result['session_path'] == temp_dir

        dialog.close()

    def test_load_empty_folder_shows_warning(self, temp_dir):
        """Selecting a folder with no registration data should warn."""
        from proxyl_analysis.ui.main_menu import MainMenuDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()
        dialog = MainMenuDialog()

        with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                   return_value=temp_dir):
            with patch('PySide6.QtWidgets.QMessageBox.warning') as mock_warn:
                dialog._load_previous_session()
                mock_warn.assert_called_once()

        dialog.close()


# ===========================================================================
# Item 1: Time axis — temporal resolution from DICOM metadata
# ===========================================================================

class TestTimeAxis:
    """Item 1: Time axis should use ~33s intervals (not 70s)."""

    def test_extract_temporal_resolution_from_real_dicom(self):
        """extract_temporal_resolution returns ~33.4s on a real source DICOM."""
        import json

        # Find source DICOM path from a real session's registration metadata
        candidates = [
            Path(__file__).parent.parent / "output" / "35354296",
            Path(__file__).parent.parent / "output" / "35354281",
            Path(__file__).parent.parent / "output" / "35354333",
        ]
        dicom_path = None
        for session in candidates:
            metrics_file = session / "registered" / "registration_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    meta = json.load(f).get("metadata", {})
                src = meta.get("dicom_path")
                if src and Path(src).exists():
                    dicom_path = src
                    break

        if dicom_path is None:
            pytest.skip("No real source DICOM available")

        from proxyl_analysis.io import extract_temporal_resolution

        result = extract_temporal_resolution(dicom_path)
        assert result is not None, "Failed to extract temporal resolution"
        assert result == pytest.approx(33.408, abs=0.1), (
            f"Expected ~33.4s per timepoint, got {result}"
        )

    def test_create_time_array_default_is_33s(self):
        """Default interval should be 33.0s, not 70s."""
        from proxyl_analysis.run_analysis import create_time_array

        arr = create_time_array(10, time_units='seconds')
        expected = np.arange(10, dtype=float) * 33.0
        np.testing.assert_allclose(arr, expected)

    def test_create_time_array_with_custom_resolution(self):
        """Custom temporal_resolution_s parameter should be used."""
        from proxyl_analysis.run_analysis import create_time_array

        arr = create_time_array(5, time_units='seconds', temporal_resolution_s=45.0)
        expected = np.arange(5, dtype=float) * 45.0
        np.testing.assert_allclose(arr, expected)

    def test_create_time_array_minutes_conversion(self):
        """Minutes array should equal seconds array / 60."""
        from proxyl_analysis.run_analysis import create_time_array

        arr_sec = create_time_array(10, time_units='seconds', temporal_resolution_s=33.408)
        arr_min = create_time_array(10, time_units='minutes', temporal_resolution_s=33.408)
        np.testing.assert_allclose(arr_min, arr_sec / 60.0)


# ===========================================================================
# Item 7: Pixel value readout on parameter maps
# ===========================================================================

class TestPixelValueReadout:
    """Item 7: Hover over parameter maps to read pixel values."""

    @pytest.fixture
    def results_dialog(self):
        """Create a ParameterMapResultsDialog with known synthetic data."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()

        # 10x10x3 maps with known values
        kb_map = np.full((10, 10, 3), np.nan)
        kb_map[5, 7, 1] = 0.4523
        kb_map[3, 2, 1] = 1.2345

        mask = np.zeros((10, 10, 3), dtype=bool)
        mask[5, 7, 1] = True
        mask[3, 2, 1] = True

        param_maps = {
            'kb_map': kb_map,
            'kd_map': np.full((10, 10, 3), np.nan),
            'knt_map': np.full((10, 10, 3), np.nan),
            'r_squared_map': np.full((10, 10, 3), np.nan),
            'a1_amplitude_map': np.full((10, 10, 3), np.nan),
            'a2_amplitude_map': np.full((10, 10, 3), np.nan),
            'mask': mask,
        }

        dialog = ParameterMapResultsDialog(
            param_maps=param_maps,
            spacing=(1.0, 1.0, 2.0),
        )
        # Navigate to z=1 where we placed known values
        dialog.z_slider.setValue(1)

        yield dialog
        dialog.close()

    def test_pixel_label_exists(self, results_dialog):
        """ParameterMapResultsDialog should have a pixel_label QLabel."""
        from PySide6.QtWidgets import QLabel

        assert hasattr(results_dialog, 'pixel_label')
        assert isinstance(results_dialog.pixel_label, QLabel)

    def test_hover_updates_label(self, results_dialog):
        """Simulating a motion event over a known pixel updates the label."""
        # Create a mock event at pixel (5, 7) — where kb = 0.4523
        event = MagicMock()
        event.inaxes = results_dialog.ax
        event.xdata = 5.0
        event.ydata = 7.0

        results_dialog._on_pixel_hover(event)

        text = results_dialog.pixel_label.text()
        assert "(5, 7)" in text
        assert "0.4523" in text
        assert "kb" in text

    def test_hover_outside_shows_dash(self, results_dialog):
        """Motion event outside axes resets label to 'Pixel: —'."""
        # First hover over a valid pixel to change the label from its default
        valid_event = MagicMock()
        valid_event.inaxes = results_dialog.ax
        valid_event.xdata = 5.0
        valid_event.ydata = 7.0
        results_dialog._on_pixel_hover(valid_event)
        assert "0.4523" in results_dialog.pixel_label.text()

        # Now hover outside — label should reset
        event = MagicMock()
        event.inaxes = None  # Outside any axes

        results_dialog._on_pixel_hover(event)

        assert results_dialog.pixel_label.text() == "Pixel: \u2014"

    def test_hover_nan_pixel_shows_dash(self, results_dialog):
        """Hovering over a NaN (unfitted) pixel shows '—' for the value."""
        # Pixel (0, 0) at z=1 is NaN in our test data
        event = MagicMock()
        event.inaxes = results_dialog.ax
        event.xdata = 0.0
        event.ydata = 0.0

        results_dialog._on_pixel_hover(event)

        text = results_dialog.pixel_label.text()
        assert "(0, 0)" in text
        assert "\u2014" in text


# ===========================================================================
# Item 8: Stride-based downsampling for parameter maps
# ===========================================================================

class TestParameterMapStride:
    """Item 8: Stride parameter for coarser-resolution parameter mapping."""

    @pytest.fixture
    def synthetic_4d(self):
        """Create small synthetic 4D data with a clear signal."""
        np.random.seed(42)
        # 10x10x1x20 — small enough to fit quickly
        data = np.random.uniform(500, 1000, (10, 10, 1, 20)).astype(np.float64)
        # Add a strong ramp so signal passes threshold checks
        for t in range(20):
            data[:, :, :, t] += t * 100
        return data

    @pytest.fixture
    def time_array(self):
        return np.linspace(0, 10, 20)

    def test_stride_output_same_shape(self, synthetic_4d, time_array):
        """stride=2 on 10×10 input still produces 10×10 output maps."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=2
        )
        assert result['kb_map'].shape == (10, 10, 1)
        assert result['mask'].shape == (10, 10, 1)

    def test_stride_fills_blocks(self, synthetic_4d, time_array):
        """Fitted value at strided position fills surrounding stride×stride block."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=2
        )
        kb = result['kb_map'][:, :, 0]
        mask = result['mask'][:, :, 0]

        # For every fitted 2×2 block, all values within should be identical
        for x in range(0, 10, 2):
            for y in range(0, 10, 2):
                block = kb[x:x+2, y:y+2]
                if mask[x, y]:
                    # All values in block should equal the corner value
                    assert np.all(block == block[0, 0]), (
                        f"Block at ({x},{y}) not filled uniformly: {block}"
                    )

    def test_stride_1_unchanged(self, synthetic_4d, time_array):
        """stride=1 produces identical results to default behavior."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result_default = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0
        )
        result_stride1 = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=1
        )
        np.testing.assert_array_equal(result_default['kb_map'], result_stride1['kb_map'])
        np.testing.assert_array_equal(result_default['mask'], result_stride1['mask'])

    def test_stride_in_options_dialog(self):
        """Dialog result dict includes 'stride' key."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "ui" / "parameter_map_options.py"
        text = src.read_text()
        assert "'stride'" in text, "Expected 'stride' key in options dialog result"
        assert "stride_spin" in text, "Expected stride_spin widget in dialog"

    def test_stride_metadata_recorded(self, synthetic_4d, time_array):
        """Output metadata contains stride value."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=4
        )
        assert 'stride' in result['metadata']
        assert result['metadata']['stride'] == 4

    def test_stride_reduces_total_positions(self, synthetic_4d, time_array):
        """stride=2 on 10×10 should report ~25 positions, not 100."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        positions_log = []

        def capture_progress(pct, current, total):
            if not positions_log or positions_log[-1] != total:
                positions_log.append(total)
            return True

        create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=1,
            progress_callback=capture_progress
        )
        total_stride1 = positions_log[-1]

        positions_log.clear()
        create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=2,
            progress_callback=capture_progress
        )
        total_stride2 = positions_log[-1]

        # stride=2 should have ~4x fewer positions
        assert total_stride1 == 100  # 10×10
        assert total_stride2 == 25   # 5×5

    def test_stride_non_divisible(self):
        """stride=3 on 10×10 handles partial edge blocks correctly."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        np.random.seed(99)
        data = np.random.uniform(500, 1000, (10, 10, 1, 20)).astype(np.float64)
        for t in range(20):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 10, 20)

        result = create_parameter_maps(
            data, time, window_size=1, z_slice=0, stride=3
        )
        kb = result['kb_map'][:, :, 0]
        mask = result['mask'][:, :, 0]

        # Output shape unchanged
        assert result['kb_map'].shape == (10, 10, 1)

        # Check that partial edge block at x=9 is handled (block 9:10 is 1 wide)
        if mask[9, 0]:
            assert not np.isnan(kb[9, 0])

        # Check block at x=6 fills 6:9 (3 wide)
        if mask[6, 0]:
            block = kb[6:9, 0:3]
            assert np.all(block == block[0, 0])

    def test_stride_larger_than_image(self):
        """stride > image dimension processes a single position per slice."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        np.random.seed(77)
        data = np.random.uniform(500, 1000, (6, 6, 1, 20)).astype(np.float64)
        for t in range(20):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 10, 20)

        result = create_parameter_maps(
            data, time, window_size=1, z_slice=0, stride=20
        )
        # Should still produce full-size output
        assert result['kb_map'].shape == (6, 6, 1)
        # Only one position fitted (0,0), so if successful, entire map filled
        assert result['metadata']['total_positions'] == 1

    def test_stride_non_square_image(self):
        """stride works correctly on non-square images (e.g. 12×8)."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        np.random.seed(55)
        data = np.random.uniform(500, 1000, (12, 8, 1, 20)).astype(np.float64)
        for t in range(20):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 10, 20)

        result = create_parameter_maps(
            data, time, window_size=1, z_slice=0, stride=4
        )
        assert result['kb_map'].shape == (12, 8, 1)
        # 12/4=3 positions in x, 8/4=2 positions in y → 6 total
        assert result['metadata']['total_positions'] == 6

    def test_stride_with_roi_mask(self):
        """stride correctly interacts with ROI mask."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        np.random.seed(33)
        data = np.random.uniform(500, 1000, (10, 10, 1, 20)).astype(np.float64)
        for t in range(20):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 10, 20)

        # ROI mask covers only top-left 6×6 region
        roi_mask = np.zeros((10, 10), dtype=bool)
        roi_mask[:6, :6] = True

        result = create_parameter_maps(
            data, time, window_size=1, z_slice=0,
            stride=2, roi_mask=roi_mask
        )
        # Output shape stays full-size
        assert result['kb_map'].shape == (10, 10, 1)
        # stride=2 on ROI: positions (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4) = 9
        assert result['metadata']['total_positions'] == 9


# ===========================================================================
# Item 8 (continued): Stride worker and integration tests
# ===========================================================================

class TestParameterMapStrideWorker:
    """Item 8: Verify ParameterMappingWorker passes stride correctly."""

    def test_worker_passes_stride_to_create_parameter_maps(self):
        """Worker forwards stride from options dict to create_parameter_maps()."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMappingWorker

        np.random.seed(42)
        data = np.random.uniform(500, 1000, (6, 6, 1, 10)).astype(np.float64)
        for t in range(10):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 5, 10)

        options = {
            'window_size': (1, 1, 1),
            'z_slice': 0,
            'kernel_type': 'sliding_window',
            'stride': 3,
        }

        captured_kwargs = {}

        def mock_create(*args, **kwargs):
            captured_kwargs.update(kwargs)
            # Return a minimal valid result
            shape = (6, 6, 1)
            return {
                'kb_map': np.zeros(shape),
                'kd_map': np.zeros(shape),
                'knt_map': np.zeros(shape),
                'r_squared_map': np.zeros(shape),
                'a1_amplitude_map': np.zeros(shape),
                'a2_amplitude_map': np.zeros(shape),
                'baseline_map': np.zeros(shape),
                't0_map': np.zeros(shape),
                'tmax_map': np.zeros(shape),
                'mask': np.zeros(shape, dtype=bool),
                'metadata': {'stride': 3},
            }

        with patch('proxyl_analysis.parameter_mapping.create_parameter_maps', mock_create):
            worker = ParameterMappingWorker(data, time, options)
            worker.run()

        assert captured_kwargs.get('stride') == 3

    def test_worker_defaults_stride_to_1(self):
        """Worker defaults stride to 1 when not in options dict."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMappingWorker

        np.random.seed(42)
        data = np.random.uniform(500, 1000, (4, 4, 1, 10)).astype(np.float64)
        for t in range(10):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 5, 10)

        # Options WITHOUT stride key
        options = {
            'window_size': (1, 1, 1),
            'z_slice': 0,
            'kernel_type': 'sliding_window',
        }

        captured_kwargs = {}

        def mock_create(*args, **kwargs):
            captured_kwargs.update(kwargs)
            shape = (4, 4, 1)
            return {
                'kb_map': np.zeros(shape),
                'kd_map': np.zeros(shape),
                'knt_map': np.zeros(shape),
                'r_squared_map': np.zeros(shape),
                'a1_amplitude_map': np.zeros(shape),
                'a2_amplitude_map': np.zeros(shape),
                'baseline_map': np.zeros(shape),
                't0_map': np.zeros(shape),
                'tmax_map': np.zeros(shape),
                'mask': np.zeros(shape, dtype=bool),
                'metadata': {'stride': 1},
            }

        with patch('proxyl_analysis.parameter_mapping.create_parameter_maps', mock_create):
            worker = ParameterMappingWorker(data, time, options)
            worker.run()

        assert captured_kwargs.get('stride') == 1


class TestParameterMapStrideIntegration:
    """Item 8: Integration tests for stride save/load roundtrip."""

    @pytest.fixture
    def stride_param_maps(self):
        """Create parameter maps with stride=4 via actual create_parameter_maps."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        np.random.seed(42)
        data = np.random.uniform(500, 1000, (10, 10, 1, 20)).astype(np.float64)
        for t in range(20):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 10, 20)

        roi_mask = np.ones((10, 10), dtype=bool)

        result = create_parameter_maps(
            data, time, window_size=1, z_slice=0, stride=4,
            roi_mask=roi_mask
        )
        return result

    def test_stride_npz_roundtrip(self, temp_dir, stride_param_maps):
        """Stride metadata survives NPZ save → load cycle."""
        from proxyl_analysis.parameter_mapping import save_parameter_maps, load_parameter_maps

        out_dir = str(Path(temp_dir) / "stride_maps")
        save_parameter_maps(stride_param_maps, (1.0, 1.0, 2.0), out_dir)
        loaded, _ = load_parameter_maps(out_dir)

        assert 'metadata' in loaded
        assert loaded['metadata']['stride'] == 4

    def test_stride_json_metadata_roundtrip(self, temp_dir, stride_param_maps):
        """Stride value is in JSON metadata file and is a native int."""
        from proxyl_analysis.parameter_mapping import save_parameter_maps

        out_dir = str(Path(temp_dir) / "stride_maps")
        save_parameter_maps(stride_param_maps, (1.0, 1.0, 2.0), out_dir)

        meta_path = Path(out_dir) / "parameter_maps_metadata.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        assert 'stride' in metadata
        assert metadata['stride'] == 4
        assert isinstance(metadata['stride'], int)

    def test_stride_dicom_export_metadata(self, temp_dir, stride_param_maps):
        """Stride value survives DICOM export in ImageComments JSON."""
        from proxyl_analysis.io import save_parameter_map_as_dicom
        import pydicom

        saved = save_parameter_map_as_dicom(
            stride_param_maps['kb_map'], 'kb_map', temp_dir,
            (1.0, 1.0, 2.0),
            source_dicom=None,
            metadata=stride_param_maps['metadata']
        )
        assert len(saved) >= 1

        ds = pydicom.dcmread(saved[0])
        parsed = json.loads(ds.ImageComments)
        assert parsed['stride'] == 4

    def test_stride_map_values_preserved_through_save_load(self, temp_dir, stride_param_maps):
        """Map values (including block-filled patterns) survive save/load."""
        from proxyl_analysis.parameter_mapping import save_parameter_maps, load_parameter_maps

        out_dir = str(Path(temp_dir) / "stride_maps")
        save_parameter_maps(stride_param_maps, (1.0, 1.0, 2.0), out_dir)
        loaded, _ = load_parameter_maps(out_dir)

        np.testing.assert_array_equal(
            loaded['kb_map'], stride_param_maps['kb_map'],
            err_msg="kb_map not preserved through save/load with stride"
        )
        np.testing.assert_array_equal(
            loaded['mask'], stride_param_maps['mask'],
            err_msg="mask not preserved through save/load with stride"
        )

    def test_stride_with_roi_roundtrip(self, temp_dir):
        """Stride + ROI mask combined roundtrip."""
        from proxyl_analysis.parameter_mapping import (
            create_parameter_maps, save_parameter_maps, load_parameter_maps
        )

        np.random.seed(42)
        data = np.random.uniform(500, 1000, (10, 10, 1, 20)).astype(np.float64)
        for t in range(20):
            data[:, :, :, t] += t * 100
        time = np.linspace(0, 10, 20)

        roi_mask = np.zeros((10, 10), dtype=bool)
        roi_mask[2:8, 2:8] = True

        result = create_parameter_maps(
            data, time, window_size=1, z_slice=0,
            stride=2, roi_mask=roi_mask
        )

        out_dir = str(Path(temp_dir) / "stride_roi_maps")
        save_parameter_maps(result, (1.0, 1.0, 2.0), out_dir)
        loaded, _ = load_parameter_maps(out_dir)

        assert loaded['metadata']['stride'] == 2
        assert 'roi_mask' in loaded
        np.testing.assert_array_equal(loaded['roi_mask'], roi_mask)
        np.testing.assert_array_equal(loaded['kb_map'], result['kb_map'])


class TestParameterMapStrideCLI:
    """Item 8: CLI --stride argument is defined and wired correctly."""

    def test_stride_argument_in_source(self):
        """run_analysis.py defines --stride argument with default=1."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "run_analysis.py"
        text = src.read_text()
        assert "'--stride'" in text
        assert "default=1" in text.split("'--stride'")[1].split("add_argument")[0]

    def test_stride_passed_to_create_parameter_maps_call(self):
        """Batch mode passes stride=args.stride to create_parameter_maps()."""
        src = Path(__file__).parent.parent / "proxyl_analysis" / "run_analysis.py"
        text = src.read_text()
        assert "stride=args.stride" in text
