"""
Tests for user feedback fixes (Plans A, C, D).

Validates:
- A1: show_roi_cb → roi_checkbox fix in parameter map export
- A2: numpy int64 JSON serialization in DICOM metadata
- A3: setQuitOnLastWindowClosed(False) in init_qt_app
- C1: Simplified difference image labels and filenames
- C2: Grayscale colormap for difference images
- C3: Fit results table export with %Enhancement and %NTE
- D1: Session loading error messages
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
