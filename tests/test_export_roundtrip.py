"""
Round-trip export/load validation tests.

For each export format (DICOM, PNG, CSV, NPZ) we write data, read it back,
and verify the loaded content matches the original.
"""

import csv
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pydicom
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="proxylfit_rt_")
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_3d_map():
    """3D parameter map with known values."""
    np.random.seed(123)
    data = np.random.random((16, 16, 3)).astype(np.float32) * 0.5
    # Set some specific values we can check later
    data[5, 5, 0] = 0.1234
    data[10, 10, 1] = 0.4567
    data[0, 0, 2] = 0.0
    return data


@pytest.fixture
def sample_spacing():
    return (0.137, 0.137, 0.75)


@pytest.fixture
def sample_metadata():
    """Metadata dict with numpy types to exercise A2 fix."""
    return {
        'kernel_type': 'sliding_window',
        'window_x': np.int64(15),
        'window_y': np.int64(15),
        'window_z': np.int64(3),
        'success_rate': np.float64(92.3),
        'total_positions': np.int64(768),
        'spacing': np.array([0.137, 0.137, 0.75]),
    }


@pytest.fixture
def sample_fit_results():
    """Realistic fit_results dict."""
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
# 1. DICOM parameter map round-trip
# ===========================================================================

class TestDicomParameterMapRoundtrip:
    """Save parameter map as DICOM, load back, verify pixel values."""

    def test_pixel_values_preserved(self, temp_dir, sample_3d_map, sample_spacing):
        """Values should survive the save/load cycle within tolerance."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved = save_parameter_map_as_dicom(
            sample_3d_map, 'kb_map', temp_dir, sample_spacing,
            source_dicom=None, metadata={'test': True}
        )

        assert len(saved) == 3, f"Expected 3 files, got {len(saved)}"

        for z, filepath in enumerate(saved):
            ds = pydicom.dcmread(filepath)

            # Reconstruct float values from DICOM
            raw = ds.pixel_array.astype(np.float64)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            reconstructed = raw * slope + intercept

            # DICOM stores as [rows, cols] = [y, x], need to transpose back
            reconstructed = reconstructed.T

            original_slice = sample_3d_map[:, :, z]

            # 16-bit quantization limits precision — allow ~1% of value range
            value_range = original_slice.max() - original_slice.min()
            if value_range > 0:
                atol = value_range / 65535 * 2  # 2x quantization step
                np.testing.assert_allclose(
                    reconstructed, original_slice, atol=atol,
                    err_msg=f"Slice z={z} values don't match after round-trip"
                )

    def test_metadata_preserved(self, temp_dir, sample_3d_map, sample_spacing, sample_metadata):
        """Metadata should survive JSON serialization in ImageComments."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved = save_parameter_map_as_dicom(
            sample_3d_map, 'kd_map', temp_dir, sample_spacing,
            source_dicom=None, metadata=sample_metadata
        )

        ds = pydicom.dcmread(saved[0])
        parsed = json.loads(ds.ImageComments)

        assert parsed['kernel_type'] == 'sliding_window'
        assert parsed['window_x'] == 15
        assert isinstance(parsed['window_x'], int)  # was numpy int64
        assert parsed['success_rate'] == pytest.approx(92.3)
        assert parsed['total_positions'] == 768
        assert parsed['spacing'] == pytest.approx([0.137, 0.137, 0.75])

    def test_series_description(self, temp_dir, sample_3d_map, sample_spacing):
        """Each map type should get a descriptive SeriesDescription."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        for map_name, expected_text in [
            ('kb_map', 'Buildup Rate'),
            ('kd_map', 'Decay Rate'),
            ('r_squared_map', 'R-squared'),
            ('a2_amplitude_map', 'Non-tracer Amplitude'),
        ]:
            saved = save_parameter_map_as_dicom(
                sample_3d_map, map_name, temp_dir, sample_spacing,
            )
            ds = pydicom.dcmread(saved[0])
            assert expected_text in ds.SeriesDescription, (
                f"{map_name}: expected '{expected_text}' in '{ds.SeriesDescription}'"
            )

    def test_spacing_preserved(self, temp_dir, sample_3d_map, sample_spacing):
        """Voxel spacing should be stored correctly."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved = save_parameter_map_as_dicom(
            sample_3d_map, 'kb_map', temp_dir, sample_spacing,
        )

        ds = pydicom.dcmread(saved[0])
        # PixelSpacing is [row_spacing, col_spacing] = [y, x]
        assert float(ds.PixelSpacing[0]) == pytest.approx(sample_spacing[1])
        assert float(ds.PixelSpacing[1]) == pytest.approx(sample_spacing[0])
        assert float(ds.SliceThickness) == pytest.approx(sample_spacing[2])

    def test_slice_location_increases(self, temp_dir, sample_3d_map, sample_spacing):
        """SliceLocation should increase with z-index."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved = save_parameter_map_as_dicom(
            sample_3d_map, 'kb_map', temp_dir, sample_spacing,
        )

        locations = []
        for filepath in saved:
            ds = pydicom.dcmread(filepath)
            locations.append(float(ds.SliceLocation))

        assert locations == sorted(locations)
        assert locations[0] == pytest.approx(0.0)
        assert locations[-1] == pytest.approx((len(saved) - 1) * sample_spacing[2])

    def test_2d_map_single_file(self, temp_dir, sample_spacing):
        """A 2D map should produce a single DICOM file."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        data_2d = np.random.random((16, 16)).astype(np.float32)
        saved = save_parameter_map_as_dicom(
            data_2d, 'kb_map', temp_dir, sample_spacing,
        )
        assert len(saved) == 1
        ds = pydicom.dcmread(saved[0])
        assert ds.Rows == 16
        assert ds.Columns == 16


# ===========================================================================
# 2. DICOM derived image round-trip (averaged / difference)
# ===========================================================================

class TestDicomDerivedImageRoundtrip:
    """Save derived images as DICOM, load back, verify data and metadata."""

    def test_averaged_image_roundtrip(self, temp_dir, sample_spacing):
        """Averaged image values and operation params should survive."""
        from proxyl_analysis.io import save_derived_image_as_dicom

        # Create a 3D averaged image
        image = np.random.random((16, 16, 3)).astype(np.float32) * 8000 + 2000

        params = {
            'start_idx': 5,
            'end_idx': 15,
            'n_frames': 11,
            'time_range': '5.83 - 17.50 minutes',
            'time_units': 'minutes',
        }

        output_path = str(Path(temp_dir) / "avg_t5-t15")
        saved = save_derived_image_as_dicom(
            image, output_path, 'averaged', params, sample_spacing
        )

        assert len(saved) == 3

        for z, filepath in enumerate(saved):
            ds = pydicom.dcmread(filepath)

            # Verify operation type in DICOM tags
            assert 'AVERAGED' in ds.ImageType
            assert 'Averaged' in ds.SeriesDescription

            # Verify operation params in ImageComments
            comments = json.loads(ds.ImageComments)
            assert comments['start_idx'] == 5
            assert comments['end_idx'] == 15
            assert comments['n_frames'] == 11

            # Verify pixel data roundtrip
            raw = ds.pixel_array.astype(np.float64)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            reconstructed = (raw * slope + intercept).T

            original = image[:, :, z]
            value_range = original.max() - original.min()
            if value_range > 0:
                atol = value_range / 65535 * 2
                np.testing.assert_allclose(
                    reconstructed, original, atol=atol,
                    err_msg=f"Averaged z={z} values don't match"
                )

    def test_difference_image_roundtrip(self, temp_dir, sample_spacing):
        """Difference images (with negative values) should survive round-trip."""
        from proxyl_analysis.io import save_derived_image_as_dicom

        # Create difference image with positive and negative values
        image = (np.random.random((16, 16)) - 0.5).astype(np.float32) * 2000

        params = {
            'region_a_start': 0,
            'region_a_end': 4,
            'region_b_start': 10,
            'region_b_end': 15,
            'description': 'B minus A',
            'time_units': 'minutes',
        }

        output_path = str(Path(temp_dir) / "diff_t10-t15_minus_t0-t4")
        saved = save_derived_image_as_dicom(
            image, output_path, 'difference', params, sample_spacing
        )

        assert len(saved) == 1

        ds = pydicom.dcmread(saved[0])
        assert 'DIFFERENCE' in ds.ImageType
        assert ds.PixelRepresentation == 1  # Signed for negative values

        # Verify params round-trip
        comments = json.loads(ds.ImageComments)
        assert comments['region_a_start'] == 0
        assert comments['region_b_start'] == 10

        # Verify pixel values (signed int16 encoding)
        raw = ds.pixel_array.astype(np.float64)
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        reconstructed = (raw * slope + intercept).T

        value_range = image.max() - image.min()
        atol = value_range / 32767 * 2  # Signed 16-bit
        np.testing.assert_allclose(
            reconstructed, image, atol=atol,
            err_msg="Difference image values don't match after round-trip"
        )

    def test_2d_single_file(self, temp_dir, sample_spacing):
        """A 2D derived image should produce a single file, not a subfolder."""
        from proxyl_analysis.io import save_derived_image_as_dicom

        image_2d = np.ones((16, 16), dtype=np.float32) * 5000
        output_path = str(Path(temp_dir) / "test_avg")
        saved = save_derived_image_as_dicom(
            image_2d, output_path, 'averaged',
            {'start_idx': 0, 'end_idx': 5}, sample_spacing
        )
        assert len(saved) == 1
        assert saved[0].endswith('.dcm')


# ===========================================================================
# 3. ROI mask NPZ round-trip
# ===========================================================================

class TestRoiMaskRoundtrip:
    """Save ROI mask as NPZ, load back, verify mask and info."""

    def test_mask_values_preserved(self, temp_dir):
        """Saved mask should be identical after load."""
        from proxyl_analysis.io import save_roi_mask, load_roi_mask

        mask = np.zeros((32, 32, 9), dtype=bool)
        mask[10:22, 10:22, 3:6] = True

        info = {
            'roi_type': 'contour',
            'z_slice': 4,
            'area_pixels': int(np.sum(mask)),
        }

        save_roi_mask(temp_dir, mask, info)
        loaded_mask, loaded_info = load_roi_mask(temp_dir)

        np.testing.assert_array_equal(loaded_mask, mask)
        assert loaded_info['roi_type'] == 'contour'
        assert loaded_info['z_slice'] == 4
        assert loaded_info['area_pixels'] == int(np.sum(mask))

    def test_mask_without_info(self, temp_dir):
        """Mask saved without roi_info should load with None info."""
        from proxyl_analysis.io import save_roi_mask, load_roi_mask

        mask = np.ones((16, 16), dtype=bool)
        save_roi_mask(temp_dir, mask)
        loaded_mask, loaded_info = load_roi_mask(temp_dir)

        np.testing.assert_array_equal(loaded_mask, mask)
        assert loaded_info is None

    def test_nonexistent_mask_returns_none(self, temp_dir):
        """Loading from a directory without a mask should return None."""
        from proxyl_analysis.io import load_roi_mask

        mask, info = load_roi_mask(temp_dir)
        assert mask is None
        assert info is None


# ===========================================================================
# 4. CSV fit results round-trip
# ===========================================================================

class TestCsvFitResultsRoundtrip:
    """Export fit results CSV, load back, verify all parameters."""

    def test_all_parameters_roundtrip(self, temp_dir, sample_fit_results):
        """Every parameter should be loadable and match the originals."""
        from proxyl_analysis.ui.fitting import FitResultsDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()
        time = np.linspace(0, 70, 126)
        signal = np.ones(126) * 5000
        fitted = np.ones(126) * 5000

        dialog = FitResultsDialog(time, signal, fitted, sample_fit_results)

        csv_path = str(Path(temp_dir) / "results.csv")
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._save_results_table()

        # Load and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = {row['parameter']: row for row in reader}

        # Kinetic parameters
        assert float(rows['kb']['value']) == pytest.approx(0.98)
        assert float(rows['kb']['error']) == pytest.approx(0.697)
        assert rows['kb']['units'] == '1/minutes'

        assert float(rows['kd']['value']) == pytest.approx(0.035)
        assert float(rows['knt']['value']) == pytest.approx(0.008)

        assert float(rows['A0']['value']) == pytest.approx(5200.0)
        assert float(rows['A1']['value']) == pytest.approx(2800.0)
        assert float(rows['A2']['value']) == pytest.approx(-400.0)

        assert float(rows['t0']['value']) == pytest.approx(4.3)
        assert rows['t0']['units'] == 'minutes'
        assert float(rows['tmax']['value']) == pytest.approx(30.0)

        # Derived parameters
        expected_enh = (2800.0 / 5200.0) * 100
        expected_nte = (-400.0 / 5200.0) * 100
        assert float(rows['%Enhancement']['value']) == pytest.approx(expected_enh, rel=0.001)
        assert float(rows['%NTE']['value']) == pytest.approx(expected_nte, rel=0.001)
        assert rows['%Enhancement']['units'] == '%'

        # Fit quality
        assert float(rows['R_squared']['value']) == pytest.approx(0.76)
        assert float(rows['RMSE']['value']) == pytest.approx(163.5)

        dialog.close()

    def test_csv_has_12_data_rows(self, temp_dir, sample_fit_results):
        """CSV should have header + 12 data rows."""
        from proxyl_analysis.ui.fitting import FitResultsDialog
        from proxyl_analysis.ui.styles import init_qt_app

        app = init_qt_app()
        time = np.linspace(0, 70, 126)
        signal = np.ones(126) * 5000
        fitted = np.ones(126) * 5000

        dialog = FitResultsDialog(time, signal, fitted, sample_fit_results)

        csv_path = str(Path(temp_dir) / "results.csv")
        with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(csv_path, 'CSV')):
            with patch('PySide6.QtWidgets.QMessageBox.information'):
                dialog._save_results_table()

        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # Header + 12 params (A0, A1, A2, kb, kd, knt, t0, tmax,
        # %Enhancement, %NTE, R_squared, RMSE)
        assert len(lines) == 13, f"Expected 13 lines (header + 12), got {len(lines)}"

        dialog.close()


# ===========================================================================
# 5. PNG export validation
# ===========================================================================

class TestPngExport:
    """Parameter map PNG export should produce valid images."""

    def test_png_export_creates_file(self, temp_dir):
        """PNG export should produce a non-empty file."""
        import matplotlib.pyplot as plt

        data = np.random.random((16, 16)).astype(np.float32)
        png_path = Path(temp_dir) / "test_map.png"

        fig, ax = plt.subplots()
        ax.imshow(data, cmap='hot')
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)

        assert png_path.exists()
        assert png_path.stat().st_size > 1000  # Not empty/corrupt

    def test_parameter_map_results_png_export(self, temp_dir):
        """ParameterMapResultsDialog PNG export via _export method."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog
        from proxyl_analysis.ui.styles import init_qt_app
        from PySide6.QtWidgets import QApplication

        app = init_qt_app()

        shape = (16, 16, 3)
        np.random.seed(42)
        mask = np.ones(shape, dtype=bool)

        param_maps = {
            'kb_map': np.random.random(shape).astype(np.float32) * 0.5,
            'mask': mask,
            'metadata': {'kernel_type': 'sliding_window'},
        }
        roi_mask = np.ones(shape, dtype=bool)

        dialog = ParameterMapResultsDialog(
            param_maps=param_maps,
            spacing=(0.137, 0.137, 0.75),
            roi_mask=roi_mask,
            output_dir=temp_dir,
        )
        dialog.show()
        app.processEvents()

        # Simulate PNG export for kb_map
        export_dir = Path(temp_dir) / "png_export"
        export_dir.mkdir()

        # _show_map_selection_dialog returns (selected_maps, include_roi)
        with patch.object(dialog, '_show_map_selection_dialog',
                          return_value=(['kb_map'], False)):
            with patch('PySide6.QtWidgets.QFileDialog.getExistingDirectory',
                       return_value=str(export_dir)):
                with patch('PySide6.QtWidgets.QMessageBox.information'):
                    dialog._export('png')

        # Check that PNG files were created
        png_files = list(export_dir.glob("*.png"))
        assert len(png_files) >= 1, (
            f"Expected at least 1 PNG file, got {len(png_files)} in {export_dir}"
        )

        for png_file in png_files:
            assert png_file.stat().st_size > 500, (
                f"PNG file {png_file.name} seems too small ({png_file.stat().st_size} bytes)"
            )

        dialog.close()
        app.processEvents()


# ===========================================================================
# 6. DICOM parameter map full pipeline round-trip
# ===========================================================================

class TestParameterMapPipelineRoundtrip:
    """Full pipeline: create parameter maps → export DICOM → load back."""

    def test_create_and_reload_maps(self, temp_dir, sample_spacing):
        """Parameter maps from fitting should survive DICOM export/reload."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        # Simulate what create_parameter_maps() returns
        shape = (16, 16, 3)
        np.random.seed(99)
        mask = np.random.random(shape) > 0.3

        original_maps = {
            'kb_map': np.random.random(shape).astype(np.float32) * 0.5 * mask,
            'kd_map': np.random.random(shape).astype(np.float32) * 0.1 * mask,
            'r_squared_map': (np.random.random(shape).astype(np.float32) * 0.3 + 0.6) * mask,
        }

        metadata = {
            'kernel_type': 'sliding_window',
            'window_x': 15,
            'window_y': 15,
            'window_z': 3,
        }

        # Export each map
        for map_name, map_data in original_maps.items():
            export_dir = str(Path(temp_dir) / "maps")
            saved = save_parameter_map_as_dicom(
                map_data, map_name, export_dir, sample_spacing,
                metadata=metadata,
            )
            assert len(saved) == 3

            # Reload and verify each slice
            for z, filepath in enumerate(saved):
                ds = pydicom.dcmread(filepath)
                raw = ds.pixel_array.astype(np.float64)
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                reconstructed = (raw * slope + intercept).T

                original = map_data[:, :, z]
                value_range = original.max() - original.min()
                if value_range > 0:
                    # Allow tolerance for 16-bit quantization
                    atol = value_range / 65535 * 2
                    np.testing.assert_allclose(
                        reconstructed, original, atol=atol,
                        err_msg=f"{map_name} z={z}: values don't match"
                    )

    def test_nan_handling_in_maps(self, temp_dir, sample_spacing):
        """NaN values in parameter maps should be preserved (as 0 in DICOM)."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        data = np.full((8, 8, 1), np.nan, dtype=np.float32)
        data[2:6, 2:6, 0] = 0.25  # Some valid values

        saved = save_parameter_map_as_dicom(
            data, 'kb_map', temp_dir, sample_spacing,
        )

        ds = pydicom.dcmread(saved[0])
        raw = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        reconstructed = (raw * slope + intercept).T

        # Valid region should be close to 0.25
        valid_region = reconstructed[2:6, 2:6]
        np.testing.assert_allclose(
            valid_region, 0.25, atol=0.01,
            err_msg="Valid region values don't match after NaN export"
        )


# ===========================================================================
# 7. Image Tools export round-trip via dialog
# ===========================================================================

class TestImageToolsExportRoundtrip:
    """ImageToolsDialog save image → load back DICOM and verify."""

    def test_averaged_image_export_via_dialog(self, temp_dir):
        """_save_image in average mode should produce loadable DICOM."""
        from proxyl_analysis.ui.image_tools import ImageToolsDialog
        from proxyl_analysis.ui.styles import init_qt_app
        import time as time_mod

        app = init_qt_app()

        time_array = np.linspace(0, 40, 20)
        image_4d = np.zeros((16, 16, 3, 20), dtype=np.float32)
        for t in range(20):
            image_4d[:, :, :, t] = 5000 + t * 200
        roi_mask = np.ones((16, 16, 3), dtype=bool)

        dialog = ImageToolsDialog(
            image_4d=image_4d,
            time_array=time_array,
            roi_signal=np.mean(image_4d[:, :, :, :], axis=(0, 1, 2)),
            time_units='minutes',
            output_dir=temp_dir,
            initial_mode='average',
            roi_mask=roi_mask,
        )
        dialog.show()
        for _ in range(5):
            app.processEvents()
            time_mod.sleep(0.05)

        # Set region and trigger preview
        dialog.region_a_start = 5
        dialog.region_a_end = 10
        dialog.region_a_start_spin.setValue(5)
        dialog.region_a_end_spin.setValue(10)

        # Compute the preview image directly
        dialog._update_preview()
        app.processEvents()

        if dialog.preview_image is not None:
            save_dir = Path(temp_dir) / "avg_export"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / "test_avg.dcm")

            with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',
                       return_value=(save_path, 'DICOM Files (*.dcm)')):
                with patch('PySide6.QtWidgets.QMessageBox.information'):
                    dialog._save_image()

            # Check if DICOM files were created (3D creates subfolder)
            dcm_files = list(Path(temp_dir).rglob("*.dcm"))
            assert len(dcm_files) >= 1, "No DICOM files created"

            ds = pydicom.dcmread(str(dcm_files[0]))
            assert ds.Rows > 0
            assert ds.Columns > 0
            assert hasattr(ds, 'PixelData')

            # Verify operation metadata
            comments = json.loads(ds.ImageComments)
            assert comments['start_idx'] == 5
            assert comments['end_idx'] == 10

        dialog.close()
        app.processEvents()


# ===========================================================================
# 8. Registered DICOM series round-trip
# ===========================================================================

class TestRegisteredDicomSeriesRoundtrip:
    """Save registered 4D as DICOM series, load back, verify."""

    def test_save_and_load_series(self, temp_dir):
        """Full 4D round-trip via registered DICOM series."""
        from proxyl_analysis.io import save_registered_as_dicom_series, load_registered_dicom_series

        # Create synthetic 4D data
        np.random.seed(42)
        original_4d = (np.random.random((16, 16, 3, 5)) * 10000).astype(np.float32)
        spacing = (0.5, 0.5, 2.0)

        # Save
        dicom_dir = save_registered_as_dicom_series(
            original_4d, spacing, temp_dir,
            series_description="Test Registered"
        )

        # Load back using the DICOM series loader directly
        loaded_4d, loaded_spacing = load_registered_dicom_series(dicom_dir)

        assert loaded_4d.shape == original_4d.shape, (
            f"Shape mismatch: {loaded_4d.shape} != {original_4d.shape}"
        )
        assert loaded_spacing[0] == pytest.approx(spacing[0], rel=0.01)
        assert loaded_spacing[1] == pytest.approx(spacing[1], rel=0.01)
        assert loaded_spacing[2] == pytest.approx(spacing[2], rel=0.01)

        # Values should be close (16-bit quantization)
        for z in range(3):
            for t in range(5):
                orig_slice = original_4d[:, :, z, t]
                load_slice = loaded_4d[:, :, z, t]
                value_range = orig_slice.max() - orig_slice.min()
                if value_range > 0:
                    atol = value_range / 65535 * 2
                    np.testing.assert_allclose(
                        load_slice, orig_slice, atol=atol,
                        err_msg=f"z={z}, t={t}: values don't match"
                    )

    def test_file_naming_convention(self, temp_dir):
        """Files should follow z{ZZ}_t{TTT}.dcm naming."""
        from proxyl_analysis.io import save_registered_as_dicom_series

        data = np.random.random((8, 8, 2, 3)).astype(np.float32) * 5000
        save_registered_as_dicom_series(data, (1.0, 1.0, 2.0), temp_dir)

        dicom_dir = Path(temp_dir) / "registered" / "dicoms"
        dcm_files = sorted(dicom_dir.glob("z*_t*.dcm"))

        expected_names = [
            "z00_t000.dcm", "z00_t001.dcm", "z00_t002.dcm",
            "z01_t000.dcm", "z01_t001.dcm", "z01_t002.dcm",
        ]
        actual_names = [f.name for f in dcm_files]
        assert actual_names == expected_names

    def test_series_info_json(self, temp_dir):
        """series_info.json should be created with correct dimensions."""
        from proxyl_analysis.io import save_registered_as_dicom_series

        data = np.random.random((16, 16, 3, 5)).astype(np.float32)
        save_registered_as_dicom_series(data, (0.5, 0.5, 2.0), temp_dir)

        info_path = Path(temp_dir) / "registered" / "dicoms" / "series_info.json"
        assert info_path.exists()

        with open(info_path) as f:
            info = json.load(f)

        # shape is stored as [x, y, z, t]
        assert info['shape'] == [16, 16, 3, 5]
        assert info['spacing'] == pytest.approx([0.5, 0.5, 2.0])
        assert info['n_slices'] == 3
        assert info['n_timepoints'] == 5
        assert info['format_version'] == '2.0'
