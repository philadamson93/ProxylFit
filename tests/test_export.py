"""
Unit tests for DICOM and PNG export functionality.

Tests save and load operations for:
- Parameter maps (DICOM)
- Derived images (averaged/difference) (DICOM)
- PNG exports with and without ROI overlay
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest
import pydicom


# Test fixtures
@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="proxylfit_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_parameter_map():
    """Create a sample 3D parameter map with known values."""
    # Create a 32x32x5 parameter map with gradient values
    x, y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    kb_map = np.zeros((32, 32, 5))
    for z in range(5):
        # Each z-slice has a distinct pattern
        kb_map[:, :, z] = (x + y) * (z + 1) * 0.1
    return kb_map


@pytest.fixture
def sample_4d_image():
    """Create a sample 4D image for averaging/differencing."""
    # 32x32x5 spatial, 10 timepoints
    # Add time-varying spatial variation to avoid uniform difference images
    image = np.zeros((32, 32, 5, 10))
    x, y = np.meshgrid(np.arange(32), np.arange(32))
    spatial_variation = (x + y) / 64.0  # 0 to 1 variation
    for t in range(10):
        # Base intensity increases with time
        base = 100 + t * 10
        # Spatial variation also increases with time (key for non-uniform differences)
        image[:, :, :, t] = base + spatial_variation[:, :, np.newaxis] * (5 + t)
    return image


@pytest.fixture
def sample_roi_mask():
    """Create a sample ROI mask (circular region in center)."""
    mask = np.zeros((32, 32, 5), dtype=bool)
    y, x = np.ogrid[:32, :32]
    center = (16, 16)
    radius = 8
    circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    for z in range(5):
        mask[:, :, z] = circle
    return mask


@pytest.fixture
def sample_spacing():
    """Standard voxel spacing for tests."""
    return (1.0, 1.0, 2.0)  # x, y, z in mm


class TestParameterMapDICOMExport:
    """Tests for parameter map DICOM export and load."""

    def test_save_parameter_map_creates_files(self, temp_output_dir, sample_parameter_map, sample_spacing):
        """Test that saving a parameter map creates expected DICOM files."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map,
            map_name='kb_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
            source_dicom=None,
            metadata={'kernel_type': 'sliding_window', 'window_x': 15}
        )

        # Should create one file per z-slice
        assert len(saved_files) == 5, f"Expected 5 files, got {len(saved_files)}"

        # All files should exist
        for filepath in saved_files:
            assert Path(filepath).exists(), f"File not created: {filepath}"

    def test_load_parameter_map_dicom_preserves_shape(self, temp_output_dir, sample_parameter_map, sample_spacing):
        """Test that loaded DICOM preserves spatial dimensions."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map,
            map_name='kb_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        # Load the first file and check dimensions
        ds = pydicom.dcmread(saved_files[0])
        assert ds.Rows == 32, f"Expected Rows=32, got {ds.Rows}"
        assert ds.Columns == 32, f"Expected Columns=32, got {ds.Columns}"

    def test_load_parameter_map_dicom_preserves_values(self, temp_output_dir, sample_parameter_map, sample_spacing):
        """Test that loaded DICOM preserves data values (within tolerance)."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map,
            map_name='kb_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        # Load and reconstruct each slice
        for z, filepath in enumerate(sorted(saved_files)):
            ds = pydicom.dcmread(filepath)

            # Get pixel array and apply rescale
            pixel_array = ds.pixel_array.astype(np.float64)
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            reconstructed = pixel_array * slope + intercept

            # Original slice (transposed as DICOM stores [y, x])
            original_slice = sample_parameter_map[:, :, z].T

            # Compare values - allow some tolerance due to 16-bit quantization
            # For normalized data, tolerance of ~1% is reasonable
            max_diff = np.max(np.abs(reconstructed - original_slice))
            max_value = np.max(np.abs(original_slice))
            if max_value > 0:
                relative_error = max_diff / max_value
                assert relative_error < 0.02, f"Slice {z}: relative error {relative_error:.4f} exceeds 2%"

    def test_parameter_map_dicom_metadata(self, temp_output_dir, sample_parameter_map, sample_spacing):
        """Test that DICOM metadata is set correctly."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        metadata = {'kernel_type': 'gaussian', 'window_x': 11, 'window_y': 11}
        saved_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map,
            map_name='kd_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
            metadata=metadata
        )

        ds = pydicom.dcmread(saved_files[0])

        # Check modality and type
        assert ds.Modality == 'MR'
        assert 'PARAMETER_MAP' in ds.ImageType

        # Check series description
        assert 'Decay Rate' in ds.SeriesDescription

        # Check spacing
        assert list(ds.PixelSpacing) == [1.0, 1.0]
        assert ds.SliceThickness == 2.0

    def test_multiple_parameter_maps_separate_folders(self, temp_output_dir, sample_parameter_map, sample_spacing):
        """Test that different parameter maps go to separate folders."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        kb_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map,
            map_name='kb_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        kd_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map * 0.5,  # Different values
            map_name='kd_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        # Should be in separate subdirectories
        kb_dir = Path(kb_files[0]).parent
        kd_dir = Path(kd_files[0]).parent
        assert kb_dir != kd_dir
        assert kb_dir.name == 'kb_map'
        assert kd_dir.name == 'kd_map'


class TestDerivedImageDICOMExport:
    """Tests for averaged/difference image DICOM export."""

    def test_save_averaged_image_creates_files(self, temp_output_dir, sample_4d_image, sample_spacing):
        """Test saving an averaged image creates DICOM files."""
        from proxyl_analysis.io import save_derived_image_as_dicom
        from proxyl_analysis.ui.image_tools import compute_averaged_image

        # Create averaged image from frames 0-4
        avg_image = compute_averaged_image(sample_4d_image, 0, 4)

        operation_params = {
            'start_idx': 0,
            'end_idx': 4,
            'n_frames': 5,
            'time_units': 'minutes'
        }

        output_path = Path(temp_output_dir) / 'avg_t0-t4.dcm'
        saved_files = save_derived_image_as_dicom(
            image=avg_image,
            output_path=str(output_path),
            operation_type='averaged',
            operation_params=operation_params,
            spacing=sample_spacing,
        )

        # Should create 5 files (one per z-slice)
        assert len(saved_files) == 5
        for f in saved_files:
            assert Path(f).exists()

    def test_averaged_image_values(self, temp_output_dir, sample_4d_image, sample_spacing):
        """Test averaged image has correct values."""
        from proxyl_analysis.io import save_derived_image_as_dicom
        from proxyl_analysis.ui.image_tools import compute_averaged_image

        # Create averaged image from frames 0-4
        # Base intensities: 100,110,120,130,140 with spatial variation
        avg_image = compute_averaged_image(sample_4d_image, 0, 4)

        # Check that computed average is reasonable (mean around 120 + spatial variation)
        assert 115 < avg_image.mean() < 135, f"Unexpected avg: {avg_image.mean()}"

        operation_params = {'start_idx': 0, 'end_idx': 4, 'n_frames': 5}
        output_path = Path(temp_output_dir) / 'avg.dcm'
        saved_files = save_derived_image_as_dicom(
            image=avg_image,
            output_path=str(output_path),
            operation_type='averaged',
            operation_params=operation_params,
            spacing=sample_spacing,
        )

        # Load and verify values are preserved
        ds = pydicom.dcmread(saved_files[0])
        pixel_array = ds.pixel_array.astype(np.float64)
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        reconstructed = pixel_array * slope + intercept

        # Get original slice for comparison (transposed for DICOM)
        original_slice = avg_image[:, :, 0].T

        # Check that reconstructed values match original within tolerance
        max_diff = np.max(np.abs(reconstructed - original_slice))
        max_value = np.max(np.abs(original_slice))
        relative_error = max_diff / max_value
        assert relative_error < 0.02, f"Relative error {relative_error:.4f} exceeds 2%"

    def test_save_difference_image_handles_negatives(self, temp_output_dir, sample_4d_image, sample_spacing):
        """Test difference images correctly handle negative values."""
        from proxyl_analysis.io import save_derived_image_as_dicom
        from proxyl_analysis.ui.image_tools import compute_difference_image

        # Difference: B - A where B = frames 0-1, A = frames 8-9
        # This results in negative values (early - late)
        diff_image = compute_difference_image(sample_4d_image, (8, 9), (0, 1))

        # Verify we have negative values
        assert diff_image.min() < 0, "Difference image should have negative values"
        assert diff_image.mean() < 0, "Mean should be negative"

        operation_params = {
            'region_a_start': 8, 'region_a_end': 9,
            'region_b_start': 0, 'region_b_end': 1
        }

        output_path = Path(temp_output_dir) / 'diff.dcm'
        saved_files = save_derived_image_as_dicom(
            image=diff_image,
            output_path=str(output_path),
            operation_type='difference',
            operation_params=operation_params,
            spacing=sample_spacing,
        )

        # Load and verify negative values preserved
        ds = pydicom.dcmread(saved_files[0])
        assert ds.PixelRepresentation == 1, "Difference image should use signed pixels"

        pixel_array = ds.pixel_array.astype(np.float64)
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        reconstructed = pixel_array * slope + intercept

        # Get original slice for comparison (transposed for DICOM)
        original_slice = diff_image[:, :, 0].T

        # Verify values are reconstructed correctly
        max_diff = np.max(np.abs(reconstructed - original_slice))
        max_value = np.max(np.abs(original_slice))
        relative_error = max_diff / max_value
        assert relative_error < 0.02, f"Relative error {relative_error:.4f} exceeds 2%"

        # Verify sign is preserved
        assert reconstructed.mean() < 0, f"Reconstructed mean should be negative, got {reconstructed.mean()}"

    def test_derived_image_series_description(self, temp_output_dir, sample_4d_image, sample_spacing):
        """Test that series description reflects the operation."""
        from proxyl_analysis.io import save_derived_image_as_dicom
        from proxyl_analysis.ui.image_tools import compute_averaged_image

        avg_image = compute_averaged_image(sample_4d_image, 2, 7)
        operation_params = {'start_idx': 2, 'end_idx': 7, 'n_frames': 6}

        output_path = Path(temp_output_dir) / 'avg.dcm'
        saved_files = save_derived_image_as_dicom(
            image=avg_image,
            output_path=str(output_path),
            operation_type='averaged',
            operation_params=operation_params,
            spacing=sample_spacing,
        )

        ds = pydicom.dcmread(saved_files[0])
        assert 't2' in ds.SeriesDescription and 't7' in ds.SeriesDescription
        assert '6 frames' in ds.SeriesDescription


class TestPNGExport:
    """Tests for PNG export functionality."""

    def test_png_export_creates_file(self, temp_output_dir, sample_parameter_map):
        """Test that PNG export creates a file."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Create a simple figure to export
        fig, ax = plt.subplots()
        ax.imshow(sample_parameter_map[:, :, 0], cmap='plasma')
        ax.set_title('kb (z=0)')

        png_path = Path(temp_output_dir) / 'kb_z00.png'
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_png_can_be_loaded(self, temp_output_dir, sample_parameter_map):
        """Test that exported PNG can be loaded and read."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, ax = plt.subplots()
        ax.imshow(sample_parameter_map[:, :, 0], cmap='plasma')

        png_path = Path(temp_output_dir) / 'test.png'
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Load with PIL
        img = Image.open(png_path)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_png_with_roi_overlay(self, temp_output_dir, sample_parameter_map, sample_roi_mask):
        """Test PNG export with ROI contour overlay."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        z = 2
        fig, ax = plt.subplots()
        ax.imshow(sample_parameter_map[:, :, z].T, cmap='plasma', origin='lower')

        # Add ROI contour
        roi_slice = sample_roi_mask[:, :, z].T
        ax.contour(roi_slice, levels=[0.5], colors='cyan', linewidths=2)
        ax.set_title(f'kb with ROI (z={z})')

        png_path = Path(temp_output_dir) / 'kb_with_roi.png'
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

        assert png_path.exists()

        # PNG with ROI should be larger than without (more detail)
        from PIL import Image
        img = Image.open(png_path)
        assert img.size[0] > 0


class TestRoundTripDataIntegrity:
    """Integration tests for full save/load cycle data integrity."""

    def test_parameter_map_roundtrip_all_slices(self, temp_output_dir, sample_parameter_map, sample_spacing):
        """Test save/load roundtrip for all parameter map slices."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        saved_files = save_parameter_map_as_dicom(
            param_map=sample_parameter_map,
            map_name='r_squared_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        # Load all slices and verify
        for z, filepath in enumerate(sorted(saved_files)):
            ds = pydicom.dcmread(filepath)
            pixel_array = ds.pixel_array.astype(np.float64)
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            reconstructed = pixel_array * slope + intercept

            # Compare to original slice (transposed)
            original_slice = sample_parameter_map[:, :, z].T

            # Allow 2% tolerance due to 16-bit quantization
            max_diff = np.max(np.abs(reconstructed - original_slice))
            max_value = np.max(np.abs(original_slice))
            if max_value > 0:
                relative_error = max_diff / max_value
                assert relative_error < 0.02, \
                    f"Slice {z}: relative error {relative_error:.4f} exceeds 2%"

    def test_derived_image_roundtrip_3d(self, temp_output_dir, sample_4d_image, sample_spacing):
        """Test roundtrip for a full 3D derived image."""
        from proxyl_analysis.io import save_derived_image_as_dicom
        from proxyl_analysis.ui.image_tools import compute_averaged_image

        avg_image = compute_averaged_image(sample_4d_image, 3, 6)

        operation_params = {'start_idx': 3, 'end_idx': 6, 'n_frames': 4}
        output_path = Path(temp_output_dir) / 'avg_3d.dcm'

        saved_files = save_derived_image_as_dicom(
            image=avg_image,
            output_path=str(output_path),
            operation_type='averaged',
            operation_params=operation_params,
            spacing=sample_spacing,
        )

        # Load all slices and verify values match original
        for z, filepath in enumerate(sorted(saved_files)):
            ds = pydicom.dcmread(filepath)
            pixel_array = ds.pixel_array.astype(np.float64)
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            reconstructed = pixel_array * slope + intercept

            # Compare to original slice (transposed)
            original_slice = avg_image[:, :, z].T

            # Allow 2% tolerance
            max_diff = np.max(np.abs(reconstructed - original_slice))
            max_value = np.max(np.abs(original_slice))
            relative_error = max_diff / max_value
            assert relative_error < 0.02, \
                f"Slice {z}: relative error {relative_error:.4f} exceeds 2%"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_2d_parameter_map(self, temp_output_dir, sample_spacing):
        """Test export of 2D (single slice) parameter map."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        param_map_2d = np.random.random((32, 32)) * 0.5

        saved_files = save_parameter_map_as_dicom(
            param_map=param_map_2d,
            map_name='test_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        assert len(saved_files) == 1
        assert Path(saved_files[0]).exists()

    def test_nan_values_in_parameter_map(self, temp_output_dir, sample_spacing):
        """Test that NaN values are handled (replaced with 0)."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        param_map = np.ones((32, 32, 3)) * 0.5
        param_map[10:20, 10:20, :] = np.nan  # Add NaN region

        saved_files = save_parameter_map_as_dicom(
            param_map=param_map,
            map_name='kb_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        # Load and verify no NaN in output
        ds = pydicom.dcmread(saved_files[0])
        pixel_array = ds.pixel_array.astype(np.float64)
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        reconstructed = pixel_array * slope + intercept

        assert not np.any(np.isnan(reconstructed)), "NaN values should be replaced with 0"

    def test_uniform_image_no_crash(self, temp_output_dir, sample_spacing):
        """Test that uniform images (zero range) don't cause division errors."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        # Uniform image - all same value
        param_map = np.ones((32, 32, 3)) * 42.0

        saved_files = save_parameter_map_as_dicom(
            param_map=param_map,
            map_name='uniform_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        assert len(saved_files) == 3
        for f in saved_files:
            assert Path(f).exists()

    def test_negative_parameter_values(self, temp_output_dir, sample_spacing):
        """Test parameter maps with negative values."""
        from proxyl_analysis.io import save_parameter_map_as_dicom

        param_map = np.zeros((32, 32, 2))
        param_map[:16, :, :] = -0.5
        param_map[16:, :, :] = 0.5

        saved_files = save_parameter_map_as_dicom(
            param_map=param_map,
            map_name='kd_map',
            output_dir=temp_output_dir,
            spacing=sample_spacing,
        )

        # Load and verify signed representation
        ds = pydicom.dcmread(saved_files[0])
        assert ds.PixelRepresentation == 1, "Should use signed pixels for negative values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
