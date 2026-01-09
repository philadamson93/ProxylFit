# IO Module

**File**: `proxyl_analysis/io.py`

The IO module handles loading DICOM data and converting it to the 4D numpy array format used throughout ProxylFit.

## Overview

ProxylFit expects 4D MRI data with dimensions `[x, y, z, t]`:
- `x, y`: In-plane spatial dimensions
- `z`: Slice dimension
- `t`: Time dimension

## Functions

### load_dicom_series()

Load a DICOM series and reshape into 4D tensor.

```python
from proxyl_analysis.io import load_dicom_series

image_4d, spacing = load_dicom_series(dicom_path)
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dicom_path` | str | Path to DICOM file or directory |

**Returns**:
| Return | Type | Description |
|--------|------|-------------|
| `image_4d` | np.ndarray | 4D array with shape [x, y, z, t] |
| `spacing` | tuple | Voxel spacing (x, y, z) in mm |

**Example**:
```python
from proxyl_analysis.io import load_dicom_series

# Load DICOM data
image_4d, spacing = load_dicom_series("path/to/dicom.dcm")

print(f"Shape: {image_4d.shape}")  # e.g., (128, 128, 9, 60)
print(f"Spacing: {spacing}")        # e.g., (0.5, 0.5, 2.0)
```

## Data Format

### Input Requirements

- **DICOM format**: Standard DICOM files (.dcm)
- **Multi-frame**: Can be single multi-frame DICOM or directory of files
- **Supported sequences**: T1, T2, dynamic contrast-enhanced

### Output Format

The output 4D array has:
- **dtype**: float64
- **Shape**: [x, y, z, t]
- **Orientation**: Based on DICOM orientation tags

### Spacing Information

Voxel spacing is extracted from DICOM headers:
- `PixelSpacing` for x, y dimensions
- `SliceThickness` or `SpacingBetweenSlices` for z dimension

## Timing Information

ProxylFit assumes **70-second intervals** between timepoints by default. This can be adjusted using the `--time-units` CLI option or by modifying the time array directly.

```python
# Default: 70 seconds per timepoint
time_array = np.arange(num_timepoints) * (70.0 / 60.0)  # in minutes
```

## Error Handling

The module handles common DICOM issues:

```python
try:
    image_4d, spacing = load_dicom_series(path)
except FileNotFoundError:
    print("DICOM file not found")
except ValueError as e:
    print(f"Invalid DICOM data: {e}")
```

## Memory Considerations

Large 4D datasets can consume significant memory:

| Dimensions | dtype | Memory |
|------------|-------|--------|
| 128x128x9x60 | float64 | ~71 MB |
| 256x256x20x100 | float64 | ~1.0 GB |
| 512x512x50x200 | float64 | ~20.5 GB |

For very large datasets, consider:
- Processing individual slices
- Using memory-mapped arrays
- Downsampling spatial dimensions

## Integration with SimpleITK

The module uses SimpleITK for DICOM handling:

```python
import SimpleITK as sitk

# Internal implementation
reader = sitk.ImageSeriesReader()
dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
reader.SetFileNames(dicom_files)
image = reader.Execute()
```

## See Also

- [Registration Module](registration.md) - Next step in pipeline
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)
