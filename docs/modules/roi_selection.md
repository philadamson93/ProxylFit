# ROI Selection Module

**File**: `proxyl_analysis/roi_selection.py`

The ROI selection module provides interactive tools for defining regions of interest on MRI slices.

## Available Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `rectangle` | Rectangular bounding box | Quick selection, regular shapes |
| `contour` | Manual free-form drawing | Irregular shapes, precise boundaries |
| `segment` | AI-based (SegmentAnything) | Complex anatomical structures |

## Functions

### select_rectangle_roi()

Interactive rectangular ROI selection (matplotlib version).

```python
from proxyl_analysis.roi_selection import select_rectangle_roi

mask = select_rectangle_roi(image_4d, z_index=4)
```

> **Recommended**: Use `select_rectangle_roi_qt()` from `ui.py` for better layout.

### select_manual_contour_roi()

Interactive contour drawing (matplotlib version).

```python
from proxyl_analysis.roi_selection import select_manual_contour_roi

mask = select_manual_contour_roi(image_4d, z_index=4)
```

> **Recommended**: Use `select_manual_contour_roi_qt()` from `ui.py` for better layout.

### select_segmentation_roi()

SegmentAnything-based segmentation.

```python
from proxyl_analysis.roi_selection import select_segmentation_roi

mask = select_segmentation_roi(
    image_4d, z_index=4,
    model_path='sam_vit_h.pth',
    model_type='vit_h'
)
```

**Requirements**: `segment-anything` and `opencv-python` packages.

### compute_roi_timeseries()

Extract mean signal time series from ROI.

```python
from proxyl_analysis.roi_selection import compute_roi_timeseries

timeseries = compute_roi_timeseries(image_4d, roi_mask)
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `image_4d` | np.ndarray | 4D image [x, y, z, t] |
| `roi_mask` | np.ndarray | 2D boolean mask [x, y] |

**Returns**: 1D array of mean intensity per timepoint.

### get_available_roi_modes()

Check which ROI modes are available.

```python
from proxyl_analysis.roi_selection import get_available_roi_modes

modes = get_available_roi_modes()
# Returns: ['rectangle', 'contour', 'segment'] or subset
```

## Qt-Based ROI Selection

For modern UI with proper layout management, use the Qt versions:

```python
from proxyl_analysis.ui import (
    select_rectangle_roi_qt,
    select_manual_contour_roi_qt
)

# Rectangle selection with Qt UI
mask = select_rectangle_roi_qt(image_4d, z_index=4)

# Contour drawing with Qt UI
mask = select_manual_contour_roi_qt(image_4d, z_index=4)
```

## ROI Mask Format

All ROI functions return a **2D boolean mask**:

```python
mask.shape  # (x_dim, y_dim) - matches image_4d[:, :, z, 0].shape
mask.dtype  # bool
mask.sum()  # Number of selected pixels
```

## Rectangle Mode Details

### Workflow
1. Display slice at timepoint 0
2. Click and drag to draw rectangle
3. Adjust by dragging corners/edges
4. Click "Accept ROI" to confirm

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| Enter | Accept selection |
| Escape | Cancel |

## Contour Mode Details

### Workflow
1. Display slice with Z-navigation
2. Click and drag to draw contour
3. Press 'C' to close contour
4. Mask is created from closed contour
5. Click "Accept ROI" to confirm

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| C | Close contour |
| R | Reset drawing |
| Up/Down | Change Z-slice |
| Enter | Accept ROI |
| Escape | Cancel |

### Drawing Tips
- Draw slowly for smoother contours
- Points are added at minimum 2-pixel intervals
- Contour must be closed to create mask
- Can navigate Z-slices while drawing

## Segment Mode Details

### Requirements
```bash
pip install segment-anything opencv-python
```

### Model Download
Download SAM checkpoint from Meta AI:
- [ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (2.4 GB, most accurate)
- [ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (1.2 GB)
- [ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (358 MB, fastest)

### Workflow
1. Click to add positive points (green +)
2. Press 'T' to toggle to negative mode
3. Click to add negative points (red x)
4. Press 'S' to run segmentation
5. Refine with more points if needed
6. Press 'C' to confirm

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| T | Toggle positive/negative mode |
| S | Run segmentation |
| C | Confirm selection |
| R | Reset all points |

## ROI Statistics

All selectors provide statistics about the selected region:

```python
stats = selector.get_roi_stats()
# Returns:
{
    'num_pixels': int,
    'mean_intensity': float,
    'std_intensity': float,
    'min_intensity': float,
    'max_intensity': float,
    'coordinates': tuple,  # For rectangle: (x_min, x_max, y_min, y_max)
    'bounding_box': tuple  # For contour/segment
}
```

## Example: Complete ROI Workflow

```python
from proxyl_analysis.io import load_dicom_series
from proxyl_analysis.registration import register_timeseries
from proxyl_analysis.ui import select_manual_contour_roi_qt
from proxyl_analysis.roi_selection import compute_roi_timeseries

# Load and register data
image_4d, spacing = load_dicom_series("data.dcm")
registered_4d, metrics = register_timeseries(image_4d, spacing)

# Select ROI using Qt UI
z_slice = 4
roi_mask = select_manual_contour_roi_qt(registered_4d, z_slice)

# Check selection
print(f"Selected {roi_mask.sum()} pixels")

# Extract time series
signal = compute_roi_timeseries(registered_4d, roi_mask)
print(f"Time series: {len(signal)} points")
```

## Troubleshooting

### No ROI Selected
**Symptom**: Function returns empty mask

**Solutions**:
1. Ensure you clicked "Accept ROI" button
2. For contour mode, ensure contour is closed (press 'C')
3. Check console for error messages

### Contour Mask Incorrect
**Symptom**: Mask doesn't match drawn contour

**Solutions**:
1. Ensure contour is fully closed
2. Draw contour in consistent direction
3. Avoid self-intersecting paths

### SegmentAnything Not Available
**Symptom**: ImportError for segment mode

**Solutions**:
```bash
pip install segment-anything opencv-python
```

## See Also

- [Qt UI Guide](../ui.md) - Modern UI components
- [Kinetic Model](model.md) - Fitting time series
- [Registration](registration.md) - Align before ROI selection
