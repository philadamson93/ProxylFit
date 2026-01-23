# T013: ROI Contour Metrics for Difference/Averaged Images

**Status**: planned
**Priority**: medium
**Created**: 2025-01-22

## Description

Carry the ROI contour through to difference and averaged image views, displaying quantitative metrics (mean ± stdev) within the contour. For difference images, also report the mean values of both input windows (Region A and Region B) that generated the subtraction.

## Current Implementation

- ROI is selected on the registered data during the main workflow
- Difference/averaged images are created in Image Tools dialog
- No contour overlay or metrics displayed on derived images

## Proposed Implementation

### Contour Display

- Two options for ROI selection:
  - **Use Existing ROI**: Reuse contour from main workflow (if available)
  - **Draw New ROI**: Draw a new contour directly on the diff/avg image
- Contour displayed as colored outline (e.g., cyan) on the image
- Checkbox to toggle contour visibility
- "Redraw ROI" button to replace current contour at any time

### Metrics Panel (Side Panel)

For **Averaged Images:**
```
┌─────────────────────────────────┐
│ ROI Metrics                     │
├─────────────────────────────────┤
│ Averaged Image (t0 - t5)        │
│                                 │
│ Within ROI:                     │
│   Mean:  1247.3 ± 89.2          │
│   Min:   1089.1                 │
│   Max:   1456.7                 │
│   Pixels: 342                   │
└─────────────────────────────────┘
```

For **Difference Images:**
```
┌─────────────────────────────────┐
│ ROI Metrics                     │
├─────────────────────────────────┤
│ Difference Image                │
│ (t0-t5) minus (t10-t15)         │
│                                 │
│ Region A (t0-t5):               │
│   Mean:  1247.3 ± 89.2          │
│                                 │
│ Region B (t10-t15):             │
│   Mean:  1089.5 ± 76.4          │
│                                 │
│ Difference (A - B):             │
│   Mean:  157.8 ± 42.1           │
│   Min:   -23.4                  │
│   Max:   289.6                  │
│   Pixels: 342                   │
│                                 │
│ % Change: +14.5%                │
└─────────────────────────────────┘
```

### Export Metadata

When saving (per T012), include metrics in:
- DICOM ImageComments tag
- Separate CSV sidecar file: `{filename}_metrics.csv`
- NPZ metadata dict

**Metrics CSV format:**
```csv
metric,region,value,std,min,max
mean_intensity,region_a,1247.3,89.2,,
mean_intensity,region_b,1089.5,76.4,,
mean_intensity,difference,157.8,42.1,-23.4,289.6
percent_change,difference,14.5,,,
n_pixels,roi,342,,,
z_slice,roi,4,,,
operation,info,difference,,,
timepoints_a,info,0-5,,,
timepoints_b,info,10-15,,,
```

## Files to Modify

### `proxyl_analysis/ui/image_tools.py`

**Add ROI overlay:**
- Accept `roi_mask` parameter in `ImageToolsDialog.__init__()`
- Draw contour on preview image using matplotlib contour
- Add "Draw New ROI" button if user wants different contour

**Add metrics panel:**
- New `QGroupBox` for "ROI Metrics" in the side panel
- Calculate and display metrics when:
  - Preview is generated
  - ROI is available or newly drawn
- Update metrics when slice changes (for 3D data)

**Add metrics calculation:**
```python
def _calculate_roi_metrics(self, image: np.ndarray, mask: np.ndarray) -> dict:
    """Calculate mean, std, min, max within ROI mask."""

def _calculate_difference_metrics(self, region_a_avg: np.ndarray,
                                   region_b_avg: np.ndarray,
                                   mask: np.ndarray) -> dict:
    """Calculate metrics for both input regions and their difference."""
```

**Modify `_update_preview()`:**
- After computing diff/avg image, calculate metrics if ROI available
- Update metrics panel display

**Modify `_save_preview()`:**
- Include metrics in exported metadata
- Save metrics JSON sidecar

### `proxyl_analysis/ui/main_menu.py`

- Pass `roi_mask` to `ImageToolsDialog` when launching from menu
- Store ROI mask in workflow state

### `proxyl_analysis/ui/roi.py`

- May need to expose contour extraction for overlay drawing

## UI Mockup

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Image Tools - Difference Image                                           │
├────────────────────────────────────────────┬─────────────────────────────┤
│                                            │ Time Selection              │
│     [Image preview with ROI contour        │ Region A: [==|----]  t0-t5  │
│      overlaid in cyan]                     │ Region B: [----==|]  t10-15 │
│                                            │                             │
│                                            │ [✓] Show ROI contour        │
│                                            │ [Draw New ROI]              │
│                                            ├─────────────────────────────┤
│                                            │ ROI Metrics                 │
│                                            │                             │
│                                            │ Region A (t0-t5):           │
│                                            │   Mean: 1247.3 ± 89.2       │
│                                            │                             │
│                                            │ Region B (t10-t15):         │
│                                            │   Mean: 1089.5 ± 76.4       │
│                                            │                             │
│                                            │ Difference:                 │
│                                            │   Mean: 157.8 ± 42.1        │
│                                            │   % Change: +14.5%          │
├────────────────────────────────────────────┴─────────────────────────────┤
│ [Cancel]                              [Export Metrics]  [Save Image]     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Dependencies

- T012 (DICOM export for derived images - for metadata export)
- Existing ROI selection workflow
