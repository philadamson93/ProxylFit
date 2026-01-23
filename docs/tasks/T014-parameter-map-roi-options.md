# T014: Parameter Map ROI and Single-Slice Options

**Status**: planned
**Priority**: medium
**Created**: 2025-01-22

## Description

Add options to run parameter mapping on:
1. A single slice instead of full volume (faster)
2. Only within an ROI contour (new or existing) instead of full FOV (faster)
3. Display mean ± stdev of parameter values within the contour with option to redraw or reuse existing ROI

## Current Implementation

- Parameter mapping runs on entire 4D volume (all slices, all voxels)
- No option to restrict to ROI or single slice
- Computationally expensive for full volume pixel-wise fitting

## Proposed Implementation

### Options Dialog

Before running parameter mapping, show options:

```
┌─────────────────────────────────────────────────────┐
│ Parameter Mapping Options                           │
├─────────────────────────────────────────────────────┤
│ Spatial Extent:                                     │
│   ○ Full volume (all slices)                        │
│   ○ Single slice: [Slice 4 ▼] (fastest)             │
│                                                     │
│ Region of Interest:                                 │
│   ○ Full field of view                              │
│   ○ Within ROI contour only (faster)                │
│       [Use Existing ROI] [Draw New ROI]             │
│                                                     │
│ Fitting Method:                                     │
│   ○ Sliding window (current, smoothed)              │
│   ○ Pixel-wise (slower, higher resolution)          │
│                                                     │
│              [Cancel]  [Run Parameter Mapping]      │
└─────────────────────────────────────────────────────┘
```

### Single Slice Mode

- Process only one z-slice instead of all 9
- Significantly faster (~9x speedup)
- Useful for quick previews or when only one slice is of interest
- Output: 2D parameter maps instead of 3D

### ROI-Restricted Mode

- Only fit voxels within the ROI mask
- Skip computation for background voxels
- Speedup proportional to ROI size vs full FOV
- Useful when only tumor region is of interest

### ROI Metrics Display

After parameter mapping completes, show metrics panel:

```
┌─────────────────────────────────────────────────────┐
│ Parameter Map ROI Metrics                           │
├─────────────────────────────────────────────────────┤
│ ROI: 342 voxels (slice 4)                           │
│                                                     │
│ Parameter        Mean ± Std        Range            │
│ ─────────────────────────────────────────────────   │
│ kb (buildup)     0.0234 ± 0.0089   [0.008, 0.051]   │
│ kd (decay)       0.0156 ± 0.0042   [0.009, 0.028]   │
│ knt (non-tracer) 0.0021 ± 0.0008   [0.001, 0.004]   │
│ R²               0.923 ± 0.045     [0.78, 0.99]     │
│                                                     │
│ [Redraw ROI]  [Export Metrics]                      │
└─────────────────────────────────────────────────────┘
```

### Redraw ROI Option

- "Redraw ROI" button opens contour selection on parameter map
- After redrawing, metrics recalculate for new ROI
- Option to save new ROI for future use

## Files to Modify

### `proxyl_analysis/parameter_mapping.py`

**Modify `create_parameter_maps()`:**
```python
def create_parameter_maps(
    registered_4d: np.ndarray,
    time_array: np.ndarray,
    injection_index: int,
    spacing: Tuple[float, float, float],
    roi_mask: np.ndarray = None,      # NEW: restrict to ROI
    z_slice: int = None,              # NEW: single slice mode
    window_size: int = 3,
    ...
) -> Dict[str, np.ndarray]:
```

**Add metrics calculation:**
```python
def calculate_parameter_roi_metrics(
    param_maps: Dict[str, np.ndarray],
    roi_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Calculate mean, std, min, max for each parameter within ROI."""
```

### `proxyl_analysis/ui/main_menu.py`

**Add parameter mapping options dialog:**
- New dialog class `ParameterMapOptionsDialog`
- Slice selector dropdown
- ROI mode radio buttons
- "Use Existing" / "Draw New" ROI buttons

**Modify parameter mapping launch:**
- Show options dialog before running
- Pass selected options to `create_parameter_maps()`

### `proxyl_analysis/ui/parameter_maps.py` (NEW FILE)

Create dedicated UI for parameter map results:
- Display parameter maps with ROI overlay
- Metrics panel showing mean ± std within ROI
- Redraw ROI functionality
- Export metrics button

### `proxyl_analysis/run_analysis.py`

**Add CLI options:**
```
--param-slice N       Run parameter mapping on single slice N only
--param-roi-only      Restrict parameter mapping to ROI region
```

## Performance Estimates

| Mode | Voxels Processed | Relative Time |
|------|------------------|---------------|
| Full volume (128×128×9) | 147,456 | 1.0x (baseline) |
| Single slice (128×128×1) | 16,384 | ~0.11x |
| ROI only (~5% of slice) | ~820 | ~0.006x |
| Single slice + ROI | ~820 | ~0.006x |

## Export Format

Metrics saved to `{dataset_dir}/parameter_maps/roi_metrics.json`:

```json
{
  "roi": {
    "type": "contour",
    "z_slice": 4,
    "n_voxels": 342
  },
  "parameters": {
    "kb": {"mean": 0.0234, "std": 0.0089, "min": 0.008, "max": 0.051},
    "kd": {"mean": 0.0156, "std": 0.0042, "min": 0.009, "max": 0.028},
    "knt": {"mean": 0.0021, "std": 0.0008, "min": 0.001, "max": 0.004},
    "r_squared": {"mean": 0.923, "std": 0.045, "min": 0.78, "max": 0.99}
  },
  "processing": {
    "mode": "single_slice",
    "slice": 4,
    "roi_restricted": true,
    "fitting_method": "sliding_window",
    "window_size": 3
  }
}
```

## Dependencies

- Existing ROI selection workflow (for "Use Existing ROI")
- T013 (shares ROI contour overlay pattern)
