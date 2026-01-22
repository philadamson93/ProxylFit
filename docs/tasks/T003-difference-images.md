# T003: Interactive Difference Images

**Status**: complete
**Priority**: medium
**Created**: 2025-01-08
**Updated**: 2025-01-21

## Description

Extend the T002 averaging UI to support selecting TWO regions on the time curve. Each region is averaged, then subtracted (Region B - Region A) to produce a difference image. This allows users to flexibly define what comparison they want to make.

## Rationale

- More flexible than predefined "contrast effect" or "proxyl effect"
- User decides which time periods to compare
- Same familiar UI paradigm as T002
- Difference images highlight signal changes
- Useful for visualization, quality assessment, and analysis

## Example Use Cases

| Use Case | Region A | Region B |
|----------|----------|----------|
| Contrast effect | Pre-injection baseline | Post-injection peak |
| Proxyl effect | Early baseline | End-of-scan |
| Custom comparison | Any user-defined range | Any user-defined range |

## UI Design

Same UI as T002, but with mode toggle to select two regions:

1. Display the ROI time series curve
2. Toggle: "Single Region (Average)" vs "Two Regions (Difference)"
3. In difference mode:
   - First click-drag defines Region A (shown in blue)
   - Second click-drag defines Region B (shown in red)
4. Preview shows the difference image (B - A)
5. "Save Difference" button exports the result

### Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│  ProxylFit - Create Difference Image                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Mode: [Single Average] [Two Regions - Difference]  ← selected
│                                                             │
│   Signal                                                    │
│     ▲                                                       │
│     │      ████████                                         │
│     │     █        ████                                     │
│     │    █            ████                                  │
│     │   █                 ████████████                      │
│     │  █                                                    │
│     │ █   [==A==]              [====B====]                  │
│     └──────────────────────────────────────────────► Time   │
│           (blue)                (red)                       │
│                                                             │
│   Region A: frames 2-6 (pre-injection)                      │
│   Region B: frames 20-30 (post-injection)                   │
│   Result: B - A (contrast effect)                           │
│                                                             │
│   [Clear Regions]  [Preview Diff]  [Save Difference]        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Extend T002 UI with two-region selection mode
- Visual distinction between Region A (blue) and Region B (red)
- Compute difference as: mean(Region B) - mean(Region A)
- Preview difference image with diverging colormap
- Save difference image to output directory
- Allow re-selection of either region
- Handle edge cases (overlapping regions, empty selection)

## Implementation Notes

```python
def select_difference_regions_qt(time_array: np.ndarray,
                                  signal: np.ndarray,
                                  image_4d: np.ndarray,
                                  time_units: str = 'minutes') -> Tuple[np.ndarray, dict]:
    """
    Interactive Qt UI for selecting two time ranges and computing difference image.

    Parameters
    ----------
    time_array : np.ndarray
        Time points for x-axis
    signal : np.ndarray
        Signal values (ROI mean) for y-axis
    image_4d : np.ndarray
        Full 4D image data [x, y, z, t]
    time_units : str
        Units for time axis display

    Returns
    -------
    difference_image : np.ndarray
        3D difference image [x, y, z] = mean(B) - mean(A)
    selection_info : dict
        {
            'region_a': {'start_idx': int, 'end_idx': int},
            'region_b': {'start_idx': int, 'end_idx': int},
            'description': str  # e.g., "frames 20-30 minus frames 2-6"
        }
    """
    pass


def compute_difference_image(image_4d: np.ndarray,
                              region_a: Tuple[int, int],
                              region_b: Tuple[int, int]) -> np.ndarray:
    """
    Compute difference image: mean(region_b) - mean(region_a).

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    region_a : tuple
        (start_idx, end_idx) for Region A
    region_b : tuple
        (start_idx, end_idx) for Region B

    Returns
    -------
    np.ndarray
        Difference image [x, y, z]
    """
    avg_a = np.mean(image_4d[:, :, :, region_a[0]:region_a[1]], axis=3)
    avg_b = np.mean(image_4d[:, :, :, region_b[0]:region_b[1]], axis=3)
    return avg_b - avg_a
```

## Visualization

- **Colormap**: Diverging (e.g., `RdBu_r` or `coolwarm`)
  - Blue = signal decrease (B < A)
  - White = no change
  - Red = signal increase (B > A)
- **Colorbar**: Show absolute intensity units
- **Center at zero**: Colormap centered at 0 for symmetric display

## Save Format

Difference images saved as:
- **NPZ**: `difference_image_A{a_start}-{a_end}_B{b_start}-{b_end}.npz`
- Contains: difference image, region definitions, metadata
- **Optional PNG/TIFF**: For quick visualization with colormap

## CLI Integration

```bash
# Launch interactive difference UI (shares UI with averaging)
python run_analysis.py --dicom data.dcm --z 4 --create-difference-image

# Or combined flag that opens unified UI
python run_analysis.py --dicom data.dcm --z 4 --image-tools
```

## Acceptance Criteria

- [x] Extend T002 UI with two-region selection mode
- [x] Toggle between single-region (average) and two-region (difference) modes
- [x] Visual distinction for Region A vs Region B
- [x] Compute difference as mean(B) - mean(A)
- [x] Preview with diverging colormap (centered at 0)
- [x] Save difference image with metadata
- [x] Handle edge cases (overlapping regions, empty selection)
- [x] Documentation updated

## Implementation Notes (Actual)

Implemented in `ImageToolsDialog` class in `proxyl_analysis/ui.py`:
- Unified dialog with T002, mode toggle at top
- Region A (blue) and Region B (red) with separate controls
- Click "Select on Plot" for each region, then click start/end
- RdBu_r diverging colormap centered at 0
- Saves as NPZ with region metadata

## Dependency

Builds on **T002 (Interactive Averaged Images)** - shares the same UI foundation with added two-region mode.
