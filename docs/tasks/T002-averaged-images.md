# T002: Interactive Averaged Image Generation

**Status**: complete
**Priority**: medium
**Created**: 2025-01-08
**Updated**: 2025-01-21

## Description

Provide an interactive UI for generating averaged images from the T1 time series. Users select a time range on the signal uptake curve (same UI paradigm as injection time selection) to define which frames to average.

## Rationale

- More flexible than predefined frame ranges
- User can interactively choose regions of interest
- Same familiar UI as injection time selection
- Averaged images reduce noise and improve SNR
- Useful for quality assessment and presentation

## UI Design

Reuse/extend the injection time selection Qt UI with region highlighting:

1. Display the ROI time series curve
2. User clicks and drags to select a time range (highlighted region)
3. Preview shows the averaged image for selected range
4. "Save Average" button exports the result
5. Option to select another region or close

### Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│  ProxylFit - Create Averaged Image                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Signal                                                    │
│     ▲                                                       │
│     │      ████████                                         │
│     │     █        ████                                     │
│     │    █            ████                                  │
│     │   █                 ████████████                      │
│     │  █  [====SELECTED====]                                │
│     │ █                                                     │
│     └──────────────────────────────────────────────► Time   │
│                                                             │
│   Click and drag on the curve to select a time range        │
│                                                             │
│   Selected: frames 5-15 (10 frames)                         │
│                                                             │
│   [Preview Average]  [Save Average]  [Close]                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Interactive region selection on time series plot
- Visual feedback showing selected range
- Preview of averaged image before saving
- Save averaged image to output directory
- Support for multiple region selections in one session
- Handle edge cases (empty selection, single frame)

## Implementation Notes

```python
def select_average_region_qt(time_array: np.ndarray,
                              signal: np.ndarray,
                              image_4d: np.ndarray,
                              time_units: str = 'minutes') -> Tuple[np.ndarray, dict]:
    """
    Interactive Qt UI for selecting time range and generating averaged image.

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
    averaged_image : np.ndarray
        3D averaged image [x, y, z]
    selection_info : dict
        {'start_idx': int, 'end_idx': int, 'n_frames': int, 'time_range': tuple}
    """
    # Qt dialog with:
    # - Time series plot with click-drag region selection
    # - Highlighted region showing selected frames
    # - Preview pane showing averaged image
    # - Save button to export
    pass


def compute_averaged_image(image_4d: np.ndarray,
                           start_idx: int,
                           end_idx: int) -> np.ndarray:
    """
    Compute averaged image over specified frame range.

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    start_idx : int
        Starting frame index (inclusive)
    end_idx : int
        Ending frame index (exclusive)

    Returns
    -------
    np.ndarray
        Averaged 3D image [x, y, z]
    """
    return np.mean(image_4d[:, :, :, start_idx:end_idx], axis=3)
```

## Save Format

Averaged images saved as:
- **NPZ**: `averaged_image_{start}-{end}.npz` containing image data and metadata
- **Optional PNG/TIFF**: For quick visualization

## CLI Integration

```bash
# Launch interactive averaging UI after ROI selection
python run_analysis.py --dicom data.dcm --z 4 --create-averaged-image

# This opens the Qt UI after ROI time series is computed
```

## Acceptance Criteria

- [x] Qt UI for interactive region selection on time curve
- [x] Click-drag to select frame range (click start, click end)
- [x] Visual highlighting of selected region
- [x] Preview averaged image before saving
- [x] Save averaged image with metadata
- [x] Handle edge cases (single frame, empty selection)
- [x] Documentation updated

## Implementation Notes (Actual)

Implemented in `ImageToolsDialog` class in `proxyl_analysis/ui.py`:
- Combined with T003 in unified dialog with mode toggle
- Click "Select on Plot" then click twice (start, end) to select range
- Start/End spinboxes synced with plot clicks
- Z-slice slider for preview navigation
- Saves as NPZ with metadata

## Relationship to T003

This task provides the foundation for T003 (Difference Images). T003 extends this UI to support selecting TWO regions and computing their difference.
