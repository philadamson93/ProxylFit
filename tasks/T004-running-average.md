# T004: Running Average Dynamic Images

**Status**: planned
**Priority**: medium
**Created**: 2025-01-08

## Description

Implement N-point running average (temporal smoothing) across the dynamic image series to reduce noise while preserving temporal dynamics.

## Rationale

- Individual frames can be noisy, especially at lower SNR
- Running average smooths noise while preserving signal trends
- Useful for visualization
- Common preprocessing step in DCE-MRI analysis

## Answered Questions

| Question | Answer |
|----------|--------|
| Averaging method | Simple average (uniform weights) |
| Use for fitting? | Visualization only by default, optional parameter to enable for fitting |

## Requirements

- Apply N-point simple moving average along time dimension
- Support configurable window size (2, 3, or more points)
- Handle edge frames appropriately
- Preserve original data, create smoothed copy
- Optional flag to use smoothed data for fitting

## Implementation Notes

```python
from scipy.ndimage import uniform_filter1d

def running_average(image_4d, window_size=3):
    """
    Apply simple running average along time dimension.

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    window_size : int
        Number of frames to average (default: 3)

    Returns
    -------
    smoothed_4d : np.ndarray
        Temporally smoothed image series (same shape)
    """
    smoothed = uniform_filter1d(image_4d, size=window_size, axis=3, mode='nearest')
    return smoothed
```

## Edge Handling

Using `mode='nearest'` - repeats edge values to maintain same output size.

## CLI Interface

```bash
# Create smoothed series for visualization (default: 3-point)
python run_analysis.py --dicom data.dcm --z 4 --smooth-temporal 3

# Use smoothed data for fitting (optional)
python run_analysis.py --dicom data.dcm --z 4 --smooth-temporal 3 --smooth-for-fitting
```

## Output

- Smoothed 4D series saved to output directory (optional)
- Visualization comparison: original vs smoothed single voxel time course

## Acceptance Criteria

- [ ] Implement simple running average with configurable window size
- [ ] Handle edge frames with nearest-neighbor padding
- [ ] CLI option `--smooth-temporal N` to enable
- [ ] CLI option `--smooth-for-fitting` to use smoothed data for kinetic fitting
- [ ] Visualize comparison of original vs smoothed
- [ ] Option to save smoothed series
- [ ] Documentation on when to use smoothing
