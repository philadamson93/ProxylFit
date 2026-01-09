# T002: Averaged Image Generation

**Status**: planned
**Priority**: medium
**Created**: 2025-01-08

## Description

Generate averaged images from the T1 time series for visualization and analysis:
- Full time series average
- Pre-injection average (first N frames before injection)
- Post-scan average (last N frames at end of scan)
- Contrast-enhanced average (N frames starting from injection)

## Rationale

- Averaged images reduce noise and improve SNR
- Pre/post comparison shows contrast effect
- Contrast-enhanced window captures peak signal
- Useful for quality assessment and presentation

## Requirements

- Compute mean across specified frame ranges
- Frame ranges derived from user-selected injection time
- Handle edge cases (fewer frames than specified)
- Save averaged images to output
- Visualize averaged images

## Answered Questions

| Question | Answer |
|----------|--------|
| Frame indexing | Code uses 0-indexed (Python standard). Frame 1 = index 0 |
| Post-injection frames | Use last N frames of scan (not fixed 122-126) |
| Contrast window | Starts at injection click, length configurable |

## Frame Range Definitions

Based on injection time selected by user:

| Image Type | Frame Range | Default N |
|------------|-------------|-----------|
| Pre-injection | `[0 : injection_idx]` or first 5 | 5 |
| Contrast-enhanced | `[injection_idx : injection_idx + N]` | configurable |
| End-of-scan | `[-N : ]` (last N frames) | 5 |
| Full average | `[0 : ]` (all frames) | all |

## Implementation Notes

```python
def compute_averaged_images(image_4d, injection_idx, contrast_window=24, n_avg=5):
    """
    Compute averaged images based on injection time.

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    injection_idx : int
        Frame index when injection occurred (from user click)
    contrast_window : int
        Number of frames after injection to average for contrast-enhanced
    n_avg : int
        Number of frames for pre/post averages
    """
    n_frames = image_4d.shape[3]

    averaged = {
        'full': np.mean(image_4d, axis=3),
        'pre_injection': np.mean(image_4d[:,:,:, max(0, injection_idx-n_avg):injection_idx], axis=3),
        'contrast_enhanced': np.mean(image_4d[:,:,:, injection_idx:injection_idx+contrast_window], axis=3),
        'end_of_scan': np.mean(image_4d[:,:,:, -n_avg:], axis=3),
    }

    return averaged
```

## CLI Interface

```bash
# Default: 5-frame averages, 24-frame contrast window
python run_analysis.py --dicom data.dcm --z 4 --create-averaged-images

# Custom contrast window
python run_analysis.py --dicom data.dcm --z 4 --create-averaged-images --contrast-window 30

# Custom averaging width
python run_analysis.py --dicom data.dcm --z 4 --create-averaged-images --avg-frames 3
```

## Acceptance Criteria

- [ ] Compute averaged images based on injection time
- [ ] Configurable contrast window length
- [ ] Configurable number of frames for pre/post averages
- [ ] Handle edge cases (short scans, injection near start/end)
- [ ] Save averaged images to output directory
- [ ] Visualize averaged images (montage)
- [ ] Documentation updated
