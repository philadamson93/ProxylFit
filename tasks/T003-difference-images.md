# T003: Difference Images

**Status**: planned
**Priority**: medium
**Created**: 2025-01-08

## Description

Generate difference images showing signal changes:
- **Contrast effect**: Post-injection minus pre-injection (shows contrast uptake)
- **Proxyl effect**: End-of-scan minus pre-injection (shows baseline shift over time)

## Rationale

- Difference images highlight signal changes due to contrast
- Separates contrast effect (acute) from proxyl effect (cumulative)
- Useful for visualization and quality assessment
- Standard approach in DCE-MRI analysis

## Definitions

| Term | Meaning |
|------|---------|
| **Contrast effect** | Signal change from contrast injection (post-injection vs pre-injection) |
| **Proxyl effect** | Baseline shift from start to end of scan (end-of-scan vs pre-injection) |

## Answered Questions

| Question | Answer |
|----------|--------|
| Which differences? | Post-contrast vs pre, and end-of-scan vs pre |
| Percent calculation? | Absolute difference only (no percent normalization needed) |
| Proxyl-effect definition | Difference in baseline vs end-of-scan |

## Requirements

- Compute difference images using averaged images from T002
- Absolute difference (no normalization)
- Visualize with appropriate colormaps
- Save difference images

## Implementation Notes

```python
def compute_difference_images(averaged_images):
    """
    Compute difference images from averaged images.

    Parameters
    ----------
    averaged_images : dict
        Output from compute_averaged_images():
        - 'pre_injection'
        - 'contrast_enhanced'
        - 'end_of_scan'

    Returns
    -------
    dict with:
        - 'contrast_effect': contrast_enhanced - pre_injection
        - 'proxyl_effect': end_of_scan - pre_injection
    """
    differences = {
        'contrast_effect': averaged_images['contrast_enhanced'] - averaged_images['pre_injection'],
        'proxyl_effect': averaged_images['end_of_scan'] - averaged_images['pre_injection'],
    }

    return differences
```

## Visualization

- **Colormap**: Diverging (e.g., `RdBu_r` or `coolwarm`)
  - Blue = signal decrease
  - White = no change
  - Red = signal increase
- **Colorbar**: Show absolute intensity units
- **Overlay option**: Show difference overlaid on anatomical image

## Dependency

This task depends on **T002 (Averaged Images)** - difference images are computed from the averaged images.

## CLI Interface

```bash
# Create difference images (requires averaged images)
python run_analysis.py --dicom data.dcm --z 4 --create-averaged-images --create-difference-images
```

## Acceptance Criteria

- [ ] Compute contrast effect difference (post - pre)
- [ ] Compute proxyl effect difference (end - pre)
- [ ] Visualize with diverging colormap
- [ ] Save difference images to output directory
- [ ] Option to overlay on anatomical
- [ ] Documentation updated
