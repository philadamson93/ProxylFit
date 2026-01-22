# T001: T2 to T1 Registration

**Status**: implemented
**Priority**: high
**Created**: 2025-01-08

## Description

Register T2-weighted MRI series to match the T1 series. T2 images provide better tumor volume definition (RANO criteria) and are more useful for segmentation and ROI selection.

## Rationale

- T2 images have better soft tissue contrast for tumor boundaries
- RANO (Response Assessment in Neuro-Oncology) criteria use T2/FLAIR for tumor measurement
- ROI selection on T2 can be transferred to T1 for kinetic analysis
- Enables multi-modal analysis workflow

## Requirements

- Load T2 DICOM series alongside T1
- Perform rigid or affine registration of T2 to T1 space
- Apply transformation to align T2 voxels with T1
- Support ROI selection on registered T2
- Transfer ROI mask to T1 coordinate space

## Answered Questions

| Question | Answer |
|----------|--------|
| Same session? | Yes, T2 and T1 acquired in same session |
| T2 type? | Single volume (not time series) |
| Resolution? | Likely different resolution (different sequence), support both same and different |

## Implementation Notes

- Leverage existing SimpleITK registration infrastructure from `registration.py`
- Use mutual information metric (works well for multi-modal)
- Support resampling for different resolutions
- Add visualization to verify T2-T1 alignment quality
- ROI selected on T2 needs coordinate transformation to T1 space

## Proposed Workflow

```
1. Load T1 4D series (existing)
2. Load T2 volume (new CLI arg: --t2 path/to/t2.dcm)
3. Register T2 → T1 reference frame (first T1 timepoint)
4. Resample T2 to T1 grid if different resolution
5. Display registered T2 for ROI selection
6. User draws ROI on T2 (better tumor visibility)
7. ROI mask already in T1 coordinates (after resampling)
8. Continue with kinetic analysis on T1
```

## CLI Interface

```bash
python run_analysis.py --dicom t1_data.dcm --t2 t2_data.dcm --z 4 --roi-mode contour
```

## Acceptance Criteria

- [x] CLI option `--t2` to load T2 DICOM
- [x] T2 registered to T1 coordinate space
- [x] Resampling works for different resolutions
- [x] Visual verification of registration quality
- [x] ROI selection works on registered T2
- [x] ROI mask in T1 coordinates for analysis
- [ ] Documentation updated
