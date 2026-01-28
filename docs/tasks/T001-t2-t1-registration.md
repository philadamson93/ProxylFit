# T001: T2 to T1 Registration

**Status**: complete
**Priority**: high
**Created**: 2025-01-08
**Updated**: 2025-01-28

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
- [x] Documentation updated

## Implementation Details (January 2025)

### UI Enhancements
- **"Load from DICOM Folder" button** - Scans folder, auto-detects PROXYL and T2 series, allows loading both together
- **Progress dialog during T2 registration** - Shows "Registering T2 to T1..." with indeterminate progress bar
- **Qt-based T2 Registration Review Dialog** - Consistent with T1 review dialog:
  - 2x3 grid: T1 Reference, Registered T2, Overlay (T1=Red/T2=Green), Difference, Original T2, Info panel
  - Z-slice navigation via spinbox and arrow keys
  - Accept button in bottom-left
  - ProxylFit styling

### Session Persistence
- Registered T2 saved as DICOM series to `output/{dataset}/registered/dicoms/T2/`
- T2 automatically loaded when resuming a session
- Files: `z00.dcm`, `z01.dcm`, ..., `series_info.json`

### Key Files
- `proxyl_analysis/registration.py` - `register_t2_to_t1()`, `_visualize_t2_t1_registration()` (batch mode)
- `proxyl_analysis/ui/registration.py` - `T2RegistrationProgressDialog`, `T2RegistrationReviewDialog`, `run_t2_registration_with_progress()`
- `proxyl_analysis/io.py` - `save_registered_t2_as_dicom()`, `load_registered_t2()`
- `proxyl_analysis/run_analysis.py` - Integration in menu loop
