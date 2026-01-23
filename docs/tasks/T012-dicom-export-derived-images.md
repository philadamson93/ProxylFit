# T012: Export Averaged/Difference Images as DICOM

**Status**: planned
**Priority**: medium
**Created**: 2025-01-22

## Description

Export averaged and difference images as DICOM files with descriptive filenames indicating the operation and timepoints used. Include option to rename before saving.

## Current Implementation

Averaged and difference images are saved as `.npz` files:
- `averaged_image_frames{start}-{end}.npz`
- `difference_image_A{a_start}-{a_end}_B{b_start}-{b_end}.npz`

## Proposed Implementation

### Filename Convention

**Averaged images:**
```
avg_t{start}-t{end}_{n_frames}frames.dcm
# Example: avg_t0-t5_6frames.dcm
```

**Difference images:**
```
diff_t{a_start}-t{a_end}_minus_t{b_start}-t{b_end}.dcm
# Example: diff_t0-t5_minus_t10-t15.dcm
```

### Rename Option

Before saving, show dialog with:
- Auto-generated filename (editable)
- Preview of operation details
- Save / Cancel buttons

### DICOM Tags

Store operation details in DICOM metadata:

| Tag | Value |
|-----|-------|
| SeriesDescription | "Averaged t0-t5 (6 frames)" or "Difference t0-t5 minus t10-t15" |
| ImageComments | Full operation description |
| SeriesNumber | Original + 2000 (averaged) or + 3000 (difference) |
| DerivationDescription | Algorithm details |
| SourceImageSequence | Reference to source registered series (if available) |

### Export Location

Save to `{dataset_dir}/image_tools/` (per T011 structure) or user-selected location.

## Files to Modify

### `proxyl_analysis/ui/image_tools.py`

- Modify `_save_preview()` method to offer DICOM export
- Add DICOM writing using pydicom
- Add rename dialog before save
- Store operation metadata in DICOM tags

### `proxyl_analysis/io.py`

Add helper function:
```python
def save_derived_image_as_dicom(
    image: np.ndarray,
    output_path: str,
    operation_type: str,  # "averaged" or "difference"
    operation_params: dict,  # timepoints, etc.
    spacing: Tuple[float, float, float],
    source_dicom: str = None
) -> str:
    """Save a derived 2D/3D image as DICOM with descriptive metadata."""
```

## UI Changes

### Save Dialog

```
┌─────────────────────────────────────────────┐
│ Save Derived Image                          │
├─────────────────────────────────────────────┤
│ Operation: Difference (t0-t5 minus t10-t15) │
│                                             │
│ Filename: [diff_t0-t5_minus_t10-t15.dcm  ]  │
│                                             │
│ Format:   ○ DICOM (Recommended)             │
│           ○ NumPy (.npz)                    │
│           ○ NIfTI (.nii.gz)                 │
│                                             │
│           [Cancel]  [Save]                  │
└─────────────────────────────────────────────┘
```

## Dependencies

- T010 (DICOM export infrastructure)
- T011 (self-contained dataset structure for save location)
