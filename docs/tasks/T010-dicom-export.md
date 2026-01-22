# T010: Export Registered Data as DICOM Series

## Status: Planned

## Overview

Change how registered 4D data is saved and loaded. Instead of saving as a single `.npz` file, save as N separate DICOM files where each DICOM represents a 3D volume (x, y, z) at a specific time point t.

## Current Implementation

**Save format:** `registered_4d_data.npz`
- Contains: `registered_4d` array [x, y, z, t] + `spacing` array
- Location: `{output_dir}/registered_4d_data.npz`
- Companion file: `registration_metrics.json`

**Data dimensions:** Typically [128, 128, 9, 26] = (x, y, z, t)

## Proposed Implementation

**Save format:** N DICOM files in a subdirectory
- Location: `{output_dir}/registered_dicoms/`
- Files: `registered_t{:03d}.dcm` (e.g., `registered_t000.dcm` through `registered_t025.dcm`)
- Each file: 3D volume [x, y, z] for that time point
- Metadata file: `registered_dicoms/series_info.json` (maps time indices, stores spacing, etc.)

## Files Requiring Changes

### 1. `proxyl_analysis/io.py`

**Add new functions:**

```python
def save_registered_as_dicom_series(
    registered_4d: np.ndarray,
    spacing: Tuple[float, float, float],
    output_dir: str,
    series_description: str = "Registered T1 DCE",
    source_dicom: str = None  # Optional: copy metadata from source
) -> str:
    """
    Save registered 4D data as a series of DICOM files.

    Parameters
    ----------
    registered_4d : np.ndarray
        4D array with shape [x, y, z, t]
    spacing : tuple
        Voxel spacing (x, y, z) in mm
    output_dir : str
        Output directory (will create 'registered_dicoms' subdirectory)
    series_description : str
        DICOM SeriesDescription tag value
    source_dicom : str, optional
        Path to source DICOM to copy patient/study metadata from

    Returns
    -------
    str
        Path to the created DICOM series directory
    """

def load_registered_dicom_series(
    series_dir: str
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a registered DICOM series back into 4D array.

    Parameters
    ----------
    series_dir : str
        Path to directory containing registered DICOM files

    Returns
    -------
    registered_4d : np.ndarray
        4D array with shape [x, y, z, t]
    spacing : tuple
        Voxel spacing (x, y, z) in mm
    """
```

**Implementation details:**
- Use `pydicom` for DICOM writing (already a dependency)
- Copy relevant DICOM tags from source if provided (PatientID, StudyInstanceUID, etc.)
- Generate new SeriesInstanceUID and SOPInstanceUIDs
- Store temporal index in DICOM tags (TemporalPositionIdentifier or similar)
- Save `series_info.json` with:
  - `n_timepoints`: int
  - `spacing`: [x, y, z]
  - `shape`: [x, y, z, t]
  - `time_indices`: list of time indices
  - `source_dicom`: original source path
  - `created_at`: timestamp

### 2. `proxyl_analysis/registration.py`

**Modify `save_registration_data()` (lines 182-236):**
- Call new `save_registered_as_dicom_series()` instead of `np.savez_compressed()`
- Keep `registration_metrics.json` as-is (still useful)
- Update path references in saved metadata

**Modify `load_registration_data()` (lines 239-283):**
- Call new `load_registered_dicom_series()` instead of `np.load()`
- Handle backward compatibility: check for both old `.npz` format and new DICOM format
- Prioritize DICOM format if both exist

```python
def load_registration_data(output_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float], List[RegistrationMetrics]]:
    output_path = Path(output_dir)

    # Check for new DICOM format first
    dicom_dir = output_path / "registered_dicoms"
    if dicom_dir.exists():
        registered_4d, spacing = load_registered_dicom_series(str(dicom_dir))
    else:
        # Fall back to legacy .npz format
        reg_data_file = output_path / "registered_4d_data.npz"
        if not reg_data_file.exists():
            raise FileNotFoundError(...)
        data = np.load(reg_data_file)
        registered_4d = data['registered_4d']
        spacing = tuple(data['spacing'])

    # Load metrics (unchanged)
    ...
```

### 3. `proxyl_analysis/run_analysis.py`

**Lines affected:**
- Line 368: `load_registration_data()` call - no change needed (function interface unchanged)
- Line 995: `timeseries_data.npz` save - keep as-is (different data)
- Any print statements referencing `.npz` files - update messages

**Update export messages:**
- Line ~652-655: Update message about registered data location

### 4. `proxyl_analysis/ui/main_menu.py`

**Update `_handle_export()` method:**
- Line ~408-426: Update message for 'registered_data' export type
- Change from: `"Registered data saved to: {output_dir}/registered_4d_data.npz"`
- Change to: `"Registered data saved to: {output_dir}/registered_dicoms/"`

### 5. `proxyl_analysis/ui/registration.py`

**Review for any direct references to .npz files:**
- The UI code uses `load_registration_data()` so should work without changes
- Verify no hardcoded `.npz` paths

### 6. Documentation Updates

**`docs/io.md`:**
- Document new DICOM export/import functions
- Document file naming convention
- Document metadata stored in series_info.json

**`docs/index.md`:**
- Update data flow section to reflect DICOM export

## DICOM Tag Strategy

For each saved DICOM file:

| Tag | Value |
|-----|-------|
| PatientID | Copy from source or generate |
| StudyInstanceUID | Copy from source or generate |
| SeriesInstanceUID | Generate new (unique to registered series) |
| SOPInstanceUID | Generate new (unique per file) |
| SeriesDescription | "Registered T1 DCE" |
| SeriesNumber | Original + 1000 (to distinguish) |
| InstanceNumber | Time index (1-based) |
| TemporalPositionIdentifier | Time index |
| NumberOfTemporalPositions | Total timepoints |
| PixelSpacing | From spacing parameter |
| SliceThickness | From spacing[2] |
| ImageOrientationPatient | Standard axial or copy from source |
| Modality | MR |

## Testing Plan

1. **Unit tests for io.py:**
   - Test save/load round-trip preserves data exactly
   - Test with various array sizes
   - Test metadata preservation

2. **Integration tests:**
   - Run full registration workflow
   - Verify DICOM files are valid (can be opened in DICOM viewers)
   - Verify load works correctly

3. **Backward compatibility:**
   - Test loading old `.npz` format still works
   - Test mixed scenarios (old data + new code)

## Migration Notes

- Old `.npz` files will still be loadable (backward compatible)
- No automatic migration of existing data needed
- New registrations will save as DICOM by default

## Dependencies

- `pydicom` - already in requirements (used for reading)
- Verify pydicom version supports writing Enhanced DICOM or multi-frame

## Estimated Scope

- **io.py**: ~150 lines new code
- **registration.py**: ~30 lines modified
- **run_analysis.py**: ~5 lines modified
- **ui/main_menu.py**: ~5 lines modified
- **Documentation**: ~50 lines
- **Tests**: ~100 lines
