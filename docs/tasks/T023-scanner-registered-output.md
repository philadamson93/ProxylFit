# T023: Fix DICOM Scanner for Registered Output

**Status**: completed
**Priority**: high
**Created**: 2026-02-20

## Description

The DICOM scanner did not recognize series from registered output directories.
When a user pointed "Load from DICOM Folder" at a registered output directory,
the scanner found all series but the T1/T2 dropdowns were stuck on "-- None --"
because the detection heuristics only matched raw scanner descriptions.

## Root Cause

`proxyl_analysis/dicom_scanner.py` lines 115-119 used detection heuristics that
only matched raw scanner descriptions:
- T1: required `"proxyl"` or `"flash" + >100 frames`
- T2: required both `"t2"` AND `"turbo"`

But registered output uses `"Registered T1 DCE"` and `"Registered T2"` -- neither matched.

## Fix

### Scanner Detection (dicom_scanner.py)

Added two `or` conditions to the detection heuristics:
- `'registered t1' in desc_lower` for PROXYL detection
- `'registered t2' in desc_lower` for T2 detection

### Loading Registered Data (run_analysis.py)

Added `_detect_registered_session()` helper that checks if a DICOM file path is
inside a `registered/dicoms/` directory structure. When `load_from_scan` selects
registered data, the code now loads via `load_registration_data()` instead of
`load_dicom_series()`, skipping unnecessary re-registration.

## Files Modified

- `proxyl_analysis/dicom_scanner.py` - Added registered output detection
- `proxyl_analysis/run_analysis.py` - Added `_detect_registered_session()` helper,
  modified both `load_from_scan` handlers (initial and menu loop)

## Acceptance Criteria

- [x] Scanner detects "Registered T1 DCE" as PROXYL series
- [x] Scanner detects "Registered T2" as T2 series
- [x] DicomScanResultsDialog combo boxes populate for registered output
- [x] Loading registered data from scanner uses correct load path
- [x] Tests cover all acceptance criteria
