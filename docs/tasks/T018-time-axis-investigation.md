Reference: docs/claude_ops.md

# T018: Investigate and Fix Time Axis

**Status**: planned
**Priority**: high
**Created**: 2026-02-06
**Source**: User feedback items #1, #1.1

## Description

Time axis values are incorrect — user reports 66 seconds displayed vs 33 seconds actual. The code hardcodes 70 seconds per frame in `create_time_array()`.

**File:** `proxyl_analysis/run_analysis.py` lines 43-70

```python
time_array = np.arange(num_timepoints, dtype=float) * (70.0 / 60.0)  # 70s per frame
```

## Plan

1. **Investigate DICOM metadata** from a sample file. Check tags:
   - `AcquisitionTime` / `ContentTime` (differences between consecutive frames)
   - `TriggerTime`
   - `TemporalPositionIdentifier`
   - `RepetitionTime` (TR)
   - Any private/vendor-specific timing tags

2. **Based on findings**, either:
   - Parse the interval from DICOM metadata automatically (most robust)
   - Update the hardcoded default to the correct value
   - Prompt the user to confirm/enter the interval on load

## Verification

- Compare displayed time values against known protocol timing
- Verify kinetic fit parameters are in correct units after the fix
