**Status: Completed** (2026-02-25)

Reference: docs/claude_ops.md

# Fix Time Axis — Read Temporal Resolution from DICOM Metadata

## Goal

Fix Item 1 from Ralph's feedback (proxylfit_firstlook.pdf): the time axis uses hardcoded 70s per timepoint, but actual temporal resolution is 33.408s per timepoint.

## Root Cause

`create_time_array()` in `run_analysis.py` hardcodes 70s. The correct interval is computable from DICOM metadata:

```
temporal_resolution = AcquisitionDuration / (NumberOfFrames / n_z_slices)
                    = 4209.408 / (1134 / 9)
                    = 33.408 seconds
```

Verified on all 3 existing source DICOMs — all yield 33.408s.

## Approach

1. Add `extract_temporal_resolution()` to `io.py` to read from DICOM tags
2. Update `create_time_array()` default from 70s to 33s, add `temporal_resolution_s` parameter
3. Update all call sites to extract and pass DICOM-derived resolution
4. Add test coverage

## Files to Modify

| File | Changes |
|------|---------|
| `proxyl_analysis/io.py` | Add `extract_temporal_resolution()` function |
| `proxyl_analysis/run_analysis.py` | Update `create_time_array()` signature + default; update 6 call sites |
| `tests/test_user_feedback_fixes.py` | Add `TestTimeAxis` class (4 tests) |
| `docs/ralph_feedback.md` | New — tracking doc for all feedback items |

## Tests

| Test | Verifies |
|------|----------|
| `test_extract_temporal_resolution_from_real_dicom` | Returns ~33.4s on real source DICOM (skip if unavailable) |
| `test_create_time_array_default_is_33s` | Default interval is 33.0s, not 70s |
| `test_create_time_array_with_custom_resolution` | Custom `temporal_resolution_s` parameter works |
| `test_create_time_array_minutes_conversion` | Minutes array = seconds array / 60 |

## Verification

1. `uv run pytest tests/test_user_feedback_fixes.py -v -k TestTimeAxis` — 4/4 passed
2. `uv run pytest tests/ -v` — 157/157 passed, no regressions
