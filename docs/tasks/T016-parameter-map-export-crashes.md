Reference: docs/claude_ops.md

# T016: Fix Parameter Map Export Crashes

**Status**: planned
**Priority**: critical
**Created**: 2026-02-06
**Source**: User feedback items #10.1, #10.2

## Description

Both PNG and DICOM export from the parameter map results dialog crash.

### PNG Export — `show_roi_cb` AttributeError

**File:** `proxyl_analysis/ui/parameter_map_options.py` lines 913, 974, 977, 998

Code references `self.show_roi_cb` which doesn't exist. The actual attribute is `self.roi_checkbox` (line 589).

**Fix:** Replace all 4 occurrences of `self.show_roi_cb` with `self.roi_checkbox`.

### DICOM Export — int64 JSON Serialization

**File:** `proxyl_analysis/io.py` line 1308

`json.dumps(metadata)` fails when metadata contains numpy `int64`/`float64` values.

**Fix:** Convert numpy types to native Python types before serialization. Add a helper that walks the dict and calls `int()`/`float()` on numpy scalars, then use it at line 1308.

**Note:** User also reports "select parameter maps to export" yields an empty `kb_map` directory. May be a side-effect of the crash or a separate issue — investigate during fix.

## Verification

- Export parameter maps as PNG → no crash, files saved
- Export parameter maps as DICOM → no crash, non-empty output
