Reference: docs/claude_ops.md

# T021: Improve Session Loading UX

**Status**: planned
**Priority**: low
**Created**: 2026-02-06
**Source**: User feedback item #2

## Description

User had difficulty loading previously registered DICOMs directly. Likely operator error, but the UX could be more forgiving.

### Potential Improvements

**Files:** `proxyl_analysis/ui/main_menu.py`, `proxyl_analysis/run_analysis.py`

- Add clearer error messages when selected directory doesn't match expected structure
- Show what structure is expected (e.g., "Looking for `registered/dicoms/z00_t000.dcm`")
- Possibly add visual indicators of valid session directories in the file browser

## Verification

- Select an invalid directory → verify helpful error message is shown
- Select a valid session directory → verify it loads correctly
