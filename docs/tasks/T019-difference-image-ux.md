Reference: docs/claude_ops.md

# T019: Difference Image UX Improvements

**Status**: planned
**Priority**: medium
**Created**: 2026-02-06
**Source**: User feedback items #3, #4, #4.1

## Description

Three UX improvements to the image tools difference/averaged image workflow.

### A. Simplify Region Wording

**File:** `proxyl_analysis/ui/image_tools.py`

**Current:**
- Region A: `"Region A (Blue) - subtracted from B"` (line 382)
- Region B: `"Region B (Red)"` (line 221)

**Change:** Simplify to `"Region A"` and `"Region B"`. Keep color coding on the plot itself. Update the instructions text to clearly state `"Result = mean(Region B) - mean(Region A)"`.

### B. Improve Suggested Filenames

**File:** `proxyl_analysis/ui/image_tools.py` line 675

**Current:** `diff_t{a_start}-t{a_end}_minus_t{b_start}-t{b_end}`

**Change:** Reorder to match the computation (B minus A): `t{b_start}-t{b_end}_minus_t{a_start}-t{a_end}`

### C. Grayscale Default for Difference Images

**File:** `proxyl_analysis/ui/image_tools.py` line 626

**Current:** `cmap='RdBu_r'` (red-blue diverging)

**Change:** Switch to `cmap='gray'`.

## Verification

- Open difference image tool, verify simplified labels
- Save a difference image, verify filename follows B-minus-A format
- Verify difference image preview displays in grayscale
