# Session State - January 15, 2025

## Summary

Planning and initial implementation session for ProxylFit UI improvements.

## What Was Done

### Planning
1. Reviewed repository structure and documentation
2. Updated task plans:
   - **T002**: Averaged Images - changed to interactive UI with region selection
   - **T003**: Difference Images - changed to two-region selection mode
   - **T004**: Temporal Smoothing - moved to **backlog** (fitting works on noisy data)
   - **T006**: Created new task for Main Workflow Menu
3. Created `docs/testing-plan.md` - comprehensive testing strategy
4. Created `docs/implementation-plan.md` - implementation order and approach

### Implementation (T006 Phase 1)
1. **`proxyl_analysis/ui.py`** - Added `MainMenuDialog` class with:
   - Experiment section (Load New T1 DICOM, Load Previous Session)
   - Data status header (T1/T2 info, registration status)
   - ROI Analysis section (T2 default source, method, z-slice)
   - Parameter Maps section (sliding window controls)
   - Image Tools section (disabled placeholders for T002/T003)
   - Export section

2. **`proxyl_analysis/run_analysis.py`** - Added menu integration:
   - `--dicom` now optional
   - `--batch` flag to skip menu
   - Menu shows after registration

## Current Status

| Task | Status |
|------|--------|
| T006 Phase 1: Main Menu | **Ready for testing** |
| T002: Averaged Images | Pending |
| T003: Difference Images | Pending |
| T005: Pixel-Level Maps | Pending |
| T004: Temporal Smoothing | Backlog |

## Next Steps

1. **Manual Testing Checkpoint 1** - Test the main menu:
   ```bash
   # Without --dicom
   python proxyl_analysis/run_analysis.py

   # With --dicom
   python proxyl_analysis/run_analysis.py --dicom /path/to/test.dcm

   # Batch mode (skips menu)
   python proxyl_analysis/run_analysis.py --dicom /path/to/test.dcm --batch
   ```

2. After testing passes, implement **T002** (Averaged Images dialog)

3. Then **T003** (Difference Images - extends T002)

4. Then **T005** (Pixel-Level Parameter Maps)

## Key Decisions Made

| Decision | Choice |
|----------|--------|
| ROI default source | T2 (better tumor contrast) |
| Image Tools prerequisite | Requires ROI selection first |
| Temporal smoothing | Backlog - not needed, fitting works on noisy data |
| Export format | NPZ only for now |
| App launch without --dicom | Allowed - menu lets user load data |

## Files Changed

```
proxyl_analysis/ui.py          # Added MainMenuDialog (~550 lines)
proxyl_analysis/run_analysis.py # Menu integration
docs/tasks/T002-averaged-images.md
docs/tasks/T003-difference-images.md
docs/tasks/T004-running-average.md (moved to backlog)
docs/tasks/T006-tools-menu.md (created)
docs/testing-plan.md (created)
docs/implementation-plan.md (created)
```

## To Resume

Tell Claude:
> "Resume ProxylFit implementation. We completed T006 Phase 1 (Main Menu).
> Next: run manual testing checkpoint 1, then implement T002 (Averaged Images)."
