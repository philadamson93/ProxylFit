# ProxylFit Implementation Plan

**Created**: 2025-01-15
**Updated**: 2025-01-28
**Status**: In Progress

## Overview

This document tracks the implementation of the main workflow menu (T006) and related features.

## Current Status

- **T007**: Registration Progress UI - complete
- **T006 Phase 1**: Main Menu with decoupled ROI workflow - complete
- **T002**: Averaged Images - complete
- **T003**: Difference Images - complete
- **T001 Enhancements**: T2 Registration UI improvements - complete

## Implementation Order

| Phase | Task | Description | Status |
|-------|------|-------------|--------|
| 0 | T007 | Registration Progress UI | **complete** |
| 1 | T006 Phase 1 | Main Menu with decoupled ROI workflow | **complete** |
| 2 | T002 | Averaged Images (time-curve region selection) | **complete** |
| 3 | T003 | Difference Images (two-region selection) | **complete** |
| 4 | T005 | Pixel-Level Parameter Maps | planned |
| 5 | T006 Phase 2 | Polish, additional export formats | planned |

**Backlog**: T004 (Temporal Smoothing)

## Completed Features

### T007: Registration Progress UI
- Background QThread for registration
- Progress dialog with live metrics plots (translation, MSE)
- Qt-based registration review dialog
- Splash window shown during registration

### T006 Phase 1: Main Menu
- Experiment section (Load New T1, Load Previous Session)
- Data status header (T1 info, T2 load button)
- ROI Analysis section with decoupled workflow:
  - "Draw ROI" button -> ROI selection -> injection time -> return to menu
  - "Run Kinetic Fit" button (enabled after ROI + injection time)
- Parameter Maps section (sliding window)
- Image Tools section (enabled after ROI + injection time)
- Export section (registered data, report, time series CSV)
- Scroll area to prevent UI occlusion

### T002/T003: Image Tools
- Unified `ImageToolsDialog` with mode toggle
- Averaged Image mode: single region selection
- Difference Image mode: two regions (B - A)
- Click-to-select interaction with synced spinboxes
- Z-slice slider for preview navigation
- NPZ export with metadata
- RdBu_r diverging colormap for difference images

### T001 Enhancements: T2 Registration UI (January 2025)
- **"Load from DICOM Folder" button** - Renamed from "Scan DICOM Folder", scans and auto-detects PROXYL/T2 series
- **T2 selection in scan dialog** - Can select both T1 and T2 together for loading
- **T2 registration progress dialog** - Shows indeterminate progress while registering T2 to T1
- **Qt-based T2 Registration Review Dialog** - Replaces matplotlib version:
  - 2x3 grid: T1 Reference, Registered T2, Overlay, Difference, Original T2, Info
  - Z-slice navigation (spinbox + arrow keys)
  - Accept button in bottom-left
- **T2 session persistence** - Registered T2 saved as DICOM to `registered/dicoms/T2/`, auto-loaded on session resume
- **Files**: `io.py` (save/load T2), `ui/registration.py` (progress + review dialogs), `run_analysis.py` (integration)

## T005: Pixel-Level Maps (Next)

### Scope
- Add pixel-level option to Parameter Maps section
- Progress bar with cancel support
- NaN for failed fits
- All parameter maps (kb, kd, knt, etc.)

### Files to Modify
- `proxyl_analysis/parameter_mapping.py` - Add `fit_pixel_level_maps()`
- `proxyl_analysis/ui.py` - Progress dialog

## Notes

- All UI uses PySide6 (Qt)
- Output format is NPZ (other formats in future)
- Registration is automatic (always runs unless `--batch` with `--skip-registration`)
- ROI workflow returns to menu after each operation (no more app exit after fitting)
