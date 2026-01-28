# Changelog

All notable changes to ProxylFit are documented here.

## [1.2.0] - January 2025

### Added
- **T2 Registration UI Enhancements**
  - "Load from DICOM Folder" button - auto-detects PROXYL and T2 series
  - T2 registration progress dialog with indeterminate progress bar
  - Qt-based T2 Registration Review Dialog (replaces matplotlib version)
    - 2x3 grid showing T1, registered T2, overlay, difference, original T2, and info
    - Z-slice navigation via spinbox and arrow keys
    - Accept button positioned at bottom-left
  - T2 session persistence - registered T2 saved as DICOM and auto-loaded on resume

### Changed
- Renamed "Scan DICOM Folder" to "Load from DICOM Folder"
- T2 visualization now uses Qt dialog for consistency with rest of app

### Files Modified
- `proxyl_analysis/io.py` - Added `save_registered_t2_as_dicom()`, `load_registered_t2()`
- `proxyl_analysis/ui/registration.py` - Added `T2RegistrationProgressDialog`, `T2RegistrationReviewDialog`, `run_t2_registration_with_progress()`
- `proxyl_analysis/run_analysis.py` - T2 loading and session resume integration

---

## [1.1.0] - January 2025

### Added
- **Qt-based UI** (`ui.py`) - Modern PySide6 interface replacing matplotlib widgets
  - `ROISelectorDialog` - Rectangle ROI selection with proper layout
  - `ManualContourDialog` - Contour drawing with Z-slice navigation
  - `InjectionTimeSelectorDialog` - Time point selection with CSV export
  - `FitResultsDialog` - Fit visualization with parameter display
- **Navigation toolbar** - Pan, zoom, and save functionality for all plots
- **Responsive layouts** - Windows resize properly without element overlap
- **Consistent styling** - Professional appearance via Qt stylesheets
- **Documentation** - Comprehensive docs in `docs/` folder
  - Installation guide
  - Qt UI guide
  - Module documentation
  - Quick start guide

### Changed
- `run_analysis.py` now uses Qt-based UI functions by default
- Updated `__init__.py` to export new Qt components
- Bumped version to 1.1.0

### Dependencies
- Added `PySide6>=6.5.0` to requirements

### Migration
The original matplotlib-based functions remain available for backwards compatibility:
```python
# Old (still works)
from proxyl_analysis.roi_selection import select_rectangle_roi

# New (recommended)
from proxyl_analysis.ui import select_rectangle_roi_qt
```

---

## [1.0.0] - November 2024

### Initial Release
- DICOM loading and 4D tensor reshaping
- Rigid registration with SimpleITK
- ROI selection modes: rectangle, contour, SegmentAnything
- Extended kinetic model fitting
- Parameter mapping with sliding window
- matplotlib-based interactive UI
- CLI entry point (`run_analysis.py`)

### Features
- Extended kinetic model: `I(t) = A0 + A1*(1-exp(-kb*(t-t0)))*exp(-kd*(t-t0)) + A2*(1-exp(-knt*(t-tmax)))`
- Automatic registration caching
- CSV export for timecourse data
- Comprehensive fit quality metrics

---

## Version Numbering

ProxylFit follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible
