# T009: Refactor ui.py into Modular Package

**Status**: planned
**Priority**: low
**Created**: 2025-01-22

## Description

Refactor the monolithic `proxyl_analysis/ui.py` file (3100+ lines) into a modular `ui/` package with separate files for each dialog type.

## Rationale

- **Maintainability**: Large files are harder to navigate and understand
- **Git conflicts**: Multiple developers working on different dialogs cause merge conflicts
- **Testing**: Easier to test individual components in isolation
- **IDE performance**: Some IDEs struggle with very large files
- **Separation of concerns**: Each dialog has distinct functionality

## Current Structure

```
proxyl_analysis/
  ui.py  (3100+ lines)
    - PROXYLFIT_STYLE (stylesheet)
    - HeaderWidget, ButtonBar (reusable components)
    - RegistrationProgressDialog
    - RegistrationReviewDialog
    - ManualContourDialog
    - ROISelectorDialog
    - InjectionTimeSelectorDialog
    - FitResultsDialog
    - MainMenuDialog
    - ImageToolsDialog
    - Helper functions (init_qt_app, compute_averaged_image, etc.)
```

## Proposed Structure

```
proxyl_analysis/
  ui/
    __init__.py           # Public API exports, init_qt_app()
    styles.py             # PROXYLFIT_STYLE, theme constants
    components.py         # HeaderWidget, ButtonBar, InstructionBox
    registration.py       # RegistrationProgressDialog, RegistrationReviewDialog
    roi.py                # ROISelectorDialog, ManualContourDialog
    injection.py          # InjectionTimeSelectorDialog
    fitting.py            # FitResultsDialog
    image_tools.py        # ImageToolsDialog, compute_averaged_image, compute_difference_image
    main_menu.py          # MainMenuDialog
```

## Implementation Steps

### Step 1: Create ui/ package structure
```python
# ui/__init__.py
from .styles import PROXYLFIT_STYLE
from .components import HeaderWidget, ButtonBar
from .registration import RegistrationProgressDialog, RegistrationReviewDialog, show_registration_review_qt
from .roi import ROISelectorDialog, ManualContourDialog, select_rectangle_roi_qt, select_manual_contour_roi_qt
from .injection import InjectionTimeSelectorDialog, select_injection_time_qt
from .fitting import FitResultsDialog, plot_fit_results_qt
from .image_tools import ImageToolsDialog, show_image_tools_dialog, compute_averaged_image, compute_difference_image
from .main_menu import MainMenuDialog, show_main_menu

def init_qt_app():
    """Initialize Qt application if not already running."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        app.setStyleSheet(PROXYLFIT_STYLE)
    return app

__all__ = [
    'init_qt_app',
    'PROXYLFIT_STYLE',
    'HeaderWidget',
    'ButtonBar',
    # ... all public exports
]
```

### Step 2: Extract styles.py
- Move `PROXYLFIT_STYLE` constant
- Add color/theme constants for consistency

### Step 3: Extract components.py
- Move `HeaderWidget`
- Move `ButtonBar`
- Move any other reusable widgets

### Step 4: Extract dialog files
- Each dialog class goes to its own file
- Include related helper functions with each dialog
- Maintain backward-compatible imports in `__init__.py`

### Step 5: Update imports throughout codebase
- Update `run_analysis.py`
- Update any other files importing from `ui`

### Step 6: Delete old ui.py
- Ensure all functionality is preserved
- Run tests to verify

## Backward Compatibility

Maintain the same public API:
```python
# These should still work:
from proxyl_analysis.ui import ROISelectorDialog
from proxyl_analysis.ui import select_rectangle_roi_qt
from proxyl_analysis.ui import init_qt_app
```

## Testing Plan

1. Run existing application through all workflows
2. Verify all dialogs open and function correctly
3. Test keyboard shortcuts
4. Test button callbacks
5. Verify styling is consistent

## Acceptance Criteria

- [ ] All dialogs work identically to before
- [ ] No breaking changes to public API
- [ ] Each file < 500 lines
- [ ] All imports work correctly
- [ ] Tests pass (if any exist)

## Notes

- This is a pure refactoring task - no functionality changes
- Can be done incrementally (one dialog at a time)
- Consider doing this after a release to avoid complicating bug fixes
