# Qt User Interface Guide

ProxylFit v1.1.0 introduces a modern Qt-based user interface built with PySide6. This replaces the previous matplotlib widgets approach, providing proper layout management and responsive design.

## Overview

The Qt UI provides:
- **Automatic layout management** - No overlapping elements
- **Resizable windows** - UI adapts to any window size
- **Navigation toolbar** - Pan, zoom, and save for all plots
- **Consistent styling** - Professional appearance across all dialogs
- **Status bar feedback** - Real-time status updates

## Dialog Components

### ROI Selector Dialog

**Purpose**: Interactive rectangle ROI selection on MRI slices.

**Features**:
- Click and drag to select rectangular region
- Interactive resize handles on corners/edges
- Real-time statistics panel (mean, std, min, max)
- Navigation toolbar for pan/zoom

**Usage**:
```python
from proxyl_analysis.ui import ROISelectorDialog, init_qt_app

app = init_qt_app()
dialog = ROISelectorDialog(image_slice, title="Select Tumor ROI")

if dialog.exec() == QDialog.Accepted:
    mask = dialog.get_mask()
    stats = dialog.get_stats()
```

**Keyboard Shortcuts**:
| Key | Action |
|-----|--------|
| Escape | Cancel and close |
| Enter | Accept ROI |

---

### Manual Contour Dialog

**Purpose**: Free-form contour drawing for irregular ROI shapes.

**Features**:
- Click and drag to draw contour
- Z-slice navigation with slider
- Close contour to create mask
- Reset functionality

**Usage**:
```python
from proxyl_analysis.ui import ManualContourDialog, init_qt_app

app = init_qt_app()
dialog = ManualContourDialog(image_4d, z_index=4)

if dialog.exec() == QDialog.Accepted:
    mask = dialog.get_mask()
    final_z = dialog.get_z_index()
```

**Keyboard Shortcuts**:
| Key | Action |
|-----|--------|
| C | Close contour |
| R | Reset drawing |
| Up Arrow | Next Z-slice |
| Down Arrow | Previous Z-slice |
| Escape | Cancel |

---

### Injection Time Selector Dialog

**Purpose**: Select the contrast injection time point from signal data.

**Features**:
- Click to select time point
- Visual marker showing selection
- CSV export functionality
- Signal statistics display

**Usage**:
```python
from proxyl_analysis.ui import InjectionTimeSelectorDialog, init_qt_app

app = init_qt_app()
dialog = InjectionTimeSelectorDialog(time, signal, time_units='minutes')

if dialog.exec() == QDialog.Accepted:
    injection_idx = dialog.get_injection_index()
```

**Buttons**:
- **Export CSV**: Save timecourse data to CSV file
- **Set Injection Time**: Accept selection and close

---

### Fit Results Dialog

**Purpose**: Display kinetic model fitting results with visualization.

**Features**:
- Main plot showing data and fitted curve
- Residuals subplot
- Parameter table with uncertainties
- Fit quality metrics (R², RMSE)
- Save plot functionality

**Usage**:
```python
from proxyl_analysis.ui import FitResultsDialog, init_qt_app

app = init_qt_app()
dialog = FitResultsDialog(time, signal, fitted_signal, fit_results)
dialog.exec()
```

---

## Drop-in Replacement Functions

These functions provide the same interface as the original matplotlib-based functions:

### select_rectangle_roi_qt()

```python
from proxyl_analysis.ui import select_rectangle_roi_qt

# Same interface as select_rectangle_roi()
mask = select_rectangle_roi_qt(image_4d, z_index=4)
```

### select_manual_contour_roi_qt()

```python
from proxyl_analysis.ui import select_manual_contour_roi_qt

# Same interface as select_manual_contour_roi()
mask = select_manual_contour_roi_qt(image_4d, z_index=4)
```

### select_injection_time_qt()

```python
from proxyl_analysis.ui import select_injection_time_qt

# Same interface as select_injection_time()
injection_idx = select_injection_time_qt(time, signal, 'minutes', './output')
```

### plot_fit_results_qt()

```python
from proxyl_analysis.ui import plot_fit_results_qt

# Same interface as plot_fit_results()
plot_fit_results_qt(time, signal, fitted_signal, fit_results, 'output.png')
```

---

## Styling

The UI uses a consistent style sheet defined in `ui.py`. Key style elements:

### Button Types

```python
# Standard button
button_bar.add_button("name", "Label", callback, "default")

# Accept button (green)
button_bar.add_button("accept", "Accept", callback, "accept")

# Cancel button (red)
button_bar.add_button("cancel", "Cancel", callback, "cancel")

# Export button (blue)
button_bar.add_button("export", "Export", callback, "export")
```

### Custom Styling

To customize the appearance, modify `PROXYLFIT_STYLE` in `ui.py`:

```python
PROXYLFIT_STYLE = """
QPushButton {
    background-color: #e0e0e0;
    border: 1px solid #b0b0b0;
    border-radius: 4px;
    padding: 8px 16px;
}
/* Add your customizations here */
"""
```

---

## Layout Architecture

Each dialog follows a consistent structure:

```
┌─────────────────────────────────────────┐
│  Header (Title + Logo)                  │
├─────────────────────────────────────────┤
│  Instructions (green box)               │
├─────────────────────────────────────────┤
│                           │             │
│  Matplotlib Canvas        │  Info Panel │
│  + Navigation Toolbar     │             │
│                           │             │
├─────────────────────────────────────────┤
│  Button Bar (Cancel ... Accept)         │
├─────────────────────────────────────────┤
│  Status Bar                             │
└─────────────────────────────────────────┘
```

This layout uses Qt's layout managers:
- `QVBoxLayout` for vertical stacking
- `QHBoxLayout` for horizontal arrangement
- `addStretch()` for flexible spacing

---

## Migrating from Matplotlib Widgets

If you have code using the old matplotlib-based UI:

**Before (v1.0.0)**:
```python
from proxyl_analysis.roi_selection import select_rectangle_roi
mask = select_rectangle_roi(image_4d, z_index)
```

**After (v1.1.0)**:
```python
from proxyl_analysis.ui import select_rectangle_roi_qt
mask = select_rectangle_roi_qt(image_4d, z_index)
```

The function signatures are identical, so migration is straightforward.

---

## Troubleshooting

### Dialog doesn't appear

Ensure Qt app is initialized:
```python
from proxyl_analysis.ui import init_qt_app
app = init_qt_app()  # Must be called before creating dialogs
```

### Window too small

Dialogs have minimum sizes set. You can resize by dragging window edges.

### Plot not updating

Call `canvas.draw()` after modifying the matplotlib figure:
```python
self.ax.plot(x, y)
self.canvas.draw()
```

### Style not applying

Ensure the application style is set:
```python
app = init_qt_app()  # This applies PROXYLFIT_STYLE automatically
```
