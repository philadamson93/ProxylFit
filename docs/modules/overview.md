# Module Overview

ProxylFit is organized into modular components, each handling a specific aspect of the MRI analysis pipeline.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_analysis.py                          │
│                    (CLI Entry Point)                            │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    io.py      │     │ registration.py │     │ roi_selection.py│
│  DICOM I/O    │────▶│ Image Alignment │────▶│  ROI Tools      │
└───────────────┘     └─────────────────┘     └─────────────────┘
                                                      │
                              ┌────────────────────────┤
                              ▼                        ▼
                    ┌─────────────────┐     ┌───────────────────┐
                    │    model.py     │     │parameter_mapping.py│
                    │ Kinetic Fitting │     │  Spatial Maps     │
                    └─────────────────┘     └───────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     ui.py       │
                    │  Qt Interface   │
                    └─────────────────┘
```

## Module Summary

| Module | File | Purpose | Key Functions |
|--------|------|---------|---------------|
| **IO** | `io.py` | DICOM loading | `load_dicom_series()` |
| **Registration** | `registration.py` | Image alignment | `register_timeseries()` |
| **ROI Selection** | `roi_selection.py` | Region selection | `select_rectangle_roi()`, `select_manual_contour_roi()` |
| **Kinetic Model** | `model.py` | Curve fitting | `fit_proxyl_kinetics()` |
| **Parameter Mapping** | `parameter_mapping.py` | Spatial maps | `create_parameter_maps()` |
| **UI** | `ui.py` | Qt interface | `ROISelectorDialog`, `ManualContourDialog` |

## Data Flow

### 1. Data Loading (`io.py`)

```
DICOM File → load_dicom_series() → 4D numpy array [x, y, z, t]
```

### 2. Registration (`registration.py`)

```
4D array → register_timeseries() → Aligned 4D array + metrics
```

### 3. ROI Selection (`roi_selection.py` + `ui.py`)

```
Aligned 4D → Interactive UI → Binary mask [x, y]
```

### 4. Time Series Extraction (`roi_selection.py`)

```
4D array + mask → compute_roi_timeseries() → 1D signal [t]
```

### 5. Kinetic Fitting (`model.py`)

```
Time + Signal → fit_proxyl_kinetics() → Parameters (kb, kd, knt, ...)
```

### 6. Parameter Mapping (`parameter_mapping.py`)

```
4D array → create_parameter_maps() → 3D parameter maps
```

## Module Dependencies

```
io.py ─────────────────────────────────────────────┐
                                                   │
registration.py ◀──────────────────────────────────┤
       │                                           │
       └──▶ roi_selection.py ◀────── ui.py ◀───────┤
                   │                    │          │
                   └────▶ model.py ◀────┘          │
                              │                    │
                              └──▶ parameter_mapping.py
```

## File Sizes (Lines of Code)

| Module | Lines | Description |
|--------|-------|-------------|
| `io.py` | ~250 | DICOM handling |
| `registration.py` | ~830 | Registration engine |
| `roi_selection.py` | ~1,250 | ROI selection tools |
| `model.py` | ~820 | Kinetic modeling |
| `parameter_mapping.py` | ~950 | Parameter maps |
| `ui.py` | ~900 | Qt UI components |
| `run_analysis.py` | ~660 | CLI entry point |

**Total**: ~5,660 lines of Python code

## Configuration

Each module can be configured through:

1. **Function parameters** - Direct control at call time
2. **CLI arguments** - Via `run_analysis.py`
3. **Environment** - Qt styling, matplotlib rcParams

## Extending ProxylFit

To add new functionality:

1. **New analysis method**: Add to `model.py` or create new module
2. **New ROI selection mode**: Add to `roi_selection.py` and `ui.py`
3. **New visualization**: Add dialog to `ui.py`
4. **New CLI option**: Update `run_analysis.py`

See individual module documentation for detailed API information.
