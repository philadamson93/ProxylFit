![ProxylFit](proxylfit.png)

# ProxylFit: PROXYL MRI Analysis

ProxylFit is a comprehensive Python toolkit for analyzing time-resolved MRI data after PROXYL contrast injection. The software provides robust image registration, interactive ROI selection, and sophisticated kinetic modeling capabilities.

## What's New in v1.1.0

- **Modern Qt-based UI** - PySide6 interface with proper layout management
- **Responsive dialogs** - Windows resize properly, no overlapping elements
- **Navigation toolbar** - Pan, zoom, and save for all plots
- **Improved styling** - Professional appearance with consistent theming

## Overview

ProxylFit implements an **extended kinetic model** to analyze Proxyl contrast dynamics in MRI data:

```
I(t) = A0 + A1*(1 - exp(-kb*(t - t0))) * exp(-kd*(t - t0)) + A2*(1 - exp(-knt*(t - tmax)))
```

### Key Features

- **Image Registration**: Rigid registration of 4D time series data with quality metrics
- **Interactive ROI Selection**: Multiple modes including rectangular, manual contour drawing, and SegmentAnything-based segmentation
- **Extended Kinetic Modeling**: Fits tracer dynamics with non-tracer effects
- **Parameter Mapping**: Spatial parameter maps using sliding window analysis
- **Modern Qt Interface**: Responsive UI with proper layout management
- **70-Second Time Intervals**: Optimized for typical Proxyl acquisition protocols

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic analysis with manual contour ROI selection
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4

# Analysis with rectangular ROI
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --roi-mode rectangle

# Create parameter maps with custom window size
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --create-parameter-maps --window-size 7
```

## Extended Kinetic Model

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| **A0** | Baseline signal | Intensity |
| **A1** | Tracer signal amplitude | Intensity |
| **A2** | Non-tracer effect amplitude | Intensity |
| **kb** | Buildup rate constant | 1/min |
| **kd** | Decay rate constant | 1/min |
| **knt** | Non-tracer effect rate constant | 1/min |
| **t0** | Tracer injection time | minutes |
| **tmax** | Non-tracer effect onset time | minutes |

### Model Components

1. **Baseline**: `A0` - constant background signal
2. **Tracer Term**: `A1*(1 - exp(-kb*(t - t0))) * exp(-kd*(t - t0))` - contrast buildup and decay
3. **Non-tracer Term**: `A2*(1 - exp(-knt*(t - tmax)))` - physiological effects

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- [**Installation Guide**](docs/installation.md) - Setup and dependencies
- [**Qt UI Guide**](docs/ui.md) - Modern user interface
- [**Module Overview**](docs/modules/overview.md) - Architecture and components

### Module Documentation

| Module | Description | Documentation |
|--------|-------------|---------------|
| IO | DICOM loading | [docs/modules/io.md](docs/modules/io.md) |
| Registration | Image alignment | [docs/modules/registration.md](docs/modules/registration.md) |
| ROI Selection | Region selection | [docs/modules/roi_selection.md](docs/modules/roi_selection.md) |
| Kinetic Model | Curve fitting | [docs/modules/model.md](docs/modules/model.md) |
| Parameter Mapping | Spatial maps | [docs/modules/parameter_mapping.md](docs/modules/parameter_mapping.md) |
| UI | Qt interface | [docs/ui.md](docs/ui.md) |

## Installation

### Requirements

- Python 3.7+ (3.9+ recommended)
- PySide6 for Qt-based UI
- numpy, scipy, matplotlib, SimpleITK

### Install via pip

```bash
git clone https://github.com/philadamson93/ProxylFit.git
cd ProxylFit
pip install -r requirements.txt
```

### Install via conda

```bash
git clone https://github.com/philadamson93/ProxylFit.git
cd ProxylFit
conda env create -f environment.yml
conda activate proxyl
```

### Optional Dependencies

```bash
# For SegmentAnything ROI selection
pip install segment-anything opencv-python
```

## Workflow

1. **Load DICOM Data** - Import 4D time series PROXYL MRI data
2. **Registration** - Rigid alignment across timepoints with quality visualization
3. **ROI Selection** - Interactive selection using Qt-based dialogs
4. **Injection Time Selection** - Define PROXYL contrast injection timepoint
5. **Kinetic Fitting** - Extended model parameter estimation
6. **Parameter Mapping** (optional) - Spatial parameter distribution analysis
7. **Results Export** - Comprehensive results and visualization

## User Interface

ProxylFit v1.1.0 features a modern Qt-based interface:

### ROI Selection Dialog
- Click and drag to select region
- Real-time statistics panel
- Navigation toolbar for pan/zoom

### Contour Drawing Dialog
- Free-form contour drawing
- Z-slice navigation slider
- Keyboard shortcuts (C=close, R=reset)

### Injection Time Selector
- Click to select time point
- CSV export functionality
- Signal statistics display

### Fit Results Viewer
- Data and fitted curve plot
- Residuals subplot
- Parameter table with uncertainties

## Output Files

### Per-Analysis Results
- `kinetic_results.txt` - Complete parameter summary
- `timeseries_data.npz` - Raw and fitted data
- `kinetic_fit.png` - Visualization plots
- `timecourse_data.csv` - Exportable timecourse data

### Registration Data
- `registered_4d_data.npz` - Registered 4D dataset
- `registration_metrics.json` - Quality metrics and metadata

### Parameter Maps
- `parameter_maps_*.npz` - Spatial parameter distributions
- Visualization plots for each parameter

## Example Complete Workflow

```bash
python proxyl_analysis/run_analysis.py \
    --dicom tumor_data.dcm \
    --z 5 \
    --roi-mode contour \
    --create-parameter-maps \
    --window-size 7 \
    --output-dir ./results
```

This produces:
- ROI-specific kinetic parameters from manual contour selection
- Spatial parameter maps across the entire image
- Comprehensive quality metrics and visualizations

## Advanced Usage

### Auto-Loading Previous Registration
```bash
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --auto-load
```

### Batch Processing (No GUI)
```bash
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --no-plot --auto-load
```

### Custom Window Sizes for Parameter Mapping
```bash
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 \
    --create-parameter-maps \
    --window-size-x 15 --window-size-y 15 --window-size-z 3
```

## API Usage

```python
from proxyl_analysis import (
    load_dicom_series,
    register_timeseries,
    fit_proxyl_kinetics
)
from proxyl_analysis.ui import (
    select_rectangle_roi_qt,
    select_manual_contour_roi_qt,
    select_injection_time_qt,
    plot_fit_results_qt
)
from proxyl_analysis.roi_selection import compute_roi_timeseries

# Load data
image_4d, spacing = load_dicom_series("data.dcm")

# Register
registered_4d, metrics = register_timeseries(image_4d, spacing)

# Select ROI (Qt UI)
roi_mask = select_manual_contour_roi_qt(registered_4d, z_index=4)

# Extract time series
signal = compute_roi_timeseries(registered_4d, roi_mask)

# Select injection time (Qt UI)
time = np.arange(len(signal)) * (70/60)  # 70s intervals in minutes
injection_idx = select_injection_time_qt(time, signal)

# Fit model
kb, kd, knt, fitted, results = fit_proxyl_kinetics(
    time[injection_idx:], signal[injection_idx:]
)

# Visualize (Qt UI)
plot_fit_results_qt(time[injection_idx:], signal[injection_idx:], fitted, results)
```

## Performance Notes

- **Extended Model**: Requires ≥8 timepoints (vs 5 for basic model)
- **Parameter Mapping**: Computationally intensive; use appropriate window sizes
- **Registration Caching**: Automatic caching reduces repeat processing time
- **Memory Usage**: Scales with 4D dataset size and window dimensions

## Troubleshooting

See the [documentation](docs/) for detailed troubleshooting guides:
- [Installation issues](docs/installation.md#troubleshooting)
- [UI problems](docs/ui.md#troubleshooting)
- [ROI selection](docs/modules/roi_selection.md#troubleshooting)
- [Fitting convergence](docs/modules/model.md#troubleshooting)

## Citation

When using ProxylFit, please cite the relevant publications and acknowledge the extended kinetic modeling framework.

---

**ProxylFit Version**: 1.1.0
**Last Updated**: January 2025
**Compatible**: Python 3.7+
