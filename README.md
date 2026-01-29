# ProxylFit

Analysis software for time-resolved PROXYL MRI data. Performs image registration, ROI-based kinetic modeling, and parameter mapping.

## Installation

```bash
git clone https://github.com/philadamson93/ProxylFit.git
cd ProxylFit
pip install -r requirements.txt
```

## Running ProxylFit

```bash
python -m proxyl_analysis
```

This opens the main menu where you can load data, draw ROIs, run analyses, and export results.

## Documentation

- **[Tutorial](docs/tutorial.md)** - Step-by-step guide with screenshots
- **[Installation Guide](docs/installation.md)** - Detailed setup and troubleshooting

## Workflow

1. **Load Data** - Select a DICOM folder or resume a previous session
2. **Registration** - Automatic rigid alignment across timepoints
3. **Draw ROI** - Contour, rectangle, or AI-assisted selection
4. **Select Injection Time** - Click on the contrast injection timepoint
5. **Kinetic Fit** - Extended model parameter estimation
6. **Parameter Maps** - Generate spatial maps of kinetic parameters
7. **Export** - DICOM, NPZ, CSV outputs

## Kinetic Model

ProxylFit fits an extended kinetic model to characterize PROXYL dynamics:

```
I(t) = A0 + A1*(1 - exp(-kb*(t-t0)))*exp(-kd*(t-t0)) + A2*(1 - exp(-knt*(t-tmax)))
```

| Parameter | Description |
|-----------|-------------|
| kb | Buildup rate constant |
| kd | Decay rate constant |
| knt | Non-tracer effect rate |

## Output

Results are saved to `output/{dataset_name}/`:
- `registered/dicoms/` - Registered DICOM series
- `kinetic_results.txt` - Fitted parameters
- `parameter_maps/` - Spatial parameter distributions
- `roi_timeseries.csv` - Time series data

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib
- SimpleITK
- PySide6
- pydicom
