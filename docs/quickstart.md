# Quick Start Guide

Get ProxylFit running in 5 minutes.

## 1. Install Dependencies

```bash
cd proxylfit
pip install -r requirements.txt
```

## 2. Run Your First Analysis

```bash
python proxyl_analysis/run_analysis.py --dicom your_data.dcm --z 4
```

## 3. Interactive Workflow

When you run the command, you'll see:

### Step 1: Registration
The software will register all timepoints to a reference frame. This may take a few minutes for the first run. Results are cached for future use.

### Step 2: ROI Selection
A Qt dialog opens showing your MRI slice:
- **Draw a contour** around your region of interest by clicking and dragging
- Press **C** to close the contour
- Click **Accept ROI** when satisfied

### Step 3: Injection Time Selection
Another dialog shows the signal time course:
- **Click** on the time point when contrast was injected
- Click **Set Injection Time** to confirm

### Step 4: Results
The kinetic model is fitted and results are displayed:
- Fitted curve overlaid on data
- Parameter values with uncertainties
- R² and RMSE fit quality metrics

## Output Files

Results are saved to `./output/`:

```
output/
├── kinetic_results.txt      # Parameter summary
├── kinetic_fit.png          # Fit visualization
├── timeseries_data.npz      # Raw data
└── registration_*/          # Cached registration
```

## Common Options

```bash
# Use rectangular ROI instead of contour
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --roi-mode rectangle

# Skip registration (faster, less accurate)
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --skip-registration

# Auto-load previous registration
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --auto-load

# Create parameter maps
python proxyl_analysis/run_analysis.py --dicom data.dcm --z 4 --create-parameter-maps
```

## Next Steps

- Read the [Qt UI Guide](ui.md) for interface details
- Learn about [ROI Selection modes](modules/roi_selection.md)
- Understand the [Kinetic Model](modules/model.md)
