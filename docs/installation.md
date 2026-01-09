# Installation Guide

## Requirements

- **Python**: 3.7 or higher (3.9+ recommended)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: 8GB+ RAM recommended for large datasets

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/proxylfit.git
cd proxylfit

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Conda Environment

```bash
# Clone the repository
git clone https://github.com/your-repo/proxylfit.git
cd proxylfit

# Create and activate conda environment
conda env create -f environment.yml
conda activate proxyl
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Numerical computing |
| scipy | >=1.7.0 | Scientific computing, curve fitting |
| matplotlib | >=3.3.0 | Plotting and visualization |
| SimpleITK | >=2.1.0 | Medical image processing |
| PySide6 | >=6.5.0 | Qt-based user interface |

### Optional Dependencies

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| segment-anything | AI-based ROI segmentation | `pip install segment-anything` |
| opencv-python | Computer vision for segmentation | `pip install opencv-python` |
| pydicom | Enhanced DICOM support | `pip install pydicom` |

## Installing Optional Features

### SegmentAnything ROI Selection

For AI-powered ROI segmentation:

```bash
pip install segment-anything opencv-python
```

You'll also need to download a SAM model checkpoint:
- [ViT-H (largest, most accurate)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [ViT-L (medium)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [ViT-B (smallest, fastest)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Verifying Installation

Run the following to verify your installation:

```bash
python -c "from proxyl_analysis import load_dicom_series, fit_proxyl_kinetics; print('Core modules OK')"
python -c "from proxyl_analysis.ui import init_qt_app; print('Qt UI OK')"
```

## Troubleshooting

### PySide6 Installation Issues

**macOS**: If you encounter issues with PySide6:
```bash
pip install --upgrade pip
pip install PySide6 --force-reinstall
```

**Linux**: You may need additional system packages:
```bash
# Ubuntu/Debian
sudo apt-get install libxcb-xinerama0 libxkbcommon-x11-0

# Fedora
sudo dnf install xcb-util-wm xcb-util-image xcb-util-keysyms
```

### SimpleITK Installation Issues

If SimpleITK fails to install:
```bash
pip install --upgrade pip setuptools wheel
pip install SimpleITK
```

### Display Issues (Headless Servers)

For running on headless servers, you may need a virtual display:
```bash
# Install Xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run python run_analysis.py --dicom data.dcm --z 4
```

Or use batch mode without GUI:
```bash
python run_analysis.py --dicom data.dcm --z 4 --no-plot
```

## Updating ProxylFit

```bash
cd proxylfit
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstalling

```bash
# If using pip
pip uninstall proxylfit

# If using conda
conda deactivate
conda env remove -n proxyl
```
