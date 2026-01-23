# T015: Application Distribution for Non-Technical Users

**Status**: planning
**Priority**: high
**Created**: 2025-01-22

## Overview

Package ProxylFit for distribution to collaborators who are not Python/CLI savvy but are experienced MR scientists familiar with tools like ImageJ. The solution must support iterative development with easy updates as we incorporate feedback.

## User Profile

- **Technical level**: Not Python/CLI savvy
- **Domain expertise**: MR scientist, familiar with ImageJ, DICOM viewers
- **Platform**: Likely macOS (based on current development), possibly Windows
- **Workflow**: Needs to process DICOM data, draw ROIs, view results
- **Update frequency**: Will receive updates as we iterate on feedback

## Requirements

1. **Easy installation**: Minimal steps, no command line required
2. **Easy updates**: Simple way to get new versions
3. **Cross-platform**: At minimum macOS, ideally Windows too
4. **Self-contained**: No need to install Python, dependencies, etc.
5. **Familiar UX**: Similar to desktop apps they already use
6. **Debug-friendly**: Easy to get error logs when things go wrong

---

## Option 1: Standalone macOS App (PyInstaller)

Create a native `.app` bundle that can be double-clicked to run.

### Implementation

```bash
# Install PyInstaller
pip install pyinstaller

# Create spec file for ProxylFit
pyinstaller --name "ProxylFit" \
    --windowed \
    --icon assets/icon.icns \
    --add-data "proxyl_analysis:proxyl_analysis" \
    --hidden-import PySide6 \
    --hidden-import pydicom \
    --hidden-import nibabel \
    proxyl_analysis/run_analysis.py
```

### Pros
- Double-click to launch - very familiar UX
- No installation required (drag to Applications)
- Self-contained, includes Python runtime
- Native macOS experience

### Cons
- Large file size (200-500 MB) due to bundled Python + Qt
- Code signing/notarization required for macOS Gatekeeper
- Separate builds needed for macOS Intel vs Apple Silicon
- Updates require re-downloading entire app
- Can be tricky to debug (hidden Python errors)

### Update Mechanism
- Manual: Share new `.app` via Box/Dropbox
- Could add in-app update checker that downloads new version

### Effort: Medium
- 2-3 days for initial setup and testing
- Need Apple Developer account for signing ($99/year)

---

## Option 2: Docker Container with Web UI

Package as Docker container with a web-based interface (Streamlit or Gradio).

### Implementation

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY proxyl_analysis/ ./proxyl_analysis/
COPY streamlit_app.py .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### New File: `streamlit_app.py`
```python
import streamlit as st
from proxyl_analysis.io import load_dicom_series
# ... wrap existing functionality in Streamlit UI
```

### Pros
- Consistent environment across all platforms
- Web UI is very accessible (works in any browser)
- Easy updates: just `docker pull` new image
- Good for debugging (can attach to container)
- Could eventually host on server for shared access

### Cons
- Requires Docker Desktop installation (can be confusing)
- Docker Desktop is large (~2GB) and resource-intensive
- Requires rewriting UI from Qt to web framework
- File access requires volume mounting (confusing for users)
- Significant development effort to recreate current Qt UI

### Update Mechanism
```bash
docker pull username/proxylfit:latest
```

### Effort: High
- 1-2 weeks to rewrite UI in Streamlit/Gradio
- Need to maintain parallel UIs or deprecate Qt version

---

## Option 3: Conda Environment + Launcher Script

Provide a conda environment file and simple launcher.

### Implementation

**`environment.yml`:**
```yaml
name: proxylfit
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pyside6
  - numpy
  - scipy
  - matplotlib
  - pydicom
  - nibabel
  - pip
  - pip:
    - segment-anything  # optional
```

**`ProxylFit.command`** (macOS double-clickable):
```bash
#!/bin/bash
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate proxylfit
python -m proxyl_analysis.run_analysis
```

### Pros
- Uses existing Qt UI (no rewrite)
- Conda is popular in scientific community
- Easy updates: `git pull` or replace folder
- Full debugging capability

### Cons
- Requires Anaconda/Miniforge installation (~500MB)
- Conda can be slow and confusing
- Environment setup can fail in various ways
- Not truly "double-click and run"

### Update Mechanism
- Git pull (if comfortable with git)
- Or replace folder contents

### Effort: Low
- 1 day to create and test

---

## Option 4: Briefcase/Toga Native App (BeeWare)

Use BeeWare toolchain to create truly native apps.

### Implementation

```bash
pip install briefcase
briefcase new  # Creates project structure
briefcase build
briefcase package
```

### Pros
- Creates proper native apps for each platform
- Smaller than PyInstaller in some cases
- Active development, good Python packaging

### Cons
- Requires rewriting UI using Toga (not Qt)
- Toga is less mature than Qt
- Still learning curve for packaging

### Effort: High
- 2+ weeks to rewrite UI in Toga

---

## Option 5: PyApp / Shiv Single-File Executable

Create a single executable file using PyApp or Shiv.

### Implementation (PyApp)
```bash
pip install pyapp
pyapp build --python 3.11 --app proxyl_analysis
```

### Pros
- Single file distribution
- Simpler than PyInstaller
- Auto-extracts and runs

### Cons
- Less mature than PyInstaller
- Still large file size
- May have compatibility issues

### Effort: Medium

---

## Option 6: Hybrid - Qt App + Simple Installer

Keep current Qt UI but create a proper installer.

### macOS: Create DMG with drag-to-install
```bash
# Use create-dmg tool
create-dmg \
  --volname "ProxylFit" \
  --window-size 600 400 \
  --app-drop-link 400 200 \
  "ProxylFit.dmg" \
  "dist/ProxylFit.app"
```

### Windows: Create MSI/EXE installer
Use NSIS or WiX to create proper Windows installer.

### Pros
- Professional installation experience
- Can include shortcuts, uninstaller
- Familiar to users

### Cons
- More complex build process
- Need to maintain installers for each platform

### Effort: Medium-High

---

## Option 7: napari Plugin

If colleague uses napari for other image analysis, could integrate as plugin.

### Pros
- Leverages existing tool they may know
- napari has good DICOM support
- Plugin ecosystem for distribution

### Cons
- Requires napari installation
- Different UI paradigm than current app
- May not fit workflow

### Effort: High (complete rewrite)

---

## Recommendation

### For Immediate Use: **Option 1 (PyInstaller macOS App)**

**Rationale:**
- Fastest path to a usable distribution
- Keeps current Qt UI (no rewrite)
- Familiar UX for ImageJ users (double-click app)
- Can iterate on feedback without changing distribution method

**Implementation Steps:**
1. Add app icon and metadata
2. Create PyInstaller spec file
3. Build and test on clean macOS system
4. Sign with Apple Developer certificate (optional but recommended)
5. Create simple "How to Use" PDF
6. Share via Box/Dropbox

**Update Workflow:**
1. Make code changes based on feedback
2. Rebuild app
3. Share new version via Box
4. User replaces old app with new one

### For Future: **Consider Option 2 (Docker + Web UI)**

If the tool becomes useful to multiple collaborators or needs to run on a shared server, investing in a web-based UI would be valuable. This could happen after the initial feedback cycle stabilizes the core functionality.

---

## Implementation Plan for Option 1

### Phase 1: Prepare for Packaging (Day 1)

1. **Create app icon**
   - Design simple icon (or use placeholder)
   - Generate `.icns` file for macOS

2. **Add entry point script**
   ```python
   # proxyl_analysis/__main__.py
   from .run_analysis import main
   if __name__ == "__main__":
       main()
   ```

3. **Create PyInstaller spec file**
   - Handle all hidden imports
   - Bundle data files (logo, etc.)

### Phase 2: Build and Test (Day 2)

1. **Build on macOS**
   ```bash
   pyinstaller ProxylFit.spec
   ```

2. **Test on clean system**
   - Test on Mac without Python installed
   - Verify all features work
   - Check file dialogs, DICOM loading, etc.

3. **Fix any missing imports/data**

### Phase 3: Polish and Distribute (Day 3)

1. **Create DMG installer** (optional but nice)

2. **Write quick start guide**
   - Installation steps
   - Basic workflow
   - How to report issues

3. **Share with colleague**

---

## Quick Start Guide (Draft)

### Installing ProxylFit

1. Download `ProxylFit.app` from the shared folder
2. Drag `ProxylFit.app` to your Applications folder
3. Double-click to run

**First Launch (macOS):**
If you see "ProxylFit cannot be opened because it is from an unidentified developer":
1. Right-click (or Control-click) on ProxylFit.app
2. Select "Open" from the menu
3. Click "Open" in the dialog

### Basic Workflow

1. **Launch ProxylFit** - double-click the app
2. **Load DICOM** - Click "Load New T1 DICOM" and select your file
3. **Wait for Registration** - Progress bar shows status
4. **Draw ROI** - Select contour mode, click on slice to draw
5. **Set Injection Time** - Click on the curve where injection occurred
6. **Run Analysis** - Click "Run Kinetic Fit" to see results

### Getting Help

If something doesn't work:
1. Take a screenshot of any error message
2. Note what you were doing when it happened
3. Send to [developer email]

---

## Files to Create

```
ProxylFit/
├── packaging/
│   ├── ProxylFit.spec          # PyInstaller configuration
│   ├── icon.icns               # macOS app icon
│   ├── icon.ico                # Windows app icon
│   ├── Info.plist              # macOS app metadata
│   └── build.sh                # Build script
├── docs/
│   └── quick-start-guide.md    # User documentation
└── proxyl_analysis/
    └── __main__.py             # Entry point for packaging
```

---

## Alternative: Even Simpler First Step

If signing/packaging is too much overhead for initial testing, we could:

1. **Share the entire project folder** via Box
2. **Include a simple launcher script** that:
   - Checks if Python is installed
   - Creates venv if needed
   - Installs dependencies
   - Launches the app

This is less polished but gets the tool in their hands faster for feedback.

```bash
#!/bin/bash
# launch_proxylfit.command

cd "$(dirname "$0")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install from python.org"
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "First run - setting up environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Launch
python -m proxyl_analysis.run_analysis
```

This approach:
- Works immediately
- Easy to update (replace folder)
- Full debugging capability
- But requires Python to be installed

---

## Decision: Folder + Launcher Script (venv-based, NO conda)

**Chosen approach**: Folder + launcher script using Python's built-in `venv`

**Why not conda?** Colleague has had trouble with conda before - avoid that complexity.

**What user needs to do:**
1. Install Python from python.org (one-time, simple installer)
2. Download ProxylFit folder from Box
3. Double-click `Launch_ProxylFit.command`

**What the launcher script does:**
1. Checks if Python is installed
2. Creates `venv/` folder on first run
3. Installs all dependencies from `requirements.txt`
4. Launches the Qt GUI

**Future**: Once features stabilize based on feedback, can upgrade to PyInstaller standalone app.

## Implementation Checklist

When ready to implement:

- [ ] Create `Launch_ProxylFit.command` (macOS)
- [ ] Create `Launch_ProxylFit.bat` (Windows, if needed)
- [ ] Create `INSTALL.txt` with simple 2-step instructions
- [ ] Ensure `requirements.txt` is complete and pinned
- [ ] Test on clean macOS system without Python dev tools
- [ ] Create `__main__.py` entry point if not exists
- [ ] Add `.gitignore` entry for `venv/` folder
