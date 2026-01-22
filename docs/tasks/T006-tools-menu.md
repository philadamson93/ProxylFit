# T006: Main Workflow Menu

**Status**: planned
**Priority**: high
**Created**: 2025-01-15

## Description

Create a main workflow menu that appears after registration completes. This consolidates CLI flags into a discoverable UI and provides a framework for adding future features. The menu is organized into logical sections that accommodate current and planned functionality.

## Rationale

- Current workflow requires knowing many CLI flags
- Users must decide analysis type before running
- Main menu makes all options discoverable
- Provides extensible framework for future features
- Simplifies CLI to minimal required arguments

## Simplified CLI

```bash
# Launch menu directly - load data from UI
python run_analysis.py

# Or specify DICOM on command line - registration runs, then menu appears
python run_analysis.py --dicom data.dcm

# Optional: preload T2 for segmentation
python run_analysis.py --dicom data.dcm --t2 t2_data.dcm

# Batch mode - skip menu, use legacy flags
python run_analysis.py --dicom data.dcm --batch --roi-mode contour --z 4
```

## Default Workflow

```
Option A (no args):  Launch App → Main Menu → Load Experiment → Registration → Ready
Option B (--dicom):  Load T1 DICOM → Auto-Registration → Main Menu → Ready
```

## UI Design

```
┌──────────────────────────────────────────────────────────────────────┐
│  ProxylFit - Analysis Menu                                    [logo] │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Experiment                                                    │  │
│  │                                                                │  │
│  │  [Load New T1 DICOM...]     [Load Previous Session...]        │  │
│  │                                                                │  │
│  │  Current: patient_scan.dcm (128×128×9×26)                     │  │
│  │  Output:  ./output/registration_patient_scan/                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  T1 Data: patient_scan.dcm (128×128×9×26)    Registration: ✓        │
│  T2 Data: Not loaded                          [Load T2 Volume...]    │
│                                                                      │
│  ════════════════════════════════════════════════════════════════    │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ROI Analysis                                                  │  │
│  │                                                                │  │
│  │  Select ROI and fit kinetic model to extract kb, kd, knt      │  │
│  │                                                                │  │
│  │  ROI Source:   ○ T2 (default)   ○ T1                          │  │
│  │  ROI Method:   ○ Rectangle  ○ Manual Contour  ○ Segment       │  │
│  │  Z-slice:      [____4____] / 8                                │  │
│  │                                                                │  │
│  │                                  [Start ROI Analysis]          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Parameter Maps                                                │  │
│  │                                                                │  │
│  │  Generate spatial parameter maps across the image              │  │
│  │                                                                │  │
│  │  ○ Sliding Window     Window: [_15_]×[_15_]×[_3_] voxels      │  │
│  │  ○ Pixel-Level        (slower, full resolution)               │  │
│  │                                                                │  │
│  │  Scope: ○ Whole image  ○ Within ROI (select first)            │  │
│  │                                                                │  │
│  │                                  [Create Parameter Maps]       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Image Tools                                                   │  │
│  │                                                                │  │
│  │  [Averaged Image]         [Difference Image]                  │  │
│  │                                                                │  │
│  │  Select time ranges on signal curve to generate processed     │  │
│  │  images. Requires ROI selection first.                        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Export                                                        │  │
│  │                                                                │  │
│  │  [Registered 4D Data]  [Registration Report]  [Time Series]   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│                                                        [Exit]        │
└──────────────────────────────────────────────────────────────────────┘
```

## Menu Sections

### Section 0: Experiment
Load new data or resume a previous session.

| Button | Description |
|--------|-------------|
| Load New T1 DICOM | Opens file dialog, loads DICOM, runs registration, resets all state |
| Load Previous Session | Opens folder dialog to select a previous output folder with saved registration |

**Load New T1 DICOM**:
- Opens file dialog to select a new T1 DICOM file
- Automatically runs registration (shows quality review window)
- Resets all analysis state (T2, ROI, results)
- Updates output directory to match new DICOM name
- User can work with multiple datasets in one session

**Load Previous Session**:
- Opens folder dialog to select a previous output directory
- Looks for `registered_4d_data.npz` and `registration_metrics.json`
- Loads registered data without re-running registration
- Restores any saved analysis results if available
- Useful for resuming work or reviewing previous analyses

### Header: Data Status
Shows currently loaded data and provides quick access to load additional data.

| Element | Description |
|---------|-------------|
| T1 Data | Shows loaded DICOM path and dimensions |
| Registration | Status indicator (Complete ✓ / Failed ✗) |
| T2 Data | Shows T2 status or "Not loaded" |
| Load T2 Button | Opens file dialog to load T2 volume |

**T2 Loading (T001)**:
- User can load T2 at any time from the menu
- T2 is automatically registered to T1 space
- Once loaded, "ROI Source: T2" option becomes available
- ROI drawn on T2 is in T1 coordinates (after registration)

### Section 1: ROI Analysis
Primary workflow for kinetic model fitting.

| Option | Description | Task |
|--------|-------------|------|
| ROI Source | T2 (default, better tumor contrast) or T1 | T001 |
| ROI Method | Rectangle / Manual Contour / Segment | existing |
| Z-slice | Spinbox to select slice | existing |
| Start | Launches ROI → Injection Time → Fitting workflow | existing |

**Note**: T2 is the default ROI source because it provides better tumor contrast for segmentation. T1 option available if T2 not loaded or user prefers it. ROI coordinates are automatically in T1 space (T2 is registered to T1).

### Section 2: Parameter Maps
Spatial parameter mapping options.

| Option | Description | Task |
|--------|-------------|------|
| Sliding Window | Current implementation with configurable window size | existing |
| Pixel-Level | Full voxel-wise fitting (slower but higher resolution) | T005 |
| Scope | Whole image or within a selected ROI | existing |

### Section 3: Image Tools
Post-processing image generation tools.

| Button | Description | Task |
|--------|-------------|------|
| Averaged Image | Select time range → generate averaged 3D image | T002 |
| Difference Image | Select two ranges → compute difference | T003 |

**Note**: These tools require ROI selection first. The time curve displayed for region selection comes from the ROI signal. If no ROI exists, buttons are disabled with tooltip "Select ROI first".

### Section 4: Export
Data export options.

| Button | Description |
|--------|-------------|
| Registered 4D Data | Save registered_4d to NPZ/NIfTI |
| Registration Report | Save quality metrics as PDF/JSON |
| Time Series | Export ROI time series to CSV |

## Feature Matrix

| Feature | Task | Menu Location | Status |
|---------|------|---------------|--------|
| Load New Experiment | T006 | Experiment: "Load New T1 DICOM" | planned |
| Load Previous Session | T006 | Experiment: "Load Previous Session" | planned |
| T2 Loading | T001 | Header: "Load T2 Volume" | implemented |
| T2-based ROI (default) | T001 | ROI Analysis: "ROI Source" | implemented |
| Averaged Images | T002 | Image Tools | planned |
| Difference Images | T003 | Image Tools | planned |
| Pixel-Level Maps | T005 | Parameter Maps: "Pixel-Level" | planned |
| Sliding Window Maps | - | Parameter Maps: "Sliding Window" | implemented |
| Temporal Smoothing | T004 | — | backlog |

## Implementation Notes

```python
class MainMenuDialog(QDialog):
    """Main workflow menu shown after registration."""

    def __init__(self,
                 registered_4d: np.ndarray,
                 spacing: Tuple[float, float, float],
                 time_array: np.ndarray,
                 dicom_path: str,
                 output_dir: str = './output',
                 registered_t2: Optional[np.ndarray] = None,
                 parent=None):
        super().__init__(parent)
        self.registered_4d = registered_4d
        self.spacing = spacing
        self.time_array = time_array
        self.dicom_path = dicom_path
        self.output_dir = output_dir
        self.registered_t2 = registered_t2  # None until loaded

        # State
        self.roi_mask = None
        self.roi_signal = None

        self._setup_ui()
        self._update_t2_status()

    def _setup_ui(self):
        """Build the menu UI."""
        layout = QVBoxLayout(self)

        # Experiment section (load new / load previous)
        self._create_experiment_section(layout)

        # Header with data status
        self._create_header(layout)

        # ROI Analysis section
        self._create_roi_section(layout)

        # Parameter Maps section
        self._create_param_maps_section(layout)

        # Image Tools section
        self._create_image_tools_section(layout)

        # Export section
        self._create_export_section(layout)

        # Exit button
        self._create_footer(layout)

    def _load_new_experiment(self):
        """Load a new T1 DICOM and run registration."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load T1 DICOM", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if file_path:
            # Reset all state
            self._reset_state()

            # Load and register
            image_4d, spacing = load_dicom_series(file_path)
            self.dicom_path = file_path
            self.spacing = spacing

            # Update output directory
            dicom_name = Path(file_path).stem
            self.output_dir = f"./output/registration_{dicom_name}"

            # Run registration (shows quality window)
            self.registered_4d, reg_metrics = register_timeseries(
                image_4d, spacing,
                output_dir=self.output_dir,
                show_quality_window=True,
                dicom_path=file_path
            )

            # Update UI
            self._update_data_status()

    def _load_previous_session(self):
        """Load a previous session from saved registration data."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Previous Session Folder", "./output"
        )
        if folder_path:
            try:
                # Reset state
                self._reset_state()

                # Load saved registration
                self.registered_4d, self.spacing, reg_metrics = load_registration_data(folder_path)
                self.output_dir = folder_path

                # Try to determine original DICOM path from metadata
                metrics_file = Path(folder_path) / "registration_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metadata = json.load(f).get('metadata', {})
                        self.dicom_path = metadata.get('dicom_path', 'Unknown')

                # Update UI
                self._update_data_status()

            except FileNotFoundError as e:
                QMessageBox.warning(self, "Load Failed", f"Could not load session: {e}")

    def _reset_state(self):
        """Reset all analysis state for a new experiment."""
        self.registered_t2 = None
        self.roi_mask = None
        self.roi_signal = None
        self._update_t2_status()

    def _load_t2_volume(self):
        """Open file dialog and load T2 volume."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load T2 DICOM", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if file_path:
            t2_volume, t2_spacing = load_t2_volume(file_path)
            t1_reference = self.registered_4d[:, :, :, 0]
            self.registered_t2, reg_info = register_t2_to_t1(
                t2_volume, t2_spacing,
                t1_reference, self.spacing
            )
            self._update_t2_status()

    def _update_t2_status(self):
        """Update T2 status display and enable/disable T2 options."""
        if self.registered_t2 is not None:
            self.t2_status_label.setText("T2 Data: Loaded ✓")
            self.t2_radio.setEnabled(True)
        else:
            self.t2_status_label.setText("T2 Data: Not loaded")
            self.t2_radio.setEnabled(False)
            self.t1_radio.setChecked(True)

    def _start_roi_analysis(self):
        """Launch ROI analysis workflow."""
        # Determine source image
        use_t2 = self.t2_radio.isChecked() and self.registered_t2 is not None
        source_image = self.registered_t2 if use_t2 else self.registered_4d[:, :, :, 0]

        # Get settings
        roi_mode = self._get_selected_roi_mode()
        z_slice = self.z_spinbox.value()
        apply_smoothing = self.smoothing_checkbox.isChecked()
        smooth_n = self.smoothing_spinbox.value() if apply_smoothing else None

        # Launch workflow...
        pass

    def _create_averaged_image(self):
        """Launch averaged image tool (T002)."""
        if self.roi_signal is None:
            # Prompt: select ROI first or use whole-image mean
            pass
        # Launch AveragedImageDialog
        pass

    def _create_difference_image(self):
        """Launch difference image tool (T003)."""
        if self.roi_signal is None:
            # Prompt: select ROI first or use whole-image mean
            pass
        # Launch DifferenceImageDialog
        pass

    def _create_parameter_maps(self):
        """Launch parameter mapping."""
        pixel_level = self.pixel_level_radio.isChecked()
        within_roi = self.within_roi_radio.isChecked()

        if pixel_level:
            # Launch pixel-level fitting (T005)
            pass
        else:
            # Launch sliding window (existing)
            window_size = (
                self.window_x_spin.value(),
                self.window_y_spin.value(),
                self.window_z_spin.value()
            )
            pass

    def _apply_temporal_smoothing(self):
        """Apply running average to 4D data (T004)."""
        n_points = self.smoothing_spinbox.value()
        # Apply and optionally save
        pass
```

## Extensibility

The menu is designed to accommodate future features:

1. **New Image Tools**: Add buttons to Image Tools section
2. **New Analysis Modes**: Add new sections or options to existing sections
3. **Preprocessing Options**: Add checkboxes/spinners to relevant sections
4. **New Export Formats**: Add buttons to Export section

Each section uses a `QGroupBox` that can be expanded with new controls without restructuring the entire dialog.

## Acceptance Criteria

- [ ] MainMenuDialog with all sections
- [ ] **Experiment section**: Load New T1 DICOM button works
- [ ] **Experiment section**: Load Previous Session button works
- [ ] **Experiment section**: Loading new data resets all state (T2, ROI, results)
- [ ] Header shows data status with T2 load button
- [ ] ROI Analysis section with source (T2 default)/method/z-slice options
- [ ] Parameter Maps section with sliding window vs pixel-level toggle
- [ ] Image Tools section with averaged/difference buttons (disabled until ROI selected)
- [ ] Export section with data export options
- [ ] T2 loading triggers registration and enables T2 options
- [ ] Menu appears after registration by default
- [ ] `--batch` flag to skip menu for scripted workflows
- [ ] Consistent styling with ProxylFit theme
- [ ] Documentation updated

## Dependencies

- **T001**: T2-T1 Registration (implemented - integrate into menu)
- **T002**: Averaged Image dialog
- **T003**: Difference Image dialog
- **T005**: Pixel-level parameter mapping

## Implementation Order

1. **T006 Phase 1**: Basic menu structure with existing features (ROI analysis, sliding window maps, export)
2. **T002**: Averaged Image dialog (establishes time-curve region selection UI)
3. **T003**: Difference Image dialog (extends T002 with two-region mode)
4. **T005**: Pixel-level maps option in Parameter Maps section
5. **T006 Phase 2**: Polish, add other export formats, refinements
