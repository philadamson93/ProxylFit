# T011: Self-Contained Dataset Directory Structure

## Status: Planned

## Overview

Reorganize output directory structure so each dataset (DICOM source) has all its results in a single self-contained directory. Currently results are scattered across multiple sibling directories.

## Current Structure (Problematic)

```
output/
├── registration_example_dicom/
│   ├── registered_4d_data.npz
│   ├── registration_metrics.json
│   ├── kinetic_results.txt
│   ├── kinetic_fit.png
│   └── timeseries_data.npz
├── parameter_maps_example_dicom/           # Separate directory!
│   ├── parameter_maps.npz
│   └── *.nii.gz
├── enhanced_parameter_maps_example_dicom/  # Another separate directory!
│   └── ...
├── averaged_image_frames0-5.npz            # In root! Not linked to dataset
└── difference_image_A0-5_B10-15.npz        # In root! Not linked to dataset
```

**Problems:**
- Related files scattered across 3+ directories
- Image tools outputs not associated with any dataset
- No way to know what analysis has been done
- Hard to share/archive a complete dataset
- No versioning for re-runs

## Proposed Structure

```
output/
└── {dataset_name}/                          # e.g., "example_dicom" or custom name
    ├── dataset_manifest.json                # NEW: Tracks all analysis performed
    ├── source_info.json                     # NEW: Source DICOM path, load time, etc.
    │
    ├── registered/                          # Registration outputs
    │   ├── dicoms/                          # NEW: DICOM series (T010)
    │   │   ├── registered_t000.dcm
    │   │   ├── registered_t001.dcm
    │   │   └── series_info.json
    │   ├── registration_metrics.json
    │   └── registered_4d_data.npz           # LEGACY: Keep for backward compat
    │
    ├── roi_analysis/                        # ROI workflow outputs
    │   ├── roi_mask.npz                     # The selected ROI
    │   ├── timeseries.csv
    │   ├── timeseries.npz
    │   ├── kinetic_fit.png
    │   └── kinetic_results.json             # Changed from .txt to .json
    │
    ├── parameter_maps/                      # Parameter mapping outputs
    │   ├── maps.npz
    │   ├── metadata.json
    │   ├── visualization.png
    │   └── nifti/                           # Optional NIfTI exports
    │       ├── kb_map.nii.gz
    │       └── ...
    │
    └── image_tools/                         # Averaged/difference images
        ├── averaged_frames0-5.npz
        └── difference_A0-5_B10-15.npz
```

## Dataset Manifest

New file `dataset_manifest.json` tracks analysis state:

```json
{
  "dataset_name": "example_dicom",
  "created_at": "2024-01-15 10:30:00",
  "updated_at": "2024-01-15 14:22:00",
  "source": {
    "dicom_path": "/path/to/original.dcm",
    "dicom_filename": "original.dcm",
    "shape": [128, 128, 9, 26],
    "spacing": [0.13, 0.13, 1.0]
  },
  "analysis": {
    "registration": {
      "completed": true,
      "timestamp": "2024-01-15 10:35:00",
      "format": "dicom",
      "n_timepoints": 26
    },
    "roi_analysis": {
      "completed": true,
      "timestamp": "2024-01-15 11:00:00",
      "roi_type": "rectangle",
      "z_slice": 4,
      "injection_index": 3
    },
    "parameter_maps": {
      "completed": false
    },
    "image_tools": {
      "averaged_images": ["frames0-5"],
      "difference_images": ["A0-5_B10-15"]
    }
  },
  "files": {
    "registered/dicoms/": "DICOM series (26 files)",
    "registered/registration_metrics.json": "Registration quality metrics",
    "roi_analysis/timeseries.csv": "Time series data",
    "roi_analysis/kinetic_fit.png": "Kinetic model fit plot"
  }
}
```

## Files Requiring Changes

### 1. `proxyl_analysis/io.py`

**Add new functions:**

```python
def create_dataset_directory(output_base: str, dataset_name: str) -> Path:
    """Create and initialize a dataset directory structure."""

def get_dataset_path(output_base: str, dataset_name: str, subdir: str = None) -> Path:
    """Get path within dataset directory, creating subdirs as needed."""

def load_dataset_manifest(dataset_dir: str) -> dict:
    """Load dataset manifest, or return empty manifest if not exists."""

def save_dataset_manifest(dataset_dir: str, manifest: dict) -> None:
    """Save updated dataset manifest."""

def update_manifest_analysis(dataset_dir: str, analysis_type: str, data: dict) -> None:
    """Update manifest with analysis completion info."""
```

### 2. `proxyl_analysis/registration.py`

**Modify `save_registration_data()` (lines 182-236):**
- Save to `{dataset_dir}/registered/` subdirectory
- Update manifest with registration info
- Call DICOM export (T010) to `registered/dicoms/`

**Modify `load_registration_data()` (lines 239-283):**
- Look in `{dataset_dir}/registered/` subdirectory
- Fall back to old flat structure for backward compatibility

### 3. `proxyl_analysis/parameter_mapping.py`

**Modify `save_parameter_maps()` (lines 524-614):**
- Save to `{dataset_dir}/parameter_maps/` subdirectory
- NIfTI files go to `parameter_maps/nifti/`
- Update manifest

**Remove separate directory creation** - use dataset directory instead

### 4. `proxyl_analysis/run_analysis.py`

**Major refactoring needed:**

- Line 304: Change from `registration_{dicom_name}` to unified dataset directory
- Line 595-600: Save kinetic results to `roi_analysis/` subdirectory
- Line 713, 758: Remove separate parameter_maps directory creation
- Line 995: Save timeseries to `roi_analysis/` subdirectory

**Add dataset initialization:**
```python
# Create dataset directory
dataset_name = Path(args.dicom).stem
dataset_dir = create_dataset_directory(args.output_dir, dataset_name)

# Initialize manifest with source info
manifest = {
    "dataset_name": dataset_name,
    "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    "source": {
        "dicom_path": str(args.dicom),
        ...
    }
}
save_dataset_manifest(dataset_dir, manifest)
```

### 5. `proxyl_analysis/ui/main_menu.py`

**Modify to use dataset directory:**
- Pass `dataset_dir` instead of separate `registration_dir`, `param_map_dir`
- Update `_handle_export()` to use new paths
- Update info display to show dataset structure

### 6. `proxyl_analysis/ui/image_tools.py`

**Modify save locations:**
- Line 559-591: Save to `{dataset_dir}/image_tools/` instead of root
- Pass `dataset_dir` through dialog initialization

### 7. `proxyl_analysis/ui/injection.py`

**Modify CSV export:**
- Line 166-184: Save to `{dataset_dir}/roi_analysis/`

### 8. `proxyl_analysis/ui/fitting.py`

**Modify plot save:**
- Default save location should be `{dataset_dir}/roi_analysis/`

## Backward Compatibility

1. **Loading old data:**
   - `load_registration_data()` checks for new structure first, then old
   - Manifest loading returns empty dict if not found

2. **Migration helper (optional):**
   ```python
   def migrate_legacy_dataset(output_dir: str, dicom_name: str) -> Path:
       """Migrate old scattered directories to new unified structure."""
   ```

## Implementation Order

1. **Phase 1: Infrastructure**
   - Add dataset directory functions to `io.py`
   - Add manifest functions

2. **Phase 2: Registration** (coordinate with T010)
   - Update `registration.py` save/load
   - Update `run_analysis.py` dataset creation

3. **Phase 3: Parameter Maps**
   - Update `parameter_mapping.py`
   - Remove separate directory creation

4. **Phase 4: UI Components**
   - Update `main_menu.py`
   - Update `image_tools.py`
   - Update `injection.py`
   - Update `fitting.py`

5. **Phase 5: Documentation**
   - Update all docs with new structure
   - Add migration guide

## Benefits

1. **Self-contained:** All results for one dataset in one place
2. **Portable:** Easy to zip and share complete analysis
3. **Trackable:** Manifest shows exactly what analysis has been done
4. **Organized:** Clear separation of workflow stages
5. **Extensible:** Easy to add new analysis types

## Dependencies

- T010 (DICOM export) should be implemented first or in parallel
- Both tasks modify similar files in `io.py` and `registration.py`
