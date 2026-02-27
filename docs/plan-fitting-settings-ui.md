# Plan: Expose Fitting Parameters & Constraints via UI

**Created**: 2026-02-27
**Status**: Future Work
**Priority**: Usability / Reproducibility

## Problem

Fitting parameters (bounds, thresholds, tolerances) are currently hardcoded in `model.py` and `parameter_mapping.py`. Users cannot adjust them without editing source code, and the settings used for a given fit are not fully recorded in output metadata.

## What's Currently Hardcoded

### Rate constant bounds (model.py, fit_proxyl_kinetics)

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| kb (buildup) | 0.001 | 2.0 /min |
| kd (decay) | 0.001 | 1.0 /min |
| knt (non-tracer) | 0.001 | 0.2 /min |
| A0 (baseline) | 0 | 2 × max(signal) |
| A1 (tracer amp) | 0 | 3 × signal_range |
| A2 (non-tracer amp) | -signal_range | +signal_range |

### Quality thresholds (parameter_mapping.py, create_parameter_maps)

| Threshold | Current Value | Purpose |
|-----------|---------------|---------|
| min_signal_threshold | 0.1 (0.15 in enhanced workflow) | Skip low-signal voxels |
| R² acceptance | 0.1 | Reject poor fits |
| CV noise cutoff | 2.0 | Skip noisy signals |

### Optimizer settings (model.py)

| Setting | Primary (TRF) | Fallback (Dogbox) |
|---------|---------------|-------------------|
| maxfev | 5000 | 2000 |
| ftol | 1e-8 | 1e-6 |
| xtol | 1e-8 | 1e-6 |

## What's Already Configurable via UI

The `ParameterMappingOptionsDialog` (ui/parameter_map_options.py) already exposes:
- Slice mode (single vs all)
- ROI processing toggle
- Kernel type and window size
- Stride
- Injection time selection

## Proposed Design

### 1. New "Fitting Settings" dialog

A modal dialog accessible from a button in the Parameter Maps section of the main menu (next to "Create Parameter Maps"). Contains three collapsible sections:

**Section A — Rate Constant Bounds**
- Spinboxes for lower/upper bounds of kb, kd, knt
- Sensible min/max limits on the spinboxes themselves to prevent nonsense values
- "Reset to Defaults" button

**Section B — Quality Thresholds**
- Signal threshold (0.0–1.0 slider or spinbox)
- R² acceptance threshold (0.0–1.0)
- CV noise cutoff (0.5–5.0)

**Section C — Optimizer (Advanced, collapsed by default)**
- Max iterations (maxfev)
- Tolerance (ftol/xtol)
- Fallback toggle (enable/disable dogbox fallback)

### 2. Settings data flow

```
FittingSettingsDialog
  → returns dict of all settings
  → passed through to create_parameter_maps() and fit_proxyl_kinetics()
  → stored in result metadata dict
  → saved to parameter_maps_metadata.json
```

### 3. Plumb settings through existing functions

- `fit_proxyl_kinetics()` — add optional `bounds_override` and `optimizer_settings` kwargs (fall back to current hardcoded values when None)
- `create_parameter_maps()` — add optional `fitting_settings` dict kwarg; pass quality thresholds + fitting kwargs through
- Keep all current defaults unchanged so existing call sites are unaffected

### 4. Save settings as metadata

Extend the metadata dict saved with parameter maps to include a `fitting_settings` key:

```python
'fitting_settings': {
    'kb_bounds': [0.001, 2.0],
    'kd_bounds': [0.001, 1.0],
    'knt_bounds': [0.001, 0.2],
    'min_signal_threshold': 0.1,
    'r_squared_threshold': 0.1,
    'cv_threshold': 2.0,
    'maxfev': 5000,
    'ftol': 1e-8,
    'xtol': 1e-8,
    'fallback_enabled': True
}
```

This also applies to single ROI kinetic fits — the settings used should be recorded in the fit results dict.

### 5. Persist user preferences across sessions

Save last-used fitting settings to the session NPZ or a small JSON sidecar so they carry over when reopening a dataset. Load them as defaults when the dialog opens.

## Files to Modify

| File | Changes |
|------|---------|
| `proxyl_analysis/ui/fitting_settings.py` | **New file** — FittingSettingsDialog widget |
| `proxyl_analysis/ui/main_menu.py` | Add "Fitting Settings" button in parameter maps section |
| `proxyl_analysis/model.py` | Add optional kwargs to `fit_proxyl_kinetics` for bounds/tolerances |
| `proxyl_analysis/parameter_mapping.py` | Pass fitting_settings through; add to metadata |
| `proxyl_analysis/run_analysis.py` | Wire settings dialog into the parameter mapping flow |

## UI Placement

The "Fitting Settings" button sits in the Parameter Maps section of the main menu, above or beside "Create Parameter Maps". It opens the settings dialog; chosen settings persist for the session and are passed along when the user proceeds to create maps or run a kinetic fit.

## Testing

- Verify defaults match current hardcoded behavior exactly (regression)
- Confirm modified bounds actually affect fitting output
- Check metadata JSON contains all settings after a parameter map run
- Test "Reset to Defaults" restores original values
- Test session persistence of custom settings
