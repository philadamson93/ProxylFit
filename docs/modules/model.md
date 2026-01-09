# Kinetic Model Module

**File**: `proxyl_analysis/model.py`

The model module implements the extended Proxyl kinetic model for fitting contrast dynamics in MRI data.

## Extended Kinetic Model

ProxylFit uses an extended model that accounts for both tracer dynamics and non-tracer physiological effects:

```
I(t) = A0 + A1*(1 - exp(-kb*(t - t0))) * exp(-kd*(t - t0)) + A2*(1 - exp(-knt*(t - tmax)))
```

### Model Components

| Component | Formula | Description |
|-----------|---------|-------------|
| Baseline | `A0` | Constant background signal |
| Tracer uptake | `A1*(1 - exp(-kb*(t-t0)))` | Contrast buildup |
| Tracer decay | `exp(-kd*(t-t0))` | Contrast washout |
| Non-tracer | `A2*(1 - exp(-knt*(t-tmax)))` | Physiological effects |

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| **A0** | Baseline signal | > 0 | Intensity |
| **A1** | Tracer amplitude | > 0 | Intensity |
| **A2** | Non-tracer amplitude | Any | Intensity |
| **kb** | Buildup rate | 0.01 - 2.0 | 1/min |
| **kd** | Decay rate | 0.01 - 1.0 | 1/min |
| **knt** | Non-tracer rate | 0.001 - 0.2 | 1/min |
| **t0** | Injection time | ≥ 0 | minutes |
| **tmax** | Non-tracer onset | ≥ 0 | minutes |

## Functions

### fit_proxyl_kinetics()

Fit the extended kinetic model to time series data.

```python
from proxyl_analysis.model import fit_proxyl_kinetics

kb, kd, knt, fitted_signal, fit_results = fit_proxyl_kinetics(
    time, signal, time_units='minutes'
)
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `time` | np.ndarray | Time points array |
| `signal` | np.ndarray | Signal intensity values |
| `time_units` | str | Units for display ('minutes', 'seconds') |

**Returns**:
| Return | Type | Description |
|--------|------|-------------|
| `kb` | float | Buildup rate constant |
| `kd` | float | Decay rate constant |
| `knt` | float | Non-tracer rate constant |
| `fitted_signal` | np.ndarray | Model-fitted curve |
| `fit_results` | dict | Complete results with errors |

**fit_results Dictionary**:
```python
{
    'A0': float,           # Baseline
    'A1': float,           # Tracer amplitude
    'A2': float,           # Non-tracer amplitude
    'kb': float,           # Buildup rate
    'kd': float,           # Decay rate
    'knt': float,          # Non-tracer rate
    't0': float,           # Injection time
    'tmax': float,         # Non-tracer onset
    'A0_error': float,     # Parameter uncertainties...
    # ... more errors ...
    'r_squared': float,    # R² fit quality
    'rmse': float,         # Root mean squared error
    'residuals': np.ndarray,
    'covariance_matrix': np.ndarray,
    'time_units': str
}
```

### select_injection_time()

Interactive selection of injection time point (matplotlib version).

```python
from proxyl_analysis.model import select_injection_time

injection_idx = select_injection_time(time, signal, 'minutes', './output')
```

> **Note**: For the Qt-based version, use `select_injection_time_qt()` from `ui.py`.

### plot_fit_results()

Visualize fitting results (matplotlib version).

```python
from proxyl_analysis.model import plot_fit_results

plot_fit_results(time, signal, fitted_signal, fit_results, 'output.png')
```

> **Note**: For the Qt-based version, use `plot_fit_results_qt()` from `ui.py`.

### calculate_derived_parameters()

Calculate derived kinetic parameters (half-lives, rate ratios).

```python
from proxyl_analysis.model import calculate_derived_parameters

derived = calculate_derived_parameters(
    kb, kd, knt,
    fit_results['kb_error'],
    fit_results['kd_error'],
    fit_results['knt_error']
)
```

**Returns**:
```python
{
    'half_life_buildup': float,     # ln(2)/kb
    'half_life_decay': float,       # ln(2)/kd
    'half_life_nontracer': float,   # ln(2)/knt
    'rate_ratio_buildup_decay': float,    # kb/kd
    'rate_ratio_buildup_nontracer': float, # kb/knt
    # ... with corresponding errors
}
```

## Fitting Algorithm

The module uses scipy's `curve_fit` with:

1. **Method**: Trust Region Reflective (`trf`) algorithm
2. **Bounds**: Physically meaningful parameter constraints
3. **Initial estimates**: Automatic estimation from data
4. **Fallback**: Alternative `dogbox` method if primary fails

### Fitting Process

```python
# Internal implementation
popt, pcov = curve_fit(
    proxyl_kinetic_model_extended,
    time, signal,
    p0=initial_params,
    bounds=(lower_bounds, upper_bounds),
    maxfev=5000,
    method='trf',
    ftol=1e-8, xtol=1e-8
)
```

## Requirements

- Minimum **8 timepoints** for extended model fitting
- Data should start at or before injection time
- Signal should show clear contrast dynamics

## Example Workflow

```python
import numpy as np
from proxyl_analysis.model import (
    fit_proxyl_kinetics,
    calculate_derived_parameters,
    print_fit_summary
)
from proxyl_analysis.ui import plot_fit_results_qt

# Prepare data (after ROI extraction)
time = np.arange(60) * (70.0 / 60.0)  # 60 timepoints, 70s intervals
signal = roi_timeseries  # From compute_roi_timeseries()

# Fit model
kb, kd, knt, fitted, results = fit_proxyl_kinetics(time, signal)

# Print summary
print_fit_summary(results)

# Calculate derived parameters
derived = calculate_derived_parameters(
    kb, kd, knt,
    results['kb_error'], results['kd_error'], results['knt_error']
)

# Visualize
plot_fit_results_qt(time, signal, fitted, results, 'fit.png')
```

## Troubleshooting

### Fitting Fails to Converge

**Symptoms**: ValueError or poor R² values

**Solutions**:
1. Ensure injection time is correctly identified
2. Check signal quality (noise, artifacts)
3. Try different initial parameter estimates
4. Increase `maxfev` for more iterations

### Parameters at Bounds

**Symptoms**: Warning about parameters hitting bounds

**Solutions**:
1. Check if bounds are appropriate for your data
2. Consider if the extended model is suitable
3. Try the simpler two-parameter model

### Large Uncertainties

**Symptoms**: Error bars >> parameter values

**Solutions**:
1. Increase signal-to-noise ratio (larger ROI)
2. Check for motion artifacts
3. Ensure adequate time resolution

## See Also

- [Parameter Mapping](parameter_mapping.md) - Spatial parameter maps
- [ROI Selection](roi_selection.md) - Extract time series
- [UI Module](../ui.md) - Qt-based visualization
