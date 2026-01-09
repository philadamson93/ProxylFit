# T005: Pixel-Level Parameter Maps (kb, kd)

**Status**: planned
**Priority**: high
**Created**: 2025-01-08

## Description

Generate kb and kd parameter maps at individual pixel/voxel level, fitting the full extended kinetic model to each voxel's time series independently.

## Rationale

- True voxel-wise analysis reveals local heterogeneity
- Current sliding window approach smooths spatial variations
- Individual pixel fitting needed for detailed tumor characterization
- Enables correlation with histology at fine spatial scale

## Current Implementation

The existing `parameter_mapping.py` uses a sliding window approach:
- Averages signal within NxNxN window
- Fits kinetic model to averaged signal
- Assigns parameters to center voxel
- Results in spatially smoothed maps

## Answered Questions

| Question | Answer |
|----------|--------|
| Quality threshold | None for now, needs further investigation |
| Model | Full 8-parameter extended model |
| Computation time | Report progress per voxel with status bar so user can kill if too slow |

## Requirements

- Fit full kinetic model to each voxel's time series independently
- Generate kb, kd, knt, and other parameter maps at native resolution
- Progress bar showing voxel-by-voxel progress
- Allow user to cancel if taking too long
- Save all parameter maps

## Implementation Notes

```python
from tqdm import tqdm

def fit_pixel_level_maps(image_4d, time, injection_idx, time_units='minutes'):
    """
    Fit kinetic model to each voxel independently.

    Parameters
    ----------
    image_4d : np.ndarray
        Shape [x, y, z, t]
    time : np.ndarray
        Time array
    injection_idx : int
        Injection time index
    time_units : str
        Time units for fitting

    Returns
    -------
    dict of parameter maps, each shape [x, y, z]
    """
    x_dim, y_dim, z_dim, t_dim = image_4d.shape
    total_voxels = x_dim * y_dim * z_dim

    # Initialize output maps
    param_names = ['kb', 'kd', 'knt', 'A0', 'A1', 'A2', 't0', 'tmax', 'r_squared']
    maps = {name: np.full((x_dim, y_dim, z_dim), np.nan) for name in param_names}

    # Trim time to post-injection
    time_fit = time[injection_idx:]

    # Progress bar
    pbar = tqdm(total=total_voxels, desc="Fitting voxels", unit="voxel")

    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(z_dim):
                signal = image_4d[i, j, k, injection_idx:]

                try:
                    kb, kd, knt, fitted, results = fit_proxyl_kinetics(
                        time_fit, signal, time_units
                    )

                    maps['kb'][i, j, k] = kb
                    maps['kd'][i, j, k] = kd
                    maps['knt'][i, j, k] = knt
                    maps['A0'][i, j, k] = results['A0']
                    maps['A1'][i, j, k] = results['A1']
                    maps['A2'][i, j, k] = results['A2']
                    maps['t0'][i, j, k] = results['t0']
                    maps['tmax'][i, j, k] = results['tmax']
                    maps['r_squared'][i, j, k] = results['r_squared']

                except Exception:
                    # Failed fit - leave as NaN
                    pass

                pbar.update(1)

    pbar.close()
    return maps
```

## Progress Reporting

Using `tqdm` for progress bar:
```
Fitting voxels:  45%|████████████               | 66,355/147,456 [12:34<15:21, 88.0 voxel/s]
```

User can press Ctrl+C to cancel if taking too long.

## Parallelization (Future Enhancement)

For faster processing, could add multiprocessing:
```python
from multiprocessing import Pool

# Not implementing initially - can add if needed
```

## CLI Interface

```bash
# Generate pixel-level parameter maps
python run_analysis.py --dicom data.dcm --z 4 --pixel-level-maps
```

## Output Files

```
output/
├── pixel_maps/
│   ├── kb_map.npy
│   ├── kd_map.npy
│   ├── knt_map.npy
│   ├── A0_map.npy
│   ├── A1_map.npy
│   ├── r_squared_map.npy
│   └── pixel_maps_visualization.png
```

## Acceptance Criteria

- [ ] Fit full kinetic model to individual voxels
- [ ] Progress bar with voxel count and ETA
- [ ] Can be cancelled with Ctrl+C
- [ ] Generate all parameter maps (kb, kd, knt, A0, A1, A2, r_squared)
- [ ] Handle failed fits gracefully (NaN values)
- [ ] Save maps to output directory
- [ ] Visualization of parameter maps
- [ ] CLI option `--pixel-level-maps`
- [ ] Documentation on interpretation and limitations
