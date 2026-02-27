# Plan: Parallel Parameter Map Generation

**Created**: 2026-02-27
**Status**: Future Work
**Priority**: Performance

## Problem

Parameter map generation is currently single-threaded. The `create_parameter_maps` function iterates over every voxel sequentially in a triple-nested loop, calling `fit_proxyl_kinetics` for each position one at a time. For large ROIs this is slow since each nonlinear curve fit (scipy.optimize) is CPU-bound.

## Why Parallelization Fits

- Each pixel's fit is **completely independent** — no shared mutable state between voxels
- The workload is CPU-bound (scipy curve fitting), not I/O-bound
- Typical ROIs contain hundreds to thousands of positions, providing ample work to distribute
- Expected speedup: **near-linear with core count** (e.g., 4-8x on typical hardware)

## Proposed Approach

### Option A: `concurrent.futures.ProcessPoolExecutor` (Recommended)

- Standard library, no new dependency
- Simple map-over-inputs pattern
- ProcessPool avoids GIL limitations for CPU-bound scipy work

### Option B: `joblib.Parallel`

- Slightly cleaner API for embarrassingly parallel loops
- Would add a new dependency (joblib)
- Well-tested with numpy/scipy workloads

### Recommendation

Use **Option A** (`ProcessPoolExecutor`) to avoid adding dependencies.

## Implementation Outline

### 1. Extract a per-pixel worker function

Pull the inner loop body (signal extraction, quality checks, curve fitting) into a standalone function that takes serializable arguments and returns a result tuple:

```python
def _fit_single_position(args):
    """Fit kinetics at a single (x, y, z) position. Designed for multiprocessing."""
    x, y, z, image_4d, time_array, window_size, kernel_type, signal_threshold, time_units = args
    # ... extract signal, quality checks, call fit_proxyl_kinetics ...
    # return (x, y, z, results_dict) or (x, y, z, None) on failure
```

### 2. Build a work list

Before entering the loop, collect all `(x, y, z)` positions that pass the ROI mask filter into a list of argument tuples.

### 3. Parallel dispatch

```python
from concurrent.futures import ProcessPoolExecutor
import os

max_workers = max(1, os.cpu_count() - 1)  # leave one core free

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(_fit_single_position, work_items, chunksize=16))
```

### 4. Collect results back into maps

Iterate over results and assign fitted values into the output arrays, same as today.

### 5. Progress reporting

- `ProcessPoolExecutor.map` doesn't support per-item callbacks directly
- Use `executor.submit` + `as_completed` instead, updating the progress callback as futures complete
- Alternatively, batch results and update progress periodically

### 6. Cancellation support

- Check a shared `threading.Event` or `multiprocessing.Event` flag
- On cancel request, call `executor.shutdown(wait=False, cancel_futures=True)` (Python 3.9+)

## Key Considerations

- **Serialization overhead**: The 4D image array must be passed to worker processes. For large datasets, consider using shared memory (`multiprocessing.shared_memory`) to avoid copying the full array per worker.
- **Stride support**: Works unchanged — just build the work list with stride-spaced positions.
- **Per-voxel logging**: Current per-fit print statements would flood output in parallel. Reduce to summary logging only.
- **Fallback**: Keep the sequential path available (e.g., `parallel=False` argument) for debugging and environments where multiprocessing is problematic.
- **GUI thread safety**: The fitting already runs in a background QThread. The parallel workers would be spawned from that thread — this is fine since ProcessPoolExecutor manages its own processes.

## Files to Modify

| File | Changes |
|------|---------|
| `proxyl_analysis/parameter_mapping.py` | Extract worker function, add parallel dispatch in `create_parameter_maps` |
| `proxyl_analysis/ui/main_menu.py` | No changes needed (progress callback interface stays the same) |

## Testing

- Compare output maps (sequential vs parallel) for bit-exact agreement
- Benchmark wall-clock time on a representative dataset
- Verify cancellation works cleanly
- Test with stride > 1
