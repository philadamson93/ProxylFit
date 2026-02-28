Reference: docs/claude_ops.md

**Status: Completed** (2026-02-27)

# T008: Parallel Parameter Map Generation

## Goal

Parallelize the per-voxel fitting loop in `create_parameter_maps` to achieve near-linear speedup with CPU core count. Each voxel's nonlinear curve fit (~50-200ms) is independent, making this an ideal candidate for `ProcessPoolExecutor`.

## Approach

- Extract the inner loop body into `_fit_single_position(args)` — a standalone worker function
- Use `multiprocessing.shared_memory.SharedMemory` to share the large 4D image array across worker processes (zero-copy)
- `_run_parallel()` dispatches work via `ProcessPoolExecutor` with `as_completed` for progress tracking
- `_run_sequential()` fallback uses the same worker function for behavioral parity and debugging
- New `parallel=True` parameter on `create_parameter_maps()` (default enabled)
- Progress/cancellation interface unchanged — UI code requires no modifications

## Files Modified

- `proxyl_analysis/parameter_mapping.py` — refactored `create_parameter_maps`, added worker infrastructure
- `tests/test_parallel_parameter_maps.py` — 10 new tests (7 unit + 3 QThread integration)
- `docs/changelog.md` — version 1.3.0 entry
- `docs/plan-parallel-parameter-maps.md` — status updated to Completed

## Verification

- 10/10 new tests pass
- 19/19 existing stride regression tests pass
- 182/182 non-real-data tests pass (real data tests have pre-existing hang)
