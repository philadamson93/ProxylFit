"""
Parameter mapping module for creating spatial maps of kinetic parameters.

This module implements sliding window parameter fitting to create 2D/3D maps
of kb (binding rate) and kd (decay rate) parameters across the entire image.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any, Union
import os
import time
from pathlib import Path
import sys

from .model import fit_proxyl_kinetics, add_proxylfit_logo, set_proxylfit_style, select_injection_time
from .roi_selection import ManualContourROISelector


# ---------------------------------------------------------------------------
# Per-voxel fitting worker (used by both the sequential path and the
# multiprocessing.Pool path). Worker processes get their context via
# `_init_pixel_worker` so that `registered_4d` is pickled once per worker
# instead of once per voxel.
# ---------------------------------------------------------------------------

_PIXEL_WORKER_CTX: Dict[str, Any] = {}


def _init_pixel_worker(registered_4d, time_array, kernel_type, window_size,
                       signal_threshold, time_units,
                       pre_injection_4d=None,
                       steady_state_time=None,
                       excluded_indices=None):
    """Pool initializer — stash the read-only fitting inputs as worker globals.

    ``pre_injection_4d`` is the slice of the original 4D volume *before*
    the injection time. When provided, the worker extracts the same
    kernel window from it for each voxel and passes the resulting
    per-voxel baseline signal to ``fit_proxyl_kinetics`` so A0 is
    pinned to the pre-injection mean — same behavior as the kinetic
    fit page. When None (no injection time, or all timepoints used),
    the per-voxel fit falls back to the 7-param free-A0 mode.

    ``steady_state_time`` (in time_units) is forwarded to every voxel's
    fit so the knt lower bound = ln(20)/t_steady applies uniformly,
    keeping the parameter map consistent with the curve-fit page's
    NTE constraint. None falls back to the legacy 0.001/min floor.

    ``excluded_indices`` (a sequence of int) is forwarded to every
    voxel's fit so the same bolus / artefact points masked on the
    kinetic-fit page are masked here too. Indices are interpreted in
    the **post-injection** array (after create_parameter_maps slices
    off the pre-injection portion). None means no exclusions.
    """
    _PIXEL_WORKER_CTX['registered_4d'] = registered_4d
    _PIXEL_WORKER_CTX['time_array'] = time_array
    _PIXEL_WORKER_CTX['kernel_type'] = kernel_type
    _PIXEL_WORKER_CTX['window_size'] = window_size
    _PIXEL_WORKER_CTX['signal_threshold'] = signal_threshold
    _PIXEL_WORKER_CTX['time_units'] = time_units
    _PIXEL_WORKER_CTX['pre_injection_4d'] = pre_injection_4d
    _PIXEL_WORKER_CTX['steady_state_time'] = steady_state_time
    _PIXEL_WORKER_CTX['excluded_indices'] = (
        list(excluded_indices) if excluded_indices else None
    )


def _fit_pixel(pos):
    """
    Fit a single (x, y, z, z_idx) position from `_PIXEL_WORKER_CTX`.

    Returns
    -------
    tuple or None
        (x, y, z_idx, kb, kd, knt, r_squared, A1, A2, A0, t0, tmax,
        A0_est, A2_est) on success, or None when the pixel is below the
        signal threshold, too noisy, or the fit fails / is too poor
        (R² <= 0.1).
    """
    x, y, z, z_idx = pos
    ctx = _PIXEL_WORKER_CTX
    registered_4d = ctx['registered_4d']
    time_array = ctx['time_array']
    kernel_type = ctx['kernel_type']
    window_size = ctx['window_size']
    signal_threshold = ctx['signal_threshold']
    time_units = ctx['time_units']
    pre_injection_4d = ctx.get('pre_injection_4d')
    steady_state_time = ctx.get('steady_state_time')
    excluded_indices = ctx.get('excluded_indices')

    # Extract signal using the configured kernel
    if kernel_type == 'sliding_window':
        window_signal = _extract_sliding_window_signal(
            registered_4d, x, y, z, window_size
        )
    else:
        window_signal = _extract_kernel_signal(
            registered_4d, x, y, z, window_size, kernel_type
        )

    # Threshold checks (same as the original sequential loop)
    max_signal = float(np.max(window_signal))
    min_signal = float(np.min(window_signal))
    signal_variation = max_signal - min_signal
    if max_signal < signal_threshold or signal_variation < signal_threshold * 0.1:
        return None

    mean_signal = float(np.mean(window_signal))
    cv = (float(np.std(window_signal)) / mean_signal) if mean_signal > 0 else float('inf')
    if cv > 2.0:
        return None

    # Per-voxel pre-injection signal: same kernel applied to the
    # pre-injection slice of the volume (when one was preserved by
    # create_parameter_maps). Pinning A0 here makes the per-voxel
    # parameter map match the curve-fit page's results for the same
    # ROI; without it, A0 is free per voxel and biases A1/A2/etc.
    pre_injection_signal = None
    if pre_injection_4d is not None and pre_injection_4d.shape[3] > 0:
        if kernel_type == 'sliding_window':
            pre_injection_signal = _extract_sliding_window_signal(
                pre_injection_4d, x, y, z, window_size
            )
        else:
            pre_injection_signal = _extract_kernel_signal(
                pre_injection_4d, x, y, z, window_size, kernel_type
            )

    # Fit. verbose=False suppresses the per-voxel "Note: <param> at upper
    # bound" / "Warning: covariance issues" / "First fitting attempt failed"
    # diagnostics that fit_proxyl_kinetics emits — useful for a single ROI
    # fit, but at thousands of voxels per run they add measurable I/O cost
    # on a Mac terminal and obscure the real progress messages.
    try:
        kb, kd, knt, _fitted, fit_results = fit_proxyl_kinetics(
            time_array, window_signal, time_units, verbose=False,
            pre_injection_signal=pre_injection_signal,
            steady_state_time=steady_state_time,
            excluded_indices=excluded_indices,
        )
    except Exception:
        return None

    if fit_results['r_squared'] <= 0.1:
        return None

    return (
        x, y, z_idx,
        kb, kd, knt,
        fit_results['r_squared'],
        fit_results['A1'], fit_results['A2'], fit_results['A0'],
        fit_results['t0'], fit_results['tmax'],
        # Initial estimates of A0 and A2 (from estimate_initial_parameters_extended).
        # These power the %NTE_est map and may be absent on legacy fit_results
        # dicts — fall back to the fitted values so the result tuple shape
        # stays fixed across worker versions.
        fit_results.get('A0_est', fit_results['A0']),
        fit_results.get('A2_est', fit_results['A2']),
    )


def create_parameter_maps(registered_4d: np.ndarray,
                         time_array: np.ndarray,
                         window_size: Union[int, Tuple[int, int, int]] = 5,
                         z_slice: Optional[int] = None,
                         min_signal_threshold: float = 0.1,
                         time_units: str = 'minutes',
                         progress_callback: Optional[callable] = None,
                         roi_mask: Optional[np.ndarray] = None,
                         kernel_type: str = 'sliding_window',
                         injection_time_index: Optional[int] = None,
                         stride: int = 1,
                         n_workers: Optional[int] = None,
                         steady_state_time: Optional[float] = None,
                         excluded_indices: Optional[list] = None) -> Dict[str, np.ndarray]:
    """
    Create spatial parameter maps using sliding window or convolution approach.
    
    Parameters
    ----------
    registered_4d : np.ndarray
        Registered 4D data with shape [x, y, z, t]
    time_array : np.ndarray
        Time points for fitting
    window_size : int or tuple of int
        Size of sliding window/kernel. If int, creates cubic window (NxNxN). 
        If tuple (wx, wy, wz), creates rectangular window (wx x wy x wz)
    z_slice : int, optional
        If provided, only process this z-slice (2D mapping)
        If None, process all slices (3D mapping)
    min_signal_threshold : float
        Minimum signal level (relative to max) to attempt fitting
    time_units : str
        Time units for fitting
    progress_callback : callable, optional
        Function to call with progress updates (progress_pct, current_position, total_positions)
    roi_mask : np.ndarray, optional
        2D boolean mask defining region of interest for parameter mapping.
        If provided, only pixels within this mask will be processed.
    kernel_type : str
        Type of kernel: 'sliding_window', 'gaussian', 'uniform', 'box'
    injection_time_index : int, optional
        Index in time_array where injection occurred. If provided, only data
        from this point onwards will be used for fitting.
    stride : int
        Step size for spatial iteration. stride=1 fits every pixel (full resolution).
        stride=N fits every Nth pixel and fills surrounding NxN blocks with the
        fitted value (nearest-neighbor fill). Output maps remain full-size.
    n_workers : int, optional
        Number of worker processes used for parallel per-voxel fitting.
        Defaults to ``min(os.cpu_count(), 8)``. Pass ``1`` to force sequential
        execution. Pass ``0`` for the same auto-select behaviour as ``None``.

    Returns
    -------
    dict
        Dictionary containing parameter maps:
        - 'kb_map': Buildup rate map
        - 'kd_map': Decay rate map
        - 'knt_map': Non-tracer effect rate map
        - 'r_squared_map': R-squared goodness of fit map
        - 'a1_amplitude_map': Tracer amplitude (A1) parameter map
        - 'a2_amplitude_map': Non-tracer amplitude (A2) parameter map
        - 'baseline_map': Baseline (A0) parameter map
        - 't0_map': Tracer onset time (t0) parameter map
        - 'tmax_map': Non-tracer onset time (tmax) parameter map
        - 'mask': Boolean mask of successfully fitted voxels
        - 'roi_mask': Copy of input ROI mask (if provided)
    """
    x_size, y_size, z_size, t_size = registered_4d.shape
    
    # Parse window size
    if isinstance(window_size, int):
        window_x, window_y, window_z = window_size, window_size, window_size
    else:
        window_x, window_y, window_z = window_size
    
    # Handle injection time selection. Keep the pre-injection slice
    # of the volume around so each per-voxel fit can pin A0 to the
    # local pre-injection mean (same A0-pinning logic as the curve-fit
    # page — keeps parameter maps and curve fits in agreement).
    pre_injection_4d = None
    if injection_time_index is not None:
        if injection_time_index > 0:
            pre_injection_4d = registered_4d[:, :, :, :injection_time_index]
        # Trim time array and image data to start from injection
        time_array = time_array[injection_time_index:]
        registered_4d = registered_4d[:, :, :, injection_time_index:]
        print(f"Using data from injection time onwards: {len(time_array)} timepoints")
        if pre_injection_4d is not None:
            print(f"Pre-injection slice retained for per-voxel A0 pinning: "
                  f"{pre_injection_4d.shape[3]} timepoints")
    
    # Determine processing dimensions
    if z_slice is not None:
        # 2D processing - single slice
        z_start, z_end = z_slice, z_slice + 1
        output_shape = (x_size, y_size, 1)
    else:
        # 3D processing - all slices
        z_start, z_end = 0, z_size
        output_shape = (x_size, y_size, z_size)
    
    # Initialize output maps for extended model
    kb_map = np.full(output_shape, np.nan)
    kd_map = np.full(output_shape, np.nan)
    knt_map = np.full(output_shape, np.nan)
    r_squared_map = np.full(output_shape, np.nan)
    a1_amplitude_map = np.full(output_shape, np.nan)  # Tracer amplitude
    a2_amplitude_map = np.full(output_shape, np.nan)  # Non-tracer amplitude
    a0_est_map = np.full(output_shape, np.nan)        # Initial-estimate baseline
    a2_est_map = np.full(output_shape, np.nan)        # Initial-estimate non-tracer amp
    baseline_map = np.full(output_shape, np.nan)
    t0_map = np.full(output_shape, np.nan)  # Tracer onset time
    tmax_map = np.full(output_shape, np.nan)  # Non-tracer onset time
    fit_mask = np.zeros(output_shape, dtype=bool)
    
    # Calculate signal threshold
    max_signal = np.max(registered_4d)
    signal_threshold = min_signal_threshold * max_signal
    
    # Build the list of (x, y, z, z_idx) positions we'll fit. Doing this once
    # up front replaces the original "count first, fit second" pattern and
    # gives multiprocessing.Pool a flat work list to chew through.
    positions = []
    for z in range(z_start, z_end):
        z_idx = z if z_slice is None else 0
        for x in range(0, x_size, stride):
            for y in range(0, y_size, stride):
                if roi_mask is not None and not roi_mask[x, y]:
                    continue
                positions.append((x, y, z, z_idx))
    total_positions = len(positions)

    print(f"Creating parameter maps using {window_x}x{window_y}x{window_z} {kernel_type} kernel (stride={stride})...")
    if roi_mask is not None:
        print(f"Processing within ROI on {'single slice' if z_slice is not None else 'all slices'}: {total_positions} positions")
    else:
        print(f"Processing {'single slice' if z_slice is not None else 'all slices'}: {total_positions} positions")
    print(f"Signal threshold: {signal_threshold:.2f}")

    # Worker count: None/0 -> auto, capped at 8 to avoid pathological RAM use
    # on machines with many cores (each worker gets a copy of registered_4d).
    if not n_workers:
        n_workers = min(os.cpu_count() or 1, 8)
    n_workers = max(1, int(n_workers))
    use_pool = n_workers > 1 and total_positions >= 8 * n_workers

    if total_positions:
        print(
            f"Fitting with {n_workers} worker"
            f"{'s' if n_workers != 1 else ''} ("
            f"{'parallel' if use_pool else 'sequential'})"
        )

    # Stash inputs on the main process so the sequential path can reuse the
    # same _fit_pixel helper as the worker processes. Worker subprocesses get
    # their own copy via the Pool initializer below.
    _init_pixel_worker(
        registered_4d, time_array, kernel_type,
        (window_x, window_y, window_z), signal_threshold, time_units,
        pre_injection_4d=pre_injection_4d,
        steady_state_time=steady_state_time,
        excluded_indices=excluded_indices,
    )

    # Counter incremented inside _store on every non-None fit result.
    # Used as the basis for success_rate so the percentage reflects
    # iterated-positions converged, not filled pixels — without this,
    # stride>1 inflates the pixel count by stride² and pushes the rate
    # past 100% (a stride=3 run with ~53% convergence reads as ~478%).
    successful_count = [0]

    def _store(result):
        """Write a fit-result tuple into the output maps."""
        if result is None:
            return
        successful_count[0] += 1
        (x, y, z_idx, kb, kd, knt, r2,
         A1, A2, A0, t0, tmax, A0_est, A2_est) = result
        x_end_blk = min(x + stride, x_size)
        y_end_blk = min(y + stride, y_size)
        kb_map[x:x_end_blk, y:y_end_blk, z_idx] = kb
        kd_map[x:x_end_blk, y:y_end_blk, z_idx] = kd
        knt_map[x:x_end_blk, y:y_end_blk, z_idx] = knt
        r_squared_map[x:x_end_blk, y:y_end_blk, z_idx] = r2
        a1_amplitude_map[x:x_end_blk, y:y_end_blk, z_idx] = A1
        a2_amplitude_map[x:x_end_blk, y:y_end_blk, z_idx] = A2
        a0_est_map[x:x_end_blk, y:y_end_blk, z_idx] = A0_est
        a2_est_map[x:x_end_blk, y:y_end_blk, z_idx] = A2_est
        baseline_map[x:x_end_blk, y:y_end_blk, z_idx] = A0
        t0_map[x:x_end_blk, y:y_end_blk, z_idx] = t0
        tmax_map[x:x_end_blk, y:y_end_blk, z_idx] = tmax
        fit_mask[x:x_end_blk, y:y_end_blk, z_idx] = True

    start_time = time.time()
    cancelled = False
    current_position = 0

    if use_pool:
        from multiprocessing import Pool

        # ~16 chunks per worker keeps progress updates smooth without much
        # IPC overhead.
        chunksize = max(1, total_positions // (n_workers * 16))
        # Note: pre_injection_4d (positional-7), steady_state_time
        # (positional-8), and excluded_indices (positional-9) passed
        # here so each Pool subprocess gets the same per-voxel
        # A0-pinning data, knt-floor, and bolus-mask as the main
        # process.
        init_args = (
            registered_4d, time_array, kernel_type,
            (window_x, window_y, window_z), signal_threshold, time_units,
            pre_injection_4d, steady_state_time, excluded_indices,
        )

        with Pool(n_workers, initializer=_init_pixel_worker, initargs=init_args) as pool:
            iterator = pool.imap_unordered(_fit_pixel, positions, chunksize=chunksize)
            for current_position, result in enumerate(iterator, 1):
                _store(result)

                if progress_callback:
                    progress_pct = 100.0 * current_position / total_positions
                    if progress_callback(progress_pct, current_position,
                                         total_positions) is False:
                        print("Parameter mapping cancelled by user.")
                        cancelled = True
                        pool.terminate()
                        break
    else:
        for current_position, pos in enumerate(positions, 1):
            if progress_callback:
                progress_pct = 100.0 * current_position / total_positions
                if progress_callback(progress_pct, current_position,
                                     total_positions) is False:
                    print("Parameter mapping cancelled by user.")
                    cancelled = True
                    break
            _store(_fit_pixel(pos))

    elapsed_time = time.time() - start_time
    # successful_fits is the count of iterated positions where the fit
    # converged — capped at total_positions, so success_rate stays in
    # [0, 100]%. fit_mask sums many more pixels than that whenever
    # stride>1 (each successful position fills a stride×stride block),
    # so don't reuse it for the rate.
    successful_fits = successful_count[0]
    filled_pixels = int(np.sum(fit_mask))
    success_rate = (100.0 * successful_fits / total_positions) if total_positions else 0.0

    print(f"Parameter mapping completed in {elapsed_time:.1f} seconds")
    print(f"Successful fits: {successful_fits}/{total_positions} ({success_rate:.1f}%)")
    if filled_pixels != successful_fits:
        print(f"  Filled pixels (stride blocks): {filled_pixels}")

    # Derived percent maps:
    #   100*A1/A0     (%Enhancement)
    #   100*A2/A0     (%NTE — fitted)
    #   100*A2_est/A0_est (%NTE_est — initial-estimate version)
    # Voxels with A0<=0 (or A0_est<=0) divide-by-zero out to NaN, matching
    # the "render as —" behavior used for derived parameters elsewhere.
    with np.errstate(divide='ignore', invalid='ignore'):
        a0_safe = np.where(baseline_map > 0, baseline_map, np.nan)
        a0_est_safe = np.where(a0_est_map > 0, a0_est_map, np.nan)
        a1_percent_map = 100.0 * a1_amplitude_map / a0_safe
        a2_percent_map = 100.0 * a2_amplitude_map / a0_safe
        a2_percent_est_map = 100.0 * a2_est_map / a0_est_safe

    result = {
        'kb_map': kb_map,
        'kd_map': kd_map,
        'knt_map': knt_map,
        'r_squared_map': r_squared_map,
        'a1_amplitude_map': a1_amplitude_map,
        'a2_amplitude_map': a2_amplitude_map,
        'a0_est_map': a0_est_map,
        'a2_est_map': a2_est_map,
        'a1_percent_map': a1_percent_map,
        'a2_percent_map': a2_percent_map,
        'a2_percent_est_map': a2_percent_est_map,
        'baseline_map': baseline_map,
        't0_map': t0_map,
        'tmax_map': tmax_map,
        'mask': fit_mask,
        'metadata': {
            'window_size': window_size,
            'window_x': window_x,
            'window_y': window_y, 
            'window_z': window_z,
            'z_slice': z_slice,
            'time_units': time_units,
            'signal_threshold': signal_threshold,
            'success_rate': success_rate,
            'processing_time': elapsed_time,
            'total_positions': total_positions,
            'successful_fits': successful_fits,
            'kernel_type': kernel_type,
            'injection_time_index': injection_time_index,
            'stride': stride,
            'steady_state_time': steady_state_time,
            'excluded_indices': list(excluded_indices) if excluded_indices else [],
        }
    }
    
    # Add ROI mask to result if provided
    if roi_mask is not None:
        result['roi_mask'] = roi_mask.copy()
    
    return result


def _extract_sliding_window_signal(image_4d: np.ndarray, x: int, y: int, z: int, 
                                  window_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Extract time series from sliding window around specified voxel.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D image data [x, y, z, t]
    x, y, z : int
        Center voxel coordinates
    window_size : tuple of int
        Size of window (window_x, window_y, window_z)
        
    Returns
    -------
    np.ndarray
        Mean time series from window region
    """
    x_size, y_size, z_size, t_size = image_4d.shape
    window_x, window_y, window_z = window_size
    
    # Calculate window bounds with boundary checking
    x_radius = window_x // 2
    y_radius = window_y // 2
    z_radius = window_z // 2
    
    x_start = max(0, x - x_radius)
    x_end = min(x_size, x + x_radius + 1)
    y_start = max(0, y - y_radius)
    y_end = min(y_size, y + y_radius + 1)
    z_start = max(0, z - z_radius)
    z_end = min(z_size, z + z_radius + 1)
    
    # Extract window region
    window_region = image_4d[x_start:x_end, y_start:y_end, z_start:z_end, :]
    
    # Return mean signal across spatial dimensions
    return np.mean(window_region, axis=(0, 1, 2))


def _extract_kernel_signal(image_4d: np.ndarray, x: int, y: int, z: int,
                          kernel_size: Tuple[int, int, int], kernel_type: str) -> np.ndarray:
    """
    Extract time series using convolution kernel around specified voxel.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D image data [x, y, z, t]
    x, y, z : int
        Center voxel coordinates
    kernel_size : tuple of int
        Size of kernel (kx, ky, kz)
    kernel_type : str
        Type of kernel: 'gaussian', 'uniform', 'box'
        
    Returns
    -------
    np.ndarray
        1D time series extracted using kernel convolution
    """
    kx, ky, kz = kernel_size
    x_size, y_size, z_size, t_size = image_4d.shape
    
    # Define kernel bounds with bounds checking
    x_start = max(0, x - kx//2)
    x_end = min(x_size, x + kx//2 + 1)
    y_start = max(0, y - ky//2)
    y_end = min(y_size, y + ky//2 + 1)
    z_start = max(0, z - kz//2)
    z_end = min(z_size, z + kz//2 + 1)
    
    # Extract region
    region = image_4d[x_start:x_end, y_start:y_end, z_start:z_end, :]
    
    # Create kernel weights
    region_shape = region.shape[:3]
    if kernel_type == 'gaussian':
        # Create 3D Gaussian kernel
        from scipy.stats import multivariate_normal
        center = np.array(region_shape) / 2
        sigma = np.array(region_shape) / 6  # Standard deviation
        
        coords = np.mgrid[0:region_shape[0], 0:region_shape[1], 0:region_shape[2]]
        coords = np.stack(coords, axis=-1)
        
        kernel = multivariate_normal.pdf(coords.reshape(-1, 3), mean=center, cov=np.diag(sigma**2))
        kernel = kernel.reshape(region_shape)
        
    elif kernel_type == 'uniform':
        # Uniform weights (same as mean)
        kernel = np.ones(region_shape)
        
    elif kernel_type == 'box':
        # Box kernel (same as sliding window)
        kernel = np.ones(region_shape)
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply kernel convolution for each timepoint
    timeseries = np.zeros(t_size)
    for t in range(t_size):
        timeseries[t] = np.sum(region[:, :, :, t] * kernel)
    
    return timeseries


def visualize_parameter_maps(param_maps: Dict[str, np.ndarray], 
                           spacing: Tuple[float, float, float],
                           output_dir: Optional[str] = None,
                           z_slice: Optional[int] = None) -> None:
    """
    Create visualization of parameter maps.
    
    Parameters
    ----------
    param_maps : dict
        Dictionary of parameter maps from create_parameter_maps()
    spacing : tuple
        Voxel spacing (x, y, z)
    output_dir : str, optional
        Directory to save plots
    z_slice : int, optional
        Z-slice to display (for 3D data)
    """
    kb_map = param_maps['kb_map']
    kd_map = param_maps['kd_map']
    knt_map = param_maps['knt_map']
    r_squared_map = param_maps['r_squared_map']
    mask = param_maps['mask']
    metadata = param_maps['metadata']
    
    # Handle 3D vs 2D display
    if kb_map.ndim == 3 and kb_map.shape[2] > 1:
        # 3D data - select middle slice if not specified
        if z_slice is None:
            z_slice = kb_map.shape[2] // 2
        kb_slice = kb_map[:, :, z_slice].T
        kd_slice = kd_map[:, :, z_slice].T
        knt_slice = knt_map[:, :, z_slice].T
        r2_slice = r_squared_map[:, :, z_slice].T
        mask_slice = mask[:, :, z_slice].T
        slice_title = f" (z={z_slice})"
    else:
        # 2D data
        kb_slice = kb_map[:, :, 0].T
        kd_slice = kd_map[:, :, 0].T
        knt_slice = knt_map[:, :, 0].T
        r2_slice = r_squared_map[:, :, 0].T
        mask_slice = mask[:, :, 0].T
        slice_title = f" (z={metadata.get('z_slice', 'single')})"
    
    # Apply consistent styling
    set_proxylfit_style()
    
    # Create figure with subplots and padding (extra bottom margin for logo)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Lower the top of the subplot area and raise the suptitle to avoid overlap
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.88, hspace=0.3, wspace=0.2)
    
    # Enhanced title with ROI information
    roi_info = ""
    if 'roi_mask' in param_maps:
        roi_pixels = np.sum(param_maps['roi_mask'])
        roi_info = f" (ROI: {roi_pixels} pixels)"
    
    kernel_info = metadata.get('kernel_type', 'sliding_window')
    window_str = f"{metadata.get('window_x', 5)}x{metadata.get('window_y', 5)}x{metadata.get('window_z', 5)}"
    
    fig.suptitle(f'ProxylFit – Parameter Maps{slice_title} - {kernel_info.title()} {window_str}{roi_info}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add ProxylFit logo in bottom-right
    add_proxylfit_logo(fig, position='bottom-right')
    
    # Check if we have ROI overlay
    roi_overlay = None
    if 'roi_mask' in param_maps:
        roi_mask_data = param_maps['roi_mask']
        if roi_mask_data.shape == kb_slice.shape:
            roi_overlay = roi_mask_data.T
        else:
            print("Warning: ROI mask shape mismatch with parameter maps")
    
    # KD map (decay rate)
    ax = axes[0, 0]
    kd_masked = np.where(mask_slice, kd_slice, np.nan)
    im1 = ax.imshow(kd_masked, cmap='plasma', origin='lower')
    ax.set_title(f"kd (decay rate) [/{metadata['time_units']}]")
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    
    # Add ROI contour if available
    if roi_overlay is not None:
        ax.contour(roi_overlay, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
    
    plt.colorbar(im1, ax=ax, fraction=0.046)
    
    # KNT map (non-tracer rate)
    ax = axes[0, 1]
    knt_masked = np.where(mask_slice, knt_slice, np.nan)
    im2 = ax.imshow(knt_masked, cmap='magma', origin='lower')
    ax.set_title(f"knt (non-tracer rate) [/{metadata['time_units']}]")
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    
    # Add ROI contour if available
    if roi_overlay is not None:
        ax.contour(roi_overlay, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
    
    plt.colorbar(im2, ax=ax, fraction=0.046)
    
    # R-squared map
    ax = axes[1, 0]
    r2_masked = np.where(mask_slice, r2_slice, np.nan)
    im3 = ax.imshow(r2_masked, cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower')
    ax.set_title('R-squared (fit quality)')
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    
    # Add ROI contour if available
    if roi_overlay is not None:
        ax.contour(roi_overlay, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
    
    plt.colorbar(im3, ax=ax, fraction=0.046)
    
    # Original MRI slice with ROI overlay (replaces ratio map)
    ax = axes[1, 1]
    reference_slice = param_maps.get('reference_slice')
    if reference_slice is not None:
        im4 = ax.imshow(reference_slice.T, cmap='gray', origin='lower')
        ax.set_title('Original MRI slice with ROI')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        # Overlay ROI contour if available
        if roi_overlay is not None:
            ax.contour(roi_overlay, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
        plt.colorbar(im4, ax=ax, fraction=0.046)
    else:
        # Fallback: display mask itself if reference not available
        im4 = ax.imshow(mask_slice, cmap='gray', origin='lower')
        ax.set_title('ROI mask (reference slice unavailable)')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        plt.colorbar(im4, ax=ax, fraction=0.046)
    
    # Add metadata text
    window_text = f"{metadata['window_x']}×{metadata['window_y']}×{metadata['window_z']}"
    metadata_text = (
        f"Window size: {window_text}\n"
        f"Success rate: {metadata['success_rate']:.1f}% ({metadata['successful_fits']}/{metadata['total_positions']})\n"
        f"Processing time: {metadata['processing_time']:.1f}s"
    )
    fig.text(0.02, 0.02, metadata_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plot_file = output_path / "parameter_maps.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Parameter maps saved to: {plot_file}")
    
    plt.show()


def save_parameter_maps(param_maps: Dict[str, np.ndarray], 
                       spacing: Tuple[float, float, float],
                       output_dir: str,
                       dicom_path: Optional[str] = None) -> None:
    """
    Save parameter maps and metadata to disk.
    
    Parameters
    ----------
    param_maps : dict
        Dictionary of parameter maps from create_parameter_maps()
    spacing : tuple
        Voxel spacing (x, y, z)
    output_dir : str
        Output directory
    dicom_path : str, optional
        Original DICOM file path for metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save parameter maps as compressed numpy arrays
    maps_file = output_path / "parameter_maps.npz"
    np.savez_compressed(
        maps_file,
        kb_map=param_maps.get('kb_map'),
        kd_map=param_maps.get('kd_map'),
        knt_map=param_maps.get('knt_map'),
        r_squared_map=param_maps.get('r_squared_map'),
        a1_amplitude_map=param_maps.get('a1_amplitude_map'),
        a2_amplitude_map=param_maps.get('a2_amplitude_map'),
        a0_est_map=param_maps.get('a0_est_map'),
        a2_est_map=param_maps.get('a2_est_map'),
        a1_percent_map=param_maps.get('a1_percent_map'),
        a2_percent_map=param_maps.get('a2_percent_map'),
        a2_percent_est_map=param_maps.get('a2_percent_est_map'),
        baseline_map=param_maps.get('baseline_map'),
        t0_map=param_maps.get('t0_map'),
        tmax_map=param_maps.get('tmax_map'),
        mask=param_maps.get('mask'),
        spacing=np.array(spacing),
        roi_mask=param_maps.get('roi_mask') if 'roi_mask' in param_maps else None
    )
    
    # Save metadata as JSON
    import json
    metadata_file = output_path / "parameter_maps_metadata.json"
    metadata = param_maps['metadata'].copy()
    metadata.update({
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dicom_path': dicom_path,
        'dicom_filename': Path(dicom_path).name if dicom_path else None,
        'spacing': spacing,
        'output_shape': param_maps['kb_map'].shape
    })
    
    # Convert numpy types to native Python for JSON serialization
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_safe)
    
    print(f"Parameter maps saved to: {maps_file}")
    print(f"Metadata saved to: {metadata_file}")


def load_parameter_maps(output_dir: str) -> Tuple[Dict[str, np.ndarray], Tuple[float, float, float]]:
    """
    Load previously saved parameter maps.
    
    Parameters
    ----------
    output_dir : str
        Directory containing saved parameter maps
        
    Returns
    -------
    param_maps : dict
        Dictionary of parameter maps
    spacing : tuple
        Voxel spacing
    """
    output_path = Path(output_dir)
    
    # Load parameter maps
    maps_file = output_path / "parameter_maps.npz"
    if not maps_file.exists():
        raise FileNotFoundError(f"Parameter maps file not found: {maps_file}")
    
    data = np.load(maps_file)
    param_maps = {
        'kb_map': data['kb_map'],
        'kd_map': data['kd_map'],
        'knt_map': data['knt_map'],
        'r_squared_map': data['r_squared_map'],
        'a1_amplitude_map': data['a1_amplitude_map'],
        'a2_amplitude_map': data['a2_amplitude_map'],
        'baseline_map': data['baseline_map'],
        't0_map': data['t0_map'],
        'tmax_map': data['tmax_map'],
        'mask': data['mask'],
    }
    # Older saves may not have the percent maps; compute on the fly so
    # legacy datasets still display them in the new viewer.
    if 'a1_percent_map' in data.files:
        param_maps['a1_percent_map'] = data['a1_percent_map']
        param_maps['a2_percent_map'] = data['a2_percent_map']
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            a0_safe = np.where(param_maps['baseline_map'] > 0,
                               param_maps['baseline_map'], np.nan)
            param_maps['a1_percent_map'] = (
                100.0 * param_maps['a1_amplitude_map'] / a0_safe
            )
            param_maps['a2_percent_map'] = (
                100.0 * param_maps['a2_amplitude_map'] / a0_safe
            )

    # %NTE_est requires the initial-estimate raw maps (a0_est_map, a2_est_map)
    # and the derived percent map. Legacy saves don't have any of these — leave
    # them out of param_maps so the dropdown / metrics gracefully skip them.
    if 'a0_est_map' in data.files:
        param_maps['a0_est_map'] = data['a0_est_map']
        param_maps['a2_est_map'] = data['a2_est_map']
    if 'a2_percent_est_map' in data.files:
        param_maps['a2_percent_est_map'] = data['a2_percent_est_map']
    elif 'a0_est_map' in data.files and 'a2_est_map' in data.files:
        with np.errstate(divide='ignore', invalid='ignore'):
            a0_est_safe = np.where(param_maps['a0_est_map'] > 0,
                                   param_maps['a0_est_map'], np.nan)
            param_maps['a2_percent_est_map'] = (
                100.0 * param_maps['a2_est_map'] / a0_est_safe
            )
    spacing = tuple(data['spacing'])

    # Optionally load ROI mask if present
    if 'roi_mask' in data and data['roi_mask'] is not None:
        loaded_roi = data['roi_mask']
        if loaded_roi.shape != ():  # np.savez stores None as 0-d array
            param_maps['roi_mask'] = loaded_roi
    
    # Load metadata if available
    metadata_file = output_path / "parameter_maps_metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        param_maps['metadata'] = metadata
    
    print(f"Loaded parameter maps with shape: {param_maps['kb_map'].shape}")
    
    return param_maps, spacing


# Progress callback example for use with create_parameter_maps
def print_progress(progress_pct: float, current: int, total: int) -> None:
    """Example progress callback that prints progress."""
    if current % 1000 == 0 or progress_pct >= 100:
        print(f"  Progress: {progress_pct:.1f}% ({current}/{total})")


def select_parameter_mapping_region(registered_4d: np.ndarray, z_index: int = None) -> np.ndarray:
    """
    Interactive selection of contour region for parameter mapping.
    
    Parameters
    ----------
    registered_4d : np.ndarray
        4D array with shape [x, y, z, t]
    z_index : int, optional
        Z-slice index for contour selection. If None, uses middle slice.
        
    Returns
    -------
    roi_mask : np.ndarray
        Boolean mask of shape [x, y] where True indicates pixels inside ROI
    """
    if z_index is None:
        z_index = registered_4d.shape[2] // 2
    
    print(f"Selecting parameter mapping region on slice {z_index}")
    print("Draw a contour around the region where you want to compute parameter maps.")
    print("Controls: Drag to draw, 'c' to close contour, 'r' to reset, then click 'Accept ROI'")
    
    # Use the existing manual contour ROI selector
    selector = ManualContourROISelector(
        registered_4d, 
        z_index, 
        title="Parameter Mapping Region Selection"
    )
    
    roi_mask = selector.show_and_select()
    
    if np.any(roi_mask):
        num_pixels = np.sum(roi_mask)
        print(f"Selected region contains {num_pixels} pixels")
    else:
        print("Warning: No region was selected")
    
    return roi_mask


def select_injection_time_for_mapping(registered_4d: np.ndarray, time_array: np.ndarray, 
                                     roi_mask: np.ndarray, time_units: str = 'minutes',
                                     z_slice: Optional[int] = None) -> int:
    """
    Select injection time based on a representative curve from the selected region.
    
    Parameters
    ----------
    registered_4d : np.ndarray
        4D image data [x, y, z, t]
    time_array : np.ndarray
        Time points array
    roi_mask : np.ndarray
        2D boolean mask defining region of interest
    time_units : str
        Time units for display
    z_slice : int, optional
        If provided, compute representative curve only on this z-slice
        
    Returns
    -------
    int
        Index in time_array where injection occurred
    """
    if not np.any(roi_mask):
        raise ValueError("ROI mask contains no selected pixels")
    
    print("Computing representative curve from selected region...")
    
    # Compute mean signal from the ROI across all timepoints and z-slices
    t_points = registered_4d.shape[3]
    representative_curve = np.zeros(t_points)
    
    if z_slice is not None:
        # Single-slice representative curve
        for t in range(t_points):
            slice_2d = registered_4d[:, :, z_slice, t]
            roi_values = slice_2d[roi_mask]
            if len(roi_values) > 0:
                representative_curve[t] = np.mean(roi_values)
        print(f"Representative curve computed from {np.sum(roi_mask)} pixels on slice z={z_slice}")
    else:
        # Multi-slice representative curve averaged across slices
        for t in range(t_points):
            slice_signals = []
            for z in range(registered_4d.shape[2]):
                slice_2d = registered_4d[:, :, z, t]
                roi_values = slice_2d[roi_mask]
                if len(roi_values) > 0:
                    slice_signals.append(np.mean(roi_values))
            
            if slice_signals:
                representative_curve[t] = np.mean(slice_signals)
        print(f"Representative curve computed from {np.sum(roi_mask)} pixels across {registered_4d.shape[2]} slices")
    print("Please click on the time point when contrast was injected.")
    
    # Use the existing injection time selection interface
    injection_index = select_injection_time(
        time_array, representative_curve, time_units, None
    )
    
    print(f"Selected injection time: {time_array[injection_index]:.2f} {time_units} (index {injection_index})")
    
    return injection_index


def enhanced_parameter_mapping_workflow(registered_4d: np.ndarray, time_array: np.ndarray,
                                       time_units: str = 'minutes', 
                                       z_slice: Optional[int] = None,
                                       kernel_type: str = 'sliding_window',
                                       kernel_size: Union[int, Tuple[int, int, int]] = (5, 5, 1),
                                       interactive: bool = True) -> Dict[str, np.ndarray]:
    """
    Complete enhanced parameter mapping workflow with contour selection and injection time selection.
    
    Parameters
    ----------
    registered_4d : np.ndarray
        Registered 4D data with shape [x, y, z, t]
    time_array : np.ndarray
        Time points for fitting
    time_units : str
        Time units for fitting and display
    z_slice : int, optional
        Z-slice for contour selection. If None, uses middle slice.
    kernel_type : str
        Type of kernel: 'sliding_window', 'gaussian', 'uniform', 'box'
    kernel_size : int
        Size of kernel (will create NxNxN cube)
        
    Returns
    -------
    dict
        Dictionary containing parameter maps and all metadata
    """
    print("="*60)
    print("ENHANCED PARAMETER MAPPING WORKFLOW")
    print("="*60)
    
    # Step 1: Select parameter mapping region
    print("\nStep 1: Select parameter mapping region")
    roi_mask = select_parameter_mapping_region(registered_4d, z_slice)
    
    if not np.any(roi_mask):
        raise ValueError("No region selected for parameter mapping")
    
    # Step 2: Kernel configuration (no prompting; use CLI/defaults)
    print(f"\nStep 2: Kernel configuration")
    # Normalize kernel_size to tuple for display consistency
    if isinstance(kernel_size, int):
        kernel_size_display = (kernel_size, kernel_size, kernel_size)
    else:
        kernel_size_display = kernel_size
    print(f"Using {kernel_type} kernel with size {kernel_size_display}")
    
    # Step 3: Select injection time based on representative curve
    print("\nStep 3: Select injection time from representative curve")
    injection_index = select_injection_time_for_mapping(
        registered_4d, time_array, roi_mask, time_units, z_slice=z_slice
    )
    
    # Step 4: Create parameter maps
    print("\nStep 4: Creating parameter maps within selected ROI")
    
    # Use higher signal threshold and more restrictive fitting for ROI-based mapping
    param_maps = create_parameter_maps(
        registered_4d=registered_4d,
        time_array=time_array,
        window_size=kernel_size,
        z_slice=z_slice,  # Only process the selected slice
        min_signal_threshold=0.15,  # Higher threshold for ROI mapping
        time_units=time_units,
        progress_callback=print_progress,
        roi_mask=roi_mask,
        kernel_type=kernel_type,
        injection_time_index=injection_index
    )
    
    # Print summary of fitting within ROI
    if 'metadata' in param_maps:
        metadata = param_maps['metadata']
        roi_pixels = np.sum(roi_mask)
        print(f"\nROI Parameter Mapping Summary:")
        print(f"  ROI contains: {roi_pixels} pixels")
        print(f"  Positions processed: {metadata['total_positions']}")
        print(f"  Successful fits: {metadata['successful_fits']} ({metadata['success_rate']:.1f}%)")
        print(f"  Kernel type: {metadata['kernel_type']}")
        print(f"  Injection time index: {metadata.get('injection_time_index', 'Not specified')}")
    
    # Add reference MRI slice used during contour selection for visualization
    try:
        if z_slice is not None:
            param_maps['reference_slice'] = registered_4d[:, :, z_slice, 0]
    except Exception:
        pass
    
    return param_maps


def _prompt_kernel_settings(default_type: str = 'sliding_window', default_size: Union[int, Tuple[int, int, int]] = (5, 5, 1), interactive: bool = True) -> Tuple[str, Tuple[int, int, int]]:
    """
    Prompt user to select kernel type and size, with sane defaults.
    
    Parameters
    ----------
    default_type : str
        Default kernel type
    default_size : int or tuple
        Default kernel size; accept single odd integer (cubic) or tuple (wx, wy, wz)
    interactive : bool
        If False, do not prompt and return defaults immediately
    
    Returns
    -------
    (kernel_type, kernel_size) : Tuple[str, int]
    """
    if not interactive:
        # Normalize size to tuple and return
        if isinstance(default_size, int):
            return default_type, (default_size, default_size, default_size)
        return default_type, default_size
    allowed_types = ['sliding_window', 'gaussian', 'uniform', 'box']
    print(f"Kernel types: {', '.join(allowed_types)}")
    try:
        type_input = input(f"Select kernel type [{default_type}]: ").strip().lower()
    except Exception:
        type_input = ''
    kernel_type = type_input if type_input in allowed_types else default_type
    if type_input and type_input not in allowed_types:
        print(f"Unrecognized kernel type '{type_input}', using default '{default_type}'.")
    
    # Normalize default size to tuple for display
    if isinstance(default_size, int):
        default_size_tuple = (default_size, default_size, default_size)
    else:
        default_size_tuple = default_size
    
    try:
        size_input = input(f"Select kernel size (odd N or NxNyxNz) [{default_size_tuple[0]}x{default_size_tuple[1]}x{default_size_tuple[2]}]: ").strip().lower()
    except Exception:
        size_input = ''
    
    def make_odd_positive(n: int) -> int:
        if n < 1:
            return 1
        return n if n % 2 == 1 else n + 1
    
    kernel_size_tuple: Tuple[int, int, int]
    if not size_input:
        kernel_size_tuple = default_size_tuple
    else:
        # Try parse formats: "N" or "NxNyxNz" or "N,N,N"
        parts = None
        try:
            # single integer
            parsed_single = int(size_input)
            n = make_odd_positive(parsed_single)
            kernel_size_tuple = (n, n, n)
        except ValueError:
            # split on x or comma
            if 'x' in size_input:
                parts = size_input.split('x')
            elif ',' in size_input:
                parts = size_input.split(',')
            if parts and len(parts) == 3:
                try:
                    wx = make_odd_positive(int(parts[0]))
                    wy = make_odd_positive(int(parts[1]))
                    wz = make_odd_positive(int(parts[2]))
                    kernel_size_tuple = (wx, wy, wz)
                except ValueError:
                    print("Invalid kernel size components. Using default.")
                    kernel_size_tuple = default_size_tuple
            else:
                print("Invalid kernel size input. Using default.")
                kernel_size_tuple = default_size_tuple
    
    return kernel_type, kernel_size_tuple