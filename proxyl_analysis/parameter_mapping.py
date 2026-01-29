"""
Parameter mapping module for creating spatial maps of kinetic parameters.

This module implements sliding window parameter fitting to create 2D/3D maps
of kb (binding rate) and kd (decay rate) parameters across the entire image.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any, Union
import time
from pathlib import Path
import sys

from .model import fit_proxyl_kinetics, add_proxylfit_logo, set_proxylfit_style, select_injection_time
from .roi_selection import ManualContourROISelector


def create_parameter_maps(registered_4d: np.ndarray, 
                         time_array: np.ndarray,
                         window_size: Union[int, Tuple[int, int, int]] = 5,
                         z_slice: Optional[int] = None,
                         min_signal_threshold: float = 0.1,
                         time_units: str = 'minutes',
                         progress_callback: Optional[callable] = None,
                         roi_mask: Optional[np.ndarray] = None,
                         kernel_type: str = 'sliding_window',
                         injection_time_index: Optional[int] = None) -> Dict[str, np.ndarray]:
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
    
    # Handle injection time selection
    if injection_time_index is not None:
        # Trim time array and image data to start from injection
        time_array = time_array[injection_time_index:]
        registered_4d = registered_4d[:, :, :, injection_time_index:]
        print(f"Using data from injection time onwards: {len(time_array)} timepoints")
    
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
    baseline_map = np.full(output_shape, np.nan)
    t0_map = np.full(output_shape, np.nan)  # Tracer onset time
    tmax_map = np.full(output_shape, np.nan)  # Non-tracer onset time
    fit_mask = np.zeros(output_shape, dtype=bool)
    
    # Calculate signal threshold
    max_signal = np.max(registered_4d)
    signal_threshold = min_signal_threshold * max_signal
    
    # Calculate total positions for progress tracking
    total_positions = 0
    for z in range(z_start, z_end):
        for x in range(x_size):
            for y in range(y_size):
                # Check if pixel is in ROI (if ROI mask is provided)
                if roi_mask is not None and not roi_mask[x, y]:
                    continue
                total_positions += 1
    
    print(f"Creating parameter maps using {window_x}x{window_y}x{window_z} {kernel_type} kernel...")
    if roi_mask is not None:
        print(f"Processing within ROI on {'single slice' if z_slice is not None else 'all slices'}: {total_positions} positions")
    else:
        print(f"Processing {'single slice' if z_slice is not None else 'all slices'}: {total_positions} positions")
    print(f"Signal threshold: {signal_threshold:.2f}")
    
    current_position = 0
    start_time = time.time()
    
    # Process each voxel position
    for z in range(z_start, z_end):
        z_idx = z if z_slice is None else 0  # Index for output arrays
        
        for x in range(x_size):
            for y in range(y_size):
                # Check if pixel is in ROI (if ROI mask is provided)
                if roi_mask is not None and not roi_mask[x, y]:
                    continue
                    
                current_position += 1
                
                # Progress reporting
                if progress_callback and current_position % 100 == 0:
                    progress_pct = 100.0 * current_position / total_positions
                    progress_callback(progress_pct, current_position, total_positions)
                
                # Extract signal using specified kernel type
                if kernel_type == 'sliding_window':
                    window_signal = _extract_sliding_window_signal(
                        registered_4d, x, y, z, (window_x, window_y, window_z)
                    )
                else:
                    window_signal = _extract_kernel_signal(
                        registered_4d, x, y, z, (window_x, window_y, window_z), kernel_type
                    )
                
                # Check if window has sufficient signal and quality
                max_signal = np.max(window_signal)
                min_signal = np.min(window_signal)
                signal_variation = max_signal - min_signal
                
                # Skip if signal too low or no meaningful variation
                if max_signal < signal_threshold or signal_variation < signal_threshold * 0.1:
                    continue
                
                # Skip if signal has too much noise (coefficient of variation too high)
                cv = np.std(window_signal) / np.mean(window_signal) if np.mean(window_signal) > 0 else float('inf')
                if cv > 2.0:  # Skip very noisy signals
                    continue
                
                # Attempt kinetic fitting
                try:
                    kb, kd, knt, fitted_signal, fit_results = fit_proxyl_kinetics(
                        time_array, window_signal, time_units
                    )
                    
                    # Check fit quality (require reasonable R-squared)
                    if fit_results['r_squared'] > 0.1:  # Minimum R² threshold
                        kb_map[x, y, z_idx] = kb
                        kd_map[x, y, z_idx] = kd
                        knt_map[x, y, z_idx] = knt
                        r_squared_map[x, y, z_idx] = fit_results['r_squared']
                        a1_amplitude_map[x, y, z_idx] = fit_results['A1']
                        a2_amplitude_map[x, y, z_idx] = fit_results['A2']
                        baseline_map[x, y, z_idx] = fit_results['A0']
                        t0_map[x, y, z_idx] = fit_results['t0']
                        tmax_map[x, y, z_idx] = fit_results['tmax']
                        fit_mask[x, y, z_idx] = True
                        # Per-voxel success log
                        try:
                            print(f"Fit success at (x={x}, y={y}, z={z_idx}): "
                                  f"kb={kb:.4f}, kd={kd:.4f}, knt={knt:.4f}, R2={fit_results['r_squared']:.3f}")
                        except Exception:
                            # Avoid any logging-related crashes
                            pass
                        
                except Exception:
                    # Fitting failed - leave as NaN
                    continue
    
    elapsed_time = time.time() - start_time
    successful_fits = np.sum(fit_mask)
    success_rate = 100.0 * successful_fits / total_positions
    
    print(f"Parameter mapping completed in {elapsed_time:.1f} seconds")
    print(f"Successful fits: {successful_fits}/{total_positions} ({success_rate:.1f}%)")
    
    result = {
        'kb_map': kb_map,
        'kd_map': kd_map,
        'knt_map': knt_map,
        'r_squared_map': r_squared_map,
        'a1_amplitude_map': a1_amplitude_map,
        'a2_amplitude_map': a2_amplitude_map,
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
            'injection_time_index': injection_time_index
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
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
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
        'r_squared_map': data['r_squared_map'],
        'amplitude_map': data['amplitude_map'],
        'baseline_map': data['baseline_map'],
        'mask': data['mask']
    }
    spacing = tuple(data['spacing'])
    
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