#!/usr/bin/env python3
"""
Command-line entry point for Proxyl MRI analysis.

Usage: python run_analysis.py --dicom path/to/file.dcm --z 4
"""

import argparse
import sys
import os
import numpy as np
import json
from pathlib import Path

# Add the parent directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent))

from proxyl_analysis.io import load_dicom_series
from proxyl_analysis.registration import register_timeseries, load_registration_data, save_registration_data
from proxyl_analysis.roi_selection import select_rectangle_roi, select_segmentation_roi, select_manual_contour_roi, compute_roi_timeseries, print_roi_mode_info, get_available_roi_modes
from proxyl_analysis.model import fit_proxyl_kinetics, plot_fit_results, print_fit_summary, calculate_derived_parameters, select_injection_time
from proxyl_analysis.parameter_mapping import create_parameter_maps, visualize_parameter_maps, save_parameter_maps, print_progress, enhanced_parameter_mapping_workflow

# Import Qt-based UI (modern, responsive layout)
from proxyl_analysis.ui import (
    select_rectangle_roi_qt,
    select_manual_contour_roi_qt,
    select_injection_time_qt,
    plot_fit_results_qt,
    init_qt_app
)


def create_time_array(num_timepoints: int, time_units: str = 'minutes') -> np.ndarray:
    """
    Create time array for the analysis.
    Each timestep is 70 seconds.
    
    Parameters
    ----------
    num_timepoints : int
        Number of time points
    time_units : str
        Units for time ('minutes' or 'seconds')
        
    Returns
    -------
    np.ndarray
        Time array
    """
    if time_units == 'minutes':
        # 70-second intervals converted to minutes
        time_array = np.arange(num_timepoints, dtype=float) * (70.0 / 60.0)
    elif time_units == 'seconds':
        # 70-second intervals
        time_array = np.arange(num_timepoints, dtype=float) * 70.0
    else:
        # Default: just use timepoint indices
        time_array = np.arange(num_timepoints, dtype=float)
    
    return time_array


def main():
    """Main analysis pipeline."""

    # Initialize Qt application for GUI components
    app = init_qt_app()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze time-resolved MRI data after Proxyl injection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dicom', 
        type=str, 
        required=True,
        help='Path to the DICOM file'
    )
    
    parser.add_argument(
        '--z', 
        type=int, 
        default=4,
        help='Z-slice index for ROI selection (0-8 for 9 slices)'
    )
    
    parser.add_argument(
        '--time-units',
        type=str,
        default='minutes',
        choices=['minutes', 'seconds', 'index'],
        help='Units for time axis'
    )
    
    parser.add_argument(
        '--skip-registration',
        action='store_true',
        help='Skip registration step (faster but less accurate)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory to save output plots and results'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting (for batch processing)'
    )
    
    parser.add_argument(
        '--load-registration',
        type=str,
        help='Load previously saved registration data from specified directory'
    )
    
    parser.add_argument(
        '--no-registration-window',
        action='store_true',
        help='Skip registration quality visualization window'
    )
    
    parser.add_argument(
        '--force-registration',
        action='store_true',
        help='Force new registration even if previous data exists'
    )
    
    parser.add_argument(
        '--auto-load',
        action='store_true',
        help='Automatically use existing registration data without prompting (for batch processing)'
    )
    
    parser.add_argument(
        '--create-parameter-maps',
        action='store_true',
        help='Create spatial parameter maps (kb and kd) using sliding window approach'
    )
    
    parser.add_argument(
        '--enhanced-parameter-maps',
        action='store_true',
        help='Enhanced parameter mapping workflow with contour selection, kernel configuration, and injection time selection'
    )
    
    parser.add_argument(
        '--kernel-type',
        type=str,
        default='sliding_window',
        choices=['sliding_window', 'gaussian', 'uniform', 'box'],
        help='Kernel type for enhanced parameter mapping'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Size of cubic sliding window for parameter mapping (NxNxN voxels) - used if individual dimensions not specified'
    )
    
    parser.add_argument(
        '--window-size-x',
        type=int,
        default=15,
        help='X dimension of sliding window (overrides --window-size if specified)'
    )
    
    parser.add_argument(
        '--window-size-y', 
        type=int,
        default=15,
        help='Y dimension of sliding window (overrides --window-size if specified)'
    )
    
    parser.add_argument(
        '--window-size-z',
        type=int,
        default=3,
        help='Z dimension of sliding window (overrides --window-size if specified)'
    )
    
    parser.add_argument(
        '--map-slice',
        type=int,
        help='Z-slice for 2D parameter mapping (if not specified, processes all slices)'
    )
    
    parser.add_argument(
        '--skip-roi-analysis',
        action='store_true',
        help='Skip ROI selection and analysis (useful when only creating parameter maps)'
    )
    
    parser.add_argument(
        '--roi-mode',
        type=str,
        default='contour',
        choices=['rectangle', 'segment', 'contour'],
        help='ROI selection mode: rectangle (bounding box), segment (SegmentAnything), or contour (manual drawing)'
    )
    
    parser.add_argument(
        '--sam-model-path',
        type=str,
        help='Path to SegmentAnything model checkpoint (optional, will auto-download if not provided)'
    )
    
    parser.add_argument(
        '--sam-model-type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SegmentAnything model type (vit_h is most accurate but largest)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dicom):
        print(f"Error: DICOM file not found: {args.dicom}")
        sys.exit(1)
    
    if not (0 <= args.z <= 8):
        print(f"Error: Z-slice index must be between 0-8, got {args.z}")
        sys.exit(1)
    
    # Validate ROI mode
    available_modes = get_available_roi_modes()
    if args.roi_mode not in available_modes:
        print(f"Error: ROI mode '{args.roi_mode}' not available.")
        print_roi_mode_info()
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Auto-detect previous registration data based on DICOM filename
    dicom_name = Path(args.dicom).stem  # Get filename without extension
    auto_registration_dir = output_dir / f"registration_{dicom_name}"
    
    # Check if we should automatically load registration data
    auto_load = False
    if not args.load_registration and not args.force_registration and auto_registration_dir.exists():
        reg_data_file = auto_registration_dir / "registered_4d_data.npz"
        metrics_file = auto_registration_dir / "registration_metrics.json"
        
        if reg_data_file.exists() and metrics_file.exists():
            print("="*60)
            print("FOUND PREVIOUS REGISTRATION DATA")
            print("="*60)
            print(f"Registration data found for: {dicom_name}")
            print(f"Location: {auto_registration_dir}")
            
            # Try to load metadata and show summary
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                metadata = metrics_data.get('metadata', {})
                summary = metrics_data.get('summary', {})
                
                if metadata:
                    print(f"Created: {metadata.get('created_at', 'Unknown')}")
                    print(f"Shape: {metadata.get('shape', 'Unknown')}")
                
                if summary:
                    print(f"Total registration time: {summary.get('total_registration_time', 0):.1f}s")
                    
            except (json.JSONDecodeError, FileNotFoundError):
                print("Could not read registration metadata")
            
            print()
            # Ask user if they want to use it (or auto-load if requested)
            if args.auto_load:
                print("Auto-loading existing registration data...")
                args.load_registration = str(auto_registration_dir)
                auto_load = True
            else:
                use_existing = input("Use existing registration data? [Y/n]: ").strip().lower()
                if use_existing in ['', 'y', 'yes']:
                    args.load_registration = str(auto_registration_dir)
                    auto_load = True
                else:
                    print("Will perform new registration...")
    
    print("="*60)
    print("PROXYL MRI ANALYSIS PIPELINE")
    print("="*60)
    print(f"DICOM file: {args.dicom}")
    print(f"Z-slice for ROI: {args.z}")
    print(f"ROI selection mode: {args.roi_mode}")
    print(f"Time units: {args.time_units}")
    print(f"Output directory: {output_dir}")
    if auto_load:
        print(f"Auto-loading registration from: {auto_registration_dir}")
    print()
    
    try:
        # Step 1: Load DICOM data (or registered data)
        if args.load_registration:
            print("Step 1: Loading previously saved registration data...")
            try:
                registered_4d, spacing, reg_metrics = load_registration_data(args.load_registration)
                print(f"  Loaded registered 4D data with shape: {registered_4d.shape} [x, y, z, t]")
                print(f"  Voxel spacing: {spacing}")
                
                # Show registration quality summary
                num_timepoints = len(reg_metrics) - 1
                print(f"  Registration completed for {num_timepoints} timepoints")
                
                # Also load original data for visualization if needed
                if not (args.no_registration_window or args.no_plot):
                    print("  Loading original DICOM for comparison...")
                    image_4d, current_spacing = load_dicom_series(args.dicom)
                    
                    # Check for spacing mismatch
                    if spacing != current_spacing:
                        print("⚠️  WARNING: Spacing mismatch detected!")
                        print(f"  Cached spacing: {spacing}")
                        print(f"  Current spacing: {current_spacing}")
                        print(f"  Translations may be incorrect by factor of {spacing[0]/current_spacing[0]:.1f}x")
                        print("  Consider using --force-registration to recompute with correct spacing")
                    
                    # Show registration visualization window
                    print("  Showing registration quality visualization...")
                    from proxyl_analysis.registration import visualize_registration_quality
                    visualize_registration_quality(image_4d, registered_4d, reg_metrics)
                else:
                    image_4d = None
                    
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Falling back to loading DICOM and performing registration...")
                args.load_registration = None
        
        if not args.load_registration:
            print("Step 1: Loading DICOM data...")
            image_4d, spacing = load_dicom_series(args.dicom)
            print(f"  Loaded 4D image with shape: {image_4d.shape} [x, y, z, t]")
            print(f"  Voxel spacing: {spacing}")
            print()
            
            # Step 2: Registration
            if args.skip_registration:
                print("Step 2: Skipping registration (as requested)")
                registered_4d = image_4d
                reg_metrics = None
            else:
                print("Step 2: Performing rigid registration...")
                show_reg_window = not args.no_registration_window
                registered_4d, reg_metrics = register_timeseries(
                    image_4d, spacing, 
                    output_dir=str(auto_registration_dir),
                    show_quality_window=show_reg_window,
                    dicom_path=args.dicom
                )
        print()
        
        # Step 3: Enhanced parameter mapping (if requested)
        if args.enhanced_parameter_maps:
            print("Step 3: Enhanced parameter mapping workflow...")
            
            # Create time array for the workflow
            time_array = create_time_array(registered_4d.shape[3], args.time_units)
            
            # Run enhanced workflow with command-line parameters
            # Determine window dimensions for enhanced workflow
            if any([args.window_size_x, args.window_size_y, args.window_size_z]):
                # Use per-dimension sizes; fallback to --window-size where missing
                enh_window_x = args.window_size_x if args.window_size_x is not None else args.window_size
                enh_window_y = args.window_size_y if args.window_size_y is not None else args.window_size
                enh_window_z = args.window_size_z if args.window_size_z is not None else args.window_size
                enh_kernel_size = (enh_window_x, enh_window_y, enh_window_z)
                print(f"  Enhanced mapping: using {enh_window_x}x{enh_window_y}x{enh_window_z} {args.kernel_type} kernel")
            else:
                # Default to NxNx1 for single-slice mapping convenience
                enh_kernel_size = (args.window_size, args.window_size, 1)
                print(f"  Enhanced mapping: using {args.window_size}x{args.window_size}x1 {args.kernel_type} kernel")
            
            param_maps = enhanced_parameter_mapping_workflow(
                registered_4d=registered_4d,
                time_array=time_array,
                time_units=args.time_units,
                z_slice=args.z if hasattr(args, 'z') else None,
                kernel_type=args.kernel_type,
                kernel_size=enh_kernel_size
            )
            
            # Visualize parameter maps
            if not args.no_plot:
                visualize_parameter_maps(param_maps, spacing, str(output_dir))
            
            # Save parameter maps (unique folder per DICOM file)
            dicom_name = Path(args.dicom).stem
            param_map_dir = output_dir / f"enhanced_parameter_maps_{dicom_name}"
            save_parameter_maps(param_maps, spacing, str(param_map_dir), args.dicom)
            print()
            
        # Step 4: Standard parameter mapping (if requested)
        elif args.create_parameter_maps:
            print("Step 3: Creating parameter maps...")
            
            # Determine window dimensions
            if any([args.window_size_x, args.window_size_y, args.window_size_z]):
                # Use individual dimensions (default to --window-size if not specified)
                window_x = args.window_size_x if args.window_size_x is not None else args.window_size
                window_y = args.window_size_y if args.window_size_y is not None else args.window_size
                window_z = args.window_size_z if args.window_size_z is not None else args.window_size
                window_size = (window_x, window_y, window_z)
                print(f"  Using {window_x}x{window_y}x{window_z} sliding window")
            else:
                # Use cubic window
                window_size = args.window_size
                print(f"  Using {args.window_size}x{args.window_size}x{args.window_size} sliding window")
            
            if args.map_slice is not None:
                print(f"  Processing single slice: {args.map_slice}")
            else:
                print(f"  Processing all {registered_4d.shape[2]} slices")
            
            # Create time array for parameter fitting
            time_array = create_time_array(registered_4d.shape[3], args.time_units)
            
            # Create parameter maps
            param_maps = create_parameter_maps(
                registered_4d=registered_4d,
                time_array=time_array,
                window_size=window_size,
                z_slice=args.map_slice,
                time_units=args.time_units,
                progress_callback=print_progress
            )
            
            # Visualize parameter maps
            if not args.no_plot:
                visualize_parameter_maps(param_maps, spacing, str(output_dir))
            
            # Save parameter maps (unique folder per DICOM file)
            dicom_name = Path(args.dicom).stem
            param_map_dir = output_dir / f"parameter_maps_{dicom_name}"
            save_parameter_maps(param_maps, spacing, str(param_map_dir), args.dicom)
            print()
        
        # Step 5: ROI selection (skip if requested)
        if not args.skip_roi_analysis:
            # Determine step number based on what parameter mapping was done
            if args.enhanced_parameter_maps or args.create_parameter_maps:
                step_num = "5"
            else:
                step_num = "3"
            print(f"Step {step_num}: ROI selection...")
            
            if args.roi_mode == 'rectangle':
                print(f"  Please select a rectangular ROI on slice {args.z}")
                print("  Using Qt-based UI with proper layout management.")
                roi_mask = select_rectangle_roi_qt(registered_4d, args.z)

            elif args.roi_mode == 'contour':
                print(f"  Please draw a contour around the ROI on slice {args.z}")
                print("  Using Qt-based UI with proper layout management.")
                roi_mask = select_manual_contour_roi_qt(registered_4d, args.z)
                
            elif args.roi_mode == 'segment':
                print(f"  Please segment the ROI using SegmentAnything on slice {args.z}")
                print("  Click to add points, 't' to toggle positive/negative mode.")
                print("  Press 's' to run segmentation, 'c' to confirm, 'r' to reset.")
                try:
                    roi_mask = select_segmentation_roi(
                        registered_4d, args.z,
                        model_path=args.sam_model_path,
                        model_type=args.sam_model_type
                    )
                except ImportError as e:
                    print(f"Error: {e}")
                    print("Falling back to rectangle selection (Qt UI)...")
                    roi_mask = select_rectangle_roi_qt(registered_4d, args.z)
                except Exception as e:
                    print(f"Segmentation failed: {e}")
                    print("Falling back to rectangle selection (Qt UI)...")
                    roi_mask = select_rectangle_roi_qt(registered_4d, args.z)
            
            if not np.any(roi_mask):
                print("Error: No ROI was selected. Exiting.")
                sys.exit(1)
            
            print(f"  ROI selected with {np.sum(roi_mask)} pixels using {args.roi_mode} mode")
            print()
            
            # Step 6: Extract time series  
            if args.enhanced_parameter_maps or args.create_parameter_maps:
                next_step = "6"
            else:
                next_step = "4"
            print(f"Step {next_step}: Computing ROI time series...")
            signal_timeseries = compute_roi_timeseries(registered_4d, roi_mask)
            
            # Create time array
            time_array = create_time_array(len(signal_timeseries), args.time_units)
            
            print(f"  Extracted {len(signal_timeseries)} time points")
            print(f"  Signal range: {np.min(signal_timeseries):.2f} to {np.max(signal_timeseries):.2f}")
            print()
            
            # Step 7: Select injection time
            if args.enhanced_parameter_maps or args.create_parameter_maps:
                next_step = "7"
            else:
                next_step = "5"
            print(f"Step {next_step}: Selecting injection time...")
            print("  Please click on the time point when contrast was injected.")
            print("  Using Qt-based UI with proper layout management.")
            injection_index = select_injection_time_qt(time_array, signal_timeseries, args.time_units, str(output_dir))
            
            # Trim data to start from injection time
            time_array_fit = time_array[injection_index:]
            signal_timeseries_fit = signal_timeseries[injection_index:]
            
            print(f"  Injection time: {time_array[injection_index]:.1f} {args.time_units}")
            print(f"  Fitting data from injection onwards: {len(signal_timeseries_fit)} points")
            print()
            
            # Step 8: Fit kinetic model
            if args.enhanced_parameter_maps or args.create_parameter_maps:
                next_step = "8"
            else:
                next_step = "6"
            print(f"Step {next_step}: Fitting kinetic model...")
            try:
                kb, kd, knt, fitted_signal, fit_results = fit_proxyl_kinetics(
                    time_array_fit, signal_timeseries_fit, args.time_units
                )
                
                # Print results
                print_fit_summary(fit_results)
                
                # Calculate derived parameters
                derived_params = calculate_derived_parameters(
                    kb, kd, knt, fit_results['kb_error'], fit_results['kd_error'], fit_results['knt_error']
                )
                
                print("\nDerived Parameters:")
                print(f"  Tracer half-life (buildup):  {derived_params['half_life_buildup']:.2f} ± {derived_params['half_life_buildup_error']:.2f} {args.time_units}")
                print(f"  Tracer half-life (decay):    {derived_params['half_life_decay']:.2f} ± {derived_params['half_life_decay_error']:.2f} {args.time_units}")
                print(f"  Non-tracer half-life:   {derived_params['half_life_nontracer']:.2f} ± {derived_params['half_life_nontracer_error']:.2f} {args.time_units}")
                print(f"  Rate ratio (kb/kd):     {derived_params['rate_ratio_buildup_decay']:.2f} ± {derived_params['rate_ratio_buildup_decay_error']:.2f}")
                print(f"  Rate ratio (kb/knt):    {derived_params['rate_ratio_buildup_nontracer']:.2f} ± {derived_params['rate_ratio_buildup_nontracer_error']:.2f}")
                
            except Exception as e:
                print(f"Error in kinetic fitting: {e}")
                sys.exit(1)
            
            print()
            
            # Step 9: Save results and plots
            if args.enhanced_parameter_maps or args.create_parameter_maps:
                next_step = "9"
            else:
                next_step = "7"
            print(f"Step {next_step}: Saving results...")
            
            # Save numerical results
            results_file = output_dir / "kinetic_results.txt"
            with open(results_file, 'w') as f:
                f.write("EXTENDED PROXYL KINETIC ANALYSIS RESULTS\n")
                f.write("="*40 + "\n\n")
                f.write(f"DICOM file: {args.dicom}\n")
                f.write(f"Z-slice: {args.z}\n")
                f.write(f"ROI mode: {args.roi_mode}\n")
                f.write(f"ROI pixels: {np.sum(roi_mask)}\n")
                f.write(f"Injection time: {time_array[injection_index]:.1f} {args.time_units} (index {injection_index})\n")
                f.write(f"Time units: {args.time_units}\n\n")
                
                f.write("FITTED PARAMETERS:\n")
                f.write(f"A0 (baseline):           {fit_results['A0']:.3f} ± {fit_results['A0_error']:.3f}\n")
                f.write(f"A1 (tracer amplitude):   {fit_results['A1']:.3f} ± {fit_results['A1_error']:.3f}\n")
                f.write(f"A2 (non-tracer ampl.):   {fit_results['A2']:.3f} ± {fit_results['A2_error']:.3f}\n")
                f.write(f"kb (buildup rate):       {kb:.4f} ± {fit_results['kb_error']:.4f} /{args.time_units}\n")
                f.write(f"kd (decay rate):         {kd:.4f} ± {fit_results['kd_error']:.4f} /{args.time_units}\n")
                f.write(f"knt (non-tracer rate):   {knt:.4f} ± {fit_results['knt_error']:.4f} /{args.time_units}\n")
                f.write(f"t0 (tracer onset):       {fit_results['t0']:.2f} ± {fit_results['t0_error']:.2f} {args.time_units}\n")
                f.write(f"tmax (non-tracer onset): {fit_results['tmax']:.2f} ± {fit_results['tmax_error']:.2f} {args.time_units}\n\n")
                
                f.write("FIT QUALITY:\n")
                f.write(f"R-squared: {fit_results['r_squared']:.4f}\n")
                f.write(f"RMSE:      {fit_results['rmse']:.3f}\n\n")
                
                f.write("DERIVED PARAMETERS:\n")
                f.write(f"Tracer half-life (buildup):  {derived_params['half_life_buildup']:.2f} ± {derived_params['half_life_buildup_error']:.2f} {args.time_units}\n")
                f.write(f"Tracer half-life (decay):    {derived_params['half_life_decay']:.2f} ± {derived_params['half_life_decay_error']:.2f} {args.time_units}\n")
                f.write(f"Non-tracer half-life:   {derived_params['half_life_nontracer']:.2f} ± {derived_params['half_life_nontracer_error']:.2f} {args.time_units}\n")
                f.write(f"Rate ratio (kb/kd):     {derived_params['rate_ratio_buildup_decay']:.2f} ± {derived_params['rate_ratio_buildup_decay_error']:.2f}\n")
                f.write(f"Rate ratio (kb/knt):    {derived_params['rate_ratio_buildup_nontracer']:.2f} ± {derived_params['rate_ratio_buildup_nontracer_error']:.2f}\n")
                
                # Add registration information if available
                if reg_metrics:
                    f.write("\n")
                    f.write("REGISTRATION QUALITY:\n")
                    
                    # Detailed translation analysis
                    translations = [m.translation for m in reg_metrics[1:]]
                    trans_x = [t[0] for t in translations]
                    trans_y = [t[1] for t in translations] 
                    trans_z = [t[2] for t in translations]
                    trans_mag = [np.linalg.norm(t) for t in translations]
                    
                    # Other metrics
                    mean_mi = np.mean([m.mutual_info for m in reg_metrics[1:]])
                    std_mi = np.std([m.mutual_info for m in reg_metrics[1:]])
                    mean_mse = np.mean([m.mean_squared_error for m in reg_metrics[1:]])
                    std_mse = np.std([m.mean_squared_error for m in reg_metrics[1:]])
                    mean_rotation = np.mean([np.linalg.norm(m.rotation) * 180 / np.pi for m in reg_metrics[1:]])
                    total_reg_time = sum([m.registration_time for m in reg_metrics[1:]])
                    
                    f.write(f"Mutual Information:          {mean_mi:.2f} ± {std_mi:.2f}\n")
                    f.write(f"Mean Squared Error:          {mean_mse:.2f} ± {std_mse:.2f}\n")
                    f.write(f"Translation (mm):\n")
                    f.write(f"  X-direction: {np.mean(trans_x):.2f} ± {np.std(trans_x):.2f}\n")
                    f.write(f"  Y-direction: {np.mean(trans_y):.2f} ± {np.std(trans_y):.2f}\n") 
                    f.write(f"  Z-direction: {np.mean(trans_z):.2f} ± {np.std(trans_z):.2f}\n")
                    f.write(f"  Magnitude:   {np.mean(trans_mag):.2f} ± {np.std(trans_mag):.2f}\n")
                    f.write(f"Mean rotation:               {mean_rotation:.2f} degrees\n")
                    f.write(f"Total reg. time:             {total_reg_time:.1f} seconds\n")
                    f.write(f"Number of timepoints registered: {len(reg_metrics)-1}\n")
            
            print(f"  Results saved to: {results_file}")
            
            # Save raw data
            data_file = output_dir / "timeseries_data.npz"
            np.savez(
                data_file,
                time_full=time_array,
                signal_full=signal_timeseries,
                time_fit=time_array_fit,
                signal_fit=signal_timeseries_fit,
                fitted_signal=fitted_signal,
                roi_mask=roi_mask,
                injection_index=injection_index,
                injection_time=time_array[injection_index]
            )
            print(f"  Raw data saved to: {data_file}")
            
            # Create and save plot
            if not args.no_plot:
                plot_file = output_dir / "kinetic_fit.png"
                plot_fit_results_qt(time_array_fit, signal_timeseries_fit, fitted_signal,
                                   fit_results, str(plot_file))
        
        print("\nAnalysis completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()