#!/usr/bin/env python3
"""
Command-line entry point for Proxyl MRI analysis.

Usage: python run_analysis.py --dicom path/to/file.dcm --z 4
"""

import argparse
import sys
import os
import time
import numpy as np
import json
from pathlib import Path

# Add the parent directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent))

from proxyl_analysis.io import load_dicom_series, load_t2_volume
from proxyl_analysis.registration import register_timeseries, load_registration_data, save_registration_data, register_t2_to_t1
from proxyl_analysis.roi_selection import select_rectangle_roi, select_segmentation_roi, select_manual_contour_roi, compute_roi_timeseries, print_roi_mode_info, get_available_roi_modes
from proxyl_analysis.model import fit_proxyl_kinetics, plot_fit_results, print_fit_summary, calculate_derived_parameters, select_injection_time
from proxyl_analysis.parameter_mapping import create_parameter_maps, visualize_parameter_maps, save_parameter_maps, print_progress, enhanced_parameter_mapping_workflow

# Import Qt-based UI (modern, responsive layout)
from proxyl_analysis.ui import (
    select_rectangle_roi_qt,
    select_manual_contour_roi_qt,
    select_injection_time_qt,
    plot_fit_results_qt,
    init_qt_app,
    show_main_menu,
    run_registration_with_progress,
    show_registration_review_qt,
    show_image_tools_dialog,
    show_parameter_map_options,
    show_parameter_map_results,
    ParameterMappingProgressDialog
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
        required=False,
        help='Path to the T1 DICOM file (optional if using menu to load)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: skip main menu and use command-line arguments directly'
    )

    parser.add_argument(
        '--t2',
        type=str,
        help='Path to T2 DICOM file for ROI selection (optional). T2 provides better tumor definition for RANO criteria.'
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
    if args.batch and not args.dicom:
        print("Error: --dicom is required when using --batch mode")
        sys.exit(1)

    if args.dicom and not os.path.exists(args.dicom):
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

    # If no DICOM provided and not in batch mode, show menu to load
    if not args.dicom and not args.batch:
        print("="*60)
        print("PROXYL MRI ANALYSIS - INTERACTIVE MODE")
        print("="*60)
        print("No DICOM file specified. Opening main menu...")
        print()

        # Show menu with no data loaded
        result = show_main_menu(
            registered_4d=None,
            spacing=None,
            time_array=None,
            dicom_path="",
            output_dir=args.output_dir,
            registered_t2=None
        )

        if result is None or result.get('action') == 'exit':
            print("Exiting.")
            sys.exit(0)

        # Handle menu result
        if result.get('action') == 'load_new':
            args.dicom = result['dicom_path']
        elif result.get('action') == 'load_previous':
            args.load_registration = result['session_path']
            # Try to find original DICOM path from metadata
            metrics_file = Path(result['session_path']) / "registration_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metadata = json.load(f).get('metadata', {})
                    args.dicom = metadata.get('dicom_path', '')
        else:
            print(f"Unexpected menu action: {result.get('action')}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Dataset directory structure (T011)
    # New structure: output/{dicom_name}/registered/...
    # Legacy structure: output/registration_{dicom_name}/...
    dicom_name = Path(args.dicom).stem  # Get filename without extension

    # Check for new structure first, then legacy
    dataset_dir = output_dir / dicom_name
    legacy_dir = output_dir / f"registration_{dicom_name}"

    # Determine which directory to use
    if dataset_dir.exists() and (dataset_dir / "registered").exists():
        auto_registration_dir = dataset_dir
    elif legacy_dir.exists():
        auto_registration_dir = legacy_dir
    else:
        # New dataset - use new structure
        auto_registration_dir = dataset_dir

    # Check if we should automatically load registration data
    auto_load = False

    def _find_registration_data(dir_path):
        """Check if registration data exists in directory (2D DICOM slice format)."""
        p = Path(dir_path)
        dicom_dir = p / "registered" / "dicoms"
        # Check for 2D slice format: z00_t00.dcm
        if dicom_dir.exists() and (dicom_dir / "z00_t00.dcm").exists():
            return p / "registered" / "registration_metrics.json"
        return None

    if not args.load_registration and not args.force_registration:
        metrics_file = _find_registration_data(auto_registration_dir)

        if metrics_file and metrics_file.exists():
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
                # Show Qt dialog instead of command line prompt
                from PySide6.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setWindowTitle("Previous Registration Found")
                msg.setText(f"Registration data found for:\n{dicom_name}")
                msg.setInformativeText("Would you like to use the existing registration data?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                result = msg.exec()

                if result == QMessageBox.Yes:
                    print("Using existing registration data...")
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
                    show_registration_review_qt(image_4d, registered_4d, reg_metrics, output_dir=args.load_registration)
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

                # Create dataset directory with full structure (T011)
                from proxyl_analysis.io import create_dataset_directory, save_dataset_manifest
                auto_registration_dir = create_dataset_directory(str(output_dir), dicom_name)

                # Initialize manifest with source info
                manifest = {
                    "dataset_name": dicom_name,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "source": {
                        "dicom_path": str(args.dicom),
                        "dicom_filename": Path(args.dicom).name,
                        "shape": list(image_4d.shape),
                        "spacing": list(spacing)
                    },
                    "analysis": {
                        "registration": {"completed": False},
                        "roi_analysis": {"completed": False},
                        "parameter_maps": {"completed": False},
                        "derived_images": {"averaged_images": [], "difference_images": []}
                    }
                }
                save_dataset_manifest(str(auto_registration_dir), manifest)

                if args.batch:
                    # Batch mode: use direct registration (no progress dialog)
                    registered_4d, reg_metrics = register_timeseries(
                        image_4d, spacing,
                        output_dir=str(auto_registration_dir),
                        show_quality_window=show_reg_window,
                        dicom_path=args.dicom
                    )
                else:
                    # Interactive mode: use progress dialog
                    registered_4d, reg_metrics = run_registration_with_progress(
                        image_4d, spacing,
                        output_dir=str(auto_registration_dir),
                        dicom_path=args.dicom
                    )
                    if registered_4d is None:
                        print("Registration cancelled.")
                        sys.exit(0)
                    # Show quality window after progress dialog closes
                    if show_reg_window:
                        show_registration_review_qt(image_4d, registered_4d, reg_metrics, output_dir=str(auto_registration_dir))
        print()

        # Show main menu (if not in batch mode)
        if not args.batch:
            # Create time array for menu
            time_array = create_time_array(registered_4d.shape[3], args.time_units)

            # ROI state preserved across menu returns
            roi_state = None
            registered_t2 = None

            # Menu loop - keep showing menu until user exits
            while True:
                # Show menu with loaded data and ROI state
                menu_result = show_main_menu(
                    registered_4d=registered_4d,
                    spacing=spacing,
                    time_array=time_array,
                    dicom_path=args.dicom,
                    output_dir=str(auto_registration_dir),
                    registered_t2=registered_t2,
                    roi_state=roi_state
                )

                if menu_result is None or menu_result.get('action') == 'exit':
                    print("Exiting.")
                    sys.exit(0)

                # Handle menu actions
                action = menu_result.get('action')

                if action == 'load_new':
                    # User wants to load different data - restart
                    print(f"Loading new DICOM: {menu_result['dicom_path']}")
                    print("Please restart the application with the new file.")
                    sys.exit(0)

                elif action == 'load_previous':
                    # User wants to load different session - restart
                    print(f"Loading session from: {menu_result['session_path']}")
                    print("Please restart the application with --load-registration flag.")
                    sys.exit(0)

                elif action == 'load_t2':
                    # Load and register T2
                    print(f"Loading T2 from: {menu_result['t2_path']}")
                    try:
                        from proxyl_analysis.io import load_t2_volume
                        t2_volume, t2_spacing = load_t2_volume(menu_result['t2_path'])
                        print(f"  T2 volume shape: {t2_volume.shape}")
                        t1_reference = registered_4d[:, :, :, 0]
                        registered_t2, reg_info = register_t2_to_t1(
                            t2_volume=t2_volume,
                            t2_spacing=t2_spacing,
                            t1_reference=t1_reference,
                            t1_spacing=spacing,
                            show_quality=True
                        )
                        print(f"  T2-T1 registration complete")
                    except Exception as e:
                        print(f"  Error loading T2: {e}")
                        registered_t2 = None
                    continue  # Return to menu with T2 loaded

                elif action == 'draw_roi':
                    # Draw ROI -> Extract time series -> Select injection time -> Return to menu
                    args.roi_mode = menu_result['roi_mode']
                    args.z = menu_result['z_slice']

                    # Choose image for ROI selection: T2 (if registered) or T1
                    if registered_t2 is not None and menu_result.get('roi_source') == 't2':
                        roi_selection_image = registered_t2[:, :, :, np.newaxis]
                        print("  Using registered T2 for ROI selection")
                    else:
                        roi_selection_image = registered_4d
                        print("  Using T1 for ROI selection")

                    # Do ROI selection
                    print(f"ROI selection ({args.roi_mode}) on slice {args.z}...")
                    if args.roi_mode == 'rectangle':
                        roi_mask = select_rectangle_roi_qt(roi_selection_image, args.z)
                    elif args.roi_mode == 'contour':
                        roi_mask = select_manual_contour_roi_qt(roi_selection_image, args.z)
                    elif args.roi_mode == 'segment':
                        try:
                            roi_mask = select_segmentation_roi(
                                roi_selection_image, args.z,
                                model_path=args.sam_model_path,
                                model_type=args.sam_model_type
                            )
                        except Exception as e:
                            print(f"Segmentation failed: {e}, falling back to contour")
                            roi_mask = select_manual_contour_roi_qt(roi_selection_image, args.z)

                    if not np.any(roi_mask):
                        print("No ROI was selected.")
                        continue  # Return to menu

                    print(f"  ROI selected with {np.sum(roi_mask)} pixels")

                    # Extract time series from T1 (always use T1 for signal)
                    roi_signal = compute_roi_timeseries(registered_4d, roi_mask)
                    print(f"  Extracted {len(roi_signal)} time points")

                    # Select injection time
                    print("Select injection time point...")
                    injection_idx = select_injection_time_qt(time_array, roi_signal, args.time_units, str(auto_registration_dir))
                    injection_time = time_array[injection_idx]
                    print(f"  Injection time: {injection_time:.1f} {args.time_units} (index {injection_idx})")

                    # Store ROI state
                    roi_state = {
                        'roi_mask': roi_mask,
                        'roi_signal': roi_signal,
                        'injection_idx': injection_idx,
                        'injection_time': injection_time
                    }
                    print("ROI and injection time set. Returning to menu.")
                    continue  # Return to menu with ROI state

                elif action == 'kinetic_fit':
                    # Run kinetic fitting on existing ROI data
                    if roi_state is None:
                        print("Error: No ROI data. Draw ROI first.")
                        continue

                    roi_mask = roi_state['roi_mask']
                    roi_signal = roi_state['roi_signal']
                    injection_idx = roi_state['injection_idx']

                    # Trim data to start from injection time
                    time_array_fit = time_array[injection_idx:]
                    signal_fit = roi_signal[injection_idx:]

                    print(f"Fitting kinetic model ({len(signal_fit)} points from injection)...")
                    try:
                        kb, kd, knt, fitted_signal, fit_results = fit_proxyl_kinetics(
                            time_array_fit, signal_fit, args.time_units
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

                        # Show fit plot (Qt-based)
                        plot_file = auto_registration_dir / "kinetic_fit.png"
                        plot_fit_results_qt(time_array_fit, signal_fit, fitted_signal,
                                           fit_results, str(plot_file))

                        # Save results
                        results_file = auto_registration_dir / "kinetic_results.txt"
                        with open(results_file, 'w') as f:
                            f.write("EXTENDED PROXYL KINETIC ANALYSIS RESULTS\n")
                            f.write("="*40 + "\n\n")
                            f.write(f"DICOM file: {args.dicom}\n")
                            f.write(f"ROI pixels: {np.sum(roi_mask)}\n")
                            f.write(f"Injection time: {time_array[injection_idx]:.1f} {args.time_units} (index {injection_idx})\n\n")
                            f.write("FITTED PARAMETERS:\n")
                            f.write(f"kb (buildup rate):       {kb:.4f} ± {fit_results['kb_error']:.4f} /{args.time_units}\n")
                            f.write(f"kd (decay rate):         {kd:.4f} ± {fit_results['kd_error']:.4f} /{args.time_units}\n")
                            f.write(f"knt (non-tracer rate):   {knt:.4f} ± {fit_results['knt_error']:.4f} /{args.time_units}\n")
                            f.write(f"\nFIT QUALITY:\n")
                            f.write(f"R-squared: {fit_results['r_squared']:.4f}\n")
                            f.write(f"RMSE:      {fit_results['rmse']:.3f}\n")

                        print(f"\nResults saved to: {results_file}")

                    except Exception as e:
                        print(f"Error in kinetic fitting: {e}")
                        import traceback
                        traceback.print_exc()

                    print("Returning to menu.")
                    continue  # Return to menu

                elif action == 'parameter_maps':
                    # Show parameter map options dialog (T014)
                    print("Opening parameter map options...")

                    options = show_parameter_map_options(
                        max_z=registered_4d.shape[2] - 1,
                        current_z=roi_state.get('z_slice', 4) if roi_state else 4,
                        existing_roi=roi_state.get('roi_mask') if roi_state else None,
                        existing_injection_idx=roi_state.get('injection_idx') if roi_state else None,
                        default_window_size=menu_result.get('window_size', (15, 15, 3))
                    )

                    if options is None:
                        print("Parameter mapping cancelled.")
                        continue

                    # Determine ROI mask to use
                    param_roi_mask = None
                    if options['roi_only']:
                        if options['reuse_roi'] and roi_state is not None:
                            param_roi_mask = roi_state['roi_mask']
                            print(f"Reusing existing ROI ({np.sum(param_roi_mask)} pixels)")
                        elif options['redraw_roi']:
                            # Draw new ROI for parameter mapping
                            z_for_roi = options['z_slice'] if options['single_slice'] else registered_4d.shape[2] // 2
                            print(f"Drawing new ROI on slice {z_for_roi}...")
                            param_roi_mask = select_manual_contour_roi_qt(registered_4d, z_for_roi)
                            if not np.any(param_roi_mask):
                                print("No ROI was drawn. Returning to menu.")
                                continue

                    # Determine injection time
                    injection_idx = None
                    if options['reuse_injection'] and roi_state is not None:
                        injection_idx = roi_state['injection_idx']
                        print(f"Reusing injection time index: {injection_idx}")
                    elif options['select_injection']:
                        # Need to get representative curve
                        if param_roi_mask is not None:
                            # Use ROI mean signal (compute_roi_timeseries imported at top of file)
                            rep_signal = compute_roi_timeseries(registered_4d, param_roi_mask)
                        elif roi_state is not None and roi_state.get('roi_signal') is not None:
                            rep_signal = roi_state['roi_signal']
                        else:
                            # Use center region
                            cx, cy = registered_4d.shape[0] // 2, registered_4d.shape[1] // 2
                            cz = options['z_slice'] if options['single_slice'] else registered_4d.shape[2] // 2
                            rep_signal = registered_4d[cx-5:cx+5, cy-5:cy+5, cz, :].mean(axis=(0, 1))

                        print("Select injection time...")
                        injection_idx = select_injection_time_qt(
                            time_array, rep_signal, args.time_units, str(auto_registration_dir)
                        )
                        print(f"Selected injection time index: {injection_idx}")

                    # Run parameter mapping with progress dialog
                    print("Creating parameter maps...")
                    progress_dialog = ParameterMappingProgressDialog(
                        registered_4d=registered_4d,
                        time_array=time_array,
                        options=options,
                        roi_mask=param_roi_mask,
                        injection_idx=injection_idx,
                        time_units=args.time_units
                    )

                    result = progress_dialog.exec()

                    if result != 1 or progress_dialog.param_maps is None:
                        print("Parameter mapping cancelled or failed.")
                        continue

                    param_maps = progress_dialog.param_maps

                    # Add reference slice for visualization/overlay
                    if options['single_slice']:
                        param_maps['reference_slice'] = registered_4d[:, :, options['z_slice'], 0]
                    else:
                        # Store entire 3D reference for z-slice navigation
                        param_maps['reference_slice'] = registered_4d[:, :, :, 0]

                    # Show results viewer (T014)
                    print("Displaying parameter map results...")
                    show_parameter_map_results(
                        param_maps=param_maps,
                        spacing=spacing,
                        roi_mask=param_roi_mask,
                        output_dir=str(auto_registration_dir),
                        source_dicom=args.dicom
                    )

                    # Save parameter maps
                    param_map_dir = auto_registration_dir / "parameter_maps"
                    save_parameter_maps(param_maps, spacing, str(param_map_dir), args.dicom)

                    print("Returning to menu.")
                    continue  # Return to menu

                elif action == 'image_tools':
                    # Launch Image Tools dialog (T002/T003/T012/T013)
                    if roi_state is None:
                        print("Error: No ROI data. Draw ROI first.")
                        continue

                    mode = menu_result.get('mode', 'average')
                    print(f"Opening Image Tools ({mode} mode)...")

                    show_image_tools_dialog(
                        image_4d=registered_4d,
                        time_array=time_array,
                        roi_signal=roi_state['roi_signal'],
                        time_units=args.time_units,
                        output_dir=str(auto_registration_dir),
                        initial_mode=mode,
                        roi_mask=roi_state.get('roi_mask'),
                        spacing=spacing,
                        source_dicom=args.dicom
                    )

                    print("Returning to menu.")
                    continue  # Return to menu

                elif action == 'export':
                    export_type = menu_result.get('export_type')
                    if export_type == 'registered_data':
                        print(f"Registered data saved to: {auto_registration_dir}")
                    elif export_type == 'registration_report':
                        print(f"Registration metrics saved to: {auto_registration_dir / 'registration_metrics.json'}")
                    elif export_type == 'timeseries' and roi_state is not None:
                        # Export time series CSV
                        csv_file = auto_registration_dir / "roi_timeseries.csv"
                        import csv
                        with open(csv_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['time', 'signal'])
                            for t, s in zip(time_array, roi_state['roi_signal']):
                                writer.writerow([t, s])
                        print(f"Time series CSV saved to: {csv_file}")
                    print("Export complete.")
                    continue  # Return to menu

                else:
                    # Unknown action, show menu again
                    continue

            # If we break out of the menu loop, skip the rest and do parameter maps
            args.skip_roi_analysis = True

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

        # T2-T1 Registration (if T2 provided via command line in batch mode)
        # T2 images provide better tumor volume definition per RANO criteria
        # In interactive mode, T2 is handled inside the menu loop
        if not args.batch:
            registered_t2 = None  # Already handled in menu loop, just ensure it's defined
        elif args.t2:
            print("Loading and registering T2 volume...")
            try:
                t2_volume, t2_spacing = load_t2_volume(args.t2)
                print(f"  T2 volume shape: {t2_volume.shape}")
                print(f"  T2 spacing: {t2_spacing}")

                # Use first timepoint of T1 as reference
                t1_reference = registered_4d[:, :, :, 0]

                # Register T2 to T1
                registered_t2, reg_info = register_t2_to_t1(
                    t2_volume=t2_volume,
                    t2_spacing=t2_spacing,
                    t1_reference=t1_reference,
                    t1_spacing=spacing,
                    show_quality=not args.no_plot
                )

                print(f"  T2-T1 registration complete")
                print(f"  Registered T2 shape: {registered_t2.shape}")
                print(f"  Using registered T2 for ROI selection (better tumor definition)")
                print()
            except FileNotFoundError:
                print(f"  Error: T2 file not found: {args.t2}")
                print(f"  Falling back to T1 for ROI selection")
                registered_t2 = None
            except Exception as e:
                print(f"  Error registering T2: {e}")
                print(f"  Falling back to T1 for ROI selection")
                registered_t2 = None
        else:
            # Batch mode without T2
            registered_t2 = None

        # Step 5: ROI selection (skip if requested)
        if not args.skip_roi_analysis:
            # Determine step number based on what parameter mapping was done
            if args.enhanced_parameter_maps or args.create_parameter_maps:
                step_num = "5"
            else:
                step_num = "3"
            print(f"Step {step_num}: ROI selection...")

            # Choose image for ROI selection: T2 (if registered) or T1
            if registered_t2 is not None:
                # Create 4D volume from T2 for ROI selection UI (add time dimension)
                roi_selection_image = registered_t2[:, :, :, np.newaxis]
                print("  Using registered T2 for ROI selection (better tumor visualization)")
            else:
                roi_selection_image = registered_4d
                print("  Using T1 for ROI selection")

            if args.roi_mode == 'rectangle':
                print(f"  Please select a rectangular ROI on slice {args.z}")
                print("  Using Qt-based UI with proper layout management.")
                roi_mask = select_rectangle_roi_qt(roi_selection_image, args.z)

            elif args.roi_mode == 'contour':
                print(f"  Please draw a contour around the ROI on slice {args.z}")
                print("  Using Qt-based UI with proper layout management.")
                roi_mask = select_manual_contour_roi_qt(roi_selection_image, args.z)

            elif args.roi_mode == 'segment':
                print(f"  Please segment the ROI using SegmentAnything on slice {args.z}")
                print("  Click to add points, 't' to toggle positive/negative mode.")
                print("  Press 's' to run segmentation, 'c' to confirm, 'r' to reset.")
                try:
                    roi_mask = select_segmentation_roi(
                        roi_selection_image, args.z,
                        model_path=args.sam_model_path,
                        model_type=args.sam_model_type
                    )
                except ImportError as e:
                    print(f"Error: {e}")
                    print("Falling back to rectangle selection (Qt UI)...")
                    roi_mask = select_rectangle_roi_qt(roi_selection_image, args.z)
                except Exception as e:
                    print(f"Segmentation failed: {e}")
                    print("Falling back to rectangle selection (Qt UI)...")
                    roi_mask = select_rectangle_roi_qt(roi_selection_image, args.z)
            
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