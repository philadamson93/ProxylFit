"""
Registration module for rigid registration of time-resolved MRI volumes.
Enhanced with save/load functionality, quality metrics, and visualization.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from pathlib import Path
import json
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict
from .model import add_proxylfit_logo, set_proxylfit_style


@dataclass
class RegistrationMetrics:
    """Container for registration quality metrics."""
    timepoint: int
    ncc: float  # Normalized cross-correlation
    mutual_info: float  # Mutual information
    mean_squared_error: float
    translation: Tuple[float, float, float]  # Translation parameters
    rotation: Tuple[float, float, float]  # Rotation parameters (in radians)
    optimizer_iterations: int
    optimizer_metric_value: float
    registration_time: float


def register_timeseries(image_4d: np.ndarray, spacing: Tuple[float, float, float],
                       output_dir: Optional[str] = None, 
                       show_quality_window: bool = True,
                       dicom_path: Optional[str] = None) -> Tuple[np.ndarray, List[RegistrationMetrics]]:
    """
    Register each timepoint volume to the first timepoint using rigid registration.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    spacing : tuple of float
        Voxel spacing (x, y, z) from DICOM metadata
    output_dir : str, optional
        Directory to save registration data and metrics
    show_quality_window : bool
        Whether to show the registration quality visualization window
        
    Returns
    -------
    registered_4d : np.ndarray
        Registered 4D array with same shape [x, y, z, t]
    metrics : List[RegistrationMetrics]
        Registration quality metrics for each timepoint
    """
    x, y, z, t = image_4d.shape
    registered_4d = np.zeros_like(image_4d)
    metrics = []
    
    # Convert reference volume (t=0) to SimpleITK image
    reference_volume = image_4d[:, :, :, 0]
    reference_image = _numpy_to_sitk(reference_volume, spacing)
    
    # Keep reference timepoint unchanged
    registered_4d[:, :, :, 0] = reference_volume
    
    # Add metrics for reference timepoint (perfect registration)
    ref_metrics = RegistrationMetrics(
        timepoint=0,
        ncc=1.0,
        mutual_info=compute_mutual_information(reference_image, reference_image),
        mean_squared_error=0.0,
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        optimizer_iterations=0,
        optimizer_metric_value=0.0,
        registration_time=0.0
    )
    metrics.append(ref_metrics)
    
    print(f"Registering {t-1} timepoints to reference (t=0)...")
    
    # Register each subsequent timepoint
    for timepoint in range(1, t):
        print(f"  Registering timepoint {timepoint}/{t-1}")
        
        start_time = time.time()
        
        # Convert moving volume to SimpleITK image
        moving_volume = image_4d[:, :, :, timepoint]
        moving_image = _numpy_to_sitk(moving_volume, spacing)
        
        # Perform rigid registration with metrics
        registered_image, transform, reg_metrics = _register_volumes_with_metrics(
            reference_image, moving_image
        )
        
        # Convert back to numpy and store
        registered_volume = _sitk_to_numpy(registered_image)
        registered_4d[:, :, :, timepoint] = registered_volume
        
        # Calculate additional quality metrics
        ncc = compute_registration_quality(reference_image, registered_image)
        mutual_info = compute_mutual_information(reference_image, registered_image)
        mse = compute_mean_squared_error(_sitk_to_numpy(reference_image), registered_volume)
        
        # Extract transform parameters
        translation = transform.GetTranslation()
        rotation = (transform.GetAngleX(), transform.GetAngleY(), transform.GetAngleZ())
        
        registration_time = time.time() - start_time
        
        # Store metrics
        timepoint_metrics = RegistrationMetrics(
            timepoint=timepoint,
            ncc=ncc,
            mutual_info=mutual_info,
            mean_squared_error=mse,
            translation=translation,
            rotation=rotation,
            optimizer_iterations=reg_metrics['iterations'],
            optimizer_metric_value=reg_metrics['final_metric'],
            registration_time=registration_time
        )
        metrics.append(timepoint_metrics)
        
        # Print detailed registration results with pixel debugging
        tx, ty, tz = translation  # These are in mm from SimpleITK
        trans_mag = np.sqrt(tx**2 + ty**2 + tz**2)
        
        # Calculate pixel values for debugging
        # Extract transform parameters and convert to pixels
        tx_pixels = tx / spacing[0] if spacing[0] > 0 else 0
        ty_pixels = ty / spacing[1] if spacing[1] > 0 else 0  
        tz_pixels = tz / spacing[2] if spacing[2] > 0 else 0
        trans_mag_pixels = np.sqrt(tx_pixels**2 + ty_pixels**2 + tz_pixels**2)
        
        print(f"    MI: {mutual_info:.2f}, MSE: {mse:.2f}")
        print(f"    Translation (mm): X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f} (mag={trans_mag:.2f})")
        print(f"    Translation (pixels): X={tx_pixels:.1f}, Y={ty_pixels:.1f}, Z={tz_pixels:.1f} (mag={trans_mag_pixels:.1f})")
        print(f"    Voxel spacing (mm): X={spacing[0]:.3f}, Y={spacing[1]:.3f}, Z={spacing[2]:.3f}")
        print(f"    Time: {registration_time:.1f}s")
    
    print("Registration complete.")
    
    # Save registration data if output directory specified
    if output_dir:
        save_registration_data(registered_4d, metrics, spacing, output_dir, dicom_path)
    
    # Show quality visualization if requested
    if show_quality_window:
        visualize_registration_quality(image_4d, registered_4d, metrics)
    
    return registered_4d, metrics


def save_registration_data(registered_4d: np.ndarray, metrics: List[RegistrationMetrics], 
                          spacing: Tuple[float, float, float], output_dir: str, 
                          dicom_path: Optional[str] = None) -> None:
    """
    Save registered 4D data and metrics to disk.
    
    Parameters
    ----------
    registered_4d : np.ndarray
        Registered 4D array
    metrics : List[RegistrationMetrics]
        Registration quality metrics
    spacing : tuple of float
        Voxel spacing
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save registered 4D data
    reg_data_file = output_path / "registered_4d_data.npz"
    np.savez_compressed(
        reg_data_file,
        registered_4d=registered_4d,
        spacing=np.array(spacing)
    )
    print(f"  Registered 4D data saved to: {reg_data_file}")
    
    # Save metrics as JSON
    metrics_file = output_path / "registration_metrics.json"
    metrics_dict = {
        'metadata': {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dicom_path': dicom_path,
            'dicom_filename': Path(dicom_path).name if dicom_path else None,
            'shape': registered_4d.shape,
            'spacing': spacing
        },
        'metrics': [asdict(m) for m in metrics],
        'summary': {
            'mean_mi': np.mean([m.mutual_info for m in metrics[1:]]),  # Exclude reference
            'std_mi': np.std([m.mutual_info for m in metrics[1:]]),
            'mean_translation': np.mean([np.linalg.norm(m.translation) for m in metrics[1:]]),
            'std_translation': np.std([np.linalg.norm(m.translation) for m in metrics[1:]]),
            'mean_registration_time': np.mean([m.registration_time for m in metrics[1:]]),
            'total_registration_time': sum([m.registration_time for m in metrics[1:]]),
            'worst_translation_timepoint': max(metrics[1:], key=lambda m: np.linalg.norm(m.translation)).timepoint,
            'best_translation_timepoint': min(metrics[1:], key=lambda m: np.linalg.norm(m.translation)).timepoint
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Registration metrics saved to: {metrics_file}")


def load_registration_data(output_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float], List[RegistrationMetrics]]:
    """
    Load previously saved registration data.
    
    Parameters
    ----------
    output_dir : str
        Directory containing saved registration data
        
    Returns
    -------
    registered_4d : np.ndarray
        Registered 4D array
    spacing : tuple of float
        Voxel spacing
    metrics : List[RegistrationMetrics]
        Registration quality metrics
    """
    output_path = Path(output_dir)
    
    # Load registered 4D data
    reg_data_file = output_path / "registered_4d_data.npz"
    if not reg_data_file.exists():
        raise FileNotFoundError(f"Registered data file not found: {reg_data_file}")
    
    data = np.load(reg_data_file)
    registered_4d = data['registered_4d']
    spacing = tuple(data['spacing'])
    
    # Load metrics
    metrics_file = output_path / "registration_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    metrics = []
    for m_dict in metrics_data['metrics']:
        metrics.append(RegistrationMetrics(**m_dict))
    
    print(f"Loaded registered 4D data with shape: {registered_4d.shape}")
    print(f"Registration data contains {len(metrics)} timepoints")
    
    return registered_4d, spacing, metrics


def visualize_registration_quality(original_4d: np.ndarray, registered_4d: np.ndarray, 
                                  metrics: List[RegistrationMetrics], z_slice: int = None) -> None:
    """
    Interactive visualization window for registration quality assessment.
    
    Parameters
    ----------
    original_4d : np.ndarray
        Original 4D data
    registered_4d : np.ndarray
        Registered 4D data  
    metrics : List[RegistrationMetrics]
        Registration quality metrics
    z_slice : int, optional
        Z-slice to display (default: middle slice)
    """
    if z_slice is None:
        z_slice = original_4d.shape[2] // 2
    
    class RegistrationViewer:
        def __init__(self):
            self.current_timepoint = 1
            self.max_timepoint = original_4d.shape[3] - 1
            self.z_slice = z_slice
            self.max_z_slice = original_4d.shape[2] - 1
            
            # Apply consistent styling
            set_proxylfit_style()
            
            # Create figure with reorganized layout - minimal bottom margin
            self.fig = plt.figure(figsize=(16, 10))
            self.fig.subplots_adjust(top=0.93, bottom=0.0, left=0.1, right=0.9, 
                                   hspace=0.28, wspace=0.3)
            # New layout: images, plots, review quality + timepoint info, buttons
            gs = self.fig.add_gridspec(5, 5, height_ratios=[2, 2, 0.12, 0.12, 0.18], width_ratios=[1, 1, 1, 1, 0.8])
            
            # Add title bar (moved higher)
            self.fig.suptitle('ProxylFit – Image Registration Review', 
                            fontsize=16, fontweight='bold', y=0.972)
            
            # Add ProxylFit logo (larger size) moved up by ~15%
            add_proxylfit_logo(self.fig, zoom=0.18, custom_xy=(0.95, 0.20))
            
            # Image displays - Reference (t=0), Registered, Difference, Unregistered
            self.ax_ref = self.fig.add_subplot(gs[0, 0])
            self.ax_registered = self.fig.add_subplot(gs[0, 1])
            self.ax_diff = self.fig.add_subplot(gs[0, 2])
            self.ax_unregistered = self.fig.add_subplot(gs[0, 3])
            
            # Review Quality instructions area (aligned with images)
            self.ax_instructions = self.fig.add_subplot(gs[0, 4])
            self.ax_instructions.axis('off')
            
            # Metrics plots - centered (3 plots in middle columns)
            self.ax_translation = self.fig.add_subplot(gs[1, 1])
            self.ax_rotation = self.fig.add_subplot(gs[1, 2])
            self.ax_mse = self.fig.add_subplot(gs[1, 3])  # MSE plot
            
            # Timepoint info area (aligned with plots)
            self.ax_timepoint_info = self.fig.add_subplot(gs[1, 4])
            self.ax_timepoint_info.axis('off')
            
            # Controls (navigation buttons) - moved up
            self.ax_controls = self.fig.add_subplot(gs[4, :])
            
            # Initialize colorbar reference
            self.diff_colorbar = None
            
            self.setup_controls()
            self.plot_metrics()
            self.setup_click_navigation()  # Setup click-to-navigate
            self.setup_reference()  # Setup reference image once
            self.update_display()
            
        def setup_controls(self):
            # All 5 buttons centered (4 navigation + 1 accept) - moved up by ~15%
            nav_button_width = 0.08
            accept_button_width = 0.10
            button_height = 0.035
            button_y = 0.16
            button_spacing = 0.02
            
            # Calculate total width for all 5 buttons
            total_width = 4 * nav_button_width + accept_button_width + 4 * button_spacing
            start_x = (1.0 - total_width) / 2  # Center all buttons
            
            ax_prev = plt.axes([start_x, button_y, nav_button_width, button_height])
            ax_next = plt.axes([start_x + nav_button_width + button_spacing, button_y, nav_button_width, button_height])
            ax_first = plt.axes([start_x + 2 * (nav_button_width + button_spacing), button_y, nav_button_width, button_height])
            ax_last = plt.axes([start_x + 3 * (nav_button_width + button_spacing), button_y, nav_button_width, button_height])
            ax_accept_btn = plt.axes([start_x + 4 * (nav_button_width + button_spacing), button_y, accept_button_width, button_height])
            
            self.btn_prev = Button(ax_prev, 'Previous')
            self.btn_next = Button(ax_next, 'Next')
            self.btn_first = Button(ax_first, 'First')
            self.btn_last = Button(ax_last, 'Last')
            
            # Style navigation buttons consistently
            for btn in [self.btn_prev, self.btn_next, self.btn_first, self.btn_last]:
                btn.label.set_fontsize(9)
            
            self.btn_prev.on_clicked(self.prev_timepoint)
            self.btn_next.on_clicked(self.next_timepoint)
            self.btn_first.on_clicked(self.first_timepoint)
            self.btn_last.on_clicked(self.last_timepoint)
            
            # Create accept button
            self.btn_accept = Button(ax_accept_btn, 'Accept\nRegistration')
            self.btn_accept.label.set_fontsize(9)
            self.btn_accept.label.set_color('green')
            self.btn_accept.label.set_weight('bold')
            self.btn_accept.on_clicked(self.accept_registration)
            
            self.accepted = False
            
            # Connect keyboard events
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            self.ax_controls.set_xlim(0, 1)
            self.ax_controls.set_ylim(0, 1)
            self.ax_controls.axis('off')
            
        def setup_click_navigation(self):
            """Setup click-to-navigate functionality for metric plots."""
            # Connect click events to all metric plots
            metric_axes = [self.ax_translation, self.ax_rotation, self.ax_mse]
            for ax in metric_axes:
                ax.figure.canvas.mpl_connect('button_press_event', 
                                           lambda event, axis=ax: self.on_plot_click(event, axis))
                # Make plots appear clickable with crosshair cursor
                ax.set_picker(True)
        
        def on_plot_click(self, event, axis):
            """Handle click events on metric plots to navigate to specific timepoint."""
            # Check if the click was on the specified axis
            if event.inaxes == axis:
                # Get the x-coordinate (timepoint) of the click
                clicked_timepoint = round(event.xdata)
                
                # Validate timepoint range (skip t=0 since we start from t=1)
                if clicked_timepoint is not None and 1 <= clicked_timepoint <= self.max_timepoint:
                    print(f"  Jumping to timepoint {clicked_timepoint}")
                    self.current_timepoint = clicked_timepoint
                    self.update_display()
            
        def setup_reference(self):
            # Set up reference image once (t=0) - this never changes initially
            self._update_reference_image()
            
        def _update_reference_image(self):
            """Update reference image for current z-slice."""
            self.ax_ref.clear()
            ref_slice = original_4d[:, :, self.z_slice, 0].T
            self.ax_ref.imshow(ref_slice, cmap='gray', origin='lower')
            self.ax_ref.set_title(f'Reference (t=0) - Z:{self.z_slice}')
            self.ax_ref.axis('off')
            
        def plot_metrics(self):
            timepoints = [m.timepoint for m in metrics]
            
            # Translation plot
            trans_mag = [np.linalg.norm(m.translation) for m in metrics]
            self.ax_translation.plot(timepoints, trans_mag, 'go-')
            self.ax_translation.set_xlabel('Timepoint', fontsize=10)
            self.ax_translation.set_ylabel('Translation (mm)', fontsize=10)
            self.ax_translation.set_title('Translation Magnitude', fontsize=11)
            self.ax_translation.grid(True, alpha=0.3)
            self.trans_marker, = self.ax_translation.plot([], [], 'ro', markersize=8)
            
            # Rotation plot
            rot_mag = [np.linalg.norm(m.rotation) * 180 / np.pi for m in metrics]
            self.ax_rotation.plot(timepoints, rot_mag, 'mo-')
            self.ax_rotation.set_xlabel('Timepoint', fontsize=10)
            self.ax_rotation.set_ylabel('Rotation (degrees)', fontsize=10)
            self.ax_rotation.set_title('Rotation Magnitude', fontsize=11)
            self.ax_rotation.grid(True, alpha=0.3)
            self.rot_marker, = self.ax_rotation.plot([], [], 'ro', markersize=8)
            
            # Difference plot (changed from MSE, changed color from red to blue)
            mse_values = [m.mean_squared_error for m in metrics]
            self.ax_mse.plot(timepoints, mse_values, 'bo-')
            self.ax_mse.set_xlabel('Timepoint', fontsize=10)
            self.ax_mse.set_ylabel('|Registered - t0|', fontsize=10)
            self.ax_mse.set_title('Image Difference', fontsize=11)
            self.ax_mse.grid(True, alpha=0.3)
            self.mse_marker, = self.ax_mse.plot([], [], 'ro', markersize=8)
            
            # Accept button is created in setup_controls()
            
            # Add instructions in the instructions area (left of buttons)
            self.ax_instructions.text(0.5, 0.9, 'Review Quality', 
                            ha='center', va='center', fontsize=11, weight='bold')
            self.ax_instructions.text(0.5, 0.7, '↑↓ z-slices', 
                            ha='center', va='center', fontsize=9)
            self.ax_instructions.text(0.5, 0.5, '←→ timepoints', 
                            ha='center', va='center', fontsize=9)
            self.ax_instructions.text(0.5, 0.3, 'Click plots to jump', 
                            ha='center', va='center', fontsize=9)
            self.ax_instructions.text(0.5, 0.1, 'Click "Accept" when ready', 
                            ha='center', va='center', fontsize=9, color='green', weight='bold')
            
        def update_display(self):
            # Get current images (reference slice never changes since it's setup once)
            orig_slice = original_4d[:, :, self.z_slice, self.current_timepoint].T
            reg_slice = registered_4d[:, :, self.z_slice, self.current_timepoint].T
            ref_slice = original_4d[:, :, self.z_slice, 0].T  # Reference for difference calculation
            
            # Clear only the axes that need updating (NOT the reference axis)
            self.ax_registered.clear()
            self.ax_diff.clear()
            self.ax_unregistered.clear()
            
            # Registered image (moved to position next to reference)
            self.ax_registered.imshow(reg_slice, cmap='gray', origin='lower')
            self.ax_registered.set_title(f'Registered t{self.current_timepoint} → t0')
            self.ax_registered.axis('off')
            
            # Difference image
            diff_slice = np.abs(ref_slice - reg_slice)
            im_diff = self.ax_diff.imshow(diff_slice, cmap='jet', origin='lower')
            self.ax_diff.set_title(f'|t0 - t{self.current_timepoint}|')
            self.ax_diff.axis('off')
            
            # Only create colorbar once
            if self.diff_colorbar is None:
                self.diff_colorbar = plt.colorbar(im_diff, ax=self.ax_diff, fraction=0.046)
            else:
                # Update existing colorbar with new data range
                self.diff_colorbar.update_normal(im_diff)
            
            # Unregistered original image
            self.ax_unregistered.imshow(orig_slice, cmap='gray', origin='lower')
            self.ax_unregistered.set_title(f'Original t{self.current_timepoint} (unregistered)')
            self.ax_unregistered.axis('off')
            
            # Update metric markers
            current_metric = metrics[self.current_timepoint]
            self.trans_marker.set_data([current_metric.timepoint], [np.linalg.norm(current_metric.translation)])
            self.rot_marker.set_data([current_metric.timepoint], [np.linalg.norm(current_metric.rotation) * 180 / np.pi])
            self.mse_marker.set_data([current_metric.timepoint], [current_metric.mean_squared_error])
            
            # Update info text with individual translation components
            tx, ty, tz = current_metric.translation
            translation_mag = np.linalg.norm(current_metric.translation)
            rotation_mag = np.linalg.norm(current_metric.rotation) * 180 / np.pi
            
            # Update info display in right column (avoiding control area overlap)
            info_text = (
                f"Timepoint: {self.current_timepoint}/{self.max_timepoint}\n"
                f"Z-slice: {self.z_slice}/{self.max_z_slice}\n\n"
                f"MI: {current_metric.mutual_info:.2f}\n"
                f"MSE: {current_metric.mean_squared_error:.2f}\n\n"
                f"Translation (mm):\n"
                f"  X: {tx:.2f}, Y: {ty:.2f}, Z: {tz:.2f}\n"
                f"  Mag: {translation_mag:.2f}\n\n"
                f"Rotation: {rotation_mag:.2f}°"
            )
            
            # Clear and update timepoint info panel (aligned with plots)
            if hasattr(self, 'timepoint_info_obj'):
                self.timepoint_info_obj.remove()
            
            self.timepoint_info_obj = self.ax_timepoint_info.text(0.5, 0.5, info_text, fontsize=9, 
                                                                 ha='center', va='center',
                                                                 transform=self.ax_timepoint_info.transAxes,
                                                                 bbox=dict(boxstyle="round,pad=0.4", 
                                                                         facecolor="lightblue", alpha=0.8))
            
            plt.draw()
            
        def prev_timepoint(self, event):
            if self.current_timepoint > 1:
                self.current_timepoint -= 1
                self.update_display()
                
        def next_timepoint(self, event):
            if self.current_timepoint < self.max_timepoint:
                self.current_timepoint += 1
                self.update_display()
                
        def first_timepoint(self, event):
            self.current_timepoint = 1
            self.update_display()
            
        def last_timepoint(self, event):
            self.current_timepoint = self.max_timepoint
            self.update_display()
        
        def _on_key_press(self, event):
            """Handle keyboard navigation."""
            if event.key == 'up':  # Next z-slice
                if self.z_slice < self.max_z_slice:
                    self.z_slice += 1
                    self._update_z_slice()
            elif event.key == 'down':  # Previous z-slice
                if self.z_slice > 0:
                    self.z_slice -= 1
                    self._update_z_slice()
            elif event.key == 'left':  # Previous timepoint
                self.prev_timepoint(event)
            elif event.key == 'right':  # Next timepoint
                self.next_timepoint(event)
        
        def _update_z_slice(self):
            """Update display for new z-slice."""
            self._update_reference_image()
            self.update_display()
            print(f"Switched to z-slice {self.z_slice}")
        
        def accept_registration(self, event):
            """Accept the registration and close window."""
            self.accepted = True
            print("Registration accepted!")
            plt.close(self.fig)
    
    viewer = RegistrationViewer()
    viewer.fig.tight_layout(rect=[0.02, 0.0, 0.98, 0.93])
    plt.show()


def _register_volumes_with_metrics(reference: sitk.Image, moving: sitk.Image) -> Tuple[sitk.Image, sitk.Transform, Dict]:
    """
    Perform rigid registration between two 3D volumes with detailed metrics.
    
    Parameters
    ----------
    reference : sitk.Image
        Reference (fixed) image
    moving : sitk.Image
        Moving image to be registered
        
    Returns
    -------
    sitk.Image
        Registered moving image
    sitk.Transform
        Final transformation
    dict
        Registration metrics
    """
    # Initialize registration framework
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set metric - Mutual Information with improved sampling
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)  # More bins for better metric
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.05)  # More sampling for better accuracy
    
    # Set interpolator - Linear
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set optimizer - Regular Step Gradient Descent with improved parameters
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,  # Reduced learning rate for more stable convergence
        minStep=1e-6,      # Smaller minimum step for finer adjustments
        numberOfIterations=2000,  # Increased iterations for better convergence
        gradientMagnitudeTolerance=1e-10  # Tighter convergence tolerance
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Set transform - 3D Euler (rotation + translation)
    initial_transform = sitk.CenteredTransformInitializer(
        reference,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform)
    
    # Set shrink factors and smoothing sigmas for multi-resolution with more levels
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])  # More resolution levels
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])  # Corresponding smoothing
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Track optimization progress
    iteration_count = [0]
    metric_values = []
    
    def iteration_callback():
        iteration_count[0] += 1
        metric_values.append(registration_method.GetMetricValue())
    
    registration_method.AddCommand(sitk.sitkIterationEvent, iteration_callback)
    
    # Execute registration
    try:
        final_transform = registration_method.Execute(reference, moving)
        
        # Apply transformation to moving image
        registered_image = sitk.Resample(
            moving,
            reference,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID()
        )
        
        # Collect metrics
        reg_metrics = {
            'iterations': iteration_count[0],
            'final_metric': registration_method.GetMetricValue() if metric_values else 0.0,
            'metric_history': metric_values,
            'optimizer_stop_condition': registration_method.GetOptimizerStopConditionDescription()
        }
        
        return registered_image, final_transform, reg_metrics
        
    except Exception as e:
        print(f"Warning: Registration failed, returning original image. Error: {e}")
        # Return identity transform
        identity_transform = sitk.Euler3DTransform()
        reg_metrics = {
            'iterations': 0,
            'final_metric': float('inf'),
            'metric_history': [],
            'optimizer_stop_condition': f"Registration failed: {e}"
        }
        return moving, identity_transform, reg_metrics


def _numpy_to_sitk(volume: np.ndarray, spacing: Tuple[float, float, float]) -> sitk.Image:
    """
    Convert numpy array to SimpleITK image with proper spacing.
    
    Parameters
    ----------
    volume : np.ndarray
        3D numpy array with shape [x, y, z]
    spacing : tuple of float
        Voxel spacing (x, y, z)
        
    Returns
    -------
    sitk.Image
        SimpleITK image object
    """
    # SimpleITK expects [z, y, x] ordering, so transpose
    volume_sitk_order = np.transpose(volume, (2, 1, 0))
    
    # Create SimpleITK image
    image = sitk.GetImageFromArray(volume_sitk_order)
    image.SetSpacing(spacing)
    
    return image


def _sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """
    Convert SimpleITK image to numpy array with [x, y, z] ordering.
    
    Parameters
    ----------
    image : sitk.Image
        SimpleITK image object
        
    Returns
    -------
    np.ndarray
        3D numpy array with shape [x, y, z]
    """
    # Get array from SimpleITK (returns [z, y, x])
    array_sitk_order = sitk.GetArrayFromImage(image)
    
    # Transpose to [x, y, z]
    array = np.transpose(array_sitk_order, (2, 1, 0))
    
    return array


def compute_registration_quality(reference: sitk.Image, registered: sitk.Image) -> float:
    """
    Compute registration quality using normalized cross-correlation.
    
    Parameters
    ----------
    reference : sitk.Image
        Reference image
    registered : sitk.Image
        Registered image
        
    Returns
    -------
    float
        Normalized cross-correlation value (higher is better)
    """
    try:
        # Compute normalized cross-correlation
        ncc_filter = sitk.NormalizedCorrelationImageFilter()
        ncc_image = ncc_filter.Execute(reference, registered)
        
        # Get mean NCC value
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(ncc_image)
        
        return stats_filter.GetMean()
        
    except Exception:
        return 0.0


def compute_mutual_information(reference: sitk.Image, registered: sitk.Image) -> float:
    """
    Compute mutual information between two images.
    
    Parameters
    ----------
    reference : sitk.Image
        Reference image
    registered : sitk.Image
        Registered image
        
    Returns
    -------
    float
        Mutual information value
    """
    try:
        # Use the same metric as the registration
        mi_metric = sitk.MattesMutualInformationImageToImageMetricv4()
        mi_metric.SetNumberOfHistogramBins(50)
        mi_metric.Initialize(reference, registered)
        
        return -mi_metric.GetValue()  # Negative because SimpleITK minimizes
        
    except Exception:
        return 0.0


def compute_mean_squared_error(reference: np.ndarray, registered: np.ndarray) -> float:
    """
    Compute mean squared error between two volumes.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference volume
    registered : np.ndarray
        Registered volume
        
    Returns
    -------
    float
        Mean squared error
    """
    try:
        return np.mean((reference - registered) ** 2)
    except Exception:
        return float('inf')


# Legacy function for compatibility
def register_timeseries_simple(image_4d: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Simple registration function that returns only the registered 4D data (legacy compatibility).
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    spacing : tuple of float
        Voxel spacing (x, y, z) from DICOM metadata
        
    Returns
    -------
    registered_4d : np.ndarray
        Registered 4D array with same shape [x, y, z, t]
    """
    registered_4d, _ = register_timeseries(image_4d, spacing, show_quality_window=False)
    return registered_4d