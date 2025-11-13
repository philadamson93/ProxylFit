"""
ROI selection module for interactive rectangle selection on MRI slices.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Polygon
from typing import Tuple, Optional, List
import warnings
from .model import add_proxylfit_logo, set_proxylfit_style

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn("SegmentAnything not available. Install with: pip install segment-anything")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Install with: pip install opencv-python")


class ROISelector:
    """Interactive rectangle selector for defining ROI on MRI slice."""
    
    def __init__(self, image_slice: np.ndarray, title: str = "Select ROI"):
        """
        Initialize ROI selector.
        
        Parameters
        ----------
        image_slice : np.ndarray
            2D image slice to display
        title : str
            Title for the plot window
        """
        self.image_slice = image_slice
        self.roi_coords = None
        self.mask = None
        self.title = title
        
        # Apply consistent styling
        set_proxylfit_style()
        
        # Set up the plot with padding (extra bottom margin for logo)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.85)
        
        # Add title with program name and context
        self.fig.suptitle(f'ProxylFit – {title}', fontsize=14, fontweight='bold', y=0.95)
        self.ax.set_title("Click and drag to select rectangular ROI", fontsize=11)
        
        # Add ProxylFit logo in bottom-right
        add_proxylfit_logo(self.fig, position='bottom-right')
        
        # Display image
        self.im = self.ax.imshow(image_slice.T, cmap='gray', origin='lower')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Add colorbar positioned to avoid logo
        cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        
        # Initialize rectangle selector
        self.rectangle_selector = RectangleSelector(
            self.ax,
            self._on_rectangle_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        # Add instructions
        self.ax.text(0.02, 0.98, 
                    "Instructions:\n1. Click and drag to select ROI\n2. Close window when done", 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _on_rectangle_select(self, eclick, erelease):
        """Callback for rectangle selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure coordinates are in correct order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Store ROI coordinates
        self.roi_coords = (x_min, x_max, y_min, y_max)
        
        # Create mask
        self.mask = np.zeros(self.image_slice.shape, dtype=bool)
        self.mask[x_min:x_max, y_min:y_max] = True
        
        # Update title with ROI info
        roi_size = (x_max - x_min) * (y_max - y_min)
        self.ax.set_title(f"{self.title}\nROI: ({x_min}, {y_min}) to ({x_max}, {y_max}), Size: {roi_size} pixels")
        
        print(f"ROI selected: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    def show_and_select(self) -> np.ndarray:
        """
        Display image and wait for user ROI selection.
        
        Returns
        -------
        np.ndarray
            Boolean mask of shape [x, y] where True indicates pixels inside ROI
        """
        plt.show()
        
        if self.mask is None:
            print("Warning: No ROI was selected")
            return np.zeros(self.image_slice.shape, dtype=bool)
        
        return self.mask
    
    def get_roi_stats(self) -> dict:
        """
        Get statistics about the selected ROI.
        
        Returns
        -------
        dict
            Dictionary containing ROI statistics
        """
        if self.mask is None:
            return {}
        
        roi_pixels = self.image_slice[self.mask]
        
        return {
            'num_pixels': np.sum(self.mask),
            'mean_intensity': np.mean(roi_pixels),
            'std_intensity': np.std(roi_pixels),
            'min_intensity': np.min(roi_pixels),
            'max_intensity': np.max(roi_pixels),
            'coordinates': self.roi_coords
        }


def select_rectangle_roi(image_4d: np.ndarray, z_index: int) -> np.ndarray:
    """
    Interactive selection of rectangular ROI on a specific slice.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    z_index : int
        Z-slice index to display for ROI selection
        
    Returns
    -------
    roi_mask : np.ndarray
        Boolean mask of shape [x, y] where True indicates pixels inside ROI
        
    Raises
    ------
    IndexError
        If z_index is out of bounds
    """
    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")
    
    # Extract slice from first timepoint
    image_slice = image_4d[:, :, z_index, 0]
    
    # Create ROI selector
    title = f"ROI Selection - Slice {z_index} (Timepoint 0)"
    selector = ROISelector(image_slice, title)
    
    # Show and get ROI selection
    roi_mask = selector.show_and_select()
    
    # Print ROI statistics
    stats = selector.get_roi_stats()
    if stats:
        print(f"\nROI Statistics:")
        print(f"  Number of pixels: {stats['num_pixels']}")
        print(f"  Mean intensity: {stats['mean_intensity']:.2f}")
        print(f"  Std intensity: {stats['std_intensity']:.2f}")
        print(f"  Min intensity: {stats['min_intensity']:.2f}")
        print(f"  Max intensity: {stats['max_intensity']:.2f}")
    
    return roi_mask


def compute_roi_timeseries(image_4d: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    Compute mean intensity time series for ROI across all timepoints.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    roi_mask : np.ndarray
        Boolean mask of shape [x, y] defining the ROI
        
    Returns
    -------
    timeseries : np.ndarray
        1D array of mean ROI intensity for each timepoint
    """
    if not np.any(roi_mask):
        raise ValueError("ROI mask contains no True values")
    
    t_points = image_4d.shape[3]
    timeseries = np.zeros(t_points)
    
    # Compute mean for each timepoint
    for t in range(t_points):
        # Extract 2D slice for this timepoint
        slice_2d = image_4d[:, :, :, t]
        
        # Apply mask and compute mean across all z-slices
        masked_values = []
        for z in range(slice_2d.shape[2]):
            slice_z = slice_2d[:, :, z]
            masked_values.extend(slice_z[roi_mask])
        
        timeseries[t] = np.mean(masked_values)
    
    return timeseries


def visualize_roi_on_slice(image_slice: np.ndarray, roi_mask: np.ndarray, 
                          title: str = "ROI Visualization") -> None:
    """
    Visualize the selected ROI overlaid on the image slice.
    
    Parameters
    ----------
    image_slice : np.ndarray
        2D image slice
    roi_mask : np.ndarray
        Boolean mask defining the ROI
    title : str
        Title for the plot
    """
    # Apply consistent styling
    set_proxylfit_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.08, right=0.88, wspace=0.3)
    fig.suptitle(f'ProxylFit – {title}', fontsize=14, fontweight='bold', y=0.95)
    
    # Add ProxylFit logo in bottom-right
    add_proxylfit_logo(fig, position='bottom-right')
    
    # Original image
    ax1.imshow(image_slice.T, cmap='gray', origin='lower')
    ax1.set_title("Original Image")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Image with ROI overlay
    overlay = image_slice.copy()
    overlay_color = np.zeros((*image_slice.shape, 3))
    overlay_color[:, :, 0] = overlay / np.max(overlay)  # Red channel
    overlay_color[:, :, 1] = overlay / np.max(overlay)  # Green channel  
    overlay_color[:, :, 2] = overlay / np.max(overlay)  # Blue channel
    
    # Highlight ROI in red
    overlay_color[roi_mask, 0] = 1.0  # Full red
    overlay_color[roi_mask, 1] = 0.3  # Reduced green
    overlay_color[roi_mask, 2] = 0.3  # Reduced blue
    
    ax2.imshow(np.transpose(overlay_color, (1, 0, 2)), origin='lower')
    ax2.set_title("ROI Overlay (Red)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


class ManualContourROISelector:
    """Interactive manual contour drawing for defining ROI on MRI slice."""
    
    def __init__(self, image_4d: np.ndarray, z_index: int = 0, title: str = "Draw ROI Contour"):
        """
        Initialize manual contour ROI selector.
        
        Parameters
        ----------
        image_4d : np.ndarray
            4D image array with shape [x, y, z, t]
        z_index : int
            Initial z-slice index to display
        title : str
            Title for the plot window
        """
        self.image_4d = image_4d
        self.z_index = z_index
        self.max_z = image_4d.shape[2] - 1
        self.image_slice = image_4d[:, :, z_index, 0]  # First timepoint
        self.mask = None
        self.title = title
        self.contour_points = []
        self.drawing = False
        self.current_path = []
        self.path_plots = []
        
        # Apply consistent styling
        set_proxylfit_style()
        
        # Set up the plot with padding (extra bottom margin for logo)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.08, right=0.88)
        self._update_title()
        
        # Add ProxylFit logo in bottom-right
        add_proxylfit_logo(self.fig, position='bottom-right')
        
        # Display image
        self.im = self.ax.imshow(self.image_slice.T, cmap='gray', origin='lower')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Add colorbar
        plt.colorbar(self.im, ax=self.ax)
        
        # Add Accept button with centered layout and extra padding
        from matplotlib.widgets import Button
        button_width = 0.12
        button_height = 0.05
        button_x = (1.0 - button_width) / 2  # Center horizontally
        ax_accept = plt.axes([button_x, 0.02, button_width, button_height])
        self.btn_accept = Button(ax_accept, 'Accept ROI')
        
        # Style button consistently
        self.btn_accept.label.set_fontsize(10)
        self.btn_accept.label.set_color('green')
        self.btn_accept.label.set_weight('bold')
        
        self.btn_accept.on_clicked(self._accept_roi)
        self.accepted = False
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Enable keyboard focus (cross-platform)
        try:
            self.fig.canvas.setFocusPolicy(2)  # Qt.StrongFocus for Qt backends
        except:
            pass  # Ignore if not available
        
        # Add instructions
        self.instruction_text = self.ax.text(0.02, 0.98, 
                    "Instructions:\n"
                    "• Drag to draw contour around ROI\n"
                    "• Arrow keys: Navigate z-slices\n"
                    "• 'c': Close contour and create mask\n" 
                    "• 'r': Reset and start over\n"
                    "• Click 'Accept ROI' when satisfied", 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    def _update_title(self):
        """Update the title with current z-slice information."""
        # Add figure title with program name and context
        self.fig.suptitle(f'ProxylFit – Manual ROI Selection (Z-slice {self.z_index}/{self.max_z})', 
                         fontsize=14, fontweight='bold', y=0.95)
        
        # Add instructions box in lower-left to avoid logo
        instruction_text = "Drag to draw contour, ↑↓ for z-slices, 'c' to close, 'r' to reset"
        if hasattr(self, 'instruction_text'):
            self.instruction_text.remove()
        
        self.instruction_text = self.ax.text(0.02, 0.02, instruction_text,
                                           transform=self.ax.transAxes,
                                           verticalalignment='bottom',
                                           bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="lightblue", alpha=0.8),
                                           fontsize=9)
    
    def _on_press(self, event):
        """Start drawing when mouse is pressed."""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        self.drawing = True
        self.current_path = [(event.xdata, event.ydata)]
        print(f"Started drawing contour at ({event.xdata:.1f}, {event.ydata:.1f})")
    
    def _on_release(self, event):
        """Stop drawing when mouse is released."""
        if not self.drawing:
            return
            
        self.drawing = False
        if len(self.current_path) > 2:
            # Add the path to our contour points
            self.contour_points.extend(self.current_path)
            print(f"Added path segment with {len(self.current_path)} points")
        self.current_path = []
    
    def _on_motion(self, event):
        """Add points while dragging."""
        if not self.drawing or event.inaxes != self.ax:
            return
            
        # Add point if we've moved a reasonable distance
        if len(self.current_path) > 0:
            last_x, last_y = self.current_path[-1]
            dist = ((event.xdata - last_x)**2 + (event.ydata - last_y)**2)**0.5
            if dist > 2:  # Minimum distance between points
                self.current_path.append((event.xdata, event.ydata))
                
                # Draw the line segment
                if len(self.current_path) > 1:
                    prev_x, prev_y = self.current_path[-2]
                    line = self.ax.plot([prev_x, event.xdata], [prev_y, event.ydata], 'r-', linewidth=2)[0]
                    self.path_plots.append(line)
                    self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'r':  # Reset
            self._reset_contour()
        elif event.key == 'c':  # Close contour
            self._close_contour()
        elif event.key == 'up':  # Next z-slice
            self._next_z_slice()
        elif event.key == 'down':  # Previous z-slice
            self._prev_z_slice()
    
    def _reset_contour(self):
        """Reset the contour and start over."""
        self.contour_points = []
        self.current_path = []
        self.mask = None
        
        # Remove all path plots
        for plot in self.path_plots:
            plot.remove()
        self.path_plots = []
        
        self.fig.canvas.draw()
        print("Reset contour")
    
    def _next_z_slice(self):
        """Navigate to next z-slice."""
        if self.z_index < self.max_z:
            self.z_index += 1
            self._update_z_slice()
    
    def _prev_z_slice(self):
        """Navigate to previous z-slice."""
        if self.z_index > 0:
            self.z_index -= 1
            self._update_z_slice()
    
    def _update_z_slice(self):
        """Update display for new z-slice."""
        # Clear current contour
        self._reset_contour()
        
        # Update image slice
        self.image_slice = self.image_4d[:, :, self.z_index, 0]
        
        # Update image display
        self.im.set_data(self.image_slice.T)
        self.im.set_clim(vmin=np.min(self.image_slice), vmax=np.max(self.image_slice))
        
        # Update title
        self._update_title()
        
        self.fig.canvas.draw()
        print(f"Switched to z-slice {self.z_index}")
    
    def _close_contour(self):
        """Close the contour and create a mask."""
        if len(self.contour_points) < 3:
            print("Need at least 3 points to create contour. Draw more!")
            return
        
        # Close the contour by connecting back to start
        if len(self.contour_points) > 0:
            start_point = self.contour_points[0]
            end_point = self.contour_points[-1]
            
            # Draw closing line
            closing_line = self.ax.plot([end_point[0], start_point[0]], 
                                      [end_point[1], start_point[1]], 'g-', linewidth=3)[0]
            self.path_plots.append(closing_line)
        
        # Create mask from contour
        self._create_mask_from_contour()
        self.fig.canvas.draw()
    
    def _create_mask_from_contour(self):
        """Create a binary mask from the drawn contour."""
        from matplotlib.path import Path
        
        if len(self.contour_points) < 3:
            return
        
        print(f"Creating mask from {len(self.contour_points)} contour points")
        print(f"Image shape: {self.image_slice.shape}")
        print(f"First few contour points: {self.contour_points[:3]}")
        
        # Create a Path object from contour points
        path = Path(self.contour_points)
        
        # Create coordinate grids matching the display coordinates
        # Since we use origin='lower' in imshow, coordinates should match directly
        height, width = self.image_slice.shape[1], self.image_slice.shape[0]  # Note: T is used in display
        
        # Create meshgrid in the same coordinate system as the plot
        x = np.arange(width)   # X coordinates (0 to image_slice.shape[0])
        y = np.arange(height)  # Y coordinates (0 to image_slice.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Flatten coordinate arrays - points should be in (x, y) format to match contour_points
        points = np.column_stack((X.ravel(), Y.ravel()))
        
        # Check which points are inside the contour
        mask_flat = path.contains_points(points)
        
        # Reshape to match the coordinate system: (width, height) then transpose for storage
        mask_display_coords = mask_flat.reshape((height, width))  # Shape for display
        self.mask = mask_display_coords.T  # Transpose to match image_slice shape (width, height)
        
        print(f"Mask shape: {self.mask.shape}")
        print(f"Mask pixels: {np.sum(self.mask)}")
        
        # Create overlay to show the mask - use the display coordinate version
        mask_for_display = np.ma.masked_where(~mask_display_coords, np.ones_like(mask_display_coords))
        overlay = self.ax.imshow(mask_for_display, cmap='Reds', alpha=0.4, origin='lower', 
                               extent=self.im.get_extent())
        self.path_plots.append(overlay)
        
        num_pixels = np.sum(self.mask)
        total_pixels = self.mask.size
        print(f"Contour closed! ROI contains {num_pixels} pixels ({100*num_pixels/total_pixels:.1f}% of image)")
        print("Click 'Accept ROI' button to confirm, or 'r' to reset and redraw")
    
    def _accept_roi(self, event):
        """Accept the current ROI and close window."""
        if self.mask is not None:
            self.accepted = True
            print("ROI accepted!")
            plt.close(self.fig)
        else:
            print("No ROI drawn yet. Draw a contour first.")
    
    def show_and_select(self) -> np.ndarray:
        """
        Display image and wait for user ROI drawing.
        
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates pixels inside drawn ROI
        """
        plt.show()
        
        if self.mask is None:
            print("Warning: No ROI contour was drawn")
            return np.zeros(self.image_slice.shape, dtype=bool)
        
        return self.mask
    
    def get_roi_stats(self) -> dict:
        """
        Get statistics about the drawn ROI.
        
        Returns
        -------
        dict
            Dictionary containing ROI statistics
        """
        if self.mask is None:
            return {}
        
        roi_pixels = self.image_slice[self.mask]
        
        # Find bounding box
        y_indices, x_indices = np.where(self.mask)
        if len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = (x_min, x_max, y_min, y_max)
        else:
            bbox = None
        
        return {
            'num_pixels': np.sum(self.mask),
            'area_fraction': np.sum(self.mask) / self.mask.size,
            'mean_intensity': np.mean(roi_pixels) if len(roi_pixels) > 0 else 0,
            'std_intensity': np.std(roi_pixels) if len(roi_pixels) > 0 else 0,
            'min_intensity': np.min(roi_pixels) if len(roi_pixels) > 0 else 0,
            'max_intensity': np.max(roi_pixels) if len(roi_pixels) > 0 else 0,
            'bounding_box': bbox,
            'num_contour_points': len(self.contour_points)
        }


class SegmentAnythingROISelector:
    """Interactive segmentation selector using SegmentAnything for tumor outline."""
    
    def __init__(self, image_slice: np.ndarray, model_path: Optional[str] = None, 
                 model_type: str = "vit_h", title: str = "Segment ROI"):
        """
        Initialize SegmentAnything ROI selector.
        
        Parameters
        ----------
        image_slice : np.ndarray
            2D image slice to display
        model_path : str, optional
            Path to SAM model checkpoint. If None, will attempt to auto-download
        model_type : str
            SAM model type ('vit_h', 'vit_l', 'vit_b')
        title : str
            Title for the plot window
        """
        if not SAM_AVAILABLE:
            raise ImportError("SegmentAnything not available. Install with: pip install segment-anything")
        
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available. Install with: pip install opencv-python")
        
        self.image_slice = image_slice
        self.mask = None
        self.title = title
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.point_mode = 'positive'  # 'positive' or 'negative'
        self.point_plots = []  # Store point plot objects for removal
        
        # Initialize SAM
        self._init_sam(model_path, model_type)
        
        # Apply consistent styling
        set_proxylfit_style()
        
        # Set up the plot with padding (extra bottom margin for logo)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.08, right=0.88)
        self._update_title()
        
        # Add ProxylFit logo in bottom-right
        add_proxylfit_logo(self.fig, position='bottom-right')
        
        # Display image (convert to RGB for SAM)
        self.image_rgb = self._prepare_image_for_sam(image_slice)
        self.predictor.set_image(self.image_rgb)
        
        # Display the image
        self.im = self.ax.imshow(image_slice.T, cmap='gray', origin='lower')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Add colorbar
        plt.colorbar(self.im, ax=self.ax)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Overlay for showing current segmentation
        self.mask_overlay = None
        self.mask_fill = None
        
        # Add instructions
        self.instruction_text = self.ax.text(0.02, 0.98, 
                    "Instructions:\n"
                    "• Click: Add point (mode shown in title)\n"
                    "• 't': Toggle positive/negative mode\n" 
                    "• 's': Run segmentation\n"
                    "• 'r': Reset points\n"
                    "• 'c': Confirm segmentation\n"
                    "• Close window when done", 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue" if self.point_mode == 'positive' else "lightcoral", alpha=0.8))
    
    def _init_sam(self, model_path: Optional[str], model_type: str) -> None:
        """Initialize SegmentAnything model."""
        try:
            if model_path is None:
                # Try to auto-download model (requires internet)
                print(f"Loading SAM model ({model_type})...")
                self.sam = sam_model_registry[model_type]()
                print("Note: For offline use, download SAM checkpoint and provide model_path")
            else:
                self.sam = sam_model_registry[model_type](checkpoint=model_path)
            
            self.predictor = SamPredictor(self.sam)
            print("SAM model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM model: {e}")
    
    def _prepare_image_for_sam(self, image_slice: np.ndarray) -> np.ndarray:
        """Convert grayscale MRI slice to RGB format for SAM."""
        # Normalize to 0-255
        normalized = ((image_slice - np.min(image_slice)) / 
                     (np.max(image_slice) - np.min(image_slice)) * 255).astype(np.uint8)
        
        # Convert to RGB by stacking
        rgb_image = np.stack([normalized, normalized, normalized], axis=-1)
        
        return rgb_image
    
    def _update_title(self):
        """Update the title with current mode information."""
        mode_text = f"[{self.point_mode.upper()} MODE]"
        color = "green" if self.point_mode == 'positive' else "red"
        
        # Add figure title with program name
        self.fig.suptitle(f'ProxylFit – SegmentAnything ROI Selection', 
                         fontsize=14, fontweight='bold', y=0.95)
        
        # Add mode indicator and instructions in lower-left to avoid logo
        instruction_text = f"{mode_text} - Click to add points, 't' to toggle, 's' to segment, 'c' to confirm"
        if hasattr(self, 'instruction_text'):
            self.instruction_text.remove()
        
        self.instruction_text = self.ax.text(0.02, 0.02, instruction_text,
                                           transform=self.ax.transAxes,
                                           verticalalignment='bottom',
                                           color=color, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="lightgray", alpha=0.8),
                                           fontsize=9)
    
    def _on_click(self, event):
        """Handle mouse clicks for adding positive/negative points."""
        if event.inaxes != self.ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        if event.button == 1:  # Left click - add point based on current mode
            if self.point_mode == 'positive':
                self.positive_points.append((x, y))
                point_plot = self.ax.plot(x, y, 'g+', markersize=15, markeredgewidth=3)[0]
                self.point_plots.append(point_plot)
                print(f"Added positive point at ({x}, {y})")
            else:  # negative mode
                self.negative_points.append((x, y))
                point_plot = self.ax.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_plots.append(point_plot)
                print(f"Added negative point at ({x}, {y})")
        
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'r':  # Reset
            self._reset_points()
        elif event.key == 'c':  # Confirm
            self._confirm_segmentation()
        elif event.key == 't':  # Toggle mode
            self._toggle_point_mode()
        elif event.key == 's':  # Segment
            self._update_segmentation()
    
    def _reset_points(self):
        """Reset all points and segmentation."""
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.point_mode = 'positive'  # Reset to positive mode
        
        # Clear plot overlays
        for plot in self.point_plots:
            plot.remove()
        self.point_plots = []
        
        if self.mask_overlay is not None:
            try:
                self.mask_overlay.remove()
            except:
                pass
            self.mask_overlay = None
        
        
        # Clear and redraw
        self.ax.clear()
        self.im = self.ax.imshow(self.image_slice.T, cmap='gray', origin='lower')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.colorbar(self.im, ax=self.ax)
        
        self._update_title()
        
        # Re-add instructions
        self.instruction_text = self.ax.text(0.02, 0.98, 
                    "Instructions:\n"
                    "• Click: Add point (mode shown in title)\n"
                    "• 't': Toggle positive/negative mode\n" 
                    "• 's': Run segmentation\n"
                    "• 'r': Reset points\n"
                    "• 'c': Confirm segmentation\n"
                    "• Close window when done", 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        self.fig.canvas.draw()
        print("Reset all points")
    
    def _toggle_point_mode(self):
        """Toggle between positive and negative point mode."""
        self.point_mode = 'negative' if self.point_mode == 'positive' else 'positive'
        self._update_title()
        
        # Update instruction box color
        self.instruction_text.set_bbox(dict(
            boxstyle="round,pad=0.3", 
            facecolor="lightblue" if self.point_mode == 'positive' else "lightcoral", 
            alpha=0.8
        ))
        
        self.fig.canvas.draw()
        print(f"Switched to {self.point_mode} mode")
    
    def _confirm_segmentation(self):
        """Confirm the current segmentation as final."""
        if self.current_mask is not None:
            self.mask = self.current_mask.copy()
            print(f"Segmentation confirmed: {np.sum(self.mask)} pixels selected")
            
            # Update title to show confirmation
            self.ax.set_title(f"{self.title} - Segmentation CONFIRMED")
            self.fig.canvas.draw()
        else:
            print("No segmentation to confirm")
    
    def _update_segmentation(self):
        """Update segmentation based on current points."""
        if not self.positive_points:
            print("No positive points added yet. Add at least one positive point first.")
            return
        
        print(f"Running segmentation with {len(self.positive_points)} positive and {len(self.negative_points)} negative points...")
        
        # Prepare points for SAM
        input_points = np.array(self.positive_points + self.negative_points)
        input_labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
        
        try:
            # Get segmentation from SAM
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Use the mask with highest score
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # SAM returns masks in (H, W) format, we need (W, H) to match our image orientation
            # Our image_slice is (W, H), SAM mask is (H, W), so we transpose
            self.current_mask = best_mask.T
            
            print(f"Selected mask {best_mask_idx} with score {scores[best_mask_idx]:.3f}")
            
            # Update overlay
            self._update_overlay()
            
            # Print segmentation stats
            num_pixels = np.sum(self.current_mask)
            total_pixels = self.current_mask.size
            print(f"Segmentation complete: {num_pixels} pixels ({100*num_pixels/total_pixels:.1f}% of image)")
            print("Press 'c' to confirm this segmentation or add more points to refine it.")
            
        except Exception as e:
            print(f"Segmentation error: {e}")
    
    def _update_overlay(self):
        """Update the segmentation overlay on the plot."""
        if self.current_mask is None:
            return
        
        # Remove previous overlay
        if self.mask_overlay is not None:
            self.mask_overlay.remove()
        
        print(f"Updating overlay - mask shape: {self.current_mask.shape}, image shape: {self.image_slice.shape}")
        
        # Create a simple colored overlay using masked array
        try:
            # Create an RGB image where the mask is colored
            mask_display = np.ma.masked_where(~self.current_mask.T, np.ones_like(self.current_mask.T))
            
            # Display the masked overlay
            self.mask_overlay = self.ax.imshow(mask_display, cmap='Reds', alpha=0.4, origin='lower', 
                                             extent=self.im.get_extent(), vmin=0, vmax=1)
            
            print("Overlay updated successfully")
            
        except Exception as e:
            print(f"Overlay creation failed: {e}")
            # Fallback: show mask boundary only
            try:
                # Find the boundary of the mask using simple edge detection
                mask_array = self.current_mask.astype(int)
                boundary = np.zeros_like(mask_array)
                
                # Simple boundary detection: find edges
                for i in range(1, mask_array.shape[0]-1):
                    for j in range(1, mask_array.shape[1]-1):
                        if mask_array[i,j] == 1:
                            # Check if on boundary (has at least one non-mask neighbor)
                            neighbors = mask_array[i-1:i+2, j-1:j+2]
                            if np.any(neighbors == 0):
                                boundary[i,j] = 1
                
                y_coords, x_coords = np.where(boundary)
                if len(x_coords) > 0:
                    self.mask_overlay = self.ax.scatter(x_coords, y_coords, c='red', s=1, alpha=0.8)
                    print(f"Showing mask boundary as {len(x_coords)} points")
                else:
                    print("No boundary found, mask might be empty or full")
                    
            except Exception as e2:
                print(f"Fallback boundary plot failed: {e2}")
        
        self.fig.canvas.draw()
    
    def show_and_select(self) -> np.ndarray:
        """
        Display image and wait for user segmentation.
        
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates pixels inside segmented region
        """
        plt.show()
        
        if self.mask is None:
            print("Warning: No segmentation was confirmed")
            return np.zeros(self.image_slice.shape, dtype=bool)
        
        return self.mask
    
    def get_roi_stats(self) -> dict:
        """
        Get statistics about the segmented ROI.
        
        Returns
        -------
        dict
            Dictionary containing ROI statistics
        """
        if self.mask is None:
            return {}
        
        roi_pixels = self.image_slice[self.mask]
        
        # Find bounding box
        y_indices, x_indices = np.where(self.mask)
        if len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = (x_min, x_max, y_min, y_max)
        else:
            bbox = None
        
        return {
            'num_pixels': np.sum(self.mask),
            'area_fraction': np.sum(self.mask) / self.mask.size,
            'mean_intensity': np.mean(roi_pixels) if len(roi_pixels) > 0 else 0,
            'std_intensity': np.std(roi_pixels) if len(roi_pixels) > 0 else 0,
            'min_intensity': np.min(roi_pixels) if len(roi_pixels) > 0 else 0,
            'max_intensity': np.max(roi_pixels) if len(roi_pixels) > 0 else 0,
            'bounding_box': bbox,
            'num_positive_points': len(self.positive_points),
            'num_negative_points': len(self.negative_points)
        }


def select_segmentation_roi(image_4d: np.ndarray, z_index: int, 
                           model_path: Optional[str] = None,
                           model_type: str = "vit_h") -> np.ndarray:
    """
    Interactive selection of segmented ROI using SegmentAnything on a specific slice.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    z_index : int
        Z-slice index to display for ROI selection
    model_path : str, optional
        Path to SAM model checkpoint
    model_type : str
        SAM model type ('vit_h', 'vit_l', 'vit_b')
        
    Returns
    -------
    roi_mask : np.ndarray
        Boolean mask of shape [x, y] where True indicates pixels inside segmented region
        
    Raises
    ------
    IndexError
        If z_index is out of bounds
    ImportError
        If SegmentAnything dependencies are not available
    """
    if not SAM_AVAILABLE:
        raise ImportError(
            "SegmentAnything not available. Install with:\n"
            "pip install segment-anything\n"
            "Also requires: pip install opencv-python"
        )
    
    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")
    
    # Extract slice from first timepoint
    image_slice = image_4d[:, :, z_index, 0]
    
    # Create segmentation selector
    title = f"Segmentation ROI Selection - Slice {z_index} (Timepoint 0)"
    selector = SegmentAnythingROISelector(image_slice, model_path, model_type, title)
    
    # Show and get ROI selection
    roi_mask = selector.show_and_select()
    
    # Print ROI statistics
    stats = selector.get_roi_stats()
    if stats:
        print(f"\nSegmentation ROI Statistics:")
        print(f"  Number of pixels: {stats['num_pixels']}")
        print(f"  Area fraction: {stats['area_fraction']:.1%}")
        print(f"  Mean intensity: {stats['mean_intensity']:.2f}")
        print(f"  Std intensity: {stats['std_intensity']:.2f}")
        print(f"  Min intensity: {stats['min_intensity']:.2f}")
        print(f"  Max intensity: {stats['max_intensity']:.2f}")
        print(f"  Points used: {stats['num_positive_points']} positive, {stats['num_negative_points']} negative")
        if stats['bounding_box']:
            bbox = stats['bounding_box']
            print(f"  Bounding box: ({bbox[0]}, {bbox[2]}) to ({bbox[1]}, {bbox[3]})")
    
    return roi_mask


def select_manual_contour_roi(image_4d: np.ndarray, z_index: int) -> np.ndarray:
    """
    Interactive manual contour drawing for ROI selection on a specific slice.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    z_index : int
        Z-slice index to display for ROI selection
        
    Returns
    -------
    roi_mask : np.ndarray
        Boolean mask of shape [x, y] where True indicates pixels inside drawn contour
        
    Raises
    ------
    IndexError
        If z_index is out of bounds
    """
    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")
    
    # Create manual contour selector
    title = f"Manual Contour ROI Selection (Timepoint 0)"
    selector = ManualContourROISelector(image_4d, z_index, title)
    
    # Show and get ROI selection
    roi_mask = selector.show_and_select()
    final_z_index = selector.z_index
    
    # Print ROI statistics
    stats = selector.get_roi_stats()
    if stats:
        print(f"\nManual Contour ROI Statistics (z-slice {final_z_index}):")
        print(f"  Number of pixels: {stats['num_pixels']}")
        print(f"  Area fraction: {stats['area_fraction']:.1%}")
        print(f"  Mean intensity: {stats['mean_intensity']:.2f}")
        print(f"  Std intensity: {stats['std_intensity']:.2f}")
        print(f"  Min intensity: {stats['min_intensity']:.2f}")
        print(f"  Max intensity: {stats['max_intensity']:.2f}")
        print(f"  Contour points used: {stats['num_contour_points']}")
        if stats['bounding_box']:
            bbox = stats['bounding_box']
            print(f"  Bounding box: ({bbox[0]}, {bbox[2]}) to ({bbox[1]}, {bbox[3]})")
    
    return roi_mask


def compare_roi_methods(image_4d: np.ndarray, z_index: int,
                       rectangle_mask: Optional[np.ndarray] = None,
                       segment_mask: Optional[np.ndarray] = None,
                       title: str = "ROI Method Comparison") -> None:
    """
    Compare different ROI selection methods on the same slice.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    z_index : int
        Z-slice index to display
    rectangle_mask : np.ndarray, optional
        Rectangular ROI mask
    segment_mask : np.ndarray, optional  
        Segmentation ROI mask
    title : str
        Title for the comparison plot
    """
    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")
    
    # Extract slice from first timepoint
    image_slice = image_4d[:, :, z_index, 0]
    
    # Determine subplot layout
    num_methods = 1 + sum(mask is not None for mask in [rectangle_mask, segment_mask])
    
    # Apply consistent styling
    set_proxylfit_style()
    
    fig, axes = plt.subplots(1, num_methods, figsize=(6*num_methods, 6))
    if num_methods == 1:
        axes = [axes]
    
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.05, right=0.88, wspace=0.2)
    fig.suptitle(f'ProxylFit – {title}', fontsize=14, fontweight='bold', y=0.95)
    
    # Add ProxylFit logo in bottom-right
    add_proxylfit_logo(fig, position='bottom-right')
    
    # Original image
    ax_idx = 0
    axes[ax_idx].imshow(image_slice.T, cmap='gray', origin='lower')
    axes[ax_idx].set_title("Original Image")
    axes[ax_idx].set_xlabel('X')
    axes[ax_idx].set_ylabel('Y')
    ax_idx += 1
    
    # Rectangle ROI overlay
    if rectangle_mask is not None:
        overlay = image_slice.copy()
        overlay_color = np.zeros((*image_slice.shape, 3))
        overlay_color[:, :, 0] = overlay / np.max(overlay)
        overlay_color[:, :, 1] = overlay / np.max(overlay)
        overlay_color[:, :, 2] = overlay / np.max(overlay)
        
        # Highlight ROI in blue
        overlay_color[rectangle_mask, 0] = 0.3  # Reduced red
        overlay_color[rectangle_mask, 1] = 0.3  # Reduced green
        overlay_color[rectangle_mask, 2] = 1.0  # Full blue
        
        axes[ax_idx].imshow(np.transpose(overlay_color, (1, 0, 2)), origin='lower')
        axes[ax_idx].set_title(f"Rectangle ROI\n({np.sum(rectangle_mask)} pixels)")
        axes[ax_idx].set_xlabel('X')
        axes[ax_idx].set_ylabel('Y')
        ax_idx += 1
    
    # Segmentation ROI overlay
    if segment_mask is not None:
        overlay = image_slice.copy()
        overlay_color = np.zeros((*image_slice.shape, 3))
        overlay_color[:, :, 0] = overlay / np.max(overlay)
        overlay_color[:, :, 1] = overlay / np.max(overlay)
        overlay_color[:, :, 2] = overlay / np.max(overlay)
        
        # Highlight ROI in green
        overlay_color[segment_mask, 0] = 0.3  # Reduced red
        overlay_color[segment_mask, 1] = 1.0  # Full green
        overlay_color[segment_mask, 2] = 0.3  # Reduced blue
        
        axes[ax_idx].imshow(np.transpose(overlay_color, (1, 0, 2)), origin='lower')
        axes[ax_idx].set_title(f"Segmentation ROI\n({np.sum(segment_mask)} pixels)")
        axes[ax_idx].set_xlabel('X')
        axes[ax_idx].set_ylabel('Y')
        ax_idx += 1
    
    plt.suptitle(f"{title} - Slice {z_index}")
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    if rectangle_mask is not None and segment_mask is not None:
        rect_pixels = np.sum(rectangle_mask)
        seg_pixels = np.sum(segment_mask)
        overlap = np.sum(rectangle_mask & segment_mask)
        union = np.sum(rectangle_mask | segment_mask)
        jaccard = overlap / union if union > 0 else 0
        
        print(f"\nROI Comparison Statistics:")
        print(f"  Rectangle ROI: {rect_pixels} pixels")
        print(f"  Segmentation ROI: {seg_pixels} pixels")
        print(f"  Overlap: {overlap} pixels")
        print(f"  Jaccard Index (IoU): {jaccard:.3f}")
        print(f"  Size ratio (seg/rect): {seg_pixels/rect_pixels:.2f}" if rect_pixels > 0 else "")


def get_available_roi_modes() -> List[str]:
    """
    Get list of available ROI selection modes based on installed dependencies.
    
    Returns
    -------
    List[str]
        List of available modes ('rectangle', 'contour', 'segment')
    """
    modes = ['rectangle', 'contour']  # Always available
    
    if SAM_AVAILABLE and CV2_AVAILABLE:
        modes.append('segment')
    
    return modes


def print_roi_mode_info() -> None:
    """Print information about available ROI selection modes."""
    modes = get_available_roi_modes()
    
    print("Available ROI Selection Modes:")
    print("  • rectangle: Interactive rectangular bounding box selection")
    print("  • contour: Manual contour drawing (drag to draw tumor outline) - DEFAULT")
    
    if 'segment' in modes:
        print("  • segment: SegmentAnything-based tumor segmentation (point-and-click)")
    else:
        print("  • segment: NOT AVAILABLE (requires: pip install segment-anything opencv-python)")
    
    print(f"\nCurrently available: {', '.join(modes)}")