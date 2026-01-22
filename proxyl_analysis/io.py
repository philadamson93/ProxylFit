"""
DICOM I/O module for loading and reshaping time-resolved MRI data.
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional


def _extract_robust_spacing(filepath: str) -> Tuple[float, float, float]:
    """
    Extract voxel spacing using multiple robust methods.
    
    Tries pydicom first (most reliable), then SimpleITK metadata, 
    then falls back to SimpleITK GetSpacing().
    
    Parameters
    ----------
    filepath : str
        Path to DICOM file
        
    Returns
    -------
    tuple of float
        Voxel spacing (x, y, z) in mm
    """
    
    # Method 1: Try pydicom (most reliable for DICOM tags)
    try:
        import pydicom
        ds = pydicom.dcmread(filepath, force=True)
        
        x_spacing = None
        y_spacing = None
        z_spacing = None
        
        # Check standard PixelSpacing
        if hasattr(ds, 'PixelSpacing') and len(ds.PixelSpacing) >= 2:
            y_spacing = float(ds.PixelSpacing[0])  # Row spacing (Y)
            x_spacing = float(ds.PixelSpacing[1])  # Column spacing (X)
            
        # Check SliceThickness
        if hasattr(ds, 'SliceThickness'):
            z_spacing = float(ds.SliceThickness)
        elif hasattr(ds, 'SpacingBetweenSlices'):
            z_spacing = float(ds.SpacingBetweenSlices)
        
        # Check multi-frame DICOM (Enhanced DICOM)
        if hasattr(ds, 'SharedFunctionalGroupsSequence'):
            for group in ds.SharedFunctionalGroupsSequence:
                if hasattr(group, 'PixelMeasuresSequence'):
                    pm = group.PixelMeasuresSequence[0]
                    if hasattr(pm, 'PixelSpacing') and len(pm.PixelSpacing) >= 2:
                        y_spacing = float(pm.PixelSpacing[0])
                        x_spacing = float(pm.PixelSpacing[1])
                    if hasattr(pm, 'SliceThickness'):
                        z_spacing = float(pm.SliceThickness)
        
        # Default Z spacing if not found
        if z_spacing is None:
            z_spacing = 1.0
            
        if x_spacing is not None and y_spacing is not None:
            spacing = (x_spacing, y_spacing, z_spacing)
            return spacing
            
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Method 2: Try SimpleITK metadata extraction
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Try to get spacing from DICOM tags
        pixel_spacing_tag = "0028|0030"  # PixelSpacing
        if reader.HasMetaDataKey(pixel_spacing_tag):
            ps_str = reader.GetMetaData(pixel_spacing_tag)
            parts = ps_str.split('\\\\')
            if len(parts) >= 2:
                y_spacing = float(parts[0])  # Row spacing
                x_spacing = float(parts[1])  # Column spacing
                
                # Get Z spacing
                z_spacing = 1.0  # Default
                slice_thickness_tag = "0018|0050"  # SliceThickness
                if reader.HasMetaDataKey(slice_thickness_tag):
                    z_spacing = float(reader.GetMetaData(slice_thickness_tag))
                
                spacing = (x_spacing, y_spacing, z_spacing)
                return spacing
                
    except Exception as e:
        pass
    
    # Method 3: Fall back to SimpleITK GetSpacing()
    try:
        image = sitk.ReadImage(filepath)
        spacing = image.GetSpacing()
        return spacing
        
    except Exception as e:
        pass
        # Final fallback
        return (1.0, 1.0, 1.0)


def load_dicom_series(filepath: str, override_spacing: Tuple[float, float, float] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a single multi-frame DICOM file and reshape to 4D tensor.
    
    The DICOM contains time-resolved data where the time dimension is encoded 
    across repeated stacks of 9 z-slices.
    
    Parameters
    ----------
    filepath : str
        Path to the DICOM file
    override_spacing : tuple of float, optional
        Manual voxel spacing (x, y, z) to use if DICOM metadata is missing/incorrect
        
    Returns
    -------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t] where:
        - x, y: spatial dimensions
        - z: slice dimension (9 slices)
        - t: time dimension
    spacing : tuple of float
        Voxel spacing (x, y, z) from DICOM metadata or override
        
    Raises
    ------
    ValueError
        If the total number of slices is not divisible by 9
    FileNotFoundError
        If the DICOM file cannot be found
    """
    try:
        # Load DICOM using SimpleITK
        image = sitk.ReadImage(filepath)
        
        # Get voxel spacing using robust extraction
        spacing = _extract_robust_spacing(filepath)
        
        # Check if spacing is default (1,1,1) - indicates missing DICOM tags
        if spacing == (1.0, 1.0, 1.0):
            print("⚠️  WARNING: SimpleITK returned default spacing (1,1,1)")
            print("   This suggests DICOM metadata is missing or not readable")
        
        
        # Handle spacing override
        if override_spacing is not None:
            spacing = override_spacing
            print(f"✅ Using manual spacing override: {spacing}")
        elif spacing == (1.0, 1.0, 1.0):
            print("\n🔧 SPACING OVERRIDE NEEDED:")
            print("SimpleITK could not read correct spacing from DICOM metadata.")
            print("If you know the correct voxel spacing, modify the function call:")
            print("Example: load_dicom_series(filepath, override_spacing=(0.13, 0.13, 1.0))")
            print("\nFor now, using default spacing - translations will be INCORRECT")
            print("Expected: actual_translation = reported_translation * (actual_spacing / 1.0)")
            print(f"If actual spacing is 0.13mm: actual_translation = reported * 0.13")
            print()
        
        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(image)
        
        # Get dimensions
        total_slices = image_array.shape[0]
        height = image_array.shape[1]
        width = image_array.shape[2]
        
        # Calculate time points
        z_slices = 9
        if total_slices % z_slices != 0:
            raise ValueError(
                f"Total slices ({total_slices}) is not divisible by {z_slices}. "
                f"Expected data with {z_slices} z-slices per timepoint."
            )
        
        t_points = total_slices // z_slices
        
        # Reshape from [total_slices, y, x] to [x, y, z, t]
        # SimpleITK returns arrays as [z, y, x], so we need to transpose
        image_4d = image_array.reshape(t_points, z_slices, height, width)
        
        # Transpose to [x, y, z, t]
        image_4d = np.transpose(image_4d, (3, 2, 1, 0))
        
        return image_4d, spacing
        
    except Exception as e:
        if "does not exist" in str(e).lower() or "cannot find" in str(e).lower():
            raise FileNotFoundError(f"DICOM file not found: {filepath}")
        else:
            raise RuntimeError(f"Error loading DICOM file: {e}")


def get_timepoint_volume(image_4d: np.ndarray, timepoint: int) -> np.ndarray:
    """
    Extract a single timepoint volume from 4D data.
    
    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    timepoint : int
        Time index to extract
        
    Returns
    -------
    volume : np.ndarray
        3D array with shape [x, y, z] for the specified timepoint
    """
    if timepoint >= image_4d.shape[3]:
        raise IndexError(f"Timepoint {timepoint} exceeds available timepoints ({image_4d.shape[3]})")
    
    return image_4d[:, :, :, timepoint]


def get_slice_at_timepoint(image_4d: np.ndarray, z_index: int, timepoint: int = 0) -> np.ndarray:
    """
    Extract a 2D slice from a specific z-index and timepoint.

    Parameters
    ----------
    image_4d : np.ndarray
        4D array with shape [x, y, z, t]
    z_index : int
        Z-slice index to extract
    timepoint : int, optional
        Time index to extract (default: 0)

    Returns
    -------
    slice_2d : np.ndarray
        2D array with shape [x, y] for the specified slice and timepoint
    """
    if z_index >= image_4d.shape[2]:
        raise IndexError(f"Z-index {z_index} exceeds available slices ({image_4d.shape[2]})")
    if timepoint >= image_4d.shape[3]:
        raise IndexError(f"Timepoint {timepoint} exceeds available timepoints ({image_4d.shape[3]})")

    return image_4d[:, :, z_index, timepoint]


def load_t2_volume(filepath: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a T2-weighted DICOM volume (single 3D volume, not time series).

    T2 images provide better tumor volume definition (RANO criteria) and are
    more useful for segmentation and ROI selection. This function loads a T2
    volume that can be registered to the T1 series for ROI selection.

    Parameters
    ----------
    filepath : str
        Path to the T2 DICOM file

    Returns
    -------
    t2_volume : np.ndarray
        3D array with shape [x, y, z]
    spacing : tuple of float
        Voxel spacing (x, y, z) from DICOM metadata

    Raises
    ------
    FileNotFoundError
        If the DICOM file cannot be found
    """
    try:
        # Load DICOM using SimpleITK
        image = sitk.ReadImage(filepath)

        # Get voxel spacing
        spacing = _extract_robust_spacing(filepath)

        if spacing == (1.0, 1.0, 1.0):
            print("WARNING: T2 spacing could not be read from DICOM metadata")

        # Convert to numpy array
        # SimpleITK returns [z, y, x], transpose to [x, y, z]
        image_array = sitk.GetArrayFromImage(image)
        t2_volume = np.transpose(image_array, (2, 1, 0))

        print(f"Loaded T2 volume with shape: {t2_volume.shape}")
        print(f"T2 spacing: {spacing}")

        return t2_volume, spacing

    except Exception as e:
        if "does not exist" in str(e).lower() or "cannot find" in str(e).lower():
            raise FileNotFoundError(f"T2 DICOM file not found: {filepath}")
        else:
            raise RuntimeError(f"Error loading T2 DICOM file: {e}")