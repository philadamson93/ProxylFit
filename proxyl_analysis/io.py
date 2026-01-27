"""
DICOM I/O module for loading and reshaping time-resolved MRI data.

Includes functions for:
- Loading multi-frame DICOM files
- Dataset directory management
- DICOM writing for registered and derived images
"""

import json
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import SimpleITK as sitk


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


# =============================================================================
# Dataset Directory Management (T011)
# =============================================================================

def create_dataset_directory(output_base: str, dataset_name: str) -> Path:
    """
    Create and initialize a dataset directory structure.

    Parameters
    ----------
    output_base : str
        Base output directory (e.g., './output')
    dataset_name : str
        Name for the dataset (typically DICOM filename stem)

    Returns
    -------
    Path
        Path to the created dataset directory
    """
    dataset_dir = Path(output_base) / dataset_name

    # Create main directory and subdirectories
    subdirs = ['registered', 'registered/dicoms', 'roi_analysis',
               'parameter_maps', 'parameter_maps/nifti', 'derived_images']

    for subdir in subdirs:
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    return dataset_dir


def get_dataset_path(dataset_dir: str, subdir: str = None) -> Path:
    """
    Get path within dataset directory, creating subdirs as needed.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    subdir : str, optional
        Subdirectory within dataset (e.g., 'registered', 'roi_analysis')

    Returns
    -------
    Path
        Full path to the requested location
    """
    path = Path(dataset_dir)
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset_manifest(dataset_dir: str) -> Dict[str, Any]:
    """
    Load dataset manifest, or return empty manifest if not exists.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory

    Returns
    -------
    dict
        Manifest data or empty dict with basic structure
    """
    manifest_path = Path(dataset_dir) / "dataset_manifest.json"

    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)

    # Return empty manifest structure
    return {
        "dataset_name": Path(dataset_dir).name,
        "created_at": None,
        "updated_at": None,
        "source": {},
        "analysis": {
            "registration": {"completed": False},
            "roi_analysis": {"completed": False},
            "parameter_maps": {"completed": False},
            "derived_images": {"averaged_images": [], "difference_images": []}
        },
        "files": {}
    }


def save_dataset_manifest(dataset_dir: str, manifest: Dict[str, Any]) -> None:
    """
    Save dataset manifest to JSON file.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    manifest : dict
        Manifest data to save
    """
    manifest_path = Path(dataset_dir) / "dataset_manifest.json"
    manifest["updated_at"] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def update_manifest_analysis(dataset_dir: str, analysis_type: str, data: Dict[str, Any]) -> None:
    """
    Update manifest with analysis completion info.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    analysis_type : str
        Type of analysis ('registration', 'roi_analysis', 'parameter_maps', 'image_tools')
    data : dict
        Data to merge into the analysis section
    """
    manifest = load_dataset_manifest(dataset_dir)

    if analysis_type not in manifest["analysis"]:
        manifest["analysis"][analysis_type] = {}

    manifest["analysis"][analysis_type].update(data)
    manifest["analysis"][analysis_type]["completed"] = True
    manifest["analysis"][analysis_type]["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')

    save_dataset_manifest(dataset_dir, manifest)


def save_roi_mask(dataset_dir: str, roi_mask: np.ndarray, roi_info: Dict[str, Any] = None) -> Path:
    """
    Save ROI mask to dataset directory.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    roi_mask : np.ndarray
        Binary ROI mask
    roi_info : dict, optional
        Additional ROI metadata (type, z_slice, etc.)

    Returns
    -------
    Path
        Path to saved ROI file
    """
    roi_path = get_dataset_path(dataset_dir, 'roi_analysis') / 'roi_mask.npz'

    save_data = {'roi_mask': roi_mask}
    if roi_info:
        save_data['roi_info'] = roi_info

    np.savez_compressed(roi_path, **save_data)
    return roi_path


def load_roi_mask(dataset_dir: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load ROI mask from dataset directory.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory

    Returns
    -------
    roi_mask : np.ndarray or None
        Binary ROI mask, or None if not found
    roi_info : dict or None
        ROI metadata, or None if not found
    """
    roi_path = Path(dataset_dir) / 'roi_analysis' / 'roi_mask.npz'

    if not roi_path.exists():
        return None, None

    data = np.load(roi_path, allow_pickle=True)
    roi_mask = data['roi_mask']
    roi_info = data['roi_info'].item() if 'roi_info' in data else None

    return roi_mask, roi_info


# =============================================================================
# DICOM Writing Functions (T010, T012)
# =============================================================================

def _generate_uid() -> str:
    """Generate a unique DICOM UID."""
    import pydicom.uid
    return pydicom.uid.generate_uid()


def _copy_dicom_metadata(source_path: str) -> Dict[str, Any]:
    """
    Extract relevant metadata from source DICOM for copying.

    Parameters
    ----------
    source_path : str
        Path to source DICOM file

    Returns
    -------
    dict
        Dictionary of DICOM metadata to copy
    """
    try:
        import pydicom
        ds = pydicom.dcmread(source_path, force=True)

        metadata = {}

        # Patient info
        if hasattr(ds, 'PatientID'):
            metadata['PatientID'] = str(ds.PatientID)
        if hasattr(ds, 'PatientName'):
            metadata['PatientName'] = str(ds.PatientName)

        # Study info
        if hasattr(ds, 'StudyInstanceUID'):
            metadata['StudyInstanceUID'] = str(ds.StudyInstanceUID)
        if hasattr(ds, 'StudyDate'):
            metadata['StudyDate'] = str(ds.StudyDate)
        if hasattr(ds, 'StudyDescription'):
            metadata['StudyDescription'] = str(ds.StudyDescription)

        # Series info (will generate new)
        if hasattr(ds, 'SeriesNumber'):
            metadata['OriginalSeriesNumber'] = int(ds.SeriesNumber)

        return metadata

    except Exception:
        return {}


def save_registered_as_dicom_series(
    registered_4d: np.ndarray,
    spacing: Tuple[float, float, float],
    output_dir: str,
    series_description: str = "Registered T1 DCE",
    source_dicom: str = None
) -> str:
    """
    Save registered 4D data as a series of 2D DICOM files.

    Each file is a single 2D slice. Files are organized as z{ZZ}_t{TTT}.dcm
    where slices iterate through all timepoints before moving to the next z-slice.

    Parameters
    ----------
    registered_4d : np.ndarray
        4D array with shape [x, y, z, t]
    spacing : tuple
        Voxel spacing (x, y, z) in mm
    output_dir : str
        Output directory (will create 'registered/dicoms' subdirectory)
    series_description : str
        DICOM SeriesDescription tag value
    source_dicom : str, optional
        Path to source DICOM to copy patient/study metadata from

    Returns
    -------
    str
        Path to the created DICOM series directory
    """
    import pydicom
    from pydicom.dataset import FileDataset

    # Create output directory
    dicom_dir = get_dataset_path(output_dir, 'registered/dicoms')

    x_dim, y_dim, z_dim, n_timepoints = registered_4d.shape
    total_files = z_dim * n_timepoints

    # Get metadata from source if available
    source_metadata = {}
    if source_dicom:
        source_metadata = _copy_dicom_metadata(source_dicom)

    # Generate UIDs
    series_uid = _generate_uid()
    study_uid = source_metadata.get('StudyInstanceUID', _generate_uid())

    # Determine series number
    original_series = source_metadata.get('OriginalSeriesNumber', 1)
    series_number = original_series + 1000

    # Save each 2D slice as a separate DICOM file
    instance_number = 1
    for z in range(z_dim):
        for t in range(n_timepoints):
            # Get 2D slice [x, y] -> need [y, x] for DICOM
            slice_2d = registered_4d[:, :, z, t]
            slice_dicom = slice_2d.T  # Transpose to [y, x]

            # Create file with z and t in filename
            filename = f"z{z:02d}_t{t:03d}.dcm"
            filepath = dicom_dir / filename

            # Create minimal DICOM dataset
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = _generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)

            # Patient/Study info from source
            ds.PatientID = source_metadata.get('PatientID', 'ANONYMOUS')
            ds.PatientName = source_metadata.get('PatientName', 'Anonymous')
            ds.StudyInstanceUID = study_uid
            ds.StudyDate = source_metadata.get('StudyDate', time.strftime('%Y%m%d'))
            ds.StudyDescription = source_metadata.get('StudyDescription', 'DCE-MRI Study')

            # Series info
            ds.SeriesInstanceUID = series_uid
            ds.SeriesNumber = series_number
            ds.SeriesDescription = series_description

            # Instance info
            ds.SOPClassUID = pydicom.uid.MRImageStorage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.InstanceNumber = instance_number

            # Slice position info
            ds.SliceLocation = float(z * spacing[2])
            ds.ImagePositionPatient = [0.0, 0.0, float(z * spacing[2])]
            ds.InStackPositionNumber = z + 1

            # Temporal info
            ds.TemporalPositionIdentifier = t + 1
            ds.NumberOfTemporalPositions = n_timepoints

            # Image info - single 2D frame
            ds.Modality = 'MR'
            ds.ImageType = ['DERIVED', 'SECONDARY', 'REGISTERED']
            ds.Rows = y_dim
            ds.Columns = x_dim
            ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [row, col]
            ds.SliceThickness = float(spacing[2])
            ds.SpacingBetweenSlices = float(spacing[2])

            # Pixel data
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0  # Unsigned
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'

            # Scale data to uint16
            slice_min = slice_dicom.min()
            slice_max = slice_dicom.max()
            if slice_max > slice_min:
                scaled = ((slice_dicom - slice_min) / (slice_max - slice_min) * 65535).astype(np.uint16)
            else:
                scaled = np.zeros_like(slice_dicom, dtype=np.uint16)

            ds.PixelData = scaled.tobytes()

            # Rescale slope/intercept for reconstruction
            ds.RescaleSlope = (slice_max - slice_min) / 65535 if slice_max > slice_min else 1.0
            ds.RescaleIntercept = slice_min

            # Save
            ds.save_as(str(filepath))
            instance_number += 1

    # Save series info JSON
    series_info = {
        'format_version': '2.0',
        'format_type': '2d_slices',
        'file_pattern': 'z{z:02d}_t{t:03d}.dcm',
        'n_slices': z_dim,
        'n_timepoints': n_timepoints,
        'total_files': total_files,
        'shape': [x_dim, y_dim, z_dim, n_timepoints],
        'spacing': list(spacing),
        'series_uid': series_uid,
        'study_uid': study_uid,
        'source_dicom': source_dicom,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'series_description': series_description
    }

    with open(dicom_dir / 'series_info.json', 'w') as f:
        json.dump(series_info, f, indent=2)

    print(f"  Saved {total_files} DICOM files ({z_dim} slices x {n_timepoints} timepoints) to: {dicom_dir}")
    return str(dicom_dir)


def load_registered_dicom_series(series_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a registered DICOM series (2D slice format) back into 4D array.

    Parameters
    ----------
    series_dir : str
        Path to directory containing registered DICOM files (z{ZZ}_t{TTT}.dcm format)

    Returns
    -------
    registered_4d : np.ndarray
        4D array with shape [x, y, z, t]
    spacing : tuple
        Voxel spacing (x, y, z) in mm

    Raises
    ------
    FileNotFoundError
        If the directory doesn't contain valid 2D slice DICOM files
    """
    import pydicom

    series_path = Path(series_dir)

    # Check for new 2D slice format
    if not (series_path / 'z00_t00.dcm').exists():
        raise FileNotFoundError(
            f"No valid DICOM series found in {series_path}.\n"
            "Expected format: z00_t00.dcm, z00_t01.dcm, ...\n"
            "Please re-run registration to generate updated DICOM output."
        )

    # Load series info
    info_path = series_path / 'series_info.json'
    if not info_path.exists():
        raise FileNotFoundError(
            f"Missing series_info.json in {series_path}.\n"
            "Please re-run registration to generate updated DICOM output."
        )

    with open(info_path, 'r') as f:
        series_info = json.load(f)

    x_dim, y_dim, z_dim, n_timepoints = series_info['shape']
    spacing = tuple(series_info['spacing'])

    # Initialize 4D array
    registered_4d = np.zeros((x_dim, y_dim, z_dim, n_timepoints), dtype=np.float64)

    # Load each 2D slice
    for z in range(z_dim):
        for t in range(n_timepoints):
            filepath = series_path / f'z{z:02d}_t{t:03d}.dcm'
            if not filepath.exists():
                raise FileNotFoundError(f"Missing DICOM file: {filepath}")

            ds = pydicom.dcmread(str(filepath))

            # Get pixel data [y, x] and apply rescale
            pixel_data = ds.pixel_array.astype(np.float64)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_data = pixel_data * ds.RescaleSlope + ds.RescaleIntercept

            # Transpose from [y, x] to [x, y] and store
            registered_4d[:, :, z, t] = pixel_data.T

    print(f"Loaded registered DICOM series with shape: {registered_4d.shape}")
    return registered_4d, spacing


def save_derived_image_as_dicom(
    image: np.ndarray,
    output_path: str,
    operation_type: str,
    operation_params: Dict[str, Any],
    spacing: Tuple[float, float, float],
    source_dicom: str = None
) -> List[str]:
    """
    Save a derived 2D/3D image as DICOM file(s) with descriptive metadata.

    For 2D images, saves a single file. For 3D images, creates a subfolder
    named {base_name} and saves one file per z-slice with naming:
    {base_name}/{base_name}_z{ZZ}.dcm

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image array [x, y] or [x, y, z]
    output_path : str
        Base path for output DICOM file(s). For 3D images, _z{ZZ} suffix is added.
    operation_type : str
        Type of operation ('averaged' or 'difference')
    operation_params : dict
        Operation parameters (timepoints, etc.)
    spacing : tuple
        Voxel spacing (x, y, z) in mm
    source_dicom : str, optional
        Path to source DICOM for metadata

    Returns
    -------
    List[str]
        List of paths to saved DICOM file(s)
    """
    import pydicom
    from pydicom.dataset import FileDataset

    output_path = Path(output_path)
    base_name = output_path.stem
    parent_dir = output_path.parent

    # Get source metadata
    source_metadata = {}
    if source_dicom:
        source_metadata = _copy_dicom_metadata(source_dicom)

    # Build series description
    if operation_type == 'averaged':
        start = operation_params.get('start_idx', 0)
        end = operation_params.get('end_idx', 0)
        n_frames = end - start + 1
        series_desc = f"Averaged t{start}-t{end} ({n_frames} frames)"
        series_offset = 2000
    else:  # difference
        a_start = operation_params.get('region_a_start', 0)
        a_end = operation_params.get('region_a_end', 0)
        b_start = operation_params.get('region_b_start', 0)
        b_end = operation_params.get('region_b_end', 0)
        series_desc = f"Difference t{a_start}-t{a_end} minus t{b_start}-t{b_end}"
        series_offset = 3000

    # Determine if 2D or 3D
    if image.ndim == 2:
        z_dim = 1
        slices = [image]  # Single slice
    else:
        z_dim = image.shape[2]
        slices = [image[:, :, z] for z in range(z_dim)]

    # Get image dimensions
    x_dim, y_dim = slices[0].shape

    # Common metadata
    series_uid = _generate_uid()
    study_uid = source_metadata.get('StudyInstanceUID', _generate_uid())
    original_series = source_metadata.get('OriginalSeriesNumber', 1)
    series_number = original_series + series_offset

    saved_files = []

    # For 3D images, create a subfolder to hold all slices
    if z_dim > 1:
        slice_dir = parent_dir / base_name
        slice_dir.mkdir(parents=True, exist_ok=True)
    else:
        parent_dir.mkdir(parents=True, exist_ok=True)

    for z, slice_2d in enumerate(slices):
        # Determine filename
        if z_dim == 1:
            filepath = parent_dir / f"{base_name}.dcm"
        else:
            filepath = slice_dir / f"{base_name}_z{z:02d}.dcm"

        # Transpose [x, y] to [y, x] for DICOM
        slice_dicom = slice_2d.T

        # Create DICOM dataset
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = _generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Patient/Study info
        ds.PatientID = source_metadata.get('PatientID', 'ANONYMOUS')
        ds.PatientName = source_metadata.get('PatientName', 'Anonymous')
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = source_metadata.get('StudyDate', time.strftime('%Y%m%d'))

        # Series info
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.SeriesDescription = series_desc

        # Instance info
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = z + 1

        # Slice position info
        ds.SliceLocation = float(z * spacing[2])
        ds.ImagePositionPatient = [0.0, 0.0, float(z * spacing[2])]
        ds.InStackPositionNumber = z + 1

        # Image info - single 2D frame
        ds.Modality = 'MR'
        ds.ImageType = ['DERIVED', 'SECONDARY', operation_type.upper()]
        ds.Rows = y_dim
        ds.Columns = x_dim
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])

        # Store operation details in ImageComments
        ds.ImageComments = json.dumps(operation_params)
        ds.DerivationDescription = f"{operation_type.capitalize()} image"

        # Pixel data - handle negative values for difference images
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'

        slice_min = slice_dicom.min()
        slice_max = slice_dicom.max()

        if slice_min < 0:
            # Use signed representation for difference images
            ds.PixelRepresentation = 1  # Signed
            if slice_max > slice_min:
                scale = 32767 / max(abs(slice_min), abs(slice_max))
                scaled = (slice_dicom * scale).astype(np.int16)
                ds.RescaleSlope = 1.0 / scale
                ds.RescaleIntercept = 0.0
            else:
                scaled = np.zeros_like(slice_dicom, dtype=np.int16)
                ds.RescaleSlope = 1.0
                ds.RescaleIntercept = 0.0
        else:
            ds.PixelRepresentation = 0  # Unsigned
            if slice_max > slice_min:
                scaled = ((slice_dicom - slice_min) / (slice_max - slice_min) * 65535).astype(np.uint16)
                ds.RescaleSlope = (slice_max - slice_min) / 65535
                ds.RescaleIntercept = slice_min
            else:
                scaled = np.zeros_like(slice_dicom, dtype=np.uint16)
                ds.RescaleSlope = 1.0
                ds.RescaleIntercept = 0.0

        ds.PixelData = scaled.tobytes()

        # Save
        ds.save_as(str(filepath))
        saved_files.append(str(filepath))

    return saved_files


def save_parameter_map_as_dicom(
    param_map: np.ndarray,
    map_name: str,
    output_dir: str,
    spacing: Tuple[float, float, float],
    source_dicom: str = None,
    metadata: Dict[str, Any] = None
) -> List[str]:
    """
    Save a parameter map as DICOM file(s) with one file per z-slice.

    Creates a subfolder named {map_name} containing all slices:
    {output_dir}/{map_name}/{map_name}_z{ZZ}.dcm

    Parameters
    ----------
    param_map : np.ndarray
        3D parameter map array [x, y, z]
    map_name : str
        Name of the parameter (e.g., 'kb_map', 'kd_map')
    output_dir : str
        Base output directory
    spacing : tuple
        Voxel spacing (x, y, z) in mm
    source_dicom : str, optional
        Path to source DICOM for metadata
    metadata : dict, optional
        Additional metadata about the parameter mapping

    Returns
    -------
    List[str]
        List of paths to saved DICOM file(s)
    """
    import pydicom
    from pydicom.dataset import FileDataset

    output_path = Path(output_dir)

    # Create subfolder for this parameter map
    map_dir = output_path / map_name
    map_dir.mkdir(parents=True, exist_ok=True)

    # Get source metadata
    source_metadata = {}
    if source_dicom:
        source_metadata = _copy_dicom_metadata(source_dicom)

    # Build series description
    map_descriptions = {
        'kb_map': 'Buildup Rate (kb)',
        'kd_map': 'Decay Rate (kd)',
        'knt_map': 'Non-tracer Rate (knt)',
        'r_squared_map': 'R-squared (fit quality)',
        'a1_amplitude_map': 'Tracer Amplitude (A1)',
        'a2_amplitude_map': 'Non-tracer Amplitude (A2)',
        'baseline_map': 'Baseline (A0)',
        't0_map': 'Tracer Onset (t0)',
        'tmax_map': 'Non-tracer Onset (tmax)'
    }
    series_desc = f"ParamMap: {map_descriptions.get(map_name, map_name)}"

    # Ensure 3D
    if param_map.ndim == 2:
        param_map = param_map[:, :, np.newaxis]

    z_dim = param_map.shape[2]
    x_dim, y_dim = param_map.shape[0], param_map.shape[1]

    # Series offset based on map type for unique series numbers
    map_offsets = {
        'kb_map': 4000, 'kd_map': 4100, 'knt_map': 4200,
        'r_squared_map': 4300, 'a1_amplitude_map': 4400, 'a2_amplitude_map': 4500,
        'baseline_map': 4600, 't0_map': 4700, 'tmax_map': 4800
    }
    series_offset = map_offsets.get(map_name, 4000)

    # Common metadata
    series_uid = _generate_uid()
    study_uid = source_metadata.get('StudyInstanceUID', _generate_uid())
    original_series = source_metadata.get('OriginalSeriesNumber', 1)
    series_number = original_series + series_offset

    saved_files = []

    for z in range(z_dim):
        slice_2d = param_map[:, :, z]

        # Determine filename
        if z_dim == 1:
            filepath = map_dir / f"{map_name}.dcm"
        else:
            filepath = map_dir / f"{map_name}_z{z:02d}.dcm"

        # Transpose [x, y] to [y, x] for DICOM
        slice_dicom = slice_2d.T

        # Create DICOM dataset
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = _generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Patient/Study info
        ds.PatientID = source_metadata.get('PatientID', 'ANONYMOUS')
        ds.PatientName = source_metadata.get('PatientName', 'Anonymous')
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = source_metadata.get('StudyDate', time.strftime('%Y%m%d'))

        # Series info
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = series_number
        ds.SeriesDescription = series_desc

        # Instance info
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = z + 1

        # Slice position info
        ds.SliceLocation = float(z * spacing[2])
        ds.ImagePositionPatient = [0.0, 0.0, float(z * spacing[2])]
        ds.InStackPositionNumber = z + 1

        # Image info
        ds.Modality = 'MR'
        ds.ImageType = ['DERIVED', 'SECONDARY', 'PARAMETER_MAP']
        ds.Rows = y_dim
        ds.Columns = x_dim
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])

        # Store metadata in ImageComments
        if metadata:
            ds.ImageComments = json.dumps(metadata)
        ds.DerivationDescription = f"Parameter map: {map_name}"

        # Pixel data - handle NaN and negative values
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'

        # Replace NaN with 0 for DICOM
        slice_clean = np.nan_to_num(slice_dicom, nan=0.0)
        slice_min = slice_clean.min()
        slice_max = slice_clean.max()

        if slice_min < 0:
            # Use signed representation
            ds.PixelRepresentation = 1
            if slice_max > slice_min:
                scale = 32767 / max(abs(slice_min), abs(slice_max))
                scaled = (slice_clean * scale).astype(np.int16)
                ds.RescaleSlope = 1.0 / scale
                ds.RescaleIntercept = 0.0
            else:
                scaled = np.zeros_like(slice_clean, dtype=np.int16)
                ds.RescaleSlope = 1.0
                ds.RescaleIntercept = 0.0
        else:
            ds.PixelRepresentation = 0
            if slice_max > slice_min:
                scaled = ((slice_clean - slice_min) / (slice_max - slice_min) * 65535).astype(np.uint16)
                ds.RescaleSlope = (slice_max - slice_min) / 65535
                ds.RescaleIntercept = slice_min
            else:
                scaled = np.zeros_like(slice_clean, dtype=np.uint16)
                ds.RescaleSlope = 1.0
                ds.RescaleIntercept = 0.0

        ds.PixelData = scaled.tobytes()

        # Save
        ds.save_as(str(filepath))
        saved_files.append(str(filepath))

    return saved_files