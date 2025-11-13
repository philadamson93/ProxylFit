"""
ProxylFit - A Python library for analyzing time-resolved MRI of rat brains after Proxyl injection.

This package provides tools for:
- Loading and reshaping DICOM data into 4D tensors
- Rigid registration of timepoint volumes
- ROI selection for kinetic analysis
- Kinetic model fitting to extract kb and kd parameters
"""

from .io import load_dicom_series
from .registration import register_timeseries, load_registration_data, save_registration_data, visualize_registration_quality
from .roi_selection import select_rectangle_roi
from .model import fit_proxyl_kinetics
from .parameter_mapping import create_parameter_maps, visualize_parameter_maps, save_parameter_maps, load_parameter_maps

__version__ = "1.0.0"
__all__ = [
    "load_dicom_series",
    "register_timeseries",
    "load_registration_data", 
    "save_registration_data",
    "visualize_registration_quality",
    "select_rectangle_roi",
    "fit_proxyl_kinetics",
    "create_parameter_maps",
    "visualize_parameter_maps",
    "save_parameter_maps",
    "load_parameter_maps"
]