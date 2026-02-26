"""
Shared fixtures for ProxylFit test suite.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def qt_app():
    """Session-scoped Qt application."""
    from proxyl_analysis.ui.styles import init_qt_app
    return init_qt_app()


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory that is cleaned up after test."""
    return tmp_path


@pytest.fixture
def screenshot_dir():
    """Directory for screenshot audit trail."""
    d = Path(__file__).parent / "screenshots"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def sample_4d():
    """Synthetic 4D data: 32x32x3x20 with realistic signal."""
    np.random.seed(42)
    data = np.zeros((32, 32, 3, 20), dtype=np.float32)
    x, y = np.meshgrid(np.arange(32), np.arange(32))
    spatial = (x + y) / 64.0

    for t in range(20):
        base = 5000 + t * 200
        data[:, :, :, t] = base + spatial[:, :, np.newaxis] * (50 + t * 10)

    return data


@pytest.fixture
def sample_roi_mask():
    """Circular ROI mask centered in a 32x32x3 volume."""
    mask = np.zeros((32, 32, 3), dtype=bool)
    y, x = np.ogrid[:32, :32]
    circle = (x - 16)**2 + (y - 16)**2 <= 8**2
    for z in range(3):
        mask[:, :, z] = circle
    return mask


@pytest.fixture
def sample_spacing():
    """Standard voxel spacing."""
    return (1.0, 1.0, 2.0)


@pytest.fixture
def sample_time_array():
    """Time array for 20 timepoints at 70s intervals in minutes."""
    return np.arange(20, dtype=float) * (70.0 / 60.0)


@pytest.fixture
def sample_fit_results():
    """Realistic fit_results dict as produced by fit_proxyl_kinetics."""
    return {
        'kb': 0.98, 'kd': 0.035, 'knt': 0.008,
        'A0': 5200.0, 'A1': 2800.0, 'A2': -400.0,
        't0': 4.3, 'tmax': 30.0,
        'kb_error': 0.697, 'kd_error': 0.010, 'knt_error': 0.003,
        'A0_error': 48.5, 'A1_error': 185.3, 'A2_error': 95.0,
        't0_error': 1.52, 'tmax_error': 4.8,
        'r_squared': 0.76, 'rmse': 163.5,
        'residuals': np.random.normal(0, 120, 126),
        'time_units': 'minutes',
    }


@pytest.fixture
def sample_param_maps():
    """Synthetic parameter maps dict for ParameterMapResultsDialog tests."""
    np.random.seed(42)
    shape = (16, 16, 3)
    return {
        'kb_map': np.random.random(shape).astype(np.float32) * 0.5,
        'kd_map': np.random.random(shape).astype(np.float32) * 0.1,
        'knt_map': np.random.random(shape).astype(np.float32) * 0.02,
        'r_squared_map': np.random.uniform(0.5, 1.0, shape).astype(np.float32),
        'a1_amplitude_map': np.random.random(shape).astype(np.float32) * 3000,
        'a2_amplitude_map': np.random.random(shape).astype(np.float32) * 500,
        'mask': np.ones(shape, dtype=bool),
        'metadata': {
            'kernel_type': 'sliding_window',
            'window_x': 15, 'window_y': 15, 'window_z': 3,
        },
        'reference_slice': np.random.random(shape).astype(np.float32) * 5000,
    }


@pytest.fixture
def sample_t2_volume():
    """Synthetic 3D T2 volume (32x32x3)."""
    np.random.seed(99)
    return np.random.random((32, 32, 3)).astype(np.float32) * 3000


@pytest.fixture
def real_session_path():
    """Path to a real registered session, or skip if unavailable."""
    candidates = [
        Path(__file__).parent.parent / "output" / "35354281",
        Path(__file__).parent.parent / "output" / "35354296",
        Path(__file__).parent.parent / "output" / "35354333",
    ]
    for p in candidates:
        if (p / "registered" / "dicoms" / "z00_t000.dcm").exists():
            return p
    pytest.skip("No real session data available")


@pytest.fixture
def sample_param_maps_full():
    """Full parameter maps dict matching exact output of create_parameter_maps().

    Includes ALL map keys, numpy-typed metadata values (np.int64, np.float64),
    and an roi_mask — exercises JSON serialization edge cases.
    """
    np.random.seed(42)
    shape = (16, 16, 3)
    mask = np.random.random(shape) > 0.3

    return {
        'kb_map': np.random.random(shape).astype(np.float64) * 0.5 * mask,
        'kd_map': np.random.random(shape).astype(np.float64) * 0.1 * mask,
        'knt_map': np.random.random(shape).astype(np.float64) * 0.02 * mask,
        'r_squared_map': (np.random.uniform(0.5, 1.0, shape) * mask).astype(np.float64),
        'a1_amplitude_map': np.random.random(shape).astype(np.float64) * 3000 * mask,
        'a2_amplitude_map': (np.random.random(shape).astype(np.float64) - 0.5) * 500 * mask,
        'baseline_map': np.random.random(shape).astype(np.float64) * 5000 + 3000,
        't0_map': np.random.random(shape).astype(np.float64) * 10 * mask,
        'tmax_map': np.random.random(shape).astype(np.float64) * 40 * mask,
        'mask': mask,
        'roi_mask': np.random.random((16, 16)) > 0.4,
        'metadata': {
            'window_size': np.int64(5),
            'window_x': np.int64(5),
            'window_y': np.int64(5),
            'window_z': np.int64(1),
            'z_slice': np.int64(4),
            'time_units': 'minutes',
            'signal_threshold': np.float64(523.7),
            'success_rate': np.float64(78.2),
            'processing_time': np.float64(12.34),
            'total_positions': np.int64(768),
            'successful_fits': np.int64(600),
            'kernel_type': 'sliding_window',
            'injection_time_index': np.int64(3),
            'stride': np.int64(1),
        },
    }


@pytest.fixture
def sample_registration_metrics():
    """List of 3 RegistrationMetrics (reference + 2 moving timepoints).

    Values are realistic and include numpy scalars as produced by np.mean() etc.
    """
    from proxyl_analysis.registration import RegistrationMetrics

    return [
        RegistrationMetrics(
            timepoint=0,
            ncc=1.0,
            mutual_info=1.5,
            mean_squared_error=0.0,
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            optimizer_iterations=0,
            optimizer_metric_value=0.0,
            registration_time=0.0,
        ),
        RegistrationMetrics(
            timepoint=1,
            ncc=0.98,
            mutual_info=1.42,
            mean_squared_error=12.5,
            translation=(0.12, -0.05, 0.03),
            rotation=(0.001, -0.002, 0.0005),
            optimizer_iterations=25,
            optimizer_metric_value=-1.38,
            registration_time=2.1,
        ),
        RegistrationMetrics(
            timepoint=2,
            ncc=0.97,
            mutual_info=1.39,
            mean_squared_error=18.3,
            translation=(0.25, -0.10, 0.08),
            rotation=(0.002, -0.003, 0.001),
            optimizer_iterations=30,
            optimizer_metric_value=-1.35,
            registration_time=2.4,
        ),
    ]


@pytest.fixture(scope="session")
def real_session_data():
    """Session-scoped fixture: loads real session 35354296 data once.

    Returns dict with registered_4d, spacing, metrics, T2, parameter maps,
    ROI mask, time array, and ROI signal. Skips if session unavailable.
    """
    from proxyl_analysis.registration import load_registration_data
    from proxyl_analysis.io import load_registered_t2
    from proxyl_analysis.roi_selection import compute_roi_timeseries
    from proxyl_analysis.run_analysis import create_time_array

    # Find a real session with parameter maps
    candidates = [
        Path(__file__).parent.parent / "output" / "35354296",
        Path(__file__).parent.parent / "output" / "35354281",
        Path(__file__).parent.parent / "output" / "35354333",
    ]
    session_path = None
    for p in candidates:
        if (p / "registered" / "dicoms" / "z00_t000.dcm").exists():
            session_path = p
            break

    if session_path is None:
        pytest.skip("No real session data available")

    # Load registered 4D data
    registered_4d, spacing, metrics = load_registration_data(str(session_path))

    # Load registered T2
    registered_t2, t2_spacing = load_registered_t2(str(session_path))

    # Load parameter maps
    npz_path = session_path / "parameter_maps" / "parameter_maps.npz"
    meta_path = session_path / "parameter_maps" / "parameter_maps_metadata.json"
    if npz_path.exists():
        npz = np.load(str(npz_path), allow_pickle=True)
        param_maps = {k: npz[k] for k in npz.keys()}
    else:
        param_maps = None

    if meta_path.exists():
        with open(meta_path, 'r') as f:
            param_metadata = json.load(f)
    else:
        param_metadata = {}

    # Extract ROI mask from parameter maps
    roi_mask_2d = param_maps['roi_mask'] if param_maps and 'roi_mask' in param_maps else None

    # Create time array
    num_tp = registered_4d.shape[3]
    time_array = create_time_array(num_tp, 'minutes')

    # Compute ROI signal (use roi_mask if available, else center region)
    if roi_mask_2d is not None and roi_mask_2d.sum() > 0:
        roi_signal = compute_roi_timeseries(registered_4d, roi_mask_2d)
    else:
        # Fallback: center 10x10 region
        cx, cy = registered_4d.shape[0] // 2, registered_4d.shape[1] // 2
        fallback_mask = np.zeros(registered_4d.shape[:2], dtype=bool)
        fallback_mask[cx-5:cx+5, cy-5:cy+5] = True
        roi_signal = compute_roi_timeseries(registered_4d, fallback_mask)
        roi_mask_2d = fallback_mask

    return {
        'session_path': str(session_path),
        'registered_4d': registered_4d,
        'spacing': spacing,
        'metrics': metrics,
        'registered_t2': registered_t2,
        't2_spacing': t2_spacing,
        'param_maps': param_maps,
        'param_metadata': param_metadata,
        'roi_mask_2d': roi_mask_2d,
        'time_array': time_array,
        'roi_signal': roi_signal,
    }
