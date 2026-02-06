#!/usr/bin/env python3
"""
Capture screenshots of all ProxylFit UI dialogs for visual verification.

Usage:
    uv run python scripts/capture_ui_screenshots.py

Outputs numbered PNG files to output/screenshots/.
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np

# Ensure project is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

SCREENSHOT_DIR = project_root / "output" / "screenshots"


def capture(dialog, filename):
    """Show dialog, let it render, capture screenshot, close."""
    app = QApplication.instance()
    dialog.show()
    for _ in range(8):
        app.processEvents()
        time_mod.sleep(0.05)
    pixmap = dialog.grab()
    filepath = SCREENSHOT_DIR / filename
    pixmap.save(str(filepath))
    dialog.close()
    app.processEvents()
    print(f"  {filename}")
    return filepath


def make_kinetic_signal(time):
    """Generate realistic PROXYL kinetic signal."""
    from proxyl_analysis.model import proxyl_kinetic_model_extended
    A0, A1, A2, kb, kd, knt, t0, tmax = 5200, 2800, -400, 0.98, 0.035, 0.008, 4.3, 30.0
    signal = proxyl_kinetic_model_extended(time, A0, A1, A2, kb, kd, knt, t0, tmax)
    signal += np.random.normal(0, 120, len(time))
    return signal, (A0, A1, A2, kb, kd, knt, t0, tmax)


def capture_main_menu_loaded():
    """MainMenuDialog with data loaded and ROI set."""
    from proxyl_analysis.ui.main_menu import MainMenuDialog

    time_array = np.linspace(0, 70, 126)
    registered_4d = np.random.random((32, 32, 9, 126)).astype(np.float32) * 8000 + 2000
    roi_mask = np.zeros((32, 32, 9), dtype=bool)
    roi_mask[10:22, 10:22, 3:6] = True
    signal, _ = make_kinetic_signal(time_array)

    dialog = MainMenuDialog(
        registered_4d=registered_4d,
        spacing=(0.137, 0.137, 0.75),
        time_array=time_array,
        dicom_path="/data/sample_experiment.dcm",
        output_dir="./output/sample",
        roi_state={
            'roi_mask': roi_mask,
            'roi_signal': signal,
            'injection_idx': 6,
            'injection_time': 4.3,
        }
    )
    capture(dialog, "01_main_menu_loaded.png")


def capture_main_menu_empty():
    """MainMenuDialog with no data loaded."""
    from proxyl_analysis.ui.main_menu import MainMenuDialog

    dialog = MainMenuDialog()
    capture(dialog, "02_main_menu_empty.png")


def capture_fit_results():
    """FitResultsDialog with kinetic fit."""
    from proxyl_analysis.ui.fitting import FitResultsDialog
    from proxyl_analysis.model import proxyl_kinetic_model_extended

    time = np.linspace(0, 70, 126)
    A0, A1, A2, kb, kd, knt, t0, tmax = 5200, 2800, -400, 0.98, 0.035, 0.008, 4.3, 30.0
    fitted_signal = proxyl_kinetic_model_extended(time, A0, A1, A2, kb, kd, knt, t0, tmax)
    noise = np.random.normal(0, 120, len(time))
    signal = fitted_signal + noise

    fit_results = {
        'kb': kb, 'kd': kd, 'knt': knt,
        'A0': A0, 'A1': A1, 'A2': A2,
        't0': t0, 'tmax': tmax,
        'kb_error': 0.697, 'kd_error': 0.010, 'knt_error': 0.003,
        'A0_error': 48.5, 'A1_error': 185.3, 'A2_error': 95.0,
        't0_error': 1.52, 'tmax_error': 4.8,
        'r_squared': 0.76, 'rmse': 163.5,
        'residuals': noise,
        'time_units': 'minutes',
    }

    dialog = FitResultsDialog(time, signal, fitted_signal, fit_results)
    capture(dialog, "03_fit_results.png")


def capture_image_tools_averaged():
    """ImageToolsDialog in averaged mode with preview."""
    from proxyl_analysis.ui.image_tools import ImageToolsDialog

    time_array = np.linspace(0, 40, 20)
    image_4d = np.zeros((32, 32, 5, 20), dtype=np.float32)
    for t in range(20):
        image_4d[:, :, :, t] = 5000 + t * 200 + np.random.normal(0, 100, (32, 32, 5))
    roi_mask = np.zeros((32, 32, 5), dtype=bool)
    roi_mask[10:22, 10:22, :] = True
    roi_signal = np.mean(image_4d[10:22, 10:22, 2, :], axis=(0, 1))

    dialog = ImageToolsDialog(
        image_4d=image_4d,
        time_array=time_array,
        roi_signal=roi_signal,
        time_units='minutes',
        output_dir='./output',
        initial_mode='average',
        roi_mask=roi_mask,
    )
    # Programmatically select region A
    dialog.region_a_start = 0
    dialog.region_a_end = 5
    dialog.region_a_start_spin.setValue(0)
    dialog.region_a_end_spin.setValue(5)

    capture(dialog, "04_image_tools_averaged.png")


def capture_image_tools_difference():
    """ImageToolsDialog in difference mode with both regions."""
    from proxyl_analysis.ui.image_tools import ImageToolsDialog

    time_array = np.linspace(0, 40, 20)
    image_4d = np.zeros((32, 32, 5, 20), dtype=np.float32)
    for t in range(20):
        image_4d[:, :, :, t] = 5000 + t * 200 + np.random.normal(0, 100, (32, 32, 5))
    roi_mask = np.zeros((32, 32, 5), dtype=bool)
    roi_mask[10:22, 10:22, :] = True
    roi_signal = np.mean(image_4d[10:22, 10:22, 2, :], axis=(0, 1))

    dialog = ImageToolsDialog(
        image_4d=image_4d,
        time_array=time_array,
        roi_signal=roi_signal,
        time_units='minutes',
        output_dir='./output',
        initial_mode='difference',
        roi_mask=roi_mask,
    )
    # Programmatically select both regions
    dialog.region_a_start = 0
    dialog.region_a_end = 4
    dialog.region_a_start_spin.setValue(0)
    dialog.region_a_end_spin.setValue(4)
    dialog.region_b_start = 10
    dialog.region_b_end = 15
    dialog.region_b_start_spin.setValue(10)
    dialog.region_b_end_spin.setValue(15)

    capture(dialog, "05_image_tools_difference.png")


def capture_parameter_map_results():
    """ParameterMapResultsDialog with synthetic maps."""
    from proxyl_analysis.ui.parameter_map_options import ParameterMapResultsDialog

    shape = (32, 32, 5)
    np.random.seed(42)
    mask = np.random.random(shape) > 0.3  # ~70% valid

    param_maps = {
        'kb_map': np.random.random(shape) * 0.5 * mask,
        'kd_map': np.random.random(shape) * 0.1 * mask,
        'knt_map': np.random.random(shape) * 0.01 * mask,
        'r_squared_map': (np.random.random(shape) * 0.3 + 0.6) * mask,
        'a1_amplitude_map': np.random.random(shape) * 3000 * mask,
        'a2_amplitude_map': (np.random.random(shape) * 500 - 250) * mask,
        'mask': mask,
        'metadata': {
            'kernel_type': 'sliding_window',
            'window_x': 15, 'window_y': 15, 'window_z': 3,
            'success_rate': 98.5,
            'processing_time': 1102.5,
            'total_positions': 2282,
        }
    }
    roi_mask = np.zeros(shape, dtype=bool)
    roi_mask[8:24, 8:24, 1:4] = True

    dialog = ParameterMapResultsDialog(
        param_maps=param_maps,
        spacing=(0.137, 0.137, 0.75),
        roi_mask=roi_mask,
        output_dir='./output',
    )
    capture(dialog, "06_parameter_map_results.png")


def capture_parameter_map_options():
    """ParameterMapOptionsDialog with existing ROI."""
    from proxyl_analysis.ui.parameter_map_options import ParameterMapOptionsDialog

    existing_roi = np.zeros((32, 32, 9), dtype=bool)
    existing_roi[10:22, 10:22, 3:6] = True

    dialog = ParameterMapOptionsDialog(
        max_z=8,
        current_z=4,
        existing_roi=existing_roi,
        existing_injection_idx=6,
    )
    capture(dialog, "07_parameter_map_options.png")


def capture_injection_time():
    """InjectionTimeSelectorDialog with signal and marker."""
    from proxyl_analysis.ui.injection import InjectionTimeSelectorDialog

    time = np.linspace(0, 70, 126)
    signal, _ = make_kinetic_signal(time)

    dialog = InjectionTimeSelectorDialog(
        time=time,
        signal=signal,
        time_units='minutes',
        output_dir='./output',
    )
    # Set injection marker programmatically
    dialog.injection_index = 6
    dialog.injection_marker.remove()
    dialog.injection_marker = dialog.ax.axvline(
        x=time[6], color='red', linewidth=3
    )
    dialog.info_widget.update_info(
        f"Selected Time: {time[6]:.2f} minutes\n"
        f"Index: 6\n"
        f"Signal: {signal[6]:.2f}"
    )
    dialog.canvas.draw()

    capture(dialog, "08_injection_time.png")


def capture_roi_selector():
    """ROISelectorDialog with image."""
    from proxyl_analysis.ui.roi import ROISelectorDialog

    # Create a more interesting image with some structure
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    image_slice = 5000 + 3000 * np.exp(-(x**2 + y**2) / 0.3) + np.random.normal(0, 200, (64, 64))

    dialog = ROISelectorDialog(image_slice.T, title="ROI Selection - Slice 4")
    capture(dialog, "09_roi_selector.png")


def capture_manual_contour():
    """ManualContourDialog with image."""
    from proxyl_analysis.ui.roi import ManualContourDialog

    # Create 4D with structure
    image_4d = np.zeros((64, 64, 9, 10), dtype=np.float32)
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    base = 5000 + 3000 * np.exp(-(x**2 + y**2) / 0.3)
    for z in range(9):
        for t in range(10):
            image_4d[:, :, z, t] = base.T + np.random.normal(0, 200, (64, 64))

    dialog = ManualContourDialog(image_4d, z_index=4)
    capture(dialog, "10_manual_contour.png")


def main():
    from proxyl_analysis.ui.styles import init_qt_app
    app = init_qt_app()

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capturing screenshots to {SCREENSHOT_DIR}/\n")

    capture_main_menu_loaded()
    capture_main_menu_empty()
    capture_fit_results()
    capture_image_tools_averaged()
    capture_image_tools_difference()
    capture_parameter_map_results()
    capture_parameter_map_options()
    capture_injection_time()
    capture_roi_selector()
    capture_manual_contour()

    n = len(list(SCREENSHOT_DIR.glob("*.png")))
    print(f"\nDone! {n} screenshots in {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
