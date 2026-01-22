"""
Qt-based UI module for ProxylFit.

Provides modern, responsive UI components using PySide6 with proper layout management.
Embeds matplotlib figures using FigureCanvasQTAgg for scientific visualization.
"""

# Core styles and initialization
from .styles import PROXYLFIT_STYLE, get_logo_path, init_qt_app

# Reusable components
from .components import (
    MatplotlibCanvas,
    LogoWidget,
    HeaderWidget,
    InstructionWidget,
    InfoWidget,
    ButtonBar
)

# Registration dialogs
from .registration import (
    RegistrationWorker,
    RegistrationProgressDialog,
    RegistrationReviewDialog,
    RegistrationSplashWindow,
    run_registration_with_progress,
    show_registration_review_qt
)

# ROI selection dialogs
from .roi import (
    ROISelectorDialog,
    ManualContourDialog,
    select_rectangle_roi_qt,
    select_manual_contour_roi_qt
)

# Injection time selection
from .injection import (
    InjectionTimeSelectorDialog,
    select_injection_time_qt
)

# Fit results
from .fitting import (
    FitResultsDialog,
    plot_fit_results_qt
)

# Image tools (T002/T003)
from .image_tools import (
    ImageToolsDialog,
    show_image_tools_dialog,
    compute_averaged_image,
    compute_difference_image
)

# Main menu
from .main_menu import (
    MainMenuDialog,
    show_main_menu
)

__all__ = [
    # Styles and initialization
    'PROXYLFIT_STYLE',
    'get_logo_path',
    'init_qt_app',

    # Components
    'MatplotlibCanvas',
    'LogoWidget',
    'HeaderWidget',
    'InstructionWidget',
    'InfoWidget',
    'ButtonBar',

    # Registration
    'RegistrationWorker',
    'RegistrationProgressDialog',
    'RegistrationReviewDialog',
    'RegistrationSplashWindow',
    'run_registration_with_progress',
    'show_registration_review_qt',

    # ROI
    'ROISelectorDialog',
    'ManualContourDialog',
    'select_rectangle_roi_qt',
    'select_manual_contour_roi_qt',

    # Injection
    'InjectionTimeSelectorDialog',
    'select_injection_time_qt',

    # Fitting
    'FitResultsDialog',
    'plot_fit_results_qt',

    # Image tools
    'ImageToolsDialog',
    'show_image_tools_dialog',
    'compute_averaged_image',
    'compute_difference_image',

    # Main menu
    'MainMenuDialog',
    'show_main_menu',
]
