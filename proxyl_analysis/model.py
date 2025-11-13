"""
Kinetic modeling module for fitting Proxyl injection curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
from pathlib import Path
import os


def add_proxylfit_logo(fig, logo_path=None, zoom=0.15, position='top-right', custom_xy=None):
    """
    Add ProxylFit logo to matplotlib figure using OffsetImage for consistent placement.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add logo to
    logo_path : str, optional
        Path to logo file (auto-detected if None)
    zoom : float
        Logo zoom level (default: 0.15, roughly 6-8% of figure width)
    position : str
        Logo position: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
    custom_xy : tuple, optional
        Custom (x, y) position override in figure fraction coordinates
    """
    try:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.image as mpimg
        
        # Find logo file if not provided
        if logo_path is None:
            logo_path = Path(__file__).parent.parent / "proxylfit.png"
            if not logo_path.exists():
                logo_path = Path("proxylfit.png")
        
        if Path(logo_path).exists():
            logo_img = mpimg.imread(str(logo_path))
            
            # Create OffsetImage with specified zoom
            imagebox = OffsetImage(logo_img, zoom=zoom, alpha=0.8)
            
            # Use custom position if provided
            if custom_xy is not None:
                xy = custom_xy
                # Determine alignment based on position
                if xy[0] > 0.5 and xy[1] > 0.5:
                    align = (1, 1)  # top-right
                elif xy[0] <= 0.5 and xy[1] > 0.5:
                    align = (0, 1)  # top-left
                elif xy[0] > 0.5 and xy[1] <= 0.5:
                    align = (1, 0)  # bottom-right
                else:
                    align = (0, 0)  # bottom-left
            else:
                # Set position coordinates
                if position == 'top-right':
                    xy = (0.95, 0.95)
                    align = (1, 1)
                elif position == 'top-left':
                    xy = (0.05, 0.95)
                    align = (0, 1)
                elif position == 'bottom-right':
                    xy = (0.95, 0.05)
                    align = (1, 0)
                elif position == 'bottom-left':
                    xy = (0.05, 0.05)
                    align = (0, 0)
                else:
                    xy = (0.95, 0.95)  # Default to top-right
                    align = (1, 1)
            
            ab = AnnotationBbox(imagebox, xy, 
                              xycoords='figure fraction',
                              frameon=False, 
                              box_alignment=align,
                              pad=0.1)
            
            # Add to figure (not individual axes) so it persists during redraws
            fig.add_artist(ab)
            
    except Exception:
        # Silently fail if logo can't be loaded or dependencies missing
        pass


def set_proxylfit_style():
    """Apply consistent ProxylFit styling to matplotlib."""
    import matplotlib.pyplot as plt
    
    # Set consistent font and styling
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'font.weight': 'normal',
        'axes.titleweight': 'bold',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.alpha': 0.3,
        'axes.grid': True,
        'grid.linewidth': 0.5
    })


def proxyl_kinetic_model_extended(t: np.ndarray, A0: float, A1: float, A2: float, kb: float, kd: float, knt: float, t0: float, tmax: float) -> np.ndarray:
    """
    Extended Proxyl kinetic model function with non-tracer effect term.
    
    I(t) = A0 + A1*(1 - exp(-kb*(t - t0))) * exp(-kd*(t - t0)) + A2*(1 - exp(-knt*(t - tmax)))
    
    Parameters
    ----------
    t : np.ndarray
        Time points (in minutes)
    A0 : float
        Baseline signal
    A1 : float
        Tracer signal amplitude
    A2 : float
        Non-tracer effect amplitude
    kb : float
        Buildup rate constant (1/min)
    kd : float
        Decay rate constant (1/min)
    knt : float
        Non-tracer effect rate constant (1/min)
    t0 : float
        Tracer injection time offset (minutes)
    tmax : float
        Non-tracer effect onset time (minutes)
        
    Returns
    -------
    np.ndarray
        Model signal values
    """
    # Handle t < t0 and t < tmax cases
    t_shifted_tracer = np.maximum(t - t0, 0)
    t_shifted_nontracer = np.maximum(t - tmax, 0)
    
    # Compute tracer term: A1*(1 - exp(-kb*(t - t0))) * exp(-kd*(t - t0))
    tracer_uptake = 1 - np.exp(-kb * t_shifted_tracer)
    tracer_decay = np.exp(-kd * t_shifted_tracer)
    tracer_term = A1 * tracer_uptake * tracer_decay
    
    # Compute non-tracer term: A2*(1 - exp(-knt*(t - tmax)))
    nontracer_term = A2 * (1 - np.exp(-knt * t_shifted_nontracer))
    
    # Complete model
    signal = A0 + tracer_term + nontracer_term
    
    return signal


def estimate_initial_parameters_extended(time: np.ndarray, signal: np.ndarray) -> Dict[str, float]:
    """
    Estimate initial parameters for extended curve fitting.
    
    Parameters
    ----------
    time : np.ndarray
        Time points (in minutes)
    signal : np.ndarray
        Signal values
        
    Returns
    -------
    dict
        Initial parameter estimates
    """
    # tail level ~ median of last ~20% of points
    tail = signal[int(0.8*len(signal)):]
    tail_level = float(np.median(tail))

    A0_est = np.mean(signal[:max(3, int(0.05*len(signal)))])  # early baseline
    A2_est = tail_level - A0_est      # can be negative  

    # keep your A1_est as before or set from peak - baseline
    A1_est = max((np.max(signal) - A0_est) - max(A2_est, 0), 0)

    # time offsets
    t0_est   = time[np.argmax(np.diff(signal) > 0.1*np.std(np.diff(signal)))] if len(time)>2 else time[0]
    tmax_est = time[np.argmax(signal)]   # onset near peak

    # rates a bit slower/more conservative
    kb_est, kd_est, knt_est = 0.05, 0.02, 0.01
        
    return {
        'A0': A0_est,
        'A1': A1_est,
        'A2': A2_est,
        'kb': kb_est,
        'kd': kd_est,
        'knt': knt_est,
        't0': t0_est,
        'tmax': tmax_est
    }


def fit_proxyl_kinetics(time: np.ndarray, signal: np.ndarray, 
                       time_units: str = 'minutes') -> Tuple[float, float, float, np.ndarray, Dict]:
    """
    Fit extended Proxyl kinetic model to extract rate parameters.
    
    Parameters
    ----------
    time : np.ndarray
        Time points (in minutes)
    signal : np.ndarray  
        Signal intensity values
    time_units : str
        Units for time (for display purposes)
        
    Returns
    -------
    kb : float
        Buildup rate constant (1/min)
    kd : float
        Decay rate constant (1/min)
    knt : float
        Non-tracer effect rate constant (1/min)
    fitted_signal : np.ndarray
        Model-fitted signal curve
    fit_results : dict
        Complete fitting results including all parameters, errors, and fit quality
        
    Raises
    ------
    ValueError
        If fitting fails or arrays have different lengths
    """
    if len(time) != len(signal):
        raise ValueError("Time and signal arrays must have same length")
    
    if len(time) < 8:
        raise ValueError("Need at least 8 time points for model fitting")
    
    # Get initial parameter estimates
    initial_params = estimate_initial_parameters_extended(time, signal)
    

    # More reasonable bounds to prevent numerical issues
    signal_range = np.max(signal) - np.min(signal)
    
    lower_bounds = [
        0,        # A0 >= 0
        0,        # A1 >= 0
        -signal_range,  # A2 can be negative
        0.001,    # kb > 0
        0.001,    # kd > 0
        0.001,    # knt > 0
        time[0],  # t0 >= first time
        time[0]   # tmax >= first time
    ]

    upper_bounds = [
        np.max(signal) * 2,  # A0 - reasonable baseline bound
        signal_range * 3,    # A1 - reasonable amplitude bound
        signal_range,    # A2 - smaller non-tracer amplitude
        2.0,      # kb <= 2.0/min (more permissive for fast binding)
        1.0,      # kd <= 1.0/min (allow faster decay)
        0.2,      # knt <= 0.2/min (allow moderate non-tracer effects)
        time[-1], # t0 <= last time
        time[-1]  # tmax <= last time
    ]
        
    # Initial guess
    p0 = [initial_params['A0'], initial_params['A1'], initial_params['A2'],
          initial_params['kb'], initial_params['kd'], initial_params['knt'],
          initial_params['t0'], initial_params['tmax']]
    
    try:
        # First attempt with standard fitting
        popt, pcov = curve_fit(
            proxyl_kinetic_model_extended,
            time,
            signal,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,  # More iterations for extended model
            method='trf',  # Trust region reflective algorithm
            ftol=1e-8,     # Tighter tolerance
            xtol=1e-8
        )
        
        A0_fit, A1_fit, A2_fit, kb_fit, kd_fit, knt_fit, t0_fit, tmax_fit = popt
        
        # Check for critical parameters at bounds (only warn for kb, kd, knt)
        tolerance = 1e-6
        critical_bounds_hit = []
        critical_params = {'kb': 3, 'kd': 4, 'knt': 5}  # indices in popt
        
        for name, idx in critical_params.items():
            param = popt[idx]
            lower = lower_bounds[idx]
            upper = upper_bounds[idx]
            if abs(param - upper) < tolerance:  # Only warn for upper bound hits on critical params
                critical_bounds_hit.append(f"{name} at upper bound")
        
        # Only print warning if critical kinetic parameters hit bounds
        if critical_bounds_hit and len(critical_bounds_hit) <= 2:  # Limit verbosity
            print(f"Note: {', '.join(critical_bounds_hit)} - consider adjusting parameter bounds")
        
        # Calculate fitted curve
        fitted_signal = proxyl_kinetic_model_extended(time, A0_fit, A1_fit, A2_fit, 
                                                     kb_fit, kd_fit, knt_fit, 
                                                     t0_fit, tmax_fit)
        
        # Calculate parameter uncertainties with robust error handling
        try:
            # Check for numerical issues in covariance matrix
            diag_elements = np.diag(pcov)
            if np.any(diag_elements < 0) or np.any(np.isinf(diag_elements)) or np.any(np.isnan(diag_elements)):
                # Use relative error estimation if covariance is bad
                print("Warning: Covariance matrix has numerical issues. Using conservative error estimates.")
                param_errors = np.abs(popt) * 0.1  # 10% relative error as fallback
            else:
                param_errors = np.sqrt(diag_elements)
                # Cap extremely large errors at 100% of parameter value
                for i in range(len(param_errors)):
                    if param_errors[i] > abs(popt[i]) * 2:  # Error > 200% of value
                        param_errors[i] = abs(popt[i])  # Cap at 100% error
        except Exception as e:
            print(f"Warning: Error calculation failed ({e}). Using conservative estimates.")
            param_errors = np.abs(popt) * 0.1  # 10% relative error as fallback
        
        # Calculate fit quality metrics
        residuals = signal - fitted_signal
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((signal - np.mean(signal)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Compile results
        fit_results = {
            'A0': A0_fit,
            'A1': A1_fit,
            'A2': A2_fit,
            'kb': kb_fit,
            'kd': kd_fit,
            'knt': knt_fit,
            't0': t0_fit,
            'tmax': tmax_fit,
            'A0_error': param_errors[0],
            'A1_error': param_errors[1],
            'A2_error': param_errors[2], 
            'kb_error': param_errors[3],
            'kd_error': param_errors[4],
            'knt_error': param_errors[5],
            't0_error': param_errors[6],
            'tmax_error': param_errors[7],
            'r_squared': r_squared,
            'rmse': rmse,
            'residuals': residuals,
            'covariance_matrix': pcov,
            'time_units': time_units
        }
        
        return kb_fit, kd_fit, knt_fit, fitted_signal, fit_results
        
    except Exception as e:
        print(f"First fitting attempt failed: {e}")
        print("Trying alternative fitting approach with relaxed constraints...")
        
        try:
            # Fallback approach with relaxed bounds and different method
            relaxed_upper = [
                np.max(signal) * 3,  # More relaxed A0
                signal_range * 5,    # More relaxed A1
                signal_range * 3,    # More relaxed A2
                1.0,      # More relaxed kb
                0.5,      # More relaxed kd
                0.1,      # More relaxed knt
                time[-1], # t0
                time[-1]  # tmax
            ]
            
            # Try with dogbox method and looser tolerances
            popt, pcov = curve_fit(
                proxyl_kinetic_model_extended,
                time,
                signal,
                p0=p0,
                bounds=(lower_bounds, relaxed_upper),
                maxfev=2000,
                method='dogbox',  # Different algorithm
                ftol=1e-6,        # Looser tolerance
                xtol=1e-6
            )
            
            A0_fit, A1_fit, A2_fit, kb_fit, kd_fit, knt_fit, t0_fit, tmax_fit = popt
            
            print("Fallback fitting succeeded with relaxed constraints.")
            
            # Calculate fitted curve
            fitted_signal = proxyl_kinetic_model_extended(time, A0_fit, A1_fit, A2_fit, 
                                                         kb_fit, kd_fit, knt_fit, 
                                                         t0_fit, tmax_fit)
            
            # Simplified error handling for fallback
            try:
                diag_elements = np.diag(pcov)
                param_errors = np.sqrt(np.abs(diag_elements))  # Use abs to handle negative values
                # Cap all errors at 100% of parameter value
                for i in range(len(param_errors)):
                    if param_errors[i] > abs(popt[i]):
                        param_errors[i] = abs(popt[i])
            except:
                param_errors = np.abs(popt) * 0.2  # 20% relative error as fallback
            
            # Calculate fit quality metrics
            residuals = signal - fitted_signal
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((signal - np.mean(signal)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            # Compile results
            fit_results = {
                'A0': A0_fit,
                'A1': A1_fit,
                'A2': A2_fit,
                'kb': kb_fit,
                'kd': kd_fit,
                'knt': knt_fit,
                't0': t0_fit,
                'tmax': tmax_fit,
                'A0_error': param_errors[0],
                'A1_error': param_errors[1],
                'A2_error': param_errors[2], 
                'kb_error': param_errors[3],
                'kd_error': param_errors[4],
                'knt_error': param_errors[5],
                't0_error': param_errors[6],
                'tmax_error': param_errors[7],
                'r_squared': r_squared,
                'rmse': rmse,
                'residuals': residuals,
                'covariance_matrix': pcov,
                'time_units': time_units,
                'fit_method': 'fallback'  # Mark as fallback fit
            }
            
            return kb_fit, kd_fit, knt_fit, fitted_signal, fit_results
            
        except Exception as e2:
            raise ValueError(f"Both standard and fallback curve fitting failed. Standard: {str(e)}, Fallback: {str(e2)}")


def plot_fit_results(time: np.ndarray, signal: np.ndarray, fitted_signal: np.ndarray,
                    fit_results: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot the original signal, fitted curve, and residuals.
    
    Parameters
    ----------
    time : np.ndarray
        Time points
    signal : np.ndarray
        Original signal
    fitted_signal : np.ndarray
        Fitted signal curve
    fit_results : dict
        Fitting results from fit_proxyl_kinetics
    save_path : str, optional
        Path to save the plot
    """
    # Apply consistent styling
    set_proxylfit_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.88, hspace=0.3)
    
    # Add title with program name
    fig.suptitle('ProxylFit – Kinetic Model Fitting Results', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Add ProxylFit logo in bottom-right (custom position to avoid residuals plot)
    add_proxylfit_logo(fig, zoom=0.12, custom_xy=(0.95, 0.02))

    
    # Main plot: signal and fit
    ax1.plot(time, signal, 'bo-', markersize=4, linewidth=2, label='Data')
    ax1.plot(time, fitted_signal, 'r-', linewidth=2, label='Fitted Model')
    ax1.set_ylabel('Signal Intensity')
    # ax1.set_title('Extended Proxyl Kinetic Model Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add fit parameters as text
    param_text = (
        f"kb (buildup) = {fit_results['kb']:.4f} ± {fit_results['kb_error']:.4f} /{fit_results['time_units']}\n"
        f"kd (decay) = {fit_results['kd']:.4f} ± {fit_results['kd_error']:.4f} /{fit_results['time_units']}\n"
        f"knt (non-tracer) = {fit_results['knt']:.4f} ± {fit_results['knt_error']:.4f} /{fit_results['time_units']}\n"
        f"R² = {fit_results['r_squared']:.4f}\n"
        f"RMSE = {fit_results['rmse']:.2f}"
    )
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8))
    
    # Residuals plot
    ax2.plot(time, fit_results['residuals'], 'go-', markersize=3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f'Time ({fit_results["time_units"]})')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fit Residuals')
    ax2.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def print_fit_summary(fit_results: Dict) -> None:
    """
    Print a summary of the fitting results.
    
    Parameters
    ----------
    fit_results : dict
        Fitting results from fit_proxyl_kinetics
    """
    print("\n" + "="*50)
    print("PROXYL KINETIC MODEL FIT RESULTS")
    print("="*50)
    print(f"Model: I(t) = A0 + A1*(1-exp(-kb*(t-t0)))*exp(-kd*(t-t0)) + A2*(1-exp(-knt*(t-tmax)))")
    print()
    print("Fitted Parameters:")
    print(f"  A0 (baseline):           {fit_results['A0']:.3f} ± {fit_results['A0_error']:.3f}")
    print(f"  A1 (tracer amplitude):   {fit_results['A1']:.3f} ± {fit_results['A1_error']:.3f}")
    print(f"  A2 (non-tracer ampl.):   {fit_results['A2']:.3f} ± {fit_results['A2_error']:.3f}")
    print(f"  kb (buildup rate):       {fit_results['kb']:.4f} ± {fit_results['kb_error']:.4f} /{fit_results['time_units']}")
    print(f"  kd (decay rate):         {fit_results['kd']:.4f} ± {fit_results['kd_error']:.4f} /{fit_results['time_units']}")
    print(f"  knt (non-tracer rate):   {fit_results['knt']:.4f} ± {fit_results['knt_error']:.4f} /{fit_results['time_units']}")
    print(f"  t0 (tracer onset):       {fit_results['t0']:.2f} ± {fit_results['t0_error']:.2f} {fit_results['time_units']}")
    print(f"  tmax (non-tracer onset): {fit_results['tmax']:.2f} ± {fit_results['tmax_error']:.2f} {fit_results['time_units']}")
    print()
    print("Fit Quality:")
    print(f"  R-squared:         {fit_results['r_squared']:.4f}")
    print(f"  RMSE:              {fit_results['rmse']:.3f}")
    print("="*50)


def calculate_derived_parameters(kb: float, kd: float, knt: float, kb_error: float, kd_error: float, knt_error: float) -> Dict[str, float]:
    """
    Calculate derived kinetic parameters.
    
    Parameters
    ----------
    kb : float
        Buildup rate constant
    kd : float
        Decay rate constant
    knt : float
        Non-tracer effect rate constant
    kb_error : float
        Error in kb
    kd_error : float
        Error in kd
    knt_error : float
        Error in knt
        
    Returns
    -------
    dict
        Derived parameters including half-lives and ratios
    """
    # Half-lives
    t_half_buildup = np.log(2) / kb if kb > 0 else np.inf
    t_half_decay = np.log(2) / kd if kd > 0 else np.inf
    t_half_nontracer = np.log(2) / knt if knt > 0 else np.inf
    
    # Rate ratios
    rate_ratio_buildup_decay = kb / kd if kd > 0 else np.inf
    rate_ratio_buildup_nontracer = kb / knt if knt > 0 else np.inf
    
    # Error propagation for derived parameters with bounds checking
    # Cap relative errors to prevent huge error propagation
    max_rel_error = 2.0  # Maximum 200% relative error
    
    if kb > 0 and kb_error/kb < max_rel_error:
        t_half_buildup_error = t_half_buildup * (kb_error / kb)
    else:
        t_half_buildup_error = t_half_buildup * max_rel_error
        
    if kd > 0 and kd_error/kd < max_rel_error:
        t_half_decay_error = t_half_decay * (kd_error / kd)
    else:
        t_half_decay_error = t_half_decay * max_rel_error
        
    if knt > 0 and knt_error/knt < max_rel_error:
        t_half_nontracer_error = t_half_nontracer * (knt_error / knt)
    else:
        t_half_nontracer_error = t_half_nontracer * max_rel_error
    
    # Rate ratio errors using propagation of uncertainty with bounds
    if kb > 0 and kd > 0:
        rel_error_kb = min(kb_error/kb, max_rel_error)
        rel_error_kd = min(kd_error/kd, max_rel_error)
        rate_ratio_buildup_decay_error = rate_ratio_buildup_decay * np.sqrt(rel_error_kb**2 + rel_error_kd**2)
    else:
        rate_ratio_buildup_decay_error = rate_ratio_buildup_decay * max_rel_error
        
    if kb > 0 and knt > 0:
        rel_error_kb = min(kb_error/kb, max_rel_error)
        rel_error_knt = min(knt_error/knt, max_rel_error)
        rate_ratio_buildup_nontracer_error = rate_ratio_buildup_nontracer * np.sqrt(rel_error_kb**2 + rel_error_knt**2)
    else:
        rate_ratio_buildup_nontracer_error = rate_ratio_buildup_nontracer * max_rel_error
    
    return {
        'half_life_buildup': t_half_buildup,
        'half_life_decay': t_half_decay,
        'half_life_nontracer': t_half_nontracer,
        'rate_ratio_buildup_decay': rate_ratio_buildup_decay,
        'rate_ratio_buildup_nontracer': rate_ratio_buildup_nontracer,
        'half_life_buildup_error': t_half_buildup_error,
        'half_life_decay_error': t_half_decay_error,
        'half_life_nontracer_error': t_half_nontracer_error,
        'rate_ratio_buildup_decay_error': rate_ratio_buildup_decay_error,
        'rate_ratio_buildup_nontracer_error': rate_ratio_buildup_nontracer_error
    }


def select_injection_time(time: np.ndarray, signal: np.ndarray, 
                         time_units: str = 'minutes', output_dir: str = './output') -> int:
    """
    Interactive selection of injection time point from signal data.
    
    Parameters
    ----------
    time : np.ndarray
        Time points
    signal : np.ndarray
        Signal values
    time_units : str
        Units for time axis
    output_dir : str
        Directory to save CSV export
        
    Returns
    -------
    int
        Index of selected injection time
    """
    class InjectionTimeSelector:
        def __init__(self):
            self.injection_index = 0
            self.selected = False
            
            # Apply consistent styling
            set_proxylfit_style()
            
            # Create figure with padding (extra bottom margin for buttons/logo)
            self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))
            self.fig.subplots_adjust(top=0.85, bottom=0.18, left=0.1, right=0.88)
            
            # Add title with program name
            self.fig.suptitle('ProxylFit – Injection Time Selection', 
                            fontsize=14, fontweight='bold', y=0.95)
            
            # Add ProxylFit logo in bottom-right (custom lower position to avoid x-axis)
            add_proxylfit_logo(self.fig, zoom=0.12, custom_xy=(0.95, 0.03))
            
            # Add horizontal separator below title
            self.ax.axhline(y=self.ax.get_ylim()[1], color='lightgray', linewidth=1, alpha=0.5)
            
            # Plot signal
            self.line, = self.ax.plot(time, signal, 'b-o', linewidth=2, markersize=4, label='Signal')
            self.ax.set_xlabel(f'Time ({time_units})')
            self.ax.set_ylabel('Signal Intensity')
            self.ax.set_title('Click on the injection time point, then click Set Injection Time', 
                             fontsize=11)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # Set y-axis to autoscale based on data (don't start at 0)
            y_min, y_max = np.min(signal), np.max(signal)
            y_range = y_max - y_min
            self.ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            
            # Initial injection marker at t=0
            self.injection_marker = self.ax.axvline(x=time[0], color='red', linewidth=3, 
                                                  label=f'Injection time: {time[0]:.1f} {time_units}')
            
            # Add text showing current selection (moved outside plot area)
            self.info_text = self.ax.text(1.02, 0.95, 
                                        f"Injection time: {time[0]:.1f} {time_units}\n"
                                        f"Index: {0}\n"
                                        f"Click plot to change", 
                                        transform=self.ax.transAxes, 
                                        verticalalignment='top',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            # Connect click and key events
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            # Add buttons with improved spacing and centering
            from matplotlib.widgets import Button
            
            # Center buttons at bottom with consistent spacing (moved down)
            button_width = 0.15
            button_height = 0.05
            button_y = 0.04  # Lowered further per UI request
            button_spacing = 0.05
            
            # Calculate centered positions
            total_width = 2 * button_width + button_spacing
            start_x = (1.0 - total_width) / 2
            
            ax_export = plt.axes([start_x, button_y, button_width, button_height])
            ax_accept = plt.axes([start_x + button_width + button_spacing, button_y, button_width, button_height])
            
            self.btn_export = Button(ax_export, 'Export CSV')
            self.btn_accept = Button(ax_accept, 'Set Injection Time')
            
            # Style buttons consistently
            self.btn_export.label.set_fontsize(10)
            self.btn_accept.label.set_fontsize(10)
            self.btn_accept.label.set_color('green')
            self.btn_accept.label.set_weight('bold')
            
            self.btn_export.on_clicked(self._export_csv)
            self.btn_accept.on_clicked(self._accept_injection_time)
            
            self.accepted = False
            
            # plt.tight_layout()
            self.fig.subplots_adjust(bottom=0.22)

            
        def _on_click(self, event):
            """Handle clicks to select injection time."""
            if event.inaxes == self.ax and event.button == 1:
                clicked_time = event.xdata
                
                # Find closest time point
                closest_idx = np.argmin(np.abs(time - clicked_time))
                self.injection_index = closest_idx
                
                # Update marker
                self.injection_marker.remove()
                self.injection_marker = self.ax.axvline(x=time[closest_idx], color='red', linewidth=3, 
                                                      label=f'Injection time: {time[closest_idx]:.1f} {time_units}')
                
                # Update text
                self.info_text.set_text(f"Injection time: {time[closest_idx]:.1f} {time_units}\n"
                                       f"Index: {closest_idx}\n"
                                       f"Signal: {signal[closest_idx]:.1f}")
                
                # Update legend
                self.ax.legend()
                self.fig.canvas.draw()
                
                print(f"Selected injection time: {time[closest_idx]:.1f} {time_units} (index {closest_idx})")
        
        def _on_key_press(self, event):
            """Handle key presses for shortcuts."""
            if event.key == 'e':
                self._export_csv(event)
        
        def _export_csv(self, event):
            """Export timecourse data to CSV."""
            import csv
            from pathlib import Path
            
            # Create output directory if needed
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            
            # Generate filename
            csv_file = Path(output_dir) / "timecourse_data.csv"
            
            try:
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f'Time ({time_units})', 'Mean Intensity'])
                    for t, s in zip(time, signal):
                        writer.writerow([f'{t:.3f}', f'{s:.6f}'])
                
                print(f"Exported timecourse data to: {csv_file}")
                
                # Update info text to show export confirmation
                current_text = self.info_text.get_text()
                self.info_text.set_text(current_text + f"\n\nExported to: {csv_file.name}")
                self.fig.canvas.draw()
                
            except Exception as e:
                print(f"Error exporting CSV: {e}")
        
        def _accept_injection_time(self, event):
            """Accept the current injection time selection and close window."""
            self.accepted = True
            print(f"Injection time set: {time[self.injection_index]:.1f} {time_units}")
            plt.close(self.fig)
                
        def show_and_select(self):
            """Show the plot and wait for selection."""
            plt.show()
            return self.injection_index
    
    # Create selector and get result
    selector = InjectionTimeSelector()
    injection_index = selector.show_and_select()
    
    print(f"Final injection time selection: {time[injection_index]:.1f} {time_units} (index {injection_index})")
    return injection_index