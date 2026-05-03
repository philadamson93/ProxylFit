"""
Kinetic modeling module for fitting Proxyl injection curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional, Sequence
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


def estimate_initial_parameters_extended(
    time: np.ndarray,
    signal: np.ndarray,
    pre_injection_signal: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Estimate initial parameters for extended curve fitting.

    Parameters
    ----------
    time : np.ndarray
        Time points (in minutes), post-injection.
    signal : np.ndarray
        Signal values, post-injection.
    pre_injection_signal : np.ndarray, optional
        Signal values *before* injection. When provided, A0_est (the
        baseline) is the mean of these points — that's the actual
        baseline of the ROI. Without it, the function falls back to
        the mean of the first ~4% of the post-injection signal, which
        is unreliable: those points are already on the buildup curve,
        not at baseline, so A0 is overestimated and A2_est (which is
        derived from tail − A0) becomes correspondingly biased.

    Returns
    -------
    dict
        Initial parameter estimates
    """
    # tail level ~ median of last ~10% of points
    tail = signal[int(0.9*len(signal)):]
    tail_level = float(np.median(tail))

    if pre_injection_signal is not None and len(pre_injection_signal) > 0:
        # Real baseline: mean of pre-injection points. This is the
        # physically meaningful A0 — what the signal looks like before
        # contrast arrives — and matches what an ImageJ user would
        # compute by averaging the same pre-injection window.
        A0_est = float(np.mean(pre_injection_signal))
    else:
        # Fallback when caller didn't supply pre-injection data. The
        # first ~4% of the post-injection signal is approximately the
        # baseline only if the user clicked injection time slightly
        # late; otherwise these points are already rising.
        A0_est = float(np.mean(signal[:max(3, int(0.04*len(signal)))]))

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
                       time_units: str = 'minutes',
                       verbose: bool = True,
                       pre_injection_signal: Optional[np.ndarray] = None,
                       steady_state_time: Optional[float] = None,
                       excluded_indices: Optional[Sequence[int]] = None,
                       ) -> Tuple[float, float, float, np.ndarray, Dict]:
    """
    Fit extended Proxyl kinetic model to extract rate parameters.

    Parameters
    ----------
    time : np.ndarray
        Time points (in minutes), post-injection.
    signal : np.ndarray
        Signal intensity values, post-injection.
    time_units : str
        Units for time (for display purposes)
    verbose : bool
        Print diagnostic ``Note``/``Warning`` messages when parameters hit
        bounds, the covariance matrix is degenerate, or the primary fit
        falls back to relaxed bounds. Set to ``False`` for parameter
        mapping (thousands of fits per run) — the messages are useful for
        a single ROI fit but become per-voxel noise at scale and slow the
        run measurably on terminals that re-render each line.
    pre_injection_signal : np.ndarray, optional
        Signal values from before the injection time. When provided,
        A0 is **fixed** to the mean of these points and dropped from
        the optimization variables. Pre-injection signal is a clean
        direct measurement of A0; letting the optimizer wiggle it
        just lets it compensate for model error in the kinetic
        terms, which biases the derived A1/A2/%Enhancement/%NTE
        values. The reported ``A0_error`` is 0 in this mode (fixed
        parameter). Strongly recommended for kinetic-fit-page calls;
        param-mapping voxels also use it when create_parameter_maps
        is told the injection time.
    steady_state_time : float, optional
        Maximum time (in ``time_units``, post-tmax) at which the
        non-tracer term should reach steady state. Used to bound the
        non-tracer rate constant from below: ``knt_lower = ln(20) /
        steady_state_time`` so that ``(1 - exp(-knt·t_steady)) ≥ 0.95``
        (i.e. NTE is within 5% of A2 by ``t_steady``). Without this
        constraint, ``knt`` can drift arbitrarily small to inflate
        the reported A2 amplitude even though the tail is nowhere
        near saturation, biasing %NTE high. Typical values: 70–100
        minutes for in-vivo PROXYL data. ``None`` (default) keeps
        the legacy lower bound of 0.001/min.
    excluded_indices : sequence of int, optional
        Indices into the post-injection ``time`` / ``signal`` arrays
        to drop from the fit. Useful for masking bolus-passage points
        (commonly indices 6–7) that aren't well described by the
        kb/kd/knt processes the model is meant to extract. The
        excluded points still appear on the rendered fit plot —
        they just don't influence the optimizer. ``None`` or an
        empty sequence keeps every point.

    Notes on ``tmax``
    -----------------
    ``tmax`` (the onset time of the non-tracer (1-exp) term) is
    always pinned to ``time[argmax(signal)]`` — the empirical signal
    peak — and dropped from the optimization. The non-tracer onset
    is only weakly identifiable against the tracer rise+decay, and
    letting it float lets ``knt`` and ``kd`` compete for explanation
    of the late-curve shape. Anchoring tmax to the observed peak
    keeps NTE as a refinement on top of the tracer fit instead of
    competing with it. ``tmax_error`` is reported as 0 (fixed). A1
    remains a free fit parameter alongside kb, kd, knt, and t0.

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

    # Keep the full input arrays around for the fitted-signal return —
    # excluded points still need a model value at their time so the
    # plot can show the fit line going through (or near) them. The fit
    # itself only sees the kept subset.
    time_full = np.asarray(time)
    signal_full = np.asarray(signal)

    if excluded_indices:
        excluded_set = {int(i) for i in excluded_indices
                        if 0 <= int(i) < len(signal_full)}
    else:
        excluded_set = set()

    if excluded_set:
        keep_mask = np.ones(len(signal_full), dtype=bool)
        keep_mask[list(excluded_set)] = False
        time_fit_in = time_full[keep_mask]
        signal_fit_in = signal_full[keep_mask]
        if len(time_fit_in) < 8:
            raise ValueError(
                "Need at least 8 time points after exclusions for model fitting"
            )
    else:
        time_fit_in = time_full
        signal_fit_in = signal_full

    # Use the masked-down arrays for everything that follows. The fit's
    # tmax pinning, knt-floor derivation, and bound construction should
    # all reflect the points the optimizer is actually looking at.
    time = time_fit_in
    signal = signal_fit_in

    # Get initial parameter estimates
    initial_params = estimate_initial_parameters_extended(
        time, signal, pre_injection_signal=pre_injection_signal,
    )
    

    # More reasonable bounds to prevent numerical issues
    signal_range = np.max(signal) - np.min(signal)

    # knt lower bound: derived from steady_state_time when caller
    # supplies it. The non-tracer term (1 - exp(-knt·Δt)) reaches 95%
    # of A2 at knt·Δt = ln(20) ≈ 3.0. Bounding knt from below at
    # ln(20)/t_steady forces the fitted NTE to actually saturate
    # within the user-specified window — preventing the optimizer
    # from running knt to ~0 (slow non-saturating term that inflates
    # A2 to absorb residuals). When steady_state_time is None we
    # fall back to the legacy 0.001 lower bound.
    if steady_state_time is not None and steady_state_time > 0:
        knt_lower = float(np.log(20.0) / steady_state_time)
    else:
        knt_lower = 0.001

    # If the derived knt lower exceeds the existing 0.2 upper, bump
    # the upper to keep the constraint feasible. This only happens
    # for very short steady_state_time (< 15 min when using ln(20));
    # at that point the user is asserting NTE is fast, so let knt be.
    knt_upper = max(0.2, knt_lower * 2.0)

    lower_bounds = [
        0,            # A0 >= 0
        0,            # A1 >= 0
        -signal_range,  # A2 can be negative
        0.001,        # kb > 0
        0.001,        # kd > 0
        knt_lower,    # knt: ≥ ln(20)/t_steady when set, else 0.001
        time[0],      # t0 >= first time
        time[0]       # tmax >= first time
    ]

    upper_bounds = [
        np.max(signal) * 2,  # A0 - reasonable baseline bound
        signal_range * 3,    # A1 - reasonable amplitude bound
        signal_range,    # A2 - smaller non-tracer amplitude
        2.0,          # kb <= 2.0/min (more permissive for fast binding)
        1.0,          # kd <= 1.0/min (allow faster decay)
        knt_upper,    # knt: 0.2 unless bumped to keep lower < upper
        time[-1],     # t0 <= last time
        time[-1]      # tmax <= last time
    ]

    # Make sure the initial knt estimate respects the new lower bound
    # so curve_fit doesn't reject p0 as out-of-bounds.
    if initial_params['knt'] < knt_lower:
        initial_params['knt'] = knt_lower * 1.05
    if initial_params['knt'] > knt_upper:
        initial_params['knt'] = knt_upper * 0.95
        
    # Initial guess
    p0 = [initial_params['A0'], initial_params['A1'], initial_params['A2'],
          initial_params['kb'], initial_params['kd'], initial_params['knt'],
          initial_params['t0'], initial_params['tmax']]

    # Build fit dispatch based on which parameters are fixed.
    #   • tmax is ALWAYS pinned to time[argmax(signal)] (the empirical
    #     signal peak). The non-tracer term's onset is hard for the
    #     optimizer to identify against the tracer rise+decay, so
    #     letting it float lets knt and kd compete and pulls tmax to
    #     unphysical values. Pinning to the observed peak anchors
    #     the second (1-exp) term and keeps NTE as a refinement on
    #     top of the tracer fit. A1 stays free along with kb, kd,
    #     knt, and t0.
    #   • A0 is fixed when pre-injection data is supplied — clean
    #     direct measurement of baseline.
    # Two dispatch paths (A0 free vs A0 fixed); tmax always fixed.
    # After the fit returns, _expand_to_8 inserts the fixed values
    # back at indices 0 (A0) and 7 (tmax) so the rest of the function
    # (param unpacking, error reporting, fit_results dict) sees the
    # canonical 8-element popt + 8×8 pcov regardless of dispatch
    # path.
    fix_a0 = (pre_injection_signal is not None
              and len(pre_injection_signal) > 0)
    A0_fixed = float(initial_params['A0']) if fix_a0 else None
    tmax_fixed = float(time[int(np.argmax(signal))])

    if fix_a0:
        # 6-param fit: drop A0 (0) and tmax (7)
        def _wrap(t, A1, A2, kb, kd, knt, t0):
            return proxyl_kinetic_model_extended(
                t, A0_fixed, A1, A2, kb, kd, knt, t0, tmax_fixed,
            )
        fit_model = _wrap
        free_indices = [1, 2, 3, 4, 5, 6]
    else:
        # 7-param fit: drop only tmax (7)
        def _wrap(t, A0, A1, A2, kb, kd, knt, t0):
            return proxyl_kinetic_model_extended(
                t, A0, A1, A2, kb, kd, knt, t0, tmax_fixed,
            )
        fit_model = _wrap
        free_indices = [0, 1, 2, 3, 4, 5, 6]

    fit_p0 = [p0[i] for i in free_indices]
    fit_lower = [lower_bounds[i] for i in free_indices]
    fit_upper = [upper_bounds[i] for i in free_indices]

    def _expand_to_8(popt_, pcov_):
        """Insert fixed A0/tmax values back into popt/pcov so the rest
        of the function sees canonical 8-element shapes regardless of
        which params were actually fit. Pcov rows/cols for fixed params
        are zero (no fit-derived uncertainty for fixed values). tmax is
        always fixed in this build, so index 7 is always filled from
        tmax_fixed; A0 (index 0) is filled when pre-injection data was
        supplied.
        """
        full_popt = np.zeros(8)
        if fix_a0:
            full_popt[0] = A0_fixed
        full_popt[7] = tmax_fixed
        for i, fi in enumerate(free_indices):
            full_popt[fi] = popt_[i]
        full_pcov = np.zeros((8, 8))
        for i, fi in enumerate(free_indices):
            for j, fj in enumerate(free_indices):
                full_pcov[fi, fj] = pcov_[i, j]
        return full_popt, full_pcov

    try:
        # First attempt with standard fitting
        popt, pcov = curve_fit(
            fit_model,
            time,
            signal,
            p0=fit_p0,
            bounds=(fit_lower, fit_upper),
            maxfev=500,    # Cap iterations; pathological pixels fall through
                           # to the relaxed-bounds dogbox retry below.
            method='trf',  # Trust region reflective algorithm
            ftol=1e-6,     # Loosened from 1e-8 — the initial estimates are
            xtol=1e-6,     # close enough that 1e-6 converges in far fewer steps.
        )
        popt, pcov = _expand_to_8(popt, pcov)

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
        if verbose and critical_bounds_hit and len(critical_bounds_hit) <= 2:
            print(f"Note: {', '.join(critical_bounds_hit)} - consider adjusting parameter bounds")
        
        # Calculate fitted curve over the FULL input time array (including
        # excluded points) so the caller can draw the model line through
        # every data point on the plot. R²/RMSE/residuals below are still
        # computed against the kept points only — those are what the
        # optimizer actually minimized.
        fitted_signal = proxyl_kinetic_model_extended(time_full, A0_fit, A1_fit, A2_fit,
                                                     kb_fit, kd_fit, knt_fit,
                                                     t0_fit, tmax_fit)
        fitted_signal_kept = proxyl_kinetic_model_extended(time, A0_fit, A1_fit, A2_fit,
                                                           kb_fit, kd_fit, knt_fit,
                                                           t0_fit, tmax_fit)
        
        # Calculate parameter uncertainties with robust error handling
        try:
            # Check for numerical issues in covariance matrix
            diag_elements = np.diag(pcov)
            if np.any(diag_elements < 0) or np.any(np.isinf(diag_elements)) or np.any(np.isnan(diag_elements)):
                # Use relative error estimation if covariance is bad
                if verbose:
                    print("Warning: Covariance matrix has numerical issues. Using conservative error estimates.")
                param_errors = np.abs(popt) * 0.1  # 10% relative error as fallback
            else:
                param_errors = np.sqrt(diag_elements)
                # Cap extremely large errors at 100% of parameter value
                for i in range(len(param_errors)):
                    if param_errors[i] > abs(popt[i]) * 2:  # Error > 200% of value
                        param_errors[i] = abs(popt[i])  # Cap at 100% error
        except Exception as e:
            if verbose:
                print(f"Warning: Error calculation failed ({e}). Using conservative estimates.")
            param_errors = np.abs(popt) * 0.1  # 10% relative error as fallback

        # Force errors for fixed parameters to zero. The conservative
        # fallbacks above (10% of |popt|) would otherwise lie about
        # uncertainty for parameters the optimizer never touched.
        fixed_indices = set(range(8)) - set(free_indices)
        for fi in fixed_indices:
            param_errors[fi] = 0.0

        # Calculate fit quality metrics over the kept (non-excluded)
        # subset only — those are what the optimizer actually saw.
        residuals = signal - fitted_signal_kept
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((signal - np.mean(signal)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals ** 2))

        # Compile results
        fit_results = {
            'A0': A0_fit,
            'A1': A1_fit,
            'A2': A2_fit,
            'A0_est': initial_params['A0'],
            'A2_est': initial_params['A2'],
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
            'excluded_indices': sorted(excluded_set),
        }

        return kb_fit, kd_fit, knt_fit, fitted_signal, fit_results

    except Exception as e:
        if verbose:
            print(f"First fitting attempt failed: {e}")
            print("Trying alternative fitting approach with relaxed constraints...")
        
        try:
            # Fallback approach with relaxed bounds and different method.
            # knt upper still respects steady_state_time so the fallback
            # can't sneak past the user's knt floor either.
            relaxed_upper = [
                np.max(signal) * 3,         # More relaxed A0
                signal_range * 5,           # More relaxed A1
                signal_range * 3,           # More relaxed A2
                1.0,                        # More relaxed kb
                0.5,                        # More relaxed kd
                max(0.1, knt_lower * 2.0),  # knt: keep ≥ 2× knt_lower
                time[-1],                   # t0
                time[-1]                    # tmax
            ]

            # Slice relaxed upper bounds to match the free-parameter
            # set (same indices used for the primary fit).
            fit_relaxed_upper = [relaxed_upper[i] for i in free_indices]

            # Try with dogbox method and looser tolerances
            popt, pcov = curve_fit(
                fit_model,
                time,
                signal,
                p0=fit_p0,
                bounds=(fit_lower, fit_relaxed_upper),
                maxfev=2000,
                method='dogbox',  # Different algorithm
                ftol=1e-6,        # Looser tolerance
                xtol=1e-6
            )
            popt, pcov = _expand_to_8(popt, pcov)

            A0_fit, A1_fit, A2_fit, kb_fit, kd_fit, knt_fit, t0_fit, tmax_fit = popt
            
            if verbose:
                print("Fallback fitting succeeded with relaxed constraints.")
            
            # Calculate fitted curve over the FULL input time array
            # (including excluded points) so the caller can plot the
            # model line through every data point. Quality metrics
            # below use the kept subset only.
            fitted_signal = proxyl_kinetic_model_extended(time_full, A0_fit, A1_fit, A2_fit,
                                                         kb_fit, kd_fit, knt_fit,
                                                         t0_fit, tmax_fit)
            fitted_signal_kept = proxyl_kinetic_model_extended(time, A0_fit, A1_fit, A2_fit,
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

            # Force errors for fixed parameters to zero — same as the
            # primary fit path. Without this, the 20% fallback would
            # report a fake uncertainty for tmax / A0 / A2 even though
            # the optimizer never moved them.
            fixed_indices = set(range(8)) - set(free_indices)
            for fi in fixed_indices:
                param_errors[fi] = 0.0

            # Calculate fit quality metrics on kept points only.
            residuals = signal - fitted_signal_kept
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((signal - np.mean(signal)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals ** 2))

            # Compile results
            fit_results = {
                'A0': A0_fit,
                'A1': A1_fit,
                'A2': A2_fit,
                'A0_est': initial_params['A0'],
                'A2_est': initial_params['A2'],
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
                'excluded_indices': sorted(excluded_set),
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
    print(f"  tmax (NTE onset, fixed): {fit_results['tmax']:.2f} {fit_results['time_units']}  [pinned to argmax(signal)]")
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