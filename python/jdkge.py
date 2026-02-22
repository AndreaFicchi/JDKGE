# -----------------------------------------------------------------------------------------------------------------
# JDKGE/python/jdkge.py
# -----------------------------------------------------------------------------------------------------------------
# JDKGE metric: Kling–Gupta Efficiency with an added Jensen–Shannon Divergence component.
#
# Computes a modified KGE augmented with an additional component assessing distributional similarity based on 
# a discretized Jensen–Shannon Divergence (JSD) between observed and simulated flow distributions.
# -----------------------------------------------------------------------------------------------------------------
#
# ----------
# References
# ----------
# Ficchì, A., Bavera, D., Grimaldi, S., Moschini, F., Pistocchi, A., Russo, C., Salamon, P.,
# and Toreti, A. (2026). Improving low and high flow simulations at once: An enhanced metric for
# hydrological model calibration. EGUsphere [preprint]. https://doi.org/10.5194/egusphere-2026-43
#
# Freedman, D. and Diaconis, P. (1981). On the histogram as a density estimator: L2 theory.
# Zeitschrift fur Wahrscheinlichkeitstheorie und Verwandte Gebiete, 57, 453–476.
# https://doi.org/10.1007/BF01025868
#
# Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition of the mean squared error and NSE 
# performance criteria: Implications for improving hydrological modelling, \emph{Journal of Hydrology}, 377, 80-91,
# https://doi.org/10.1016/j.jhydrol.2009.08.003, 2009.
#
# Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper Danube basin under an ensemble of climate 
# change scenarios, Journal of Hydrology, 424-425, 264-277, https://doi.org/10.1016/j.jhydrol.2012.01.011, 2012.
#
# Lin, J. (1991). Divergence measures based on the Shannon entropy.
# IEEE Transactions on Information Theory, 37, 145–151. https://doi.org/10.1109/18.61115
#
# ------
# Author
# ------
# Author: Andrea Ficchì, Politecnico di Milano, Italy
# -----------------------------------------------------------------------------------------------------------------

import numpy as np

try: # try to run as a package keeping relative imports
    from .jsd_fd_log import jsd_fd_log
    from .utils import filter_nan
except ImportError:  # running as a script (no package context)
    from jsd_fd_log import jsd_fd_log
    from utils import filter_nan


def jdkge(obs, sim, dt_h=24.0, verbose=False):
    """
    Compute the Joint Divergence Kling-Gupta Efficiency (JDKGE) metric. 
    JDKGE is an enhanced calibration objective function that augments the modified KGE with an additional 
    distributional similarity component based on the Jensen–Shannon Divergence (JSD).

    The metric is computed after removing NaN-paired values. Components are:
      - r    : Pearson correlation between observed and simulated
      - beta : bias ratio (mean(sim) / mean(obs))
      - alpha: variability ratio (CV(sim) / CV(obs)), where CV = std/mean
      - JSD  : discretized Jensen–Shannon Divergence (bits, log base 2)

    The JDKGE score is: JDKGE = 1 - sqrt((r-1)^2 + (beta-1)^2 + (alpha-1)^2 + JSD^2)

    ----------
    Parameters
    ----------
    obs : array_like, shape (n,)
        Observed streamflow values.
    sim : array_like, shape (n,)
        Simulated streamflow values.
    dt_h : float, default=24.0
        Observation time step in hours (e.g., 24 for daily data). Internally converted to seconds for the JSD computation.
    verbose: boolean, default=False
        Flag for logging message with JDKGE (value and components) output.

    -------
    Returns
    -------
    jdkge_value : float
        JDKGE value.
    r : float
        Correlation component.
    beta : float
        Bias ratio component.
    alpha : float
        Variability ratio component.
    JSD : float
        Jensen–Shannon Divergence component (bits).

    ------
    Raises
    ------
    ValueError
        If fewer than 2 valid paired samples remain after filtering.
    """
    obs_f, sim_f = filter_nan(obs, sim)

    if obs_f.size < 2:
        raise ValueError("Need at least 2 paired non-NaN samples to compute correlation and JDKGE.")

    # --- r component (Pearson correlation) ---
    r = float(np.corrcoef(obs_f, sim_f)[0, 1])

    mean_o = float(np.mean(obs_f))
    mean_s = float(np.mean(sim_f))
    std_o = float(np.std(obs_f))
    std_s = float(np.std(sim_f))

    # --- beta component (ratio of means): mean(sim) / mean(obs) ---
    if mean_s == 0.0 and mean_o == 0.0:
        beta = 1.0  # fallback to avoid NaN/Inf during calibration
    else:
        beta = mean_s / mean_o

    # --- alpha component (CV ratio): (sd(sim)/mean(sim)) / (sd(obs)/mean(obs)) ---
    # alpha = (std_s / mean_s) / (std_o / mean_o)
    # CVsim
    if mean_s == 0.0:
        if std_s == 0.0:
            cv_sim = 1.0
        else:
            cv_sim = 99999.0
    else:
        cv_sim = std_s / mean_s

    # CVobs
    if mean_o == 0.0:
        if std_o == 0.0:
            cv_obs = 1.0
        else:
            cv_obs = 99999.0
    else:
        cv_obs = std_o / mean_o

    # alpha (CV ratio)
    if cv_sim == 0.0 and cv_obs == 0.0:
        alpha = 1.0
    else:
        alpha = cv_sim / cv_obs

    # --- JSD component ---
    ts_s = float(dt_h) * 3600.0
    JSD = float(jsd_fd_log(obs_f, sim_f, ts_s))

    # fallback to avoid NaN/Inf during calibration
    if not np.isfinite(JSD):
        JSD = 1.0 

    # --- JDKGE metric ---
    jdkge_value = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (alpha - 1.0) ** 2 + JSD**2)

    if not np.isfinite(jdkge_value):
        print("WARNING: nan or nonfinite JDKGE found")
        jdkge_value = -999.0

    if verbose:
        print("JDKGE value:")
        print(f"JDKGE = {jdkge_value}")
        print("JDKGE components:")
        print(f"r = {r}")
        print(f"beta = {beta}")
        print(f"alpha = {alpha}")
        print(f"JSD = {JSD}")

    return float(jdkge_value), float(r), float(beta), float(alpha), float(JSD)
