# ------------------------------------------------------------------------------------------------------------
# JDKGE/python/jsd_fd_log.py
# ------------------------------------------------------------------------------------------------------------
# Calculation of the Jensen–Shannon Divergence (JSD) component used in the JDKGE metric.
#
# jsd_fd_log computes a discretized estimate of the Jensen–Shannon Divergence (JSD) between observed and
# simulated streamflow time series. The JSD is estimated from empirical distributions obtained
# from log-transformed flows and scale-invariant binning based on an adapted Freedman–Diaconis
# (FD) rule. Additive smoothing is used to avoid zero-probability bins.
# Using log-transformed flows increases sensitivity to low flows. 
# JSD is bounded between 0 and 1 (in bits, due to base-2 logarithms use), where 0 indicates perfect agreement.
# ------------------------------------------------------------------------------------------------------------
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
# Lin, J. (1991). Divergence measures based on the Shannon entropy.
# IEEE Transactions on Information Theory, 37, 145–151. https://doi.org/10.1109/18.61115
#
# ------
# Author
# ------
# Andrea Ficchì, Politecnico di Milano, Italy
# ------------------------------------------------------------------------------------------------------------

# import required modules
import numpy as np
from scipy.stats import iqr

def jsd_fd_log(obs, sim, ts_s=86400):
    """
    Compute the discretized Jensen–Shannon Divergence (JSD) used in the JDKGE metric.

    The function:
      1) removes non-finite and negative values,
      2) replaces zeros with a small epsilon,
      3) applies a natural log transformation of streamflow values (obs and sim),
      4) builds equal-width histogram bins using an adapted Freedman–Diaconis rule with
         timestep scaling and safety bounds on bin count,
      5) applies additive smoothing and computes the discrete JSD using log base 2.

    ----------
    Parameters
    ----------
    obs:  (1D array of float) observed streamflow values. 
    sim:  (1D array of float) simulated streamflow values (paired with obs).
    ts_s: (float) data time step in seconds (e.g., 86400 for daily data).

    -------
    Returns
    -------
    jsd: (float) Jensen–Shannon Divergence in bits (use of log base 2).

    --------
    See Also
    --------
    numpy.histogram : Histogram construction.
    scipy.stats.iqr : Interquartile range used in the FD binning strategy.
    """

    # Fixed hyper-parameters (reported here for clarity)
    epsilon = 1e-6
    min_nbins = 25
    max_nbins = 100

    # Define time-step scaling factor
    ts_factor = ts_s / 86400

    # Remove non-finite and non-positive data
    obs = obs[np.isfinite(obs) & (obs >= 0)]
    sim = sim[np.isfinite(sim) & (sim >= 0)]

    # epsilon adjustments for safety and data distribution consistency
    if len(obs[obs > 0]) > 0:
        obs_min_nonzero = np.min(obs[obs > 0])
    else:
        obs_min_nonzero = epsilon
    if len(sim[sim > 0]) > 0:
        sim_min_nonzero = np.min(sim[sim > 0])
    else:
        sim_min_nonzero = epsilon
    epsilon_max = min(obs_min_nonzero, sim_min_nonzero)

    if epsilon > epsilon_max:
        epsilon = epsilon_max * 1e-1

    # Replace zero values with epsilon 
    obs[obs == 0] = epsilon
    sim[sim == 0] = epsilon

    # Log transformation of flow values
    obs_log = np.log(obs)
    sim_log = np.log(sim)

    # Combine data for support range to ensure bins are computed over the entire range of data (obs & sim)
    all_data = np.concatenate([obs_log, sim_log])
    obs_log_min = np.min(all_data)
    obs_log_max = np.max(all_data)

    # Freedman–Diaconis (FD) binning rule
    iqr_val = iqr(obs_log, nan_policy='omit')

    if iqr_val == 0:
        bin_width = (obs_log_max - obs_log_min) / min_nbins # fallback bin width
    else:
        bin_width = 2 * iqr_val / len(obs_log)**(1/3)  # FD-based bin width

    # Get number of bins (FD + safety limits)
    absolute_min_width = epsilon * 100    
    bin_width = max(bin_width, absolute_min_width)
    range_width = obs_log_max - obs_log_min
    n_bins = max(min(int(np.ceil((ts_factor**(1/3)) * range_width / bin_width)), max_nbins), min_nbins)

    # Generate bin edges 
    bin_edges = np.linspace(obs_log_min, obs_log_max, n_bins + 1)  
    bin_edges = bin_edges[np.isfinite(bin_edges)]                  

    bin_edges[0] = min(bin_edges[0], obs_log_min)
    bin_edges[-1] = max(bin_edges[-1], obs_log_max)

    # Compute the histogram for observed and simulated data with density normalization
    p_hist, _ = np.histogram(obs_log, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(sim_log, bins=bin_edges, density=True)

    # Avoid zero probabilities by adding a small epsilon
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon

    # Normalize histograms to ensure they represent probability distributions
    p_hist /= np.nansum(p_hist)
    q_hist /= np.nansum(q_hist)
    
    # Compute the mixture distribution
    mix_hist = 0.5 * (p_hist + q_hist)
    
    # Calculate JSD using base-2 logarithm
    jsd = 0.5 * np.nansum(p_hist * np.log2(p_hist / mix_hist)) + 0.5 * np.nansum(q_hist * np.log2(q_hist / mix_hist))

    return jsd
