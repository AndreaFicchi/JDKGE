# ------------------------------------------------------------------------------------------------------------
# JDKGE/python/utils.py
# ------------------------------------------------------------------------------------------------------------
# Utilities for the JDKGE metric.
#
# Currently including: 
# - NaN filtering for paired observed/simulated series.
# ------------------------------------------------------------------------------------------------------------

import numpy as np


def filter_nan(obs, sim):
    """
    Removes paired entries from simulated and observed data where either observed or simulated values are NaN.

    ----------
    Parameters
    ----------
    obs : array_like
        Observed values. Must be the same length as `sim`.
    sim : array_like
        Simulated values.
    -------
    Returns
    -------
    obs_f : numpy.ndarray
        Filtered observed values as a 1D array.
    sim_f : numpy.ndarray
        Filtered simulated values as a 1D array.

    ------
    Raises
    ------
    ValueError
        If inputs have different sizes.
    """

    obs_arr = np.asarray(obs, dtype=float).ravel()
    sim_arr = np.asarray(sim, dtype=float).ravel()

    if sim_arr.size != obs_arr.size:
        raise ValueError(f"sim and obs must have the same length. Got {sim_arr.size} and {obs_arr.size}.")

    mask = ~np.isnan(sim_arr) & ~np.isnan(obs_arr)

    return obs_arr[mask], sim_arr[mask]
