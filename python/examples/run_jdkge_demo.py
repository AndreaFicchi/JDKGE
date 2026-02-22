# JDKGE/python/examples/run_jdkge_demo.py
"""
Demo script that computes JDKGE and KGE' on sample observed/simulated streamflow series.

- Observed and simulated streamflow time series are loaded from a local CSV file if available, otherwise generated synthetically.
- If created synthetically, simulated streamflow is created by perturbing observations with multiplicative noise + low-flow-focused
  perturbations, then enforcing non-negativity.
- JDKGE is computed and compared with KGE' (Kling et al. (2012)'s version with ratio of coeff. of variation, without the JSD term).

Run examples
------------
From repo root:
    python -m JDKGE.python.examples.run_jdkge_demo

Optional data file
------------------
Place the combined CSV file with the sample data at:
    JDKGE/python/data/L0123001_Qmm_obs_sim.csv

Expected columns:
    date,Qobs_mm,Qsim_mm
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

# Set working directory to the script's directory for consistent relative paths
os.chdir(Path(__file__).resolve().parent)
print("Current working directory:", os.getcwd())

# Add .../JDKGE/python to sys.path for importing jdkge, assuming this run_jdkge_demo script is in .../JDKGE/python/examples/
pkg_dir = Path(__file__).resolve().parents[1]
if str(pkg_dir) not in sys.path:
    print("str(pkg_dir):", str(pkg_dir))
    sys.path.insert(0, str(pkg_dir))

from jdkge import jdkge

def load_obs_sim_csv(path: Path):
    """
    Load a combined observed/simulated streamflow CSV exported from R (airGR sample data for basin L0123001).

    Expected columns
    ----------------
    - date (optional but recommended; ISO format YYYY-MM-DD)
    - Qobs_mm
    - Qsim_mm

    Returns
    -------
    qobs_w_gaps : np.ndarray
        Observed streamflow values (may include NaNs).
    qsim_w_gaps : np.ndarray
        Simulated streamflow values (may include NaNs).
    dates : np.ndarray (datetime64[D]) | None
        Dates if present, otherwise None.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    data = np.genfromtxt(path, delimiter=",", dtype=None, names=True, encoding="utf-8")
    names = list(data.dtype.names or [])

    required = {"Qobs_mm", "Qsim_mm"}
    missing = required - set(names)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}. Found: {names}")

    qobs_w_gaps = np.asarray(data["Qobs_mm"], dtype=float)
    qsim_w_gaps = np.asarray(data["Qsim_mm"], dtype=float)

    dates = None
    if "date" in names:
        dates = np.asarray(data["date"], dtype="datetime64[D]")

    return qobs_w_gaps, qsim_w_gaps, dates


def _synthetic_qobs_with_gaps(n: int = 5000, seed: int = 123) -> np.ndarray:
    """
    Generate a synthetic observed streamflow series with NaNs and zeros for demo purposes.
    """
    rng = np.random.default_rng(seed)

    q = rng.gamma(shape=1.2, scale=2.0, size=n)

    zero_idx = rng.choice(n, size=max(1, n // 50), replace=False)
    nan_idx = rng.choice(n, size=max(1, n // 40), replace=False)
    q[zero_idx] = 0.0
    q[nan_idx] = np.nan

    return q


def _plogis(x: np.ndarray) -> np.ndarray:
    """
    Logistic function, R's plogis().
    """
    return 1.0 / (1.0 + np.exp(-x))


def _build_synthetic_simulation(qobs_w_gaps: np.ndarray, seed: int = 123) -> np.ndarray:
    """
    Create simulated series by perturbing observed series.
    """
    rng = np.random.default_rng(seed)

    bias_factor = 1.05
    cv_noise = 0.25

    noise_sd = np.sqrt(np.log(1.0 + cv_noise**2))
    noise_mult = np.exp(rng.normal(loc=0.0, scale=noise_sd, size=qobs_w_gaps.size))

    qsim = bias_factor * qobs_w_gaps * noise_mult

    q_thr = np.nanquantile(qobs_w_gaps, 0.25, method="linear")
    med_pos = np.nanmedian(qobs_w_gaps[qobs_w_gaps > 0])
    add_noise_sd = 0.2 * med_pos if np.isfinite(med_pos) else 0.0

    if q_thr != 0:
        lf_w = _plogis((q_thr - qobs_w_gaps) / (q_thr * 0.15))
    else:
        lf_w = np.zeros_like(qobs_w_gaps)

    lf_w = np.where(np.isnan(lf_w), 0.0, lf_w)

    qsim = qsim * (1.0 + lf_w * 0.5) + rng.normal(
        loc=0.9 * q_thr, scale=add_noise_sd, size=qobs_w_gaps.size
    ) * lf_w

    qsim = np.maximum(qsim, 0.0)
    return qsim


def _kge_2012(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float, float, float]:
    """
    KGE' Kling et al. (2012)'s modified KGE (with ratio of coeff. of var.) for comparison: 
    KGE' = 1 - sqrt((r-1)^2 + (beta-1)^2 + (alpha-1)^2),
    using alpha = CV(sim)/CV(obs).
    """
    sim = np.asarray(sim, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()

    r = float(np.corrcoef(obs, sim)[0, 1])

    mean_o = float(np.mean(obs))
    mean_s = float(np.mean(sim))
    std_o = float(np.std(obs))
    std_s = float(np.std(sim))

    beta = mean_s / mean_o
    alpha = (std_s / mean_s) / (std_o / mean_o)

    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (alpha - 1.0) ** 2)
    return float(kge), float(r), float(beta), float(alpha)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    script_par_dir = script_dir.parent 
    data_path = script_par_dir / "data" / "L0123001_Qmm_obs_sim.csv"

    print("[demo] Loading Qobs and Qsim data...")
    try:
        qobs_w_gaps, qsim_w_gaps, dates = load_obs_sim_csv(data_path)
        print(f"[demo] Loaded observed+simulated streamflow from: {data_path}")
    except FileNotFoundError:
        print(f"[demo] No data file found at: {data_path}")
        print("[demo] Using synthetic observed+simulated streamflow for this demo.")
        dates = None
        qobs_w_gaps = _synthetic_qobs_with_gaps(n=10000, seed=123)
        qsim_w_gaps = _build_synthetic_simulation(qobs_w_gaps, seed=123)

    # Keep paired finite, non-negative values only
    idx_ok = (
        np.isfinite(qobs_w_gaps)
        & np.isfinite(qsim_w_gaps)
        & (qobs_w_gaps >= 0)
        & (qsim_w_gaps >= 0)
    )
    qobs = qobs_w_gaps[idx_ok]
    qsim = qsim_w_gaps[idx_ok]

    n_missing_obs = int(np.sum(~np.isfinite(qobs_w_gaps)))
    n_missing_sim = int(np.sum(~np.isfinite(qsim_w_gaps)))
    print(f"[demo] Missing values: obs={n_missing_obs}, sim={n_missing_sim}")
    print(f"[demo] Paired valid samples: {qobs.size}")

    print("[demo] Computing JDKGE and KGE'...")
    jdkge_value, r, beta, alpha, jsd = jdkge(obs=qobs, sim=qsim, dt_h=24.0, verbose=True)

    print("\n--- JDKGE results ---")
    print(f"JDKGE = {jdkge_value:.6f}")
    print(f"r     = {r:.6f}")
    print(f"beta  = {beta:.6f}")
    print(f"alpha = {alpha:.6f}")
    print(f"JSD   = {jsd:.6f}")

    kge_value, r_kge, beta_kge, alpha_kge = _kge_2012(sim=qsim, obs=qobs)
    print("\n--- KGE' (Kling et al. (2012)'s modified KGE) results ---")
    print(f"KGE'  = {kge_value:.6f}")
    print(f"r     = {r_kge:.6f}")
    print(f"beta  = {beta_kge:.6f}")
    print(f"alpha = {alpha_kge:.6f}")

    # Quick plot (optional)
    print("Plotting sample hydrographs")
    
    # Quick plot (optional)
    try:
        import matplotlib.pyplot as plt  # optional dependency
        fig, ax = plt.subplots()
        if dates is not None and dates.size == qobs_w_gaps.size:
            ax.plot(dates, qobs_w_gaps, label="Qobs (with gaps)")
            ax.plot(dates, qsim_w_gaps, label="Qsim (with gaps)")
        else:
            ax.plot(qobs_w_gaps, label="Qobs (with gaps)")
            ax.plot(qsim_w_gaps, label="Qsim (with gaps)")

        ax.set_ylabel("Streamflow (mm/d)")
        ax.legend()
        ax.set_title("JDKGE demo: observed vs simulated")
        
        # Text box with metrics (axes coordinates: top-left)
        text = f"JDKGE = {jdkge_value:.4f}\nKGE'  = {kge_value:.4f}"
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        plt.show()
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())