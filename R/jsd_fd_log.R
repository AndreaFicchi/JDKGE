#***********************************************************************************************************************************
#***********************************************************************************************************************************
#' 
#' @title  Calculation of the Jensen-Shannon Divergence (JSD) component used in the JDKGE metric
#' 
#' @description computes a discretized estimate of the Jensen-Shannon Divergence (JSD) 
#' between observed and simulated streamflow time series.
#' The JSD is estimated from empirical distributions obtained from the log-transformed flows
#' and scale-invariant binning based on an adapted Freedman-Diaconis (FD) rule.
#' 
#' The use of log-transformed flows allows to increase the sensitivity to low-flows and a scale-invariant 
#' 
#' JSD measures the divergence between the two data distributions (empirical probability distributions).
#' As base-2 logarithms are used, JSD is bounded between 0 and 1 (in bits), with 0 indicating a perfect
#' fit between observations and simulations.
#' 
#' --==== Parameters to be passed as inputs to the function ====--
#' @param obs         Numeric vector of observed (streamflow) values.
#' @param sim         Numeric vector of simulated (streamflow) values (paired with \code{obs}).
#' @param timestep_s  Numeric scalar giving the data time step in seconds (default: 86400 for daily data).
#' 
#' --==== Outputs of the function ====--
#' @return JSD        Numeric scalar representing the JSD value (in bits) used as component in the JDKGE metric.
#' 
#' @details
#' This JSD implementation follows the discretized JSD formulation described in 
#' Ficchì et al. (under review), including log-transformation of flows, 
#' timestep-invariant Freedman-Diaconis binning, and additive smoothing.
#' The function performs the following steps:
#' \itemize{
#'   \item Pre-processing of zero and non-positive values.
#'   \item Log-transformation of streamflows to increase sensitivity to low flows.
#'   \item Construction of empirical distributions using equal-width bins
#'         determined via an adapted Freedman-Diaconis rule.
#'   \item Additive smoothing to avoid zero-probability bins.
#'   \item Computation of the discrete Jensen-Shannon Divergence.
#' }
#' 
#' @seealso 
#' \code{\link{jdkge}} for the full JDKGE metric,
#' \code{\link{hist}} for histogram construction,
#' \code{\link{IQR}} for the interquartile range used in the FD binning strategy.
#' 
#' @references
#' Ficchì, A., Bavera, D., Grimaldi, S., Moschini, F., Pistocchi, A., Russo, C., Salamon, P., and Toreti, A.: 
#' Improving low and high flow simulations at once: An enhanced metric for hydrological model calibration, 
#' \emph{EGUsphere [preprint]}, https://doi.org/10.5194/egusphere-2026-43, 2026.
#' 
#' Freedman, D. and Diaconis, P.: On the histogram as a density estimator: L2 theory,
#' \emph{Zeitschrift fur Wahrscheinlichkeitstheorieund Verwandte Gebiete}, 57, 453-476, https://doi.org/10.1007/BF01025868, 1981.
#' 
#' Lin, J.: Divergence measures based on the Shannon entropy, \emph{IEEE Transactions on Information Theory}, 37, 145-151, 
#' https://doi.org/10.1109/18.61115, 1991.
#' 
#' @author  Andrea Ficchì, Politecnico di Milano, Italy
#' 
#' @export
#'       
#***********************************************************************************************************************************

jsd_fd_log <- function(obs, sim, timestep_s = 86400) {
  
  # Fixed hyper-parameters (reported here for clarity)
  epsilon = 1e-6
  min_nbins = 25
  max_nbins = 100
  
  # Define time-scale invariance factor 
  ts_factor <- timestep_s / 86400
  
  # Remove non-finite and any negative values
  obs <- obs[is.finite(obs) & obs >= 0] 
  sim <- sim[is.finite(sim) & sim >= 0] 
  
  obs_min_nonzero <- min(obs[obs > 0], na.rm = TRUE)
  sim_min_nonzero <- min(sim[sim > 0], na.rm = TRUE)
  epsilon_max <- min(obs_min_nonzero, sim_min_nonzero)
  
  if (epsilon > epsilon_max) {
    warning("The provided epsilon is too large; adjusting it to make it lower than the smallest non-zero obs/sim.")
    epsilon <- epsilon_max * 1e-1
  }
  
  # Replace zero values with epsilon 
  obs[obs == 0] <- epsilon
  sim[sim == 0] <- epsilon
  
  # Log transformation of flows
  obs_log <- log(obs)
  sim_log <- log(sim)
  
  # Combine data for support range
  all_data <- c(obs_log, sim_log)
  obs_log_min <- min(all_data, na.rm = TRUE)
  obs_log_max <- max(all_data, na.rm = TRUE)
  
  # Freedman-Diaconis rule for bin width and number of bins
  iqr_val <- IQR(obs_log, na.rm = TRUE)
  
  if (iqr_val == 0) {
    bin_width <- (obs_log_max - obs_log_min) / min_nbins
    warning("Warning: IQR of obs_log is zero! Fallback with default bin_width (based on obs_log range): ", round(bin_width, 5))
  } else {
    bin_width <- 2 * iqr_val / length(obs_log)^(1/3) 
  }
  
  absolute_min_width <- min(epsilon*(10^2), 1e-1) 
  
  bin_width <- max(bin_width, absolute_min_width)

  # Data-based binning adjustment
  range_width <- obs_log_max - obs_log_min  
  
  n_bins <- max(min( ceiling( (ts_factor^(1/3))*range_width / bin_width), max_nbins), min_nbins) 
  
  # Log-based binning 
  bin_edges <- seq(obs_log_min, obs_log_max, length.out = n_bins + 1)
  
  # Validate bin edges
  bin_edges <- bin_edges[is.finite(bin_edges)]
  
  ## Consistency checks - needed for numeric approximations
  bin_edges[1] <- min(bin_edges[1], obs_log_min) 
  bin_edges[length(bin_edges)] <- max(tail(bin_edges, 1), obs_log_max)
  
  # Compute histograms
  p_hist <- hist(obs_log, breaks = bin_edges, plot = FALSE)$density
  q_hist <- hist(sim_log, breaks = bin_edges, plot = FALSE)$density
  
  # Add epsilon and normalize
  p_hist <- p_hist + epsilon
  q_hist <- q_hist + epsilon
  p_hist <- p_hist / sum(p_hist, na.rm = TRUE)
  q_hist <- q_hist / sum(q_hist, na.rm = TRUE)
  
  mix_hist <- 0.5 * (p_hist + q_hist)
  
  # Compute JSD using base-2 logarithms
  js_div_p <- ifelse(p_hist > 0, p_hist * log2(p_hist / mix_hist), 0)
  js_div_q <- ifelse(q_hist > 0, q_hist * log2(q_hist / mix_hist), 0)
  
  JSD <- 0.5 * sum(js_div_p, na.rm = TRUE) + 0.5 * sum(js_div_q, na.rm = TRUE)
  
  return(JSD)
  
}
