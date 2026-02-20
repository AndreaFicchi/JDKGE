#***********************************************************************************************************************************
#***********************************************************************************************************************************
#'
#' @title Joint Divergence Kling-Gupta Efficiency (JDKGE)
#'
#' @description
#' computes the Joint Divergence Kling-Gupta Efficiency (JDKGE), an enhanced
#' calibration objective function that augments the modified Kling-Gupta Efficiency
#' (KGE') with an additional component based on the Jensen-Shannon Divergence (JSD).
#'
#' JDKGE combines four components: (i) Pearson correlation, r, (ii) relative
#' variability as the ratio of coefficients of variation (\eqn{\alpha}), (iii) bias as
#' the ratio of means (\eqn{\beta}), and (iv) a discretized JSD estimate computed from
#' empirical distributions of log-transformed flows (see \code{\link{jsd_fd_log}}).
#' The overall score is defined as one minus the Euclidean distance to the ideal point
#' \eqn{(r,\alpha,\beta,\mathrm{JSD}) = (1,1,1,0)}.
#'
#' --==== Parameters to be passed as inputs to the function ====--
#' @param obs Numeric vector of observed streamflow values.
#' @param sim Numeric vector of simulated streamflow values (paired with \code{obs}).
#' @param timestep_s Numeric scalar giving the data time step in seconds (default 86400 for daily data).
#' @param verbose Logical; if \code{TRUE} (default), prints the overall score and component values.
#' @param check_inputs Logical; if \code{TRUE}, validates inputs (types, lengths, and \code{timestep_s} value).
#'        Set to \code{FALSE} for speed in calibration loops (default).
#'
#' --==== Outputs of the function ====--
#' @return A list with the following elements:
#' \describe{
#'   \item{CritValue}{Numeric scalar, the JDKGE value.}
#'   \item{CritName}{Character string with the metric name, i.e., \code{JDKGE}.}
#'   \item{SubCritValues}{Numeric vector of length 4 with components \eqn{(r,\alpha,\beta,\mathrm{JSD})}.}
#'   \item{SubCritNames}{Character vector of component names: \code{c("r","alpha","beta","jsd")}.}
#'   \item{CritBestValue}{Numeric scalar, the ideal JDKGE value (1).}
#' }
#'
#' @details
#' Missing or non-finite values in \code{obs} and \code{sim} are removed pairwise prior to computation.
#' The JSD component is computed via \code{\link{jsd_fd_log}}, which applies (i) pre-processing (removing
#' non-positive values), (ii) log-transformation to increase sensitivity to low flows, (iii) a
#' timestep-adjusted Freedman-Diaconis binning rule, (iv) additive smoothing to avoid
#' zero-probability bins, and (v) computation of the discrete JSD component.
#'
#' @seealso
#' \code{\link{jsd_fd_log}} for the JSD component implementation;
#' \code{\link[stats]{sd}}, \code{\link[base]{mean}} for basic statistics.
#'
#' @references
#' Ficchì, A., Bavera, D., Grimaldi, S., Moschini, F., Pistocchi, A., Russo, C., Salamon, P., and Toreti, A.: 
#' Improving low and high flow simulations at once: An enhanced metric for hydrological model calibration, 
#' \emph{EGUsphere [preprint]}, https://doi.org/10.5194/egusphere-2026-43, 2026.
#' 
#' Freedman, D. and Diaconis, P.: On the histogram as a density estimator: L2 theory,
#' \emph{Zeitschrift fur Wahrscheinlichkeitstheorieund Verwandte Gebiete}, 57, 453-476, https://doi.org/10.1007/BF01025868, 1981.
#' 
#' Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition of the mean squared error and NSE performance criteria:
#' Implications for improving hydrological modelling, \emph{Journal of Hydrology}, 377, 80-91,
#' https://doi.org/10.1016/j.jhydrol.2009.08.003, 2009.
#'
#' Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios, 
#' \emph{Journal of Hydrology}, 424-425, 264-277, https://doi.org/10.1016/j.jhydrol.2012.01.011, 2012
#' 
#' Lin, J.: Divergence measures based on the Shannon entropy, \emph{IEEE Transactions on Information Theory}, 37, 145-151, 
#' https://doi.org/10.1109/18.61115, 1991.
#'
#' @author
#' Andrea Ficchì, Politecnico di Milano, Italy
#'
#' @export
#'
#***********************************************************************************************************************************

jdkge <- function(obs, sim, timestep_s = 86400, verbose = TRUE, check_inputs = FALSE){
  
  if(check_inputs){
    
    if (!is.numeric(obs) || !is.numeric(sim)){
      stop(sprintf(
        "Invalid inputs: 'obs' and 'sim' must be numeric vectors (got obs: %s, sim: %s).",
        paste(class(obs), collapse = "/"),
        paste(class(sim), collapse = "/")
      ), call. = FALSE)
    }
    
    if (length(obs) != length(sim)){
      stop(sprintf(
        "Invalid inputs: 'obs' and 'sim' must have the same length (got length(obs)=%d, length(sim)=%d).",
        length(obs), length(sim)
      ), call. = FALSE)
    }
    
    if (!is.numeric(timestep_s) || length(timestep_s) != 1L || !is.finite(timestep_s) || timestep_s <= 0) {
      stop(sprintf(
        "Invalid input: 'timestep_s' must be a finite numeric scalar > 0 (got %s).",
        paste(timestep_s, collapse=", ")
      ), call. = FALSE)
    }
    
  }
  
  CritName <- "JDKGE"; 
  CritBestValue <- +1;
  
  TS_ignore <- !is.finite(obs) | !is.finite(sim) ;  ## time steps to ignore (TS_ignore)
  
  if(sum(!TS_ignore)==0){ OutputsCrit <- list(NA); names(OutputsCrit) <- c("CritValue"); return(OutputsCrit); }
  if(sum(!TS_ignore)==1){ OutputsCrit <- list(NA); names(OutputsCrit) <- c("CritValue"); return(OutputsCrit); } 
  
  # initialization of variables
  CritValue <- NA 
  SubCritValues <- rep(NA, 4)
  SubCritNames <- c("r", "alpha", "beta", "jsd")
  SubCritPrint <- rep(NA, 4)
  
  meanVarObs <- mean(obs[!TS_ignore])
  meanVarSim <- mean(sim[!TS_ignore])
  
  #####__Compute_the_JDKGE_components_______________________________#####
  
  ### Correlation component (Pearson's r)
  SubCritPrint[1L] <- paste0(CritName, " cor(sim, obs, \"pearson\") =")
  Numer <- sum((obs[!TS_ignore] - meanVarObs) * 
                 (sim[!TS_ignore] - meanVarSim))
  Deno1 <- sqrt(sum((obs[!TS_ignore] - meanVarObs)^2))
  Deno2 <- sqrt(sum((sim[!TS_ignore] - meanVarSim)^2))
  if (Numer == 0) {
    if (Deno1 == 0 & Deno2 == 0) {
      Crit <- 1
    }
    else {
      Crit <- 0
    }
  }
  else {
    Crit <- Numer/(Deno1 * Deno2)
  }
  if (is.numeric(Crit) & is.finite(Crit)) {
    SubCritValues[1L] <- Crit
  }
  
  ### Alpha component (Coeff. of Var. ratio)
  SubCritPrint[2L] <- paste0(CritName, " cv(sim)/cv(obs)          =")
  if (meanVarSim == 0) {
    if (sd(sim[!TS_ignore]) == 0) {
      CVsim <- 1
    }
    else {
      CVsim <- 99999
    }
  }
  else {
    CVsim <- sd(sim[!TS_ignore])/meanVarSim
  }
  if (meanVarObs == 0) {
    if (sd(obs[!TS_ignore]) == 0) {
      CVobs <- 1
    }
    else {
      CVobs <- 99999
    }
  }
  else {
    CVobs <- sd(obs[!TS_ignore])/meanVarObs
  }
  if (CVsim == 0 & CVobs == 0) {
    Crit <- 1
  }
  else {
    Crit <- CVsim/CVobs
  }
  if (is.numeric(Crit) & is.finite(Crit)) {
    SubCritValues[2L] <- Crit
  }
  
  ### Beta component (ratio of means)
  SubCritPrint[3L] <- paste0(CritName, " mean(sim)/mean(obs)      =")
  if (meanVarSim == 0 & meanVarObs == 0) {
    Crit <- 1
  }
  else {
    Crit <- meanVarSim/meanVarObs
  }
  if (is.numeric(Crit) & is.finite(Crit)) {
    SubCritValues[3L] <- Crit
  }
  
  ### JSD component (JSD discretized smoothed estimate with adjusted FD rule)
  SubCritPrint[4L] <- paste0(CritName, " JSD(sim, obs)            =")
  
  Crit <- jsd_fd_log(obs=obs[!TS_ignore], sim=sim[!TS_ignore], timestep_s = timestep_s)
  
  # Safety check 
  if (is.numeric(Crit) & is.finite(Crit)) {
    SubCritValues[4L] <- Crit
  } else {
    SubCritValues[4L] <- 0 # fallback to avoid any NaN in calibration function 
  }
  
  #####__Compute_the_JDKGE_metric___________________________________#####
  if (sum(is.na(SubCritValues)) == 0) {
    CritValue <- (1 - sqrt((SubCritValues[1L] - 1)^2 + 
                             (SubCritValues[2L] - 1)^2 + (SubCritValues[3L] - 1)^2 + (SubCritValues[4L])^2) )
  }
  
  if (verbose) {
    message(sprintf("Crit. %s = %.4f", CritName, CritValue))
    message(paste("\tSubCrit.", SubCritPrint, sprintf("%.4f", 
                                                      SubCritValues), "\n", sep = " "))
  }
  
  OutputsCrit <- list(CritValue = CritValue, CritName = CritName, 
                      SubCritValues = SubCritValues, SubCritNames = SubCritNames, 
                      CritBestValue = CritBestValue)
  
  # Return output list with JDKGE value and the four components
  return(OutputsCrit)
  
}
