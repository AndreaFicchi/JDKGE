# ============================================================
# Example usage of the JDKGE metric (with airGR sample data)
# Basin: L0123001  
# Observed streamflow: BasinObs$Qmm
# Simulated streamflow: synthetic perturbation of BasinObs$Qmm
# ============================================================

#####__Packages____________________________________________________________#####

if (!requireNamespace("airGR", quietly = TRUE)){
  install.packages("airGR")
}

if(!requireNamespace("here", quietly = TRUE)){
  install.packages("here")
}

if(!requireNamespace("hydroGOF", quietly = TRUE)){
  install.packages("hydroGOF")
}

library(airGR)
library(here)
library(hydroGOF)

#####__Load_data___________________________________________________________#####

cat("Project root detected by here():\n")
print(here())

# Source functions from the same repository
source(here("R", "jdkge.R"))
source(here("R", "jsd_fd_log.R"))

# IMPORTANT: check that the jdkge() and jsd_fd_log() functions have been loaded,
# and adjust paths if needed; e.g., 
# source("R/jsd_fd_log.R"); source("R/jdkge.R");

data(L0123001, package = "airGR")
summary(BasinObs, digits = 2)
print(paste("Streamflow observations (Qobs, mm/d) loaded into BasinObs$Qmm (", class(BasinObs$Qmm), ")", sep="" ) ) 

# Observed streamflow (mm/day)
Qobs_w_gaps <- BasinObs$Qmm

# Plot Qobs
plot(Qobs_w_gaps, type="l", ylab="Streamflow (mm/d)")

# Check presence of missing values
print(paste("Missing values: ", sum(is.na(Qobs_w_gaps)), sep="" ) )

# Keep finite, non-negative values only
idx_ok <- is.finite(Qobs_w_gaps) & (Qobs_w_gaps >= 0)
Qobs <- Qobs_w_gaps[idx_ok]

#####__Create_synthetic_simulation_data____________________________________#####

# We perturb observations with multiplicative noise + mild bias
set.seed(123)

bias_factor <- 1.05          # positive bias 
cv_noise    <- 0.25          # coefficient of variation of multiplicative noise
noise_mult  <- exp(rnorm(length(Qobs_w_gaps), mean = 0, sd = sqrt(log(1 + cv_noise^2))))

Qsim_w_gaps <- bias_factor * Qobs_w_gaps * noise_mult

# We further perturb low flows with additive noise
q_thr <- quantile(Qobs_w_gaps, probs = 0.25, na.rm = TRUE, type = 7)
add_noise_sd <- 0.2 * median(Qobs_w_gaps[Qobs_w_gaps > 0], na.rm = TRUE)

# Smooth low-flow weight
lf_w <- numeric(length(Qobs_w_gaps))
lf_w <- plogis((q_thr - Qobs_w_gaps) / (q_thr*0.15) )
lf_w[is.na(lf_w)] <- 0

# extra multiplicative and additive bias concentrated in the low-flow tail
Qsim_w_gaps <- Qsim_w_gaps*(1+lf_w*0.5) + rnorm(length(Qobs_w_gaps), mean = 0.9*q_thr, sd = add_noise_sd)*lf_w

# Enforce non-negativity
Qsim_w_gaps <- pmax(Qsim_w_gaps, 0)

# Plot Qsim
lines(Qsim_w_gaps, col="red", cex=0.25)
Qsim <- Qsim_w_gaps[idx_ok]

#####__Compute_the_JDKGE_metric____________________________________________#####

timestep_s <- 86400  # daily data

jdkge_out <- jdkge(obs = Qobs, sim = Qsim, timestep_s = timestep_s, verbose = TRUE)


#####__Print_outputs_and_compare_JDKGE_vs_KGE'_____________________________#####

# Print JDKGE outputs
cat("\n--- JDKGE results ---\n")
print(paste(jdkge_out$CritName, " = ", round(jdkge_out$CritValue, 5), sep=""), quote=F)
print(data.frame(Component = jdkge_out$SubCritNames, Value = jdkge_out$SubCritValues))

# Compare with KGE' (hydroGOF) 
cat("\n--- KGE' results ---\n")
kge_out <- KGE(sim=Qsim, obs=Qobs, s=c(1,1,1), na.rm=TRUE, method= "2012",
               out.type = "full")
print(paste("KGE' = ", round(kge_out$KGE.value, 5), sep=""), quote=F)
names(kge_out$KGE.elements)[names(kge_out$KGE.elements)=="Gamma"] <- "alpha" # make KGE components names consistent with JDKGE case
print(kge_out$KGE.elements)

