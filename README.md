# Joint Divergence Kling-Gupta Efficiency (JDKGE)

This repository hosts the source code of the Joint Divergence Kling-Gupta Efficiency (JDKGE) metric, proposed to improve hydrological model calibration by enhancing performance across both low- and high-flow conditions.

JDKGE enhances the Kling-Gupta Efficiency (KGE) by incorporating an additional component based on the Jensen–Shannon Divergence (JSD), used as a measure of distributional similarity between observed and simulated streamflows. The JSD component is estimated from the empirical distributions obtained by partitioning (binning) the data domain into intervals (bins). Bin widths are determined using an adapted Freedman-Diaconis (FD) rule, adjusted to ensure invariance with respect to the model timestep. Log-transformed flows are used in the JSD component to place greater emphasis on low flows when computing the distributional similarity.

For a detailed methodological description and evaluation results, please refer to the manuscript currently under review: 

Ficchì, A., Bavera, D., Grimaldi, S., Moschini, F., Pistocchi, A., Russo, C., Salamon, P., and Toreti, A.: Improving low and high flow simulations at once: An enhanced metric for hydrological model calibration, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-43, 2026.
