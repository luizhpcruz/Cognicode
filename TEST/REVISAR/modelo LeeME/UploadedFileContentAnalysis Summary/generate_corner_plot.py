#!/usr/bin/env python
# coding: utf-8

# Generate Corner Plot from Saved MCMC Samples

import numpy as np
import matplotlib.pyplot as plt
import corner
import os

# --- 1. Load Saved MCMC Samples ---
samples_file = "/home/ubuntu/mcmc_samples_hz_fullcov.npy"
samples = np.load(samples_file)

print(f"Loaded MCMC samples ({samples.shape}) from {samples_file}")

# --- 2. Create Corner Plot ---

# Corrected function call: corner.corner
fig = corner.corner(samples, labels=["H$_0$", "$\Omega_m$"],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})

# Add Planck 2018 values for comparison (optional)
planck_H0 = 67.4
planck_H0_err = 0.5
planck_Om = 0.315
planck_Om_err = 0.007
# You can uncomment these lines to add Planck contours/points
# corner.overplot_lines(fig, [planck_H0, planck_Om], color="blue", linestyle="dashed")
# corner.overplot_points(fig, [[planck_H0, planck_Om]], marker="s", color="blue")

plt.suptitle("MCMC Posterior for Flat $\Lambda$CDM from H(z) Data (Full Covariance)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout
plot_path = "/home/ubuntu/mcmc_corner_plot_hz_fullcov_corrected.png"
plt.savefig(plot_path)
print(f"Corrected corner plot saved to {plot_path}")

