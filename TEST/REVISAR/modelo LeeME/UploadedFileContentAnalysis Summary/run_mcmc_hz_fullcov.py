#!/usr/bin/env python
# coding: utf-8

# CC fit (flat LCDM) with Full Covariance Matrix

# Adapted from CC_fit.ipynb in https://gitlab.com/mmoresco/CCcovariance

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det, pinv
import emcee
import corner # For plotting MCMC results
import os

# --- 1. Load Data and Full Covariance Matrix ---

# Define directory containing the data
dir_data = "."

# Load H(z) data (z, Hz, stat_err)
# Using HzTable_MM_BC03.dat (15 points)
data_file_hz = os.path.join(dir_data, "HzTable_MM_BC03.dat")
z, H_obs, _ = np.loadtxt(data_file_hz, usecols=(0, 1, 3), unpack=True, skiprows=1) # sigma_stat not needed directly as we use full cov matrix

print(f"Loaded {len(z)} H(z) data points from {data_file_hz}")

# Load the full covariance matrix (Stat + Sys)
cov_matrix_file = "/home/ubuntu/cov_matrix_total_hz.dat"
cov_matrix_total = np.loadtxt(cov_matrix_file)

print(f"Loaded full covariance matrix ({cov_matrix_total.shape[0]}x{cov_matrix_total.shape[1]}) from {cov_matrix_file}")

# Calculate the inverse of the covariance matrix
try:
    inv_cov_matrix = inv(cov_matrix_total)
    print("Successfully inverted the total covariance matrix.")
except np.linalg.LinAlgError:
    print("Error: Could not invert the total covariance matrix. Using pseudo-inverse.")
    # Use pseudo-inverse as a fallback, though this might indicate issues
    inv_cov_matrix = pinv(cov_matrix_total)

# --- 2. Define Cosmological Model and Likelihood ---

# Flat LCDM H(z) model
def Hz_model(params, z):
    H0, Omegam = params
    if Omegam < 0 or Omegam > 1: # Physical bounds
        return np.full_like(z, np.nan)
    return H0 * np.sqrt(Omegam * (1 + z)**3 + (1 - Omegam))

# Log-Likelihood function using the full covariance matrix
def lnlike(params, z, Hz_obs, inv_cov_matr):
    H_model = Hz_model(params, z)
    if np.any(np.isnan(H_model)): # Check for invalid model predictions
        return -np.inf
    
    residual = Hz_obs - H_model
    # Use matrix multiplication for chi2 with full covariance
    chi2 = residual.T @ inv_cov_matr @ residual 
    return -0.5 * chi2

# Define flat priors for the parameters [H0, Omega_m]
def lnprior(params):
    H0, Omegam = params
    # Wide, uninformative priors
    if 50.0 < H0 < 100.0 and 0.01 < Omegam < 0.99:
        return 0.0
    return -np.inf

# Define the posterior probability function
def lnprob(params, z, Hz_obs, inv_cov_matr):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(params, z, Hz_obs, inv_cov_matr)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# --- 3. MCMC Setup and Run ---

# Initial guess for parameters [H0, Omega_m]
initial_params = np.array([70.0, 0.3])
ndim = len(initial_params)  # Number of parameters
nwalkers = 50  # Number of MCMC walkers
nsteps = 2000  # Increased steps for potentially complex posterior
burnin = 500   # Increased burn-in

# Initialize walkers in a small ball around the initial guess
pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)

print(f"Setting up MCMC with {nwalkers} walkers, {nsteps} steps (burn-in: {burnin})...")

# Set up the MCMC sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(z, H_obs, inv_cov_matrix))

# Run the MCMC sampler
sampler.run_mcmc(pos, nsteps, progress=True)

print("MCMC run completed.")

# --- 4. Process Results ---

# Discard burn-in steps and flatten the chain
samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

print(f"Shape of flattened samples after burn-in: {samples.shape}")

# Calculate best-fit parameters (median of the posterior)
H0_mcmc, Omegam_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print(f"Best-fit H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f}")
print(f"Best-fit Omega_m = {Omegam_mcmc[0]:.3f} +{Omegam_mcmc[1]:.3f} / -{Omegam_mcmc[2]:.3f}")

# Calculate min chi2, AIC, BIC using the median parameters
best_params = [H0_mcmc[0], Omegam_mcmc[0]]
min_lnlike = lnlike(best_params, z, H_obs, inv_cov_matrix)
min_chi2 = -2 * min_lnlike
n_data = len(z) # Should be 15
k_params = ndim

AIC = min_chi2 + 2 * k_params
BIC = min_chi2 + k_params * np.log(n_data)

print(f"Minimum Chi2 (at median params): {min_chi2:.2f}")
print(f"AIC: {AIC:.2f}")
print(f"BIC: {BIC:.2f}")

# --- 5. Save Results and Plot ---

# Save samples
np.save("/home/ubuntu/mcmc_samples_hz_fullcov.npy", samples)

# Create corner plot
fig = corner.figure(samples, labels=["H$_0$", "$\Omega_m$"],
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
plot_path = "/home/ubuntu/mcmc_corner_plot_hz_fullcov.png"
plt.savefig(plot_path)
print(f"Corner plot saved to {plot_path}")

# Save summary statistics
summary_content = f"""
# MCMC Fit Results (Flat LCDM with H(z) Data - Full Covariance)

Data Points (n): {n_data}
Parameters (k): {k_params}

Best Fit (Median +/- 1-sigma):
H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f} km/s/Mpc
Omega_m = {Omegam_mcmc[0]:.3f} +{Omegam_mcmc[1]:.3f} / -{Omegam_mcmc[2]:.3f}

Goodness of Fit (at median params):
Min Chi2 = {min_chi2:.2f}
Chi2/dof = {min_chi2 / (n_data - k_params):.2f}
AIC = {AIC:.2f}
BIC = {BIC:.2f}
"""
summary_file = "/home/ubuntu/mcmc_fit_summary_hz_fullcov.txt"
with open(summary_file, "w") as f:
    f.write(summary_content)
print(f"Summary statistics saved to {summary_file}")


