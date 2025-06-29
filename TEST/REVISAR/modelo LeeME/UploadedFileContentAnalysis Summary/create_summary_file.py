#!/usr/bin/env python
# coding: utf-8

# Create Summary Statistics File from Saved MCMC Samples

import numpy as np
from numpy.linalg import inv, pinv
import os

# --- 1. Load Data, Samples, and Covariance Matrix ---

# Define directory containing the data
dir_data = "."

# Load H(z) data (z, Hz)
data_file_hz = os.path.join(dir_data, "HzTable_MM_BC03.dat")
z, H_obs, _ = np.loadtxt(data_file_hz, usecols=(0, 1, 3), unpack=True, skiprows=1)

# Load the full covariance matrix
cov_matrix_file = "/home/ubuntu/cov_matrix_total_hz.dat"
cov_matrix_total = np.loadtxt(cov_matrix_file)

# Calculate the inverse of the covariance matrix
try:
    inv_cov_matrix = inv(cov_matrix_total)
except np.linalg.LinAlgError:
    inv_cov_matrix = pinv(cov_matrix_total)

# Load MCMC samples
samples_file = "/home/ubuntu/mcmc_samples_hz_fullcov.npy"
samples = np.load(samples_file)

print(f"Loaded data, covariance matrix, and {len(samples)} MCMC samples.")

# --- 2. Define Likelihood Function (copied from MCMC script) ---

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

# --- 3. Calculate Statistics ---

# Calculate best-fit parameters (median of the posterior)
H0_mcmc, Omegam_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

# Calculate min chi2, AIC, BIC using the median parameters
best_params = [H0_mcmc[0], Omegam_mcmc[0]]
min_lnlike = lnlike(best_params, z, H_obs, inv_cov_matrix)
min_chi2 = -2 * min_lnlike
n_data = len(z) # Should be 15
k_params = 2 # H0, Omega_m

AIC = min_chi2 + 2 * k_params
BIC = min_chi2 + k_params * np.log(n_data)

print("Recalculated statistics:")
print(f"Best-fit H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f}")
print(f"Best-fit Omega_m = {Omegam_mcmc[0]:.3f} +{Omegam_mcmc[1]:.3f} / -{Omegam_mcmc[2]:.3f}")
print(f"Minimum Chi2 (at median params): {min_chi2:.2f}")
print(f"AIC: {AIC:.2f}")
print(f"BIC: {BIC:.2f}")

# --- 4. Save Summary Statistics ---

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


