#!/usr/bin/env python
# coding: utf-8

# Generate Full Covariance Matrix for H(z) Data

# Adapted from CC_covariance.ipynb in https://gitlab.com/mmoresco/CCcovariance

import numpy as np
import os

# --- 1. Load Data ---

# Define directory containing the data
dir_data = "."

# Load H(z) data including systematic error components
# Columns: 0:z, 1:IMF_err%, 2:stlib_err%, 3:H(z), 4:stat_err(sigma_H), 5:mod_err%(SFH?), 6:mod_ooo_err%(SPS?)
# Note: The exact mapping of columns 5 and 6 to SFH/SPS might need verification from Moresco's papers,
# but we follow the structure suggested by the variable names in the notebook snippet.
data_file = os.path.join(dir_data, "hz_data_moresco2020.dat")
z, imf_err_perc, stlib_err_perc, Hz, sigma_stat, mod_err_perc, mod_ooo_err_perc = np.loadtxt(
    data_file, usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True
)

print(f"Loaded {len(z)} H(z) data points and systematic error components.")

# --- 2. Calculate Statistical Covariance Matrix (Diagonal) ---
cov_mat_stat = np.diag(sigma_stat**2)

# --- 3. Calculate Systematic Covariance Matrix Components ---

# Convert percentage errors to fractional errors relative to H(z)
imf_frac = imf_err_perc / 100.0
stlib_frac = stlib_err_perc / 100.0
mod_frac = mod_err_perc / 100.0      # Assuming col 5 is SFH/mod
mod_ooo_frac = mod_ooo_err_perc / 100.0 # Assuming col 6 is SPS/mod_ooo

# Initialize systematic component matrices
cov_mat_imf = np.zeros_like(cov_mat_stat)
cov_mat_stlib = np.zeros_like(cov_mat_stat)
cov_mat_mod = np.zeros_like(cov_mat_stat)
cov_mat_mod_ooo = np.zeros_like(cov_mat_stat)

# Calculate systematic components assuming full correlation (conservative approach)
# Cov[i,j] = (H[i] * sys_err_frac[i]) * (H[j] * sys_err_frac[j])
for i in range(len(z)):
    for j in range(len(z)):
        cov_mat_imf[i, j] = (Hz[i] * imf_frac[i]) * (Hz[j] * imf_frac[j])
        cov_mat_stlib[i, j] = (Hz[i] * stlib_frac[i]) * (Hz[j] * stlib_frac[j])
        cov_mat_mod[i, j] = (Hz[i] * mod_frac[i]) * (Hz[j] * mod_frac[j])
        cov_mat_mod_ooo[i, j] = (Hz[i] * mod_ooo_frac[i]) * (Hz[j] * mod_ooo_frac[j])

# Sum systematic components
# Note: The repo's README mentions Cov_met and Cov_young as diagonal, but the data file doesn't seem
# to explicitly list them. We are using the components available in the data file (IMF, stlib, mod, mod_ooo)
# which contribute to Cov_model in the README's notation.
cov_mat_syst = cov_mat_imf + cov_mat_stlib + cov_mat_mod + cov_mat_mod_ooo

# --- 4. Calculate Total Covariance Matrix ---
cov_mat_total = cov_mat_stat + cov_mat_syst

# --- 5. Save the Total Covariance Matrix ---
cov_output_file = "/home/ubuntu/cov_matrix_total_hz.dat"
np.savetxt(cov_output_file, cov_mat_total)

print(f"Total covariance matrix ({len(z)}x{len(z)}) saved to {cov_output_file}")

# Optional: Verify matrix properties (e.g., positive definite)
try:
    np.linalg.cholesky(cov_mat_total)
    print("Verification: Total covariance matrix is positive definite.")
except np.linalg.LinAlgError:
    print("Warning: Total covariance matrix is not positive definite. Check calculations or data.")


