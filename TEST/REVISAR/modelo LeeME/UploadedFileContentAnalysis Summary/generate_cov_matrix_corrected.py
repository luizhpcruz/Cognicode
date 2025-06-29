#!/usr/bin/env python
# coding: utf-8

# Generate Full Covariance Matrix for H(z) Data

# Adapted from CC_covariance.ipynb in https://gitlab.com/mmoresco/CCcovariance

import numpy as np
import os
from scipy.interpolate import interp1d

# --- 1. Load Data ---

# Define directory containing the data
dir_data = "."

# Load H(z) data (z, Hz, stat_err)
# Using HzTable_MM_BC03.dat which contains 15 points.
# Columns: 0:z, 1:Hz, 3:stat_contr
data_file_hz = os.path.join(dir_data, "HzTable_MM_BC03.dat")
z_hz, Hz, sigma_stat = np.loadtxt(data_file_hz, usecols=(0, 1, 3), unpack=True, skiprows=1)

print(f"Loaded {len(z_hz)} H(z) data points (z, Hz, sigma_stat) from {data_file_hz}")

# Load systematic error components data (z_sys, sys_err_perc%)
# Using data_MM20.dat which contains the systematic components over a redshift range
# Corrected Columns: 0:z, 1:IMF_err%, 2:stlib_err%, 3:mod_err%(SFH?), 4:mod_ooo_err%(SPS?)
data_file_sys = os.path.join(dir_data, "hz_data_moresco2020.dat")
z_sys, imf_err_perc, stlib_err_perc, mod_err_perc, mod_ooo_err_perc = np.loadtxt(
    data_file_sys, usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=1
)

print(f"Loaded systematic error components from {data_file_sys}")

# --- 2. Interpolate Systematic Errors to Data Redshifts ---

# Create interpolation functions for each systematic component percentage
interp_imf = interp1d(z_sys, imf_err_perc, kind="linear", fill_value="extrapolate")
interp_stlib = interp1d(z_sys, stlib_err_perc, kind="linear", fill_value="extrapolate")
interp_mod = interp1d(z_sys, mod_err_perc, kind="linear", fill_value="extrapolate")
interp_mod_ooo = interp1d(z_sys, mod_ooo_err_perc, kind="linear", fill_value="extrapolate")

# Interpolate to the redshifts of the H(z) data points (z_hz)
imf_err_perc_interp = interp_imf(z_hz)
stlib_err_perc_interp = interp_stlib(z_hz)
mod_err_perc_interp = interp_mod(z_hz)
mod_ooo_err_perc_interp = interp_mod_ooo(z_hz)

# --- 3. Calculate Statistical Covariance Matrix (Diagonal) ---
cov_mat_stat = np.diag(sigma_stat**2)

# --- 4. Calculate Systematic Covariance Matrix Components ---

# Convert interpolated percentage errors to fractional errors relative to H(z)
imf_frac = imf_err_perc_interp / 100.0
stlib_frac = stlib_err_perc_interp / 100.0
mod_frac = mod_err_perc_interp / 100.0
mod_ooo_frac = mod_ooo_err_perc_interp / 100.0

# Initialize systematic component matrices
n_points = len(z_hz)
cov_mat_imf = np.zeros((n_points, n_points))
cov_mat_stlib = np.zeros((n_points, n_points))
cov_mat_mod = np.zeros((n_points, n_points))
cov_mat_mod_ooo = np.zeros((n_points, n_points))

# Calculate systematic components assuming full correlation
for i in range(n_points):
    for j in range(n_points):
        cov_mat_imf[i, j] = (Hz[i] * imf_frac[i]) * (Hz[j] * imf_frac[j])
        cov_mat_stlib[i, j] = (Hz[i] * stlib_frac[i]) * (Hz[j] * stlib_frac[j])
        cov_mat_mod[i, j] = (Hz[i] * mod_frac[i]) * (Hz[j] * mod_frac[j])
        cov_mat_mod_ooo[i, j] = (Hz[i] * mod_ooo_frac[i]) * (Hz[j] * mod_ooo_frac[j])

# Sum systematic components (Cov_model part)
cov_mat_syst_model = cov_mat_imf + cov_mat_stlib + cov_mat_mod + cov_mat_mod_ooo

# Add diagonal systematic components (Cov_met)
# Load met_contr from HzTable_MM_BC03.dat (column 4)
# *** CORRECTED unpacking: Load only the 4 specified columns ***
_, _, _, met_contr = np.loadtxt(data_file_hz, usecols=(0, 1, 3, 4), unpack=True, skiprows=1)
cov_mat_met = np.diag(met_contr**2) # Assuming met_contr is the error, not variance

# Total Systematic Covariance (excluding Cov_young which is not readily available here)
cov_mat_syst = cov_mat_syst_model + cov_mat_met

# --- 5. Calculate Total Covariance Matrix ---
cov_mat_total = cov_mat_stat + cov_mat_syst

# --- 6. Save the Total Covariance Matrix ---
cov_output_file = "/home/ubuntu/cov_matrix_total_hz.dat"
np.savetxt(cov_output_file, cov_mat_total)

print(f"Total covariance matrix ({n_points}x{n_points}) including interpolated systematics saved to {cov_output_file}")

# Optional: Verify matrix properties
try:
    np.linalg.cholesky(cov_mat_total)
    print("Verification: Total covariance matrix is positive definite.")
except np.linalg.LinAlgError:
    print("Warning: Total covariance matrix is not positive definite. Check calculations or data.")


