
import numpy as np
from modules.cosmology.friedmann import H_z

def chi_squared(observed_z, observed_H, errors, H0, Omega_m, Omega_r, Omega_Lambda):
    model_H = H_z(observed_z, H0, Omega_m, Omega_r, Omega_Lambda)
    return np.sum(((observed_H - model_H) / errors)**2)
