
import numpy as np
from modules.energetics.observational_constraints import chi_squared

# Dados de exemplo
z = np.array([0.01, 0.1])
H_obs = np.array([70.0, 75.0])
err = np.array([5.0, 5.0])

H0 = 67.66
Omega_m = 0.3111
Omega_r = 5e-5
Omega_Lambda = 0.6889

chi2 = chi_squared(z, H_obs, err, H0, Omega_m, Omega_r, Omega_Lambda)
print(f"Chi² = {chi2:.2f}")
