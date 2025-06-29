
import numpy as np

def H_z(z, H0, Omega_m, Omega_r, Omega_Lambda):
    return H0 * np.sqrt(
        Omega_r * (1 + z)**4 +
        Omega_m * (1 + z)**3 +
        (1 - Omega_m - Omega_r - Omega_Lambda) * (1 + z)**2 +
        Omega_Lambda
    )
