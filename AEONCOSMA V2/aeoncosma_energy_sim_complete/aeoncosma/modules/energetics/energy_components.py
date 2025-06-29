
import numpy as np

def rho_r(z, Omega_r, H0):
    return Omega_r * (1 + z)**4 * H0**2

def rho_m(z, Omega_m, H0):
    return Omega_m * (1 + z)**3 * H0**2

def rho_lambda(Omega_Lambda, H0):
    return Omega_Lambda * H0**2
