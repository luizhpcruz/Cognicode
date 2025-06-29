
from .energy_components import rho_r, rho_m, rho_lambda

def total_rho(z, Omega_r, Omega_m, Omega_Lambda, H0):
    return rho_r(z, Omega_r, H0) + rho_m(z, Omega_m, H0) + rho_lambda(Omega_Lambda, H0)
