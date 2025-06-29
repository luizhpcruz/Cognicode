# modelo.py

import numpy as np
from parametros import Mpl, mA, xi, Omega_m, Omega_r, Omega_ond0, n

def rho_ond(a):
    return Omega_ond0 * a**(-n)

def H_squared(a, A, dA):
    rho_A = 0.5 * (dA**2 + mA**2 * A**2) + 3 * xi * (H_squared_naive(a, A, dA)) * A**2
    return (Omega_m/a**3 + Omega_r/a**4 + rho_A + rho_ond(a)) / (3 * Mpl**2)

def H_squared_naive(a, A, dA):  # usado apenas dentro de H_squared para evitar recursão circular
    return (Omega_m/a**3 + Omega_r/a**4 + 0.5*(dA**2 + mA**2 * A**2) + rho_ond(a)) / (3*Mpl**2)

def dH_dt(a, A, dA, H):
    epsilon = 1e-5
    H_plus = np.sqrt(H_squared_naive(a + epsilon, A, dA))
    H_minus = np.sqrt(H_squared_naive(a - epsilon, A, dA))
    da_dt = H * a
    return (H_plus - H_minus) / (2 * epsilon) * da_dt / a

def deriv(t, y):
    a, A, dA = y
    H = np.sqrt(H_squared_naive(a, A, dA))
    R = 6 * (dH_dt(a, A, dA, H) + 2 * H**2)
    dadt = H * a
    d2A = -3 * H * dA - (mA**2 + xi * R) * A
    return [dadt, dA, d2A]
