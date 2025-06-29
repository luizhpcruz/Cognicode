"""Equações do modelo vetorial modificado"""

import numpy as np
# Assumindo que 'parametros.py' foi renomeado para 'model_parameters'
from model_parameters import vector_params_default

# Parâmetros padrão do campo vetorial
Mpl = 1.0          # Massa de Planck (unidade natural)
mA = 0.1           # Massa do campo vetorial
xi = 1.2           # Acoplamento não-mínimo com curvatura

def rho_ond(a, R0=vector_params_default["R0"], n=vector_params_default["n"]):
    """Densidade de energia transiente."""
    return R0 * a**(-n)

def H_squared_naive(a, A, dA, Omega_m, Omega_r):
    """Usado para evitar recursão circular na equação de Friedmann."""
    return (Omega_m/a**3 + Omega_r/a**4 + 0.5*(dA**2 + mA**2 * A**2) + rho_ond(a)) / (3*Mpl**2)

def dH_dt(a, A, dA, H, Omega_m, Omega_r):
    """Calcula a derivada de H para a curvatura escalar."""
    epsilon = 1e-5
    H_plus = np.sqrt(H_squared_naive(a + epsilon, A, dA, Omega_m, Omega_r))
    H_minus = np.sqrt(H_squared_naive(a - epsilon, A, dA, Omega_m, Omega_r))
    da_dt = H * a
    return (H_plus - H_minus) / (2 * epsilon) * da_dt / a

def deriv(t, y, Omega_m, Omega_r):
    """Equações diferenciais para a integração (y = [a, A, dA/dt])."""
    a, A, dA = y
    H = np.sqrt(H_squared_naive(a, A, dA, Omega_m, Omega_r))
    R = 6 * (dH_dt(a, A, dA, H, Omega_m, Omega_r) + 2 * H**2)
    dadt = H * a
    d2A = -3 * H * dA - (mA**2 + xi * R) * A
    return [dadt, dA, d2A]