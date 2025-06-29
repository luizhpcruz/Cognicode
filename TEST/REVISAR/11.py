# 10.py - Script atualizado com correções e comparação bayesiana dos modelos cosmológicos

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.integrate import solve_ivp

# Dados simulados (z, H(z), erro)
data = np.array([
    [0.07, 69, 19.6],
    [0.12, 68.6, 26.2],
    [0.2, 72.9, 29.6],
    [0.28, 88.8, 36.6],
    [0.4, 95, 17],
    [0.6, 98.3, 14.1],
    [0.8, 103.9, 15.2],
    [1.0, 120, 23.6],
    [1.3, 168, 17.4],
    [1.43, 177, 18.2],
    [1.53, 140, 14],
    [1.75, 202, 40.4],
    [2.3, 224, 8.6],
    [2.34, 222, 8.5],
    [2.36, 226, 9.3],
])

z_data, H_data, H_err = data.T

# Modelo Lambda-CDM
def H_lcdm(z, H0, Omega_m):
    Omega_lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z) ** 3 + Omega_lambda)

def log_likelihood_lcdm(theta):
    H0, Omega_m = theta
    model = H_lcdm(z_data, H0, Omega_m)
    return -0.5 * np.sum(((H_data - model) / H_err) ** 2)

def log_prior_lcdm(theta):
    H0, Omega_m = theta
    if 40 < H0 < 100 and 0 < Omega_m < 1:
        return 0.0
    return -np.inf

def log_probability_lcdm(theta):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(theta)

# Modelo Modificado

def H_mod(z, H0, Omega_m, rond0, n):
    Omega_lambda = 1.0 - Omega_m - rond0
    arg = Omega_m * (1 + z) ** 3 + Omega_lambda + rond0 * (1 + z) ** (2 * n)
    if np.any(arg < 0):
        return np.inf
    return H0 * np.sqrt(arg)

def log_likelihood_mod(theta):
    H0, Omega_m, rond0, n = theta
    model = H_mod(z_data, H0, Omega_m, rond0, n)
    if np.any(model == np.inf):
        return -np.inf
    return -0.5 * np.sum(((H_data - model) / H_err) ** 2)

def log_prior_mod(theta):
    H0, Omega_m, rond0, n = theta
    if 40 < H0 < 100 and 0 < Omega_m < 1 and 0 < rond0 < 3 and 0 < n < 5:
        return 0.0
    return -np.inf

def log_probability_mod(theta):
    lp = log_prior_mod(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_mod(theta)

# Configuração do MCMC
ndim_lcdm = 2
ndim_mod = 4
nwalkers = 50
nsteps = 5000

# Lambda-CDM
print("\n--- Simulação para o Modelo Lambda-CDM ---")
initial_lcdm = np.array([70, 0.3])
pos_lcdm = initial_lcdm + 1e-4 * np.random.randn(nwalkers, ndim_lcdm)
sampler_lcdm = emcee.EnsembleSampler(nwalkers, ndim_lcdm, log_probability_lcdm)
sampler_lcdm.run_mcmc(pos_lcdm, nsteps, progress=True)

samples_lcdm = sampler_lcdm.get_chain(discard=1000, thin=15, flat=True)
fig1 = corner.corner(samples_lcdm, labels=["H0", "Omega_m"], truths=[None, None])
plt.show()

H0_lcdm_mcmc, Omega_m_lcdm_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                      zip(*np.percentile(samples_lcdm, [16, 50, 84], axis=0)))
print("\nResultados do ajuste bayesiano (Lambda-CDM):")
print(f"H0 = {H0_lcdm_mcmc[0]:.3f} +{H0_lcdm_mcmc[1]:.3f} -{H0_lcdm_mcmc[2]:.3f}")
print(f"Omega_m = {Omega_m_lcdm_mcmc[0]:.3f} +{Omega_m_lcdm_mcmc[1]:.3f} -{Omega_m_lcdm_mcmc[2]:.3f}")

model_lcdm_fit = H_lcdm(z_data, H0_lcdm_mcmc[0], Omega_m_lcdm_mcmc[0])
chi2_lcdm = np.sum(((H_data - model_lcdm_fit) / H_err) ** 2)
print(f"Chi^2 (Lambda-CDM) = {chi2_lcdm:.2f}")

# Modelo Modificado
print("\n--- Simulação para o Modelo de Energia Escura Modificada ---")
initial_mod = np.array([70, 0.3, 1.0, 1.0])
pos_mod = initial_mod + 1e-4 * np.random.randn(nwalkers, ndim_mod)
sampler_mod = emcee.EnsembleSampler(nwalkers, ndim_mod, log_probability_mod)
sampler_mod.run_mcmc(pos_mod, nsteps, progress=True)

samples_mod = sampler_mod.get_chain(discard=1000, thin=15, flat=True)
fig2 = corner.corner(samples_mod, labels=["H0", "Omega_m", "Rond0", "n"], truths=[None]*4)
plt.show()

H0_mod_mcmc, Omega_m_mod_mcmc, rond0_mcmc, n_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                         zip(*np.percentile(samples_mod, [16, 50, 84], axis=0)))
print("\nResultados do ajuste bayesiano (Modelo Modificado):")
print(f"H0 = {H0_mod_mcmc[0]:.3f} +{H0_mod_mcmc[1]:.3f} -{H0_mod_mcmc[2]:.3f}")
print(f"Omega_m = {Omega_m_mod_mcmc[0]:.3f} +{Omega_m_mod_mcmc[1]:.3f} -{Omega_m_mod_mcmc[2]:.3f}")
print(f"Rond0 = {rond0_mcmc[0]:.3f} +{rond0_mcmc[1]:.3f} -{rond0_mcmc[2]:.3f}")
print(f"n = {n_mcmc[0]:.3f} +{n_mcmc[1]:.3f} -{n_mcmc[2]:.3f}")

model_mod_fit = H_mod(z_data, H0_mod_mcmc[0], Omega_m_mod_mcmc[0], rond0_mcmc[0], n_mcmc[0])
chi2_mod = np.sum(((H_data - model_mod_fit) / H_err) ** 2)
print(f"Chi^2 (Modelo Modificado) = {chi2_mod:.2f}")
