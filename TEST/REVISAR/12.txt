import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import emcee
import corner
from scipy.interpolate import interp1d

# Dados observacionais simulados (poderiam ser do Pantheon+ ou BAO reais)
z_data = np.linspace(0.01, 1.5, 30)
H_data = 70 * (1 + z_data)**0.5 + np.random.normal(0, 2, size=z_data.shape)
sigma_H = np.full_like(z_data, 2.0)

# Modelo Lambda-CDM
def H_LCDM(z, H0, Omega_m):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

# Modelo Modificado com termo Rond0*(1+z)^n
def H_mod(z, H0, Omega_m, Rond0, n):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m - Rond0) + Rond0 * (1 + z)**n)

# Likelihood para o modelo Lambda-CDM
def log_likelihood_lcdm(theta):
    H0, Omega_m = theta
    model = H_LCDM(z_data, H0, Omega_m)
    return -0.5 * np.sum(((H_data - model)/sigma_H)**2)

# Likelihood para o modelo modificado
def log_likelihood_mod(theta):
    H0, Omega_m, Rond0, n = theta
    model = H_mod(z_data, H0, Omega_m, Rond0, n)
    return -0.5 * np.sum(((H_data - model)/sigma_H)**2)

# Priors

def log_prior_lcdm(theta):
    H0, Omega_m = theta
    if 40 < H0 < 100 and 0 < Omega_m < 1:
        return 0.0
    return -np.inf

def log_prior_mod(theta):
    H0, Omega_m, Rond0, n = theta
    if 40 < H0 < 100 and 0 < Omega_m < 1 and 0 < Rond0 < 3 and 0 < n < 3:
        return 0.0
    return -np.inf

# Posterior

def log_posterior_lcdm(theta):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(theta)

def log_posterior_mod(theta):
    lp = log_prior_mod(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_mod(theta)

# MCMC para Lambda-CDM
ndim_lcdm, nwalkers = 2, 50
pos_lcdm = [np.array([70, 0.3]) + 1e-4*np.random.randn(ndim_lcdm) for _ in range(nwalkers)]
sampler_lcdm = emcee.EnsembleSampler(nwalkers, ndim_lcdm, log_posterior_lcdm)
print("--- Simulação para o Modelo Lambda-CDM ---")
sampler_lcdm.run_mcmc(pos_lcdm, 5000, progress=True)
samples_lcdm = sampler_lcdm.get_chain(discard=1000, flat=True)

# MCMC para Modelo Modificado
ndim_mod = 4
pos_mod = [np.array([70, 0.3, 1.0, 1.0]) + 1e-4*np.random.randn(ndim_mod) for _ in range(nwalkers)]
sampler_mod = emcee.EnsembleSampler(nwalkers, ndim_mod, log_posterior_mod)
print("\n--- Simulação para o Modelo de Energia Escura Modificada ---")
sampler_mod.run_mcmc(pos_mod, 5000, progress=True)
samples_mod = sampler_mod.get_chain(discard=1000, flat=True)

# Estatísticas dos resultados
import warnings
warnings.filterwarnings("ignore")
def print_summary(samples, labels):
    for i, label in enumerate(labels):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{label} = {mcmc[1]:.3f} +{q[1]:.3f} -{q[0]:.3f}")

def chi2(theta, model_func):
    model = model_func(z_data, *theta)
    return np.sum(((H_data - model)/sigma_H)**2)

print("\nResultados do ajuste bayesiano (Lambda-CDM):")
print_summary(samples_lcdm, ["H0", "Omega_m"])
print(f"Chi^2 (Lambda-CDM) = {chi2(np.median(samples_lcdm, axis=0), H_LCDM):.2f}")

print("\nResultados do ajuste bayesiano (Modelo Modificado):")
print_summary(samples_mod, ["H0", "Omega_m", "Rond0", "n"])
print(f"Chi^2 (Modelo Modificado) = {chi2(np.median(samples_mod, axis=0), H_mod):.2f}")

# Gráficos
fig1 = corner.corner(samples_lcdm, labels=["H0", "Omega_m"], color="black")
fig1.savefig("revisar11_lcdm.png")

fig2 = corner.corner(samples_mod, labels=["H0", "Omega_m", "Rond0", "n"], color="black")
fig2.savefig("revisar11_modificado.png")
