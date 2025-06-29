
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import emcee
import corner
import os

# --- Carregamento dos dados reais de H(z) ---
data = pd.read_csv("data/hz_data_real.csv")
z_data = data["z"].values
Hz_data = data["Hz"].values
Hz_err = data["err"].values

# --- Definição do modelo cosmológico CPL + EDE ---
def w_eff(z, w0, wa, f_EDE, z_c=3000):
    cpl = w0 + wa * z / (1 + z)
    tanh_term = 0.5 * (1 - np.tanh(5 * np.log((1 + z) / z_c)))
    return f_EDE * tanh_term + (1 - f_EDE) * cpl

def Omega_DE(z, Omega_de0, w0, wa, f_EDE):
    if z <= 1e-9:
        return Omega_de0
    integrand = lambda zp: (1 + w_eff(zp, w0, wa, f_EDE)) / (1 + zp)
    integral, _ = quad(integrand, 0, z)
    return Omega_de0 * np.exp(-3 * integral)

def H_model(z, H0, Omega_m, w0, wa, f_EDE):
    Hz = []
    for zi in z:
        Omega_de_z = Omega_DE(zi, 1.0 - Omega_m, w0, wa, f_EDE)
        Hz.append(H0 * np.sqrt(Omega_m * (1 + zi)**3 + Omega_de_z))
    return np.array(Hz)

def ln_likelihood(theta):
    H0, Omega_m, w0, wa, f_EDE = theta
    model = H_model(z_data, H0, Omega_m, w0, wa, f_EDE)
    return -0.5 * np.sum(((Hz_data - model) / Hz_err)**2)

def ln_prior(theta):
    H0, Omega_m, w0, wa, f_EDE = theta
    if 50 < H0 < 90 and 0.1 < Omega_m < 0.5 and -2 < w0 < -0.3 and -2 < wa < 2 and 0 <= f_EDE <= 0.5:
        return 0.0
    return -np.inf

def ln_posterior(theta):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta)

# --- Configuração do MCMC ---
ndim = 5
nwalkers = 32
nsteps = 3000
initial = [70.0, 0.3, -1.0, 0.2, 0.1]
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

# --- Execução ---
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior)
sampler.run_mcmc(pos, nsteps, progress=True)

# --- Salvando resultados ---
samples = sampler.get_chain(discard=1000, thin=15, flat=True)
np.save("report/mcmc_samples.npy", samples)

# --- Gerando corner plot ---
fig = corner.corner(samples, labels=["H0", "Ωm", "w0", "wa", "f_EDE"], 
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig("report/figs/corner_mcmc.png")
