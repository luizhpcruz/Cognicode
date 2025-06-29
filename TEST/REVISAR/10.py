import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

# --- 1. Dados Observacionais ---
data_hz = np.array([
    [0.24, 79.69, 2.32],
    [0.35, 84.40, 1.90],
    [0.57, 92.40, 1.50],
    [0.73, 97.30, 1.20],
    [1.00, 103.00, 1.10]
])
z_obs = data_hz[:, 0]
H_obs = data_hz[:, 1]
sigma_H = data_hz[:, 2]

# --- 2. Definição dos Modelos Cosmológicos ---
def hubble_lcdm(z, H0, Omega_m):
    arg = Omega_m * (1 + z)**3 + (1 - Omega_m)
    if np.any(arg < 0):
        return -np.inf
    return H0 * np.sqrt(arg)

def hubble_modified(z, H0, Omega_m, Rond0, n):
    term_de = Rond0 * (1 + z)**n
    arg = Omega_m * (1 + z)**3 + term_de
    if np.any(arg < 0) or Rond0 < 0:
        return -np.inf
    return H0 * np.sqrt(arg)

def hubble_wcdm(z, H0, Omega_m, w):
    term_de = (1 - Omega_m) * (1 + z)**(3 * (1 + w))
    arg = Omega_m * (1 + z)**3 + term_de
    if np.any(arg < 0):
        return -np.inf
    return H0 * np.sqrt(arg)

# --- 3. Função de Verossimilhança ---
def log_likelihood_lcdm(theta, z_obs, H_obs, sigma_H):
    H0, Omega_m = theta
    H_model = hubble_lcdm(z_obs, H0, Omega_m)
    if not np.isfinite(H_model).all():
        return -np.inf
    chi2 = np.sum(((H_obs - H_model) / sigma_H)**2)
    return -0.5 * chi2

def log_likelihood_modified(theta, z_obs, H_obs, sigma_H):
    H0, Omega_m, Rond0, n = theta
    H_model = hubble_modified(z_obs, H0, Omega_m, Rond0, n)
    if not np.isfinite(H_model).all():
        return -np.inf
    chi2 = np.sum(((H_obs - H_model) / sigma_H)**2)
    return -0.5 * chi2

def log_likelihood_wcdm(theta, z_obs, H_obs, sigma_H):
    H0, Omega_m, w = theta
    H_model = hubble_wcdm(z_obs, H0, Omega_m, w)
    if not np.isfinite(H_model).all():
        return -np.inf
    chi2 = np.sum(((H_obs - H_model) / sigma_H)**2)
    return -0.5 * chi2

# --- 4. Priors ---
def log_prior_lcdm(theta):
    H0, Omega_m = theta
    if 60 < H0 < 85 and 0.0 < Omega_m < 1.0:
        return -0.5 * ((H0 - 70) / 5)**2
    return -np.inf

def log_prior_modified(theta):
    H0, Omega_m, Rond0, n = theta
    if 60 < H0 < 85 and 0.0 < Omega_m < 1.0 and 0.0 < Rond0 < 2.0 and -5.0 < n < 5.0:
        return -0.5 * ((H0 - 70) / 5)**2
    return -np.inf

def log_prior_wcdm(theta):
    H0, Omega_m, w = theta
    if 60 < H0 < 85 and 0.0 < Omega_m < 1.0 and -2.0 < w < 0.0:
        return -0.5 * ((H0 - 70) / 5)**2
    return -np.inf

# --- 5. Posterior ---
def log_probability_lcdm(theta, z_obs, H_obs, sigma_H):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(theta, z_obs, H_obs, sigma_H)

def log_probability_modified(theta, z_obs, H_obs, sigma_H):
    lp = log_prior_modified(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_modified(theta, z_obs, H_obs, sigma_H)

def log_probability_wcdm(theta, z_obs, H_obs, sigma_H):
    lp = log_prior_wcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_wcdm(theta, z_obs, H_obs, sigma_H)

# --- 6. MCMC ---
def run_mcmc(log_prob_func, initial_state, nwalkers, nsteps, ndim, args):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, args=args)
    sampler.run_mcmc(initial_state, nsteps, progress=True)
    return sampler

nwalkers = 50
nsteps = 5000
burn_in = 1000

# --- Simulação Lambda-CDM ---
print("\n--- Simulação para o Modelo Lambda-CDM ---")
ndim_lcdm = 2
initial_lcdm = np.array([70.0, 0.3]) + 1e-4 * np.random.randn(nwalkers, ndim_lcdm)
sampler_lcdm = run_mcmc(log_probability_lcdm, initial_lcdm, nwalkers, nsteps, ndim_lcdm, (z_obs, H_obs, sigma_H))
flat_samples_lcdm = sampler_lcdm.get_chain(discard=burn_in, flat=True)
labels_lcdm = ["H0", "Omega_m"]

print("\nResultados do ajuste bayesiano (Lambda-CDM):")
for i in range(ndim_lcdm):
    mcmc = np.percentile(flat_samples_lcdm[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{labels_lcdm[i]} = {mcmc[1]:.3f} +{q[1]:.3f} -{q[0]:.3f}")

best_params_lcdm = np.median(flat_samples_lcdm, axis=0)
H_model_lcdm = hubble_lcdm(z_obs, *best_params_lcdm)
chi2_lcdm = np.sum(((H_obs - H_model_lcdm) / sigma_H)**2)
print(f"Chi^2 (Lambda-CDM) = {chi2_lcdm:.2f}")

fig_lcdm = corner.corner(flat_samples_lcdm, labels=labels_lcdm, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f")
fig_lcdm.suptitle("Distribuições Posteriores para o Modelo Lambda-CDM", fontsize=16)
plt.savefig("posterior_lcdm.png")
plt.close(fig_lcdm)

# --- Simulação Modificado ---
print("\n--- Simulação para o Modelo de Energia Escura Modificada ---")
ndim_modified = 4
initial_modified = np.array([70.0, 0.30, 0.70, 0.0]) + 1e-4 * np.random.randn(nwalkers, ndim_modified)
sampler_modified = run_mcmc(log_probability_modified, initial_modified, nwalkers, nsteps, ndim_modified, (z_obs, H_obs, sigma_H))
flat_samples_modified = sampler_modified.get_chain(discard=burn_in, flat=True)
labels_modified = ["H0", "Omega_m", "Rond0", "n"]

print("\nResultados do ajuste bayesiano (Modelo Modificado):")
for i in range(ndim_modified):
    mcmc = np.percentile(flat_samples_modified[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{labels_modified[i]} = {mcmc[1]:.3f} +{q[1]:.3f} -{q[0]:.3f}")

best_params_modified = np.median(flat_samples_modified, axis=0)
H_model_modified = hubble_modified(z_obs, *best_params_modified)
chi2_modified = np.sum(((H_obs - H_model_modified) / sigma_H)**2)
print(f"Chi^2 (Modelo Modificado) = {chi2_modified:.2f}")

fig_modified = corner.corner(flat_samples_modified, labels=labels_modified, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f")
fig_modified.suptitle("Distribuições Posteriores para o Modelo de Energia Escura Modificada", fontsize=16)
plt.savefig("posterior_modified.png")
plt.close(fig_modified)
