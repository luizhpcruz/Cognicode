import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.stats import norm
from scipy.integrate import quad

# Dados observacionais de H(z)
z_data = np.array([0.24, 0.35, 0.57, 0.73, 1.0])
H_obs = np.array([79.69, 84.4, 92.4, 97.3, 103.0])
sigma_H = np.array([2.32, 1.9, 1.5, 1.2, 1.1])

# Modelo de H(z) - LambdaCDM
def H_model(z, H0, Omega_m):
    Omega_lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)

# Função de log-verossimilhança
def log_likelihood(theta):
    H0, Omega_m = theta
    if Omega_m < 0.0 or Omega_m > 1.0 or H0 < 50.0 or H0 > 100.0:
        return -np.inf
    H_model_vals = H_model(z_data, H0, Omega_m)
    chi2 = np.sum(((H_obs - H_model_vals) / sigma_H)**2)
    return -0.5 * chi2

# Priors informativos
def log_prior(theta):
    H0, Omega_m = theta
    lp_H0 = norm.logpdf(H0, loc=70.0, scale=5.0)  # Prior normal para H0
    lp_Omega_m = 0.0 if 0.0 < Omega_m < 1.0 else -np.inf  # Prior uniforme para Omega_m
    return lp_H0 + lp_Omega_m

# Função de log-posterior
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Configuração do sampler MCMC
ndim = 2  # Número de parâmetros: H0 e Omega_m
nwalkers = 50
nsteps = 5000
initial = np.array([70.0, 0.3])  # Valores iniciais para H0 e Omega_m
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

# Execução do sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(pos, nsteps, progress=True)

# Análise dos resultados
samples = sampler.get_chain(discard=1000, thin=15, flat=True)
H0_mcmc, Omega_m_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples, [16, 50, 84], axis=0)))

print(f"H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f}/-{H0_mcmc[2]:.2f} km/s/Mpc")
print(f"Omega_m = {Omega_m_mcmc[0]:.3f} +{Omega_m_mcmc[1]:.3f}/-{Omega_m_mcmc[2]:.3f}")

# Visualização dos traços das cadeias
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["H0", "Omega_m"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("Passo")
plt.tight_layout()
plt.show()

# Visualização das distribuições posteriores
fig = corner.corner(samples, labels=labels, truths=[H0_mcmc[0], Omega_m_mcmc[0]])
plt.show()
