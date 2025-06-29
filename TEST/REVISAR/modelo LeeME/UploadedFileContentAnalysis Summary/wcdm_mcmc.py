import numpy as np
import emcee
import corner
import os
import matplotlib.pyplot as plt # Adicionado

# --- Dados H(z) + covariância ---
dir_data = "/home/ubuntu/workspace/content_analysis"
data_file_hz = os.path.join(dir_data, "HzTable_MM_BC03.dat")
z_obs, H_obs = np.loadtxt(data_file_hz, usecols=(0, 1), unpack=True, skiprows=1)

cov_matrix_file = os.path.join(dir_data, "cov_matrix_total_hz.dat")
cov_matrix = np.loadtxt(cov_matrix_file)
inv_cov_matrix = np.linalg.inv(cov_matrix)

n_data = len(z_obs)

# --- Modelo wCDM ---
def hubble_wcdm(z, H0, Om, w):
    # Assume um universo plano, Om + O_lambda = 1
    # O_lambda = 1 - Om
    # Termo de energia escura para wCDM: (1-Om) * (1+z)**(3*(1+w))
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om) * (1+z)**(3*(1+w)))

# --- Log-verossimilhança ---
def log_likelihood(params, z_obs, H_obs, inv_cov):
    H0, Om, w = params
    H_th = hubble_wcdm(z_obs, H0, Om, w)
    residual = H_obs - H_th
    chi2 = residual.T @ inv_cov @ residual
    return -0.5 * chi2

# --- Priors ---
def log_prior(params):
    H0, Om, w = params
    # Priors planos e largos
    if 50.0 < H0 < 100.0 and 0.01 < Om < 0.99 and -2.0 < w < 0.0:
        return 0.0
    return -np.inf

# --- Log-probabilidade posterior ---
def log_probability(params, z_obs, H_obs, inv_cov):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, z_obs, H_obs, inv_cov)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# --- Configuração MCMC ---
initial_guess = [70.0, 0.3, -1.0] # H0, Om, w
ndim = len(initial_guess)
nwalkers = 50
nsteps = 5000 # Aumentado para melhor convergência
burnin = 1000 # Aumentado para melhor convergência

pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                               args=(z_obs, H_obs, inv_cov_matrix))

print("Iniciando simulação MCMC para o modelo wCDM...")
sampler.run_mcmc(pos, nsteps, progress=True)
print("Simulação MCMC concluída.")

# --- Processar resultados ---
samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

# --- Resumo --- 
H0_mcmc, Om_mcmc, w_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print(f"\nResultados do MCMC (wCDM):")
print(f"H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f} km/s/Mpc")
print(f"Omega_m = {Om_mcmc[0]:.3f} +{Om_mcmc[1]:.3f} / -{Om_mcmc[2]:.3f}")
print(f"w = {w_mcmc[0]:.3f} +{w_mcmc[1]:0.3f} / -{w_mcmc[2]:0.3f}") # Corrigido para 3 casas decimais

# --- Critérios AIC/BIC ---
best_params = [H0_mcmc[0], Om_mcmc[0], w_mcmc[0]]
min_lnlike = log_likelihood(best_params, z_obs, H_obs, inv_cov_matrix)
min_chi2 = -2 * min_lnlike
k = ndim # Número de parâmetros

AIC = min_chi2 + 2 * k
BIC = min_chi2 + k * np.log(n_data)

print(f"Minimum Chi2 (at median params): {min_chi2:.2f}")
print(f"AIC: {AIC:.2f}")
print(f"BIC: {BIC:.2f}")

# --- Corner plot ---
fig = corner.corner(samples, labels=["H$_0$", "$\Omega_m$", "w"],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})

plt.suptitle("MCMC Posterior for Flat wCDM from H(z) Data (Full Covariance)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plot_path = "/home/ubuntu/wcdm_corner_plot.png"
plt.savefig(plot_path)
print(f"Corner plot salvo em: {plot_path}")

# Salvar resultados em arquivo de texto
summary_file = "/home/ubuntu/wcdm_mcmc_summary.txt"
with open(summary_file, "w") as f:
    f.write(f"Resultados do MCMC (wCDM):\n")
    f.write(f"H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f} km/s/Mpc\n")
    f.write(f"Omega_m = {Om_mcmc[0]:.3f} +{Om_mcmc[1]:.3f} / -{Om_mcmc[2]:.3f}\n")
    f.write(f"w = {w_mcmc[0]:.3f} +{w_mcmc[1]:0.3f} / -{w_mcmc[2]:0.3f}\n")
    f.write(f"Minimum Chi2 (at median params): {min_chi2:.2f}\n")
    f.write(f"AIC: {AIC:.2f}\n")
    f.write(f"BIC: {BIC:.2f}\n")
print(f"Resumo dos resultados salvo em: {summary_file}")


