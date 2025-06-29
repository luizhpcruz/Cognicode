import numpy as np
import pandas as pd
import emcee

# Carregar dados H(z) e matriz de covariância
# Usando HzTable_MM_BC03.dat para H(z) observacional
data_hz = np.loadtxt("/home/ubuntu/workspace/content_analysis/HzTable_MM_BC03.dat", usecols=(0, 1), unpack=True, skiprows=1)
z_obs = data_hz[0]
H_obs = data_hz[1]

cov_matrix = np.loadtxt("/home/ubuntu/workspace/content_analysis/cov_matrix_total_hz.dat")

# Modelo ΛCDM
def hubble_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))

# Log-verossimilhança com covariância
def log_likelihood(params, z, H_obs, inv_cov):
    H0, Om = params
    H_th = hubble_lcdm(z, H0, Om)
    residual = H_obs - H_th
    # Adiciona uma pequena penalidade para evitar valores não físicos de Om
    if not (0.01 < Om < 0.99): 
        return -np.inf
    return -0.5 * residual.T @ inv_cov @ residual

# MCMC
nwalkers = 50
initial_guess = [70.0, 0.3]
ndim = len(initial_guess)
p0 = initial_guess + 0.1 * np.random.randn(nwalkers, ndim)

# Inverter a matriz de covariância
inv_cov_matrix = np.linalg.inv(cov_matrix)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, 
                               args=(z_obs, H_obs, inv_cov_matrix))

print("Iniciando simulação MCMC...")
sampler.run_mcmc(p0, 2000, progress=True)
print("Simulação MCMC concluída.")

# Processar resultados
samples = sampler.get_chain(discard=500, thin=15, flat=True)

H0_mcmc, Omegam_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print(f"\nResultados do MCMC:")
print(f"H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f} km/s/Mpc")
print(f"Omega_m = {Omegam_mcmc[0]:.3f} +{Omegam_mcmc[1]:.3f} / -{Omegam_mcmc[2]:.3f}")

# Salvar os resultados em um arquivo de texto
with open("/home/ubuntu/mcmc_results_summary.txt", "w") as f:
    f.write(f"Resultados do MCMC:\n")
    f.write(f"H0 = {H0_mcmc[0]:.2f} +{H0_mcmc[1]:.2f} / -{H0_mcmc[2]:.2f} km/s/Mpc\n")
    f.write(f"Omega_m = {Omegam_mcmc[0]:.3f} +{Omegam_mcmc[1]:.3f} / -{Omegam_mcmc[2]:.3f}\n")

print("Resultados salvos em /home/ubuntu/mcmc_results_summary.txt")


