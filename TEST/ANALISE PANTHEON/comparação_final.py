import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
import emcee
import time
import warnings
import json

warnings.filterwarnings("ignore", category=RuntimeWarning)
total_start_time = time.time()

try:
    pantheon_data = pd.read_csv('pantheon_data.csv')
    valid_indices = pantheon_data['muerr'] > 0
    pantheon_data = pantheon_data[valid_indices]
    
    z_data = pantheon_data['zHD'].values
    mu_data = pantheon_data['mu'].values
    mu_err = pantheon_data['muerr'].values
    N_data_points = len(z_data)
    
    print(f"Dados do Pantheon carregados. {N_data_points} pontos de dados válidos.")
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    exit()

def H_LCDM(z, H0, Omega_m):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

def H_modificado(z, H0, Omega_m, Rond0, n):
    termo_de = (1 - Omega_m - Rond0) + Rond0 * (1 + z)**n
    return H0 * np.sqrt(np.maximum(0, Omega_m * (1 + z)**3 + termo_de))

interpolator_cache = {}
def get_mu_interpolator(params, model_func):
    param_key = (model_func.__name__,) + tuple(params)
    if param_key in interpolator_cache:
        return interpolator_cache[param_key]

    c = 299792.458
    z_grid = np.linspace(0, max(z_data) * 1.1, 200)
    
    # CORREÇÃO 1: Usando o integrador 'quad', mais robusto
    integrand = lambda z_prime: c / model_func(z_prime, *params)
    
    dist_lum_grid = np.zeros_like(z_grid)
    for i in range(1, len(z_grid)):
        try:
            integral_step, _ = quad(integrand, 0, z_grid[i])
            dist_lum_grid[i] = integral_step
        except Exception:
            return None 
            
    dist_lum_grid = (1 + z_grid) * dist_lum_grid
    dist_lum_grid[dist_lum_grid <= 0] = 1e-9
    
    mu_grid = 5 * np.log10(dist_lum_grid) + 25
    interpolator = interp1d(z_grid, mu_grid, kind='cubic', fill_value="extrapolate")
    interpolator_cache[param_key] = interpolator
    return interpolator

def log_likelihood(theta, model_func):
    mu_interpolator = get_mu_interpolator(theta, model_func)
    if mu_interpolator is None: return -np.inf
    model_mu = mu_interpolator(z_data)
    chi2 = np.sum(((mu_data - model_mu) / mu_err)**2)
    return -0.5 * chi2

def log_prior_lcdm(theta):
    H0, Omega_m = theta
    if 40 < H0 < 100 and 0.01 < Omega_m < 1.0: return 0.0
    return -np.inf

def log_prior_mod(theta):
    H0, Omega_m, Rond0, n = theta
    if 40 < H0 < 100 and 0.01 < Omega_m < 0.5 and -2.0 < Rond0 < 2.0 and -5.0 < n < 5.0: return 0.0
    return -np.inf

def log_posterior_lcdm(theta):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta, H_LCDM)

def log_posterior_mod(theta):
    lp = log_prior_mod(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta, H_modificado)

def run_analysis(name, posterior_func, ndim, p0_center):
    print(f"\n--- Iniciando Análise MCMC para o Modelo: {name} ---")
    nwalkers, nsteps, nburn = 100, 6000, 2000
    pos = [p0_center + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior_func)
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    samples = sampler.get_chain(discard=nburn, flat=True, thin=30)
    median_params = np.median(samples, axis=0)
    
    min_chi2 = -2 * log_likelihood(median_params, H_LCDM if name=='ΛCDM' else H_modificado)
    return median_params, min_chi2

params_lcdm, chi2_lcdm = run_analysis("ΛCDM", log_posterior_lcdm, 2, [73, 0.3])
params_mod, chi2_mod = run_analysis("Modificado", log_posterior_mod, 4, [73, 0.3, 0.7, -1.0])

k_lcdm, k_mod = 2, 4
aic_lcdm = chi2_lcdm + 2 * k_lcdm
bic_lcdm = chi2_lcdm + k_lcdm * np.log(N_data_points)
aic_mod = chi2_mod + 2 * k_mod
bic_mod = chi2_mod + k_mod * np.log(N_data_points)

# --- Apresentação e Salvamento dos Resultados Finais ---
print("\nSalvando resultados...")

summary_text = f"""
========================================
      RESULTADO DA COMPARAÇÃO DE MODELOS
========================================
Data: Pantheon+ (subconjunto de {N_data_points} pontos)

Modelo: ΛCDM (Lambda-CDM)
  - Parâmetros (k): {k_lcdm}
  - Melhor ajuste (χ²_min): {chi2_lcdm:.2f}
  - AIC: {aic_lcdm:.2f}
  - BIC: {bic_lcdm:.2f}
  - Parâmetros Medianos:
    - H0 = {params_lcdm[0]:.2f} km/s/Mpc
    - Ωm = {params_lcdm[1]:.3f}

----------------------------------------

Modelo: Modificado (Vetor-Tensor)
  - Parâmetros (k): {k_mod}
  - Melhor ajuste (χ²_min): {chi2_mod:.2f}
  - AIC: {aic_mod:.2f}
  - BIC: {bic_mod:.2f}
  - Parâmetros Medianos:
    - H0 = {params_mod[0]:.2f} km/s/Mpc
    - Ωm = {params_mod[1]:.3f}
    - R0 = {params_mod[2]:.3f}
    - n  = {params_mod[3]:.3f}
========================================
"""
print(summary_text)

# CORREÇÃO 2: Adicionando 'encoding="utf-8"' para salvar caracteres especiais
with open("resultados_finais.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("Resultados salvos em 'resultados_finais.txt'.")
total_end_time = time.time()
print(f"Análise completa finalizada em {total_end_time - total_start_time:.2f} segundos.")