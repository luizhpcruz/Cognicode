
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from scipy.linalg import inv

# --- 1. Dados Observacionais (Cosmic Chronometers H(z)) ---
# Dados de H(z) de Moresco et al. 2020 (HzTable_MM_BC32.txt)
# Fonte: https://raw.githubusercontent.com/Ahmadmehrabi/Cosmic_chronometer_data/refs/heads/main/HzTable_MM_BC32.txt
# z, H(z), sigma_H
data = np.array([
    [0.070, 69.0, 19.6],
    [0.090, 69.0, 12.0],
    [0.120, 68.6, 26.2],
    [0.170, 83.0, 8.0],
    [0.179, 75.0, 4.0],
    [0.199, 75.0, 5.0],
    [0.200, 72.9, 29.6],
    [0.270, 77.0, 14.0],
    [0.280, 88.8, 36.6],
    [0.352, 83.0, 14.0],
    [0.380, 81.5, 1.9],
    [0.4004, 77.0, 10.2],
    [0.424, 87.1, 11.2],
    [0.449, 92.8, 12.9],
    [0.470, 89.0, 3.0],
    [0.4783, 80.9, 9.0],
    [0.480, 97.0, 62.0],
    [0.593, 104.0, 13.0],
    [0.679, 92.0, 8.0],
    [0.750, 98.8, 33.6],
    [0.781, 105.0, 12.0],
    [0.875, 125.0, 17.0],
    [0.880, 90.0, 40.0],
    [0.900, 117.0, 23.0],
    [1.037, 104.0, 13.0],
    [1.363, 160.0, 33.6],
    [1.430, 177.0, 18.0],
    [1.530, 140.0, 14.0],
    [1.750, 202.0, 40.0],
    [1.965, 186.0, 50.0],
    [2.360, 224.0, 8.0],
    [2.340, 222.0, 7.0],     # Delubac et al. (2015)
    [2.400, 227.0, 8.0],     # Font-Ribera et al. (2014)
    [2.330, 224.0, 8.6],     # Zarrouk et al. (2018)
    [1.526, 168.0, 17.0],    # Wang et al. (2017)
    [1.037, 154.0, 20.0],    # Zhao et al. (2019)
    [0.978, 148.0, 17.0],    # Ata et al. (2018)
    [0.865, 139.0, 20.0],    # Kazin et al. (2014)
    [0.700, 126.6, 8.0],     # Alam et al. (2021)
    [0.570, 100.3, 3.7],     # Beutler et al. (2011)
    [0.510, 96.8, 3.4]       # Blake et al. (2012)
])

z_obs = data[:, 0]
H_obs = data[:, 1]
sigma_H_obs = data[:, 2]

# Matriz de covariância diagonal (para simplicidade, usando apenas erros estatísticos)
# Para uma análise mais robusta, uma matriz de covariância completa seria necessária.
# No entanto, para demonstração e para evitar complexidade de download/geração, usamos a diagonal.
cov_matrix = np.diag(sigma_H_obs**2)
inv_cov_matrix = inv(cov_matrix)

# --- 2. Modelos Cosmológicos ---

def hubble_lcdm(z, H0, Om):
    """Função de Hubble para o modelo ΛCDM plano."""
    Ol = 1.0 - Om
    return H0 * np.sqrt(Om * (1 + z)**3 + Ol)

def hubble_wcdm(z, H0, Om, w):
    """Função de Hubble para o modelo wCDM plano."""
    Ol = 1.0 - Om
    return H0 * np.sqrt(Om * (1 + z)**3 + Ol * (1 + z)**(3 * (1 + w)))

# --- 3. Funções de Log-Verossimilhança e Log-Prior ---

def log_likelihood(params, z, H_obs, inv_cov, model_type):
    if model_type == 'LCDM':
        H0, Om = params
        # Priors simples para H0 e Om
        if not (50 < H0 < 100 and 0.05 < Om < 0.5):
            return -np.inf
        H_th = hubble_lcdm(z, H0, Om)
    elif model_type == 'wCDM':
        H0, Om, w = params
        # Priors simples para H0, Om e w
        if not (50 < H0 < 100 and 0.05 < Om < 0.5 and -2.0 < w < 0.0):
            return -np.inf
        H_th = hubble_wcdm(z, H0, Om, w)
    else:
        raise ValueError("Tipo de modelo desconhecido.")

    residual = H_obs - H_th
    return -0.5 * residual.T @ inv_cov @ residual

def log_prior_lcdm(params):
    H0, Om = params
    if 50 < H0 < 100 and 0.05 < Om < 0.5:
        return 0.0
    return -np.inf

def log_prior_wcdm(params):
    H0, Om, w = params
    if 50 < H0 < 100 and 0.05 < Om < 0.5 and -2.0 < w < 0.0:
        return 0.0
    return -np.inf

def log_probability_lcdm(params, z, H_obs, inv_cov):
    lp = log_prior_lcdm(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, z, H_obs, inv_cov, 'LCDM')

def log_probability_wcdm(params, z, H_obs, inv_cov):
    lp = log_prior_wcdm(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, z, H_obs, inv_cov, 'wCDM')

# --- 4. Configuração e Execução do MCMC ---

def run_mcmc(log_prob_fn, initial_guess, ndim, nwalkers, nsteps, args, labels, model_name):
    print(f"\n--- Executando MCMC para o modelo {model_name} ---")
    p0 = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=args)
    state = sampler.run_mcmc(p0, nsteps, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)
    
    flat_samples = sampler.get_chain(discard=nsteps//2, thin=15, flat=True)
    return flat_samples, sampler

# --- Configuração ΛCDM ---
initial_guess_lcdm = np.array([70.0, 0.3])
ndim_lcdm = len(initial_guess_lcdm)
nwalkers_lcdm = 32
nsteps_lcdm = 5000

# --- Configuração wCDM ---
initial_guess_wcdm = np.array([70.0, 0.3, -1.0])
ndim_wcdm = len(initial_guess_wcdm)
nwalkers_wcdm = 32
nsteps_wcdm = 5000

# --- Executar MCMC para ΛCDM ---
flat_samples_lcdm, sampler_lcdm = run_mcmc(
    log_probability_lcdm, initial_guess_lcdm, ndim_lcdm, nwalkers_lcdm, nsteps_lcdm,
    (z_obs, H_obs, inv_cov_matrix), ['H0', 'Om'], 'ΛCDM'
)

# --- Executar MCMC para wCDM ---
flat_samples_wcdm, sampler_wcdm = run_mcmc(
    log_probability_wcdm, initial_guess_wcdm, ndim_wcdm, nwalkers_wcdm, nsteps_wcdm,
    (z_obs, H_obs, inv_cov_matrix), ['H0', 'Om', 'w'], 'wCDM'
)

# --- 5. Análise de Resultados ---

def analyze_results(flat_samples, sampler, z_obs, H_obs, inv_cov, model_type, labels):
    best_fit_params = []
    for i in range(flat_samples.shape[1]):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = f"{mcmc[1]:.2f} +{q[1]:.2f} / -{q[0]:.2f}"
        best_fit_params.append(mcmc[1])
        print(f"  {labels[i]} = {txt}")
    
    best_fit_params = np.array(best_fit_params)
    
    # Calcular chi2 para o melhor ajuste
    chi2_val = -2 * log_likelihood(best_fit_params, z_obs, H_obs, inv_cov, model_type)
    dof = len(H_obs) - len(best_fit_params)
    print(f"  Chi2 = {chi2_val:.2f}")
    print(f"  Degrees of Freedom (dof) = {dof}")
    print(f"  Reduced Chi2 = {chi2_val/dof:.2f}")
    
    # Calcular AIC e BIC
    aic_val = chi2_val + 2 * len(best_fit_params)
    bic_val = chi2_val + len(best_fit_params) * np.log(len(H_obs))
    print(f"  AIC = {aic_val:.2f}")
    print(f"  BIC = {bic_val:.2f}")
    
    return best_fit_params, chi2_val, aic_val, bic_val

print("\n--- Resultados do Modelo ΛCDM ---")
best_fit_lcdm, chi2_lcdm, aic_lcdm, bic_lcdm = analyze_results(
    flat_samples_lcdm, sampler_lcdm, z_obs, H_obs, inv_cov_matrix, 'LCDM', ['H0', 'Om']
)

print("\n--- Resultados do Modelo wCDM ---")
best_fit_wcdm, chi2_wcdm, aic_wcdm, bic_wcdm = analyze_results(
    flat_samples_wcdm, sampler_wcdm, z_obs, H_obs, inv_cov_matrix, 'wCDM', ['H0', 'Om', 'w']
)

# --- Comparação de Modelos ---
print("\n--- Comparação de Modelos ---")
print(f"  ΔAIC (wCDM - ΛCDM) = {aic_wcdm - aic_lcdm:.2f}")
print(f"  ΔBIC (wCDM - ΛCDM) = {bic_wcdm - bic_lcdm:.2f}")

# --- 6. Visualização dos Resultados ---

# Corner plot para ΛCDM
fig_lcdm = corner.corner(
    flat_samples_lcdm, labels=['H0', 'Om'],
    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f",
    truths=best_fit_lcdm
)
fig_lcdm.suptitle('Distribuição Posterior dos Parâmetros (ΛCDM)', fontsize=16)
plt.savefig('corner_plot_lcdm.png')

# Corner plot para wCDM
fig_wcdm = corner.corner(
    flat_samples_wcdm, labels=['H0', 'Om', 'w'],
    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f",
    truths=best_fit_wcdm
)
fig_wcdm.suptitle('Distribuição Posterior dos Parâmetros (wCDM)', fontsize=16)
plt.savefig('corner_plot_wcdm.png')

# Plot H(z) vs z
plt.figure(figsize=(10, 6))
plt.errorbar(z_obs, H_obs, yerr=sigma_H_obs, fmt='o', color='black', label='Dados H(z) Observacionais')

z_plot = np.linspace(min(z_obs), max(z_obs), 500)

# Curva de melhor ajuste ΛCDM
H_lcdm_best_fit = hubble_lcdm(z_plot, best_fit_lcdm[0], best_fit_lcdm[1])
plt.plot(z_plot, H_lcdm_best_fit, color='red', linestyle='-', label=f'ΛCDM (H0={best_fit_lcdm[0]:.2f}, Om={best_fit_lcdm[1]:.2f})')

# Curva de melhor ajuste wCDM
H_wcdm_best_fit = hubble_wcdm(z_plot, best_fit_wcdm[0], best_fit_wcdm[1], best_fit_wcdm[2])
plt.plot(z_plot, H_wcdm_best_fit, color='blue', linestyle='--', label=f'wCDM (H0={best_fit_wcdm[0]:.2f}, Om={best_fit_wcdm[1]:.2f}, w={best_fit_wcdm[2]:.2f})')

plt.xlabel('Redshift (z)')
plt.ylabel('H(z) (km/s/Mpc)')
plt.title('Comparação de H(z) Observacional com Modelos ΛCDM e wCDM')
plt.legend()
plt.grid(True)
plt.savefig('hz_comparison_plot.png')

plt.show()

print("\nAnálise concluída. Verifique os arquivos 'corner_plot_lcdm.png', 'corner_plot_wcdm.png' e 'hz_comparison_plot.png' para os gráficos.")

# Manter o console aberto em sistemas Windows
import os
if os.name == 'nt':
    input("Pressione Enter para sair...")

