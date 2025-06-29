# Conteúdo do script completo atualizado com suporte ao Pantheon+, H(z) e modelos ΛCDM e wCDM
codigo_txt = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import emcee
from scipy.linalg import inv
from scipy.integrate import quad
import os

# --- Carregar dados Pantheon+ ---
def carregar_dados_pantheon(diretorio_base):
    caminho_dados = os.path.join(diretorio_base, "Pantheon+SH0ES.dat")
    colunas = ['UID', 'zHEL', 'zCMB', 'RA', 'DEC', 'Peak_MJD', 'm_b_corr', 'm_b_corr_err']
    df = pd.read_csv(caminho_dados, delim_whitespace=True, comment='#', names=colunas, header=None)
    return df['zCMB'].values, df['m_b_corr'].values, df['m_b_corr_err'].values

# --- Carregar dados H(z) ---
dados_hz = np.array([
    [0.070, 69.0, 19.6], [0.090, 69.0, 12.0], [0.120, 68.6, 26.2],
    [0.170, 83.0, 8.0], [0.179, 75.0, 4.0], [0.199, 75.0, 5.0],
    [0.200, 72.9, 29.6], [0.270, 77.0, 14.0], [0.280, 88.8, 36.6],
    [0.352, 83.0, 14.0], [0.380, 81.5, 1.9], [0.4004, 77.0, 10.2],
    [0.424, 87.1, 11.2], [0.449, 92.8, 12.9], [0.470, 89.0, 3.0],
    [0.4783, 80.9, 9.0], [0.480, 97.0, 62.0], [0.593, 104.0, 13.0],
    [0.679, 92.0, 8.0], [0.750, 98.8, 33.6], [0.781, 105.0, 12.0],
    [0.875, 125.0, 17.0], [0.880, 90.0, 40.0], [0.900, 117.0, 23.0],
    [1.037, 104.0, 13.0], [1.363, 160.0, 33.6], [1.430, 177.0, 18.0],
    [1.530, 140.0, 14.0], [1.750, 202.0, 40.0], [1.965, 186.0, 50.0],
    [2.360, 224.0, 8.0]
])
z_obs, H_obs, sigma_H_obs = dados_hz[:,0], dados_hz[:,1], dados_hz[:,2]
inv_cov_matrix = inv(np.diag(sigma_H_obs**2))

# --- Modelos cosmológicos ---
def hubble_lcdm(z, H0, Om): return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))
def hubble_wcdm(z, H0, Om, w): return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + w)))

def distancia_luminosidade(z, H0, Om, w=-1.0, modelo='LCDM'):
    c = 299792.458
    def E(z_):
        Ol = 1 - Om
        if modelo == 'LCDM': return np.sqrt(Om * (1 + z_)**3 + Ol)
        elif modelo == 'wCDM': return np.sqrt(Om * (1 + z_)**3 + Ol * (1 + z_)**(3 * (1 + w)))
    return np.array([(1 + zi) * c / H0 * quad(lambda z_: 1 / E(z_), 0, zi)[0] for zi in z])

def distancia_modulo(z, H0, Om, w=-1.0, modelo='LCDM'):
    dL = distancia_luminosidade(z, H0, Om, w, modelo)
    return 5 * np.log10(dL * 1e6 / 10)

# --- Likelihoods ---
def log_likelihood(params, z, H_obs, inv_cov, modelo):
    if modelo == 'LCDM':
        H0, Om = params
        if not (50 < H0 < 100 and 0.05 < Om < 0.5): return -np.inf
        H_th = hubble_lcdm(z, H0, Om)
    elif modelo == 'wCDM':
        H0, Om, w = params
        if not (50 < H0 < 100 and 0.05 < Om < 0.5 and -2.0 < w < 0.0): return -np.inf
        H_th = hubble_wcdm(z, H0, Om, w)
    residual = H_obs - H_th
    return -0.5 * residual.T @ inv_cov @ residual

def log_likelihood_pantheon(params, z, mu_obs, mu_err, modelo='LCDM'):
    if modelo == 'LCDM':
        H0, Om = params
        mu_model = distancia_modulo(z, H0, Om, modelo='LCDM')
    elif modelo == 'wCDM':
        H0, Om, w = params
        mu_model = distancia_modulo(z, H0, Om, w, modelo='wCDM')
    return -0.5 * np.sum(((mu_obs - mu_model) / mu_err)**2)

# --- Log-prob combinado ---
def log_prior_combinado(params, modelo):
    if modelo == 'LCDM':
        H0, Om = params
        return 0.0 if 50 < H0 < 100 and 0.05 < Om < 0.5 else -np.inf
    elif modelo == 'wCDM':
        H0, Om, w = params
        return 0.0 if 50 < H0 < 100 and 0.05 < Om < 0.5 and -2.0 < w < 0.0 else -np.inf
    return -np.inf

def log_probability_combinado(params, z_Hz, H_obs, inv_cov_Hz, z_SN, mu_SN, mu_err_SN, modelo):
    lp = log_prior_combinado(params, modelo)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(params, z_Hz, H_obs, inv_cov_Hz, modelo) + \
               log_likelihood_pantheon(params, z_SN, mu_SN, mu_err_SN, modelo)

# --- Rodar MCMC ---
def run_mcmc_combinado(modelo, initial_guess, ndim, z_Hz, H_obs, inv_cov_Hz, z_SN, mu_SN, mu_err_SN, nwalkers=32, nsteps=5000):
    p0 = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_combinado,
                                     args=(z_Hz, H_obs, inv_cov_Hz, z_SN, mu_SN, mu_err_SN, modelo))
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler.get_chain(discard=nsteps//2, thin=15, flat=True), sampler

# --- Análise ---
def analyze_results_combinado(flat_samples, modelo, z_Hz, H_obs, inv_cov_Hz, z_SN, mu_SN, mu_err_SN, labels):
    best_fit = [np.percentile(flat_samples[:, i], 50) for i in range(len(labels))]
    chi2_total = -2 * log_likelihood(params=best_fit, z=z_Hz, H_obs=H_obs, inv_cov=inv_cov_Hz, modelo=modelo) \
                 -2 * log_likelihood_pantheon(params=best_fit, z=z_SN, mu_obs=mu_SN, mu_err=mu_err_SN, modelo=modelo)
    dof = len(H_obs) + len(mu_SN) - len(best_fit)
    aic = chi2_total + 2 * len(best_fit)
    bic = chi2_total + len(best_fit) * np.log(len(H_obs) + len(mu_SN))
    print(f"Melhor ajuste: {dict(zip(labels, best_fit))}")
    print(f"Chi² = {chi2_total:.2f}, Red. Chi² = {chi2_total/dof:.2f}, AIC = {aic:.2f}, BIC = {bic:.2f}")
    return best_fit, chi2_total, aic, bic

# --- Plot µ(z) ---
def plot_mu_vs_z(z_SN, mu_SN, mu_err_SN, best_fit_params, modelo):
    z_plot = np.linspace(min(z_SN), max(z_SN), 500)
    if modelo == 'LCDM':
        H0, Om = best_fit_params
        mu_teorico = distancia_modulo(z_plot, H0, Om, modelo='LCDM')
        legenda = f'ΛCDM (H₀={H0:.1f}, Ωₘ={Om:.2f})'
    elif modelo == 'wCDM':
        H0, Om, w = best_fit_params
        mu_teorico = distancia_modulo(z_plot, H0, Om, w, modelo='wCDM')
        legenda = f'wCDM (H₀={H0:.1f}, Ωₘ={Om:.2f}, w={w:.2f})'
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_SN, mu_SN, yerr=mu_err_SN, fmt='o', color='black', markersize=4, label='Pantheon+')
    plt.plot(z_plot, mu_teorico, 'r-', linewidth=2, label=legenda)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Módulo de Distância μ(z)')
    plt.title('Pantheon+ vs Modelo')
    plt.grid(True)
    plt.legend()
    plt.savefig('pantheon_fit_plot.png')
    plt.show()
"""
with open("/mnt/data/codigo_completo_cosmologia.txt", "w", encoding="utf-8") as f:
    f.write(codigo_txt.strip())

"/mnt/data/codigo_completo_cosmologia.txt"

