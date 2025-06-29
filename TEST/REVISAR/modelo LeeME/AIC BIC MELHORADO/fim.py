#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Cosmológica Híbrida: wCDM + EDE com Cosmic Chronometers
"""

# Verificação de dependências
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import corner
    import emcee
    from scipy.integrate import quad
    print("Todas as dependências foram carregadas com sucesso!")
except ImportError as e:
    print(f"ERRO: Dependência faltando: {e}")
    print("Por favor, instale as dependências necessárias com:")
    print("pip install numpy pandas matplotlib corner emcee scipy")
    exit(1)

# Constantes físicas
c = 299792.458  # Velocidade da luz em km/s

# ================================================
# 1. DADOS OBSERVACIONAIS
# ================================================

# Dados H(z) de Cosmic Chronometers (Moresco et al. 2016)
hz_data = np.array([
    [0.1791, 74.91, 3.8069262],
    [0.1993, 74.96, 4.9881352],
    [0.3519, 82.78, 13.94843],
    [0.3802, 83.0, 13.54],
    [0.4004, 76.97, 10.18],
    [0.4247, 87.08, 11.24],
    [0.4497, 92.78, 12.9],
    [0.4783, 80.91, 9.044],
    [0.5929, 103.8, 12.49752],
    [0.6797, 91.6, 7.961872],
    [0.7812, 104.5, 12.19515],
    [0.8754, 125.1, 16.70085],
    [1.037, 153.7, 19.6736],
    [1.363, 160.0, 32.63],
    [1.965, 186.5, 49.58]
])

z_vals = hz_data[:, 0]
Hz_vals = hz_data[:, 1]
errHz_vals = hz_data[:, 2]
cov_matrix = np.diag(errHz_vals**2)

print(f"Dados carregados: {len(z_vals)} pontos H(z)")

# ================================================
# 2. MODELO COSMOLÓGICO HÍBRIDO (wCDM + EDE)
# ================================================

def hubble_hybrid(z, H0, Om, w, Omega_EDE, n):
    """Equação de Hubble para o modelo híbrido wCDM + EDE"""
    Omega_DE = 1 - Om - Omega_EDE
    return H0 * np.sqrt(Om * (1+z)**3 + Omega_EDE * (1+z)**n + Omega_DE * (1+z)**(3*(1+w)))

# ================================================
# 3. FUNÇÃO DE VEROSSIMILHANÇA
# ================================================

def log_likelihood(params, z, Hz, cov):
    H0, Om, w, Omega_EDE, n = params
    H_pred = np.array([hubble_hybrid(z_i, H0, Om, w, Omega_EDE, n) for z_i in z])
    residual = Hz - H_pred
    inv_cov = np.linalg.inv(cov)
    chi2 = residual.T @ inv_cov @ residual
    return -0.5 * chi2

def log_prior(params):
    H0, Om, w, Omega_EDE, n = params
    if (50.0 < H0 < 100.0 and 
        0.05 < Om < 0.95 and 
        -2.0 < w < -0.5 and 
        0.0 < Omega_EDE < 0.5 and 
        0.5 < n < 3.0):
        return 0.0
    return -np.inf

def log_probability(params, z, Hz, cov):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, z, Hz, cov)

# ================================================
# 4. ANÁLISE MCMC
# ================================================

def run_mcmc():
    print("\nIniciando análise MCMC para modelo híbrido wCDM + EDE...")
    
    ndim = 5  # H0, Om, w, Omega_EDE, n
    nwalkers = 32
    nsteps = 4000
    
    initial_guess = np.array([70.0, 0.3, -1.0, 0.1, 1.5])
    p0 = initial_guess + 0.05 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(z_vals, Hz_vals, cov_matrix)
    )
    
    print(f"Executando MCMC com {nwalkers} caminhantes e {nsteps} passos...")
    sampler.run_mcmc(p0, nsteps, progress=True)
    print("MCMC concluído!")
    
    burn_in = 1000
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    H0_samples = samples[:, 0]
    Om_samples = samples[:, 1]
    w_samples = samples[:, 2]
    Omega_EDE_samples = samples[:, 3]
    n_samples = samples[:, 4]
    
    H0_mean = np.mean(H0_samples)
    H0_std = np.std(H0_samples)
    Om_mean = np.mean(Om_samples)
    Om_std = np.std(Om_samples)
    w_mean = np.mean(w_samples)
    w_std = np.std(w_samples)
    Omega_EDE_mean = np.mean(Omega_EDE_samples)
    Omega_EDE_std = np.std(Omega_EDE_samples)
    n_mean = np.mean(n_samples)
    n_std = np.std(n_samples)
    
    print("\n" + "="*50)
    print("RESULTADOS MCMC (wCDM + EDE)")
    print("="*50)
    print(f"H0 = {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc")
    print(f"Ωm = {Om_mean:.3f} ± {Om_std:.3f}")
    print(f"w = {w_mean:.3f} ± {w_std:.3f}")
    print(f"Ω_EDE = {Omega_EDE_mean:.3f} ± {Omega_EDE_std:.3f}")
    print(f"n = {n_mean:.3f} ± {n_std:.3f}")
    
    best_idx = np.argmax(sampler.get_log_prob(discard=burn_in, flat=True))
    best_params = samples[best_idx]
    best_chi2 = -2 * log_likelihood(best_params, z_vals, Hz_vals, cov_matrix)
    
    print(f"\nMelhor ajuste: H0 = {best_params[0]:.2f}, Ωm = {best_params[1]:.3f}, w = {best_params[2]:.3f}, Ω_EDE = {best_params[3]:.3f}, n = {best_params[4]:.3f}")
    print(f"χ² mínimo = {best_chi2:.2f}")
    
    n_params = 5
    n_data = len(z_vals)
    aic = best_chi2 + 2 * n_params
    bic = best_chi2 + n_params * np.log(n_data)
    
    print(f"AIC = {aic:.2f}")
    print(f"BIC = {bic:.2f}")
    
    return samples, best_params

# ================================================
# 5. VISUALIZAÇÃO
# ================================================

def create_plots(samples, best_params):
    print("\nGerando gráficos...")
    
    fig_corner = plt.figure(figsize=(15, 15))
    corner.corner(
        samples, 
        labels=["H₀", "Ω$_m$", "w", "Ω$_\text{EDE}$", "n"], 
        truths=best_params,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, 
        title_kwargs={"fontsize": 12}
    )
    corner_file = "mcmc_hybrid_corner_plot.png"
    plt.savefig(corner_file, dpi=150)
    print(f"Corner plot salvo como: {corner_file}")
    plt.close(fig_corner)
    
    fig_hubble = plt.figure(figsize=(12, 8))
    plt.errorbar(z_vals, Hz_vals, yerr=errHz_vals, fmt='o', markersize=8, capsize=4, label='Dados H(z) (CC)', color='blue')
    
    z_model = np.linspace(0, 2.0, 100)
    Hz_model_hybrid = [hubble_hybrid(z, *best_params) for z in z_model]
    Hz_model_wcdm = [hubble_hybrid(z, best_params[0], best_params[1], best_params[2], 0.0, 1.0) for z in z_model]
    
    plt.plot(z_model, Hz_model_hybrid, 'r-', linewidth=2, label='Híbrido (wCDM + EDE)')
    plt.plot(z_model, Hz_model_wcdm, 'g--', linewidth=2, label='wCDM para comparação')
    
    plt.xlabel('Redshift (z)', fontsize=14)
    plt.ylabel('H(z) [km/s/Mpc]', fontsize=14)
    plt.title('Função de Hubble H(z) - Modelo Híbrido', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(40, 250)
    plt.tight_layout()
    
    hubble_file = "hubble_hybrid_fit.png"
    plt.savefig(hubble_file, dpi=150)
    print(f"Gráfico H(z) salvo como: {hubble_file}")
    plt.show()
    plt.close(fig_hubble)

# ================================================
# 6. EXECUÇÃO PRINCIPAL
# ================================================

if __name__ == "__main__":
    print("="*70)
    print("ANÁLISE COSMOLÓGICA HÍBRIDA (wCDM + EDE) COM DADOS DE COSMIC CHRONOMETERS")
    print("="*70)
    
    samples, best_params = run_mcmc()
    create_plots(samples, best_params)
    
    print("\nAnálise concluída com sucesso!")
    input("\nPressione Enter para sair...")
