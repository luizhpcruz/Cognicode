import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from scipy.integrate import quad
from scipy.optimize import minimize
import multiprocessing

# Constantes físicas
c = 299792.458  # Velocidade da luz em km/s

# ================================================
# 1. DADOS OBSERVACIONAIS
# ================================================

# Dados H(z) "hardcodados" diretamente no script
# ESTES DADOS NÃO PRECISAM SER RECARREGADOS EM CADA PROCESSO
print("Carregando dados H(z) diretamente no script...")
z_h_data = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 1.75])
H_data = np.array([69.0, 75.0, 78.0, 90.0, 105.0, 125.0, 150.0, 175.0, 200.0])
H_err = np.array([12.0, 10.0, 9.0, 15.0, 17.0, 20.0, 25.0, 30.0, 35.0])
print(f"Dados H(z) carregados: {len(z_h_data)} pontos.")

# Carregando dados Pantheon+SH0ES.dat (ainda de arquivo externo devido ao tamanho)
# ESTES DADOS TAMBÉM SÓ PRECISAM SER CARREGADOS UMA VEZ
file_path_sn = "data/Pantheon+SH0ES.dat"
print(f"Carregando dados Pantheon+SH0ES de {file_path_sn}...")
try:
    sn_df = pd.read_csv(file_path_sn, sep=r'\s+', comment='#')
    
    sn_df = sn_df.rename(columns={'zHD': 'z', 'MU_SH0ES': 'mu', 'MU_SH0ES_ERR_DIAG': 'sigma_mu'})
    print(f"Dados Pantheon+SH0ES.dat carregados com sucesso. Colunas disponíveis: {sn_df.columns.tolist()}")

    if 'z' not in sn_df.columns or 'mu' not in sn_df.columns or 'sigma_mu' not in sn_df.columns:
        raise ValueError("Colunas 'z', 'mu' ou 'sigma_mu' não encontradas após renomeamento. Verifique o cabeçalho do arquivo e os nomes originais.")

    z_sn_data = sn_df['z'].values
    mu_sn_data = sn_df['mu'].values
    mu_sn_err = sn_df['sigma_mu'].values
    N_sn = len(z_sn_data)
    print(f"Dados Pantheon+SH0ES processados: {N_sn} pontos.")

except FileNotFoundError:
    print(f"ERRO ao carregar {file_path_sn}: Arquivo não encontrado.")
    print(f"Certifique-se de que '{file_path_sn}' existe na pasta 'data' dentro do diretório do script.")
    exit(1)
except Exception as e:
    print(f"ERRO ao carregar {file_path_sn}: {e}")
    print("Verifique o formato do arquivo e o delimitador.")
    exit(1)

# ================================================
# 2. MODELOS COSMOLÓGICOS (Funções E(z))
# ================================================

def Omega_ond(z, Omega_ond0, n):
    return Omega_ond0 * (1 + z)**n

def Omega_EDE(z, Omega_EDE0, zc, m=3):
    return Omega_EDE0 * np.exp(-(z/zc)**m)

def E_modificado(z, Omega_m, Omega_ond0, n, Omega_EDE0, zc):
    sum_omegas = Omega_m + Omega_ond0 + Omega_EDE0
    Omega_lambda_eff = 1.0 - sum_omegas
    
    term_lambda = max(0, Omega_lambda_eff) 
    
    return np.sqrt(Omega_m * (1 + z)**3 + term_lambda + \
                   Omega_ond(z, Omega_ond0, n) + Omega_EDE(z, Omega_EDE0, zc))


# ================================================
# 3. MÓDULO DE DISTÂNCIA
# ================================================

def integral_dL(z, H0, E_func, *E_args):
    integrand = lambda z_prime: 1.0 / E_func(z_prime, *E_args)
    try:
        result, _ = quad(integrand, 0, z, limit=100)
    except Exception:
        return np.inf
    return (c / H0) * result * (1 + z)

def mu_model(z_vals, H0, E_func, *E_args):
    mu = []
    for z in z_vals:
        dL_val = integral_dL(z, H0, E_func, *E_args)
        if dL_val == 0 or np.isinf(dL_val) or np.isnan(dL_val):
            mu.append(np.inf)
        else:
            mu.append(5 * np.log10(dL_val) + 25)
    return np.array(mu)

# ================================================
# 4. FUNÇÕES DE VEROSSIMILHANÇA E POSTERIOR
# ================================================

def ln_likelihood_Hz(theta, z_obs, H_obs, H_err):
    H0, Omega_m, Omega_ond0, n, Omega_EDE0, zc = theta
    
    E_vals = np.array([E_modificado(z, Omega_m, Omega_ond0, n, Omega_EDE0, zc) for z in z_obs])
    model_Hz = H0 * E_vals
    
    if np.any(H_err == 0) or np.any(np.isnan(model_Hz)) or np.any(np.isinf(model_Hz)):
        return -np.inf
    
    sigma2 = H_err**2
    return -0.5 * np.sum((H_obs - model_Hz)**2 / sigma2)

def ln_likelihood_SN(theta, z_obs, mu_obs, mu_err):
    H0, Omega_m, Omega_ond0, n, Omega_EDE0, zc = theta
    model_mu = mu_model(z_obs, H0, E_modificado, Omega_m, Omega_ond0, n, Omega_EDE0, zc)
    
    if np.any(mu_err == 0) or np.any(np.isnan(model_mu)) or np.any(np.isinf(model_mu)):
        return -np.inf
        
    sigma2 = mu_err**2
    return -0.5 * np.sum((mu_obs - model_mu)**2 / sigma2)

def ln_prior(theta):
    H0, Omega_m, Omega_ond0, n, Omega_EDE0, zc = theta
    if (60 <= H0 <= 80 and
        0.01 <= Omega_m <= 0.6 and
        0 <= Omega_ond0 <= 0.2 and
        -5 <= n <= 1 and
        0 <= Omega_EDE0 <= 0.2 and
        100 <= zc <= 10000):
        return 0.0
    return -np.inf

def ln_posterior(theta, z_h_data, H_data, H_err, z_sn_data, mu_sn_data, mu_sn_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    ll_hz = ln_likelihood_Hz(theta, z_h_data, H_data, H_err)
    ll_sn = ln_likelihood_SN(theta, z_sn_data, mu_sn_data, mu_sn_err)
    
    if not np.isfinite(ll_hz) or not np.isfinite(ll_sn):
        return -np.inf
        
    return lp + ll_hz + ll_sn

# ================================================
# 5. MCMC COM EMCEE
# ================================================

# ENVOLVA O CÓDIGO PRINCIPAL DENTRO DE if __name__ == '__main__':
if __name__ == '__main__':
    # Parâmetros: H0, Omega_m, Omega_ond0, n, Omega_EDE0, zc
    ndim = 6
    nwalkers = 50
    nsteps = 2000 # Você pode reduzir para 200 ou 500 para testes mais rápidos

    p0_initial = np.array([70.0, 0.3, 0.01, -3.0, 0.05, 3000.0])
    pos = p0_initial + 1e-4 * np.random.randn(nwalkers, ndim)


    print("\nIniciando simulação MCMC...")

    # Configurando o pool de processamento paralelo
    # Usará todos os núcleos disponíveis por padrão
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                        args=(z_h_data, H_data, H_err, z_sn_data, mu_sn_data, mu_sn_err),
                                        pool=pool)

        sampler.run_mcmc(pos, nsteps, progress=True)
    print("Simulação MCMC concluída.")

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    labels = ["H0", r"$\Omega_m$", r"$\Omega_{ond0}$", "$n$", r"$\Omega_{EDE0}$", "$z_c$"]
    best_params = [np.median(flat_samples[:, i]) for i in range(ndim)]
    errors_minus = [best_params[i] - np.percentile(flat_samples[:, i], 16) for i in range(ndim)]
    errors_plus = [np.percentile(flat_samples[:, i], 84) - best_params[i] for i in range(ndim)]

    print("\n--- Resultados MCMC ---")
    for i, label in enumerate(labels):
        print(f"{label} = {best_params[i]:.3f} +{errors_plus[i]:.3f}/-{errors_minus[i]:.3f}")


    # ================================================
    # 6. ANÁLISE E PLOTS
    # ================================================

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(labels[i])
        ax.axhline(best_params[i], color='red', linestyle='--', label=f'Mediana: {best_params[i]:.2f}')
        ax.legend(fontsize=8, loc='best')
    axes[-1].set_xlabel("Número de Passos")
    plt.suptitle("Cadeias MCMC para Parâmetros do Modelo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig("mcmc_chains.png")
    # plt.show()

    fig = corner.corner(flat_samples, labels=labels, truths=best_params, show_titles=True, title_fmt=".2e",
                        quantiles=[0.16, 0.5, 0.84], smooth=True, hist_kwargs={'density': True})
    plt.suptitle("Diagrama de Contorno (Corner Plot) dos Parâmetros do Modelo")
    plt.savefig("mcmc_corner_plot.png")
    # plt.show()

    # ================================================
    # 7. VISUALIZAÇÃO DOS RESULTADOS DO AJUSTE
    # ================================================

    z_model_plot = np.linspace(min(z_h_data.min(), z_sn_data.min()), max(z_h_data.max(), z_sn_data.max()), 100)

    plt.figure(figsize=(10, 6))
    plt.errorbar(z_h_data, H_data, yerr=H_err, fmt='o', color='blue', label='Dados H(z) Observacionais (Cosmic Chronometers)')

    E_vals_model = np.array([E_modificado(z, *best_params[1:]) for z in z_model_plot])
    H_model_plot = best_params[0] * E_vals_model

    plt.plot(z_model_plot, H_model_plot, 'r-', linewidth=2, label='Modelo Ajustado H(z) (MCMC)')
    plt.xlabel('Redshift (z)', fontsize=12)
    plt.ylabel('H(z) [km/s/Mpc]', fontsize=12)
    plt.title('Ajuste de H(z) com o Modelo Híbrido', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Hz_fit_model.png")
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(z_sn_data, mu_sn_data, yerr=mu_sn_err, fmt='o', color='purple', label=r'Dados Pantheon+SH0ES ($\mu(z)$)')

    mu_model_plot = mu_model(z_model_plot, best_params[0], E_modificado, *best_params[1:])

    plt.plot(z_model_plot, mu_model_plot, 'g-', linewidth=2, label=r'Modelo Ajustado $\mu(z)$ (MCMC)')
    plt.xlabel('Redshift (z)', fontsize=12)
    plt.ylabel(r'Módulo de Distância ($\mu$)', fontsize=12)
    plt.title('Ajuste do Módulo de Distância com o Modelo Híbrido', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("mu_fit_model.png")
    # plt.show()


    print("\nScript de análise MCMC concluído. Verifique os arquivos 'mcmc_chains.png', 'mcmc_corner_plot.png', 'Hz_fit_model.png' e 'mu_fit_model.png' para os resultados.")