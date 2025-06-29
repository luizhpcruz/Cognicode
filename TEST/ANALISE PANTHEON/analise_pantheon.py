import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
import emcee
import corner
import matplotlib.pyplot as plt
import time

# --- Iniciar cronômetro ---
start_time = time.time()

# --- Carregar e Filtrar os Dados ---
try:
    pantheon_data = pd.read_csv('pantheon_data.csv')
    
    # CORREÇÃO 1: Filtrar dados com erros nulos ou inválidos
    valid_indices = pantheon_data['muerr'] > 0
    pantheon_data = pantheon_data[valid_indices]
    
    z_data = pantheon_data['zHD'].values
    mu_data = pantheon_data['mu'].values
    mu_err = pantheon_data['muerr'].values
    
    print(f"Dados do Pantheon carregados e filtrados com sucesso. {len(z_data)} pontos de dados válidos.")

except FileNotFoundError:
    print("Erro: Arquivo 'pantheon_data.csv' não foi encontrado na pasta.")
    exit()

# --- Modelo Teórico e Otimização com Interpolação ---

# O H(z) do modelo é o mesmo
def H_LCDM(z, H0, Omega_m):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

# Cache para armazenar os interpoladores e evitar recalcular
interpolator_cache = {}

def get_mu_interpolator(H0, Omega_m):
    """
    Função que cria e armazena em cache um interpolador para mu(z).
    Esta função é mais lenta, mas só será chamada quando os parâmetros mudarem.
    """
    param_key = (H0, Omega_m)
    if param_key in interpolator_cache:
        return interpolator_cache[param_key]

    c = 299792.458  # Velocidade da luz em km/s
    
    # Criar uma grade de redshift para o pré-cálculo
    z_grid = np.linspace(0, max(z_data) * 1.1, 200)
    
    integrand = lambda z_prime: c / H_LCDM(z_prime, H0, Omega_m)
    
    dist_lum_grid = np.zeros_like(z_grid)
    dist_lum_grid[0] = 0
    
    # Pré-calcular a integral
    integrated_dist = 0
    for i in range(1, len(z_grid)):
        integral_step, _ = quad(integrand, z_grid[i-1], z_grid[i])
        integrated_dist += integral_step
        dist_lum_grid[i] = integrated_dist
        
    dist_lum_grid = (1 + z_grid) * dist_lum_grid
    dist_lum_grid[dist_lum_grid <= 0] = 1e-9

    mu_grid = 5 * np.log10(dist_lum_grid) + 25
    
    # Criar e armazenar o interpolador
    interpolator = interp1d(z_grid, mu_grid, kind='cubic', fill_value="extrapolate")
    interpolator_cache[param_key] = interpolator
    
    return interpolator

# --- Funções para a Análise Bayesiana (MCMC) ---

def log_likelihood(theta):
    H0, Omega_m = theta
    
    # Checagem de prior dentro da likelihood para velocidade
    if not (40 < H0 < 100 and 0.01 < Omega_m < 1.0):
        return -np.inf
    
    try:
        # OTIMIZAÇÃO: Obter o interpolador (rápido se já foi calculado)
        mu_interpolator = get_mu_interpolator(H0, Omega_m)
        # Calcular o modelo usando a interpolação (extremamente rápido)
        model_mu = mu_interpolator(z_data)
    except Exception:
        return -np.inf

    chi2 = np.sum(((mu_data - model_mu) / mu_err)**2)
    return -0.5 * chi2

# O prior agora é mais simples, pois a checagem principal está na likelihood
def log_prior(theta):
    return 0.0

def log_posterior(theta):
    # A log_likelihood já inclui o prior para eficiência
    return log_likelihood(theta)

# --- Execução da Análise MCMC ---
ndim, nwalkers, nsteps = 2, 50, 5000
pos = [np.array([70, 0.3]) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
print("Iniciando MCMC OTIMIZADO com dados de Supernovas do Pantheon...")
sampler.run_mcmc(pos, nsteps, progress=True)

# --- Finalização e Resultados ---
end_time = time.time()
print(f"Análise MCMC finalizada em {end_time - start_time:.2f} segundos.")

samples = sampler.get_chain(discard=1000, thin=15, flat=True)

labels = [r"$H_0$", r"$\Omega_m$"]
fig = corner.corner(samples, labels=labels, show_titles=True, 
                    quantiles=[0.16, 0.5, 0.84], truths=None) # Removido truths para dados reais
plt.suptitle("Posteriores com Dados do Pantheon", fontsize=16)
plt.show()