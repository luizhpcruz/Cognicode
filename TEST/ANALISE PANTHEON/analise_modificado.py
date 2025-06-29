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
    valid_indices = pantheon_data['muerr'] > 0
    pantheon_data = pantheon_data[valid_indices]
    
    z_data = pantheon_data['zHD'].values
    mu_data = pantheon_data['mu'].values
    mu_err = pantheon_data['muerr'].values
    
    print(f"Dados do Pantheon carregados. {len(z_data)} pontos de dados válidos para a análise.")

except FileNotFoundError:
    print("Erro: Arquivo 'pantheon_data.csv' não foi encontrado na pasta.")
    exit()

# --- Modelo Modificado com 4 Parâmetros ---

def H_modificado(z, H0, Omega_m, Rond0, n):
    # Esta é a forma do seu modelo como definido nos seus scripts anteriores
    termo_de = (1 - Omega_m - Rond0) + Rond0 * (1 + z)**n
    # Garantir que o termo sob a raiz não seja negativo
    termo_raiz = Omega_m * (1 + z)**3 + termo_de
    termo_raiz[termo_raiz < 0] = 0 # Evita erros matemáticos
    return H0 * np.sqrt(termo_raiz)

# Cache para o interpolador do modelo modificado
interpolator_cache_mod = {}

def get_mu_interpolator_mod(H0, Omega_m, Rond0, n):
    param_key = (H0, Omega_m, Rond0, n)
    if param_key in interpolator_cache_mod:
        return interpolator_cache_mod[param_key]

    c = 299792.458
    z_grid = np.linspace(0, max(z_data) * 1.1, 200)
    
    integrand = lambda z_prime: c / H_modificado(z_prime, H0, Omega_m, Rond0, n)
    
    dist_lum_grid = np.zeros_like(z_grid)
    integrated_dist = 0
    for i in range(1, len(z_grid)):
        try:
            integral_step, _ = quad(integrand, z_grid[i-1], z_grid[i])
            integrated_dist += integral_step
        except Exception:
            # Se a integral falhar, o modelo é ruim, retorna -inf na likelihood
            return None 
        dist_lum_grid[i] = integrated_dist
        
    dist_lum_grid = (1 + z_grid) * dist_lum_grid
    dist_lum_grid[dist_lum_grid <= 0] = 1e-9

    mu_grid = 5 * np.log10(dist_lum_grid) + 25
    
    interpolator = interp1d(z_grid, mu_grid, kind='cubic', fill_value="extrapolate")
    interpolator_cache_mod[param_key] = interpolator
    
    return interpolator

# --- Funções para a Análise Bayesiana (MCMC) ---

def log_likelihood_mod(theta):
    H0, Omega_m, Rond0, n = theta
    
    # Checagem de prior para velocidade
    if not (40 < H0 < 100 and 0.01 < Omega_m < 0.5 and -2.0 < Rond0 < 2.0 and -5.0 < n < 5.0):
        return -np.inf
    
    mu_interpolator = get_mu_interpolator_mod(H0, Omega_m, Rond0, n)
    if mu_interpolator is None:
        return -np.inf # Parâmetros ruins levaram a falha na integração

    model_mu = mu_interpolator(z_data)
        
    chi2 = np.sum(((mu_data - model_mu) / mu_err)**2)
    return -0.5 * chi2

def log_posterior_mod(theta):
    return log_likelihood_mod(theta) # Prior já está na likelihood

# --- Execução da Análise MCMC ---
ndim, nwalkers, nsteps = 4, 100, 6000 # Mais walkers/passos para um espaço de 4D
# Posição inicial dos walkers
pos = [np.array([73, 0.3, 0.7, -1.0]) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_mod)
print("Iniciando MCMC OTIMIZADO para o Modelo Modificado...")
sampler.run_mcmc(pos, nsteps, progress=True)

# --- Finalização e Resultados ---
end_time = time.time()
print(f"Análise MCMC finalizada em {end_time - start_time:.2f} segundos.")

samples = sampler.get_chain(discard=1500, thin=30, flat=True)

labels = [r"$H_0$", r"$\Omega_m$", r"$R_0$", r"$n$"]
fig = corner.corner(samples, labels=labels, show_titles=True, 
                    quantiles=[0.16, 0.5, 0.84])
plt.suptitle("Posteriores para o Modelo Modificado com Dados do Pantheon", fontsize=16)
plt.show()