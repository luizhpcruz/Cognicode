# Este script realiza o ajuste bayesiano de dois modelos cosmológicos aos dados H(z):
# 1. Modelo Lambda-CDM padrão
# 2. Modelo Modificado com um termo extra de energia escura Rond0*(1+z)^n

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# --- Carregar os dados observacionais ---
dados = np.loadtxt("Hz_Pantheon_BAO_CMB.txt")
z_data, Hz_data, Hz_err = dados[:,0], dados[:,1], dados[:,2]

# --- Constantes ---
c = 299792.458  # km/s

# --- Modelo Lambda-CDM ---
def modelo_LCDM(params, z):
    H0, Omega_m = params
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

# --- Modelo Modificado ---
def modelo_modificado(params, z):
    H0, Omega_m, Rond0, n = params
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m - Rond0 * (1 + z)**n) + Rond0 * (1 + z)**n)

# --- Funções de verossimilhança ---
def ln_likelihood_lcdm(params):
    H_model = modelo_LCDM(params, z_data)
    return -0.5 * np.sum(((Hz_data - H_model) / Hz_err)**2)

def ln_likelihood_modificado(params):
    H_model = modelo_modificado(params, z_data)
    return -0.5 * np.sum(((Hz_data - H_model) / Hz_err)**2)

# --- Priors uniformes ---
def ln_prior_lcdm(params):
    H0, Omega_m = params
    if 40 < H0 < 100 and 0.01 < Omega_m < 1.0:
        return 0.0
    return -np.inf

def ln_prior_modificado(params):
    H0, Omega_m, Rond0, n = params
    if 40 < H0 < 100 and 0.01 < Omega_m < 1.0 and 0.0 < Rond0 < 3.0 and 0.0 < n < 3.0:
        return 0.0
    return -np.inf

def ln_posterior_lcdm(params):
    lp = ln_prior_lcdm(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_lcdm(params)

def ln_posterior_modificado(params):
    lp = ln_prior_modificado(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_modificado(params)

# --- MCMC ---
def run_mcmc(model, ln_posterior, ndim, nwalkers=50, nsteps=5000, p0_spread=1e-1):
    np.random.seed(42)
    p0 = np.random.rand(nwalkers, ndim) * p0_spread
    if model == "lcdm":
        p0[:,0] = 70 + p0[:,0]*10
        p0[:,1] = 0.3 + p0[:,1]*0.1
    elif model == "modificado":
        p0[:,0] = 70 + p0[:,0]*10
        p0[:,1] = 0.3 + p0[:,1]*0.1
        p0[:,2] = 1.0 + p0[:,2]*0.5
        p0[:,3] = 1.0 + p0[:,3]*0.5

    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior)
    sampler.run_mcmc(p0, nsteps, progress=True)
    samples = sampler.get_chain(discard=1000, flat=True)
    return samples

# --- Executar ajustes ---
print("--- Simulação para o Modelo Lambda-CDM ---")
samples_lcdm = run_mcmc("lcdm", ln_posterior_lcdm, ndim=2)

print("\n--- Simulação para o Modelo de Energia Escura Modificada ---")
samples_mod = run_mcmc("modificado", ln_posterior_modificado, ndim=4)

# --- Estatísticas dos ajustes ---
def estatisticas(samples, nomes_param):
    resultados = {}
    for i, nome in enumerate(nomes_param):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        resultados[nome] = (mcmc[1], q[0], q[1])
    return resultados

def chi2_min(model_func, best_params):
    H_model = model_func(best_params, z_data)
    return np.sum(((Hz_data - H_model) / Hz_err)**2)

result_lcdm = estatisticas(samples_lcdm, ["H0", "Omega_m"])
result_mod = estatisticas(samples_mod, ["H0", "Omega_m", "Rond0", "n"])

print("\nResultados do ajuste bayesiano (Lambda-CDM):")
for k, (med, err1, err2) in result_lcdm.items():
    print(f"{k} = {med:.3f} +{err2:.3f} -{err1:.3f}")
print(f"Chi^2 (Lambda-CDM) = {chi2_min(modelo_LCDM, [result_lcdm[k][0] for k in result_lcdm]):.2f}")

print("\nResultados do ajuste bayesiano (Modelo Modificado):")
for k, (med, err1, err2) in result_mod.items():
    print(f"{k} = {med:.3f} +{err2:.3f} -{err1:.3f}")
print(f"Chi^2 (Modelo Modificado) = {chi2_min(modelo_modificado, [result_mod[k][0] for k in result_mod]):.2f}")

# --- Corner plots ---
corner.corner(samples_lcdm, labels=["H0", "Omega_m"], show_titles=True)
plt.savefig("revisar12_lcdm.png", dpi=300)

corner.corner(samples_mod, labels=["H0", "Omega_m", "Rond0", "n"], show_titles=True)
plt.savefig("revisar12_modificado.png", dpi=300)
plt.close('all')
