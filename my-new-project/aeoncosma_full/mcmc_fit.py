import emcee
import corner
import numpy as np
from core.hybrid_cosmology import HIDEMCosmology

# Configuração do MCMC para HIDEM
def log_prior_HIDEM(theta):
    H0, Omega_m, Omega_seed, m, a_c = theta
    if (50 < H0 < 90 and 0.1 < Omega_m < 0.5 and 
        0 < Omega_seed < 0.3 and 0.1 < m < 2.0 and 0.1 < a_c < 1.5):
        return 0.0
    return -np.inf

def log_likelihood_HIDEM(theta):
    H0, Omega_m, Omega_seed, m, a_c = theta
    cosmo = HIDEMCosmology(H0=H0, Omega_m=Omega_m, 
                          Omega_seed=Omega_seed, m=m, a_c=a_c)
    model = np.array([cosmo.H(z) for z in z_data])
    return -0.5 * np.sum(((Hz_data - model) / Hz_err)**2)

# Execução
sampler_HIDEM = emcee.EnsembleSampler(nwalkers, ndim, 
                                     lambda x: log_prior_HIDEM(x) + log_likelihood_HIDEM(x))
sampler_HIDEM.run_mcmc(pos_HIDEM, 1000, progress=True)