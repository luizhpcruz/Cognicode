from scipy.integrate import quad
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# Dados observacionais já carregados (a serem reutilizados)
z_data = np.array([0.070, 0.100, 0.120, 0.170, 0.179, 0.199, 0.200, 0.270, 0.280, 0.352,
                   0.380, 0.400, 0.440, 0.450, 0.480, 0.570, 0.593, 0.600, 0.730, 0.781,
                   0.875, 0.880, 0.900, 1.037, 1.300, 1.363, 1.430, 1.530, 1.750, 1.965,
                   2.340, 2.360])
H_data = np.array([69.0, 69.0, 68.6, 83.0, 75.0, 75.0, 72.9, 77.0, 88.8, 83.0,
                   83.0, 95.0, 84.4, 82.6, 87.9, 92.4, 104.0, 87.9, 97.3, 105.0,
                   125.0, 90.0, 117.0, 154.0, 168.0, 160.0, 177.0, 140.0, 202.0, 186.5,
                   222.0, 226.0])
sigma_H_data = np.array([19.6, 12.0, 26.2, 8.0, 4.0, 5.0, 29.6, 14.0, 36.6, 14.0,
                         13.5, 17.0, 7.1, 12.1, 7.0, 4.5, 13.0, 17.8, 7.0, 12.0,
                         17.0, 40.0, 23.0, 20.0, 17.0, 33.6, 18.0, 14.0, 40.0, 50.4,
                         32.0, 28.0])

# Modelo CPL
def H_CPL(z, H0, Omega_m, w0, wa):
    Omega_DE = 1.0 - Omega_m

    def integrand(z_prime):
        return (1 + w0 + wa * z_prime / (1 + z_prime)) / (1 + z_prime)

    integral = np.array([quad(integrand, 0, zi)[0] for zi in np.atleast_1d(z)])
    exponent = 3 * integral
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_DE * np.exp(exponent))

# Log-likelihood
def log_likelihood(theta):
    H0, Omega_m, w0, wa = theta
    if not (50.0 < H0 < 100.0 and 0.01 < Omega_m < 1.0 and -2.0 < w0 < 0.0 and -2.0 < wa < 2.0):
        return -np.inf
    H_model = H_CPL(z_data, H0, Omega_m, w0, wa)
    chi2 = np.sum(((H_data - H_model) / sigma_H_data) ** 2)
    return -0.5 * chi2

# Inicialização do MCMC
ndim = 4
nwalkers = 50
initial_guess = [68.4, 0.15, -0.8, 0.7]
pos = initial_guess + 1e-2 * np.random.randn(nwalkers, ndim)

# Rodar MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
sampler.run_mcmc(pos, 5000, progress=True)

# Descartar o burn-in
samples = sampler.get_chain(discard=1000, flat=True)

# Fazer gráfico corner
fig = corner.corner(samples, labels=["H0", "Ωm", "w0", "wa"], truths=initial_guess)
plt.show()
import numpy as np

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
labels = ["H0", "Omega_m", "w0", "wa"]

for i in range(len(labels)):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{labels[i]} = {mcmc[1]:.3f} (+{q[1]:.3f}, -{q[0]:.3f})")

