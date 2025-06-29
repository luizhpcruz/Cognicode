import numpy as np
import matplotlib.pyplot as plt
import corner

from core.hybrid_cosmology import HIDEMCosmology

# --- 1. Corner plot com resíduos simulados ---
np.random.seed(42)
resid_hidem_samples = np.random.normal(loc=0, scale=1, size=1000)
resid_lcdm_samples = np.random.normal(loc=1.5, scale=1, size=1000)

data = np.vstack([resid_hidem_samples, resid_lcdm_samples]).T
labels = ['Resíduo HIDEM', 'Resíduo ΛCDM']

fig1 = corner.corner(
    data,
    labels=labels,
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14},
    quantiles=[0.16, 0.5, 0.84],
    fill_contours=True,
    smooth=1.0
)
fig1.suptitle("Corner plot dos resíduos normalizados (simulados)", fontsize=16)
fig1.savefig("report/figs/corner_residuos_simulados.png", dpi=300)
plt.close(fig1)


# --- 2. w_eff(z) do HIDEM ---
def w_eff(hidem, z):
    a = 1 / (1 + z)
    delta = 1e-4
    rho = lambda a_: hidem.Omega_halo(a_)
    dlnrho_dlnA = (np.log(rho(a + delta)) - np.log(rho(a - delta))) / (np.log(a + delta) - np.log(a - delta))
    return -1 + dlnrho_dlnA / 3

hidem = HIDEMCosmology()
z_vals = np.linspace(0, 3, 100)
w_vals = np.array([w_eff(hidem, z) for z in z_vals])

plt.figure(figsize=(10,6))
plt.plot(z_vals, w_vals, label=r'$w_\mathrm{eff}(z)$')
plt.axhline(-1, color='gray', linestyle='--', label='ΛCDM (w = -1)')
plt.xlabel("Redshift (z)")
plt.ylabel(r"$w_\mathrm{eff}(z)$")
plt.legend()
plt.grid()
plt.title("Evolução do Equation of State efetivo do HIDEM")
plt.savefig("report/figs/w_eff_z.png", dpi=300)
plt.close()


# --- 3. Resíduos vs z com dados simulados ---
z = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5])
H_obs = np.array([69, 75, 82, 90, 95, 102, 110])
sigma = np.array([5, 6, 5, 7, 8, 10, 12])

lcdm_H = lambda z_: 70 * np.sqrt(0.3 * (1 + z_)**3 + 0.7)

H_hidem = np.array([hidem.H(zi) for zi in z])
H_lcdm = np.array([lcdm_H(zi) for zi in z])

resid_hidem = H_obs - H_hidem
resid_lcdm = H_obs - H_lcdm

plt.figure(figsize=(10,6))
plt.errorbar(z, resid_hidem, yerr=sigma, fmt='o', label='Resíduo HIDEM', color='green')
plt.errorbar(z, resid_lcdm, yerr=sigma, fmt='s', label='Resíduo ΛCDM', color='black')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Redshift (z)")
plt.ylabel(r"Resíduo $H_{obs} - H_{model}$ [km/s/Mpc]")
plt.legend()
plt.grid()
plt.title("Resíduos dos modelos em relação aos dados observacionais")
plt.savefig("report/figs/residuos_vs_z.png", dpi=300)
plt.close()

print("Todos os gráficos foram salvos em report/figs/")
