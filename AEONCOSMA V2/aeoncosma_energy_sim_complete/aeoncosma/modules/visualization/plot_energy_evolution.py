
import numpy as np
import matplotlib.pyplot as plt
from modules.energetics.energy_components import rho_r, rho_m, rho_lambda

def plot_energy_densities(z_range, Omega_r, Omega_m, Omega_Lambda, H0):
    rhos_r = rho_r(z_range, Omega_r, H0)
    rhos_m = rho_m(z_range, Omega_m, H0)
    rhos_l = rho_lambda(Omega_Lambda, H0) * np.ones_like(z_range)
    total = rhos_r + rhos_m + rhos_l

    plt.plot(z_range, rhos_r / total, label='Radiação')
    plt.plot(z_range, rhos_m / total, label='Matéria')
    plt.plot(z_range, rhos_l / total, label='Energia escura')
    plt.xlabel("Redshift z")
    plt.ylabel("Fração da densidade total")
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.title("Evolução das densidades energéticas")
    plt.savefig("energy_evolution.png")
    plt.close()
