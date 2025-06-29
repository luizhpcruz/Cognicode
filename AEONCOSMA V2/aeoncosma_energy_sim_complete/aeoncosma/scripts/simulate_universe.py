
import numpy as np
from modules.visualization.plot_energy_evolution import plot_energy_densities
from modules.visualization.pie_chart_energies import plot_pie

H0 = 67.66
Omega_m = 0.3111
Omega_r = 5e-5
Omega_Lambda = 0.6889

z_range = np.logspace(-3, 4, 500) - 1
plot_energy_densities(z_range, Omega_r, Omega_m, Omega_Lambda, H0)
plot_pie(Omega_r, Omega_m, Omega_Lambda)

print("Simulação concluída. Gráficos salvos.")
