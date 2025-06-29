import os
from classy_wrapper import Classy
import numpy as np
import matplotlib.pyplot as plt

# Carrega configuração do CLASS e parâmetros
cfg_yaml = "hidem_config.yaml"
mod_class = "C:/Users/Luiz/OneDrive/Área de Trabalho/AEONCOSMA V2/aeon2/aeoncosma_full/core/class_modificado"

# Lista de z para plot
z = np.linspace(0, 2, 100)

# Exemplo de parâmetros iniciais
params = {
    'hidem_Omega_seed': 0.672,
    'hidem_m': -0.822,
    'hidem_ac': 1.312,
    'omega_b': 0.0224,
    'omega_cdm': 0.12,
    'H0': 70.0,
    'tau_reio': 0.055,
    'A_s': 2.1e-9,
    'n_s': 0.965
}

# Instancia o CLASS
cosmo = Classy(mod_class,
               extra_args={'has_hidem_module': True},
               **params)

cosmo.compute()

Hz = [cosmo.H(zv) for zv in z]
plt.plot(z, Hz, label='HIDEM - exemplo')
plt.xlabel('z')
plt.ylabel('H(z)')
plt.legend()
plt.grid()
plt.savefig('report/figs/HIDEM_example.png')
print("Gráfico salvo em report/figs/HIDEM_example.png")
