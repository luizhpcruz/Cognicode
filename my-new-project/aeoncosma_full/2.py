import numpy as np
import matplotlib.pyplot as plt
import corner

# Simula dados reais — substitua pelos seus dados observacionais
z = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5])
H_obs = np.array([69, 75, 82, 90, 95, 102, 110])
sigma = np.array([5, 6, 5, 7, 8, 10, 12])

# Importa classes dos seus modelos
from core.hybrid_cosmology import HIDEMCosmology

hidem = HIDEMCosmology()
lcdm_H = lambda z_: 70 * np.sqrt(0.3*(1+z_)**3 + 0.7)  # LCDM simples

H_hidem = np.array([hidem.H(zi) for zi in z])
H_lcdm = np.array([lcdm_H(zi) for zi in z])

resid_hidem = (H_obs - H_hidem) / sigma
resid_lcdm = (H_obs - H_lcdm) / sigma

data = np.vstack([resid_hidem, resid_lcdm]).T
labels = ['Resíduo HIDEM', 'Resíduo ΛCDM']

fig = corner.corner(data, labels=labels, show_titles=True, title_fmt=".2f",
                    title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14})
fig.suptitle("Corner plot dos resíduos normalizados", fontsize=16)
plt.savefig("report/figs/corner_residuos.png", dpi=300)
plt.show()
