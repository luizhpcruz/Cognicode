import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Parâmetros da simulação
n_galaxies = 50000
z0 = 0.5
np.random.seed(42)

# Distribuição de redshift (z^2 * exp(-z/z0))
z = np.random.gamma(shape=3, scale=z0 / 3, size=n_galaxies)
z = z[z < 1.2]
n_galaxies = len(z)

# Magnitude i simulada
mag_i = 18 + 4 * z + np.random.normal(0, 1.0, n_galaxies)
mag_i = np.clip(mag_i, 15, 25)

# Cor g-i simulada (Galáxias mais vermelhas em z mais alto e/ou mais massivas implicitamente via mag_i)
color_gi = 0.8 + 0.5 * np.tanh(2 * (z - 0.3)) + np.random.normal(0, 0.2, n_galaxies)

# Massa estelar (log10) - Correlação com mag_i
log_mass = 11 - 0.2 * (mag_i - 20) + np.random.normal(0, 0.3, n_galaxies)

# Shear com ruído
shear_g1 = np.random.normal(0, 0.03, n_galaxies)
shear_g2 = np.random.normal(0, 0.03, n_galaxies)

# Tamanho efetivo (arcseg) - Correlação com massa e mag_i
size_arcsec = 0.6 + 0.3 * (log_mass - 10.5) - 0.05 * (mag_i - 20) + np.random.normal(0, 0.15, n_galaxies)
size_arcsec = np.clip(size_arcsec, 0.1, 2.5)

# --- NOVAS PROPRIEDADES --- 

# Taxa de Formação Estelar (SFR) em log10(M_sun/yr)
# Modelo simplificado: depende da massa, cor e redshift
# SFR maior para massas intermediárias (quenching em altas massas), mais azuis, e z mais alto (pico ~z=2, mas simplificado aqui)
log_sfr_base = 0.5 * (log_mass - 10.5) - 1.0 * (color_gi - 1.0) + 0.5 * z 
# Adiciona um turnover/quenching para altas massas
quenching = -1.0 * np.maximum(0, log_mass - 11.2)**2
log_sfr = log_sfr_base + quenching + np.random.normal(0, 0.4, n_galaxies) # Adiciona scatter
log_sfr = np.clip(log_sfr, -3, 2.5) # Limita a faixa de SFR

# Metalicidade (log10(Z/Z_sun))
# Modelo simplificado: Relação Massa-Metalicidade com evolução em z
# Z maior para maior massa e menor z
log_metallicity = -0.5 + 0.3 * (log_mass - 10.5) - 0.2 * z + np.random.normal(0, 0.15, n_galaxies)
log_metallicity = np.clip(log_metallicity, -1.5, 0.5) # Limita a faixa

# --- FIM NOVAS PROPRIEDADES ---

# RA e Dec com estruturas
ra = np.random.uniform(0, 1, n_galaxies)
dec = np.random.uniform(0, 1, n_galaxies)

# Aglomerados
cluster_coords = [(0.3, 0.3), (0.7, 0.5), (0.5, 0.8)]
for (cx, cy) in cluster_coords:
    idx = np.random.choice(n_galaxies, size=int(n_galaxies * 0.03), replace=False)
    ra[idx] = np.random.normal(cx, 0.015, len(idx))
    dec[idx] = np.random.normal(cy, 0.015, len(idx))

# Filamento
n_filament = int(n_galaxies * 0.1)
filament_idx = np.random.choice(n_galaxies, size=n_filament, replace=False)
t = np.linspace(0, 1, n_filament)
ra[filament_idx] = 0.4 + 0.2 * t + np.random.normal(0, 0.01, n_filament)
dec[filament_idx] = 0.2 + 0.5 * t + np.random.normal(0, 0.01, n_filament)

# Compilar DataFrame
data = pd.DataFrame({
    "z": z,
    "mag_i": mag_i,
    "g_i": color_gi,
    "log_mass": log_mass,
    "log_sfr": log_sfr,             # Nova coluna
    "log_metallicity": log_metallicity, # Nova coluna
    "shear_g1": shear_g1,
    "shear_g2": shear_g2,
    "size_arcsec": size_arcsec,
    "ra": ra,
    "dec": dec
})

print("Catálogo sintético gerado com SFR e Metalicidade.")
print(data.head())

# Gráficos (Atualizado para 3x3)
plt.style.use("default") # Reset style
plt.figure(figsize=(18, 18))
spec = gridspec.GridSpec(3, 3)

# Linha 1
ax1 = plt.subplot(spec[0, 0])
pc1 = ax1.hexbin(data["ra"], data["dec"], gridsize=100, cmap="plasma", bins="log")
ax1.set_title("Distribuição Angular com Estruturas (RA/Dec)")
ax1.set_xlabel("RA normalizado")
ax1.set_ylabel("Dec normalizado")
plt.colorbar(pc1, ax=ax1, label="log(N galáxias)")

ax2 = plt.subplot(spec[0, 1])
pc2 = ax2.scatter(data["z"], data["size_arcsec"], s=2, alpha=0.4, color="deepskyblue")
ax2.set_xlabel("Redshift z")
ax2.set_ylabel("Tamanho efetivo (arcseg)")
ax2.set_title("Tamanho vs Redshift")

ax3 = plt.subplot(spec[0, 2])
pc3 = ax3.hexbin(data["log_mass"], data["size_arcsec"], gridsize=50, cmap="inferno")
ax3.set_xlabel("log₁₀(M*/M☉)")
ax3.set_ylabel("Tamanho efetivo (arcseg)")
ax3.set_title("Tamanho vs Massa Estelar")
plt.colorbar(pc3, ax=ax3, label="N galáxias")

# Linha 2
ax4 = plt.subplot(spec[1, 0])
pc4 = ax4.scatter(data["mag_i"], data["log_mass"], s=2, alpha=0.4, c=data["z"], cmap="viridis")
ax4.set_xlabel("mag_i")
ax4.set_ylabel("log₁₀(M*/M☉)")
ax4.set_title("Magnitude i vs Massa Estelar")
plt.colorbar(pc4, ax=ax4, label="Redshift")

ax5 = plt.subplot(spec[1, 1])
pc5 = ax5.scatter(data["log_mass"], data["log_sfr"], s=2, alpha=0.4, c=data["g_i"], cmap="coolwarm_r") # Cor por g-i
ax5.set_xlabel("log₁₀(M*/M☉)")
ax5.set_ylabel("log₁₀(SFR / [M☉/yr])")
ax5.set_title("SFR vs Massa Estelar (Cor: g-i)")
plt.colorbar(pc5, ax=ax5, label="Cor g-i")

ax6 = plt.subplot(spec[1, 2])
pc6 = ax6.scatter(data["log_mass"], data["log_metallicity"], s=2, alpha=0.4, c=data["z"], cmap="magma") # Cor por z
ax6.set_xlabel("log₁₀(M*/M☉)")
ax6.set_ylabel("log₁₀(Z/Z☉)")
ax6.set_title("Metalicidade vs Massa Estelar (Cor: z)")
plt.colorbar(pc6, ax=ax6, label="Redshift")

# Linha 3
ax7 = plt.subplot(spec[2, 0])
ax7.hist(data["log_sfr"], bins=40, color="lightgreen", edgecolor="black")
ax7.set_xlabel("log₁₀(SFR / [M☉/yr])")
ax7.set_ylabel("Número de galáxias")
ax7.set_title("Histograma de SFR")

ax8 = plt.subplot(spec[2, 1])
ax8.hist(data["log_metallicity"], bins=40, color="gold", edgecolor="black")
ax8.set_xlabel("log₁₀(Z/Z☉)")
ax8.set_ylabel("Número de galáxias")
ax8.set_title("Histograma de Metalicidade")

ax9 = plt.subplot(spec[2, 2])
ax9.quiver(data["ra"][::400], data["dec"][::400],
           data["shear_g1"][::400], data["shear_g2"][::400],
           scale=0.01, width=0.0015, color="black", alpha=0.8)
ax9.set_title("Mapa de Shear (γ₁, γ₂)")
ax9.set_xlabel("RA")
ax9.set_ylabel("Dec")
# ax9.set_facecolor("lightgrey") # Remover fundo preto se não usar quiver

plt.suptitle("Catálogo Sintético de Galáxias com SFR e Metalicidade", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Salva a figura
plot_filename = "/home/ubuntu/catalog_plots_extended.png"
plt.savefig(plot_filename)
print(f"Gráfico do catálogo estendido salvo em: {plot_filename}")

# Salva o código modificado
code_filename = "/home/ubuntu/catalog_generator_extended.py"
with open(code_filename, "w") as f:
    # Escreve o próprio conteúdo do script atual no arquivo
    with open(__file__, "r") as current_script:
        f.write(current_script.read())
print(f"Código do catálogo estendido salvo em: {code_filename}")

