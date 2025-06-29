import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # Certifique-se de ter scikit-learn instalado

# ===============================
# Parte 1: Simulação Cósmica
# ===============================

# Parâmetros da simulação (AJUSTADOS)
grid_size = 100            # Tamanho da grade (100x100x100)
timesteps = 100            # Número de etapas de tempo
collapse_threshold = 0.8   # Limiar de densidade total para colapso
grav_collapse_factor = 0.07  # Fator de colapso gravitacional

# Inicialização das densidades
dark_matter_density = np.random.rand(grid_size, grid_size, grid_size) * 0.3
baryonic_matter_density = np.random.rand(grid_size, grid_size, grid_size) * 0.044
dark_energy_density = np.full((grid_size, grid_size, grid_size), 0.7)

# Array para rastrear células colapsadas (0 = não colapsada, 1 = colapsada)
collapsed_cells = np.zeros((grid_size, grid_size, grid_size), dtype=int)

def evolve(dark_matter, baryonic_matter, dark_energy, collapsed, steps, threshold, collapse_factor):
    """Função para evoluir as densidades durante as etapas simuladas."""
    print(f"Iniciando evolução por {steps} passos (Limiar={threshold}, Fator Colapso={collapse_factor})...")
    for step in range(steps):
        # Identifica células não colapsadas
        active_mask = (collapsed == 0)

        # 1. Expansão (afetando células ativas)
        expansion = dark_energy[active_mask] * 0.01  # Expansão simplificada
        dark_matter[active_mask] -= expansion
        baryonic_matter[active_mask] -= expansion
        
        # Garante que as densidades não fiquem negativas
        dark_matter[active_mask] = np.clip(dark_matter[active_mask], 0, None)
        baryonic_matter[active_mask] = np.clip(baryonic_matter[active_mask], 0, None)
        
        # 2. Colapso gravitacional (para células ativas)
        total_matter_active = dark_matter[active_mask] + baryonic_matter[active_mask]
        gravitational_collapse = total_matter_active * collapse_factor
        
        # Redistribui a matéria colapsada
        dark_matter[active_mask] += gravitational_collapse * 0.8
        baryonic_matter[active_mask] += gravitational_collapse * 0.2

        # 3. Verifica se a densidade ultrapassou o limiar
        current_total_density = dark_matter[active_mask] + baryonic_matter[active_mask]
        newly_collapsed_mask_local = (current_total_density > threshold)
        
        # Mapeia a máscara local para a grade completa
        indices_active = np.where(active_mask)
        indices_newly_collapsed = tuple(idx[newly_collapsed_mask_local] for idx in indices_active)
        
        if len(indices_newly_collapsed[0]) > 0:
            # Marca as células como colapsadas e "zera" a matéria nelas
            collapsed[indices_newly_collapsed] = 1
            dark_matter[indices_newly_collapsed] = 0 
            baryonic_matter[indices_newly_collapsed] = 0

        if (step + 1) % 10 == 0:
            novas_celulas = len(indices_newly_collapsed[0])
            total_colapsado = np.sum(collapsed)
            print(f"  Passo {step+1}/{steps} concluído. Novas células colapsadas: {novas_celulas}. Total colapsado: {total_colapsado}")
            
    print("Evolução concluída.")
    return dark_matter, baryonic_matter, collapsed

# Executa a simulação e captura os resultados
final_dark_matter, final_baryonic_matter, final_collapsed = evolve(
    dark_matter_density.copy(), 
    baryonic_matter_density.copy(), 
    dark_energy_density.copy(),
    collapsed_cells.copy(),
    timesteps, 
    collapse_threshold,
    grav_collapse_factor
)

# ===============================
# Parte 2: Visualização dos Resultados da Simulação
# ===============================

plt.style.use("dark_background")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
slice_index = grid_size // 2  # Seleciona a fatia central no eixo z

# 1. Matéria Escura Final
im_dm = axs[0].imshow(final_dark_matter[:, :, slice_index], cmap="inferno", vmin=0)
axs[0].set_title(f"Matéria Escura Final (Fatia z={slice_index})")
fig.colorbar(im_dm, ax=axs[0], label="Densidade")

# 2. Matéria Bariônica Final
im_bm = axs[1].imshow(final_baryonic_matter[:, :, slice_index], cmap="viridis", vmin=0)
axs[1].set_title(f"Matéria Bariônica Final (Fatia z={slice_index})")
fig.colorbar(im_bm, ax=axs[1], label="Densidade")

# 3. Células Colapsadas
im_collapsed = axs[2].imshow(final_collapsed[:, :, slice_index], cmap="binary", vmin=0, vmax=1)
axs[2].set_title(f"Células Colapsadas (Fatia z={slice_index})")
cbar_collapsed = fig.colorbar(im_collapsed, ax=axs[2], ticks=[0, 1])
cbar_collapsed.set_ticklabels(["Não Colapsada", "Colapsada"])

plt.suptitle(f"Simulação Ajustada ({timesteps} Passos, Limiar={collapse_threshold}, Fator Colapso={grav_collapse_factor})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Define um diretório de saída válido para salvar os resultados (ajuste conforme necessário)
output_dir = r"C:\Users\Luiz\OneDrive\Área de Trabalho\TEST\REVISAR\modelo LeeME"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plot_filename = os.path.join(output_dir, "simulation_results_adjusted.png")
plt.savefig(plot_filename)
print(f"Gráfico dos resultados ajustados salvo em: {plot_filename}")

# ===============================
# Parte 3: Aplicação do DBSCAN nos Pontos Colapsados
# ===============================

# Extraímos os pontos colapsados da fatia central da simulação
collapsed_slice = final_collapsed[:, :, slice_index]
points = np.argwhere(collapsed_slice == 1)  # Cada entrada é (y, x)

# Definindo os conjuntos de parâmetros para testar
eps_values = [1, 2, 3]         # Valores de ε (distância máxima)
min_samples_values = [3, 5]    # Número mínimo de pontos para formar um cluster

n_rows = len(eps_values)
n_cols = len(min_samples_values)
fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_  # Ruído é identificado por -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        ax = axs2[i, j]
        sc = ax.scatter(points[:, 1], points[:, 0], c=labels, cmap='Paired', s=15)
        ax.set_title(f"DBSCAN: ε={eps}, min_samples={min_samples}\nClusters: {n_clusters}")
        ax.invert_yaxis()  # Ajusta o eixo Y para corresponder à orientação do imshow
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        
# Ajusta o layout e salva o gráfico do DBSCAN
plt.tight_layout()
dbscan_plot_filename = os.path.join(output_dir, "dbscan_results_adjusted.png")
plt.savefig(dbscan_plot_filename)
print(f"Gráfico dos resultados do DBSCAN salvo em: {dbscan_plot_filename}")
plt.show()

# ===============================
# Parte 4: Salvando o Código (Opcional)
# ===============================
code_filename = os.path.join(output_dir, "simulation_code_adjusted.py")
try:
    # __file__ pode não estar definido em alguns ambientes interativos
    with open(code_filename, "w", encoding="utf-8") as fout:
        with open(__file__, "r", encoding="utf-8") as fin:
            fout.write(fin.read())
    print(f"Código ajustado salvo em: {code_filename}")
except Exception as e:
    print(f"Não foi possível salvar o código automaticamente: {e}")