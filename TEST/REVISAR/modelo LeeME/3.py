import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score  # Importa a Silhouette Score

# ===============================
# Parte 1: Simulação Cósmica
# ===============================

# Parâmetros da simulação
grid_size = 100            # Grade 100x100x100
timesteps = 100            # Número de passos na evolução
collapse_threshold = 0.8   # Limiar para colapso (densidade total)
grav_collapse_factor = 0.07  # Fator de colapso gravitacional

# Inicializa as densidades
dark_matter_density = np.random.rand(grid_size, grid_size, grid_size) * 0.3
baryonic_matter_density = np.random.rand(grid_size, grid_size, grid_size) * 0.044
dark_energy_density = np.full((grid_size, grid_size, grid_size), 0.7)
collapsed_cells = np.zeros((grid_size, grid_size, grid_size), dtype=int)

def evolve(dark_matter, baryonic_matter, dark_energy, collapsed, steps, threshold, collapse_factor):
    """
    Evolui as densidades por um número de passos e marca células que atingem o limiar de colapso.
    """
    print(f"\nIniciando evolução por {steps} passos (Limiar={threshold}, Fator Colapso={collapse_factor})...\n")
    for step in range(steps):
        active_mask = (collapsed == 0)  # células não colapsadas

        # 1. Expansão (afeta apenas células ativas)
        expansion = dark_energy[active_mask] * 0.01
        dark_matter[active_mask] -= expansion
        baryonic_matter[active_mask] -= expansion
        dark_matter[active_mask] = np.clip(dark_matter[active_mask], 0, None)
        baryonic_matter[active_mask] = np.clip(baryonic_matter[active_mask], 0, None)

        # 2. Colapso gravitacional para células ativas
        total_matter_active = dark_matter[active_mask] + baryonic_matter[active_mask]
        gravitational_collapse = total_matter_active * collapse_factor
        dark_matter[active_mask] += gravitational_collapse * 0.8
        baryonic_matter[active_mask] += gravitational_collapse * 0.2

        # 3. Verifica se estes pontos excedem o limiar de colapso
        current_total_density = dark_matter[active_mask] + baryonic_matter[active_mask]
        newly_collapsed_mask_local = (current_total_density > threshold)
        indices_active = np.where(active_mask)
        indices_newly_collapsed = tuple(idx[newly_collapsed_mask_local] for idx in indices_active)
        if len(indices_newly_collapsed[0]) > 0:
            collapsed[indices_newly_collapsed] = 1
            dark_matter[indices_newly_collapsed] = 0 
            baryonic_matter[indices_newly_collapsed] = 0

        if (step + 1) % 10 == 0:
            novas = len(indices_newly_collapsed[0])
            total_colapsado = np.sum(collapsed)
            print(f"Passo {step+1:3d}/{steps} concluído. Novas células colapsadas: {novas:6d}. Total colapsado: {total_colapsado:6d}")
    print("\nEvolução concluída.\n")
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

# Seleciona a fatia central (eixo z)
slice_index = grid_size // 2
dm_slice  = final_dark_matter[:, :, slice_index]
bm_slice  = final_baryonic_matter[:, :, slice_index]
col_slice = final_collapsed[:, :, slice_index]

# ===============================
# Parte 2: Visualização e Salvamento dos Gráficos da Simulação
# ===============================

plt.style.use("dark_background")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Matéria Escura
im_dm = axs[0].imshow(dm_slice, cmap="inferno", vmin=0)
axs[0].set_title(f"Matéria Escura (Fatia z={slice_index})")
fig.colorbar(im_dm, ax=axs[0], label="Densidade")

# 2. Matéria Bariônica
im_bm = axs[1].imshow(bm_slice, cmap="viridis", vmin=0)
axs[1].set_title(f"Matéria Bariônica (Fatia z={slice_index})")
fig.colorbar(im_bm, ax=axs[1], label="Densidade")

# 3. Células Colapsadas
im_collapsed = axs[2].imshow(col_slice, cmap="binary", vmin=0, vmax=1)
axs[2].set_title(f"Células Colapsadas (Fatia z={slice_index})")
cbar = fig.colorbar(im_collapsed, ax=axs[2], ticks=[0, 1])
cbar.ax.set_yticklabels(["Não Colapsada", "Colapsada"])

plt.suptitle(f"Simulação Cósmica ({timesteps} passos, Limiar={collapse_threshold}, Fator Colapso={grav_collapse_factor})")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Salva o gráfico da simulação na pasta do código
output_dir = os.getcwd()  # Diretório atual
plot_filename = os.path.join(output_dir, "simulation_results.png")
plt.savefig(plot_filename)
print(f"Gráfico da simulação salvo em: {plot_filename}")
plt.close(fig)

# ===============================
# Parte 3: Visualização e Salvamento dos Gráficos do DBSCAN
# ===============================

# Extraímos os pontos colapsados da fatia central para o DBSCAN
points = np.argwhere(col_slice == 1)  # Cada linha: (row, col)
grid_shape = col_slice.shape  # Ex: (100, 100)

# Testa conjuntos de parâmetros: (ε e min_samples)
eps_values = [1, 2, 3]
min_samples_values = [3, 5]

n_rows = len(eps_values)
n_cols = len(min_samples_values)
fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_  # -1 identifica ruído
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        ax = axs2[i, j]
        ax.scatter(points[:, 1], points[:, 0], c=labels, cmap='Paired', s=15)
        ax.set_title(f"DBSCAN: ε={eps}, min_samples={min_samples}\nClusters: {n_clusters}")
        ax.invert_yaxis()  # Alinha com a orientação do imshow
        ax.set_xlabel("Coordenada X")
        ax.set_ylabel("Coordenada Y")
        
plt.tight_layout()
dbscan_filename = os.path.join(output_dir, "dbscan_results.png")
plt.savefig(dbscan_filename)
print(f"Gráfico do DBSCAN salvo em: {dbscan_filename}")
plt.close(fig2)

# ===============================
# Parte 4: Seleção Automática de ε com k-Distance e KneeLocator
# ===============================

# Define um valor de min_samples para a seleção automática (por exemplo, 5)
min_samples_auto = 5

# Calcula as distâncias do min_samples_auto-ésimo vizinho para todos os pontos
neighbors = NearestNeighbors(n_neighbors=min_samples_auto)
neighbors_fit = neighbors.fit(points)
distances, indices = neighbors_fit.kneighbors(points)

# Seleciona a distância do min_samples_auto-ésimo vizinho e ordena
k_distances = np.sort(distances[:, min_samples_auto - 1])

# Plota o gráfico de k-distância e salva o resultado
plt.figure(figsize=(8, 4))
plt.plot(k_distances, marker="o", linestyle="--")
plt.xlabel("Pontos ordenados")
plt.ylabel(f"Distância do {min_samples_auto}-ésimo vizinho")
plt.title("Gráfico de k-distância para seleção de ε")
plt.grid(True)
kdist_filename = os.path.join(output_dir, "k_distance_plot.png")
plt.savefig(kdist_filename)
plt.close()
print(f"Gráfico de k-distância salvo em: {kdist_filename}")

# Usa o KneeLocator para encontrar o "joelho"
kneedle = KneeLocator(range(len(k_distances)), k_distances, S=1.0, curve="convex", direction="increasing")
if kneedle.knee is not None:
    epsilon_auto = k_distances[kneedle.knee]
    print(f"Valor de ε selecionado automaticamente: {epsilon_auto:.4f}")
else:
    epsilon_auto = None
    print("Não foi possível identificar automaticamente um valor de ε.")

# Se um valor automático de ε foi encontrado, aplica o DBSCAN com esse valor e calcula a Silhouette Score
if epsilon_auto is not None:
    db_auto = DBSCAN(eps=epsilon_auto, min_samples=min_samples_auto).fit(points)
    labels_auto = db_auto.labels_
    n_clusters_auto = len(set(labels_auto)) - (1 if -1 in labels_auto else 0)
    if n_clusters_auto > 1:
        score_auto = silhouette_score(points, labels_auto)
        print(f"Silhouette Score para o DBSCAN automático: {score_auto:.4f}")
    else:
        print("Não há clusters suficientes para calcular a Silhouette Score.")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 1], points[:, 0], c=labels_auto, cmap="Paired", s=15)
    plt.gca().invert_yaxis()
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.title(f"DBSCAN com ε (auto) = {epsilon_auto:.4f}\nmin_samples = {min_samples_auto}, Clusters: {n_clusters_auto}")
    auto_dbscan_filename = os.path.join(output_dir, "dbscan_auto_results.png")
    plt.savefig(auto_dbscan_filename)
    plt.close()
    print(f"Gráfico do DBSCAN com ε automático salvo em: {auto_dbscan_filename}")