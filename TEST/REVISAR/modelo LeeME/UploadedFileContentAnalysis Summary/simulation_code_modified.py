import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da simulação
grid_size = 100  # Tamanho da grade (100x100x100)
timesteps = 50   # Número de etapas de tempo
collapse_threshold = 1.5 # Limiar de densidade total para colapso (ajustável)

# Inicialização das densidades
dark_matter_density = np.random.rand(grid_size, grid_size, grid_size) * 0.3
baryonic_matter_density = np.random.rand(grid_size, grid_size, grid_size) * 0.044
dark_energy_density = np.full((grid_size, grid_size, grid_size), 0.7)

# Array para rastrear células colapsadas (0 = não colapsada, 1 = colapsada)
collapsed_cells = np.zeros((grid_size, grid_size, grid_size), dtype=int)

# Função para simular a evolução (modificada)
def evolve(dark_matter, baryonic_matter, dark_energy, collapsed, steps, threshold):
    print(f"Iniciando evolução por {steps} passos...")
    for step in range(steps):
        # Identifica células não colapsadas
        active_mask = (collapsed == 0)

        # 1. Expansão (afeta apenas células ativas)
        expansion = dark_energy[active_mask] * 0.01 # Expansão simplificada
        dark_matter[active_mask] -= expansion
        baryonic_matter[active_mask] -= expansion

        # Garante que densidades não sejam negativas após expansão
        dark_matter[active_mask] = np.clip(dark_matter[active_mask], 0, None)
        baryonic_matter[active_mask] = np.clip(baryonic_matter[active_mask], 0, None)

        # 2. Colapso Gravitacional (considera toda a matéria em células ativas)
        total_matter_active = dark_matter[active_mask] + baryonic_matter[active_mask]
        gravitational_collapse = total_matter_active * 0.05 # Colapso simplificado
        
        # Redistribui matéria colapsada (apenas em células ativas)
        dark_matter[active_mask] += gravitational_collapse * 0.8
        baryonic_matter[active_mask] += gravitational_collapse * 0.2

        # 3. Verifica Limiar de Colapso (em células ativas)
        current_total_density = dark_matter[active_mask] + baryonic_matter[active_mask]
        newly_collapsed_mask_local = (current_total_density > threshold)
        
        # Mapeia a máscara local de volta para a grade completa
        indices_active = np.where(active_mask)
        indices_newly_collapsed = tuple(idx[newly_collapsed_mask_local] for idx in indices_active)

        if len(indices_newly_collapsed[0]) > 0:
            print(f"  Passo {step+1}: {len(indices_newly_collapsed[0])} novas células colapsaram.")
            # Marca como colapsada
            collapsed[indices_newly_collapsed] = 1
            # Remove matéria da célula colapsada (ou fixa em valor alto, opcional)
            dark_matter[indices_newly_collapsed] = 0 
            baryonic_matter[indices_newly_collapsed] = 0
            # Opcional: Fixar energia escura também?
            # dark_energy[indices_newly_collapsed] = 0 

        if (step + 1) % 10 == 0:
            print(f"  Passo {step+1}/{steps} concluído. Total colapsado: {np.sum(collapsed)}")

    print("Evolução concluída.")
    return dark_matter, baryonic_matter, collapsed

# Executa a simulação modificada
final_dark_matter, final_baryonic_matter, final_collapsed = evolve(
    dark_matter_density.copy(), # Passa cópias para não modificar os arrays originais
    baryonic_matter_density.copy(), 
    dark_energy_density.copy(),
    collapsed_cells.copy(),
    timesteps, 
    collapse_threshold
)

# --- Visualização --- 

plt.style.use('dark_background') # Usar fundo escuro para melhor visualização com 'inferno'
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
slice_index = grid_size // 2

# 1. Matéria Escura Final
im_dm = axs[0].imshow(final_dark_matter[:, :, slice_index], cmap='inferno', vmin=0)
axs[0].set_title(f'Matéria Escura Final (Fatia z={slice_index})')
fig.colorbar(im_dm, ax=axs[0], label='Densidade')

# 2. Matéria Bariônica Final
im_bm = axs[1].imshow(final_baryonic_matter[:, :, slice_index], cmap='viridis', vmin=0)
axs[1].set_title(f'Matéria Bariônica Final (Fatia z={slice_index})')
fig.colorbar(im_bm, ax=axs[1], label='Densidade')

# 3. Células Colapsadas
# Usar um mapa de cores binário (e.g., 'gray') ou destacar pontos
im_collapsed = axs[2].imshow(final_collapsed[:, :, slice_index], cmap='binary', vmin=0, vmax=1)
axs[2].set_title(f'Células Colapsadas (Fatia z={slice_index})')
# Adicionar uma barra de cores simples para indicar 0 e 1
cbar_collapsed = fig.colorbar(im_collapsed, ax=axs[2], ticks=[0, 1])
cbar_collapsed.set_ticklabels(['Não Colapsada', 'Colapsada'])

plt.suptitle(f'Resultado da Simulação Simplificada após {timesteps} Passos (Limiar Colapso={collapse_threshold})', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta layout

# Salva a figura
plot_filename = '/home/ubuntu/simulation_results.png'
plt.savefig(plot_filename)
print(f"Gráfico dos resultados salvo em: {plot_filename}")

# Salva o código modificado
code_filename = '/home/ubuntu/simulation_code_modified.py'
# (O código já está sendo salvo neste arquivo pelo file_write)
print(f"Código modificado salvo em: {code_filename}")


