import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import entropy

def fibonacci_sequence_cover(n):
    a, b = "L", "S"
    while len(b) < n:
        a, b = b, a + b
    return b[:n]

# --- NOVO: Função de evolução simbólica ---
def evolve_sequence(seq: str) -> str:
    """
    Regra simples: um gene vira 'L' se pelo menos um vizinho for 'L', senão vira 'S'.
    """
    n = len(seq)
    new_seq = ''
    for i in range(n):
        left = seq[i-1] if i > 0 else 'S'
        right = seq[i+1] if i < n-1 else 'S'
        if seq[i] == 'L' or left == 'L' or right == 'L':
            new_seq += 'L'
        else:
            new_seq += 'S'
    return new_seq

# --- Modificado: aceita lista de sequências para plotar evolução ---
def plot_symbolic_evolution(sequences, save_gif=False, gif_name='evolucao.gif'):
    """
    Plota a evolução de uma cadeia simbólica ao longo das gerações,
    sobrepondo mapa de entropia e taxa de variação de L. Pode salvar animação como GIF.
    """
    n_gen = len(sequences)
    n_nos = len(sequences[0])
    # Cálculo de entropia e taxa de variação de L
    entropias = []
    l_counts = []
    for seq in sequences:
        p = np.mean([1 if c == 'L' else 0 for c in seq])
        entropias.append(entropy([p, 1-p], base=2) if 0 < p < 1 else 0)
        l_counts.append(np.sum([1 if c == 'L' else 0 for c in seq]))
    l_deltas = np.diff(l_counts, prepend=l_counts[0])

    fig, axs = plt.subplots(2, 1, figsize=(n_nos/2, n_gen/2 + 3), gridspec_kw={'height_ratios': [3, 1]})
    ax = axs[0]
    for gen, seq in enumerate(sequences):
        vetores = np.array([1 if c == 'L' else 0 for c in seq])
        angles = np.random.uniform(0, 2 * np.pi, n_nos)
        u = vetores * np.cos(angles)
        v = vetores * np.sin(angles)
        x = np.arange(n_nos)
        y = np.full(n_nos, gen)
        ax.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy', color='crimson', alpha=0.7)
    ax.set_title('Evolução Simbólica Vetorial')
    ax.set_yticks(range(n_gen))
    ax.set_yticklabels([f'Gen {i}' for i in range(n_gen)])
    ax.set_xticks(np.arange(n_nos))
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    # Sobrepor entropia
    ax2 = ax.twinx()
    ax2.plot(entropias, range(n_gen), 'b--', label='Entropia', alpha=0.7)
    ax2.set_ylabel('Entropia', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.invert_yaxis()
    # Taxa de variação de L
    axs[1].plot(range(n_gen), l_deltas, 'g-o', label='ΔL')
    axs[1].set_title('Taxa de Variação de L por Geração')
    axs[1].set_xlabel('Geração')
    axs[1].set_ylabel('ΔL')
    axs[1].grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Animação temporal ---
    if save_gif:
        fig_anim, ax_anim = plt.subplots(figsize=(n_nos/2, 3))
        def update(gen):
            ax_anim.clear()
            seq = sequences[gen]
            vetores = np.array([1 if c == 'L' else 0 for c in seq])
            angles = np.random.uniform(0, 2 * np.pi, n_nos)
            u = vetores * np.cos(angles)
            v = vetores * np.sin(angles)
            x = np.arange(n_nos)
            y = np.zeros(n_nos)
            ax_anim.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy', color='crimson')
            ax_anim.set_title(f'Geração {gen}')
            ax_anim.set_xticks(x)
            ax_anim.set_yticks([])
            ax_anim.set_ylim(-1, 1)
        ani = animation.FuncAnimation(fig_anim, update, frames=n_gen, repeat=False)
        ani.save(gif_name, writer='pillow', fps=2)
        plt.close(fig_anim)
        print(f'GIF salvo como {gif_name}')

def plot_fibonacci_vectors(n_nos: int = 21, custom_seq: str = None):
    """
    Gera um gráfico de setas (quiver plot) para visualizar uma cadeia vetorial
    com ativação baseada na sequência de Fibonacci ou em uma sequência customizada.
    """
    # Função que gera uma sequência binária tipo Fibonacci (L=ativo, S=inativo)
    def fibonacci_sequence_cover(n):
        a, b = "L", "S"
        while len(b) < n:
            a, b = b, a + b
        return b[:n]

    if custom_seq:
        seq = custom_seq.upper()[:n_nos]
        if len(seq) < n_nos:
            seq += 'S' * (n_nos - len(seq))
        fib_seq = seq
    else:
        fib_seq = fibonacci_sequence_cover(n_nos)

    # Vetores: 1 se ativo ("L"), 0 se inativo ("S")
    vetores = np.array([1 if c == 'L' else 0 for c in fib_seq])

    # Direções aleatórias para visualização vetorial
    angles = np.random.uniform(0, 2 * np.pi, n_nos)
    u = vetores * np.cos(angles)
    v = vetores * np.sin(angles)

    # Posições espaciais dos nós
    x = np.arange(n_nos)
    y = np.zeros(n_nos)

    # Visualização com quiver
    plt.figure(figsize=(10, 2))
    plt.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy', color='crimson')
    plt.title('Cadeia Vetorial com Ativação Customizada')
    plt.yticks([])
    plt.xticks(x)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Experimento de IA Simbólica Evolutiva!")
    seq = input("Digite uma sequência de L/S (ou deixe vazio para Fibonacci): ").strip().upper()
    n_gen = input("Quantas gerações evoluir? (padrão=10): ").strip()
    n_gen = int(n_gen) if n_gen.isdigit() else 10
    save_gif = input("Salvar animação GIF? (s/n): ").strip().lower() == 's'
    if seq:
        base_seq = seq[:40]
        if len(base_seq) < 40:
            base_seq += 'S' * (40 - len(base_seq))
    else:
        base_seq = fibonacci_sequence_cover(40)
    # Evolução simbólica
    sequences = [base_seq]
    for _ in range(n_gen-1):
        next_seq = evolve_sequence(sequences[-1])
        sequences.append(next_seq)
    plot_symbolic_evolution(sequences, save_gif=save_gif)