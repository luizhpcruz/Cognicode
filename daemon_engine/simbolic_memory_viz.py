import numpy as np
import matplotlib.pyplot as plt
import collections

def plot_symbolic_tape(memoria: list, title: str, n_estados: int = 4, show_silence: bool = True):
    """
    Plota uma fita de memória simbólica como uma imagem colorida.
    """
    if show_silence:
        colors = ["black", "crimson", "gold", "royalblue", "lightgray"] # 5 estados (4 + None)
        color_map = [colors[val] if val is not None else colors[-1] for val in memoria]
        
    else:
        colors = plt.cm.get_cmap("Set1", n_estados)
        color_map = [colors(val / (n_estados - 1)) for val in memoria]

    plt.figure(figsize=(12, 1.5))
    plt.bar(range(len(memoria)), [1]*len(memoria), color=color_map, edgecolor='white')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_entropy_cycles(historico_A, historico_B=None, title="Entropia Simbólica"):
    """
    Plota a evolução da entropia ao longo de ciclos.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(historico_A) + 1), historico_A, marker='o', label="Fita A")
    if historico_B:
        plt.plot(range(1, len(historico_B) + 1), historico_B, marker='s', label="Fita B")
    plt.title(title)
    plt.xlabel("Ciclo")
    plt.ylabel("Entropia (bits)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()