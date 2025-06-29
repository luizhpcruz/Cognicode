"""Regras de transição simbólica e colapsos."""
import numpy as np
import collections

# --- Funções de Memória ---
def inicializar_memoria(num_celulas, num_estados):
    """Inicializa uma fita de memória com estados discretos."""
    memoria = [None] * num_celulas
    for i in range(0, num_celulas, 2):
        memoria[i] = np.random.randint(num_estados)
    return memoria

def colapsar(memoria):
    """
    Aplica a regra de colapso simbólico (ex: 0, 1 -> None, None).
    """
    memoria_colapsada = []
    i = 0
    while i < len(memoria):
        a = memoria[i]
        b = memoria[i+1] if i + 1 < len(memoria) else None
        if a == 0 and b == 1:
            memoria_colapsada.extend([None, None])
            i += 2
        else:
            memoria_colapsada.append(a)
            i += 1
    return memoria_colapsada

def mutacao_estado(a, b, num_estados):
    """Aplica uma regra de transição com mutação probabilística."""
    if a is None or b is None:
        return None
    base = (2 * a + b) % num_estados
    mut = np.random.choice([0, 1], p=[0.9, 0.1])
    return (base + mut) % num_estados

def ciclo_simbolico(memoria, num_estados, direcao=1):
    """
    Executa um ciclo completo de evolução simbólica na fita de memória.
    """
    nova = list(memoria)
    indices = range(2, len(memoria)) if direcao == 1 else range(len(memoria) - 3, -1, -1)
    for i in indices:
        i1 = (i - direcao) % len(memoria)
        i2 = (i - 2 * direcao) % len(memoria)
        if memoria[i1] is not None and memoria[i2] is not None:
            nova[i] = mutacao_estado(memoria[i1], memoria[i2], num_estados)
    return colapsar(nova)

def p_acop_dyn(H_src, H_dst):
    """Calcula a probabilidade de acoplamento dinamicamente."""
    delta = H_src - H_dst
    return 1 / (1 + np.exp(-5 * delta))

def acoplar_silencios(mem_origem, mem_destino, prob, max_estado):
    """Acopla duas fitas através de seus estados de silêncio."""
    for idx, val in enumerate(mem_origem):
        if val is None and np.random.rand() < prob:
            if mem_destino[idx] is None:
                mem_destino[idx] = max_estado
    return mem_destino