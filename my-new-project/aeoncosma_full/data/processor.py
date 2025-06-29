
import numpy as np

def carregar_amostras(filepath, delimiter=' '):
    return np.loadtxt(filepath, delimiter=delimiter)

def estatisticas_basicas(dados):
    stats = []
    for i in range(dados.shape[1]):
        coluna = dados[:, i]
        stats.append({
            'media': np.mean(coluna),
            'mediana': np.median(coluna),
            'desvio': np.std(coluna),
            'min': np.min(coluna),
            'max': np.max(coluna)
        })
    return stats
