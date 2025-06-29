
import matplotlib.pyplot as plt

def plotar_histogramas(dados, labels=None):
    for i in range(dados.shape[1]):
        coluna = dados[:, i]
        label = f"Coluna {i+1}" if labels is None else labels[i]
        plt.figure(figsize=(6,3))
        plt.hist(coluna, bins=30, color='steelblue', edgecolor='black')
        plt.title(f'Distribuição: {label}')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
