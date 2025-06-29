
from aeoncosma.data.processor import carregar_amostras, estatisticas_basicas
from aeoncosma.data.visualizer import plotar_histogramas

def main():
    filepath = 'output/amostras_mcmc.txt'
    amostras = carregar_amostras(filepath)
    stats = estatisticas_basicas(amostras)

    for i, s in enumerate(stats):
        print(f"[Parâmetro {i+1}] Média: {s['media']:.3f} | ±{s['desvio']:.3f}")

    plotar_histogramas(amostras)

if __name__ == '__main__':
    main()
