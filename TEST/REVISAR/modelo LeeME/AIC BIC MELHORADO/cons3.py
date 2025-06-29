# Simulação em Python: DNA Cósmico com gatilhos evolutivos
import numpy as np

# Definir eventos cósmicos e intensidade de "mutação"
eventos = [
    ("Big Bang (início)", 0.05),
    ("Inflação Cósmica", 0.5),
    ("Recombinação", 0.1),
    ("Idade das Trevas", 0.02),
    ("Formação de Galáxias", 0.3),
    ("Reionização", 0.1),
    ("Expansão Acelerada", 0.05),
]
tamanho_genoma = 100  # número de "genes" no DNA Cósmico
# Genoma inicial aleatório (valores entre 0 e 1)
DNA = np.random.rand(tamanho_genoma)

# Função de mutação: adiciona ruído gaussiano aos "genes"
def mutacionar(dna, intensidade):
    ruido = np.random.normal(scale=intensidade, size=dna.shape)
    return dna + ruido

# Simular cada evento aplicando mutação ao DNA
historia = []
for nome, intensidade in eventos:
    DNA = mutacionar(DNA, intensidade)
    # Normalizar ou padronizar para manter valores entre 0 e 1 (opcional)
    DNA = (DNA - DNA.min()) / (DNA.max() - DNA.min())
    historia.append((nome, DNA.copy()))

# Exemplo: imprimir métricas após cada evento
print("Evolução do DNA Cósmico em eventos-chave:")
for nome, dna in historia:
    print(f"{nome}: variância = {np.var(dna):.4f}, média = {np.mean(dna):.4f}")
