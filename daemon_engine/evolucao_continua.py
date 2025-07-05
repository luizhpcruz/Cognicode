import numpy as np
import matplotlib.pyplot as plt
import csv

# Parâmetros configuráveis
n_pop = 30  # número de populações paralelas
n_gen = 30  # número de gerações
n_elem = 40  # elementos por vetor
mut_rate_ini = 0.2  # taxa de mutação inicial
mut_rate_final = 0.05  # taxa de mutação final
seed = 42  # semente para reprodutibilidade
elitismo = True  # mantém o melhor indivíduo sempre
np.random.seed(seed)

# --- Exemplo: vetor inicial com propriedades químicas (hidrofobicidade de aminoácidos da insulina) ---
hidrofobicidade = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}
# Sequência da cadeia A da insulina humana
seq_insulina = 'GIVEQCCTSICSLYQLENYCN'
vetor_quimico = np.array([hidrofobicidade[a] for a in seq_insulina])
vetor_quimico = np.pad(vetor_quimico, (0, n_elem - len(vetor_quimico)), 'constant', constant_values=0)

# Funções do pipeline evolutivo

def update(estado, t):
    """Atualiza o vetor de estados usando seno e exponencial."""
    novo = np.zeros_like(estado)
    for i in range(len(estado)):
        left = estado[i-1] if i > 0 else 0
        right = estado[i+1] if i < len(estado)-1 else 0
        novo[i] = np.sin(left + right + t/5) * np.exp(-estado[i]) + 0.1 * np.random.randn()
    return np.clip(novo, 0, 1)

def mutar(estado, taxa):
    """Aplica mutação gaussiana em cada elemento com probabilidade taxa."""
    novo = estado.copy()
    for i in range(len(novo)):
        if np.random.rand() < taxa:
            novo[i] += np.random.normal(0, 0.2)
    return np.clip(novo, 0, 1)

def crossover_uniforme(pai, mae):
    """Crossover uniforme entre dois vetores."""
    mask = np.random.rand(len(pai)) < 0.5
    filho = np.where(mask, pai, mae)
    return filho

def fitness(estado):
    """Função de fitness: média dos estados."""
    return np.mean(estado)

def salvar_csv(nome, matriz):
    with open(nome, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matriz)

# Inicialização: populações baseadas em dados químicos + ruído
populacoes = [vetor_quimico + np.random.normal(0, 0.1, n_elem) for _ in range(n_pop)]
historico_fitness = []
historico_melhor = []
historico_std = []

for t in range(1, n_gen+1):
    # Taxa de mutação adaptativa
    mut_rate = mut_rate_ini + (mut_rate_final - mut_rate_ini) * (t-1)/(n_gen-1)
    # Atualiza cada população
    populacoes = [update(p, t) for p in populacoes]
    # Avalia fitness (média dos estados)
    fit = np.array([fitness(p) for p in populacoes])
    historico_fitness.append(fit.copy())
    historico_std.append(np.std(fit))
    # Seleciona os melhores
    idx = np.argsort(fit)[-n_pop//2:]
    melhores = [populacoes[i] for i in idx]
    # Elitismo: guarda o melhor indivíduo
    elite = melhores[-1].copy() if elitismo else None
    # Gera nova população por crossover uniforme e mutação adaptativa
    nova_pop = []
    while len(nova_pop) < n_pop:
        pais = np.random.choice(len(melhores), 2, replace=False)
        filho = crossover_uniforme(melhores[pais[0]], melhores[pais[1]])
        filho = mutar(filho, mut_rate)
        nova_pop.append(filho)
    # Substitui o pior pelo elite
    if elitismo:
        idx_pior = np.argmin([fitness(p) for p in nova_pop])
        nova_pop[idx_pior] = elite
    populacoes = nova_pop
    # Guarda o melhor padrão
    historico_melhor.append(elite if elitismo else melhores[-1].copy())

# Salva histórico de fitness e melhor padrão
salvar_csv('historico_fitness.csv', historico_fitness)
salvar_csv('melhor_padrao.csv', [historico_melhor[-1]])

# Visualização da evolução do fitness
historico_fitness = np.array(historico_fitness)
historico_std = np.array(historico_std)
plt.figure(figsize=(10,4))
plt.plot(np.max(historico_fitness, axis=1), label='Melhor')
plt.plot(np.mean(historico_fitness, axis=1), label='Média')
plt.fill_between(range(n_gen),
                 np.mean(historico_fitness, axis=1) - historico_std,
                 np.mean(historico_fitness, axis=1) + historico_std,
                 color='gray', alpha=0.2, label='Desvio padrão')
plt.xlabel('Geração')
plt.ylabel('Fitness (média dos estados)')
plt.title('Evolução do Fitness')
plt.legend()
plt.tight_layout()
plt.show()

# Visualização dos melhores padrões ao longo das gerações
plt.figure(figsize=(10,4))
plt.imshow(historico_melhor, aspect='auto', cmap='viridis')
plt.colorbar(label='Estado')
plt.xlabel('Elemento')
plt.ylabel('Geração')
plt.title('Melhores padrões por geração')
plt.tight_layout()
plt.show()
