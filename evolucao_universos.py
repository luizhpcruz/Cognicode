import csv
import os
import time
import argparse
from cosmic_dna_planets import CosmicDNA
from optimizer import RLParameterOptimizer
from cosmic_resonance import CosmicResonance

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description='Experimento Evolutivo de Universos')
parser.add_argument('--pop', type=int, default=20, help='Tamanho da população')
parser.add_argument('--gen', type=int, default=10, help='Número de gerações')
parser.add_argument('--trials', type=int, default=50, help='Tentativas por DNA')
args = parser.parse_args()

POP_SIZE = args.pop
GENERATIONS = args.gen
TRIALS = args.trials

os.makedirs('logs', exist_ok=True)
log_rows = []
best_dnas = []

population = [CosmicDNA() for _ in range(POP_SIZE)]
parent_map = {dna.fingerprint: None for dna in population}

def log_stats(scores, g):
    rewards = [s[0] for s in scores]
    print(f"Geração {g+1}: Média={sum(rewards)/len(rewards):.4f} | Máx={max(rewards):.4f} | Mín={min(rewards):.4f}")

for g in range(GENERATIONS):
    print(f"\n🌌 Geração {g+1}")
    start = time.time()
    scores = []
    for dna in population:
        try:
            optimizer = RLParameterOptimizer(dna)
            best_params = optimizer.find_optimal_params(num_trials=TRIALS)
            reward = optimizer._run_headless_simulation(best_params)
            scores.append((reward, dna, best_params))
            log_rows.append({
                'geracao': g+1,
                'dna_fingerprint': dna.fingerprint,
                'genes': str(dna.genes),
                'params': str(best_params),
                'reward': reward,
                'parent': parent_map.get(dna.fingerprint)
            })
        except Exception as e:
            print(f"[ERRO] DNA {dna.fingerprint}: {e}")
    # Seleciona os melhores DNAs
    scores.sort(reverse=True, key=lambda x: x[0])
    top_dnas = [dna for _, dna, _ in scores[:POP_SIZE // 2]]
    # Salva o melhor DNA da geração
    best_dnas.append({'geracao': g+1, 'dna_fingerprint': scores[0][1].fingerprint, 'genes': str(scores[0][1].genes), 'params': str(scores[0][2]), 'reward': scores[0][0]})
    # Gera nova população com mutações
    new_population = top_dnas.copy()
    for dna in top_dnas:
        mutated = dna.mutate()
        parent_map[mutated.fingerprint] = dna.fingerprint
        new_population.append(mutated)
    population = new_population
    log_stats(scores, g)
    print(f"Tempo da geração: {time.time()-start:.2f}s")

print("\n🌠 Evolução finalizada.")

# Salva o log em CSV
with open('logs/log_evolucao_universos.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['geracao', 'dna_fingerprint', 'genes', 'params', 'reward', 'parent']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in log_rows:
        writer.writerow(row)
print("Log completo salvo em logs/log_evolucao_universos.csv")

# Salva os melhores DNAs de cada geração
with open('logs/melhores_dnas.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['geracao', 'dna_fingerprint', 'genes', 'params', 'reward']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in best_dnas:
        writer.writerow(row)
print("Melhores DNAs salvos em logs/melhores_dnas.csv")
