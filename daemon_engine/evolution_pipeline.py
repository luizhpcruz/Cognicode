from .ensemble_generator import generate_universe_ensemble
from .consciousness_tracker import calculate_fitness
from .simulation_universe import Universe # Importa a classe Universe
import numpy as np

def run_evolutionary_cycle(num_generations: int, num_universes_per_gen: int, simulation_time: float):
    """
    Roda o ciclo de evolução do Projeto DAEMON.
    Gera -> Simula -> Avalia -> Seleciona -> Mutaciona.
    """
    # Frequências vibracionais para a população inicial
    vibrational_frequencies = np.linspace(0.5, 2.0, num_universes_per_gen)
    
    # Geração inicial
    universes = generate_universe_ensemble(num_universes_per_gen, {'matter_density': 0.3}, vibrational_frequencies)
    
    best_fitness_per_gen = []

    print("Iniciando Evolução Simbólica do Projeto DAEMON...")

    for gen in range(num_generations):
        # 1. Simulação
        for u in universes:
            # Garante que a simulação evolui por um certo número de passos
            dt = simulation_time / 100
            for _ in range(100): # Roda 100 passos de evolução por geração
                u.evolve_universe(time_step=dt, all_universes=universes)
        
        # 2. Avaliação
        fitness_scores = [calculate_fitness(u.__dict__) for u in universes]
        best_fitness = max(fitness_scores)
        best_fitness_index = np.argmax(fitness_scores)
        best_universe = universes[best_fitness_index]
        
        best_fitness_per_gen.append(best_fitness)

        print(f"Geração {gen+1}/{num_generations} - Melhor Aptidão: {best_fitness:.4f}")
        
        # 3. Seleção e Mutação (Para a próxima geração)
        # Seleciona o melhor universo e o mutaciona para a próxima geração
        parent_dna_for_next_gen = best_universe.cosmic_dna.fingerprint
        universes = generate_universe_ensemble(num_universes_per_gen, {'matter_density': 0.3}, vibrational_frequencies, parent_dna=parent_dna_for_next_gen)
        
    print("Evolução finalizada. Gerando relatório de aptidão.")
    return best_fitness_per_gen, best_universe.cosmic_dna