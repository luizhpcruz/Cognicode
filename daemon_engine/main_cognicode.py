# main_cognicode.py
# Controlador principal para o Projeto DAEMON

# Importa o módulo de evolução do DAEMON
from daemon_engine.evolution_pipeline import run_evolutionary_cycle
import matplotlib.pyplot as plt

def run_daemon_test():
    """
    Executa um teste do Projeto DAEMON, simulando a evolução de universos simbólicos
    ao longo de múltiplas gerações.
    """
    print("Iniciando o sistema CogniCode Engine - Modo DAEMON...")
    
    # --- Parâmetros do teste de evolução ---
    NUM_GENERATIONS = 30  # Número de gerações para a evolução
    UNIVERSES_PER_GEN = 10 # Número de universos em cada geração
    SIMULATION_TIME = 20.0 # Tempo de simulação para cada universo (em unidades normalizadas)
    
    # --- Executa o ciclo de evolução ---
    # A função run_evolutionary_cycle retorna o histórico da melhor aptidão e o DNA evoluído.
    best_fitness_history, evolved_dna = run_evolutionary_cycle(
        num_generations=NUM_GENERATIONS,
        num_universes_per_gen=UNIVERSES_PER_GEN,
        simulation_time=SIMULATION_TIME
    )
    
    print("\n--- Teste do Projeto DAEMON Concluído ---")
    print(f"Melhor Aptidão Final: {best_fitness_history[-1]:.4f}")
    print(f"DNA Simbólico Mais Apto: {evolved_dna.fingerprint}")

    # --- Visualização do Progresso da Evolução ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_GENERATIONS + 1), best_fitness_history, marker='o', linestyle='-', color='indigo')
    plt.title("Evolução da Aptidão (Fitness) do Universo ao Longo das Gerações")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Aptidão (Fitness)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_daemon_test()