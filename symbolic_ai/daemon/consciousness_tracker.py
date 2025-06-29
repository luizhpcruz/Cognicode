import numpy as np

def calculate_fitness(universe_history: dict) -> float:
    """
    Calcula a aptidão (fitness) de um universo com base na estabilidade de sua entropia.
    Um universo mais 'apto' tem um fluxo de informação estável.
    """
    # Exemplo simples: calcula a média da entropia simbólica ao longo do tempo.
    # Você pode refinar isso para medir a variação, picos, etc.
    entropy_history = universe_history.get('symbolic_entropy_history', [])
    if not entropy_history:
        return 0.0
    
    # A fitness é maior quanto mais estável for a entropia (baixa std)
    return 10.0 / (np.std(entropy_history) + 1e-6) # Evita divisão por zero