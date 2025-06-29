"""Memória vetorial discreta"""
import numpy as np

class VectorMemory:
    """
    Representa uma fita de memória simbólica, com estados discretos e um histórico de evolução.
    """
    def __init__(self, size: int):
        """
        Inicializa a memória com estados aleatórios.
        Args:
            size: O tamanho da fita de memória.
        """
        self.size = size
        self.state = np.random.choice([0, 1], size=size)
        self.history = [self.state.copy()]

    def step(self, rule_func):
        """
        Aplica uma função de regra de transição para evoluir a memória em um passo.
        Args:
            rule_func: Uma função que define a regra de transição simbólica.
        """
        new_state = rule_func(self.state)
        self.state = new_state
        self.history.append(new_state.copy())
        return self.state