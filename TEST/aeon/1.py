# cosmic_dna_planets.py

import numpy as np
import random

class CosmicDNA:
    def __init__(self, parent_dna: str = None):
        """
        Representa o 'DNA Cósmico' de um universo, uma string simbólica
        que codifica características evolutivas.
        """
        if parent_dna:
            self.sequence = self._mutate_dna(parent_dna)
        else:
            # DNA inicial aleatório para o primeiro universo ou universos sem pai
            self.sequence = ''.join(random.choice('ACGT') for _ in range(10)) # Exemplo: 10 bases

    def _mutate_dna(self, dna: str, mutation_rate: float = 0.1) -> str:
        """Simula mutações aleatórias no DNA."""
        mutated_seq = list(dna)
        for i in range(len(mutated_seq)):
            if random.random() < mutation_rate:
                mutated_seq[i] = random.choice('ACGT')
        return "".join(mutated_seq)

    def evolve(self, success_metric: float):
        """
        Permite que o DNA evolua com base em uma métrica de 'sucesso' do universo.
        (Ex: picos de consciência, estabilidade, etc.)
        """
        if success_metric > 0.5: # Exemplo: se o universo está "bem sucedido", a mutação é menor
            self.sequence = self._mutate_dna(self.sequence, mutation_rate=0.01)
        else: # Caso contrário, mais chance de mutação para explorar novas "soluções"
            self.sequence = self._mutate_dna(self.sequence, mutation_rate=0.05)


class TerrestrialPlanet:
    def __init__(self, planet_id: int, universe_vibrational_frequency: float):
        """
        Representa um planeta tipo Terra com atributos vibracionais e estágios de evolução.
        """
        self.planet_id = planet_id
        self.local_vibrational_frequency = universe_vibrational_frequency * (1 + (random.random() - 0.5) * 0.2) # +/- 10%
        self.habitability_potential = random.random() # 0 a 1: potencial para vida
        self.complexity_score = 0.0 # Pontuação de complexidade/vida emergente (0 a 1)
        self.consciousness_level = 0.0 # Nível de consciência (0 a 1)
        self.evolution_stage = "Abiogênese" # Estágio atual: Abiogênese -> Simples -> Complexa -> Inteligente -> Avançada

    def evolve_complexity(self, universe_phi_dynamics: float):
        """
        Evolui a complexidade do planeta, influenciada pela dinâmica do campo escalar do universo,
        sua própria frequência vibracional e potencial de habitabilidade.
        """
        growth_factor = (universe_phi_dynamics * self.habitability_potential *
                         (self.local_vibrational_frequency / 1.0)) * 0.005 # Fator de crescimento

        self.complexity_score += growth_factor
        self.complexity_score = min(self.complexity_score, 1.0) # Limita a complexidade máxima

        self._update_evolution_stage()
        self._update_consciousness_level()

    def _update_evolution_stage(self):
        """Atualiza o estágio de evolução do planeta com base na complexidade."""
        if self.complexity_score < 0.2:
            self.evolution_stage = "Abiogênese"
        elif 0.2 <= self.complexity_score < 0.4:
            self.evolution_stage = "Vida Simples"
        elif 0.4 <= self.complexity_score < 0.6:
            self.evolution_stage = "Vida Complexa"
        elif 0.6 <= self.complexity_score < 0.8:
            self.evolution_stage = "Vida Inteligente"
        else:
            self.evolution_stage = "Civilização Avançada"

    def _update_consciousness_level(self):
        """Calcula o nível de consciência com base na complexidade, com um limiar."""
        # A consciência só "desperta" após um certo nível de complexidade
        if self.complexity_score >= 0.5: # Limiar para a emergência da consciência significativa
            self.consciousness_level = (self.complexity_score - 0.5) * 2.0 # Escala de 0 a 1 a partir de 0.5
            self.consciousness_level = min(self.consciousness_level, 1.0) # Garante que não excede 1
        else:
            self.consciousness_level = 0.0


class PlanetarySystem:
    def __init__(self, num_planets: int, universe_vibrational_frequency: float):
        self.planets = [TerrestrialPlanet(i, universe_vibrational_frequency) for i in range(num_planets)]

    def update_planets_complexity(self, universe_phi_dynamics: float):
        for planet in self.planets:
            planet.evolve_complexity(universe_phi_dynamics)

    def get_average_complexity(self) -> float:
        if not self.planets:
            return 0.0
        return np.mean([p.complexity_score for p in self.planets])

    def get_average_consciousness_level(self) -> float:
        if not self.planets:
            return 0.0
        return np.mean([p.consciousness_level for p in self.planets])
