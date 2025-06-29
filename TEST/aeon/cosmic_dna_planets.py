import numpy as np
import random
import json
import hashlib

class CosmicDNA:
    def __init__(self, genes=None, fitness_score=0.0, generation=0):
        if genes:
            self.genes = genes
        else:
            self.genes = {
                'bio_diversity_focus': random.uniform(0.1, 0.9),
                'tech_innovation_focus': random.uniform(0.1, 0.9),
                'social_cohesion_focus': random.uniform(0.1, 0.9),
                'adaptability_gene': random.uniform(0.1, 0.9),
                'conflict_resolution_focus': random.uniform(0.1, 0.9),
                'collective_awareness_drive': random.uniform(0.1, 0.9)
            }
        self.fitness_score = fitness_score
        self.generation = generation
        self.fingerprint = self._calculate_fingerprint()

    def _calculate_fingerprint(self) -> str:
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.sha3_256(gene_str.encode()).hexdigest()[:16]

    def mutate(self) -> 'CosmicDNA':
        new_genes = self.genes.copy()
        for key in new_genes:
            if np.random.random() < 0.1:
                new_genes[key] += np.random.normal(0, 0.1)
                new_genes[key] = max(0.0, min(1.0, new_genes[key]))
        return CosmicDNA(new_genes, generation=self.generation + 1)

    def evolve(self, success_metric: float):
        # Evolução baseada em sucesso: mutação se performance for baixa (exemplo simples)
        if success_metric < 0.5:
            mutated = self.mutate()
            self.genes = mutated.genes
            self.fingerprint = mutated.fingerprint
            self.generation += 1
            self.fitness_score = 0.0

# Simplificado: não detalho PlanetaryResearchDomain para foco no essencial

class TerrestrialPlanet:
    def __init__(self, planet_id: int, universe_vibrational_frequency: float):
        self.planet_id = planet_id
        self.local_vibrational_frequency = universe_vibrational_frequency * (1 + (random.random() - 0.5) * 0.2)
        self.complexity_score = 0.0
        self.evolution_stage = "Abiogênese"
    
    def evolve_complexity(self, universe_phi_dynamics: float):
        growth_factor = 0.01 * (1 + universe_phi_dynamics * 0.01)
        self.complexity_score += growth_factor
        self.complexity_score = min(self.complexity_score, 1.0)
        self._update_evolution_stage()

    def _update_evolution_stage(self):
        cs = self.complexity_score
        if cs < 0.2:
            self.evolution_stage = "Abiogênese"
        elif cs < 0.4:
            self.evolution_stage = "Vida Simples"
        elif cs < 0.6:
            self.evolution_stage = "Vida Complexa"
        elif cs < 0.8:
            self.evolution_stage = "Vida Inteligente"
        else:
            self.evolution_stage = "Civilização Avançada"

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
