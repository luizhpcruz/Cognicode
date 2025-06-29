from cosmic_resonance import CosmicResonance
from cosmic_dna_planets import CosmicDNA, PlanetarySystem
import numpy as np
import random

class Universe:
    def __init__(self, universe_id: int, initial_conditions: dict, vibrational_frequency: float, parent_dna: str = None):
        self.universe_id = universe_id
        self.matter_density = initial_conditions.get('matter_density', 0.3)
        self.radiation_density = initial_conditions.get('radiation_density', 0.001)

        self.cosmic_resonance = CosmicResonance(universe_id, vibrational_frequency=vibrational_frequency)
        self.cosmic_dna = CosmicDNA()  # Inicializa DNA aleatório

        # Se tem DNA pai, "herda" genes simplificadamente (exemplo)
        if parent_dna:
            # Para simplicidade, herdamos genes mas sem leitura do fingerprint real
            # Você pode implementar a herança real conforme precisar
            self.cosmic_dna.genes['bio_diversity_focus'] = 0.5
            self.cosmic_dna.genes['tech_innovation_focus'] = 0.5

        self.planetary_system = PlanetarySystem(num_planets=2, universe_vibrational_frequency=vibrational_frequency)

        self.hubble_constant = self._calculate_hubble_initial()
        self.time = 0.0

        self.hubble_history = [self.hubble_constant]
        self.phi_history = [self.cosmic_resonance.phi]
        self.rho_phi_history = [self.cosmic_resonance.get_energy_density_phi()]
        self.symbolic_entropy_history = [0.0]
        self.average_planet_complexity_history = [0.0]

    def _calculate_hubble_initial(self) -> float:
        rho_total = self.matter_density + self.radiation_density + self.cosmic_resonance.get_energy_density_phi()
        G = 6.674e-11
        return np.sqrt((8 * np.pi * G / 3) * rho_total)

    def _update_hubble_constant(self):
        G = 6.674e-11
        rho_phi = self.cosmic_resonance.get_energy_density_phi()
        rho_total = self.matter_density + self.radiation_density + rho_phi
        self.hubble_constant = np.sqrt((8 * np.pi * G / 3) * rho_total)

    def _calculate_symbolic_entropy_feedback(self) -> float:
        feedback_from_phi = np.abs(self.cosmic_resonance.phi) * 0.05
        feedback = feedback_from_phi + 0.005
        return min(feedback, 1.0)

    def simulate_quantum_entanglement(self, all_universes: list):
        # Simulação simples de entrelaçamento
        if self.average_planet_complexity_history[-1] > 0.8:
            others = [u for u in all_universes if u.universe_id != self.universe_id]
            if others:
                target = random.choice(others)
                target.cosmic_resonance.phi_dot += (self.cosmic_resonance.phi - target.cosmic_resonance.phi) * 0.0001

    def evolve_universe(self, time_step: float, all_universes: list = None):
        self.time += time_step

        if all_universes:
            self.simulate_quantum_entanglement(all_universes)

        symbolic_entropy_feedback = self._calculate_symbolic_entropy_feedback()
        self.symbolic_entropy_history.append(symbolic_entropy_feedback)

        self.cosmic_dna.evolve(symbolic_entropy_feedback)

        self.cosmic_resonance.update_potential_from_complexity(symbolic_entropy_feedback)
        self.cosmic_resonance.evolve_phi(self.hubble_constant)
        self._update_hubble_constant()

        self.planetary_system.update_planets_complexity(self.cosmic_resonance.get_energy_density_phi())
        self.average_planet_complexity_history.append(self.planetary_system.get_average_complexity())

        self.hubble_history.append(self.hubble_constant)
        self.phi_history.append(self.cosmic_resonance.phi)
        self.rho_phi_history.append(self.cosmic_resonance.get_energy_density_phi())
