from .field_resonance import FieldResonance
from .symbolic_dna_module import SymbolicDNA
import numpy as np

class Universe:
    """
    Representa um universo simulado, com sua própria física de campo, DNA e sistemas internos.
    """
    def __init__(self, universe_id: int, initial_conditions: dict, vibrational_frequency: float, parent_dna: str = None):
        self.universe_id = universe_id
        self.matter_density = initial_conditions.get('matter_density', 0.3)
        self.radiation_density = initial_conditions.get('radiation_density', 0.001)
        self.field_resonance = FieldResonance(universe_id, vibrational_frequency=vibrational_frequency)
        self.cosmic_dna = SymbolicDNA()
        # Herda genes do DNA pai
        if parent_dna:
            self.cosmic_dna.genes['collective_awareness_drive'] = 0.9 # Exemplo de herança
        self.hubble_constant = self._calculate_hubble_initial()
        self.time = 0.0
        self.hubble_history = [self.hubble_constant]
        self.phi_history = [self.field_resonance.phi]
        self.symbolic_entropy_history = [0.0] # Histórico de entropia simbólica

    def _calculate_hubble_initial(self) -> float:
        G = 6.674e-11
        rho_phi = self.field_resonance.get_energy_density_phi()
        rho_total = self.matter_density + self.radiation_density + rho_phi
        return np.sqrt((8 * np.pi * G / 3) * rho_total)

    def _update_hubble_constant(self):
        G = 6.674e-11
        rho_phi = self.field_resonance.get_energy_density_phi()
        rho_total = self.matter_density + self.radiation_density + rho_phi
        self.hubble_constant = np.sqrt((8 * np.pi * G / 3) * rho_total)

    def evolve_universe(self, time_step, all_universes):
        """Exemplo de evolução: atualiza phi, Hubble, entropia simbólica e evolui o DNA."""
        H = self.hubble_constant
        phi, phi_dot, potential = self.field_resonance.evolve_phi(H)
        self.phi_history.append(phi)
        self._update_hubble_constant()
        self.hubble_history.append(self.hubble_constant)
        import random
        entropy = self.symbolic_entropy_history[-1] + random.uniform(-0.05, 0.05)
        self.symbolic_entropy_history.append(entropy)
        self.cosmic_dna.evolve(entropy)
        self.time += time_step