import numpy as np

class FieldResonance:
    """
    Simula a dinâmica de um campo vetorial (phi) que responde a estímulos.
    """
    def __init__(self, universe_id: int, initial_phi=0.1, initial_phi_dot=0.01, vibrational_frequency=1.0):
        self.universe_id = universe_id
        self.phi = initial_phi
        self.phi_dot = initial_phi_dot
        self.vibrational_frequency = vibrational_frequency
        self.potential_V = self._calculate_potential(self.phi)
        self.dt = 0.01

    def _calculate_potential(self, phi: float) -> float:
        """Calcula o potencial de energia do campo."""
        return 0.5 * phi**2

    def update_potential_from_complexity(self, symbolic_entropy_feedback: float):
        """Ajusta o potencial do campo com base no feedback de entropia simbólica."""
        adjustment_factor = 1.0 + (symbolic_entropy_feedback * 0.5)
        self.potential_V = self._calculate_potential(self.phi) * adjustment_factor

    def evolve_phi(self, H: float):
        """Evolui o campo phi usando uma equação de movimento modificada."""
        dV_dphi = self.phi
        f_nu = self.vibrational_frequency * np.sin(self.phi * np.pi)
        phi_ddot = -3 * H * self.phi_dot - dV_dphi + f_nu
        self.phi_dot += phi_ddot * self.dt
        self.phi += self.phi_dot * self.dt
        self.potential_V = self._calculate_potential(self.phi)
        return self.phi, self.phi_dot, self.potential_V

    def get_energy_density_phi(self) -> float:
        """Calcula a densidade de energia do campo."""
        return 0.5 * self.phi_dot**2 + self.potential_V