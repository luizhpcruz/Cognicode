import numpy as np
from scipy.integrate import quad

class HIDEMCosmology:
    def __init__(self, H0=70, Omega_m=0.3, Omega_seed=0.05, m=0.5, a_c=0.5):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_seed = Omega_seed
        self.m = m  # Expoente de decaimento
        self.a_c = a_c  # Escala crítica
        self.Omega_k = 0.0  # Universo plano

    def Omega_halo(self, a):
        """Densidade de energia escura induzida por halos"""
        return self.Omega_seed * (a ** -self.m) * np.exp(-a / self.a_c)

    def H(self, z):
        a = 1 / (1 + z)
        Omega_halo_z = self.Omega_halo(a)
        Omega_tot = (self.Omega_m * (1 + z)**3 +
                     self.Omega_k * (1 + z)**2 +
                     Omega_halo_z)
        return self.H0 * np.sqrt(np.clip(Omega_tot, 1e-30, None))

    # Métodos para análise de BAO
    def comoving_distance(self, z):
        c = 299792.458  # km/s
        return c * quad(lambda z: 1 / self.H(z), 0, z)[0]
