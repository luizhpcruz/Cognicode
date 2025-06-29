import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

# Constante da velocidade da luz em km/s
c_km_s = 299792.458

class CosmologyModel:
    """
    Classe base para modelos cosmológicos.
    Fornece métodos para H(z) e dL(z).
    """
    def __init__(self, H0, Omega_m, Omega_r=0.0, Omega_k=0.0):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r # Densidade de radiação (geralmente negligenciável em z baixos)
        self.Omega_k = Omega_k # Densidade de curvatura
        self.Omega_de0 = 1.0 - self.Omega_m - self.Omega_r - self.Omega_k # Assegura um universo plano por padrão se Omega_k=0

    def E_z(self, z):
        """
        Calcula E(z) = H(z)/H0. Deve ser implementado por classes filhas.
        """
        raise NotImplementedError("E_z method must be implemented by subclasses.")

    def H(self, z):
        """
        Calcula H(z) em km/s/Mpc.
        """
        return self.H0 * self.E_z(z)

    def dL(self, z):
        """
        Calcula a distância de luminosidade d_L(z) em Mpc.
        Assume que H0 está em km/s/Mpc e c está em km/s.
        """
        # Distância de comoving em Mpc
        integrand = lambda z_prime: 1.0 / self.E_z(z_prime)
        try:
            # Integração robusta, especialmente para z=0
            integral, _ = quad(integrand, 0, z, epsabs=1.49e-08, epsrel=1.49e-08)
        except Exception as e:
            # Em caso de erro na integração (ex: parâmetros fora do limite), retornar um valor grande para likelihood ser baixa
            print(f"Erro na integração para dL(z) em z={z}: {e}. Retornando inf.")
            return np.inf

        # Constante de Hubble em s^-1 para consistência de unidades (1 Mpc = 3.086e19 km)
        H0_s_inv = self.H0 / (c_km_s * 1e3 / 3.086e19) # (km/s/Mpc) / (km/Mpc) = s^-1
        # d_C em Mpc = (c / H0) * integral (onde H0 está em km/s/Mpc e c em km/s, precisa de um ajuste de unidade)
        # c_km_s é km/s, H0 é km/s/Mpc. Então c_km_s/H0 dá Mpc.
        d_c_Mpc = (c_km_s / self.H0) * integral

        # Se houver curvatura, a distância de comoving é mais complexa
        if self.Omega_k == 0:
            return (1 + z) * d_c_Mpc
        elif self.Omega_k > 0: # Universo fechado
            sqrt_Omega_k = np.sqrt(self.Omega_k)
            return (1 + z) * (c_km_s / self.H0 / sqrt_Omega_k) * np.sin(sqrt_Omega_k * integral)
        else: # Universo aberto
            sqrt_abs_Omega_k = np.sqrt(abs(self.Omega_k))
            return (1 + z) * (c_km_s / self.H0 / sqrt_abs_Omega_k) * np.sinh(sqrt_abs_Omega_k * integral)


class LCDMCosmology(CosmologyModel):
    """
    Implementa o modelo Lambda-CDM plano.
    Parâmetros: H0, Omega_m.
    Omega_Lambda é inferido como 1 - Omega_m (para universo plano).
    """
    def __init__(self, H0, Omega_m):
        super().__init__(H0, Omega_m, Omega_k=0.0) # Assumindo plano por padrão
        self.Omega_Lambda = self.Omega_de0 # No LCDM plano, Omega_de0 é Omega_Lambda

    def E_z(self, z):
        """
        Calcula E(z) = H(z)/H0 para Lambda-CDM plano.
        """
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)

class HIDEMCosmology(CosmologyModel):
    """
    Implementa o modelo HIDEM (seu modelo modificado).
    Parâmetros: H0, Omega_m, Omega_seed, m, a_c.
    Assume Omega_Lambda = 0 e que o restante do universo é preenchido por Omega_seed.
    Ou, mais realisticamente, se Omega_seed é uma *adição* à energia escura,
    podemos reescrever E(z) para incluir tanto Omega_Lambda quanto Omega_seed.

    Com base nos seus PDFs, o modelo é:
    H(z)^2 = H0^2 * [Omega_m(1+z)^3 + Omega_Lambda + Omega_ond,0 * (1+z)^n]
    Onde Omega_ond,0 = Omega_seed e n = m ou um parâmetro diferente.

    Seu código Python original usava:
    Omega_de_z = Omega_de0 * exp(-3 * integral (1+w_eff)/(1+z))
    onde w_eff(z) = f_EDE * tanh_term + (1 - f_EDE) * cpl
    cpl = w0 + wa * z / (1+z)
    tanh_term = 0.5 * (1 - tanh(5 * log((1+z)/z_c)))

    Seu último uso de "HIDEM" referia-se a "Omega_seed * (1+z)**self.m * np.exp(-(1+z)/self.a_c)".
    Vamos adotar esta última forma que você usou no corner plot.
    Assumimos que é uma componente adicional e que o universo é plano com Omega_Lambda como 1 - Omega_m - Omega_seed_at_z0

    Vamos usar a forma mais literal que você indicou para o modelo "HIDEM" no código original
    do corner plot, onde 'Omega_seed', 'm', e 'a_c' são parâmetros da componente adicional.
    E para um universo plano, $Omega_Lambda = 1 - Omega_m - Omega_halo(z=0)$.
    """
    def __init__(self, H0, Omega_m, Omega_seed, m, a_c):
        super().__init__(H0, Omega_m, Omega_k=0.0)
        self.Omega_seed = Omega_seed
        self.m = m
        self.a_c = a_c

        # Calcular Omega_halo no z=0 para garantir que Omega_de0 seja consistente com a condição de planura
        self.Omega_halo_0 = self.Omega_halo_z(0) # Componente da energia escura do HIDEM em z=0
        self.Omega_Lambda_effective = 1.0 - self.Omega_m - self.Omega_halo_0 # Assumindo que Omega_seed é a 'ond' e o restante é Lambda

    def Omega_halo_z(self, z):
        """
        Calcula a densidade da componente 'halo' (HIDEM) em função de z.
        Corresponde à sua Omega_seed * (1+z)**m * exp(-(1+z)/a_c)
        """
        # Evitar problemas com (1+z) próximo de zero para m negativo ou expoente muito grande
        z_safe = np.maximum(z, 1e-8)
        try:
            return self.Omega_seed * (1 + z_safe)**self.m * np.exp(-(1 + z_safe) / self.a_c)
        except OverflowError:
            return np.inf # Retornar infinito se o cálculo for muito grande/pequeno para evitar problemas
        except RuntimeWarning: # Para np.exp de número muito grande
             return np.inf


    def E_z(self, z):
        """
        Calcula E(z) = H(z)/H0 para o modelo HIDEM.
        H(z)^2/H0^2 = Omega_m*(1+z)^3 + Omega_Lambda_effective + Omega_halo_z(z)
        """
        # Evitar valores negativos sob a raiz quadrada
        term_matter = self.Omega_m * (1 + z)**3
        term_lambda = self.Omega_Lambda_effective
        term_halo = self.Omega_halo_z(z)

        # Garantir que a soma é não-negativa para evitar sqrt de negativos
        total_energy_density = term_matter + term_lambda + term_halo

        if total_energy_density < 0:
            return np.inf # Penaliza se os parâmetros levarem a densidade negativa

        return np.sqrt(total_energy_density)