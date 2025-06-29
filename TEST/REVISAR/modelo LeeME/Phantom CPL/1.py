import numpy as np
from scipy.integrate import quad

# Parâmetros do modelo Phantom-CPL
H0 = 70.51              # km/s/Mpc
Om = 0.291
w0 = -1.08

# Conversão de H0 para s⁻¹ (1 Mpc = 3.085677581e22 metros)
H0_si = 70.51 * 1e3 / 3.0856775814913673e22  # H₀ [s⁻¹]

# Função Hubble
def H(a):
    return H0_si * np.sqrt(Om / a**3 + (1 - Om) / a**(3 * (1 + w0)))

def integrand(a):
    return 1 / (a * H(a))

# Limite prático para "infinito" (Big Rip)
a_max = 1e5  # fator de escala muito maior que hoje

# Integração de a = 1 (hoje) até a_max
t_rip_sec, _ = quad(integrand, 1, a_max, limit=200)

# Correção: segundos por Giga-ano (1 Gyr = 3.1536e16 segundos)
seconds_per_gyr = 3.1536e16
t_rip_gy = t_rip_sec / seconds_per_gyr

print(f"Tempo até o Big Rip ≈ {t_rip_gy:.2f} bilhões de anos")