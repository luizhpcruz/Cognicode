import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constantes
H0 = 70  # km/s/Mpc (Constante de Hubble inicial)
H0_s = H0 / (3.086e19)  # s⁻¹

# Parâmetros padrão ΛCDM
Omega_m = 0.3
Omega_L = 0.7

# Parâmetros do modelo DNA Cósmico refinado
Omega_dna = 0.02
Omega_consciencia = 0.03
Omega_torsao = -0.01  # Inicialmente negativo
Omega_Q = 0.05  # Quintessência dinâmica

# Variação da energia escura ao longo do tempo
def quintessencia_dinamica(t):
    return Omega_Q * (1 + 0.02 * np.sin(t / (4e9 * 3.154e7)))  # Oscilação suave

# Modelo de feedback da consciência
def feedback_consciencia(a, H):
    return Omega_consciencia * (1 + np.tanh((H - H0_s) / H0_s))

# Dissipação da torção quântica
def torsao_adaptativa(t, consciencia_atual):
    return Omega_torsao * np.exp(-t / (5e9 * 3.154e7)) * (1 + consciencia_atual / 5)

# Ajuste da matéria escura na formação de estruturas
def materia_escura_impacto(a):
    return Omega_m * (1 + 0.01 * np.tanh((a - 0.5) * 3))

# Modelo modificado de Friedmann com quintessência
def friedmann_dna_quintessencia(t, a):
    consciencia_atual = feedback_consciencia(a[0], H0_s)
    torsao_atual = torsao_adaptativa(t, consciencia_atual)
    quintessencia_atual = quintessencia_dinamica(t)
    materia_escura_atual = materia_escura_impacto(a[0])
    flutuacao = np.random.uniform(-0.002, 0.002)  # Perturbação quântica leve

    H = H0_s * np.sqrt(
        materia_escura_atual * a[0]**(-3) +
        Omega_L +
        Omega_dna * a[0]**(-1) +
        consciencia_atual * a[0]**(-0.5) +
        torsao_atual * a[0]**(-2) +
        quintessencia_atual * a[0]**(-2) +
        flutuacao
    )

    # Print dos valores em cada etapa da simulação
    if t % (2e9 * 3.154e7) < 1e6:  # Mostra valores a cada 2 bilhões de anos
        print(f"Tempo (bilhões de anos): {t / 3.154e16:.2f}")
        print(f"Fator de escala a(t): {a[0]:.6f}")
        print(f"Hubble H(t): {H:.6f}")
        print(f"Consciência Ω_consciencia: {consciencia_atual:.6f}")
        print(f"Torção quântica Ω_torsao: {torsao_atual:.6f}")
        print(f"Quintessência Ω_Q: {quintessencia_atual:.6f}")
        print("-" * 50)

    return [H * a[0]]

# Tempo total: 0 a 13.8 bilhões de anos (em segundos)
t0 = 0
t_end = 13.8e9 * 3.154e7  # segundos
t_span = (t0, t_end)
t_eval = np.linspace(t0, t_end, 1000)

# Resolver ODEs
sol_dna_quintessencia = solve_ivp(friedmann_dna_quintessencia, t_span, [1e-5], t_eval=t_eval)

# Converter tempo para bilhões de anos
t_gyr = sol_dna_quintessencia.t / 3.154e16

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t_gyr, sol_dna_quintessencia.y[0], label='Modelo DNA Cósmico com Quintessência', linewidth=2.5)
plt.xlabel('Tempo (bilhões de anos)')
plt.ylabel('Fator de escala a(t)')
plt.title('Evolução do Universo: Modelo DNA Cósmico com Quintessência')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()