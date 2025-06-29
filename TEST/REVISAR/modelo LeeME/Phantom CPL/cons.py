import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constantes
H0 = 70  # km/s/Mpc
H0_s = H0 / (3.086e19)  # s⁻¹

# Parâmetros padrão ΛCDM
Omega_m = 0.3
Omega_L = 0.7

# Parâmetros do modelo com DNA Cósmico
Omega_dna = 0.02
Omega_consciencia = 0.03
Omega_torsao = -0.01  # Inicialmente negativo, mas será amortecido

# Função de amortecimento para Ω_torsao
def amortecer_torsao(t):
    return Omega_torsao * np.exp(-t / (5e9 * 3.154e7))  # Dissipação gradual ao longo do tempo

# Função para reduzir Ω_dna ao longo do tempo
def reduzir_dna(t):
    return Omega_dna * np.exp(-t / (10e9 * 3.154e7))  # Saturação do conhecimento acumulado

# Função para aumentar Ω_consciencia se o universo acelerar demais
def ajustar_consciencia(a, H):
    return Omega_consciencia * (1 + np.tanh((H - H0_s) / H0_s))  # Ajuste conforme taxa de expansão

# Modelo com componentes aprimorados
def friedmann_dna_modificado(t, a):
    torsao_atual = amortecer_torsao(t)
    dna_atual = reduzir_dna(t)
    consciencia_atual = ajustar_consciencia(a[0], H0_s * np.sqrt(Omega_m * a[0]**(-3) + Omega_L))
    flutuacao = np.random.uniform(-0.002, 0.002)  # Pequena variação caótica
    
    H = H0_s * np.sqrt(
        Omega_m * a[0]**(-3) +
        Omega_L +
        dna_atual * a[0]**(-1) +
        consciencia_atual * a[0]**(-0.5) +  # Interação dinâmica da consciência
        torsao_atual * a[0]**(-2) +
        flutuacao  # Perturbação quântica leve
    )
    return [H * a[0]]

# Tempo total: 0 a 13.8 bilhões de anos (em segundos)
t0 = 0
t_end = 13.8e9 * 3.154e7  # segundos
t_span = (t0, t_end)
t_eval = np.linspace(t0, t_end, 1000)

# Resolver ODEs
sol_dna_modificado = solve_ivp(friedmann_dna_modificado, t_span, [1e-5], t_eval=t_eval)

# Converter tempo para bilhões de anos
t_gyr = sol_dna_modificado.t / 3.154e16

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t_gyr, sol_dna_modificado.y[0], label='Modelo DNA Cósmico Modificado', linewidth=2.5)
plt.xlabel('Tempo (bilhões de anos)')
plt.ylabel('Fator de escala a(t)')
plt.title('Evolução do Universo: Modelo DNA Cósmico com Novos Parâmetros')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()