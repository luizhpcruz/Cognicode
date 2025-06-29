import numpy as np
import matplotlib.pyplot as plt

# Parâmetros iniciais
rho0 = 1.0            # Densidade inicial
t_max = 50            # Tempo total da simulação
dt = 0.1              # Passo de tempo
G = 0.5               # Constante gravitacional (reduzida para estabilidade)
Lambda = 0.05         # Constante da energia escura
P_quantica = 0.02     # Pressão quântica estabilizante
rho_max = 1e6         # Limite superior da densidade (limiar físico)

# Arrays de registro
tempos = np.arange(0, t_max, dt)
densidades = []

# Estado inicial
rho = rho0
for t in tempos:
    # Forças atuantes
    F_grav = G * rho**2
    F_repulsa = Lambda * rho
    F_quantic = P_quantica / (rho + 1e-6)

    # Derivada da densidade
    drho_dt = F_grav - F_repulsa - F_quantic

    # Atualização da densidade
    rho += drho_dt * dt
    rho = min(rho, rho_max)  # Aplica teto
    rho = max(rho, 0)        # Garante densidade não negativa

    densidades.append(rho)

# Gráfico da evolução da densidade
plt.figure(figsize=(10, 5))
plt.plot(tempos, densidades, label='Densidade do aglomerado')
plt.axhline(y=10, color='r', linestyle='--', label='Limiar de colapso (BH)')
plt.title("Evolução de um Proto-Buraco Negro com Forças Competitivas")
plt.xlabel("Tempo")
plt.ylabel("Densidade")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
