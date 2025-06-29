import numpy as np
import matplotlib.pyplot as plt

def H_squared(z, H0, Omega_m, Omega_r, Omega_lambda0, Omega_lambda1,
              Omega_osc0, omega, Omega_grav0, alpha,
              Omega_lens0, beta, Omega_s0, delta, Omega_s1, gamma,
              Omega_n0, kappa, omega_n, omega_g, intensity_factor):
    """
    Calcula H^2(z) incluindo um campo neutro oculto com progressão de intensidade:

    Omega_neutra(z) = Omega_n0 * exp(-kappa * z) * sin^2(omega_n * z) * cos^2(omega_g * z) * intensity_factor
    """
    
    term_m = Omega_m * (1 + z) ** 3      # Matéria
    term_r = Omega_r * (1 + z) ** 4      # Radiação
    term_lambda = Omega_lambda0 + Omega_lambda1 * np.exp(-gamma * z)
    term_osc = Omega_osc0 * np.cos(omega * z) ** 2
    term_grav = Omega_grav0 * np.exp(-alpha * z)
    term_lens = Omega_lens0 * np.log(1 + beta * z)
    term_string = Omega_s0 * np.exp(-delta * z) + Omega_s1 * np.cos(gamma * z) ** 2
    
    # Termo novo: energia neutra interagindo apenas com efeitos físicos e amplificada por intensity_factor
    term_neutra = Omega_n0 * np.exp(-kappa * z) * np.sin(omega_n * z) ** 2 * np.cos(omega_g * z) ** 2 * intensity_factor

    return H0**2 * (term_m + term_r + term_lambda + term_osc + term_grav + term_lens + term_string + term_neutra)

# Parâmetros da simulação
H0 = 70                     # Constante de Hubble em km/s/Mpc
Omega_m = 0.3               # Densidade de matéria
Omega_r = 9e-5              # Densidade de radiação
Omega_lambda0 = 0.7         # Energia escura constante
Omega_lambda1 = 0.0         # Energia escura variável (não usada aqui)
Omega_osc0 = 0.01           # Amplitude da oscilação
omega = 1.0                 # Frequência das oscilações
Omega_grav0 = 0.01          # Correção gravitacional extra
alpha = 0.5                 # Taxa de decaimento gravitacional
Omega_lens0 = 0.005         # Correção associada à lente gravitacional
beta = 0.2                  # Fator logarítmico para lentes
Omega_s0 = 0.01             # Componente exponencial da energia das cordas
delta = 0.3                 # Taxa de decaimento das cordas
Omega_s1 = 0.01             # Amplitude da oscilação das cordas
gamma = 1.0                 # Frequência usada na energia escura e nas cordas

# Parâmetros do novo campo neutro
Omega_n0 = 0.02             # Energia inicial do campo neutro
kappa = 0.1                 # Taxa de decaimento da energia neutra
omega_n = 2.0               # Frequência local da energia neutra
omega_g = 0.5               # Frequência global modulando todas as interações

# Testar diferentes intensidades da energia neutra
intensity_values = [0.5, 1.0, 2.0, 5.0]

# Criar um array de valores de redshift
z = np.linspace(0, 10, 300)

# Plot do resultado
plt.figure(figsize=(10, 6))

for intensity_factor in intensity_values:
    # Calcular H²(z) e H(z) para cada intensidade
    H2_z = H_squared(z, H0, Omega_m, Omega_r, Omega_lambda0, Omega_lambda1,
                      Omega_osc0, omega, Omega_grav0, alpha,
                      Omega_lens0, beta, Omega_s0, delta, Omega_s1, gamma,
                      Omega_n0, kappa, omega_n, omega_g, intensity_factor)

    # Aplicação do valor absoluto para evitar problemas com pequenos valores negativos numéricos.
    H_z = np.sqrt(np.abs(H2_z))
    
    # Plotando a curva para cada intensidade
    plt.plot(z, H_z, label=f'Intensidade = {intensity_factor}')

# Configuração do gráfico
plt.xlabel('Redshift (z)')
plt.ylabel('Constante de Hubble H(z) (km/s/Mpc)')
plt.title('Expansão Cósmica com Progressão de Energia Neutra')
plt.legend()
plt.grid(True)

# Salvar o gráfico como imagem
plt.savefig("grafico_H_z_progressao_neutra.png", dpi=300)

# Exibir o gráfico na tela
plt.show()

# Impressão de alguns valores no terminal para a maior intensidade testada
print("\nValores de H(z) para alguns pontos de z (Intensidade máxima):")
for i in range(0, len(z), 50):  # Pulando alguns pontos para não sobrecarregar
    print(f"z = {z[i]:.2f}, H(z) = {H_z[i]:.2f} km/s/Mpc")