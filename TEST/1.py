import numpy as np
import matplotlib.pyplot as plt

# Intervalo de tempo: de 0.1 bilhão (1e8) a 13.8 bilhões de anos (1.38e10)
t = np.linspace(1e8, 1.38e10, 500)  # tempo em anos

# Teste 1: Valores padrão
params_1 = {
    "A_IR": 1.0e-4, "tau_IR": 1.0e10, "T_IR": 5.0e9,
    "A_vis": 5.0e-5, "tau_vis": 1.2e10, "T_vis": 3.0e9
}

# Teste 2: Amplitude maior
params_2 = {
    "A_IR": 2.0e-4, "tau_IR": 1.0e10, "T_IR": 5.0e9,
    "A_vis": 1.0e-4, "tau_vis": 1.2e10, "T_vis": 3.0e9
}

# Teste 3: Tempo de decaimento menor (efeito mais curto)
params_3 = {
    "A_IR": 1.0e-4, "tau_IR": 5.0e9, "T_IR": 5.0e9,
    "A_vis": 5.0e-5, "tau_vis": 6.0e9, "T_vis": 3.0e9
}

# Lista de testes
test_cases = [params_1, params_2, params_3]
labels = ["Padrão", "Amplitude Maior", "Decaimento mais rápido"]

plt.figure(figsize=(12, 6))

# Loop pelos testes
for i, params in enumerate(test_cases):
    rho_IR = params["A_IR"] * np.exp(-t / params["tau_IR"]) * np.cos(t / params["T_IR"])
    rho_vis = params["A_vis"] * np.exp(-t / params["tau_vis"]) * np.sin(t / params["T_vis"])
    rho_light = rho_IR + rho_vis

    plt.plot(t / 1e9, rho_light, label=f"Total (Luz) - {labels[i]}", linewidth=2)

    # Exibir resultados no terminal para alguns pontos-chave
    print(f"\nTeste: {labels[i]}")
    print(f"{'Tempo (bilhões de anos)':<25}{'ρ_light(t)':<20}")
    print("-" * 50)
    for j in range(0, len(t), 100):  # Exibir a cada 100 pontos para reduzir a saída
        print(f"{t[j] / 1e9:<25.2f}{rho_light[j]:<20.5e}")

plt.xlabel("Tempo (bilhões de anos)")
plt.ylabel("Densidade (unidades da densidade crítica)")
plt.title("Comparação da Evolução da Energia da Luz para Diferentes Parâmetros")
plt.legend()
plt.grid(True)
plt.show()