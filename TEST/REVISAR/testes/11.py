import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize

def load_hubble_data(filepath):
    """
    Carrega os dados de H(z) de um arquivo CSV.
    Espera colunas: 'z', 'H_z', 'sigma_H'
    """
    return pd.read_csv(filepath)

def H_CPL(z, H0, Omega_m, w0, wa):
    """
    Modelo com energia escura dinâmica parametrizada por CPL.

    Parâmetros:
    - z: Redshift
    - H0: Parâmetro de Hubble hoje (km/s/Mpc)
    - Omega_m: Densidade de matéria
    - w0: Valor atual do parâmetro de equação de estado
    - wa: Variação do parâmetro de equação de estado

    Retorna:
    - H(z): Valor do parâmetro de Hubble no redshift z
    """
    Omega_DE = 1.0 - Omega_m

    def integrand(z_prime):
        return (1 + w0 + wa * z_prime / (1 + z_prime)) / (1 + z_prime)

    integral, _ = quad(integrand, 0, z)
    exponent = 3 * integral
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_DE * np.exp(exponent))

def chi_squared_CPL(params, z_data, H_data, sigma_H_data):
    H0, Omega_m, w0, wa = params
    H_model = np.array([H_CPL(z, H0, Omega_m, w0, wa) for z in z_data])
    chi2 = np.sum(((H_data - H_model) / sigma_H_data) ** 2)
    return chi2

def fit_model_CPL(z_data, H_data, sigma_H_data, initial_guesses, bounds):
    result = minimize(chi_squared_CPL, initial_guesses, args=(z_data, H_data, sigma_H_data), bounds=bounds)
    if result.success:
        optimal_params = result.x
        chi2 = result.fun
        return optimal_params, chi2
    else:
        return None, None

def plot_model_fit(z_data, H_data, sigma_H_data, H_model_pred, model_name="Modelo"):
    plt.errorbar(z_data, H_data, yerr=sigma_H_data, fmt='o', label='Dados Observacionais')
    plt.plot(z_data, H_model_pred, label=f'Ajuste {model_name}', color='red')
    plt.xlabel('Redshift (z)')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.title(f'Ajuste do {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(z_data, H_data, H_model_pred, sigma_H_data, model_name="Modelo"):
    residuals = H_data - H_model_pred
    plt.errorbar(z_data, residuals, yerr=sigma_H_data, fmt='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Resíduos')
    plt.title(f'Resíduos do {model_name}')
    plt.grid(True)
    plt.show()

def main():
    # Carregar os dados observacionais
    hubble_data = load_hubble_data("hubble_data.csv")
    z_data = hubble_data['z'].values
    H_data = hubble_data['H_z'].values
    sigma_H_data = hubble_data['sigma_H'].values

    # Ajuste do Modelo CPL
    print("\n--- Ajustando o Modelo CPL ---")
    initial_guesses_cpl = [70.0, 0.3, -1.0, 0.0]  # H0, Omega_m, w0, wa
    bounds_cpl = [(50.0, 100.0), (0.01, 1.0), (-2.0, 0.0), (-2.0, 2.0)]

    optimal_params_cpl, chi2_cpl = fit_model_CPL(z_data, H_data, sigma_H_data, initial_guesses_cpl, bounds_cpl)

    if optimal_params_cpl is not None:
        H0_cpl, Omega_m_cpl, w0_cpl, wa_cpl = optimal_params_cpl
        print(f"Parâmetros ótimos (CPL): H0={H0_cpl:.3f}, Omega_m={Omega_m_cpl:.3f}, w0={w0_cpl:.3f}, wa={wa_cpl:.3f}")
        print(f"Chi-quadrado mínimo (CPL): {chi2_cpl:.2f}")

        # Previsões do modelo CPL para plotagem
        H_model_pred_cpl = np.array([H_CPL(z, H0_cpl, Omega_m_cpl, w0_cpl, wa_cpl) for z in z_data])

        # Plotagem dos resultados
        plot_model_fit(z_data, H_data, sigma_H_data, H_model_pred_cpl, model_name="Modelo CPL")
        plot_residuals(z_data, H_data, H_model_pred_cpl, sigma_H_data, model_name="Modelo CPL")
    else:
        print("Falha ao ajustar o Modelo CPL.")

if __name__ == "__main__":
    main()
