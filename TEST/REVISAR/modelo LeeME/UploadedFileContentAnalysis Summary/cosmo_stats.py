import numpy as np

# Dados observacionais
z = np.array([0.24, 0.35, 0.57, 0.73, 1.00])
H_obs = np.array([79.69, 84.40, 92.40, 97.30, 103.00])
sigma_H = np.array([2.32, 1.90, 1.50, 1.20, 1.10])

# Parâmetros cosmológicos padrão (Pantheon+/SH0ES, BAO, CMB)
H0_Pantheon = 73.4  # km/s/Mpc
H0_BAO = 67.64
H0_CMB = 67.4
Omega_m_Pantheon = 0.338
Omega_m_CMB = 0.315

# Parâmetros estimados via MCMC (seu ajuste)
H0_LCDM = 79.01
Omega_m_LCDM = 0.107

# Função H(z) para ΛCDM
def H_LCDM(z, H0, Omega_m):
    Omega_Lambda = 1 - Omega_m
    # Adicionar uma pequena constante para evitar divisão por zero ou log de zero se Omega_m for muito próximo de 1 ou 0
    # Embora neste caso Omega_Lambda = 1 - Omega_m, em outros modelos pode ser diferente.
    # A verificação np.sqrt garante que o argumento não seja negativo.
    term = Omega_m * (1 + z)**3 + Omega_Lambda
    if np.any(term < 0):
        # Retornar um valor grande ou NaN para indicar parâmetros inválidos
        # Isso pode acontecer se Omega_m > 1
        return np.full_like(z, np.nan)
    return H0 * np.sqrt(term)

# Calcular H(z) modelo ΛCDM e modelos padrão
H_model_LCDM = H_LCDM(z, H0_LCDM, Omega_m_LCDM)
H_model_Pantheon = H_LCDM(z, H0_Pantheon, Omega_m_Pantheon)
H_model_CMB = H_LCDM(z, H0_CMB, Omega_m_CMB)

# Calcular χ² para cada modelo
def chi2(H_obs, H_model, sigma_H):
    # Verificar se H_model contém NaN (parâmetros inválidos)
    if np.any(np.isnan(H_model)):
        return np.inf # Retorna infinito se o modelo for inválido
    return np.sum(((H_obs - H_model) / sigma_H) ** 2)

chi2_LCDM = chi2(H_obs, H_model_LCDM, sigma_H)
chi2_Pantheon = chi2(H_obs, H_model_Pantheon, sigma_H)
chi2_CMB = chi2(H_obs, H_model_CMB, sigma_H)

# Número de parâmetros
k_LCDM = 2
k_CPL = 4 # Embora não usado aqui, definido no contexto anterior
n = len(z)

# Critérios AIC e BIC
def aic(chi2_val, k):
    if not np.isfinite(chi2_val):
        return np.inf
    return chi2_val + 2 * k

def bic(chi2_val, k, n):
    if not np.isfinite(chi2_val):
        return np.inf
    return chi2_val + k * np.log(n)

AIC_LCDM = aic(chi2_LCDM, k_LCDM)
BIC_LCDM = bic(chi2_LCDM, k_LCDM, n)
AIC_Pantheon = aic(chi2_Pantheon, k_LCDM)
BIC_Pantheon = bic(chi2_Pantheon, k_LCDM, n)
AIC_CMB = aic(chi2_CMB, k_LCDM)
BIC_CMB = bic(chi2_CMB, k_LCDM, n)

# Comparação de AIC e BIC
delta_AIC_Pantheon = AIC_Pantheon - AIC_LCDM
delta_BIC_Pantheon = BIC_Pantheon - BIC_LCDM
delta_AIC_CMB = AIC_CMB - AIC_LCDM
delta_BIC_CMB = BIC_CMB - BIC_LCDM

# Exibir resultados
print(f"χ²_LCDM (Local Fit H0={H0_LCDM}, Ωm={Omega_m_LCDM}): {chi2_LCDM:.2f}")
print(f"χ²_Pantheon+ (H0={H0_Pantheon}, Ωm={Omega_m_Pantheon}): {chi2_Pantheon:.2f}")
print(f"χ²_CMB (H0={H0_CMB}, Ωm={Omega_m_CMB}): {chi2_CMB:.2f}")
print("\n--- Critérios de Informação ---")
print(f"AIC_LCDM (Local Fit): {AIC_LCDM:.2f}")
print(f"BIC_LCDM (Local Fit): {BIC_LCDM:.2f}")
print(f"AIC_Pantheon+: {AIC_Pantheon:.2f}")
print(f"BIC_Pantheon+: {BIC_Pantheon:.2f}")
print(f"AIC_CMB: {AIC_CMB:.2f}")
print(f"BIC_CMB: {BIC_CMB:.2f}")
print("\n--- Comparação com o Ajuste Local (Δ = Modelo - Local Fit) ---")
print(f"ΔAIC (Pantheon+ vs Local): {delta_AIC_Pantheon:.2f}")
print(f"ΔBIC (Pantheon+ vs Local): {delta_BIC_Pantheon:.2f}")
print(f"ΔAIC (CMB vs Local): {delta_AIC_CMB:.2f}")
print(f"ΔBIC (CMB vs Local): {delta_BIC_CMB:.2f}")

# Interpretação básica (escala de Jeffreys para ΔBIC, adaptada para AIC)
print("\n--- Interpretação (Referência: Kass & Raftery 1995 para BIC, adaptada) ---")
if delta_BIC_Pantheon > 10:
    print("Evidência muito forte contra Pantheon+ em favor do Ajuste Local (baseado em BIC).")
elif delta_BIC_Pantheon > 6:
    print("Evidência forte contra Pantheon+ em favor do Ajuste Local (baseado em BIC).")
elif delta_BIC_Pantheon > 2:
    print("Evidência positiva contra Pantheon+ em favor do Ajuste Local (baseado em BIC).")
else:
    print("Evidência não significativa entre Pantheon+ e Ajuste Local (baseado em BIC).")

if delta_BIC_CMB > 10:
    print("Evidência muito forte contra CMB em favor do Ajuste Local (baseado em BIC).")
elif delta_BIC_CMB > 6:
    print("Evidência forte contra CMB em favor do Ajuste Local (baseado em BIC).")
elif delta_BIC_CMB > 2:
    print("Evidência positiva contra CMB em favor do Ajuste Local (baseado em BIC).")
else:
    print("Evidência não significativa entre CMB e Ajuste Local (baseado em BIC).")

