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
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

# Calcular H(z) modelo ΛCDM e modelos padrão
H_model_LCDM = H_LCDM(z, H0_LCDM, Omega_m_LCDM)
H_model_Pantheon = H_LCDM(z, H0_Pantheon, Omega_m_Pantheon)
H_model_CMB = H_LCDM(z, H0_CMB, Omega_m_CMB)

# Calcular χ² para cada modelo
def chi2(H_obs, H_model, sigma_H):
    return np.sum(((H_obs - H_model) / sigma_H) ** 2)

chi2_LCDM = chi2(H_obs, H_model_LCDM, sigma_H)
chi2_Pantheon = chi2(H_obs, H_model_Pantheon, sigma_H)
chi2_CMB = chi2(H_obs, H_model_CMB, sigma_H)

# Número de parâmetros
k_LCDM = 2
k_CPL = 4
n = len(z)

# Critérios AIC e BIC
def aic(chi2, k):
    return chi2 + 2 * k

def bic(chi2, k, n):
    return chi2 + k * np.log(n)

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
print(f"χ²_LCDM: {chi2_LCDM:.2f}")
print(f"χ²_Pantheon+: {chi2_Pantheon:.2f}")
print(f"χ²_CMB: {chi2_CMB:.2f}")
print(f"AIC_LCDM: {AIC_LCDM:.2f}, BIC_LCDM: {BIC_LCDM:.2f}")
print(f"AIC_Pantheon+: {AIC_Pantheon:.2f}, BIC_Pantheon+: {BIC_Pantheon:.2f}")
print(f"AIC_CMB: {AIC_CMB:.2f}, BIC_CMB: {BIC_CMB:.2f}")
print(f"ΔAIC_Pantheon: {delta_AIC_Pantheon:.2f}, ΔBIC_Pantheon: {delta_BIC_Pantheon:.2f}")
print(f"ΔAIC_CMB: {delta_AIC_CMB:.2f}, ΔBIC_CMB: {delta_BIC_CMB:.2f}")