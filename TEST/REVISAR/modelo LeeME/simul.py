# Função para Hubble sem o termo vibratório (modelo simplificado)
def hubble_wzcdm_no_vibration(z, H0, Om, w0, wa):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + w0)))

# Parâmetros ajustados sem vibração
params_wzcdm_no_vib = (71.32, 0.398, -2.03, 0.462)

# Z valores para o modelo de Hubble
z_model = np.linspace(0, 2, 100)

# Calcular H(z) para o modelo sem vibração
Hz_wzcdm_no_vib = [hubble_wzcdm_no_vibration(z, *params_wzcdm_no_vib) for z in z_model]

# Calcular o chi² para o modelo sem vibração
chi2_wzcdm_no_vib = calculate_chi2(Hz_real, z_real, z_model, Hz_wzcdm_no_vib, errHz_real)
print(f"χ² para o modelo sem vibração: {chi2_wzcdm_no_vib:.2f}")

# Plotar gráfico para comparação
plt.figure(figsize=(12, 8))

# Dados reais CMB & BAO
plt.errorbar(z_cmb_bao, Hz_cmb_bao, yerr=errHz_cmb_bao, fmt='o', markersize=6, capsize=4, color='black', label="CMB & BAO (Planck, SDSS, BOSS)")

# Dados reais Pantheon+ (supernovas)
plt.errorbar(z_pantheon, mu_pantheon, yerr=err_mu_pantheon, fmt='o', markersize=6, capsize=4, color='blue', label="Supernovas Pantheon+")

# Modelo sem vibração
plt.plot(z_model, Hz_wzcdm_no_vib, 'g-', label="Modelo w(z)CDM sem vibração")

# Ajustes dos gráficos
plt.xlabel("Redshift (z)", fontsize=14)
plt.ylabel("H(z) [km/s/Mpc] ou µ(z)", fontsize=14)
plt.title("Comparação entre o modelo w(z)CDM sem vibração e os dados reais (CMB, BAO, Pantheon+)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Exibir e salvar
plt.tight_layout()
plt.savefig("comparacao_sem_vibracao_reais.png", dpi=150)
plt.show()