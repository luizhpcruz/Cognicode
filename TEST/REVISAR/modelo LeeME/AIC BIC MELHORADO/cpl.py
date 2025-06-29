import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# --- 1. Parâmetros ajustados dos modelos ---
best_params_lcdm = [69.83, 0.29, 147.1]
best_params_wcdm = [70.12, 0.28, -1.05, 146.8]
best_params_cpl = [70.52, 0.292, -1.082, 0.102, 144.3]

errors_lcdm = [0.48, 0.02, 3.5]
errors_wcdm = [0.51, 0.03, 0.08, 3.7]
errors_cpl = [0.62, 0.012, 0.042, 0.132, 3.8]

chi2_lcdm = 1598.35
chi2_wcdm = 1592.50
chi2_cpl = 1590.24

n_data = 1083  # Total: Pantheon+ (1046) + H(z) (31) + BAO (5) + SH0ES (1)
k_lcdm = 3
k_wcdm = 4
k_cpl = 5

# --- 2. AIC e BIC ---
def compute_aic_bic(chi2, k, n):
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    return aic, bic

aic_lcdm, bic_lcdm = compute_aic_bic(chi2_lcdm, k_lcdm, n_data)
aic_wcdm, bic_wcdm = compute_aic_bic(chi2_wcdm, k_wcdm, n_data)
aic_cpl, bic_cpl = compute_aic_bic(chi2_cpl, k_cpl, n_data)

delta_aic_lcdm = aic_lcdm - aic_cpl
delta_aic_wcdm = aic_wcdm - aic_cpl
delta_bic_lcdm = bic_lcdm - bic_cpl
delta_bic_wcdm = bic_wcdm - bic_cpl

# --- 3. Tabela LaTeX ---
table_data = [
    ["H$_0$ (km/s/Mpc)", f"{best_params_cpl[0]:.2f}", f"± {errors_cpl[0]:.2f}"],
    ["$\\Omega_m$", f"{best_params_cpl[1]:.3f}", f"± {errors_cpl[1]:.3f}"],
    ["$w_0$", f"{best_params_cpl[2]:.3f}", f"± {errors_cpl[2]:.3f}"],
    ["$w_a$", f"{best_params_cpl[3]:.3f}", f"± {errors_cpl[3]:.3f}"],
    ["$r_d$ (Mpc)", f"{best_params_cpl[4]:.1f}", f"± {errors_cpl[4]:.1f}"]
]

latex_table = tabulate(table_data, headers=["Parâmetro", "Valor", "Incerteza"], tablefmt="latex")
print(latex_table)

# --- 4. Banda 1σ para w(z) ---
np.random.seed(0)
n_samples = 5000
w0_samples = np.random.normal(best_params_cpl[2], errors_cpl[2], n_samples)
wa_samples = np.random.normal(best_params_cpl[3], errors_cpl[3], n_samples)

z_plot = np.linspace(0, 2.5, 100)
wz_samples = np.array([w0 + wa * z_plot / (1 + z_plot) for w0, wa in zip(w0_samples, wa_samples)])

wz_median = np.percentile(wz_samples, 50, axis=0)
wz_lower = np.percentile(wz_samples, 16, axis=0)
wz_upper = np.percentile(wz_samples, 84, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(z_plot, wz_median, 'b-', label='w(z) mediano')
plt.fill_between(z_plot, wz_lower, wz_upper, color='blue', alpha=0.3, label='1σ')
plt.axhline(-1, color='r', linestyle='--', label='ΛCDM (w = -1)')
plt.xlabel('Redshift (z)')
plt.ylabel('w(z)')
plt.title('Evolução de w(z) com banda de 1σ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('wz_band_cpl.png')
plt.show()

# --- 5. Resultados estatísticos ---
print("\nComparação estatística com o modelo CPL como base:")
print(f"ΔAIC (ΛCDM - CPL): {delta_aic_lcdm:.2f}")
print(f"ΔBIC (ΛCDM - CPL): {delta_bic_lcdm:.2f}")
print(f"ΔAIC (wCDM - CPL): {delta_aic_wcdm:.2f}")
print(f"ΔBIC (wCDM - CPL): {delta_bic_wcdm:.2f}")
