import numpy as np
from scipy.integrate import quad, cumtrapz
import matplotlib.pyplot as plt
from astropy import constants as const

# --- Constantes Fundamentais ---
c = const.c.to('km/s').value  # Velocidade da luz em km/s
k_b = const.k_B.to('eV/K').value  # Constante de Boltzmann em eV/K
h_planck = const.h.to('eV s').value  # Constante de Planck em eV·s

# --- Parâmetros do Modelo Híbrido ---
H0 = 70.52                  # Constante de Hubble (km/s/Mpc)
Om = 0.292                  # Densidade de matéria
f_ede = 0.07                # Fração de energia escura precoce
z_c = 0.42                  # Redshift de transição da EDE
w0_phantom = -1.08          # Parâmetro de equação de estado fantasma (z=0)
wa_phantom = 0.10           # Variação com redshift da equação de estado

# --- Funções do Modelo Fotônico-Fantasma ---
def eta_nu(nu):
    """Função de acoplamento espectral (ressonância em 0.95 eV)"""
    nu_res = 2.3e14  # Hz (0.95 eV)
    sigma = 0.15 * nu_res
    return np.exp(-0.5 * ((nu - nu_res)/sigma)**2)

def planck_spectrum(nu, T):
    """Densidade espectral de energia por frequência (eV·s/m³)"""
    term1 = 8*np.pi*h_planck*nu**3 / c**3
    term2 = 1.0 / (np.exp(h_planck*nu/(k_b*T)) - 1)
    return term1 * term2

def rho_phantom(z, T_cmb0=2.725):
    """Densidade de energia fantasma derivada de fótons (eV/m³)"""
    T_cmb = T_cmb0 * (1 + z)
    nu_min, nu_max = 1e11, 1e17  # De rádio a UV
    
    integrand = lambda nu: eta_nu(nu) * planck_spectrum(nu, T_cmb)
    rho, _ = quad(integrand, nu_min, nu_max, limit=500)
    
    return rho

# --- Componentes do Modelo Híbrido EDE+CPL+Fótons ---
def Omega_ede(z):
    """Componente de Energia Escura Precoce (EDE)"""
    return f_ede * (1 + z)**3 / ((1 + z)**3 + (1 + z_c)**3)

def w_phantom(z):
    """Equação de estado para a componente fantasma (formato CPL)"""
    return w0_phantom + wa_phantom * z / (1 + z)

def phantom_energy_density(z):
    """Densidade de energia da componente fantasma (sem acoplamento fotônico)"""
    # Integral para a densidade de energia da componente CPL
    z_vals = np.linspace(0, z, 100)
    integrand = 3 * (1 + w_phantom(z_vals)) / (1 + z_vals)
    integral = cumtrapz(integrand, z_vals, initial=0)[-1]
    return np.exp(integral)

def hubble_hybrid(z):
    """Função Hubble para o modelo híbrido EDE+CPL+Fótons"""
    # Densidade de matéria
    rho_m = Om * (1 + z)**3
    
    # Componente de energia escura precoce
    ede = Omega_ede(z)
    
    # Componente fantasma (CPL + acoplamento fotônico)
    rho_ph = rho_phantom(z)
    rho_ph0 = rho_phantom(0)
    phantom_cpl = (1 - Om - f_ede) * phantom_energy_density(z)
    
    # Termo de acoplamento fotônico (normalizado)
    photon_coupling = 0.3 * (rho_ph / rho_ph0)  # Fração ajustável
    
    # Soma de todas as componentes
    total_energy = rho_m + ede + phantom_cpl + photon_coupling
    
    return H0 * np.sqrt(total_energy)

# --- Simulação e Análise ---
z_vals = np.linspace(0, 10, 200)
Hz_vals = np.array([hubble_hybrid(z) for z in z_vals])

# Cálculo da equação de estado efetiva
w_eff = np.zeros_like(z_vals)
dz = 0.01

for i, z in enumerate(z_vals):
    if 0 < z < z_vals[-1] - dz:
        H_minus = hubble_hybrid(z - dz)
        H_plus = hubble_hybrid(z + dz)
        dH_dz = (H_plus - H_minus) / (2 * dz)
        H = hubble_hybrid(z)
        w_eff[i] = -1 + (1 + z) * dH_dz / (3 * H)

# --- Visualização dos Resultados ---
plt.figure(figsize=(15, 10))

# 1. Evolução da Função Hubble
plt.subplot(221)
plt.plot(z_vals, Hz_vals, 'r-', lw=2.5)
plt.xlabel('Redshift (z)')
plt.ylabel('H(z) (km s⁻¹ Mpc⁻¹)')
plt.title('Função Hubble no Modelo Híbrido')
plt.grid(True)

# 2. Equação de Estado Efetiva
plt.subplot(222)
plt.plot(z_vals, w_eff, 'b-', lw=2.5)
plt.axhline(-1, color='k', linestyle='--', label='w = -1 (ΛCDM)')
plt.axhline(w0_phantom, color='g', linestyle=':', label='w₀ fantasma')
plt.xlabel('Redshift (z)')
plt.ylabel('$w_{eff}(z)$')
plt.title('Equação de Estado Efetiva')
plt.legend()
plt.grid(True)

# 3. Contribuições Relativas
rho_components = np.zeros((len(z_vals), 4))

for i, z in enumerate(z_vals):
    rho_m = Om * (1 + z)**3
    ede = Omega_ede(z)
    phantom_cpl = (1 - Om - f_ede) * phantom_energy_density(z)
    rho_ph = rho_phantom(z)
    rho_ph0 = rho_phantom(0)
    photon_coupling = 0.3 * (rho_ph / rho_ph0)
    
    total = rho_m + ede + phantom_cpl + photon_coupling
    rho_components[i] = [
        rho_m / total,
        ede / total,
        phantom_cpl / total,
        photon_coupling / total
    ]

plt.subplot(223)
labels = ['Matéria', 'EDE', 'Fantasma (CPL)', 'Acoplamento Fotônico']
for j in range(4):
    plt.plot(z_vals, rho_components[:, j], label=labels[j], lw=2)
    
plt.xlabel('Redshift (z)')
plt.ylabel('Fração da Densidade de Energia')
plt.title('Contribuições Relativas dos Componentes')
plt.legend()
plt.grid(True)

# 4. Comparação com Modelos Padrão
plt.subplot(224)
# Modelo ΛCDM
Hz_lcdm = H0 * np.sqrt(Om*(1+z_vals)**3 + (1 - Om))

# Modelo wCDM com w = -1.08
Hz_wcdm = H0 * np.sqrt(Om*(1+z_vals)**3 + (1 - Om)*(1+z_vals)**(3*(1-1.08)))

plt.plot(z_vals, Hz_vals, 'r-', lw=2.5, label='Híbrido (EDE+CPL+Fótons)')
plt.plot(z_vals, Hz_lcdm, 'b--', lw=2, label='ΛCDM')
plt.plot(z_vals, Hz_wcdm, 'g:', lw=2, label='wCDM (w=-1.08)')
plt.xlabel('Redshift (z)')
plt.ylabel('H(z) (km s⁻¹ Mpc⁻¹)')
plt.title('Comparação com Modelos Padrão')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('hybrid_model_results.png', dpi=300)
plt.show()

# --- Análise Quantitativa ---
# Calcular H0 efetivo local
H0_local = hubble_hybrid(0)

# Calcular parâmetro de desaceleração em z=0
q0 = -1 + (1 + 0) * (hubble_hybrid(0.01) - hubble_hybrid(-0.01)) / (2*0.01) / (3 * H0_local)

# Calcular redshift de transição (q=0)
z_trans = None
for i in range(1, len(z_vals)):
    if w_eff[i] < -1/3 and w_eff[i-1] >= -1/3:
        z_trans = z_vals[i]
        break

# --- Salvar Resultados ---
np.savetxt("hybrid_cosmo_model.txt", 
           np.column_stack((z_vals, Hz_vals, w_eff, rho_components)),
           header="z H(z) w_eff(z) frac_materia frac_ede frac_phantom_CPL frac_photon_coupling")

print("\n" + "="*60)
print("RESULTADOS DO MODELO HÍBRIDO EDE+CPL+FÓTONS")
print("="*60)
print(f"H₀ local: {H0_local:.2f} km/s/Mpc")
print(f"Parâmetro de desaceleração (q₀): {q0:.3f}")
print(f"Redshift de transição (aceleração cósmica): z = {z_trans:.2f}")
print(f"w_eff(z=0): {w_eff[0]:.3f}")
print(f"w_eff(z=1): {w_eff[np.argmin(np.abs(z_vals-1))]:.3f}")
print("\nComponentes em z=0:")
print(f"- Matéria: {rho_components[0,0]*100:.1f}%")
print(f"- EDE: {rho_components[0,1]*100:.1f}%")
print(f"- Fantasma CPL: {rho_components[0,2]*100:.1f}%")
print(f"- Acoplamento fotônico: {rho_components[0,3]*100:.1f}%")
print("\nFigura salva em: hybrid_model_results.png")