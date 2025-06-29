# DNA Cósmico Expandido Integrado com ΛCDM, Termodinâmica, Bounce e Validação com Dados Observacionais
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import emcee
import pandas as pd

# ------------------------------------------------
# Classe do modelo cósmico (TREC + DNA Cósmico)
# ------------------------------------------------
class CosmicDNA:
    def __init__(self, sequence_length=100):
        self.bases = ['G','C','Ψ','Φ','Ω','Λ','A','T']
        self.sequence = list(np.random.choice(self.bases, sequence_length))
        self.temperature = 1e9  # K
        self.entropy = 0.

    def update_thermodynamics(self):
        decay = 1e7
        self.temperature = max(0, self.temperature - decay)
        self.entropy += self.temperature * 1e-6
        if self.entropy > 1e6:
            self.sequence = ['Ω','Φ','Λ']*(len(self.sequence)//3)

    def apply_quantum_torsion(self):
        if self.sequence.count('Λ')/len(self.sequence) > 0.7:
            i=np.random.randint(len(self.sequence)-3)
            self.sequence[i:i+3]=['Φ','Ψ','Λ']

    def quantum_entanglement(self, other):
        if self.sequence.count('Ψ')/len(self.sequence) > 0.6:
            p=np.random.randint(5,15)
            self.sequence[p:], other.sequence[p:]=other.sequence[p:], self.sequence[p:]

    def simulate_evolution(self, steps=10, other=None):
        for _ in range(steps):
            self.update_thermodynamics()
            self.apply_quantum_torsion()
            if other: self.quantum_entanglement(other)

# ------------------------------------------------
# Equações de Friedmann para ΛCDM + Ω_ond (rotogênica) + Quintessência CPL
# ------------------------------------------------
def H_model(a, H0, Omega_m, Omega_L, Omega_ond=0, w0=-1, wa=0, model='lcdm'):
    if model == 'trec':
        return H0 * np.sqrt(Omega_m/a**3 + Omega_L + Omega_ond/a)
    elif model == 'cpl':
        w = w0 + wa*(1 - a)
        return H0 * np.sqrt(Omega_m/a**3 + (1 - Omega_m)*a**(-3*(1+w)))
    else:  # ΛCDM padrão
        return H0 * np.sqrt(Omega_m/a**3 + Omega_L)

def friedmann_eq(t, a, H0, Omega_m, Omega_L, Omega_ond=0, w0=-1, wa=0, model='lcdm'):
    return [a[0] * H_model(a[0], H0, Omega_m, Omega_L, Omega_ond, w0, wa, model)]

# ------------------------------------------------
# Likelihoods para TREC e Quintessência CPL
# ------------------------------------------------
def load_pantheon(filepath):
    df = pd.read_csv(filepath)
    return df['z'].values, df['mu'].values, df['mu_err'].values

def luminosity_distance(z, H0, Omega_m, Omega_L, Omega_ond=0, w0=-1, wa=0, model='lcdm'):
    c = 299792.458
    if model == 'trec':
        Ez = lambda zz: np.sqrt(Omega_m*(1+zz)**3 + Omega_L + Omega_ond*(1+zz)**4)
    elif model == 'cpl':
        Ez = lambda zz: np.sqrt(Omega_m*(1+zz)**3 + (1 - Omega_m)*(1+zz)**(3*(1+w0+wa))*np.exp(-3*wa*zz/(1+zz)))
    else:
        Ez = lambda zz: np.sqrt(Omega_m*(1+zz)**3 + Omega_L)
    from scipy.integrate import quad
    dH = c/H0
    dcs = np.array([quad(lambda zz: 1/Ez(zz),0,zi)[0] for zi in z]) * dH
    return (1+z)*dcs

def log_likelihood_lcdm(theta, z, mu, mu_err):
    H0, Omega_m, Omega_L = theta
    mu_model = 5*np.log10(luminosity_distance(z,H0,Omega_m,Omega_L)) + 25
    return -0.5 * np.sum(((mu-mu_model)/mu_err)**2)

def log_likelihood_trec(theta, z, mu, mu_err):
    H0, Omega_m, Omega_L, Omega_ond = theta
    mu_model = 5*np.log10(luminosity_distance(z,H0,Omega_m,Omega_L,Omega_ond,model='trec')) + 25
    return -0.5 * np.sum(((mu-mu_model)/mu_err)**2)

def log_likelihood_cpl(theta, z, mu, mu_err):
    H0, Omega_m, w0, wa = theta
    mu_model = 5*np.log10(luminosity_distance(z,H0,Omega_m,1-Omega_m,w0=w0,wa=wa,model='cpl')) + 25
    return -0.5 * np.sum(((mu-mu_model)/mu_err)**2)

# ------------------------------------------------
# Comparação estatística
# ------------------------------------------------
def compute_aic_bic(logL, k, n):
    aic = 2*k - 2*logL
    bic = k*np.log(n) - 2*logL
    return aic, bic

# ------------------------------------------------
# Execução
# ------------------------------------------------
if __name__ == '__main__':
    z, mu, mu_err = load_pantheon('pantheon.csv')
    N = len(z)

    # ΛCDM
    p1 = [70, 0.3, 0.7]
    ll1 = log_likelihood_lcdm(p1, z, mu, mu_err)
    aic1, bic1 = compute_aic_bic(ll1, k=3, n=N)

    # TREC
    p2 = [70, 0.3, 0.7, 0.02]
    ll2 = log_likelihood_trec(p2, z, mu, mu_err)
    aic2, bic2 = compute_aic_bic(ll2, k=4, n=N)

    # Quintessência (CPL)
    p3 = [70, 0.3, -1, 0.2]
    ll3 = log_likelihood_cpl(p3, z, mu, mu_err)
    aic3, bic3 = compute_aic_bic(ll3, k=4, n=N)

    print("\n### COMPARAÇÃO MODELOS ###")
    print(f"ΛCDM:        logL={ll1:.2f}, AIC={aic1:.2f}, BIC={bic1:.2f}")
    print(f"TREC:         logL={ll2:.2f}, AIC={aic2:.2f}, BIC={bic2:.2f}")
    print(f"Quintessência: logL={ll3:.2f}, AIC={aic3:.2f}, BIC={bic3:.2f}")

    # Comparação com dados observacionais
    obs = pd.read_csv('observational_data.csv')
    def compare_param(param, model_vals, obs_df):
        plt.figure(figsize=(6,4))
        for m, val, err in model_vals:
            plt.errorbar(m, val, xerr=0, yerr=err, fmt='o', label=m)
        for idx, row in obs_df.dropna(subset=[param]).iterrows():
            plt.errorbar(row['model'], row[param], yerr=row[param+'_err'], fmt='s', c='gray', alpha=0.7)
        plt.ylabel(param)
        plt.title(f"Comparação: {param}")
        plt.legend()
        plt.show()

    model_vals_om = [('ΛCDM', p1[1], 0.01), ('TREC', p2[1], 0.015), ('CPL', p3[1], 0.02)]
    compare_param('Omega_m', model_vals_om, obs)

    model_vals_h0 = [('ΛCDM', p1[0], 0.5), ('TREC', p2[0], 0.5), ('CPL', p3[0], 0.5)]
    compare_param('H0', model_vals_h0, obs)

    tex = obs.to_latex(index=False, float_format="%.3f")
    with open('obs_comparison.tex', 'w') as f:
        f.write(tex)
