import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import emcee
import corner

class CPLExtendedFriedmann:
    """
    Implementação da equação de Friedmann com energia escura dinâmica usando parametrização CPL.
    
    Parâmetros:
    H0 : float
        Constante de Hubble atual [km/s/Mpc]
    Om : float
        Densidade de matéria atual (Ω_m)
    w0 : float
        Parâmetro de equação de estado atual da energia escura
    wa : float
        Parâmetro de evolução da equação de estado da energia escura
    """
    def __init__(self, H0=70.0, Om=0.3, w0=-1.0, wa=0.0):
        self.H0 = H0
        self.Om = Om
        self.w0 = w0
        self.wa = wa
        
        # Densidade de radiação (calculada a partir da temperatura do CMB)
        self.h = self.H0 / 100.0
        self.Or = 2.469e-5 / self.h**2  # Ω_r h² = 2.469e-5
        
        # Densidade de energia escura (universo plano)
        self.Ode = 1.0 - self.Om - self.Or
        
    def w_z(self, z):
        """Equação de estado da energia escura em função do redshift"""
        return self.w0 + self.wa * z / (1 + z)
    
    def integrand(self, z):
        """Integrando para o termo exponencial da energia escura"""
        return (1 + self.w_z(z)) / (1 + z)
    
    def exp_term(self, z):
        """Calcula o termo exponencial da energia escura"""
        integral, _ = quad(self.integrand, 0, z)
        return np.exp(3 * integral)
    
    def H_z(self, z):
        """Taxa de Hubble em função do redshift"""
        if np.isscalar(z):
            z = np.array([z])
        
        Hz = np.zeros_like(z)
        for i, zi in enumerate(z):
            # Termos de matéria e radiação
            matter_term = self.Om * (1 + zi)**3
            radiation_term = self.Or * (1 + zi)**4
            
            # Termo de energia escura
            de_term = self.Ode * self.exp_term(zi)
            
            # Equação de Friedmann
            Hz[i] = self.H0 * np.sqrt(matter_term + radiation_term + de_term)
        
        return Hz[0] if np.isscalar(z) else Hz
    
    def H_z_vectorized(self, z):
        """Versão vetorizada da taxa de Hubble para melhor desempenho"""
        z = np.asarray(z)
        Hz = np.zeros_like(z)
        
        # Para cada redshift, calcular o termo exponencial
        for i, zi in enumerate(z):
            integral, _ = quad(self.integrand, 0, zi)
            exp_val = np.exp(3 * integral)
            
            matter = self.Om * (1 + zi)**3
            radiation = self.Or * (1 + zi)**4
            dark_energy = self.Ode * exp_val
            
            Hz[i] = self.H0 * np.sqrt(matter + radiation + dark_energy)
        
        return Hz
    
    def optimized_H_z(self, z):
        """
        Versão otimizada usando a forma fechada do modelo CPL.
        Válida quando a equação de estado é dada por w(z) = w0 + wa * z/(1+z)
        """
        z = np.asarray(z)
        a = 1 / (1 + z)  # Fator de escala
        
        # Termos de matéria e radiação
        matter_term = self.Om * (1 + z)**3
        radiation_term = self.Or * (1 + z)**4
        
        # Termo de energia escura (forma fechada)
        de_term = self.Ode * (1 + z)**(3*(1 + self.w0 + self.wa)) * np.exp(-3 * self.wa * (1 - a))
        
        return self.H0 * np.sqrt(matter_term + radiation_term + de_term)

# Função para ajustar o modelo aos dados observacionais
def fit_cpl_friedmann(z_data, Hz_data, Hz_err, 
                      H0_prior=(60, 80), Om_prior=(0.1, 0.5),
                      w0_prior=(-2.0, -0.5), wa_prior=(-2, 2)):
    """
    Ajusta o modelo CPL a dados H(z) usando MCMC
    
    Parâmetros:
    z_data, Hz_data, Hz_err : arrays
        Dados observacionais de redshift, H(z) e erros
    H0_prior, Om_prior, w0_prior, wa_prior : tuplas
        Intervalos de priors para os parâmetros
    """
    def log_likelihood(params):
        H0, Om, w0, wa = params
        model = CPLExtendedFriedmann(H0, Om, w0, wa)
        Hz_model = model.optimized_H_z(z_data)
        chi2 = np.sum(((Hz_data - Hz_model) / Hz_err)**2)
        return -0.5 * chi2
    
    def log_prior(params):
        H0, Om, w0, wa = params
        if (H0_prior[0] < H0 < H0_prior[1] and
            Om_prior[0] < Om < Om_prior[1] and
            w0_prior[0] < w0 < w0_prior[1] and
            wa_prior[0] < wa < wa_prior[1]):
            return 0.0
        return -np.inf
    
    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)
    
    # Configuração do MCMC
    nwalkers = 32
    ndim = 4
    nsteps = 5000
    
    # Posições iniciais
    initial_pos = np.array([
        [70, 0.3, -1.0, 0.0] + 0.1 * np.random.randn(ndim)
        for _ in range(nwalkers)
    ])
    
    # Executar MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    
    return sampler

# Exemplo de uso com dados simulados
if __name__ == "__main__":
    # Gerar dados simulados
    np.random.seed(42)
    z_sample = np.linspace(0, 2, 30)
    true_model = CPLExtendedFriedmann(H0=70, Om=0.3, w0=-1.0, wa=0.1)
    Hz_true = true_model.optimized_H_z(z_sample)
    Hz_observed = Hz_true + 5 * np.random.randn(len(z_sample))  # Adicionar ruído
    
    # Ajustar modelo aos dados
    sampler = fit_cpl_friedmann(z_sample, Hz_observed, np.full_like(Hz_observed, 5))
    
    # Analisar resultados
    samples = sampler.get_chain(discard=1000, thin=50, flat=True)
    med_params = np.median(samples, axis=0)
    std_params = np.std(samples, axis=0)
    
    print("Parâmetros ajustados:")
    print(f"H0 = {med_params[0]:.1f} ± {std_params[0]:.1f} km/s/Mpc")
    print(f"Ωm = {med_params[1]:.3f} ± {std_params[1]:.3f}")
    print(f"w0 = {med_params[2]:.2f} ± {std_params[2]:.2f}")
    print(f"wa = {med_params[3]:.2f} ± {std_params[3]:.2f}")
    
    # Plotar resultados
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_sample, Hz_observed, yerr=5, fmt='o', label='Dados observados')
    
    best_fit_model = CPLExtendedFriedmann(*med_params)
    z_fine = np.linspace(0, 2, 100)
    plt.plot(z_fine, best_fit_model.optimized_H_z(z_fine), 'r-', label='Melhor ajuste')
    plt.plot(z_fine, true_model.optimized_H_z(z_fine), 'k--', label='Verdadeiro')
    
    plt.xlabel('Redshift (z)')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.title('Modelo CPL de Energia Escura Dinâmica')
    plt.legend()
    plt.grid(True)
    plt.savefig('cpl_friedmann_fit.png', dpi=150)
    
    # Corner plot
    labels = ["$H_0$", "$\Omega_m$", "$w_0$", "$w_a$"]
    fig = corner.corner(
        samples, 
        labels=labels,
        truths=[70, 0.3, -1.0, 0.1],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    fig.savefig('cpl_corner_plot.png', dpi=150)