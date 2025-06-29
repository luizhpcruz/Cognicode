import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.linalg import cholesky, eigh
from scipy.optimize import curve_fit
from astropy.cosmology import w0waCDM
from sklearn.covariance import MinCovDet
import requests
from io import StringIO
import os

# Configurações gerais
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.linestyle': '--'
})
np.random.seed(42)

# =============================================
# 1. Carregamento e Pré-processamento de Dados
# =============================================

class DataLoader:
    """Carrega e processa dados cosmológicos de fontes online"""
    
    def __init__(self):
        self.data_sources = {
            'Hz': "https://gitlab.com/mmoresco/CCcov/-/raw/main/HzData_2023.csv",
            'cov_imf': "https://gitlab.com/mmoresco/CCcov/-/raw/main/sys_cov_imf.csv",
            'cov_lib': "https://gitlab.com/mmoresco/CCcov/-/raw/main/sys_cov_lib.csv",
            'bao': "https://sdss.org/dr18/data/bao/latest/BAO_consensus_results_DR18.csv"
        }
        
    def load_data(self):
        """Carrega todos os conjuntos de dados"""
        self.hz_data = self._load_csv(self.data_sources['Hz'])
        self.cov_imf = self._load_cov_matrix(self.data_sources['cov_imf'])
        self.cov_lib = self._load_cov_matrix(self.data_sources['cov_lib'])
        self.bao_data = self._load_csv(self.data_sources['bao'])
        
        # Aplicar correções iniciais
        self._apply_quasar_correction()
        self._create_full_covariance()
        
        return {
            'z': self.hz_data['z'].values,
            'Hz': self.hz_data['Hz_corrected'].values,
            'cov_total': self.cov_total,
            'bao': self.bao_data
        }
    
    def _load_csv(self, url):
        """Carrega dados CSV de uma URL"""
        response = requests.get(url)
        return pd.read_csv(StringIO(response.text))
    
    def _load_cov_matrix(self, url):
        """Carrega matriz de covariância de uma URL"""
        response = requests.get(url)
        return pd.read_csv(StringIO(response.text), header=None).values
    
    def _apply_quasar_correction(self):
        """Aplica correção para dados de quasares"""
        def correction(z, Hz):
            return Hz * (1 - 0.05 * np.tanh((z - 2.5)/0.4))
        
        self.hz_data['Hz_corrected'] = correction(
            self.hz_data['z'], 
            self.hz_data['Hz']
        )
        
        # Calcular pesos adaptativos
        weights = np.where(self.hz_data['z'] > 2.0, 0.7, 1.0)
        self.weights = weights / self.hz_data['stat_err']**2
    
    def _create_full_covariance(self):
        """Cria matriz de covariância total"""
        stat_cov = np.diag(self.hz_data['stat_err']**2)
        sys_cov = self.cov_imf + self.cov_lib  # Adicione outras fontes conforme necessário
        
        self.cov_total = stat_cov + sys_cov
        self._validate_covariance()
        
        # Aplicar pesos
        weight_matrix = np.sqrt(np.outer(self.weights, self.weights))
        self.cov_total = self.cov_total * weight_matrix
    
    def _validate_covariance(self):
        """Valida e corrige matriz de covariância"""
        try:
            cholesky(self.cov_total, lower=True)
        except np.linalg.LinAlgError:
            print("Matriz não é positiva definida! Aplicando correção...")
            self.cov_total = self.make_positive_definite(self.cov_total)
    
    @staticmethod
    def make_positive_definite(cov_matrix, epsilon=1e-6):
        """Corrige matriz para ser positiva definida"""
        eigvals, eigvecs = eigh(cov_matrix)
        adjusted_eigvals = np.maximum(eigvals, epsilon)
        return eigvecs @ np.diag(adjusted_eigvals) @ eigvecs.T

# =============================================
# 2. Modelos Cosmológicos e Funções de Ajuste
# =============================================

class CosmologyModels:
    """Implementa modelos cosmológicos e funções de ajuste"""
    
    @staticmethod
    def hz_model_lcdm(z, H0, Om0):
        """Modelo ΛCDM"""
        return H0 * np.sqrt(Om0 * (1+z)**3 + (1 - Om0))
    
    @staticmethod
    def hz_model_cpl(z, H0, Om0, w0, wa):
        """Modelo CPL (w(z) = w0 + wa * z/(1+z))"""
        return w0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa).H(z).value
    
    @staticmethod
    def bao_model(z, H0, Om0, w0=-1.0, wa=0.0):
        """Modelo teórico para BAO"""
        cosmo = w0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
        da = cosmo.angular_diameter_distance(z).value
        c = 299792.458  # km/s
        Hz = cosmo.H(z).value
        return (z * c * da**2 / Hz)**(1/3)

# =============================================
# 3. Análise MCMC
# =============================================

class MCMCAnalyzer:
    """Executa análise MCMC para modelos cosmológicos"""
    
    def __init__(self, data):
        self.data = data
        self.samplers = {}
    
    def run_mcmc(self, model_type, nwalkers=50, nsteps=5000):
        """Executa MCMC para um modelo específico"""
        ndim = 2 if model_type == 'lcdm' else 4
        initial_pos = self._get_initial_positions(ndim, model_type)
        
        sampler = emcee.EnsembleSampler(
            nwalkers, 
            ndim, 
            lambda params: self.log_probability(params, model_type)
        )
        
        sampler.run_mcmc(initial_pos, nsteps, progress=True)
        self.samplers[model_type] = sampler
        return sampler
    
    def _get_initial_positions(self, ndim, model_type):
        """Gera posições iniciais para walkers"""
        if model_type == 'lcdm':
            center = [67.0, 0.3]
        else:
            center = [67.0, 0.3, -1.0, 0.0]
        
        return [center + 1e-3 * np.random.randn(ndim) for _ in range(nwalkers)]
    
    def log_probability(self, params, model_type):
        """Função de probabilidade logarítmica"""
        # Priors
        lp = self.log_prior(params, model_type)
        if not np.isfinite(lp):
            return -np.inf
        
        # Likelihood
        return lp + self.log_likelihood(params, model_type)
    
    def log_prior(self, params, model_type):
        """Priors para parâmetros"""
        H0, Om0 = params[:2]
        if not (50 < H0 < 80 and 0.1 < Om0 < 0.6):
            return -np.inf
        
        if model_type == 'cpl':
            w0, wa = params[2:]
            if not (-3 < w0 < 0 and -5 < wa < 5):
                return -np.inf
        return 0.0
    
    def log_likelihood(self, params, model_type):
        """Log-verossimilhança para dados H(z)"""
        z = self.data['z']
        Hz_obs = self.data['Hz']
        cov = self.data['cov_total']
        
        if model_type == 'lcdm':
            Hz_model = CosmologyModels.hz_model_lcdm(z, *params[:2])
        else:
            Hz_model = CosmologyModels.hz_model_cpl(z, *params)
        
        residual = Hz_obs - Hz_model
        return -0.5 * residual @ np.linalg.solve(cov, residual)
    
    def combined_log_likelihood(self, params, model_type):
        """Likelihood combinada H(z) + BAO"""
        lh_hz = self.log_likelihood(params, model_type)
        
        # BAO likelihood
        bao_data = self.data['bao']
        z_bao = bao_data['z'].values
        bao_meas = bao_data['measurement'].values
        bao_err = bao_data['error'].values
        
        if model_type == 'lcdm':
            bao_model = CosmologyModels.bao_model(z_bao, params[0], params[1])
        else:
            bao_model = CosmologyModels.bao_model(z_bao, params[0], params[1], params[2], params[3])
        
        chi2_bao = np.sum(((bao_meas - bao_model) / bao_err)**2)
        return lh_hz - 0.5 * chi2_bao

# =============================================
# 4. Análise de Resultados e Visualização
# =============================================

class ResultAnalyzer:
    """Analisa e visualiza resultados do MCMC"""
    
    def __init__(self, sampler, data, model_type):
        self.sampler = sampler
        self.data = data
        self.model_type = model_type
        self.samples = sampler.get_chain(discard=500, thin=50, flat=True)
    
    def compute_summary(self):
        """Calcula estatísticas resumidas"""
        med = np.median(self.samples, axis=0)
        q16 = np.percentile(self.samples, 16, axis=0)
        q84 = np.percentile(self.samples, 84, axis=0)
        return {
            'median': med,
            'lower_error': med - q16,
            'upper_error': q84 - med
        }
    
    def plot_corner(self, filename=None):
        """Gera corner plot"""
        if self.model_type == 'lcdm':
            labels = ["$H_0$", "$\Omega_m$"]
            truths = [67.4, 0.315]
        else:
            labels = ["$H_0$", "$\Omega_m$", "$w_0$", "$w_a$"]
            truths = [67.4, 0.315, -1.0, 0.0]
        
        fig = corner.corner(
            self.samples,
            labels=labels,
            truths=truths,
            quantiles=[0.16, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_hz_fit(self, filename=None):
        """Plota ajuste de H(z) com dados"""
        plt.figure(figsize=(10, 6))
        
        # Dados observacionais
        plt.errorbar(
            self.data['z'], 
            self.data['Hz'],
            yerr=np.sqrt(np.diag(self.data['cov_total'])),
            fmt='o', color='#3498db', ecolor='#2980b9',
            capsize=4, label='Dados H(z)'
        )
        
        # Melhor ajuste
        params = np.median(self.samples, axis=0)
        z_fit = np.linspace(0.01, 2.5, 100)
        
        if self.model_type == 'lcdm':
            hz_fit = CosmologyModels.hz_model_lcdm(z_fit, *params)
        else:
            hz_fit = CosmologyModels.hz_model_cpl(z_fit, *params)
        
        plt.plot(z_fit, hz_fit, 'r-', linewidth=2.5, label='Melhor ajuste')
        
        # ΛCDM padrão para comparação
        hz_lcdm = CosmologyModels.hz_model_lcdm(z_fit, 67.4, 0.315)
        plt.plot(z_fit, hz_lcdm, 'k--', alpha=0.7, label='ΛCDM padrão')
        
        plt.xlabel('Redshift (z)')
        plt.ylabel('H(z) [km s⁻¹ Mpc⁻¹]')
        plt.title(f'Ajuste do modelo {self.model_type.upper()} aos dados H(z)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def calculate_tension(self, h0_ref=73.04, h0_ref_err=1.04):
        """Calcula tensão de Hubble"""
        h0_samples = self.samples[:, 0]
        h0_median = np.median(h0_samples)
        h0_err = (np.percentile(h0_samples, 84) - np.percentile(h0_samples, 16)) / 2
        
        tension = np.abs(h0_median - h0_ref) / np.sqrt(h0_err**2 + h0_ref_err**2)
        return tension

# =============================================
# 5. Fluxo Principal de Análise
# =============================================

def main():
    # Etapa 1: Carregar dados
    print("Carregando dados...")
    loader = DataLoader()
    data = loader.load_data()
    
    # Etapa 2: Executar análises MCMC
    print("\nExecutando MCMC para ΛCDM...")
    analyzer = MCMCAnalyzer(data)
    sampler_lcdm = analyzer.run_mcmc('lcdm', nsteps=3000)
    
    print("\nExecutando MCMC para CPL...")
    sampler_cpl = analyzer.run_mcmc('cpl', nsteps=5000)
    
    # Etapa 3: Analisar resultados
    print("\nAnalisando resultados...")
    results_lcdm = ResultAnalyzer(sampler_lcdm, data, 'lcdm')
    results_cpl = ResultAnalyzer(sampler_cpl, data, 'cpl')
    
    summary_lcdm = results_lcdm.compute_summary()
    summary_cpl = results_cpl.compute_summary()
    
    # Etapa 4: Calcular tensão de Hubble
    tension = results_lcdm.calculate_tension()
    
    # Etapa 5: Gerar visualizações
    print("\nGerando visualizações...")
    os.makedirs("plots", exist_ok=True)
    results_lcdm.plot_corner("plots/corner_lcdm.png")
    results_cpl.plot_corner("plots/corner_cpl.png")
    results_lcdm.plot_hz_fit("plots/hz_fit_lcdm.png")
    results_cpl.plot_hz_fit("plots/hz_fit_cpl.png")
    
    # Etapa 6: Salvar resultados
    print("\nSalvando resultados...")
    np.savez("results.npz", 
             samples_lcdm=sampler_lcdm.get_chain(flat=True),
             samples_cpl=sampler_cpl.get_chain(flat=True),
             summary_lcdm=summary_lcdm,
             summary_cpl=summary_cpl,
             tension=tension)
    
    # Etapa 7: Relatório final
    print("\n===== RESULTADOS FINAIS =====")
    print(f"ΛCDM: H0 = {summary_lcdm['median'][0]:.1f} +{summary_lcdm['upper_error'][0]:.1f}/-{summary_lcdm['lower_error'][0]:.1f}")
    print(f"       Ωm = {summary_lcdm['median'][1]:.3f} +{summary_lcdm['upper_error'][1]:.3f}/-{summary_lcdm['lower_error'][1]:.3f}")
    
    print(f"\nCPL:  H0 = {summary_cpl['median'][0]:.1f} +{summary_cpl['upper_error'][0]:.1f}/-{summary_cpl['lower_error'][0]:.1f}")
    print(f"       Ωm = {summary_cpl['median'][1]:.3f} +{summary_cpl['upper_error'][1]:.3f}/-{summary_cpl['lower_error'][1]:.3f}")
    print(f"       w0 = {summary_cpl['median'][2]:.2f} +{summary_cpl['upper_error'][2]:.2f}/-{summary_cpl['lower_error'][2]:.2f}")
    print(f"       wa = {summary_cpl['median'][3]:.2f} +{summary_cpl['upper_error'][3]:.2f}/-{summary_cpl['lower_error'][3]:.2f}")
    
    print(f"\nTensão de Hubble: {tension:.1f}σ")
    print("\nAnálise concluída! Resultados salvos em 'results.npz' e pasta 'plots'")

if __name__ == "__main__":
    main()