import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import curve_fit
import pandas as pd
from getdist import plots, MCSamples
import camb
from astropy.cosmology import Planck18
import corner
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display

plt.style.use('dark_background')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.facecolor'] = '0.1'
plt.rcParams['figure.facecolor'] = '0.15'

# ================================================================
# CLASSE COSMOLÓGICA AVANÇADA COM NOVAS FUNCIONALIDADES
# ================================================================
class AdvancedCosmology:
    def __init__(self, H0=70, Om=0.3, Ol=0.7, w=-1.0, Ob=0.048, ns=0.965, sigma8=0.81):
        self.H0 = H0  # km/s/Mpc
        self.Om = Om  # Matéria total
        self.Ol = Ol  # Energia escura
        self.w = w    # Equação de estado
        self.Ob = Ob  # Matéria bariônica
        self.ns = ns  # Índice espectral
        self.sigma8 = sigma8  # Amplitude das flutuações
        
        self.h = H0/100
        self.Ok = 1 - Om - Ol  # Curvatura
        self.Tcmb = 2.7255     # Temperatura CMB
        self.current_z = 0      # Redshift atual para visualização

    def hubble(self, z):
        """Parâmetro Hubble em função do redshift"""
        return self.H0 * np.sqrt(self.Om*(1+z)**3 + self.Ok*(1+z)**2 + self.Ol*(1+z)**(3*(1+self.w)))

    def age(self, z=0):
        """Idade do universo em Gyr até redshift z"""
        def integrand(z_prime):
            return 1/((1+z_prime)*self.hubble(z_prime))
        return quad(integrand, z, np.inf)[0] / (3.154e17)  # Converter para Gyr

    def luminosity_distance(self, z):
        """Distância luminosa em Mpc"""
        def integrand(z_prime):
            return 1/self.hubble(z_prime)
        dc = (3e5/self.H0) * quad(integrand, 0, z)[0]  # Distância comóvel
        return dc * (1+z)  # Distância luminosa

    def comoving_volume(self, z):
        """Volume comóvel até redshift z [Gpc³]"""
        def integrand(z_prime):
            return (self.luminosity_distance(z_prime)/(1+z_prime))**2 / self.hubble(z_prime)
        return 4*np.pi*(quad(integrand, 0, z)[0] / (3.086e22*1e3))**3  # Converter para Gpc³

    def cmb_spectrum(self, lmax=2500):
        """Gerar espectro CMB usando CAMB"""
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.Ob*self.h**2, omch2=(self.Om-self.Ob)*self.h**2, 
                         mnu=0.06, omk=self.Ok, tau=0.054)
        pars.InitPower.set_params(As=2e-9, ns=self.ns)
        pars.set_for_lmax(lmax, lens_potential_accuracy=1)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        return powers['total'][2:lmax+1,0], powers['total'][2:lmax+1,1:4]

    def matter_power_spectrum(self, z=0, k_max=10, npoints=200):
        """Espectro de potência da matéria"""
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.Ob*self.h**2, omch2=(self.Om-self.Ob)*self.h**2)
        pars.InitPower.set_params(As=2e-9, ns=self.ns)
        pars.set_matter_power(redshifts=[z], kmax=k_max)
        results = camb.get_results(pars)
        k, z, Pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=k_max, npoints=npoints)
        return k, Pk[0]

    def growth_function(self, z):
        """Fator de crescimento linear das perturbações"""
        def integrand(a):
            return 1/(a*self.hubble(1/a-1))**3
        a = 1/(1+z)
        integral = quad(integrand, 0, a)[0]
        return self.hubble(z) * integral * (5*self.Om/2)
    
    # NOVA FUNCIONALIDADE: Visualização 3D da estrutura em grande escala
    def generate_lss_3d(self, size=256, box_size=500):
        """Gerar visualização 3D da estrutura em grande escala"""
        from scipy.fft import fftn, ifftn, fftfreq
        
        # Gerar campo gaussiano aleatório com espectro de potência correto
        k = fftfreq(size) * 2 * np.pi * size / box_size
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        
        # Aplicar filtro do espectro de potência
        k_mag[k_mag == 0] = np.inf
        pk_interp = np.interp(k_mag.ravel(), *self.matter_power_spectrum())
        pk_interp = pk_interp.reshape(k_mag.shape)
        pk_interp[k_mag == np.inf] = 0
        
        # Gerar campo de densidade
        noise = np.random.normal(0, 1, (size, size, size)) + 1j*np.random.normal(0, 1, (size, size, size))
        delta_k = noise * np.sqrt(pk_interp)
        delta = np.real(ifftn(delta_k))
        
        # Normalizar
        delta = (delta - delta.mean()) / delta.std()
        return delta

# ================================================================
# INTERFACE INTERATIVA (NOVA FUNCIONALIDADE)
# ================================================================
class CosmologyExplorer:
    def __init__(self, models):
        self.models = models
        self.current_model = list(models.keys())[0]
        self.current_z = 0
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_ui()
        self.update_plots()
    
    def setup_ui(self):
        """Configurar interface interativa"""
        gs = self.fig.add_gridspec(3, 3)
        
        # Gráficos principais
        self.ax_hubble = self.fig.add_subplot(gs[0, 0])
        self.ax_cmb = self.fig.add_subplot(gs[0, 1])
        self.ax_pk = self.fig.add_subplot(gs[0, 2])
        self.ax_3d = self.fig.add_subplot(gs[1:, :], projection='3d')
        
        # Controles
        ax_slider = self.fig.add_axes([0.25, 0.05, 0.5, 0.03])
        self.z_slider = Slider(ax_slider, 'Redshift (z)', 0, 10, valinit=0)
        self.z_slider.on_changed(self.update_z)
        
        ax_buttons = self.fig.add_axes([0.1, 0.02, 0.15, 0.1])
        ax_buttons.axis('off')
        self.model_selector = widgets.RadioButtons(
            options=list(self.models.keys()),
            description='Modelo:',
            disabled=False
        )
        self.model_selector.observe(self.update_model, names='value')
        
        # Adicionar widgets ao matplotlib
        display(self.model_selector)
    
    def update_z(self, val):
        self.current_z = val
        self.update_plots()
    
    def update_model(self, change):
        self.current_model = change.new
        self.update_plots()
    
    def update_plots(self):
        """Atualizar todos os gráficos"""
        model = self.models[self.current_model]
        
        # Limpar eixos
        for ax in [self.ax_hubble, self.ax_cmb, self.ax_pk, self.ax_3d]:
            ax.clear()
        
        # 1. Evolução do Hubble
        z = np.linspace(0, 10, 100)
        self.ax_hubble.plot(z, model.hubble(z), label=self.current_model)
        self.ax_hubble.axvline(self.current_z, color='red', linestyle='--')
        self.ax_hubble.set_xlabel('Redshift (z)')
        self.ax_hubble.set_ylabel('H(z) [km/s/Mpc]')
        self.ax_hubble.set_title('Evolução do Parâmetro Hubble')
        self.ax_hubble.legend()
        self.ax_hubble.grid(alpha=0.3)
        
        # 2. Espectro CMB
        ell = np.arange(2, 2501)
        cl = model.cmb_spectrum()[0]
        self.ax_cmb.plot(ell, cl[:2500-1], label=self.current_model)
        self.ax_cmb.set_xscale('log')
        self.ax_cmb.set_xlabel('Multipolo (ℓ)')
        self.ax_cmb.set_ylabel('$D_ℓ$ [μK²]')
        self.ax_cmb.set_title('Espectro CMB TT')
        self.ax_cmb.legend()
        self.ax_cmb.grid(alpha=0.3)
        
        # 3. Espectro de Potência
        k, Pk = model.matter_power_spectrum(z=self.current_z)
        self.ax_pk.plot(k, Pk, label=f'z={self.current_z:.1f}')
        self.ax_pk.set_xscale('log')
        self.ax_pk.set_yscale('log')
        self.ax_pk.set_xlabel('k [h/Mpc]')
        self.ax_pk.set_ylabel('P(k)')
        self.ax_pk.set_title('Espectro de Potência da Matéria')
        self.ax_pk.legend()
        self.ax_pk.grid(alpha=0.3)
        
        # 4. Visualização 3D (NOVA FUNCIONALIDADE)
        delta = model.generate_lss_3d(size=64)
        x, y, z = np.mgrid[0:64, 0:64, 0:64]
        mask = delta > 1.0  # Mostrar apenas regiões densas
        self.ax_3d.scatter(x[mask], y[mask], z[mask], c=delta[mask], 
                          cmap='viridis', s=2, alpha=0.5)
        self.ax_3d.set_title(f'Estrutura em Grande Escala (z={self.current_z:.1f})')
        self.ax_3d.grid(False)
        
        self.fig.tight_layout()

# ================================================================
# EXECUÇÃO PRINCIPAL
# ================================================================
if __name__ == "__main__":
    # Criar modelos comparativos
    models = {
        'ΛCDM Padrão': AdvancedCosmology(Om=0.31, Ol=0.69, w=-1.0),
        'Energia Fantasma': AdvancedCosmology(Om=0.29, Ol=0.71, w=-1.1),
        'Quintessência': AdvancedCosmology(Om=0.28, Ol=0.72, w=-0.9),
        'Matéria Extra': AdvancedCosmology(Om=0.32, Ol=0.68, w=-1.0),
        'Consenso Atual': AdvancedCosmology(Om=0.30, Ol=0.70, w=-1.0)
    }
    
    # Iniciar interface interativa
    explorer = CosmologyExplorer(models)
    
    # Manter a janela aberta
    plt.show()