import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy import constants as const
from astropy import units as u

class CosmicDNA:
    def __init__(self, sequence_length=256):
        # Parâmetros cosmológicos reais (Planck 2018)
        self.H0 = 67.4  # km/s/Mpc - Constante de Hubble
        self.Ω_m = 0.311  # Densidade de matéria (bariônica + escura)
        self.Ω_Λ = 0.6889  # Densidade de energia escura
        self.Ω_r = 9.2e-5  # Densidade de radiação
        self.T_CMB = 2.725  # K - Temperatura atual da CMB
        
        # Bases cósmicas com proporções realistas
        self.bases = ['G', 'C', 'A', 'T'] + ['Ψ']*5 + ['Φ']*3 + ['Ω']*30 + ['Λ']*68
        self.base_pairs = {'G': 'C', 'Ψ': 'Φ', 'Ω': 'Λ', 'A': 'T'}
        
        # Inicialização com proporções observadas
        self.sequence = list(np.random.choice(
            ['Ω']*int(0.27*sequence_length) + 
            ['Λ']*int(0.68*sequence_length) + 
            ['A']*int(0.05*sequence_length) + 
            ['Ψ', 'Φ', 'G', 'C']*5,
            size=sequence_length
        ))
        
        # Mapeamento de cores para visualização
        self.color_map = {
            'G': '#FFD700', 'C': '#32CD32',  # Bases convencionais
            'Ψ': '#9400D3', 'Φ': '#4B0082',  # Forças quânticas e consciência
            'Ω': '#1E90FF', 'Λ': '#00BFFF',  # Matéria e energia escura
            'A': '#FF8C00', 'T': '#FF4500'   # Matéria bariônica e espaço-tempo
        }
        
        # Estado termodinâmico inicial (Big Bang)
        self.temperature = 1e32  # Temperatura de Planck (K)
        self.entropy = 0
        self.time = 0  # Tempo cósmico (anos)
        self.scale_factor = 1e-30  # Fator de escala inicial
        
        # Parâmetros de evolução
        self.redshift = 1100  # Redshift inicial (época da recombinação)
        self.structure_formation_started = False
        
    def friedmann_equation(self, t, a):
        """Equação de Friedmann para expansão cósmica"""
        H = self.H0 * np.sqrt(
            self.Ω_r / a**4 + 
            self.Ω_m / a**3 + 
            self.Ω_Λ
        )
        return H
    
    def update_cosmic_evolution(self, dt=1e7):
        """Atualiza a evolução cósmica usando ΛCDM"""
        # Resolver equação de Friedmann
        sol = solve_ivp(self.friedmann_equation, [self.time, self.time + dt], 
                        [self.scale_factor], method='RK45')
        
        # Atualizar parâmetros
        self.scale_factor = sol.y[0, -1]
        self.time += dt
        
        # Calcular redshift
        self.redshift = (1 / self.scale_factor) - 1
        
        # Atualizar temperatura (resfriamento adiabático)
        self.temperature = self.T_CMB * (1 + self.redshift)
        
        # Atualizar entropia (proporcional ao volume cósmico)
        self.entropy = np.log(self.scale_factor**3) if self.scale_factor > 0 else 0
        
        # Disparar formação de estruturas em z≈10
        if self.redshift < 10 and not self.structure_formation_started:
            self.trigger_structure_formation()
            self.structure_formation_started = True
    
    def trigger_structure_formation(self):
        """Inicia a formação de estruturas cósmicas (galáxias, aglomerados)"""
        # Padrões associados à formação de estruturas
        structure_patterns = ['GΩA', 'ΩΩG', 'AΩA', 'TΩT']
        
        # Inserir padrões na sequência
        for pattern in structure_patterns:
            insert_pos = np.random.randint(0, len(self.sequence)-3)
            self.sequence[insert_pos:insert_pos+3] = list(pattern)
    
    def baryon_acoustic_oscillations(self):
        """Simula o efeito das oscilações acústicas de bárions (BAO)"""
        # BAO ocorrem em escalas de ~150 Mpc
        pattern_length = max(3, int(0.05 * len(self.sequence))
        
        # Criar padrão oscilatório
        oscillation = ['A', 'T', 'G'] * (pattern_length // 3)
        
        # Inserir em posição aleatória
        insert_pos = np.random.randint(0, len(self.sequence)-pattern_length)
        self.sequence[insert_pos:insert_pos+pattern_length] = oscillation
    
    def quantum_fluctuations(self):
        """Aplica flutuações quânticas baseadas na temperatura"""
        # Taxa de flutuação proporcional à temperatura
        fluctuation_rate = min(0.5, self.temperature / 1e12)
        
        for i in range(len(self.sequence)):
            if np.random.random() < fluctuation_rate:
                # Preserva o emparelhamento de bases
                if self.sequence[i] in self.base_pairs:
                    self.sequence[i] = np.random.choice(
                        [b for b in self.bases if b != self.sequence[i]]
                    )
    
    def supernova_feedback(self):
        """Efeito de feedback de supernovas na formação estelar"""
        # Ocorre quando estrelas massivas morrem
        if 'A' in self.sequence and 'G' in self.sequence:
            # Substitui matéria por campos de radiação
            a_positions = [i for i, b in enumerate(self.sequence) if b == 'A']
            if a_positions:
                idx = np.random.choice(a_positions)
                self.sequence[idx] = 'Ψ'  # Forças quânticas/radiação
    
    def agn_feedback(self):
        """Feedback de Núcleos Galácticos Ativos (AGNs)"""
        # Disparado por buracos negros supermassivos (padrões específicos)
        if 'ΩΦΩ' in ''.join(self.sequence) or 'ΛGΛ' in ''.join(self.sequence):
            # Remove matéria próxima
            for i in range(len(self.sequence)):
                if self.sequence[i] == 'A' and np.random.random() < 0.3:
                    self.sequence[i] = 'Λ'  # Conversão em energia escura
    
    def evolve(self, time_step=1e8):
        """Evolui o sistema cósmico através do tempo"""
        # Atualizar parâmetros cósmicos
        self.update_cosmic_evolution(time_step)
        
        # Aplicar processos físicos em diferentes eras
        if self.redshift > 3000:  # Era da Radiação
            self.quantum_fluctuations()
        elif 3000 > self.redshift > 10:  # Era da Matéria
            if np.random.random() < 0.2:
                self.baryon_acoustic_oscillations()
        else:  # Era da Energia Escura
            self.supernova_feedback()
            if np.random.random() < 0.1:
                self.agn_feedback()
    
    def plot_cosmic_evolution(self):
        """Visualiza a evolução das características cósmicas"""
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Dados de composição
        time_points = np.logspace(3, 10, 100)  # De 1,000 anos a 10 bilhões de anos
        Ω_m = [self.Ω_m / a**3 for a in np.linspace(1e-3, 1, 100)]
        Ω_Λ = [self.Ω_Λ for _ in range(100)]
        Ω_r = [self.Ω_r / a**4 for a in np.linspace(1e-3, 1, 100)]
        
        # Plot composição
        ax1.plot(time_points, Ω_m, 'b-', label='Matéria (Ω_m)')
        ax1.plot(time_points, Ω_Λ, 'r-', label='Energia Escura (Ω_Λ)')
        ax1.plot(time_points, Ω_r, 'g-', label='Radiação (Ω_r)')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Tempo (anos)')
        ax1.set_ylabel('Densidade de Energia')
        ax1.set_title('Evolução Cósmica ΛCDM')
        ax1.grid(True, which="both", ls="-")
        ax1.legend()
        
        # Segundo eixo para temperatura
        ax2 = ax1.twinx()
        temperatures = [self.T_CMB / a for a in np.linspace(1e-3, 1, 100)]
        ax2.plot(time_points, temperatures, 'm--', label='Temperatura (K)')
        ax2.set_ylabel('Temperatura (K)', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def plot_fractal_dna(self):
        """Visualização fractal do DNA cósmico com cores baseadas em ΛCDM"""
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Parâmetros cosmológicos para escalonamento
        z = self.redshift
        a = self.scale_factor
        
        for depth in range(7):  # 7 níveis fractais
            scale = 0.6**depth * np.log(1 + z)
            for i, base in enumerate(self.sequence[:100]):  # Limitar a 100 bases para visualização
                # Coordenadas fractalmente aninhadas
                theta = 4 * np.pi * i/len(self.sequence) + depth*np.pi/3
                r = scale * (1 + 0.2*np.sin(i/10))  # Adicionar oscilações BAO
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = depth * 2 * a  # Escalonado pelo fator de expansão
                
                # Tamanho do ponto baseado na importância cosmológica
                size_map = {'Ω': 300, 'Λ': 250, 'A': 200, 'Ψ': 150, 'Φ': 150, 'G': 100, 'C': 100, 'T': 100}
                size = size_map.get(base, 100) * scale
                
                ax.scatter(x, y, z, s=size, c=self.color_map.get(base, 'gray'), alpha=0.8, depthshade=False)
        
        ax.set_title(f'DNA Cósmico Fractal (z = {z:.1f}, a = {a:.3f})', fontsize=16)
        ax.set_xlabel('Eixo X (BAO Scale)')
        ax.set_ylabel('Eixo Y (BAO Scale)')
        ax.set_zlabel('Expansão Cósmica')
        
        # Adicionar legenda
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_map['Ω'], markersize=10, label='Matéria Escura'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_map['Λ'], markersize=10, label='Energia Escura'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_map['A'], markersize=10, label='Matéria Bariônica'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_map['Ψ'], markersize=10, label='Forças Quânticas'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.color_map['Φ'], markersize=10, label='Consciência')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.show()
    
    def analyze_composition(self):
        """Analisa a composição do DNA cósmico e compara com observações"""
        counts = {base: self.sequence.count(base) for base in set(self.sequence)}
        total = len(self.sequence)
        
        # Resultados simulados
        sim_dark_matter = counts.get('Ω', 0) / total
        sim_dark_energy = counts.get('Λ', 0) / total
        sim_baryonic = counts.get('A', 0) / total
        
        # Dados observacionais (Planck)
        obs_dark_matter = 0.27
        obs_dark_energy = 0.689
        obs_baryonic = 0.05
        
        # Plot comparativo
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Matéria Escura', 'Energia Escura', 'Matéria Bariônica']
        sim_values = [sim_dark_matter, sim_dark_energy, sim_baryonic]
        obs_values = [obs_dark_matter, obs_dark_energy, obs_baryonic]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, sim_values, width, label='Simulação')
        ax.bar(x + width/2, obs_values, width, label='Observações (Planck)')
        
        ax.set_ylabel('Fração do Universo')
        ax.set_title('Comparação com Dados Observacionais')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        plt.show()
        
        return {
            'simulated': {
                'dark_matter': sim_dark_matter,
                'dark_energy': sim_dark_energy,
                'baryonic_matter': sim_baryonic
            },
            'observed': {
                'dark_matter': obs_dark_matter,
                'dark_energy': obs_dark_energy,
                'baryonic_matter': obs_baryonic
            }
        }

# Simulação principal
if __name__ == "__main__":
    # Inicializar o universo com parâmetros reais
    cosmos = CosmicDNA(sequence_length=1000)
    
    print("=== Evolução Cósmica com Parâmetros ΛCDM Reais ===")
    print(f"Parâmetros iniciais: H0 = {cosmos.H0} km/s/Mpc, Ω_m = {cosmos.Ω_m}, Ω_Λ = {cosmos.Ω_Λ}")
    
    # Evoluir através das eras cósmicas
    cosmic_eras = [
        ("Era de Planck", 1e-43, 1e-32),
        ("Era da Inflação", 1e-32, 1e-30),
        ("Era da Radiação", 1e-30, 1e12),
        ("Era da Matéria", 1e12, 4e17),
        ("Era da Energia Escura", 4e17, 4.3e17)
    ]
    
    for era_name, start_time, end_time in cosmic_eras:
        print(f"\n--- {era_name} ({start_time:.1e} - {end_time:.1e} anos) ---")
        steps = 5
        time_step = (end_time - start_time) / steps
        
        for step in range(steps):
            cosmos.evolve(time_step)
            state = cosmos.sequence[:10]  # Amostra da sequência
            print(f"Passo {step+1}: z = {cosmos.redshift:.1f}, T = {cosmos.temperature:.2e} K, a = {cosmos.scale_factor:.2e}")
            print(f"Amostra de DNA: {''.join(state)}...")
    
    # Resultados finais
    print("\n=== Resultados Finais ===")
    print(f"Tempo cósmico atual: {cosmos.time:.2e} anos")
    print(f"Redshift atual: z = {cosmos.redshift:.4f}")
    print(f"Temperatura CMB: {cosmos.temperature:.4f} K")
    
    # Análise comparativa
    composition = cosmos.analyze_composition()
    print("\nComposição do Universo:")
    print(f"Matéria Escura: Simulada={composition['simulated']['dark_matter']:.4f}, Observada={composition['observed']['dark_matter']:.4f}")
    print(f"Energia Escura: Simulada={composition['simulated']['dark_energy']:.4f}, Observada={composition['observed']['dark_energy']:.4f}")
    
    # Visualizações
    cosmos.plot_cosmic_evolution()
    cosmos.plot_fractal_dna()