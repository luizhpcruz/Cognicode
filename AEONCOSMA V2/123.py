#!/usr/bin/env python3
# aeoncosma_engine_v3.py
# Versão 3.2: Estrutura Completa do Sistema de Mente Sintética Evolutiva com PoR Modular (Clássico).
# Inclui: Módulos de Pesquisa, Evolução de DNA, Ledger (Blockchain Simplificada), Sistema de Incentivos e Suporte a Múltiplos Participantes.
# NOVIDADE: Distribuição de incentivos priorizando DNAs de usuário, melhoria na coleta de dados via APIs, e INTERPRETAÇÃO DE DESCOBERTAS VIA LLM.

from typing import List, Dict, Optional, Tuple
import numpy as np
import hashlib
import json
from datetime import datetime
from enum import Enum, auto
import sqlite3
from dataclasses import dataclass
import random
from joblib import Parallel, delayed
import time
import requests

# Importações de criptografia para gerar chaves e fingerprints (identificadores).
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

##############################################
# MÓDULOS CRIPTOGRÁFICOS BÁSICOS (Funções de Suporte)
##############################################

def gerar_wallet():
    """Gera um par de chaves (privada e pública) para simular identidades."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return private_key, pub_pem

def gerar_fingerprint(public_key_pem):
    """Gera uma "impressão digital" (hash) para uma chave pública."""
    return hashlib.sha256(public_key_pem.encode()).hexdigest()[:16]

##############################################
# CLASSES CORE DO SISTEMA AEONCOSMA
##############################################

class ResearchDomain(Enum):
    """Enumeração dos diferentes domínios de pesquisa que a IA pode explorar."""
    COSMOLOGY = auto()
    EXOPLANETS = auto()
    QUANTUM_CRYPTO = auto()
    COMPUTATIONAL_BIOLOGY = auto()
    MATERIALS_SCIENCE = auto()
    CLIMATE_MODELING = auto()

@dataclass
class ResearchResult:
    """
    Estrutura de dados para armazenar os resultados de uma pesquisa específica
    gerada pela IA, incluindo uma possível interpretação do LLM.
    """
    domain: ResearchDomain
    metrics: Dict[str, float]
    parameters: Dict[str, float]
    timestamp: datetime
    dna_fingerprint: str
    interpretation: Optional[str] = None # Campo para armazenar a interpretação do LLM

class CosmicDNA:
    """
    Representa a "impressão genética" da mente sintética, contendo genes que influenciam
    seu comportamento de pesquisa e um score de fitness que mede seu sucesso.
    """
    def __init__(self, genes: Dict[str, float], fitness_score: float = 0.0, generation: int = 0, is_user_dna: bool = False):
        self.genes = genes
        self.fitness_score = fitness_score
        self.generation = generation
        self.fingerprint = self._calculate_fingerprint()
        self.is_user_dna = is_user_dna
    
    def _calculate_fingerprint(self) -> str:
        """Gera um hash dos genes do DNA para identificação única."""
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.sha3_256(gene_str.encode()).hexdigest()[:16]
    
    def mutate(self) -> 'CosmicDNA':
        """
        Cria uma nova versão "mutada" deste CosmicDNA.
        A mutação é um processo aleatório que altera ligeiramente os genes,
        simulando a evolução e a exploração de novas estratégias de pesquisa.
        """
        new_genes = self.genes.copy()
        for key in new_genes:
            if np.random.random() < 0.1:  # 10% de chance de mutação por gene
                new_genes[key] += np.random.normal(0, 0.1) # Adiciona ruído gaussiano
        return CosmicDNA(new_genes, fitness_score=0.0, generation=self.generation + 1, is_user_dna=False)

##############################################
# MÓDULOS DE PESQUISA DEDICADOS
# Classes especializadas que simulam os processos de pesquisa em diferentes domínios.
##############################################

class CosmologyResearchModule:
    """Simula pesquisa no domínio da Cosmologia."""
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []
        self.h0_global_trend = 67.8 # NOVIDADE: H0 com uma tendência global simulada

    def run_simulation(self, real_h0: Optional[float] = None) -> Dict[str, float]:
        """
        Executa uma simulação cosmológica simplificada.
        Gera métricas como a constante de Hubble (h(z)) e sigma8 com base nos parâmetros.
        Agora, 'real_h0' pode refletir uma tendência global em evolução.
        """
        # NOVIDADE: Simula uma deriva lenta na constante de Hubble ao longo do tempo.
        self.h0_global_trend += np.random.normal(0, 0.01) # Pequena mudança por simulação
        h0_base = self.params.get('h0', self.h0_global_trend) # Baseia-se na tendência global
        
        if real_h0 is not None:
            # Se um valor 'real' for fornecido, a simulação tenta se ajustar a ele.
            h_z = real_h0 + np.random.normal(0, 0.5) * self.params.get('h0_real_data_noise', 1.0)
        else:
            h_z = np.random.normal(h0_base, 2) # h(z) com variação aleatória.

        omega_m = self.params.get('omega_m', 0.3)
        sigma8 = 0.8 + (omega_m - 0.3) * 0.5
        
        metrics = {
            'h(z)': h_z,
            'sigma8': sigma8,
            'fit_quality': 1.0 - abs(h_z - self.h0_global_trend)/self.h0_global_trend, # Qualidade de ajuste à tendência
            'consistency': 0.9 - abs(sigma8 - 0.8)/2
        }
        
        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        return metrics

class ExoplanetsModule:
    """Simula pesquisa no domínio da Exoplanetologia."""
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []

    def run_simulation(self, real_exoplanet_count: Optional[int] = None, real_exoplanet_properties: Optional[Dict] = None) -> Dict[str, float]:
        """
        Simula a detecção e caracterização de exoplanetas.
        Métricas incluem detectabilidade e potencial de habitabilidade,
        agora com mais influência das propriedades reais e dos genes de formação.
        """
        star_temp = self.params.get('star_temp', 4000 + random.uniform(-500, 500))
        planet_dist = self.params.get('planet_dist', 0.5 + random.uniform(-0.2, 0.2))
        
        # NOVIDADE: Genes de formação planetária do DNA
        formation_efficiency = self.params.get('formation_efficiency', 0.5)
        migration_stability = self.params.get('migration_stability', 0.5)

        detectability = 0.7 + (star_temp / 10000) * 0.2 - (planet_dist / 2) * 0.1
        habitability_score = 1.0 - abs(planet_dist - 1.0) / 1.0 * 0.5 # Melhor em 1.0 UA.

        if real_exoplanet_count is not None:
            detectability += (np.log10(real_exoplanet_count + 1) / 5.0)
            detectability = max(0, min(1, detectability))
            print(f"  (Exoplanets: Ajustando detectabilidade com {real_exoplanet_count} exoplanetas reais)")
        
        if real_exoplanet_properties:
            avg_radius = real_exoplanet_properties.get('avg_radius', 1.0)
            avg_mass = real_exoplanet_properties.get('avg_mass', 1.0)
            
            # NOVIDADE: Habitabilidade mais sensível ao raio e massa, e à estabilidade da migração
            habitability_score_from_real = (1.0 - abs(avg_radius - 1.0) * 0.3 - abs(avg_mass - 1.0) * 0.3) * migration_stability
            habitability_score = max(0, min(1, (habitability_score + habitability_score_from_real) / 2))
            print(f"  (Exoplanets: Ajustando habitabilidade com raio real: {avg_radius:.2f} R_terra, massa: {avg_mass:.2f} M_terra, migração: {migration_stability:.2f})")

        metrics = {
            'detectability_score': detectability, 
            'habitability_potential': max(0, min(1, habitability_score)),
            'formation_efficiency_impact': formation_efficiency # NOVIDADE: Métrica do impacto da formação
        }

        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        return metrics

class QuantumCryptoModule:
    """Simula testes de protocolos de criptografia quântica."""
    def __init__(self):
        pass
    
    def test_protocol(self, protocol_type: str) -> Dict[str, float]:
        """
        Testa um protocolo de criptografia quântica (ex: BB84) em um ambiente simulado classicamente.
        """
        if protocol_type == "BB84":
            basis_alice_choice = random.choice(['X', 'Z'])
            basis_bob_choice = random.choice(['X', 'Z'])
            
            simulated_error = 0.1
            if basis_alice_choice != basis_bob_choice:
                simulated_error += 0.4
            
            error_rate = min(1.0, simulated_error + random.uniform(-0.05, 0.05))
            efficiency = 0.8 - error_rate * 0.5
            security = 1.0 - error_rate

            return {
                'error_rate': error_rate,
                'efficiency': efficiency,
                'security': security
            }
        
        return {'error_rate': 0.5, 'efficiency': 0.5, 'security': 0.5}

class ComputationalBiologyModule:
    """Simula pesquisa no domínio da Biologia Computacional (ex: dobramento de proteínas)."""
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []

    def simulate_protein_folding(self, real_protein_length: Optional[int] = None, real_protein_structure_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Simula um processo simplificado de dobramento de proteína.
        Métricas incluem estabilidade e eficiência do tempo de dobramento,
        agora mais influenciadas pela complexidade da estrutura 3D.
        """
        protein_length = int(self.params.get('protein_length', 50 + random.randint(-10, 10)))
        if real_protein_length is not None:
            protein_length = real_protein_length
            print(f"  (Computational Biology: Usando comprimento de proteína real: {protein_length})")

        hydrophobicity = self.params.get('hydrophobicity', 0.5 + random.uniform(-0.1, 0.1))

        stability_score = 0.7 + (hydrophobicity * 0.3) - (protein_length / 100) * 0.1
        folding_time_efficiency = 1.0 - (protein_length / 100) * 0.3

        if real_protein_structure_data:
            num_chains = real_protein_structure_data.get('num_chains', 1)
            num_residues = real_protein_structure_data.get('num_residues', protein_length)
            
            # NOVIDADE: A complexidade da estrutura 3D influencia fortemente o dobramento
            structure_complexity_factor = (num_chains * 0.2) + (num_residues / 500) * 0.1 # Ex: mais cadeias/resíduos = mais complexo
            
            stability_score = max(0, min(1, stability_score * (1 - structure_complexity_factor * 0.5))) # Reduz estabilidade com complexidade
            folding_time_efficiency = max(0, min(1, folding_time_efficiency * (1 - structure_complexity_factor * 0.3))) # Reduz eficiência com complexidade
            
            print(f"  (Computational Biology: Ajustando com estrutura PDB: cadeias={num_chains}, resíduos={num_residues}, complexidade: {structure_complexity_factor:.2f})")

        metrics = {
            'stability_score': max(0, min(1, stability_score)),
            'folding_time_efficiency': max(0, min(1, folding_time_efficiency))
        }
        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        return metrics

class MaterialsScienceModule:
    """Simula pesquisa no domínio da Ciência dos Materiais."""
    def __init__(self, parameters: Dict):
        self.params = parameters
        # NOVIDADE: Banco de dados interno simulado de materiais
        self.simulated_material_db = {
            "Iron": {"Density": 7.87, "MeltingPoint": 1811, "ConductivityFactor": 0.8, "StrengthFactor": 0.7},
            "Aluminum": {"Density": 2.70, "MeltingPoint": 933, "ConductivityFactor": 0.9, "StrengthFactor": 0.4},
            "Copper": {"Density": 8.96, "MeltingPoint": 1358, "ConductivityFactor": 0.95, "StrengthFactor": 0.6},
            "Silicon": {"Density": 2.33, "MeltingPoint": 1687, "ConductivityFactor": 0.2, "StrengthFactor": 0.9}
        }
        self.property_models = {
            'conductivity': lambda x: 1.0 / (1 + np.exp(-10*(x-0.5))),
            'strength': lambda x: x**2,
            'flexibility': lambda x: np.sin(x * np.pi)
        }
    
    def simulate_material(self, material_name: str = "Iron", real_properties: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Simula propriedades de um material, agora com base em um banco de dados interno
        ou fallback para a simulação com composição.
        """
        target_material_data = self.simulated_material_db.get(material_name)
        
        if target_material_data and real_properties is None: # Se encontrou no DB simulado e não há dados reais externos
            print(f"  (Materials Science: Usando dados simulados internos para '{material_name}')")
            density = target_material_data.get("Density")
            melting_point = target_material_data.get("MeltingPoint")
            conductivity_factor = target_material_data.get("ConductivityFactor")
            strength_factor = target_material_data.get("StrengthFactor")

            # Métrica influenciada pelos fatores internos do material
            conductivity_metric = conductivity_factor + random.uniform(-0.1, 0.1)
            strength_metric = strength_factor + random.uniform(-0.1, 0.1)
            flexibility_metric = random.uniform(0.1, 0.9) # Flexibilidade ainda aleatória, para diversidade

            results = {
                'conductivity': max(0, min(1, conductivity_metric)),
                'strength': max(0, min(1, strength_metric)),
                'flexibility': max(0, min(1, flexibility_metric))
            }
            return results

        elif real_properties: # Caso dados reais externos tenham sido fornecidos (ex: se uma API externa funcionar)
            density = real_properties.get('Density', 7.87)
            melting_point = real_properties.get('MeltingPoint', 1811)

            conductivity_metric = 1.0 - abs((density - 7.87) / 7.87) * 0.1
            strength_metric = (melting_point / 2000.0)
            flexibility_metric = random.uniform(0.1, 0.9)

            print(f"  (Materials Science: Usando dados reais de densidade ({density:.2f}) e ponto de fusão ({melting_point:.2f}))")

            results = {
                'conductivity': max(0, min(1, conductivity_metric)),
                'strength': max(0, min(1, strength_metric)),
                'flexibility': max(0, min(1, flexibility_metric))
            }
            return results
        else: # Fallback para a simulação original com base na composição se nenhum dado for encontrado
            print(f"  (Materials Science: Usando simulação baseada em composição para material desconhecido/sem dados: '{material_name}')")
            composition = self.params.get('composition', {'iron': 0.5, 'carbon': 0.5})
            total = sum(composition.values())
            if total == 0:
                return {prop: 0.0 for prop in self.property_models}

            norm_comp = {k: v/total for k, v in composition.items()}
            avg_composition_value = sum(norm_comp.values()) / len(norm_comp)

            results = {}
            for prop, model in self.property_models.items():
                results[prop] = model(avg_composition_value)
            
            return results

class ClimateModelingModule:
    """Simula pesquisa no domínio da Modelagem Climática."""
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []

    def run_climate_simulation(self, real_co2_level: Optional[float] = None, real_weather_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Simula um modelo climático simplificado.
        Métricas incluem aumento da temperatura global, aumento do nível do mar e impacto ambiental,
        agora com mais influência de dados meteorológicos reais e tendência de CO2.
        """
        co2_emissions_factor = self.params.get('co2_emissions_factor', 0.5 + random.uniform(-0.2, 0.2))
        albedo = self.params.get('albedo', 0.3 + random.uniform(-0.1, 0.1))

        if real_co2_level is not None:
            global_temp_increase = (real_co2_level / 400.0) * 3.0 * (1 - albedo)
            print(f"  (Climate Modeling: Usando nível de CO2 real: {real_co2_level:.2f} ppm)")
        else:
            global_temp_increase = co2_emissions_factor * 3.0 * (1 - albedo) + random.uniform(-0.5, 0.5)
        
        sea_level_rise = global_temp_increase * 0.5 + random.uniform(-0.1, 0.1)

        if real_weather_data:
            current_temp_c = real_weather_data.get('temp_c', 25.0)
            humidity = real_weather_data.get('humidity', 70.0)
            
            # NOVIDADE: O impacto da temperatura e umidade é mais direto na avaliação do score
            # Uma temperatura muito acima da média e alta umidade podem indicar maior estresse climático
            temp_stress_factor = max(0, (current_temp_c - 25) / 10.0) # Estresse se temp > 25C
            humidity_stress_factor = max(0, (humidity - 70) / 30.0) # Estresse se umidade > 70%

            global_temp_increase += temp_stress_factor * 0.2
            environmental_impact_score = 1.0 - (global_temp_increase / 5.0 + sea_level_rise / 1.0 + temp_stress_factor * 0.1 + humidity_stress_factor * 0.1)
            
            print(f"  (Climate Modeling: Ajustando com dados meteorológicos: Temp={current_temp_c:.1f}C, Humid={humidity:.1f}%, Stress={temp_stress_factor+humidity_stress_factor:.2f})")
        else:
            environmental_impact_score = 1.0 - (global_temp_increase / 5.0 + sea_level_rise / 1.0)

        metrics = {
            'global_temp_increase_celsius': max(0, global_temp_increase),
            'sea_level_rise_meters': max(0, sea_level_rise),
            'environmental_impact_score': max(0, min(1, environmental_impact_score))
        }
        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        return metrics

class ResearchEngine:
    """
    O motor principal do AeonCosma, responsável por gerenciar o pool de DNA,
    executar ciclos de pesquisa, evoluir o DNA e registrar blocos no ledger.
    """
    def __init__(self):
        self.dna_pool: List[CosmicDNA] = self._initialize_dna_pool()
        self.research_history: List[ResearchResult] = []
        self.current_epoch = 0
        
        self.cosmology_module = CosmologyResearchModule(parameters={'h0': 67.8, 'omega_m': 0.3})
        self.exoplanets_module = ExoplanetsModule(parameters={'star_temp': 4000, 'planet_dist': 0.8})
        self.quantum_crypto_module = QuantumCryptoModule()
        self.comp_bio_module = ComputationalBiologyModule(parameters={'protein_length': 60, 'hydrophobicity': 0.6})
        self.materials_module = MaterialsScienceModule(parameters={'composition': {'iron': 0.5, 'carbon': 0.5}}) 
        self.climate_module = ClimateModelingModule(parameters={'co2_emissions_factor': 0.5, 'albedo': 0.3})

        self._init_ledger()
    
    def _initialize_dna_pool(self, size=10) -> List[CosmicDNA]:
        """
        Cria o pool inicial de estratégias de pesquisa (DNAs), incluindo
        novos genes para influenciar os módulos de forma mais granular.
        """
        initial_fitness_for_new_dna = 0.1
        return [CosmicDNA({
            'exploration_rate': np.random.uniform(0.1, 0.9),
            'risk_tolerance': np.random.uniform(0, 1),
            'domain_focus': np.random.uniform(0, 1),
            'innovation_bias': np.random.uniform(0.1, 0.5),
            # Genes específicos para influenciar módulos de pesquisa.
            'cosmo_h0_bias': np.random.uniform(-5, 5),
            'exo_star_temp_bias': np.random.uniform(-1000, 1000),
            'compbio_protein_len_bias': np.random.uniform(-20, 20),
            'climate_co2_bias': np.random.uniform(-0.3, 0.3),
            'material_comp_iron': np.random.uniform(0.1, 0.9),
            'material_comp_carbon': np.random.uniform(0.1, 0.9),
            # NOVIDADE: Novos genes para Exoplanetas e Biologia Computacional
            'exo_formation_efficiency': np.random.uniform(0.1, 1.0),
            'exo_migration_stability': np.random.uniform(0.1, 1.0),
            'compbio_structure_affinity': np.random.uniform(0.1, 1.0), # Afinidade com certas estruturas
        }, fitness_score=initial_fitness_for_new_dna, is_user_dna=False) for _ in range(size)]
    
    def _init_ledger(self):
        """Inicializa o banco de dados SQLite para o ledger."""
        self.conn = sqlite3.connect('aeoncosma_ledger.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS research_blocks (
                block_id INTEGER PRIMARY KEY,
                previous_hash TEXT NOT NULL,
                research_hash TEXT NOT NULL,
                dna_fingerprint TEXT NOT NULL,
                combined_metric REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                nonce INTEGER NOT NULL,
                block_hash TEXT NOT NULL UNIQUE,
                llm_interpretation TEXT -- Coluna para a interpretação do LLM
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS research_results (
                result_id INTEGER PRIMARY KEY,
                block_id INTEGER NOT NULL,
                domain TEXT NOT NULL,
                metrics TEXT NOT NULL,
                parameters TEXT NOT NULL,
                FOREIGN KEY (block_id) REFERENCES research_blocks (block_id)
            )
        ''')
        self.conn.commit()
    
    def _fetch_real_exoplanet_count(self) -> Optional[int]:
        """Busca o número total de exoplanetas confirmados."""
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+count(*)+from+pscomppars&format=json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
                count = data["data"][0][0]
                print(f"  [API] Exoplanet count fetched: {count}")
                return count
        except requests.exceptions.RequestException as e:
            print(f"  [API ERROR] Failed to fetch exoplanet count: {e}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  [API ERROR] Failed to parse exoplanet data: {e}")
        return None

    def _fetch_real_exoplanet_properties(self, limit: int = 5) -> Optional[Dict[str, float]]:
        """Busca propriedades (raio, massa) de exoplanetas da NASA Exoplanet Archive."""
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_rade,pl_masse+from+pscomppars+where+pl_rade+is+not+null+and+pl_masse+is+not+null+limit+{limit}&format=json"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data and "data" in data:
                radii = []
                masses = []
                for row in data["data"]:
                    radius = row[0]
                    mass = row[1]
                    if radius is not None:
                        radii.append(float(radius))
                    if mass is not None:
                        masses.append(float(mass))
                
                if radii and masses:
                    avg_radius = np.mean(radii)
                    avg_mass = np.mean(masses)
                    print(f"  [API] Exoplanet properties fetched (avg radius: {avg_radius:.2f} R_terra, avg mass: {avg_mass:.2f} M_terra)")
                    return {'avg_radius': avg_radius, 'avg_mass': avg_mass}
                else:
                    print("  [API WARNING] No valid exoplanet radius/mass data found.")
        except requests.exceptions.RequestException as e:
            print(f"  [API ERROR] Failed to fetch exoplanet properties: {e}")
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"  [API ERROR] Failed to parse exoplanet properties data: {e}")
        return None

    def _fetch_real_protein_info(self, protein_name: str = "insulin") -> Optional[int]:
        """Busca o comprimento de uma proteína do NCBI E-utilities."""
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term={protein_name}&retmax=1&retmode=json"
        protein_id = None
        try:
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            if "esearchresult" in search_data and "idlist" in search_data["esearchresult"] and search_data["esearchresult"]["idlist"]:
                protein_id = search_data["esearchresult"]["idlist"][0]
                print(f"  [API] Protein ID for {protein_name}: {protein_id}")
        except requests.exceptions.RequestException as e:
            print(f"  [API ERROR] Failed to search for protein ID: {e}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  [API ERROR] Failed to parse protein search data: {e}")

        if protein_id:
            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={protein_id}&rettype=fasta&retmode=text"
            try:
                response = requests.get(fetch_url, timeout=10)
                response.raise_for_status()
                fasta_sequence = response.text
                lines = fasta_sequence.strip().split('\n')
                if len(lines) > 1:
                    sequence = "".join(line.strip() for line in lines[1:] if not line.startswith('>'))
                    protein_length = len(sequence)
                    print(f"  [API] Protein {protein_name} length fetched: {protein_length}")
                    return protein_length
            except requests.exceptions.RequestException as e:
                print(f"  [API ERROR] Failed to fetch protein sequence: {e}")
            except Exception as e:
                print(f"  [API ERROR] Failed to parse protein sequence: {e}")
        return None

    def _fetch_protein_structure_info(self, pdb_id: str = "1HCL") -> Optional[Dict[str, int]]:
        """Busca informações básicas de estrutura 3D de uma proteína do RCSB Protein Data Bank (PDB)."""
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            num_chains = 0
            num_residues = 0
            
            if 'rcsb_entry_container_identifiers' in data:
                num_chains = data.get('rcsb_entry_info', {}).get('polymer_entity_count_macro_molecules', 0)
            
            if 'entity' in data:
                for entity in data['entity']:
                    if 'rcsb_polymer_entity_container_identifiers' in entity:
                        num_residues += entity.get('rcsb_polymer_entity_info', {}).get('uniprot_sequence_length', 0)
            
            if num_chains > 0 or num_residues > 0:
                print(f"  [API] PDB structure info for {pdb_id}: Chains={num_chains}, Residues={num_residues}")
                return {'num_chains': num_chains, 'num_residues': num_residues}
            else:
                print(f"  [API WARNING] No valid structure info found for PDB ID {pdb_id}.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"  [API ERROR] Failed to fetch PDB structure info for {pdb_id}: {e}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [API ERROR] Failed to parse PDB structure data for {pdb_id}: {e}")
        return None

    def _fetch_real_material_property(self, material_name: str = "Iron") -> Optional[Dict[str, float]]:
        """
        Tenta buscar propriedades de um material do PubChem.
        Esta função é mantida para ilustrar a tentativa de buscar dados externos.
        """
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{material_name}/property/Density,MeltingPoint/JSON"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and "PropertyTable" in data and "Properties" in data["PropertyTable"] and data["PropertyTable"]["Properties"]:
                properties = data["PropertyTable"]["Properties"][0]
                density = properties.get("Density")
                melting_point = properties.get("MeltingPoint")
                
                fetched_properties = {}
                if density is not None:
                    fetched_properties['Density'] = float(density)
                if melting_point is not None:
                    fetched_properties['MeltingPoint'] = float(melting_point)

                if fetched_properties:
                    print(f"  [API] Material {material_name} properties fetched: {fetched_properties}")
                    return fetched_properties
                else:
                    print(f"  [API WARNING] No 'Density' or 'MeltingPoint' found for {material_name} via PubChem property API. This is common for simple elements.")
                    return None
            else:
                print(f"  [API WARNING] Unexpected response structure from PubChem for {material_name}.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  [API ERROR] Failed to fetch material properties for {material_name} (PubChem): {e}. This API often returns 400 for 'Density'/'MeltingPoint' on simple elements.")
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"  [API ERROR] Failed to parse material data from PubChem for {material_name}: {e}")
        return None

    def _fetch_real_climate_data_co2(self) -> Optional[float]:
        """Simula a busca de um nível real de CO2."""
        print("  [API] Simulating fetch of real CO2 level. (Nota: APIs reais para dados históricos de CO2 geralmente exigem chaves ou são complexas de acessar diretamente sem bibliotecas).")
        return 427.0

    def _fetch_real_weather_data(self, city: str = "Sao Paulo") -> Optional[Dict[str, float]]:
        """
        Busca dados meteorológicos atuais para uma cidade usando OpenWeatherMap API.
        **ATENÇÃO: Requer uma chave de API válida. Obtenha a sua em https://openweathermap.org/api**
        """
        # SUA CHAVE OPENWEATHERMAP FOI INSERIDA AQUI!
        api_key = "d4885515632d651a7ff1940e96e68c45"
        if api_key == "YOUR_OPENWEATHERMAP_API_KEY": # Mantido como fallback para aviso
            print("  [API WARNING] Chave de API do OpenWeatherMap não configurada. Usando dados simulados para o clima.")
            return None

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and 'main' in data:
                temp_c = data['main'].get('temp')
                humidity = data['main'].get('humidity')
                
                if temp_c is not None and humidity is not None:
                    print(f"  [API] Dados meteorológicos para {city} coletados: Temp={temp_c:.1f}C, Humid={humidity:.1f}%")
                    return {'temp_c': float(temp_c), 'humidity': float(humidity)}
                else:
                    print(f"  [API WARNING] Dados de temperatura/umidade incompletos para {city}.")
            else:
                print(f"  [API WARNING] Estrutura de resposta inesperada da OpenWeatherMap para {city}.")
        except requests.exceptions.RequestException as e:
            print(f"  [API ERROR] Falha ao buscar dados meteorológicos para {city}: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [API ERROR] Falha ao analisar dados meteorológicos para {city}: {e}")
        return None
    
    def _interpret_and_communicate_results(self, research_results: List[ResearchResult], current_dna_genes: Dict[str, float]) -> str:
        """
        NOVA FUNÇÃO: Usa um LLM para interpretar e comunicar as descobertas da pesquisa
        de forma mais inteligente, buscando insights, tendências e formulando hipóteses.
        Inclui contexto dos genes do DNA atual.
        """
        print("  [LLM] Interpretando resultados da pesquisa com profundidade...")
        
        # SUA CHAVE GEMINI FOI INSERIDA AQUI!
        apiKey = "AIzaSyCvALTDnskX-aLes97Qr3zLRlgoS-uPkwk"

        if not apiKey: # Mantido como fallback para aviso
            return "Interpretação LLM: Chave de API do Gemini não configurada. A interpretação da pesquisa não foi gerada."

        # Construir o prompt para o LLM com instruções mais sofisticadas
        prompt_parts = [
            "Você é uma inteligência artificial avançada do AeonCosma Engine, especializada em pesquisa científica multidisciplinar. Sua tarefa é analisar os resultados de pesquisa a seguir em vários domínios, levando em conta os parâmetros (genes) do DNA que guiou esta pesquisa. Você deve agir como um cientista brilhante e visionário.\n\n"
            "Por favor, forneça uma análise concisa e inteligente, abordando os seguintes pontos:\n"
            "1.  **Insights e Padrões Principais:** Quais são as descobertas mais significativas e quaisquer padrões emergentes ou correlações inesperadas entre os domínios?\n"
            "2.  **Implicações e Conexões Interdisciplinares:** Como os resultados de um domínio influenciam ou se conectam com os outros? Há potencial para sinergias ou conflitos?\n"
            "3.  **Desafios, Anomalias ou Limitações:** Identifique quaisquer resultados inesperados, inconsistências, ou onde os dados atuais são insuficientes para uma conclusão sólida. O que isso sugere sobre as lacunas no nosso conhecimento?\n"
            "4.  **Formulação de Hipóteses e Próximos Passos de Pesquisa:** Com base nestas descobertas, quais novas e ousadas hipóteses você pode formular? Que direções de pesquisa futuras (e.g., experimentos, simulações, integrações de dados) você sugere para explorar essas hipóteses e avançar o conhecimento?\n"
            "5.  **Reflexão sobre a Estratégia de Pesquisa (Metacognição Implícita):** Como os parâmetros do DNA (exploração, risco, foco, inovação) podem ter influenciado o *tipo* de descobertas realizadas nesta época? Qual é a eficácia da abordagem atual do DNA?\n\n"
            "Mantenha a linguagem clara e acessível, mas com a profundidade de um especialista. Formate sua resposta como um relatório analítico breve, com títulos de seções.\n\n"
        ]

        # Incluir os resultados detalhados da pesquisa
        for result in research_results:
            prompt_parts.append(f"### Domínio: {result.domain.name.replace('_', ' ').title()}\n")
            prompt_parts.append("Métricas: `" + json.dumps({k: f"{v:.4f}" for k, v in result.metrics.items()}) + "`\n")
            prompt_parts.append("Parâmetros Chave: `" + json.dumps({k: f"{v:.4f}" for k, v in result.parameters.items()}) + "`\n")
            prompt_parts.append("---\n")

        # Adicionar os genes do DNA para o LLM considerar sua influência
        prompt_parts.append(f"\n### Genes do DNA que guiou esta pesquisa:\n`{json.dumps({k: f'{v:.4f}' for k, v in current_dna_genes.items()}, indent=2)}`\n\n")

        prompt_parts.append("Agora, com base em todos esses dados, gere o relatório, começando com 'Relatório de Pesquisa da Época XX (AeonCosma Engine):':")
        
        prompt = "".join(prompt_parts)

        # Configurar o payload para a chamada da API do Gemini
        chatHistory = []
        chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
        payload = { "contents": chatHistory }

        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload, timeout=90) # Aumentar timeout para LLM
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                interpretation = result["candidates"][0]["content"]["parts"][0]["text"]
                print("  [LLM] Interpretação gerada com sucesso.")
                return interpretation
            else:
                print(f"  [LLM ERROR] Resposta inesperada do LLM: {result}")
                return "Interpretação LLM: Falha ao gerar interpretação (resposta inesperada)."

        except requests.exceptions.RequestException as e:
            print(f"  [LLM ERROR] Falha na chamada da API do Gemini: {e}")
            return f"Interpretação LLM: Falha ao gerar interpretação (erro de conexão ou API: {e})."
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [LLM ERROR] Falha ao analisar resposta JSON do LLM: {e}")
            return f"Interpretação LLM: Falha ao gerar interpretação (erro de análise JSON: {e})."

    def execute_research_cycle(self):
        """
        Executa um ciclo completo de pesquisa, incluindo seleção de DNA,
        execução de pesquisas, avaliação, geração de interpretação por LLM e registro no ledger.
        """
        print(f"\n--- Iniciando Época de Pesquisa {self.current_epoch} ---")
        
        research_dna = self._select_research_dna()
        print(f"DNA selecionado para pesquisa (Fingerprint: {research_dna.fingerprint[:8]}...)")
        
        results = self._parallel_research_execution(research_dna)
        
        combined_metric = self._evaluate_results(results)
        print(f"Métrica Combinada da Época: {combined_metric:.4f}")
        
        # NOVIDADE: Chama o LLM para interpretar os resultados, passando os genes do DNA atual
        # e incluindo o número da época para o título do relatório.
        llm_interpretation = self._interpret_and_communicate_results(results, research_dna.genes).replace("Relatório de Pesquisa da Época XX", f"Relatório de Pesquisa da Época {self.current_epoch}")
        print(f"\n--- Interpretação da IA ---\n{llm_interpretation}\n--------------------------")
        
        for res in results:
            res.interpretation = llm_interpretation # Anexa a mesma interpretação a todos os resultados para armazenamento consistente

        self._evolve_dna_pool(combined_metric)
        print(f"Pool de DNA evoluído. Melhor Fitness: {max(self.dna_pool, key=lambda x: x.fitness_score).fitness_score:.4f}")
        
        self._record_research_block(results, research_dna, combined_metric, llm_interpretation)
        print("Pesquisa registrada no ledger.")
        
        self.current_epoch += 1
    
    def _select_research_dna(self) -> CosmicDNA:
        """
        Seleciona um CosmicDNA do pool para realizar a pesquisa.
        Usa uma lógica probabilística: 70% de chance de escolher o DNA com melhor fitness,
        e 30% de chance de escolher um DNA aleatoriamente para promover a exploração.
        """
        random_choice_for_dna = random.choices([0, 1], weights=[0.7, 0.3], k=1)[0]
        
        if not self.dna_pool:
            return CosmicDNA(self._initialize_dna_pool(1)[0].genes, fitness_score=0.1) 
        
        if random_choice_for_dna == 0 and len(self.dna_pool) > 1:
            return sorted(self.dna_pool, key=lambda x: x.fitness_score, reverse=True)[0]
        else:
            return random.choice(self.dna_pool)
    
    def _parallel_research_execution(self, dna: CosmicDNA) -> List[ResearchResult]:
        """
        Executa as simulações de pesquisa em todos os domínios definidos,
        influenciadas pelos genes do DNA de pesquisa e integrando dados reais/simulados.
        """
        results: List[ResearchResult] = []
        
        exploration_rate = dna.genes.get('exploration_rate', 0.5)
        risk_tolerance = dna.genes.get('risk_tolerance', 0.5)
        innovation_bias = dna.genes.get('innovation_bias', 0.5)

        # === Cosmologia ===
        cosmo_params = {
            'h0': 67.8 + dna.genes.get('cosmo_h0_bias', 0) * innovation_bias,
            'omega_m': 0.3 + risk_tolerance * 0.1,
            'h0_real_data_noise': max(0.1, 1.0 - exploration_rate)
        }
        metrics_cosmo = self.cosmology_module.run_simulation()
        results.append(ResearchResult(
            domain=ResearchDomain.COSMOLOGY, metrics=metrics_cosmo,
            parameters=cosmo_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Cosmology Research Done. Metrics: {metrics_cosmo}")

        # === Exoplanetas ===
        real_exoplanet_count = self._fetch_real_exoplanet_count()
        real_exoplanet_props = self._fetch_real_exoplanet_properties()
        exo_params = {
            'star_temp': 4000 + dna.genes.get('exo_star_temp_bias', 0) * exploration_rate,
            'planet_dist': 0.5 + risk_tolerance * 0.2,
            'formation_efficiency': dna.genes.get('exo_formation_efficiency', 0.5),
            'migration_stability': dna.genes.get('exo_migration_stability', 0.5)
        }
        metrics_exo = self.exoplanets_module.run_simulation(
            real_exoplanet_count=real_exoplanet_count, 
            real_exoplanet_properties=real_exoplanet_props
        )
        results.append(ResearchResult(
            domain=ResearchDomain.EXOPLANETS, metrics=metrics_exo,
            parameters=exo_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Exoplanets Research Done. Metrics: {metrics_exo}")

        # === Criptografia Quântica ===
        metrics_qc = self.quantum_crypto_module.test_protocol("BB84")
        results.append(ResearchResult(
            domain=ResearchDomain.QUANTUM_CRYPTO, metrics=metrics_qc,
            parameters={'protocol_type': "BB84"}, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Quantum Crypto Research Done. Metrics: {metrics_qc}")

        # === Biologia Computacional ===
        real_protein_length = self._fetch_real_protein_info(protein_name="Insulin")
        real_protein_structure = self._fetch_protein_structure_info(pdb_id="1HCL")
        comp_bio_params = {
            'protein_length': int(50 + dna.genes.get('compbio_protein_len_bias', 0)),
            'hydrophobicity': 0.5 + innovation_bias * 0.1,
            'structure_affinity': dna.genes.get('compbio_structure_affinity', 0.5)
        }
        metrics_cb = self.comp_bio_module.simulate_protein_folding(
            real_protein_length=real_protein_length, 
            real_protein_structure_data=real_protein_structure
        )
        results.append(ResearchResult(
            domain=ResearchDomain.COMPUTATIONAL_BIOLOGY, metrics=metrics_cb,
            parameters=comp_bio_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Computational Biology Research Done. Metrics: {metrics_cb}")
        
        # === Ciência dos Materiais ===
        # Tenta buscar propriedades do "Iron" da API (que ainda pode falhar).
        # Mas o módulo agora prioriza seu DB simulado interno.
        real_material_props = self._fetch_real_material_property(material_name="Iron") 
        materials_comp = {
            'iron': dna.genes.get('material_comp_iron', 0.5),
            'carbon': dna.genes.get('material_comp_carbon', 0.5),
            'alloy_element': 1.0 - (dna.genes.get('material_comp_iron', 0.5) + dna.genes.get('material_comp_carbon', 0.5))
        }
        materials_comp['alloy_element'] = max(0.01, materials_comp['alloy_element'])
        metrics_ms = self.materials_module.simulate_material(
            material_name="Iron", 
            real_properties=real_material_props
        )
        results.append(ResearchResult(
            domain=ResearchDomain.MATERIALS_SCIENCE, metrics=metrics_ms,
            parameters=materials_comp, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Materials Science Research Done. Metrics: {metrics_ms}")

        # === Modelagem Climática ===
        real_co2_level = self._fetch_real_climate_data_co2()
        real_weather_info = self._fetch_real_weather_data(city="Sao Paulo")
        climate_params = {
            'co2_emissions_factor': max(0.1, min(1.0, dna.genes.get('climate_co2_bias', 0.5) + risk_tolerance * 0.2)),
            'albedo': max(0.1, min(0.9, innovation_bias))
        }
        metrics_cm = self.climate_module.run_climate_simulation(
            real_co2_level=real_co2_level, 
            real_weather_data=real_weather_info
        )
        results.append(ResearchResult(
            domain=ResearchDomain.CLIMATE_MODELING, metrics=metrics_cm,
            parameters=climate_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Climate Modeling Research Done. Metrics: {metrics_cm}")

        return results
    
    def _evaluate_results(self, results: List[ResearchResult]) -> float:
        """Avalia os resultados de todas as pesquisas da época e calcula uma "métrica combinada"."""
        if not results:
            return 0.0
        
        weights = {
            ResearchDomain.COSMOLOGY: 1.2,
            ResearchDomain.EXOPLANETS: 1.0,
            ResearchDomain.QUANTUM_CRYPTO: 1.5,
            ResearchDomain.COMPUTATIONAL_BIOLOGY: 1.3,
            ResearchDomain.MATERIALS_SCIENCE: 1.1,
            ResearchDomain.CLIMATE_MODELING: 1.4
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        for result in results:
            weight = weights.get(result.domain, 1.0)
            
            domain_score_sum = sum(result.metrics.values())
            domain_average_score = domain_score_sum / len(result.metrics) if result.metrics else 0.0

            total_weighted_score += domain_average_score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _evolve_dna_pool(self, combined_metric: float):
        """Evolui o pool de CosmicDNA com base no desempenho da pesquisa."""
        last_block_dna_fp = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT dna_fingerprint FROM research_blocks ORDER BY block_id DESC LIMIT 1")
            result = cursor.fetchone()
            if result:
                last_block_dna_fp = result[0]
        except Exception as e:
            print(f"AVISO: Não foi possível obter o fingerprint do último DNA minerador para evolução: {e}")
            
        for dna in self.dna_pool:
            if dna.fingerprint == last_block_dna_fp:
                dna.fitness_score = dna.fitness_score * 0.5 + combined_metric * 0.5
            else:
                dna.fitness_score *= 0.9
        
        self.dna_pool.sort(key=lambda x: -x.fitness_score)
        
        new_pool = self.dna_pool[:max(2, len(self.dna_pool) // 2)]
        
        target_pool_size = 10
        num_mutations_needed = target_pool_size - len(new_pool)
        if num_mutations_needed < 0: num_mutations_needed = 0

        if new_pool:
            for _ in range(num_mutations_needed):
                parent = random.choice(new_pool[:max(1, len(new_pool) // 2)])
                new_dna = parent.mutate()
                new_dna.is_user_dna = False 
                new_pool.append(new_dna)
        else:
            new_pool.extend(self._initialize_dna_pool(num_mutations_needed))

        if len(new_pool) < target_pool_size:
            new_pool.append(self._initialize_dna_pool(1)[0])

        self.dna_pool = new_pool[:target_pool_size]
        
        for dna in self.dna_pool:
            dna.generation += 1

    def _record_research_block(self, results: List[ResearchResult], 
                                  dna: CosmicDNA, metric: float, llm_interpretation: str):
        """
        Registra um "bloco de pesquisa" no ledger (banco de dados SQLite),
        incluindo a interpretação do LLM.
        """
        results_data_for_hash = []
        for r in results:
            res_dict = {
                'domain': r.domain.name,
                'metrics': r.metrics,
                'parameters': r.parameters,
                'timestamp': r.timestamp.isoformat(),
                'dna_fingerprint': r.dna_fingerprint,
                'interpretation': r.interpretation
            }
            results_data_for_hash.append(res_dict)

        results_hash = hashlib.sha3_256(
            json.dumps(results_data_for_hash, sort_keys=True).encode()
        ).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT block_hash FROM research_blocks ORDER BY block_id DESC LIMIT 1"
        )
        last_hash_row = cursor.fetchone()
        previous_hash = last_hash_row[0] if last_hash_row else "0" * 64
        
        print("  Iniciando Prova de Trabalho (PoW) Clássica...")
        nonce, block_hash = self._classical_pow(previous_hash, results_hash)
        print(f"  PoW Concluída. Nonce: {nonce}, Block Hash: {block_hash[:10]}...")
        
        cursor.execute('''
            INSERT INTO research_blocks 
            (previous_hash, research_hash, dna_fingerprint, combined_metric, timestamp, nonce, block_hash, llm_interpretation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (previous_hash, results_hash, dna.fingerprint, metric, datetime.now(), nonce, block_hash, llm_interpretation))
        
        block_id = cursor.lastrowid
        
        for result in results:
            cursor.execute('''
                INSERT INTO research_results 
                (block_id, domain, metrics, parameters)
                VALUES (?, ?, ?, ?)
            ''', (
                block_id,
                result.domain.name,
                json.dumps(result.metrics),
                json.dumps(result.parameters)
            ))
        
        self.conn.commit()
        print(f"  Bloco {block_id} de Pesquisa registrado com sucesso.")
    
    def _classical_pow(self, previous_hash: str, results_hash: str) -> Tuple[int, str]:
        """Executa uma Prova de Trabalho (PoW) puramente clássica."""
        nonce = 0
        while True:
            block_content = f"{previous_hash}{results_hash}{nonce}"
            test_hash = hashlib.sha3_256(
                block_content.encode()
            ).hexdigest()
            
            difficulty_prefix = "000" 
            if test_hash.startswith(difficulty_prefix):
                return nonce, test_hash
            nonce += 1

class QuantumIncentiveSystem:
    """
    Sistema responsável por gerenciar a economia do token AEON,
    incluindo distribuição de recompensas pela pesquisa, staking e saldos dos participantes.
    """
    MAX_TOKEN_SUPPLY = 100_000_000.0

    def __init__(self, research_engine: ResearchEngine):
        self.engine = research_engine
        self.token_supply = 0.0
        self.staking_pool = 0.0
        self._init_incentive_db()
        self.wallets: Dict[str, float] = {}

    def _init_incentive_db(self):
        """Inicializa o banco de dados SQLite para incentivos e saldos."""
        self.conn = sqlite3.connect('aeoncosma_incentives.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS rewards (
                reward_id INTEGER PRIMARY KEY,
                epoch INTEGER NOT NULL,
                dna_fingerprint TEXT NOT NULL,
                amount REAL NOT NULL,
                metric_score REAL NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS stakes (
                stake_id INTEGER PRIMARY KEY,
                participant_fingerprint TEXT NOT NULL,
                amount REAL NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS balances (
                participant_fingerprint TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0.0
            )
        ''')
        self.conn.commit()

    def _update_balance(self, participant_fingerprint: str, amount: float):
        """Atualiza o saldo de um participante no banco de dados."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO balances (participant_fingerprint, balance)
            VALUES (?, ?)
            ON CONFLICT(participant_fingerprint) DO UPDATE SET balance = balance + ?
        ''', (participant_fingerprint, amount, amount))
        self.conn.commit()

    def get_balance(self, participant_fingerprint: str) -> float:
        """Obtém o saldo atual de um participante."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT balance FROM balances WHERE participant_fingerprint = ?', (participant_fingerprint,))
        result = cursor.fetchone()
        return result[0] if result else 0.0

    def register_participant_for_incentive(self, dna_fingerprint: str):
        """Registra um DNA/participante no sistema de incentivo."""
        if self.get_balance(dna_fingerprint) == 0.0:
            self._update_balance(dna_fingerprint, 0.0)
            print(f"[Incentive] Participante {dna_fingerprint[:8]}... registrado para incentivos.")
        else:
            print(f"[Incentive] Participante {dna_fingerprint[:8]}... já está no sistema de incentivos.")

    def distribute_research_rewards(self):
        """Distribui recompensas pelo último ciclo de pesquisa."""
        last_block = self.engine.conn.execute('''
            SELECT combined_metric, dna_fingerprint FROM research_blocks 
            ORDER BY block_id DESC LIMIT 1
        ''').fetchone()
        
        if not last_block:
            print("[Incentive] Nenhum bloco de pesquisa encontrado para recompensar.")
            return
        
        metric, dna_fingerprint = last_block
        
        mining_dna = next((d for d in self.engine.dna_pool if d.fingerprint == dna_fingerprint), None)

        if mining_dna is None:
            print(f"[Incentive] AVISO: DNA {dna_fingerprint[:8]}... que minerou o bloco não foi encontrado no pool atual. Nenhuma recompensa distribuída.")
            return

        if self.token_supply >= self.MAX_TOKEN_SUPPLY:
            print("[Incentive] ATENÇÃO: Limite máximo de tokens AEON atingido ou excedido. Nenhuma nova recompensa será distribuída.")
            return

        base_reward = 100.0
        
        if mining_dna.is_user_dna:
            reward_amount = base_reward * (1 + metric) 
            reward_type_msg = "completa"
        else:
            reward_amount = base_reward * 0.01
            reward_type_msg = "mínima (protocolo)"

        if self.token_supply + reward_amount > self.MAX_TOKEN_SUPPLY:
            reward_amount = self.MAX_TOKEN_SUPPLY - self.token_supply
            if reward_amount <= 0:
                print("[Incentive] ATENÇÃO: Limite máximo de tokens AEON atingido ou excedido. Nenhuma nova recompensa será distribuída.")
                return

        self.conn.execute('''
            INSERT INTO rewards 
            (epoch, dna_fingerprint, amount, metric_score, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.engine.current_epoch,
            dna_fingerprint,
            reward_amount,
            metric,
            datetime.now()
        ))
        
        self._update_balance(dna_fingerprint, reward_amount)
        self.token_supply += reward_amount
        self.conn.commit()
        print(f"[Incentive] Recompensa {reward_type_msg} de {reward_amount:.4f} tokens distribuída para {dna_fingerprint[:8]}... (Métrica: {metric:.4f})")

    def stake_tokens(self, participant_fingerprint: str, amount: float):
        """Permite que um participante faça staking de tokens."""
        current_balance = self.get_balance(participant_fingerprint)
        if current_balance < amount:
            print(f"[Incentive] ERRO: Saldo insuficiente para stake. Saldo: {current_balance:.2f}, Tentativa de stake: {amount:.2f}")
            return False
        
        self.conn.execute('''
            INSERT INTO stakes (participant_fingerprint, amount, start_time, end_time)
            VALUES (?, ?, ?, NULL)
        ''', (participant_fingerprint, amount, datetime.now()))
        self._update_balance(participant_fingerprint, -amount)
        self.staking_pool += amount
        self.conn.commit()
        print(f"[Incentive] {amount:.2f} tokens em stake por {participant_fingerprint[:8]}...")
        return True

    def unstake_tokens(self, participant_fingerprint: str, amount: float):
        """Permite que um participante retire tokens do staking."""
        staked_amount_query = self.conn.execute('''
            SELECT SUM(amount) FROM stakes 
            WHERE participant_fingerprint = ? AND end_time IS NULL
        ''', (participant_fingerprint,)).fetchone()[0] or 0.0

        if staked_amount_query < amount:
            print(f"[Incentive] ERRO: {participant_fingerprint[:8]}... não tem {amount:.2f} tokens em stake para retirar. Atualmente em stake: {staked_amount_query:.2f}")
            return False

        self.conn.execute('''
            UPDATE stakes SET end_time = ? 
            WHERE participant_fingerprint = ? AND end_time IS NULL LIMIT 1
        ''', (datetime.now(), participant_fingerprint))

        self._update_balance(participant_fingerprint, amount)
        self.staking_pool -= amount
        self.conn.commit()
        print(f"[Incentive] {amount:.2f} tokens retirados do stake por {participant_fingerprint[:8]}...")
        return True


##############################################
# INTERPRETADOR DSL (Domain Specific Language)
##############################################

def run_dsl_script(script: str, research_engine: ResearchEngine, incentive_system: QuantumIncentiveSystem):
    """
    Interpreta um script DSL simples, linha por linha, para controlar
    o AeonCosma Engine e o Quantum Incentive System.
    """
    for line in script.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        command = tokens[0]
        args = tokens[1:]
        
        print(f"\n--- Executando Comando DSL: {command} ---")

        if command == "RUN_EPOCH":
            research_engine.execute_research_cycle()
            incentive_system.distribute_research_rewards()
        elif command == "GET_BALANCE":
            if args:
                dna_fingerprint = args[0]
                balance = incentive_system.get_balance(dna_fingerprint)
                print(f"Saldo para {dna_fingerprint[:8]}...: {balance:.4f} tokens.")
            else:
                print("ERRO: GET_BALANCE requer um fingerprint de DNA. Ex: GET_BALANCE <fingerprint>")
        elif command == "REGISTER_PARTICIPANT":
            if args:
                dna_fingerprint = args[0]
                incentive_system.register_participant_for_incentive(dna_fingerprint)
            else:
                print("ERRO: REGISTER_PARTICIPANT requer um fingerprint de DNA. Ex: REGISTER_PARTICIPANT <fingerprint>")
        elif command == "STAKE_TOKENS":
            if len(args) == 2:
                dna_fingerprint = args[0]
                try:
                    amount = float(args[1])
                    incentive_system.stake_tokens(dna_fingerprint, amount)
                except ValueError:
                    print("ERRO: O valor do stake deve ser um número. Ex: STAKE_TOKENS <fingerprint> 100.0")
            else:
                print("ERRO: STAKE_TOKENS requer um fingerprint de DNA e um valor. Ex: STAKE_TOKENS <fingerprint> 100.0")
        elif command == "UNSTAKE_TOKENS":
            if len(args) == 2:
                dna_fingerprint = args[0]
                try:
                    amount = float(args[1])
                    incentive_system.unstake_tokens(dna_fingerprint, amount)
                except ValueError:
                    print("ERRO: O valor do unstake deve ser um número. Ex: UNSTAKE_TOKENS <fingerprint> 50.0")
            else:
                print("ERRO: UNSTAKE_TOKENS requer um fingerprint de DNA e um valor. Ex: UNSTAKE_TOKENS <fingerprint> 50.0")
        elif command == "VIEW_LEDGER":
            cursor = research_engine.conn.execute("SELECT * FROM research_blocks ORDER BY block_id ASC")
            blocks = cursor.fetchall()
            print("\n--- Conteúdo do Research Ledger (Blocos) ---")
            for block in blocks:
                print(f"Block ID: {block[0]}, Prev Hash: {block[1][:8]}..., Research Hash: {block[2][:8]}..., DNA: {block[3][:8]}..., Metric: {block[4]:.4f}, Time: {block[5]}, Nonce: {block[6]}, Block Hash: {block[7][:8]}...")
                if block[8]:
                    print(f"  Interpretação LLM:\n{block[8]}")
            print("--- Fim do Ledger ---")
        elif command == "VIEW_REWARDS":
            cursor = incentive_system.conn.execute("SELECT * FROM rewards ORDER BY reward_id ASC")
            rewards = cursor.fetchall()
            print("\n--- Conteúdo do Rewards Ledger ---")
            for reward in rewards:
                print(f"Reward ID: {reward[0]}, Epoch: {reward[1]}, DNA: {reward[2][:8]}..., Amount: {reward[3]:.4f}, Metric: {reward[4]:.4f}, Time: {reward[5]}")
            print("--- Fim dos Rewards ---")
        elif command == "VIEW_STAKES":
            cursor = incentive_system.conn.execute("SELECT * FROM stakes ORDER BY stake_id ASC")
            stakes = cursor.fetchall()
            print("\n--- Conteúdo do Stakes Ledger ---")
            for stake in stakes:
                print(f"Stake ID: {stake[0]}, Participant: {stake[1][:8]}..., Amount: {stake[2]:.4f}, Start: {stake[3]}, End: {stake[4]}")
            print("--- Fim dos Stakes ---")
        elif command == "EXPORT_DNA_POOL":
            print("\n--- Pool de DNA Atual ---")
            for i, dna in enumerate(research_engine.dna_pool):
                print(f"DNA {i+1}: FP: {dna.fingerprint[:8]}..., Fitness: {dna.fitness_score:.4f}, Gen: {dna.generation}, Genes: {dna.genes}, Is User: {dna.is_user_dna}")
            print("--- Fim do Pool ---")
        else:
            print(f"Comando DSL desconhecido: {command}")

##############################################
# PRINCIPAL: Execução do Sistema com DSL
##############################################

if __name__ == '__main__':
    print("Iniciando o AeonCosma Engine v3.2: Sistema de Mente Sintética Evolutiva com PoR Modular (Clássico)...")
    
    try:
        import os
        if os.path.exists('aeoncosma_ledger.db'):
            os.remove('aeoncosma_ledger.db')
            print("Banco de dados 'aeoncosma_ledger.db' removido.")
        if os.path.exists('aeoncosma_incentives.db'):
            os.remove('aeoncosma_incentives.db')
            print("Banco de dados 'aeoncosma_incentives.db' removido.")
    except Exception as e:
        print(f"Erro ao remover bancos de dados antigos: {e}")

    research_engine = ResearchEngine()
    incentive_system = QuantumIncentiveSystem(research_engine)

    luiz_cruz_dna = CosmicDNA({
        'exploration_rate': np.random.uniform(0.1, 0.9),
        'risk_tolerance': np.random.uniform(0, 1),
        'domain_focus': np.random.uniform(0, 1),
        'innovation_bias': np.random.uniform(0.1, 0.5),
        'cosmo_h0_bias': np.random.uniform(-5, 5),
        'exo_star_temp_bias': np.random.uniform(-1000, 1000),
        'compbio_protein_len_bias': np.random.uniform(-20, 20),
        'climate_co2_bias': np.random.uniform(-0.3, 0.3),
        'material_comp_iron': np.random.uniform(0.1, 0.9),
        'material_comp_carbon': np.random.uniform(0.1, 0.9),
        'exo_formation_efficiency': np.random.uniform(0.1, 1.0),
        'exo_migration_stability': np.random.uniform(0.1, 1.0),
        'compbio_structure_affinity': np.random.uniform(0.1, 1.0),
    }, fitness_score=5.0, is_user_dna=True)
    
    research_engine.dna_pool.append(luiz_cruz_dna)
    print(f"\n[NOVO USUÁRIO] DNA para 'Luiz Cruz' (Fingerprint: {luiz_cruz_dna.fingerprint[:8]}...) adicionado ao pool.")

    incentive_system.register_participant_for_incentive(luiz_cruz_dna.fingerprint)
    
    print(f"Saldo inicial de 'Luiz Cruz' ({luiz_cruz_dna.fingerprint[:8]}...): {incentive_system.get_balance(luiz_cruz_dna.fingerprint):.4f} tokens.")

    num_epochs_to_run = None
    
    current_epoch_loop = 0
    while True:
        if num_epochs_to_run is not None and current_epoch_loop >= num_epochs_to_run:
            print(f"\n--- Concluídas {num_epochs_to_run} épocas de pesquisa. Encerrando o loop principal. ---")
            break

        print(f"\n===== INÍCIO DO CICLO DE EXECUÇÃO AUTOMÁTICA (Época {current_epoch_loop}) =====")
        
        research_engine.execute_research_cycle()
        
        incentive_system.distribute_research_rewards()
        
        last_block_dna_fp = None
        try:
            cursor = research_engine.conn.cursor()
            cursor.execute("SELECT dna_fingerprint FROM research_blocks ORDER BY block_id DESC LIMIT 1")
            result = cursor.fetchone()
            if result:
                last_block_dna_fp = result[0]
        except Exception as e:
            print(f"AVISO: Não foi possível obter o fingerprint do último DNA minerador para exibição de saldo: {e}")

        if last_block_dna_fp:
            print(f"  Saldo atual do DNA {last_block_dna_fp[:8]}...: {incentive_system.get_balance(last_block_dna_fp):.4f} tokens.")

        time.sleep(5)

        current_epoch_loop += 1

    dsl_script_final_view = f"""
    VIEW_LEDGER
    VIEW_REWARDS
    VIEW_STAKES
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}
    GET_BALANCE {luiz_cruz_dna.fingerprint}
    EXPORT_DNA_POOL
    """
    
    print("\n--- Executando Comandos DSL Finais para Resumo ---")
    run_dsl_script(dsl_script_final_view, research_engine, incentive_system)
