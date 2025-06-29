#!/usr/bin/env python3
# aeoncosma_engine_v2.py

from typing import List, Dict, Optional, Tuple
import numpy as np
import hashlib
import json
from datetime import datetime
from enum import Enum, auto
import sqlite3
from dataclasses import dataclass
import random
from joblib import Parallel, delayed # Para execução paralela em ResearchEngine

# Qiskit imports (re-introduzidos e necessários para o QuantumCryptoModule e _quantum_pow)
from qiskit import QuantumCircuit, execute
from qiskit_aer import Aer # Para Aer.get_backend('aer_simulator')

# Cryptography imports (para carteira e assinaturas, usadas nas funções gerar_wallet e gerar_fingerprint)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

##############################################
# MÓDULOS CRIPTOGRÁFICOS BÁSICOS (Funções de Suporte)
# Funções essenciais para a Prova de Pesquisa e transações.
##############################################

def gerar_wallet():
    """
    Gera um par de chaves (privada e pública) usando criptografia de curva elíptica (SECP256R1).
    Retorna a chave privada (objeto) e a chave pública (PEM formatado em string).
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return private_key, pub_pem

def gerar_fingerprint(public_key_pem):
    """
    Gera uma "impressão digital" (fingerprint) para uma chave pública.
    É um hash SHA256 da chave pública, truncado para os primeiros 16 caracteres.
    Funciona como um identificador único para o participante.
    """
    return hashlib.sha256(public_key_pem.encode()).hexdigest()[:16]

##############################################
# CLASSES CORE DO SISTEMA AEONCOSMA
##############################################

class ResearchDomain(Enum):
    COSMOLOGY = auto()
    EXOPLANETS = auto()
    QUANTUM_CRYPTO = auto()
    COMPUTATIONAL_BIOLOGY = auto()
    MATERIALS_SCIENCE = auto()
    CLIMATE_MODELING = auto()

@dataclass
class ResearchResult:
    domain: ResearchDomain
    metrics: Dict[str, float]
    parameters: Dict[str, float]
    timestamp: datetime
    dna_fingerprint: str  # Link para o DNA que gerou esta pesquisa

class CosmicDNA:
    def __init__(self, genes: Dict[str, float], fitness_score: float = 0.0, generation: int = 0):
        self.genes = genes
        self.fitness_score = fitness_score
        self.generation = generation
        self.fingerprint = self._calculate_fingerprint()
    
    def _calculate_fingerprint(self) -> str:
        """Gera hash única para esta configuração de DNA"""
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.sha3_256(gene_str.encode()).hexdigest()[:16]
    
    def mutate(self) -> 'CosmicDNA':
        """Cria nova versão mutada deste DNA"""
        new_genes = self.genes.copy()
        for key in new_genes:
            if np.random.random() < 0.1:  # 10% chance de mutação por gene
                new_genes[key] += np.random.normal(0, 0.1)
        return CosmicDNA(new_genes, fitness_score=0.0, generation=self.generation + 1)

##############################################
# MÓDULOS DE PESQUISA DEDICADOS
# Classes especializadas para simulações e testes em diferentes domínios.
##############################################

class CosmologyResearchModule:
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []
    
    def run_simulation(self) -> Dict[str, float]:
        """Executa simulação cosmológica com parâmetros atuais"""
        # Implementação real usaria frameworks como CLASS ou CAMB
        h0 = self.params.get('h0', 67.8)
        omega_m = self.params.get('omega_m', 0.3)
        
        # Simulação estocástica (simplificada)
        h_z = np.random.normal(h0, 2)
        sigma8 = 0.8 + (omega_m - 0.3) * 0.5 # Exemplo de métrica relacionada a omega_m
        
        metrics = {
            'h(z)': h_z,
            'sigma8': sigma8,
            'fit_quality': 1.0 - abs(h_z - 70)/70, # Qualidade de ajuste a um valor alvo (ex: 70)
            'consistency': 0.9 - abs(sigma8 - 0.8)/2 # Consistência com valor esperado (ex: 0.8)
        }
        
        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        
        return metrics

class ExoplanetsModule:
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []

    def run_simulation(self) -> Dict[str, float]:
        """Simula a detecção e caracterização de exoplanetas."""
        # Parâmetros influenciados pelo DNA (ex: foco em tipo de estrela, distância)
        star_temp = self.params.get('star_temp', 4000 + random.uniform(-500, 500))
        planet_dist = self.params.get('planet_dist', 0.5 + random.uniform(-0.2, 0.2))
        
        # Métricas simplificadas
        detectability = 0.7 + (star_temp / 10000) * 0.2 - (planet_dist / 2) * 0.1
        habitability_score = 1.0 - abs(planet_dist - 1.0) / 1.0 * 0.5 # Melhor em 1.0 UA
        
        metrics = {
            'detectability_score': max(0, min(1, detectability)), # Entre 0 e 1
            'habitability_potential': max(0, min(1, habitability_score))
        }

        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        return metrics

class QuantumCryptoModule:
    def __init__(self):
        self.backend = Aer.get_backend('aer_simulator') # Requer Qiskit e Qiskit Aer
    
    def test_protocol(self, protocol_type: str) -> Dict[str, float]:
        """Testa protocolo de criptografia quântica (ex: BB84) em simulador."""
        qc = QuantumCircuit(2, 2)
        
        if protocol_type == "BB84":
            # Implementação simplificada do BB84 para simulação
            # Alice e Bob escolhem bases aleatoriamente
            basis_alice_choice = random.choice(['X', 'Z'])
            basis_bob_choice = random.choice(['X', 'Z'])
            
            # Alice prepara um qubit aleatório em uma das bases
            initial_bit = random.randint(0, 1)
            if initial_bit == 1:
                qc.x(0) # Inverte para |1> se bit inicial for 1
            if basis_alice_choice == 'X':
                qc.h(0) # Aplica Hadamard se a base de Alice for X
            
            # Bob mede o qubit. Ele escolhe a base de medição (Hadamard para X)
            if basis_bob_choice == 'X':
                qc.h(0) # Aplica Hadamard para medir na base X
            
            qc.measure(0, 0) # Mede o qubit no bit clássico
            
            # Executar o circuito no simulador
            job = execute(qc, self.backend, shots=1024)
            counts = job.result().get_counts()
            
            # Simplificação para cálculo de métricas de erro/eficiência/segurança
            # Em um BB84 real, Bob publicaria suas bases e manteria os bits onde bases coincidem.
            # Aqui, simulamos o "vazamento" e a taxa de erro.
            
            # Assume que Bob lê o resultado. Idealmente, se bases coincidem, resultado é original bit.
            # Se bases não coincidem, resultado é 50/50.
            
            # Esta é uma simulação *muito* simplificada do BB84 para gerar métricas.
            # Não replica a troca de chaves bit a bit, mas sim a qualidade geral.
            
            # Taxa de erro simulada: maior se bases não coincidem, ou se há "ruído"
            simulated_error = 0.1 # Erro base
            if basis_alice_choice != basis_bob_choice:
                simulated_error += 0.4 # Aumenta o erro se as bases são diferentes
            
            # Para simplificar e garantir resultados numéricos
            error_rate = min(1.0, simulated_error + random.uniform(-0.05, 0.05)) # Adiciona ruído
            efficiency = 0.8 - error_rate * 0.5 # Diminui eficiência com erro
            security = 1.0 - error_rate # Segurança é inversamente proporcional ao erro

            return {
                'error_rate': error_rate,
                'efficiency': efficiency,
                'security': security
            }
        
        return {'error_rate': 0.5, 'efficiency': 0.5, 'security': 0.5} # Default para outros protocolos

class ComputationalBiologyModule:
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []

    def simulate_protein_folding(self) -> Dict[str, float]:
        """Simula um processo simplificado de dobramento de proteína."""
        # Parâmetros de DNA podem influenciar complexidade, tempo de simulação, etc.
        protein_length = int(self.params.get('protein_length', 50 + random.randint(-10, 10)))
        hydrophobicity = self.params.get('hydrophobicity', 0.5 + random.uniform(-0.1, 0.1))

        # Métricas simplificadas: estabilidade e tempo para dobrar
        stability_score = 0.7 + (hydrophobicity * 0.2) - (protein_length / 100) * 0.1
        folding_time_efficiency = 1.0 - (protein_length / 100) * 0.3 # Quanto menor a proteína, mais rápido

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
    def __init__(self):
        self.property_models = {
            'conductivity': lambda x: 1.0 / (1 + np.exp(-10*(x-0.5))),
            'strength': lambda x: x**2,
            'flexibility': lambda x: np.sin(x * np.pi) # Exemplo: maior em 0.5
        }
    
    def simulate_material(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Simula propriedades de material com dada composição."""
        # Normalizar composição para que a soma seja 1
        total = sum(composition.values())
        if total == 0: # Evitar divisão por zero se composition for vazia ou zeros
            # Retorna valores padrão ou zero para evitar erros
            return {prop: 0.0 for prop in self.property_models}

        norm_comp = {k: v/total for k, v in composition.items()}
        
        # Calcular uma 'propriedade média' ou 'parâmetro de entrada' para os modelos.
        # Aqui, vamos criar uma 'eficácia de composição' ponderada se os elementos forem importantes.
        # Para simplificar, faremos uma média simples dos valores normalizados.
        # Em um modelo real, a composição 'norm_comp' seria usada para calcular propriedades específicas.
        avg_composition_value = sum(norm_comp.values()) / len(norm_comp) # Simplificação

        results = {}
        for prop, model in self.property_models.items():
            results[prop] = model(avg_composition_value) # Aplica o modelo à média da composição
        
        return results

class ClimateModelingModule:
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.history = []

    def run_climate_simulation(self) -> Dict[str, float]:
        """Simula um modelo climático simplificado."""
        # Parâmetros influenciados pelo DNA (e.g., emissões, sensibilidade climática)
        co2_emissions_factor = self.params.get('co2_emissions_factor', 0.5 + random.uniform(-0.2, 0.2)) # Fator de 0 a 1
        albedo = self.params.get('albedo', 0.3 + random.uniform(-0.1, 0.1)) # Reflectividade da Terra

        # Métricas simplificadas de impacto
        global_temp_increase = co2_emissions_factor * 3.0 * (1 - albedo) + random.uniform(-0.5, 0.5)
        sea_level_rise = global_temp_increase * 0.5 + random.uniform(-0.1, 0.1)

        metrics = {
            'global_temp_increase_celsius': max(0, global_temp_increase),
            'sea_level_rise_meters': max(0, sea_level_rise),
            'environmental_impact_score': 1.0 - (global_temp_increase / 5.0 + sea_level_rise / 1.0) # Menor é melhor
        }
        self.history.append({
            'timestamp': datetime.now(),
            'parameters': self.params.copy(),
            'metrics': metrics.copy()
        })
        return metrics

class ResearchEngine:
    def __init__(self):
        self.dna_pool: List[CosmicDNA] = self._initialize_dna_pool()
        self.research_history: List[ResearchResult] = []
        self.current_epoch = 0
        
        # Inicializa os módulos de pesquisa dedicados
        # Os parâmetros iniciais podem ser genéricos, o DNA os ajustará dinamicamente
        self.cosmology_module = CosmologyResearchModule(parameters={'h0': 67.8, 'omega_m': 0.3})
        self.exoplanets_module = ExoplanetsModule(parameters={'star_temp': 4000, 'planet_dist': 0.8})
        self.quantum_crypto_module = QuantumCryptoModule() # Não recebe parâmetros do DNA diretamente
        self.comp_bio_module = ComputationalBiologyModule(parameters={'protein_length': 60, 'hydrophobicity': 0.6})
        self.materials_module = MaterialsScienceModule() # Recebe composição via dna.genes
        self.climate_module = ClimateModelingModule(parameters={'co2_emissions_factor': 0.5, 'albedo': 0.3})

        self._init_ledger()
    
    def _initialize_dna_pool(self, size=10) -> List[CosmicDNA]:
        """Cria pool inicial de estratégias de pesquisa com genes que influenciam módulos."""
        return [CosmicDNA({
            'exploration_rate': np.random.uniform(0.1, 0.9),
            'risk_tolerance': np.random.uniform(0, 1),
            'domain_focus': np.random.uniform(0, 1), # Este gene pode ser usado para ponderar qual módulo rodar
            'innovation_bias': np.random.uniform(0.1, 0.5),
            # Novos genes para influenciar módulos específicos
            'cosmo_h0_bias': np.random.uniform(-5, 5), # Ajuste para h0
            'exo_star_temp_bias': np.random.uniform(-1000, 1000), # Ajuste para temp estelar
            'compbio_protein_len_bias': np.random.uniform(-20, 20), # Ajuste para comprimento de proteina
            'climate_co2_bias': np.random.uniform(-0.3, 0.3), # Ajuste para emissões de CO2
            'material_comp_iron': np.random.uniform(0.1, 0.9), # Composição de material (ferro)
            'material_comp_carbon': np.random.uniform(0.1, 0.9), # Composição de material (carbono)
        }) for _ in range(size)]
    
    def _init_ledger(self):
        """Inicializa banco de dados para registro imutável"""
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
                block_hash TEXT NOT NULL UNIQUE
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
    
    def execute_research_cycle(self):
        """Executa um ciclo completo de pesquisa e aprendizado"""
        print(f"\n--- Iniciando Época de Pesquisa {self.current_epoch} ---")
        # 1. Selecionar DNA para pesquisa
        research_dna = self._select_research_dna()
        print(f"DNA selecionado para pesquisa (Fingerprint: {research_dna.fingerprint[:8]}...)")
        
        # 2. Executar pesquisas em múltiplos domínios
        results = self._parallel_research_execution(research_dna)
        
        # 3. Avaliar e integrar resultados
        combined_metric = self._evaluate_results(results)
        print(f"Métrica Combinada da Época: {combined_metric:.4f}")
        
        # 4. Atualizar DNA pool com base no desempenho
        self._evolve_dna_pool(combined_metric)
        print(f"Pool de DNA evoluído. Melhor Fitness: {max(self.dna_pool, key=lambda x: x.fitness_score).fitness_score:.4f}")
        
        # 5. Registrar no ledger distribuído (blockchain simplificado)
        self._record_research_block(results, research_dna, combined_metric)
        print("Pesquisa registrada no ledger.")
        
        self.current_epoch += 1
    
    def _select_research_dna(self) -> CosmicDNA:
        """
        Seleciona DNA para pesquisa usando um método influenciado por quântica simulada.
        Prioriza o melhor DNA, mas permite exploração aleatória.
        """
        qc = QuantumCircuit(1, 1) # 1 qubit, 1 bit clássico
        qc.h(0) # Aplica Hadamard para superposição
        qc.measure(0, 0) # Mede o qubit
        
        # O erro ModuleNotFoundError para Aer.get_backend ou execute é resolvido
        # se o Qiskit estiver corretamente instalado.
        try:
            job = execute(qc, Aer.get_backend('aer_simulator'), shots=1)
            result = job.result()
            counts = result.get_counts()
            random_bit = int(list(counts.keys())[0][0]) # Pega o resultado do bit (0 ou 1)
        except Exception as e:
            print(f"AVISO: Falha na simulação quântica para seleção de DNA ({e}). Usando aleatoriedade clássica.")
            random_bit = random.randint(0, 1) # Fallback clássico

        # Lógica de seleção: 70% de chance de selecionar o melhor, 30% de explorar aleatoriamente
        # Usamos o random_bit para simular essa probabilidade: se random_bit é 0 (70% chance), pegamos o melhor.
        # Se random_bit é 1 (30% chance), exploramos. (Esta é uma simplificação do uso do bit).
        if not self.dna_pool: # Evitar erro se pool estiver vazio
            return CosmicDNA(self._initialize_dna_pool(1)[0].genes) # Cria um DNA base se não houver
        
        if random_bit == 0 and len(self.dna_pool) > 1: # Selecionar o melhor (se houver mais de um)
            # Ordena por fitness score (decrescente) e pega o primeiro
            return sorted(self.dna_pool, key=lambda x: x.fitness_score, reverse=True)[0]
        else: # Explorar aleatoriamente ou se o pool for muito pequeno
            return random.choice(self.dna_pool)
    
    def _parallel_research_execution(self, dna: CosmicDNA) -> List[ResearchResult]:
        """
        Executa pesquisas em múltiplos domínios, com parâmetros influenciados pelo DNA.
        A seleção de quais módulos rodar pode ser mais inteligente ou probabilisticamente
        influenciada pelo 'domain_focus' do DNA. Aqui, rodamos todos com parâmetros adaptados.
        """
        results: List[ResearchResult] = []
        
        # Parâmetros específicos para cada módulo, derivados do DNA
        cosmo_params = {
            'h0': 67.8 + dna.genes.get('cosmo_h0_bias', 0) * dna.genes.get('innovation_bias', 1),
            'omega_m': 0.3 + dna.genes.get('risk_tolerance', 0) * 0.1
        }
        exo_params = {
            'star_temp': 4000 + dna.genes.get('exo_star_temp_bias', 0) * dna.genes.get('exploration_rate', 1),
            'planet_dist': 0.5 + dna.genes.get('risk_tolerance', 0) * 0.2
        }
        comp_bio_params = {
            'protein_length': int(50 + dna.genes.get('compbio_protein_len_bias', 0)),
            'hydrophobicity': 0.5 + dna.genes.get('innovation_bias', 0) * 0.1
        }
        materials_comp = { # Composição de exemplo influenciada pelo DNA
            'iron': dna.genes.get('material_comp_iron', 0.5),
            'carbon': dna.genes.get('material_comp_carbon', 0.5),
            'alloy_element': 1.0 - (dna.genes.get('material_comp_iron', 0.5) + dna.genes.get('material_comp_carbon', 0.5)) # Garante soma 1
        }
        materials_comp['alloy_element'] = max(0.01, materials_comp['alloy_element']) # Evita zero para simulação
        
        climate_params = {
            'co2_emissions_factor': max(0.1, min(1.0, dna.genes.get('climate_co2_bias', 0.5) + dna.genes.get('risk_tolerance', 0) * 0.2)),
            'albedo': max(0.1, min(0.9, dna.genes.get('innovation_bias', 0.3)))
        }

        # Execução dos módulos
        # Note: QuantumCryptoModule não recebe parâmetros do DNA diretamente neste setup,
        # mas pode ser expandido para que o DNA influencie o tipo de protocolo, qubits, etc.
        
        # Cosmology
        metrics_cosmo = self.cosmology_module.run_simulation()
        results.append(ResearchResult(
            domain=ResearchDomain.COSMOLOGY, metrics=metrics_cosmo,
            parameters=cosmo_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Cosmology Research Done. Metrics: {metrics_cosmo}")

        # Exoplanets
        metrics_exo = self.exoplanets_module.run_simulation()
        results.append(ResearchResult(
            domain=ResearchDomain.EXOPLANETS, metrics=metrics_exo,
            parameters=exo_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Exoplanets Research Done. Metrics: {metrics_exo}")

        # Quantum Crypto (BB84 hardcoded por enquanto)
        metrics_qc = self.quantum_crypto_module.test_protocol("BB84")
        results.append(ResearchResult(
            domain=ResearchDomain.QUANTUM_CRYPTO, metrics=metrics_qc,
            parameters={'protocol_type': "BB84"}, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Quantum Crypto Research Done. Metrics: {metrics_qc}")

        # Computational Biology (Protein Folding hardcoded por enquanto)
        metrics_cb = self.comp_bio_module.simulate_protein_folding()
        results.append(ResearchResult(
            domain=ResearchDomain.COMPUTATIONAL_BIOLOGY, metrics=metrics_cb,
            parameters=comp_bio_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Computational Biology Research Done. Metrics: {metrics_cb}")
        
        # Materials Science
        metrics_ms = self.materials_module.simulate_material(materials_comp)
        results.append(ResearchResult(
            domain=ResearchDomain.MATERIALS_SCIENCE, metrics=metrics_ms,
            parameters=materials_comp, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Materials Science Research Done. Metrics: {metrics_ms}")

        # Climate Modeling
        metrics_cm = self.climate_module.run_climate_simulation()
        results.append(ResearchResult(
            domain=ResearchDomain.CLIMATE_MODELING, metrics=metrics_cm,
            parameters=climate_params, timestamp=datetime.now(), dna_fingerprint=dna.fingerprint
        ))
        print(f"  Climate Modeling Research Done. Metrics: {metrics_cm}")

        return results
    
    def _evaluate_results(self, results: List[ResearchResult]) -> float:
        """Combina métricas de múltiplos domínios"""
        if not results:
            return 0.0
        
        # Ponderar por domínio de pesquisa - ajuste os pesos conforme a importância
        weights = {
            ResearchDomain.COSMOLOGY: 1.2,
            ResearchDomain.EXOPLANETS: 1.0,
            ResearchDomain.QUANTUM_CRYPTO: 1.5, # Geralmente de alto valor pela segurança
            ResearchDomain.COMPUTATIONAL_BIOLOGY: 1.3,
            ResearchDomain.MATERIALS_SCIENCE: 1.1,
            ResearchDomain.CLIMATE_MODELING: 1.4 # Impacto potencialmente alto
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        for result in results:
            weight = weights.get(result.domain, 1.0) # Peso padrão de 1.0 se não especificado
            
            # Soma das métricas individuais dentro de um domínio
            # Tratamento para garantir que as métricas sejam valores úteis (e.g., normalizar entre 0 e 1)
            domain_score_sum = sum(result.metrics.values())
            
            # Divide pela quantidade de métricas para ter uma média por domínio
            # Se não houver métricas ou o dicionário estiver vazio, o score do domínio é 0
            if result.metrics:
                domain_average_score = domain_score_sum / len(result.metrics)
            else:
                domain_average_score = 0.0

            total_weighted_score += domain_average_score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _evolve_dna_pool(self, combined_metric: float):
        """Evolui o pool de DNA com base no desempenho"""
        # Atualizar fitness do DNA atual
        for dna in self.dna_pool:
            # Recompensa o DNA que foi usado para a pesquisa atual
            if dna.fingerprint == self._select_research_dna().fingerprint: # Identifica o DNA que gerou a métrica
                dna.fitness_score = dna.fitness_score * 0.5 + combined_metric * 0.5 # Mais peso na nova métrica
            else:
                dna.fitness_score *= 0.9 # DNAs não usados decaem um pouco no fitness
        
        # Seleção e mutação
        self.dna_pool.sort(key=lambda x: -x.fitness_score)
        new_pool = self.dna_pool[:max(2, len(self.dna_pool)//2)] # Mantém a metade superior (mínimo de 2)
        
        # Adiciona mutações a partir dos melhores pais
        num_mutations = len(self.dna_pool) - len(new_pool) - 1 # Espaço para novas e uma aleatória
        for _ in range(num_mutations):
            if new_pool: # Garante que haja pais para mutar
                parent = random.choice(new_pool[:max(1, len(new_pool)//2)]) # Seleciona dos melhores da nova pool
                new_pool.append(parent.mutate())
        
        # Adiciona algumas novas aleatórias para exploração
        new_pool.append(self._initialize_dna_pool(1)[0]) # Adiciona um DNA totalmente novo
        self.dna_pool = new_pool[:10] # Limita o tamanho do pool para evitar crescimento descontrolado
        
        # Garante que todos os DNAs na pool tenham suas gerações atualizadas se evoluíram
        for dna in self.dna_pool:
            dna.generation += 1 # Incrementar geração para todos que sobrevivem ou são novos

    def _record_research_block(self, results: List[ResearchResult], 
                                  dna: CosmicDNA, metric: float):
        """Registra pesquisa no ledger distribuído (SQLite para simulação blockchain)"""
        # Gerar hash dos resultados
        results_data_for_hash = []
        for r in results:
            # Convert ResearchResult to a dict that can be JSON serialized
            # Enum needs to be converted to its value for serialization
            res_dict = {
                'domain': r.domain.name,
                'metrics': r.metrics,
                'parameters': r.parameters,
                'timestamp': r.timestamp.isoformat(), # Convert datetime to string
                'dna_fingerprint': r.dna_fingerprint
            }
            results_data_for_hash.append(res_dict)

        results_hash = hashlib.sha3_256(
            json.dumps(results_data_for_hash, sort_keys=True).encode()
        ).hexdigest()
        
        # Obter hash do último bloco
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT block_hash FROM research_blocks ORDER BY block_id DESC LIMIT 1"
        )
        last_hash_row = cursor.fetchone()
        previous_hash = last_hash_row[0] if last_hash_row else "0" * 64 # Hash inicial para o primeiro bloco
        
        # Prova de trabalho quântica (ou híbrida)
        print("  Iniciando Prova de Trabalho (PoW) Híbrida...")
        nonce, block_hash = self._quantum_pow(previous_hash, results_hash)
        print(f"  PoW Concluída. Nonce: {nonce}, Block Hash: {block_hash[:10]}...")
        
        # Inserir bloco
        cursor.execute('''
            INSERT INTO research_blocks 
            (previous_hash, research_hash, dna_fingerprint, combined_metric, timestamp, nonce, block_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (previous_hash, results_hash, dna.fingerprint, metric, datetime.now(), nonce, block_hash))
        
        block_id = cursor.lastrowid # ID do bloco recém-inserido
        
        # Inserir resultados detalhados
        for result in results:
            cursor.execute('''
                INSERT INTO research_results 
                (block_id, domain, metrics, parameters)
                VALUES (?, ?, ?, ?)
            ''', (
                block_id,
                result.domain.name, # Enum para string
                json.dumps(result.metrics),
                json.dumps(result.parameters)
            ))
        
        self.conn.commit()
        print(f"  Bloco {block_id} de Pesquisa registrado com sucesso.")
    
    def _quantum_pow(self, previous_hash: str, results_hash: str) -> Tuple[int, str]:
        """
        Prova de Trabalho (PoW) com contribuição quântica.
        A contribuição quântica é um valor (bits de um circuito) que entra no hash do PoW,
        tornando a mineração "quântica-aware".
        """
        # Circuito quântico para gerar uma parte da string que será hashificada na PoW
        # Simula uma contribuição probabilística ou um "problema quântico" a ser resolvido
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0,1], [0,1]) # Mede ambos os qubits para obter um resultado de 2 bits (e.g., '01', '10')
        
        quantum_bits_str = "00" # Fallback caso o Qiskit falhe
        try:
            # Executa o circuito no simulador Aer
            job = execute(qc, Aer.get_backend('aer_simulator'), shots=1)
            result = job.result()
            counts = result.get_counts()
            # Pega a string de resultado mais comum (como '00', '01', '10', '11')
            quantum_bits_str = list(counts.keys())[0] 
        except Exception as e:
            print(f"    AVISO: Falha na execução quântica para PoW ({e}). Usando string '00'.")
            
        nonce = 0
        while True:
            # Concatena todos os dados relevantes para o hash do bloco
            # previous_hash: Garante a ordem da cadeia
            # results_hash: Representa os dados da pesquisa do bloco
            # dna.fingerprint: Identifica o "minerador" de pesquisa
            # quantum_bits_str: A contribuição "quântica"
            # nonce: O número a ser variado para encontrar o hash válido
            block_content = f"{previous_hash}{results_hash}{quantum_bits_str}{nonce}"
            test_hash = hashlib.sha3_256(
                block_content.encode()
            ).hexdigest()
            
            # Dificuldade: Quantos zeros iniciais o hash deve ter
            # '000' é fácil para testes; aumente para '0000', '00000' para maior dificuldade.
            difficulty_prefix = "000" 
            if test_hash.startswith(difficulty_prefix):
                return nonce, test_hash
            nonce += 1
            # Para evitar loops infinitos em testes ou em caso de dificuldade muito alta
            # if nonce > 1_000_000: # Limite de nonce para depuração
            #     raise Exception("Não foi possível encontrar nonce em tempo razoável.")

class QuantumIncentiveSystem:
    def __init__(self, research_engine: ResearchEngine):
        self.engine = research_engine
        self.token_supply = 0.0 # Total de tokens emitidos
        self.staking_pool = 0.0 # Tokens em stake para validação
        self._init_incentive_db()
        self.wallets: Dict[str, float] = {} # Simula saldos de carteira para participantes

    def _init_incentive_db(self):
        """Inicializa banco de dados para registro de incentivos e staking."""
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
                participant_fingerprint TEXT NOT NULL, # Identificador do participante (pode ser o fingerprint do DNA)
                amount REAL NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME
            )
        ''')
        # Tabela para saldos, mais robusta que um dict em memória para persistência
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
        """Registra um DNA/participante no sistema de incentivo se ainda não estiver lá."""
        if self.get_balance(dna_fingerprint) == 0.0: # Simples verificação de existência
            self._update_balance(dna_fingerprint, 0.0) # Inicializa com saldo zero
            print(f"[Incentive] Participante {dna_fingerprint[:8]}... registrado para incentivos.")
        else:
            print(f"[Incentive] Participante {dna_fingerprint[:8]}... já está no sistema de incentivos.")


    def distribute_research_rewards(self):
        """
        Distribui recompensas pelo último ciclo de pesquisa registrado no ledger principal.
        Recompensas são baseadas na combined_metric e creditadas ao dna_fingerprint do bloco.
        """
        # Obter métrica combinada e dna_fingerprint do último bloco minerado
        last_block = self.engine.conn.execute('''
            SELECT combined_metric, dna_fingerprint FROM research_blocks 
            ORDER BY block_id DESC LIMIT 1
        ''').fetchone()
        
        if not last_block:
            print("[Incentive] Nenhum bloco de pesquisa encontrado para recompensar.")
            return
        
        metric, dna_fingerprint = last_block
        
        # Calcular recompensa baseada na métrica
        base_reward = 100.0 # Tokens base por bloco
        # A recompensa escala com a qualidade da pesquisa (métrica combinada)
        # Mais 1 para garantir que a métrica sempre multiplique por >= 1
        reward_amount = base_reward * (1 + metric) 
        
        # Registrar a recompensa no banco de dados de incentivos
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
        
        # Atualizar o saldo do participante que gerou o DNA
        self._update_balance(dna_fingerprint, reward_amount)
        self.token_supply += reward_amount
        self.conn.commit()
        print(f"[Incentive] Recompensa de {reward_amount:.4f} tokens distribuída para {dna_fingerprint[:8]}... (Métrica: {metric:.4f})")

    def stake_tokens(self, participant_fingerprint: str, amount: float):
        """Permite que um participante faça stake de tokens."""
        current_balance = self.get_balance(participant_fingerprint)
        if current_balance < amount:
            print(f"[Incentive] ERRO: Saldo insuficiente para stake. Saldo: {current_balance:.2f}, Tentativa de stake: {amount:.2f}")
            return False
        
        self.conn.execute('''
            INSERT INTO stakes (participant_fingerprint, amount, start_time, end_time)
            VALUES (?, ?, ?, NULL)
        ''', (participant_fingerprint, amount, datetime.now()))
        self._update_balance(participant_fingerprint, -amount) # Remove tokens do saldo
        self.staking_pool += amount
        self.conn.commit()
        print(f"[Incentive] {amount:.2f} tokens staked por {participant_fingerprint[:8]}...")
        return True

    def unstake_tokens(self, participant_fingerprint: str, amount: float):
        """Permite que um participante retire tokens do stake."""
        # Simplificação: Apenas remove da pool. Em um sistema real, verificaria stakes ativos.
        # Poderia ser mais complexo, com períodos de unbonding, etc.
        
        # Verificar se o participante realmente tem essa quantidade em stake (simplificado)
        # Para um sistema real, você precisaria de uma lógica mais robusta para rastrear stakes individuais.
        staked_amount_query = self.conn.execute('''
            SELECT SUM(amount) FROM stakes 
            WHERE participant_fingerprint = ? AND end_time IS NULL
        ''', (participant_fingerprint,)).fetchone()[0] or 0.0

        if staked_amount_query < amount:
            print(f"[Incentive] ERRO: {participant_fingerprint[:8]}... não tem {amount:.2f} tokens em stake para retirar. Atualmente staked: {staked_amount_query:.2f}")
            return False

        # Marcar stake como finalizado (simplificado, para manter histórico)
        # Em um sistema real, você gerenciaria IDs de stake específicos.
        self.conn.execute('''
            UPDATE stakes SET end_time = ? 
            WHERE participant_fingerprint = ? AND end_time IS NULL LIMIT 1 -- Retira apenas um dos stakes ativos
        ''', (datetime.now(), participant_fingerprint))

        self._update_balance(participant_fingerprint, amount) # Devolve tokens ao saldo
        self.staking_pool -= amount
        self.conn.commit()
        print(f"[Incentive] {amount:.2f} tokens unstaked por {participant_fingerprint[:8]}...")
        return True


##############################################
# INTERPRETADOR DSL (Domain Specific Language)
# Permite controlar o sistema AeonCosma.
##############################################

def run_dsl_script(script: str, research_engine: ResearchEngine, incentive_system: QuantumIncentiveSystem):
    """
    Interpreta um script DSL simples, linha por linha, para controlar o AeonCosma Engine
    e o Quantum Incentive System.
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
                dna_fingerprint = args[0] # Assume que o fingerprint já é conhecido
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
                print(f"DNA {i+1}: FP: {dna.fingerprint[:8]}..., Fitness: {dna.fitness_score:.4f}, Gen: {dna.generation}, Genes: {dna.genes}")
            print("--- Fim do Pool ---")
        else:
            print(f"Comando DSL desconhecido: {command}")

##############################################
# PRINCIPAL: Execução do Sistema com DSL
##############################################

if __name__ == '__main__':
    print("Iniciando o AeonCosma Engine v2.2: Sistema de Mente Sintética Evolutiva com PoR Modular e Quântico...")
    
    # Excluir bancos de dados antigos para uma execução limpa
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

    # Exemplo de DSL Script para demonstrar um ciclo de pesquisa e recompensa
    dsl_script = f"""
    # --- Passo 1: Registrar um DNA inicial para receber recompensas ---
    # Pegamos o fingerprint do primeiro DNA do pool inicial
    REGISTER_PARTICIPANT {research_engine.dna_pool[0].fingerprint}
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}

    # --- Passo 2: Executar um ciclo de pesquisa e mineração de recompensa ---
    RUN_EPOCH
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}
    VIEW_LEDGER
    VIEW_REWARDS
    EXPORT_DNA_POOL

    # --- Passo 3: Simular staking de tokens (exemplo) ---
    STAKE_TOKENS {research_engine.dna_pool[0].fingerprint} 50.0
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}
    VIEW_STAKES

    # --- Passo 4: Executar mais um ciclo de pesquisa e mineração ---
    RUN_EPOCH
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}
    VIEW_LEDGER
    VIEW_REWARDS
    EXPORT_DNA_POOL

    # --- Passo 5: Retirar tokens do stake (exemplo) ---
    UNSTAKE_TOKENS {research_engine.dna_pool[0].fingerprint} 20.0
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}
    VIEW_STAKES

    # --- Passo 6: Exibir o estado final ---
    VIEW_LEDGER
    VIEW_REWARDS
    VIEW_STAKES
    GET_BALANCE {research_engine.dna_pool[0].fingerprint}
    EXPORT_DNA_POOL
    """
    
    print("\n--- Executando Script DSL ---")
    run_dsl_script(dsl_script, research_engine, incentive_system)
