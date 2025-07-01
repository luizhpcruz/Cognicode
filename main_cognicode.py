"""
main_cognicode.py
Script principal do framework CogniCode: orquestra simulações, evolução e análise.
"""
from evolution.evolution_pipeline import EvolutionPipeline
from incentives.incentive_system import IncentiveSystem
from symbolic_ai.symbolic_dna import SymbolicDNA
import numpy as np

if __name__ == '__main__':
    print("Iniciando o CogniCode: Framework de Simulação e IA Evolutiva...")
    evolution_pipeline = EvolutionPipeline()
    incentive_system = IncentiveSystem(evolution_pipeline)
    user_dna = SymbolicDNA({
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
    evolution_pipeline.dna_pool.append(user_dna)
    print(f"\n[NOVO USUÁRIO] DNA simbólico adicionado ao pool.")
    incentive_system.register_participant(user_dna.fingerprint)
    print(f"Saldo inicial: {incentive_system.get_balance(user_dna.fingerprint):.4f} tokens.")
    while True:
        try:
            user_input = input("\nDigite comando DSL (ou 'EXIT' para sair): ").strip()
            if user_input.upper() in ("EXIT", "QUIT", "SAIR"):
                print("Encerrando CogniCode. Até logo!")
                break
            from cli.dsl_interpreter import run_dsl_script
            run_dsl_script(user_input, evolution_pipeline, incentive_system)
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário. Encerrando.")
            break
        except Exception as e:
            print(f"Erro ao executar comando: {e}")