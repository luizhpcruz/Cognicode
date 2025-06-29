import numpy as np
from scipy.integrate import quad
from cobaya.run import run
from getdist import plots, loadMCSamples
import os # Importar para manipulação de diretórios
import json # Importar para salvar dados em formato JSON

class HIDEMCosmology:
    def __init__(self, H0=70, Omega_m=0.3, Omega_seed=0.1, m=-0.8, a_c=1.3):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_seed = Omega_seed
        self.m = m
        self.a_c = a_c

    def E(self, z):
        a = 1 / (1 + z)
        # Atenção: a fórmula do termo HIDEM aqui é diferente da do hybrid_cosmology.py
        # Verifique qual é a correta ou a que você deseja usar.
        term_hidem = self.Omega_seed * (1 + (a / self.a_c)**(1 / self.m))**(-self.m)
        return np.sqrt(self.Omega_m * (1 + z)**3 + term_hidem)

    def H(self, z):
        return self.H0 * self.E(z)

def run_hidem_chain():
    # Definir um diretório de saída para os resultados e gráficos
    output_dir = "aeoncosma/output"
    os.makedirs(output_dir, exist_ok=True) # Criar diretório se não existir

    # Caminho base para os arquivos da cadeia, cobaya vai criar a pasta 'chains' dentro
    chain_output_path = os.path.join(output_dir, "chains", "hidem")

    # Modificar o hidem.yaml para salvar a cadeia no diretório de saída
    # Isso exigiria carregar e modificar o yaml em tempo de execução ou passar como argumento para o run
    # Por simplicidade, vamos assumir que o cobaya pode ser configurado para salvar na pasta desejada
    # Uma forma mais robusta seria ler o hidem.yaml, modificá-lo e passá-lo para run()
    info = {
        "theory": {
            "classy": {
                "path": "./core/class_modificado",
                "extra_args": {
                    "use_HIDEM": True,
                    "Omega_seed": 0.1,
                    "m": -0.8,
                    "a_c": 1.3
                }
            }
        },
        "params": {
            "H0": [60, 80],
            "Omega_m": [0.1, 0.5],
            "Omega_seed": [0.01, 0.9],
            "m": [-2, 1],
            "a_c": [0.1, 2.0]
        },
        "likelihood": {
            "planck_2018_highl_plik": {},
            "bao_boss_dr12": {}
        },
        "sampler": {
            "mcmc": {
                "max_samples": 50000,
                "Rminus1_stop": 0.01,
                "output": chain_output_path # Define onde a cadeia será salva
            }
        }
    }
    updated_info, sampler = run(info) # Passa o dicionário de info diretamente
    print(f"✅ Cadeia HIDEM executada. Resultados salvos em: {chain_output_path}.txt")
    return sampler, updated_info

def plot_corner(sampler_name="hidem", output_dir="aeoncosma/output/figs"):
    os.makedirs(output_dir, exist_ok=True) # Garante que o diretório de figuras exista
    # Carrega as amostras do diretório onde foram salvas
    samples = loadMCSamples(os.path.join("aeoncosma/output/chains", sampler_name))
    g = plots.getSubplotPlotter()
    g.triangle_plot(samples, ['H0', 'Omega_m', 'Omega_seed', 'm', 'a_c'])
    plot_path = os.path.join(output_dir, f"{sampler_name}_corner.png")
    g.export(plot_path, dpi=300)
    print(f"✅ Gráfico de canto salvo em: {plot_path}")

def compare_lnB(output_dir="aeoncosma/output/results"):
    os.makedirs(output_dir, exist_ok=True) # Garante que o diretório de resultados exista
    s_hidem = loadMCSamples(os.path.join("aeoncosma/output/chains", "hidem"))
    s_lcdm = loadMCSamples(os.path.join("aeoncosma/output/chains", "lcdm"))
    lnB = s_hidem.getLogZ() - s_lcdm.getLogZ()
    print(f"Bayes Factor (lnB) = {lnB:.1f}")

    # Salvar o Bayes Factor em um arquivo
    results_data = {"lnB_HIDEM_vs_LCDM": lnB}
    results_path = os.path.join(output_dir, "bayes_factor_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"✅ Bayes Factor salvo em: {results_path}")

def save_mcmc_summary_results(sampler, updated_info, model_name, output_dir="aeoncosma/output/results"):
    os.makedirs(output_dir, exist_ok=True) # Garante que o diretório de resultados exista

    # Extrair estatísticas do sampler
    # Note: cobaya.yaml já calcula AIC e BIC. Você pode acessá-los do updated_info
    # ou dos relatórios gerados por cobaya.
    # Por exemplo, updated_info.get('chi2s', {}).get('planck_2018_highl_plik', 0)
    # ou de forma mais robusta, ler do arquivo de output do cobaya se ele gerar um sumário.

    # Para este exemplo, vamos simular como você extrairia alguns dados
    # Em um cenário real, você teria que parsear os resultados detalhados do cobaya
    # ou usar funções específicas para extrair AIC/BIC.
    # O getdist.loadMCSamples().getMCMCSamples() tem métodos para isso.
    samples = loadMCSamples(os.path.join("aeoncosma/output/chains", model_name))
    
    # Calcular chi2, AIC, BIC manualmente ou extrair se disponíveis
    # Isso é um exemplo simplificado, cobaya e getdist podem fornecer isso.
    # O valor de chi2 é a log-likelihood mínima (negativa)
    try:
        chi2_min = -2 * samples.getLogLike() # Ou samples.minimum_chi2
    except:
        chi2_min = None # Caso não seja encontrado
    
    # AIC = -2 * log(L) + 2*k, onde k é o número de parâmetros
    # BIC = -2 * log(L) + k * log(n), onde n é o número de pontos de dados (ou "tamanho da amostra")
    # Para ser preciso, AIC/BIC devem vir do cobaya ou serem calculados com o número correto de parâmetros e dados
    k = len(samples.getParamNames().getNames())
    n = samples.get_nsamples() # Número de amostras na cadeia (pode não ser o 'n' para BIC)

    aic = chi2_min + 2 * k if chi2_min is not None else None
    bic = chi2_min + k * np.log(n) if chi2_min is not None and n > 0 else None

    param_results = {}
    for p in samples.getParamNames().getNames():
        mean = samples.get='p'['mean'] # Acesso direto ao dicionário de estatísticas do parâmetro
        sigma = samples.get='p'['err'] # Acesso direto ao dicionário de estatísticas do parâmetro
        # Ou usar samples.getMCMCSamples().getParamSample(p).mean() e .sigma()
        param_results[p] = {"mean": mean, "sigma": sigma}
        
    results_data = {
        "model": model_name,
        "chi2": chi2_min,
        "aic": aic,
        "bic": bic,
        "parameters": param_results
    }

    results_path = os.path.join(output_dir, f"{model_name}_summary_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"✅ Resultados sumários de {model_name} salvos em: {results_path}")
    return results_data