import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Carregar o arquivo PantheonPlus_SH0ES.dat ---
pantheon_file = "PantheonPlus_SH0ES.dat"
try:
    # Usando sep='\s+' para separar por espaços (qualquer quantidade de espaço)
    pantheon_data = pd.read_csv(pantheon_file, sep='\s+', comment='#')
    print("PantheonPlus_SH0ES.dat carregado com sucesso.")
except Exception as e:
    print("Erro ao carregar PantheonPlus_SH0ES.dat:", e)

print("Cabeçalho do PantheonPlus_SH0ES.dat:")
print(pantheon_data.head())

# --- 2. Carregar o arquivo Feb19_DESI_BAO.csv ---
bao_file = "Feb19_DESI_BAO.csv"
try:
    bao_data = pd.read_csv(bao_file, sep='\s+', header=None)
    print("Feb19_DESI_BAO.csv carregado com sucesso.")
except Exception as e:
    print("Erro ao carregar Feb19_DESI_BAO.csv:", e)

print("Cabeçalho (primeiras linhas) do Feb19_DESI_BAO.csv:")
print(bao_data.head())

# Se o número de colunas for igual a 7, atribuímos nomes provisórios:
if bao_data.shape[1] == 7:
    bao_data.columns = ['z', 'dummy2', 'dummy3', 'dummy4', 'dummy5', 'dummy6', 'D_V']
    print("Colunas atribuídas ao BAO:", bao_data.columns.tolist())
else:
    print("Formato inesperado nos dados BAO; número de colunas:", bao_data.shape[1])

# --- 3. Plot comparativo dos dados ---

plt.figure(figsize=(10, 6))

# Processar os dados do Pantheon:
if not pantheon_data.empty:
    # Verifica se as colunas 'z' e 'mu' existem; caso contrário, tenta usar as duas primeiras colunas
    if 'z' in pantheon_data.columns and 'mu' in pantheon_data.columns:
        plt.errorbar(pantheon_data['z'], pantheon_data['mu'],
                     yerr=pantheon_data.get('mu_err', None),
                     fmt='o', label="PantheonPlus_SH0ES", color='blue')
    else:
        # Se não houver, tenta renomear as duas primeiras colunas para 'z' e 'mu'
        pandas_columns = pantheon_data.columns.tolist()
        if len(pantheon_data.columns) >= 2:
            pantheon_data = pantheon_data.rename(columns={pandas_columns[0]: "z", pandas_columns[1]: "mu"})
            plt.errorbar(pantheon_data['z'], pantheon_data['mu'],
                         fmt='o', label="PantheonPlus_SH0ES", color='blue')
        else:
            print("Os dados do PantheonPlus_SH0ES.dat não possuem colunas suficientes.")
else:
    print("O DataFrame PantheonPlus_SH0ES.dat está vazio.")

# Plot dos dados BAO:
if 'z' in bao_data.columns and 'D_V' in bao_data.columns:
    plt.plot(bao_data['z'], bao_data['D_V'], 'rs-', label="Feb19_DESI_BAO")
else:
    print("Os dados BAO não apresentam as colunas esperadas 'z' ou 'D_V'.")

plt.xlabel("Redshift (z)")
plt.ylabel("Medida de Distância (unidades arbitrárias)")
plt.title("Comparativo: Dados PantheonPlus_SH0ES vs. Feb19_DESI_BAO")
plt.legend()
plt.grid(True)
plt.savefig("Pantheon_BAO_comparison_updated.png", dpi=300)
plt.show()