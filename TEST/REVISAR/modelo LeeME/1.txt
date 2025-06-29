import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import pandas as pd
from datetime import datetime

# Criar diretório de saída
output_dir = os.path.join(os.getcwd(), "resultados_simulacao")
os.makedirs(output_dir, exist_ok=True)

# Gerar dados simulados
n_samples = 1000
n_features = 2
n_clusters = 3
X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Aplicar DBSCAN para detecção de aglomerados
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Salvar dados simulados e rótulos em CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"simulacao_{timestamp}.csv"
csv_path = os.path.join(output_dir, csv_filename)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Cluster'] = labels
df.to_csv(csv_path, index=False)

# Plotar e salvar gráfico dos aglomerados detectados
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
plt.title('Aglomerados Detectados com DBSCAN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plot_filename = f"grafico_aglomerados_{timestamp}.png"
plot_path = os.path.join(output_dir, plot_filename)
plt.savefig(plot_path)
plt.show()

print(f"Simulação concluída. Dados salvos em: {csv_path}")
print(f"Gráfico salvo em: {plot_path}")
