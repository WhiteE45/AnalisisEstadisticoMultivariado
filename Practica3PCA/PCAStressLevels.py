import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carga del dataset
data = pd.read_csv('StressLevelDataset.csv')

# Análisis de correlación
correlation_matrix = data.corr()

# Mapa de calor de la correlación
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True, square=True)
plt.title('Mapa de calor de la matriz de correlación')
plt.show()

# Variables altamente correlacionadas
threshold = 0.75
high_corr_pairs = [
    (var1, var2, correlation_matrix.loc[var1, var2])
    for var1 in correlation_matrix.columns
    for var2 in correlation_matrix.columns
    if var1 != var2 and abs(correlation_matrix.loc[var1, var2]) > threshold
]
if high_corr_pairs:
    print("\nPares de variables con alta correlación (umbral > {:.2f}):".format(threshold))
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1} y {var2}: {corr:.2f}")
else:
    print("\nNo se encontraron pares de variables con alta correlación por encima del umbral.")

# Estandarización de los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Aplicar PCA
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# Explicación de la varianza
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Gráfica de varianza explicada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Varianza explicada acumulada')
plt.xlabel('Número de componentes principales')
plt.ylabel('Varianza explicada acumulada')
plt.grid()
plt.show()

# Selección del número óptimo de componentes
optimal_components = np.argmax(cumulative_variance >= 0.9) + 1  # 90% de varianza explicada
print(f"\nEl número óptimo de componentes principales es: {optimal_components}")

# Análisis de las variables que más contribuyen
componentes_principales = pd.DataFrame(
    pca.components_,
    columns=data.columns,
    index=[f'PC{i+1}' for i in range(len(pca.components_))]
)

print("\nMatriz de pesos de las variables en cada componente principal:")
print(componentes_principales)

# Mostrar las primeras filas de los datos transformados
pca_df = pd.DataFrame(pca_result[:, :optimal_components], columns=[f'PC{i+1}' for i in range(optimal_components)])
print("\nDatos transformados (primeras filas):")
print(pca_df.head())

# Mostrar contribución de variables originales a cada componente principal
top_features_per_pc = {}
for i in range(optimal_components):
    pc_name = f'PC{i+1}'
    sorted_features = componentes_principales.loc[pc_name].abs().sort_values(ascending=False)
    top_features = sorted_features.index[:5]  # Las 5 variables con mayor peso
    top_features_per_pc[pc_name] = top_features.tolist()
    print(f"\nLas variables más importantes para {pc_name}:")
    print(sorted_features.head(5))
    