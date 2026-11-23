import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("supermarket_dataset_power.csv")
columnas_numericas = ["cantidad", "ventas_totales", "total_articulos", "valor_descuento", "precio_unitario"]
X = df[columnas_numericas]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inercias = []
k_values = range(1, 11)  # K de 1 a 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values, inercias, marker="o")
plt.xlabel("Numero de clusters (k)")
plt.ylabel("Inercia")
plt.title("Metodo del Codo para K")
plt.show()
