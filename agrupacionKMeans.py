from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('supermarket_dataset_power.csv')
except FileNotFoundError:
    print("No se encontró el archivo 'supermarket_dataset_power.csv'.")
    exit()

# Agregamos los datos por producto: sumamos la cantidad total y las ventas totales
print("Agrupando productos...")
product_data = df.groupby('nombre_producto').agg(
    cantidad_total=('cantidad', 'sum'),
    ventas_totales=('ventas_totales', 'sum')
).reset_index()

print("Datos agregados listos para clustering:")
print(product_data.head())


# --- Paso 2: Escalar los Datos ---
# Seleccionamos las características para el clustering
features = product_data[['cantidad_total', 'ventas_totales']]

# Escalamos los datos para que KMeans funcione correctamente
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# --- Paso 3: Aplicar KMeans ---
# Usaremos K=3 (Poco, Medio, Muy vendido) basado en el análisis del codo previo.
# Puedes cambiar este número si quieres más o menos grupos.
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(features_scaled)

# Añadimos la etiqueta del clúster a nuestros datos
product_data['cluster'] = kmeans.labels_


# --- Paso 4: Analizar e Interpretar los Clusters ---
# Calculamos el promedio de cada clúster para entender qué es cada uno
cluster_analysis = product_data.groupby('cluster')[['cantidad_total', 'ventas_totales']].mean()
print("\n--- Análisis de Clusters (Promedios) ---")
print(cluster_analysis)

# Asignamos etiquetas legibles (esto puede cambiar dependiendo de los datos)
# Ordenamos por 'ventas_totales' para asignar las etiquetas
cluster_analysis_sorted = cluster_analysis.sort_values(by='ventas_totales')

mapping = {
    cluster_analysis_sorted.index[0]: 'Poco vendido (Cluster 0)',
    cluster_analysis_sorted.index[1]: 'Venta media (Cluster 1)',
    cluster_analysis_sorted.index[2]: 'Muy vendido (Cluster 2)'
}
product_data['segmento'] = product_data['cluster'].map(mapping)


# --- Paso 5: Mostrar los Resultados (Gráfica y Texto) ---

print("\n--- Productos Segmentados ---")
print(product_data.head())

# Guardar los resultados en un nuevo CSV
product_data.to_csv('productos_segmentados_con_codigo.csv', index=False)
print("\nResultados guardados en 'productos_segmentados_con_codigo.csv'")

# Imprimir ejemplos de cada clúster para la estrategia
print("\n--- EJEMPLOS PARA ESTRATEGIA DE CROSS-SELLING ---")
print("\nProductos 'Muy vendidos' (Tus productos gancho):")
print(product_data[product_data['segmento'] == 'Muy vendido (Cluster 2)'][['nombre_producto', 'ventas_totales']].head())

print("\nProductos 'Poco vendidos' (Tus productos objetivo):")
print(product_data[product_data['segmento'] == 'Poco vendido (Cluster 0)'][['nombre_producto', 'ventas_totales']].head())

# --- Paso 6: Visualizar los Clusters ---
print("\nGenerando gráfica de clusters...")
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=product_data,
    x='cantidad_total',
    y='ventas_totales',
    hue='segmento',  # Color por el segmento que creamos
    palette='deep',  # Paleta de colores
    s=100,           # Tamaño de los puntos
    alpha=0.7        # Transparencia
)
plt.title('Segmentación de Productos (KMeans)')
plt.xlabel('Cantidad Total Vendida')
plt.ylabel('Ventas Totales ($)')
plt.legend(title='Segmento')
plt.grid(True)
plt.show()
# Guardar la gráfica
# plot_filename = 'grafica_clusters_productos.png'
# plt.savefig(plot_filename)
# print(f"Gráfica guardada como '{plot_filename}'")