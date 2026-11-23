import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import os

class EntrenamientoDatos:
    def __init__(self, data_path='supermarket_dataset_power.csv'):
        """
        Inicializa la clase de entrenamiento.
        """
        self.data_path = data_path
        self.df_full = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.n_clusters = 3  # Definimos 3 clusters: Bajo, Medio, Alto
        self.is_trained = False

    def cargar_y_dividir_datos(self):
        """
        Carga el dataset y aplica la división 40/30/30 solicitada por el Director.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.data_path}")

        # 1. Cargar datos completos
        self.df_full = pd.read_csv(self.data_path)
        
        # Limpieza básica
        if self.df_full.isnull().sum().sum() > 0:
            self.df_full.fillna(0, inplace=True)

        # 2. Primera división: Separar el 30% para PRUEBAS (Test)
        # El 70% restante (temp_df) será para Entrenamiento y Validación
        temp_df, self.test_df = train_test_split(
            self.df_full, test_size=0.30, random_state=42
        )

        # 3. Segunda división: Separar el 70% restante.
        # Queremos que el 'train' original sea el 40% del total.
        # Dividimos el restante en proporción 57% / 43% aprox.
        self.train_df, self.val_df = train_test_split(
            temp_df, test_size=0.42, random_state=42 
        )
        
        return {
            "train": len(self.train_df),
            "val": len(self.val_df),
            "test": len(self.test_df)
        }

    def entrenar_modelo(self):
        """
        Entrena el modelo K-Means usando SOLO el conjunto de Entrenamiento (40%).
        """
        if self.train_df is None:
            self.cargar_y_dividir_datos()

        # Seleccionamos las columnas numéricas relevantes para el agrupamiento
        X_train = self.train_df[['cantidad', 'ventas_totales']]
        
        # Ajustamos el escalador solo con los datos de entrenamiento
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Entrenamos KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        self.is_trained = True
        
        return "Modelo entrenado exitosamente con el set de Entrenamiento."

    def validar_resultados(self):
        """
        Valida el modelo con el conjunto de Validación (30%) y retorna el Silhouette Score.
        """
        if not self.is_trained:
            return "Error: El modelo no ha sido entrenado."

        X_val = self.val_df[['cantidad', 'ventas_totales']]
        # Usamos el escalador ya ajustado (transform, no fit_transform)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Predecimos los clusters para validación
        labels = self.kmeans.predict(X_val_scaled)
        
        # Calculamos la métrica de calidad
        score = silhouette_score(X_val_scaled, labels)
        return score

    def obtener_estrategia_cross_selling(self):
        """
        Genera la sugerencia de venta cruzada usando los patrones aprendidos.
        """
        if not self.is_trained:
            return None, None

        # Para la sugerencia final, analizamos todo el dataset usando el conocimiento adquirido
        df_analysis = self.df_full.copy()
        X_full = df_analysis[['cantidad', 'ventas_totales']]
        X_full_scaled = self.scaler.transform(X_full)
        
        df_analysis['cluster'] = self.kmeans.predict(X_full_scaled)
        
        # Identificar clusters (Bajo vs Alto rendimiento)
        cluster_stats = df_analysis.groupby('cluster')['cantidad'].mean().sort_values()
        worst_cluster = cluster_stats.index[0]
        best_cluster = cluster_stats.index[-1]
        
        # Seleccionar productos de ejemplo
        prod_bajo = df_analysis[df_analysis['cluster'] == worst_cluster].sample(1).iloc[0]['nombre_producto']
        prod_alto = df_analysis[df_analysis['cluster'] == best_cluster].sample(1).iloc[0]['nombre_producto']
        
        return prod_bajo, prod_alto