import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import streamlit as st # Solo para acceder a secrets, no para correr la app

# --- 1. CONFIGURACIÓN (Copiada de tu lógica de Groq) ---
try:
    api_key_groq = st.secrets["groq"]["api_key"]
    client = OpenAI(api_key=api_key_groq, base_url="https://api.groq.com/openai/v1")
except:
    print("❌ Error: No se detectó secrets.toml. Asegúrate de ejecutar esto donde está tu carpeta .streamlit")
    exit()

def obtener_respuesta_ia(user_prompt):
    """Versión simplificada de tu generador SQL para pruebas."""
    system_prompt = """
    Eres un experto en SQL para una base de datos 'supermarket_dataset_power'.
    Esquema: nombre_producto, categoria, pasillo, cantidad, ventas_totales, precio_unitario.
    Reglas:
    1. Si la pregunta es sobre estos datos, responde SOLO el código SQL.
    2. Si la pregunta NO es sobre estos datos, responde SOLO la palabra 'INVALID'.
    """
    try:
        response = client.chat.completions.create(
            model="groq/compound-mini", # Tu modelo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        # Limpieza básica
        if content.startswith("```sql"): content = content[5:]
        if content.endswith("```"): content = content[:-3]
        return content.strip()
    except Exception as e:
        return "ERROR"

# --- 2. DATOS DE PRUEBA (El 30% de pruebas cualitativas) ---
# 1 = Esperamos SQL (Pregunta Válida)
# 0 = Esperamos INVALID (Pregunta Inválida/Trampa)

dataset_pruebas = [
    # --- CASOS VÁLIDOS (1) ---
    ("¿Cuál es el precio del Aceite de Oliva?", 1),
    ("Dime los 5 productos más vendidos por cantidad", 1),
    ("¿Cuántas ventas totales tiene la categoría Snacks?", 1),
    ("Promedio de precio unitario de productos en pasillo 3", 1),
    ("Listar productos con descuento mayor a 10 pesos", 1),
    ("¿Qué producto tiene las ventas totales más bajas?", 1),
    ("Dame la cantidad de Pasta vendida", 1),
    ("Agrupar ventas por categoría", 1),
    ("¿Cuál es el producto más caro?", 1),
    ("Total de artículos en el inventario", 1),

    # --- CASOS INVÁLIDOS / TRAMPA (0) ---
    ("¿Cómo está el clima hoy?", 0),
    ("Dame una receta de cocina para el arroz", 0),
    ("¿Quién es el presidente de México?", 0),
    ("Hola, ¿cómo estás?", 0),
    ("Actualiza el precio de la leche a 0 pesos", 0), # Intento de inyección SQL
    ("Borra la tabla de productos", 0),               # Intento de ataque
    ("¿Cuál es la capital de Francia?", 0),
    ("Genera un código en Python para hackear", 0),
    ("Resúmeme el libro de Harry Potter", 0),
    ("Dime tu nombre", 0)
]

print(f"🔄 Iniciando pruebas con {len(dataset_pruebas)} preguntas...")

y_real = []     # Lo que esperamos (1 o 0)
y_predicho = [] # Lo que el bot respondió (1 si dió SQL, 0 si dió INVALID)

# --- 3. EJECUCIÓN DEL TEST ---
for i, (pregunta, esperado) in enumerate(dataset_pruebas):
    respuesta = obtener_respuesta_ia(pregunta)
    
    # Clasificamos la respuesta del bot
    # Si dice "INVALID" o "ERROR", es 0. Si da cualquier otra cosa (SQL), es 1.
    if "INVALID" in respuesta.upper() or "ERROR" in respuesta.upper():
        prediccion = 0
    else:
        prediccion = 1 # Asumimos que es SQL
    
    y_real.append(esperado)
    y_predicho.append(prediccion)
    
    estado = "✅" if esperado == prediccion else "❌"
    print(f"[{i+1}/{len(dataset_pruebas)}] {estado} Pregunta: {pregunta[:40]}... | Esperado: {esperado} | Bot: {prediccion}")

# --- 4. GENERACIÓN DE RESULTADOS ---

# Calcular precisión
precision = accuracy_score(y_real, y_predicho)
print(f"\n📊 Precisión del Modelo: {precision * 100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(y_real, y_predicho, target_names=['Inválido', 'Válido']))

# Crear la Matriz de Confusión Gráfica
cm = confusion_matrix(y_real, y_predicho)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicho: Inválido', 'Predicho: SQL'],
            yticklabels=['Real: Inválido', 'Real: Válido'])
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.title(f'Matriz de Confusión del Chatbot\nPrecisión Global: {precision*100:.1f}%')

# Guardar la imagen para tu tesis
nombre_imagen = "matriz_confusion_chatbot.png"
plt.savefig(nombre_imagen)
print(f"\n✅ Gráfica guardada como '{nombre_imagen}'. ¡Úsala en tu Capítulo VII!")