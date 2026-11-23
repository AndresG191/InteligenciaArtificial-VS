import mysql.connector
import streamlit as st
import pandas as pd
from openai import OpenAI  # Importamos OpenAI (para Groq)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from EntrenamientoDatos import EntrenamientoDatos

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="IAssistant", page_icon="🐦", layout="wide")

st.title("IAssistant 🐦")
st.write("Tu asistente virtual. Hazme una pregunta sobre los datos de la tienda.")

# --- INICIALIZACIÓN DE LA API (MODIFICADO PARA GROQ) ---
@st.cache_resource
def init_groq_client():
    """Inicializa el cliente de Groq."""
    try:
        # Leemos la clave de Groq desde secrets.toml
        api_key_groq = st.secrets["groq"]["api_key"]
        
        if not api_key_groq:
            st.error("Error: 'api_key' de [groq] está vacía en secrets.toml.")
            st.stop()
            
        # Apuntamos el cliente de OpenAI a la URL de Groq
        client = OpenAI(
            api_key=api_key_groq,
            base_url="https://api.groq.com/openai/v1"
        )
        st.success("Modelo de Groq (IAssistant 🐦) inicializado.")
        return client
        
    except KeyError:
        st.error("Error: La sección [groq] o 'api_key' no se encontró en .streamlit/secrets.toml.")
        st.stop()
    except Exception as e:
        st.error(f"Error general al inicializar Groq: {e}")
        st.stop()

# Llama a la función de inicialización
client = init_groq_client()

# --- Funciones de Base de Datos ---
# (Esta sección se queda EXACTAMENTE IGUAL)

@st.cache_resource
def init_connection():
    """Inicializa la conexión a la base de datos MySQL."""
    try:
        return mysql.connector.connect(**st.secrets["mysql"])
    except mysql.connector.Error as err:
        st.error(f"Error al conectar con la base de datos: {err}")
        return None

def run_query(query):
    """Ejecuta una consulta SQL y devuelve los resultados como un DataFrame."""
    conn = init_connection()
    if conn:
        try:
            if not conn.is_connected():
                conn.reconnect()
            with conn.cursor() as cursor:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
                return df
        except mysql.connector.Error as err:
            st.error(f"Error de SQL: {err}")
            return None
    return None

# --- Lógica de IA (Usando tu archivo externo) ---
@st.cache_resource
def iniciar_motor_ia():
    # Crea la instancia de tu clase que está en el otro archivo
    motor = EntrenamientoDatos('supermarket_dataset_power.csv')
    # Entrena automáticamente al iniciar (siguiendo la regla 40/30/30)
    motor.cargar_y_dividir_datos()
    motor.entrenar_modelo()
    return motor

@st.cache_data
def generar_sugerenciaML():
    st.info("Consultando al motor de IA (Clase EntrenamientoDatos)...", icon="🧠")
    
    # Obtenemos el motor ya entrenado
    motor = iniciar_motor_ia()
    
    # Validación (30%)
    score = motor.validar_resultados()
    
    # Estrategia
    prod_bajo, prod_alto = motor.obtener_estrategia_cross_selling()
    
    if prod_bajo and prod_alto:
        mensaje = (
            f"### Reporte de Validación del Modelo\n"
            f"El modelo ha sido entrenado y validado siguiendo la metodología **40% Entrenamiento / 30% Validación**.\n\n"
            f"- **Calidad del Agrupamiento (Silhouette Score):** `{score:.4f}`\n"
            f"---\n"
            f"### Estrategia de Venta Cruzada Sugerida\n"
            f"Basado en el análisis de patrones de venta:\n\n"
            f"Hemos detectado que el producto **{prod_bajo}** tiene un desempeño bajo.\n"
            f"**Recomendación:** Crear un paquete promocional combinándolo con **{prod_alto}** (Producto Estrella) para impulsar su rotación."
        )
    else:
        mensaje = "Error: No se pudo generar la sugerencia. Verifica los datos."
        
    return mensaje

def generate_sql_from_prompt(user_prompt: str):

    # El prompt del sistema es el mismo
    system_prompt = """
    Eres un asistente experto en MySQL. Tu tarea es generar una consulta SQL basada en la pregunta del usuario.
    La tabla es 'supermarket_dataset_power' y su esquema es: 
    - `nombre_producto` (VARCHAR)
    - `categoria` (VARCHAR)
    - `pasillo` (VARCHAR)
    - `cantidad` (INT): Unidades vendidas
    - `ventas_totales` (DECIMAL): Ingreso total
    - `precio_unitario` (DECIMAL)
    
    Reglas:
    1. SOLO responde con el código de la consulta SQL. Nada más.
    2. Si la pregunta no se puede responder con SQL o es irrelevante, responde con 'INVALID'.
    """
    
    try:
        # Usamos el cliente global 'client' de Groq
        response = client.chat.completions.create(
            
            # --- AQUÍ ESTÁ EL CAMBIO ---
            # Modelo actualizado, gratuito y activo en Groq
            model="groq/compound-mini", 
            # ---------------------------

            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        sql_query = response.choices[0].message.content.strip()
        
        # Limpiar la respuesta (esta parte se queda igual)
        if sql_query.startswith("```sql"):
            sql_query = sql_query[5:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
            
        return sql_query.strip()
    
    except Exception as e:
        st.error(f"Error al generar la consulta SQL con el modelo de Groq: {e}")
        return None

# --- Interfaz del Chat ---
# (Esta sección se queda EXACTAMENTE IGUAL)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        else:
            st.markdown(message["content"])

st.warning("**Advertencia:** Este es un proyecto de demostración.", icon="⚠️")

if prompt := st.chat_input("¿Qué te gustaría saber?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            
            # Lógica de activación de ML (se queda igual)
            if "promocion" in prompt.lower() or "sugerencia" in prompt.lower() or "oferta" in prompt.lower():
                response_content = generar_sugerenciaML()
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content}) 
                
            else:
                # Generar una consulta sql
                sql_query = generate_sql_from_prompt(prompt)
                
                if sql_query and sql_query.upper() != 'INVALID':
                    with st.expander("Consulta SQL generada"):
                        st.code(sql_query, language="sql")
                    
                    result_df = run_query(sql_query)
                    
                    if result_df is not None and not result_df.empty:
                        st.dataframe(result_df)
                        st.session_state.messages.append({"role": "assistant", "content": result_df})
                    else:
                        response_text = "La consulta no devolvió resultados."
                        st.write(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                elif sql_query and sql_query.upper() == 'INVALID':
                    response_text = "Lo siento, no puedo responder esa pregunta. Por favor, haz una consulta relacionada con los productos de la tienda."
                    st.write(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Hubo un error al generar o ejecutar la consulta."})