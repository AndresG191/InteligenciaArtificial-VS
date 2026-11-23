import mysql.connector
import streamlit as st
from openai import OpenAI
import pandas as pd

st.title("IAssistant 🐦")
st.write("Tu asistente virtual.")

#inicializar
try:
    client = OpenAI(
        api_key=st.secrets["openrouter"]["api_key"],
        base_url="https://openrouter.ai/api/v1"
    )
except KeyError:
    st.error("No se encontró la API clave para OpenRouter")
    st.stop()
except Exception as e:
    st.error(f"No se pudo configurar el cliente de OpenRouter:{e}")
    st.stop()

#Database e IA seccion------------
@st.cache_resource
def init_connection():
    try:
        return mysql.connector.connect(**st.secrets["mysql"])
    except mysql.connector.Error as err:
        st.error(f"Error al conectar a la base de datos: {err}")
        return None
    
def run_quey(query):
    conn= init_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
                return df
        except mysql.connector.Error as err:
            st.error(f"Error de SQL: {err}")
            return None
        return None
    
#esta funcion es el cerebro de la IA
def generate_sql_from_prompt(user_prompt):

    system_prompt="""
    La base de datos que se te otorgará, contiene informacion sobre productos de una tienda
    llamada 'supermarket_dataset_power'.

    El esquema de la tabla de 'supermarket_dataset_power' es la siguiente:
    - 'categoria' (VARCHAR): Pasillo en donnde se encuentra el producto ejemplo: ('carnes', 'frutas', 'verduras')
    - 'nombre_producto' (VARCHAR): Nombre del Producto.
    - 'cantidad' (INT): Cantidad de productos que los usuarios compraron.
    - 'ventas_totales' (DECIMAL): Son las ventas totales que existen.
    - 'precio_unitario' (DECIMAL): Precio unitario que existe en cada producto.
    - 'total_articulos' (DECIMAL): La cantidad de articulos que existen dentro de una transaccion.
    - 'descuento' (VARCHAR): Identifica si existe descuento en los productos.
    - 'valor_descuento' (DECIMAL): La Cantidad de descuento que tiene cada articulo en caso de que lo tenga.
    - 'fecha_caducidad' (DATE): Indica la fecha de caducidad de un producto.
    """

    try:
        response = client.chat.completions.create(
            model = "deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature = 0
        )
        sql_query = response.choices[0].messages.content.strip()

        if sql_query.startswith("'''sql"):
            sql_query = sql_query[5:]
        if sql_query.endswith("'''"):
            sql_query = sql_query[:-3]

        return sql_query.strip()

    except Exception as e:
        st.error(f"Error al querer generar consulta SQL con modelo de IA: {e}")
        return None

#interfaz de chat

if "messages" not in st.session_state:
    st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"])
            else:
                st.markdown(message["content"])

if prompt := st.chat_input("¿Que te gustaria preguntar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            sql_query = generate_sql_from_prompt(prompt)

            if sql_query and sql_query.upper() != 'INVALID':
                with st.expander("Consulta SQL generada"):
                    st.code(sql_query, language="sql")

                result_df = run_quey(sql_query)

                if result_df is not None:
                    if not result_df.empty:
                        st.dataframe(result_df)
                        st.session_state.messages.append({"role": "assistant", 
                                                          "content": result_df})
                    else:
                        response_text = "La consulta no pudo devolver resultados."
                        st.write(response_text)
                        st.session_state.messages.append({"role": "assistant", 
                                                          "content": "Ocurrió un herror al ejecutar la consulta."})
                elif sql_query and sql_query.upper() == 'INVALID':
                    response_text = "Lo siento, no puedo responder a esa pregunta. Porfavor solamente haz preguntas relacionadas a los productos en tienda."
                    st.write(response_text)
                    st.session_state.message.append({"role": "assistant", "content": response_text})
                else:
                    st.session_state.messages.append({"role": "assistant", "content":"Hubo un error al generar la consulta" })