import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv
from groq import Groq
import requests

# Cargar variables de entorno
load_dotenv()

# Inicializar cliente de IA
qclient = Groq()

st.title('📊 Análisis de Votos')


@st.cache_data(show_spinner=False)
def cargar_excel(archivo_xlsx: bytes) -> pd.DataFrame:
    df = pd.read_excel(archivo_xlsx, usecols=["text"])

    df["text"] = df["text"].astype(str)
    # Eliminamos filas vacías
    df.dropna(subset=["text"], inplace=True)
    return df

@st.cache_data(show_spinner=False)
def etiquetar_votos_vectorizado(text_series: pd.Series) -> pd.Series:
    # Etiqueta los votos usando expresiones regulares en modo vectorizado para mejorar rendimiento.
    # Convertimos a minúsculas de una vez
    text_lower = text_series.str.lower()
    
    # Voto Nulo
    etiquetas = pd.Series(["Voto Nulo"] * len(text_series), index=text_series.index)
    
    # Reemplazar por "Voto Noboa" donde aplique
    noboa_mask = text_lower.str.contains(r"\bnoboa\b", na=False)
    etiquetas[noboa_mask] = "Voto Noboa"
    
    # Reemplazar por "Voto Luisa" donde aplique
    luisa_mask = text_lower.str.contains(r"\b(luisa|gonzález)\b", na=False)
    etiquetas[luisa_mask] = "Voto Luisa"
    
    return etiquetas

# Lectura del archivo

xlsx_file = st.file_uploader("Sube un archivo XLSX con los datos electorales", type="xlsx")

if not xlsx_file:
    st.info("Por favor, sube un archivo XLSX para comenzar.")
    st.stop()

try:
    # Carga del dataframe cacheado
    df = cargar_excel(xlsx_file)
    if df.empty:
        st.warning("El archivo no contiene datos o la columna 'text' está vacía.")
        st.stop()
except Exception as e:
    st.error(f"Error al procesar el archivo: {e}")
    st.stop()

# Muestreo y etiquetado
st.write(f"*Total de filas en el dataset:* {len(df):,}")
sample_size = st.slider(
    "Selecciona el tamaño de la muestra",
    min_value=500,
    max_value=len(df),
    value=min(5000, len(df)),
    step=500
)

# Extraemos muestra
sample_df = df.sample(n=sample_size, random_state=42)

# Etiquetado vectorizado
sample_df["Etiqueta"] = etiquetar_votos_vectorizado(sample_df["text"])
conteo_votos = sample_df["Etiqueta"].value_counts()

# Visualización
st.subheader("Vista previa de la muestra:")
st.dataframe(sample_df.head(10))

# Gráfico de resultados
fig, ax = plt.subplots(figsize=(5, 4))
colores = {
    "Voto Noboa": "blue",
    "Voto Luisa": "red",
    "Voto Nulo": "gray"
}
conteo_votos.plot(
    kind='bar',
    color=[colores.get(etq, 'gray') for etq in conteo_votos.index],
    ax=ax
)
ax.set_xlabel("Tipo de Voto")
ax.set_ylabel("Cantidad")
ax.set_title("Distribución de Votos en la Muestra")
st.pyplot(fig)

# Mostrar la cantidad de votos nulos y conclusión
votos_nulos = conteo_votos.get("Voto Nulo", 0)
st.write(f"*Cantidad de votos nulos en la muestra:* {votos_nulos}")

mayoria = conteo_votos.idxmax()
if mayoria == "Voto Nulo":
    conclusion = "La mayoría de los votos son nulos."
else:
    conclusion = f"La mayoría de los votos son para {mayoria}."
st.subheader("Conclusión")
st.write(conclusion)

# Sección de preguntas al bot
st.subheader("Preguntas sobre los datos")
user_question = st.chat_input("Escribe tu pregunta aquí")

if user_question:
    # Para evitar mandar todo el dataframe, creamos un pequeño resumen de la muestra.
    resumen_conteo = conteo_votos.to_dict()
    total_muestra = len(sample_df)

    # Generamos un texto breve que describa la muestra
    datos_resumen = (
        f"En una muestra de {total_muestra} filas, se obtuvo:\n"
        f"- Voto Noboa: {resumen_conteo.get('Voto Noboa', 0)}\n"
        f"- Voto Luisa: {resumen_conteo.get('Voto Luisa', 0)}\n"
        f"- Voto Nulo: {resumen_conteo.get('Voto Nulo', 0)}\n"
        f"Conclusión inicial: {conclusion}"
    )

    prompt = (
        "Eres un analista electoral. "
        "Recibes datos de una muestra de votos etiquetados en un proceso electoral. "
        "Habla únicamente en español.\n\n"
        f"Datos resumidos de la muestra:\n{datos_resumen}\n\n"
        f"Pregunta del usuario: {user_question}\n\n"
        "Proporciona la mejor respuesta posible con base en la información anterior."
    )

    with st.chat_message("user"):
        st.write(user_question)

    response_container = st.chat_message("assistant")
    response_text = ""

    try:
        # Petición en streaming al modelo
        stream_response = qclient.chat.completions.create(
            messages=[
                {"role": "system", "content": "You will only answer in Spanish"},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-specdec",
            stream=True
        )

        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
                response_container.markdown(response_text)

    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión al procesar la pregunta: {e}")
    except Exception as e:
        st.error(f"Error al procesar la pregunta: {e}")