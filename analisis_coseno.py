import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter, OrderedDict
from nltk.tokenize import TreebankWordTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Tokenizador y stopwords
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('spanish'))

# Función para cargar el archivo CSV desde GitHub
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/cuerpo_documentos_p2_gr_4.csv'
    return pd.read_csv(url)

# Función para calcular la similitud coseno
def sim_coseno(vec1, vec2):
    vec1 = np.array(list(vec1.values()))
    vec2 = np.array(list(vec2.values()))
    dot_prod = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0  # Evitar división por cero
    return dot_prod / (norm_1 * norm_2)

# Función para vectorizar documentos
def BoW_vec(docs: list, tokenizer):
    doc_tokens = []
    for doc in docs:
        tokens = tokenizer.tokenize(doc.lower())
        tokens = [word for word in tokens if word not in stop_words]  # Eliminar stopwords
        doc_tokens += [sorted(tokens)]
    all_doc_tokens = sum(doc_tokens, [])
    lexico = sorted(set(all_doc_tokens))
    zero_vector = OrderedDict((token, 0) for token in lexico)
    document_bow_vectors = []
    for i, doc in enumerate(docs):
        vec = zero_vector.copy()
        tokens = tokenizer.tokenize(doc.lower())
        tokens = [word for word in tokens if word not in stop_words]  # Eliminar stopwords
        token_counts = Counter(tokens)
        for key, value in token_counts.items():
            vec[key] = value
        document_bow_vectors.append(vec)
    return document_bow_vectors

# Cargar datos
data = load_data()

# Mostrar contenido del archivo en una tabla
st.title('Análisis de Documentos')
st.write('Contenido del archivo:')
st.dataframe(data)

# Inputs de texto
palabra = st.text_input('Input de palabra')
oracion = st.text_input('Input de oración')

# Procesar input de palabra
if palabra:
    palabra = palabra.lower()
    data['frecuencia'] = data['body'].apply(lambda x: x.lower().split().count(palabra))
    doc_max_freq = data.loc[data['frecuencia'].idxmax()]
    st.write(f"Documento con mayor frecuencia de la palabra '{palabra}':")
    st.write(f"Título: {doc_max_freq['title']}")
    st.write(f"Frecuencia: {doc_max_freq['frecuencia']}")

# Procesar input de oración
if oracion:
    oracion_tokens = tokenizer.tokenize(oracion.lower())
    oracion_tokens = [word for word in oracion_tokens if word not in stop_words]

    # Vectorizar documentos y oración
    document_BoW_vector = BoW_vec(docs=data['body'].to_list(), tokenizer=tokenizer)
    oracion_vec = Counter(oracion_tokens)
    oracion_vec = OrderedDict((token, oracion_vec[token]) for token in document_BoW_vector[0].keys())

    # Calcular similitud coseno
    similitudes = [sim_coseno(oracion_vec, doc_vec) for doc_vec in document_BoW_vector]
    doc_max_similitud = data.iloc[np.argmax(similitudes)]
    max_similitud = max(similitudes)

    st.write(f"Documento más similar a la oración '{oracion}' respecto a la similitud coseno:")
    st.write(f"Título: {doc_max_similitud['title']}")
    st.write(f"Similitud coseno: {max_similitud}")

    # Calcular suma de frecuencias
    data['suma_frecuencias'] = data['body'].apply(lambda x: sum([x.lower().split().count(token) for token in oracion_tokens]))
    doc_max_suma_frecuencias = data.loc[data['suma_frecuencias'].idxmax()]
    max_suma_frecuencias = max(data['suma_frecuencias'])

    st.write(f"Documento con la mayor suma de frecuencias de tokens de la oración '{oracion}':")
    st.write(f"Título: {doc_max_suma_frecuencias['title']}")
    st.write(f"Suma de frecuencias: {max_suma_frecuencias}")