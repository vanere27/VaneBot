import streamlit as st
import os

# --- Importaciones ---
from langchain_community.document_loaders import PyPDFLoader
# Splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
# Embeddings Locales (Rápido y ligero)
from langchain_community.embeddings import HuggingFaceEmbeddings
# LLM Local (Ollama)
from langchain_ollama import ChatOllama

# --- Configuración de la Página ---
st.set_page_config(page_title="ChatBot con Ollama")
st.title("VaneBot  ")

# --- Procesamiento del PDF ---
@st.cache_resource
def configurar_vectorstore():
    if not os.path.exists("conocimiento.pdf"):
        st.error("Por favor, sube el archivo 'conocimiento.pdf' a la carpeta del proyecto.")
        return None
    
    # 1. Cargar PDF
    loader = PyPDFLoader("conocimiento.pdf")
    docs = loader.load()
    
    # 2. Dividir texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    # 3. Crear Embeddings (Usamos HuggingFace local para no saturar Ollama)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Crear Base de Datos
    vector_store = Chroma.from_documents(splits, embeddings)
    return vector_store

# Inicializar la base de conocimiento
with st.spinner("Procesando documento..."):
    try:
        vector_store = configurar_vectorstore()
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
        vector_store = None

# --- Chat con Ollama ---
if vector_store:
    # Configuramos Ollama (Debe estar instalado y corriendo en tu PC)
    # Si tu PC es lenta, cambia 'llama3.1' por 'llama3.2' (es más ligero)
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Pregunta algo sobre el documento..."):
        # Guardar y mostrar pregunta
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("pensando..."):
                try:
                    # 1. Buscar contexto
                    docs_relacionados = vector_store.similarity_search(prompt, k=3)
                    contexto = "\n\n".join([doc.page_content for doc in docs_relacionados])
                    
                    # 2. Construir prompt
                    prompt_final = f"""
                    Instrucción: Eres un asistente útil. Responde la pregunta basándote 
                    ÚNICAMENTE en el siguiente contexto proporcionado.
                    
                    CONTEXTO:
                    {contexto}
                    
                    PREGUNTA:
                    {prompt}
                    """
                    
                    # 3. Invocar a Ollama
                    respuesta_completa = ""
                    placeholder = st.empty() # Para efecto de escritura tipo streaming
                    
                    # Usamos stream para que se vea como escribe poco a poco
                    for chunk in llm.stream(prompt_final):
                        respuesta_completa += chunk.content
                        placeholder.markdown(respuesta_completa + "▌")
                    
                    placeholder.markdown(respuesta_completa)
                    
                    # 4. Guardar respuesta
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
                    
                except Exception as e:
                    st.error(f"Error de conexión con Ollama. ¿Está corriendo la aplicación?: {e}")