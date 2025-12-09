# VaneBot

Este proyecto es un asistente conversacional basado en la arquitectura RAG. Permite "chatear" con un documento PDF utilizando inteligencia artificial 100% local, garantizando privacidad y funcionamiento sin internet.

## Requisitos Previos

1.  **Python 3.9** o superior.
2.  **Ollama**: Debe estar instalado y ejecutándose en segundo plano. [Descargar Ollama aquí](https://ollama.com).
3.  **Memoria RAM**: Mínimo 4GB (Optimizado para funcionar en equipos de 8GB).

##  Instalación

### 1. Preparar el modelo de IA (Solo una vez)
Abre tu terminal y ejecuta el siguiente comando para descargar la versión ligera de Llama 3.2:

```bash
ollama pull llama3.2:1b

Configurar el entorno de Python:

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Activar entorno (Mac/Linux)
source venv/bin/activate

Instalar dependencias:

Instala las librerías necesarias ejecutando:

Bash

pip install -r requirements.txt

Ejecución
Asegúrate de que la aplicación de escritorio Ollama esté abierta (busca el ícono en tu barra de tareas).

Asegúrate de que el archivo conocimiento.pdf esté en la carpeta del proyecto.

Ejecuta el comando:

Bash

streamlit run app.py
El navegador se abrirá automáticamente con la interfaz del ChatBot.

 Tecnologías Usadas
Frontend: Streamlit

Orquestación: LangChain

LLM (Cerebro): Meta Llama 3.2 (1B) vía Ollama

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Base de Datos Vectorial: ChromaDB
