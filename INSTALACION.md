# 🛠️ Guía de instalación · Asistente de campañas Ben & Frank

Esta guía explica cómo descargar el proyecto desde GitHub y dejarlo corriendo en tu computador, paso a paso. No necesitas experiencia previa con Git ni con Ollama.

> ⏱️ Tiempo estimado: 30–45 minutos. La mayor parte es la descarga de los modelos.

---

## 0. Antes de empezar: ¿mi equipo sirve?

| Recurso | Mínimo | Recomendado (equipo de prueba) |
|---|---|---|
| Sistema operativo | Windows 10/11, macOS o Linux | Windows 11 |
| RAM | 16 GB | 16 GB |
| GPU | Opcional (sin GPU funciona, pero lento) | NVIDIA con 8 GB+ de VRAM (probado en RTX 3060 12 GB) |
| Espacio en disco | 15 GB libres | 20 GB libres |
| Internet | Solo para instalar y descargar modelos | — |

El modelo principal (`qwen2.5:7b`) ocupa unos 5 GB. Si tu equipo tiene poca memoria, puedes usar `llama3.2:3b` (ver [Solución de problemas](#-solución-de-problemas)).

---

## 1. Instalar los programas necesarios

Instala estos tres programas. Si ya los tienes, salta al paso 2.

### 1.1 Python 3.10 o superior
- Descárgalo de [python.org/downloads](https://www.python.org/downloads/).
- **Windows:** en el instalador, marca la casilla **"Add Python to PATH"** antes de pulsar *Install*.

Comprueba la instalación en una terminal nueva:
```bash
python --version
```
Debe mostrar `Python 3.10` o superior.

### 1.2 Git
- Descárgalo de [git-scm.com](https://git-scm.com/downloads) e instálalo con las opciones por defecto.

```bash
git --version
```

> ¿No quieres instalar Git? Puedes descargar el proyecto como ZIP (ver paso 2, opción B).

### 1.3 Ollama (el LLM local)
- Descárgalo de [ollama.com/download](https://ollama.com/download) e instálalo.
- En Windows y macOS queda corriendo en segundo plano (verás el ícono de la llama en la barra de tareas).

```bash
ollama --version
```

---

## 2. Descargar el proyecto

Abre una terminal (en Windows: **PowerShell**) en la carpeta donde quieras guardar el proyecto, por ejemplo `Documentos`.

### Opción A: con Git (recomendada)
```bash
git clone https://github.com/TU_USUARIO/asistente-campanas-ia-generativa.git
cd asistente-campanas-ia-generativa
```

### Opción B: sin Git (ZIP)
1. En la página del repositorio en GitHub, pulsa el botón verde **Code → Download ZIP**.
2. Descomprime el archivo.
3. Abre una terminal dentro de la carpeta descomprimida (la que contiene `app.py`).

> ✅ A partir de aquí, **todos los comandos se ejecutan desde la carpeta raíz del proyecto**, es decir, la que contiene `app.py`.

---

## 3. Crear el entorno virtual e instalar dependencias

El entorno virtual aísla las librerías del proyecto para que no choquen con otras instalaciones de Python.

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Sabrás que el entorno está activo porque la terminal muestra `(.venv)` al inicio de la línea. **Cada vez que abras una terminal nueva para trabajar en el proyecto, vuelve a activarlo.**

> 💡 Si usas [`uv`](https://docs.astral.sh/uv/), puedes reemplazar los comandos anteriores por `uv venv` y `uv pip install -r requirements.txt`.

La instalación puede tardar varios minutos.

---

## 4. Descargar los modelos de Ollama

Con Ollama abierto, ejecuta:

```bash
ollama pull qwen2.5:7b       # modelo principal (obligatorio)
ollama pull llama3.2:3b      # opcional: para comparar arquitecturas
ollama pull llama3.1:8b      # opcional: juez local de la evaluación
```

Comprueba que quedaron instalados:
```bash
ollama list
```

> La primera vez que se construye el índice del RAG también se descarga el modelo de embeddings local. Es automático, solo necesita internet esa vez.

---

## 5. Configurar las claves (archivo `.env`)

El proyecto usa **LangSmith** para registrar las trazas y, opcionalmente, **OpenRouter** para el juez independiente de la evaluación.

1. Crea una cuenta gratuita en [smith.langchain.com](https://smith.langchain.com) y genera una API key en **Settings → API Keys**.
2. (Opcional) Crea una cuenta en [openrouter.ai](https://openrouter.ai) y genera una API key, solo si vas a correr la evaluación con juez externo.
3. Copia el archivo de ejemplo:

   **Windows**
   ```powershell
   copy .env.example .env
   ```
   **macOS / Linux**
   ```bash
   cp .env.example .env
   ```

4. Abre `.env` con cualquier editor de texto y completa los valores:
   ```
   LANGSMITH_API_KEY=lsv2_tu_clave_aqui
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=asistente-campanas-ben-frank
   OPENROUTER_API_KEY=sk-or-tu_clave_aqui
   ```

> 🔒 El archivo `.env` está en `.gitignore`: **nunca lo subas a GitHub** ni compartas tus claves.

---

## 6. Preparar los datos (solo la primera vez)

### 6.1 Base de Q3 e índice del RAG (notebook)
1. Abre la carpeta del proyecto en **VS Code** (o ejecuta `jupyter notebook` con el entorno activo).
2. Abre el notebook que está en `notebooks/`.
3. Arriba a la derecha, en **Select Kernel**, elige el Python del entorno `.venv` del proyecto.
4. Ejecuta las celdas de la sección de **ingesta de datos y construcción del RAG**. Esto crea la base DuckDB de Q3 y el índice Chroma.

> ⚠️ Cierra la app de Streamlit mientras ejecutas estas celdas. DuckDB bloquea el archivo de la base y no permite que dos procesos lo usen a la vez.

### 6.2 Base de Q2 (para el servidor MCP)
Con el entorno activo, desde la raíz del proyecto:
```bash
python generar_datos_q2.py
```

---

## 7. Abrir la interfaz

Con el entorno activo y Ollama corriendo:

```bash
streamlit run app.py
```

Se abrirá el navegador en **http://localhost:8501**. Si no se abre solo, copia esa dirección en el navegador.

**Windows:** también puedes darle doble clic a `iniciar_asistente.bat`.

El servidor MCP de Q2 lo inicia la app automáticamente cuando hace falta; no tienes que arrancarlo a mano.

Para detener la app, vuelve a la terminal y pulsa `Ctrl + C`.

---

## 8. Comprobar que todo funciona

Prueba estas preguntas en el chat:

1. *¿Qué plataforma tuvo el menor costo por cita en US durante agosto?* → usa las herramientas sobre DuckDB (Q3).
2. *¿Qué es el VTR?* → usa el RAG.
3. *Compara Q2 vs Q3.* → usa el servidor MCP.

Debajo de cada respuesta debe aparecer la **ruta de agentes** y el desplegable **💰 Costo por agente**. En [smith.langchain.com](https://smith.langchain.com), dentro del proyecto `asistente-campanas-ben-frank`, deberían aparecer las trazas.

---

## 🧯 Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `python no se reconoce como comando` | Python no quedó en el PATH | Reinstala Python marcando **Add Python to PATH**, o usa `py` en lugar de `python` |
| PowerShell: *"la ejecución de scripts está deshabilitada en este sistema"* al activar `.venv` | Política de ejecución de Windows | Ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y responde `S` |
| `streamlit no se reconoce` | El entorno virtual no está activo | Activa `.venv` (paso 3) y vuelve a intentar |
| `No module named 'src'` | Estás en otra carpeta | Ve a la carpeta raíz del proyecto (la que tiene `app.py`) con `cd` |
| `ModuleNotFoundError` de otra librería | Dependencias incompletas | Con el entorno activo: `pip install -r requirements.txt` |
| `Connection refused` / no conecta con Ollama | Ollama no está corriendo | Abre la app de Ollama, o ejecuta `ollama serve` en otra terminal |
| `model "qwen2.5:7b" not found` | Falta descargar el modelo | `ollama pull qwen2.5:7b` |
| Error al abrir la base DuckDB (archivo bloqueado) | El notebook y la app usan la base a la vez | Cierra una de las dos (reinicia el kernel del notebook o detén Streamlit) |
| Error porque no existe la base o el índice Chroma | No se hizo el paso 6 | Ejecuta la sección de ingesta del notebook y `generar_datos_q2.py` |
| Respuestas muy lentas o falta de memoria | El equipo no tiene GPU o tiene poca RAM/VRAM | Cierra otros programas, o cambia el modelo a `llama3.2:3b` en `src/config.py` (responde peor con herramientas) |
| No aparecen trazas en LangSmith | Clave o variables mal escritas | Revisa `.env`: `LANGSMITH_API_KEY` y `LANGSMITH_TRACING=true`. Reinicia la app |
| El puerto 8501 está ocupado | Otra app de Streamlit abierta | Ciérrala, o usa `streamlit run app.py --server.port 8502` |

---

## 🔄 Para actualizar a la última versión

Si descargaste con Git:
```bash
git pull
pip install -r requirements.txt
```
Si hubo cambios en los datos o en el corpus del RAG, vuelve a ejecutar el paso 6.

---

## ✅ Resumen rápido (para quien ya tiene todo instalado)

```powershell
git clone https://github.com/TU_USUARIO/asistente-campanas-ia-generativa.git
cd asistente-campanas-ia-generativa
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
copy .env.example .env        # y completar las claves
# Ejecutar la ingesta del notebook en notebooks/
python generar_datos_q2.py
streamlit run app.py
```
