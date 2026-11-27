# 👓 Asistente de campañas · Ben & Frank

Asistente conversacional en español que analiza campañas de pauta en redes sociales. Corre con un **LLM local (Ollama)** y está construido con **LangChain/LangGraph**. Usa **RAG**, herramientas deterministas sobre **DuckDB**, un servidor **MCP** y trazabilidad con **LangSmith**. Lo coordina una **arquitectura multiagente**.

> Proyecto final de la materia **Tendencias de IA Generativa**, Maestría en Ciencia de Datos, Universidad ICESI.

⚠️ **Datos sintéticos.** Por confidencialidad, todos los datos son sintéticos y usan la marca Ben & Frank como caso de estudio. No contienen información real de ninguna empresa ni de ningún anunciante.

---

## ✨ Qué hace

Responde preguntas de negocio sobre campañas de **Facebook, TikTok, Snapchat, Pinterest y Reddit** en **Estados Unidos, Canadá y Puerto Rico**:

- *¿Qué plataforma tuvo el menor costo por cita en US durante agosto?*
- *¿Detectas problemas de tracking o gasto sin entrega?*
- *¿Cuáles son los 3 creativos de TikTok con peor VTR?*
- *¿El lower funnel cumplió el objetivo de costo por cita del brief?*
- *Grafica la evolución semanal del costo por cita por plataforma en US.*
- *Compara Q2 vs Q3.*

| Periodo | Fuente de datos |
|---|---|
| **Q3 2026** (jul–sep) | DuckDB local, consultado con las tools nativas del agente |
| **Q2 2026** (abr–jun) | Servidor MCP local `campanas-q2` (stdio) |

Las cifras siempre salen de consultas a los datos. El LLM no las calcula ni las inventa. El RAG aporta el contexto conceptual: brief, objetivos y definiciones de métricas.

---

## 🏗️ Arquitectura

```
                         ┌──────────────────────┐
   Usuario ──► Streamlit │      🧭 Supervisor    │  (router)
                         └──────────┬───────────┘
            ┌──────────────┬────────┴──────┬────────────────┐
            ▼              ▼               ▼                ▼
       📊 Datos      🚨 Anomalías    📚 Conocimiento   🔌 Datos Q2 (MCP)
     (DuckDB Q3)     (DuckDB Q3)      (RAG · Chroma)    (servidor MCP)
            └──────────────┴────────┬──────┴────────────────┘
                                    ▼
                              ✍️ Redactor
                                    ▼
                               🔎 Crítico ──► (si rechaza, el redactor reescribe)
                                    ▼
                                Respuesta
                     (todo se traza en LangSmith)
```

**Patrones de diseño multiagente**
- **Router:** el supervisor clasifica la pregunta y la envía al especialista que corresponde.
- **Paralelización:** en las preguntas mixtas, varios especialistas trabajan a la vez (por ejemplo, Datos + Conocimiento).
- **Evaluador–optimizador:** el crítico revisa la respuesta del redactor y pide correcciones si hace falta.

La interfaz también permite usar el **agente único** (versión anterior) para comparar las dos arquitecturas.

### Herramientas

| Herramienta | Qué hace |
|---|---|
| `consultar_metricas` | Consulta agregada de métricas sobre DuckDB (inversión, impresiones, clics, CTR, VTR, citas, costo por cita…) |
| `buscar_conocimiento` | Recupera contexto del corpus RAG (Chroma) |
| `generar_grafico` | Genera gráficos Plotly a partir de `consultar_metricas` |
| Detección de anomalías | Señala problemas de tracking y gasto sin entrega |
| Tools del servidor MCP | Exponen las métricas de Q2 |

Las herramientas pasan por **guardarraíles** compartidos que validan los argumentos y registran cada ejecución.

### Cálculo de costo

Para cada respuesta, la app muestra el **costo por agente**: llamadas, tokens, un **costo sombra en USD** y un estimado de energía. El costo sombra es lo que costaría la misma respuesta con un modelo comercial de referencia. Así se puede comparar el costo del agente único con el del multiagente.

---

## 🧰 Stack

| Componente | Tecnología |
|---|---|
| LLM principal | `qwen2.5:7b` vía **Ollama** (comparación: `llama3.2:3b`) |
| Orquestación | **LangChain** + **LangGraph** |
| Observabilidad y evaluación | **LangSmith** |
| Juez externo de la evaluación | Modelo independiente (OpenRouter / `llama3.1:8b`), distinto del que responde |
| Datos estructurados | **DuckDB** |
| RAG | **Chroma** + embeddings locales |
| Protocolo de herramientas | **MCP** (servidor local por stdio) |
| Interfaz | **Streamlit** + Plotly |

**Equipo de prueba:** Ryzen 5 3600X · 16 GB RAM · RTX 3060 12 GB.

---

## 📁 Estructura del proyecto

```
asistente-campanas/
├── app.py                    # Interfaz Streamlit
├── iniciar_asistente.bat     # Lanzador con doble clic (Windows)
├── generar_datos_q2.py       # Genera la base sintética de Q2 (para el MCP)
├── requirements.txt
├── .env.example              # Variables de entorno necesarias (sin valores)
├── src/
│   ├── config.py             # Modelos, rutas, precios de referencia
│   ├── tools.py              # Herramientas deterministas
│   ├── guardarrailes.py      # Validación y registro de herramientas
│   ├── costos.py             # Contador de costos por agente
│   ├── agent.py              # Agente único
│   └── grafo.py              # Grafo multiagente (LangGraph)
├── notebooks/                # Notebook principal: datos, RAG, agentes y evaluación
├── data/                     # Datos sintéticos (CSV / DuckDB)
└── docs/                     # Informe y presentación
```

---

## 🚀 Instalación y uso

### 1. Requisitos
- Python 3.10 o superior
- [Ollama](https://ollama.com) instalado y corriendo
- Una cuenta de [LangSmith](https://smith.langchain.com) para las trazas

### 2. Clonar e instalar
```bash
git clone https://github.com/TU_USUARIO/asistente-campanas-ia-generativa.git
cd asistente-campanas-ia-generativa
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Descargar los modelos
```bash
ollama pull qwen2.5:7b
ollama pull llama3.2:3b      # opcional, para comparar
ollama pull llama3.1:8b      # juez de la evaluación
```

### 4. Variables de entorno
Copia `.env.example` a `.env` y completa tus claves:
```
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=asistente-campanas-ben-frank
OPENROUTER_API_KEY=
```

### 5. Preparar los datos (una sola vez)
1. Ejecuta el notebook de `notebooks/`. Crea la base DuckDB de Q3 y el índice Chroma del RAG.
2. Genera la base de Q2 para el servidor MCP:
   ```bash
   python generar_datos_q2.py
   ```

### 6. Lanzar la interfaz
```bash
streamlit run app.py
```
En Windows también puedes darle doble clic a `iniciar_asistente.bat`. El servidor MCP de Q2 lo arranca la app sola cuando hace falta.

---

## 🔍 Cómo verificar el multiagente

- En la barra lateral, deja **Arquitectura = Multiagente**.
- Debajo de cada respuesta aparece la **ruta** de agentes, por ejemplo: `Ruta mixta · 📊 Datos → 📚 Conocimiento → ✍️ Redactor · crítico ✅ aprobado`.
- El desplegable **💰 Costo por agente** muestra una fila por agente.
- En LangSmith, la traza `multiagente` muestra los nodos `supervisor`, los especialistas y el `critico`.

---

## 📊 Evaluación

El notebook compara el **agente único** con el **multiagente** sobre un conjunto de preguntas de prueba. Las respuestas las califica un **juez independiente**, un modelo distinto del que responde, para evitar la autoevaluación. Además se registra el costo de cada arquitectura. Los resultados completos están en el informe (`docs/`).

**Hallazgo clave:** `llama3.2:3b` tendía a saltarse las herramientas y a inventar plataformas que no estaban en los datos. `qwen2.5:7b` llama las herramientas de forma consistente, y por eso quedó como modelo principal.

---

## 📄 Documentación

- 📘 Informe final: `docs/`
- 🎤 Presentación: `docs/`

## 👤 Autor

**Fer**, Maestría en Ciencia de Datos, Universidad ICESI
