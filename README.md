# Asistente de campañas sociales — Ben & Frank (Tendencias de IA Generativa)

LLM local (Ollama) + RAG (Chroma) + agente LangChain con 4 herramientas + trazas y evaluación en LangSmith + interfaz Streamlit.
Los datos son **sintéticos** y simulan un pull diario del data lake de social (jul–sep 2026, US/CA/PR).

## Estructura
```
data/raw/        ben_frank_social_q3.csv (12.167 filas), catalogo_creativos.csv, anomalias_plantadas.json
data/processed/  campanas.duckdb, chroma/
docs_rag/        corpus del RAG: glosario, plataformas, funnel, brief Q3, convenciones y benchmarks
src/
  config.py          rutas y modelos
  generar_datos.py   generador sintético (semilla 42, reproducible)      ✅ probado
  ingest.py          CSV -> DuckDB                                       ✅ probado
  tools.py           capa determinista: métricas, KPIs, ranking, anomalías ✅ probado
  construir_eval.py  dataset de evaluación con respuestas desde DuckDB   ✅ probado
  rag.py             índice Chroma con embeddings locales                ⏳ fase 3
  agent.py           agente + herramientas @tool + memoria               ⏳ fase 3
  evaluate.py        experimentos LangSmith (llama3.2:3b vs qwen2.5:7b)   ⏳ fase 3
app.py             interfaz Streamlit                                    ⏳ fase 3
eval/preguntas_eval.json  17 preguntas (numéricas, ranking, anomalías, conceptuales)
```

## Herramientas del agente
| Herramienta | Qué hace |
|---|---|
| `consultar_metricas` | Filtra/agrupa (fecha, semana, mes, país, site, funnel, creativo) y devuelve CPM, CPC, CTR, tasa LPV, costo por cita, VTR, CPV… |
| `ranking` | Top/bottom N por cualquier KPI (sabe que en costos menor es mejor) |
| `detectar_anomalias` | Gasto sin entrega, tracking caído, CTR sospechoso, sobrecosto de CPM |
| `buscar_conocimiento` | RAG sobre `docs_rag/` |

## Puesta en marcha (Windows/Linux, RTX 3060 12 GB)
```bash
# 1. Ollama: https://ollama.com  luego
ollama pull qwen2.5:7b && ollama pull bge-m3     # llama3.2:3b ya está instalado
# 2. Entorno
python -m venv .venv && .venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
# 3. Datos e índices
python -m src.generar_datos
python -m src.ingest
python -m src.construir_eval
python -m src.rag
# 4. LangSmith (variables de entorno)
set LANGSMITH_API_KEY=...   set LANGSMITH_TRACING=true
# 5. Usar
streamlit run app.py
python -m src.evaluate
```

## Anomalías plantadas (verdad conocida)
1. 2026-08-14 · Facebook US · tracking caído (impresiones sin clicks/LPV)
2. 2026-09-03 · TikTok CA · gasto sin entrega
3. 2026-07-22 · Pinterest US · pico de CTR sin LPV (posible bot)
4. 2026-09-15 a 17 · Snapchat PR · CPM ~3x
