"""Configuración central del proyecto."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")   # LANGSMITH_* y OPENROUTER_API_KEY, desde la raíz del proyecto

DATA_RAW = ROOT / "data" / "raw" / "ben_frank_social_q3.csv"
CATALOGO = ROOT / "data" / "raw" / "catalogo_creativos.csv"
ANOMALIAS = ROOT / "data" / "raw" / "anomalias_plantadas.json"
DUCKDB_PATH = ROOT / "data" / "processed" / "campanas.duckdb"
DOCS_RAG = ROOT / "docs_rag"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma"
EVAL_PATH = ROOT / "eval" / "preguntas_eval.json"

# Modelos locales (Ollama)
LLM_PRINCIPAL = "ollama:qwen2.5:7b"     # ya instalado en el PC; se usa en TODOS los agentes
LLM_COMPARACION = "ollama:llama3.2:3b"    # mejor uso de herramientas; experimento A/B en LangSmith
EMBEDDINGS = "bge-m3"                    # alternativa: nomic-embed-text

LANGSMITH_PROJECT = "asistente-campanas-ben-frank"
TABLA = "social_diario"

# --------------------------------------------------------------------------- juez externo
# Juez de otra familia que el modelo que responde: reduce el sesgo de autoevaluación.
# Orden: OPENROUTER_API_KEY -> Haiku; OPENAI_API_KEY -> GPT-6 Luna; sin llaves -> juez LOCAL de otra familia.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_JUEZ = "anthropic/claude-haiku-5.5"   # vía OpenRouter
LLM_JUEZ_OPENAI = "gpt-6-luna"             # vía API de OpenAI (requiere créditos de API)
LLM_JUEZ_LOCAL = "ollama:llama3.1:8b"      # gratis; familia Llama (Meta) != Qwen. Instalar: ollama pull llama3.1:8b

# --------------------------------------------------------------------------- costos
# USD por millón de tokens (entrada, salida). Precios de referencia de la clase (oct. 2026);
# verificar en openrouter.ai/models antes de reportar.
PRECIOS_USD_POR_MILLON = {
    "deepseek-v4.1-flash": (0.028, 0.12),   # recomendado para trabajadores
    "gpt-6-luna":          (0.10, 0.50),
    "claude-haiku-5.5":    (0.10, 0.50),    # el juez
    "gemini-3.7-flash":    (0.75, 3.75),    # orquestador / techo de calidad
}
# Modelos locales: no pagan tokens. Se reportan con "precio sombra" usando este modelo de referencia.
MODELO_REFERENCIA_SOMBRA = "deepseek-v4.1-flash"

# Energía del PC local (costo real de correr Ollama)
POTENCIA_GPU_W = 170          # RTX 3060 a plena carga (aprox.)
TARIFA_KWH_COP = 900.0        # EJEMPLO: reemplaza con el valor por kWh de tu factura de energía

# --------------------------------------------------------------------------- multiagente
TOPE_USD_POR_TAREA = 0.01     # tope por pregunta (costo sombra + real); al superarlo se entrega lo que haya
MAX_RONDAS_CRITICO = 3        # máximo de revisiones del crítico por pregunta (diapositiva 6)
ROUTER_MODO = "llm"           # "llm": el supervisor clasifica con el LLM | "reglas": palabras clave, sin LLM
CRITICO_LLM = "mixta"         # cuándo usa LLM el crítico: "siempre" | "mixta" | "nunca"
                              # (la verificación determinista de cifras corre siempre, cuesta 0 tokens)
