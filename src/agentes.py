"""
Los tres especialistas del multiagente. Cada uno es un agente ReAct (create_agent) con:
  - solo las herramientas de su rol (menos firmas en el prompt = menos tokens de entrada),
  - un prompt corto con las reglas de v5 que le aplican,
  - el MISMO middleware de guardarraíles que el agente único (src/guardarrailes.py).

No tienen checkpointer propio: la memoria de la conversación vive en el grafo (src/grafo.py).
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from src.config import LLM_PRINCIPAL
from src.guardarrailes import limpiar_argumentos
from src.herramientas import POR_NOMBRE

CONTEXTO = """Datos: campañas sociales diarias de Ben & Frank (gafas) del 2026-07-01 al 2026-09-30, países US, CA y PR,
plataformas facebook, tiktok, snapchat, pinterest y reddit; funnels upper, middle y lower.
Meses: julio = 2026-07-01 a 2026-07-31, agosto = 2026-08-01 a 2026-08-31, septiembre = 2026-09-01 a 2026-09-30.
Responde siempre en español, conciso."""

PROMPT_DATOS = f"""Eres el especialista en DATOS de un equipo de análisis de medios digitales.
{CONTEXTO}
Reglas:
- SIEMPRE llama una herramienta antes de responder. Todo número sale de una herramienta; nunca inventes cifras.
- Valor único ("¿cuál fue el CPM en US?", "¿cuánto se invirtió?"): consultar_metricas sin agrupar_por.
- "¿Cuál tuvo el mayor/menor X?": ranking con dimension y kpi exactos, UNA sola vez; su primera fila es la respuesta.
- Para comparar plataformas, países o creativos: consultar_metricas con agrupar_por (ej. ["site"]).
- Si el usuario no menciona país, plataforma o funnel, NO envíes esos parámetros: la respuesta es del total.
  Nunca envíes listas de valores: pais, site y funnel aceptan un solo valor.
- Glosario: citas = subscriptions (cantidad); costo por cita = costo_por_cita; inversión o gasto = media_cost;
  impresiones = impressions; VTR o tasa de completado = vtr (no view_rate).
- Lee el KPI por su nombre exacto en la salida de la herramienta.
- Si piden un gráfico o tendencia usa generar_grafico; ya se muestra en pantalla, comenta en 2-3 frases lo que se ve.
- Copia las cifras tal como salen de la herramienta. Da la cifra clave y una recomendación breve si aplica."""

PROMPT_ANOMALIAS = f"""Eres el especialista en ANOMALÍAS de entrega y tracking.
{CONTEXTO}
Reglas:
- SIEMPRE llama detectar_anomalias antes de responder; si mencionan un mes, pasa sus fechas.
- Si el usuario no menciona país o plataforma, NO envíes pais ni site (la búsqueda cubre todo).
  Nunca envíes listas de valores: pais y site aceptan un solo valor.
- Reporta cada hallazgo con fecha (YYYY-MM-DD), país, plataforma, tipo y detalle, copiando las cifras de la herramienta.
- Si la herramienta no encuentra nada, dilo claramente. No inventes problemas."""

PROMPT_CONOCIMIENTO = f"""Eres el especialista en CONOCIMIENTO: definiciones de métricas, plataformas, funnel,
brief de la campaña Q3, objetivos, benchmarks y convenciones de nombres.
{CONTEXTO}
Reglas:
- SIEMPRE llama buscar_conocimiento antes de responder.
- Responde solo con lo que diga la base de conocimiento e indica la fuente entre corchetes.
- No calcules resultados de campaña: eso lo hace otro especialista."""

ESPECIALISTAS = {
    "datos": (PROMPT_DATOS, ["consultar_metricas", "ranking", "generar_grafico"]),
    "anomalias": (PROMPT_ANOMALIAS, ["detectar_anomalias"]),
    "conocimiento": (PROMPT_CONOCIMIENTO, ["buscar_conocimiento"]),
}

# Instrucción extra por especialista cuando la pregunta es mixta (cifra + objetivo).
# Evita palabras de orden ("más", "menor", "mejor") para no interferir con el middleware.
TAREA_MIXTA = {
    "datos": "Tu parte: calcula solo la cifra que pide la pregunta con consultar_metricas, sin ranking. "
             "No busques el objetivo; de eso se encarga otro especialista.",
    "conocimiento": "Tu parte: busca solo el objetivo o benchmark que menciona la pregunta. "
                    "No calcules resultados; de eso se encarga otro especialista.",
}


def crear_especialistas(modelo: str = LLM_PRINCIPAL, llm=None) -> dict:
    """Devuelve {"datos": agente, "anomalias": agente, "conocimiento": agente}.
    `llm` permite inyectar un modelo ya creado (pruebas o el mismo objeto para todos)."""
    llm = llm or init_chat_model(modelo, temperature=0)
    agentes = {}
    for nombre, (prompt, herramientas) in ESPECIALISTAS.items():
        agentes[nombre] = create_agent(
            llm, tools=[POR_NOMBRE[h] for h in herramientas], system_prompt=prompt,
            middleware=[limpiar_argumentos], name=f"especialista_{nombre}",
        ).with_config(metadata={"agente": nombre}, tags=[f"agente:{nombre}"], run_name=f"especialista_{nombre}")
    return agentes
