"""
Agente único (v6): 5 herramientas nativas (Q3) + 4 herramientas MCP (Q2) + guardarraíles + memoria.
Las herramientas viven en src/herramientas.py, el middleware en src/guardarrailes.py
y las tools MCP llegan desde src/mcp_cliente.py (servidor local mcp_server/server.py).
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from src.config import LLM_PRINCIPAL
# Se re-exportan para que el notebook y app.py sigan funcionando con "from src.agent import ..."
from src.herramientas import (HERRAMIENTAS, KPIS_VALIDOS, MENOR_ES_MEJOR, buscar_conocimiento,  # noqa: F401
                              consultar_metricas, detectar_anomalias, generar_grafico, ranking)
from src.guardarrailes import REGISTRO, limpiar_argumentos  # noqa: F401
from src.mcp_cliente import herramientas_mcp

SYSTEM_PROMPT = """Eres un analista senior de medios digitales para la marca de gafas Ben & Frank.
Datos disponibles (US, CA y PR; mismas 5 plataformas y mismos KPIs en ambos trimestres):
- Q3 (2026-07-01 a 2026-09-30): consultar_metricas, ranking, detectar_anomalias, generar_grafico.
- Q2 (2026-04-01 a 2026-06-30): consultar_metricas_q2, ranking_q2, detectar_anomalias_q2, resumen_q2.
Reglas de trimestre:
- Si la pregunta es de abril, mayo, junio o "Q2", usa SOLO las herramientas que terminan en _q2.
- Si es de julio, agosto, septiembre, "Q3" o no menciona fecha, usa las herramientas de Q3.
- Para comparar Q2 vs Q3, llama la herramienta de cada trimestre con los mismos filtros y compara las cifras.
- Meses Q2: abril = 2026-04-01 a 2026-04-30, mayo = 2026-05-01 a 2026-05-31, junio = 2026-06-01 a 2026-06-30.
Reglas:
- SIEMPRE llama al menos una herramienta antes de responder. Tienes acceso completo a los datos de la campaña.
- Todo número debe salir de una herramienta; nunca respondas con conocimiento general ni inventes cifras.
- Solo existen estas plataformas: facebook, tiktok, snapchat, pinterest, reddit.
- Para comparar plataformas, países o creativos usa agrupar_por (ej. agrupar_por=["site"]).
- Para preguntas de "cuál tuvo el menor/mayor X" usa ranking con dimension y kpi exactos
  (ej. "¿qué plataforma tuvo el menor costo por cita?" -> ranking(dimension="site", kpi="costo_por_cita")).
- Para un valor único ("¿cuál fue el CPM en US?", "¿cuánto se invirtió?") usa consultar_metricas sin agrupar_por.
- No agregues filtros que el usuario no pidió (por ejemplo funnel), y no generes gráficos si no los pidió.
- Lee el valor del KPI por su nombre exacto: costo_por_cita no es lo mismo que tasa_conversion.
- Glosario: citas (citas de examen visual) = subscriptions, NO lpv. Inversión o gasto = media_cost.
  Impresiones = impressions. Tasa de completado = vtr.
- Si el usuario no menciona país, plataforma o funnel, NO envíes esos parámetros: la respuesta es del total.
- Meses de 2026: julio = 2026-07-01 a 2026-07-31, agosto = 2026-08-01 a 2026-08-31, septiembre = 2026-09-01 a 2026-09-30.
- Preguntas conceptuales ("qué es", "qué KPIs debo usar", "cómo se calcula", benchmarks, objetivos del brief)
  se responden con buscar_conocimiento, no con consultar_metricas.
- Llama ranking UNA sola vez: su primera fila ya es la respuesta. No repitas la misma llamada.
- "¿Cumplió el objetivo / el benchmark?": NO uses ranking. Saca la cifra con consultar_metricas
  (con el filtro que pida, ej. funnel="lower") y el objetivo con buscar_conocimiento; luego compara.
- "Citas" es la cantidad (subscriptions); "costo por cita" es el costo. VTR es tasa de completado (vtr), no view_rate.
- Responde siempre en español.
- Si piden un gráfico, tendencia o comparación visual usa generar_grafico. El gráfico ya se muestra
  en pantalla: no escribas código ni tablas para dibujarlo, solo comenta en 2-3 frases lo que se ve.
- Compara funnels con sus KPIs propios (upper: CPM/VTR, middle: CTR/CPC, lower: costo por cita).
- Responde en español, conciso, con las cifras clave y una recomendación cuando aplique."""


def crear_agente(modelo: str = LLM_PRINCIPAL):
    llm = init_chat_model(modelo, temperature=0)
    tools = HERRAMIENTAS + herramientas_mcp()  # nativas (Q3) + MCP (Q2)
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver(),
                        middleware=[limpiar_argumentos])


def preguntar(agente, texto: str, thread_id: str = "demo") -> str:
    out = agente.invoke({"messages": [{"role": "user", "content": texto}]},
                        {"configurable": {"thread_id": thread_id}})
    return out["messages"][-1].content


def figuras_del_turno(mensajes) -> list[str]:
    """JSON de las figuras Plotly generadas desde la última pregunta del usuario."""
    inicio = max(i for i, m in enumerate(mensajes) if m.type == "human")
    return [m.artifact for m in mensajes[inicio + 1:]
            if m.type == "tool" and getattr(m, "artifact", None) and m.name == "generar_grafico"]
