"""
Agente LangChain con 5 herramientas + memoria.
consultar_metricas, ranking, detectar_anomalias, buscar_conocimiento (RAG) y generar_grafico.
"""
from typing import Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from src import tools as T
from src import graficos as G
from src.config import LLM_PRINCIPAL
from src.rag import retriever

_ret = None

Dim = Literal["fecha", "semana", "mes", "pais", "site", "funnel", "creative"]


@tool
def consultar_metricas(agrupar_por: Optional[list[Dim]] = None, fecha_inicio: Optional[str] = None,
                       fecha_fin: Optional[str] = None, pais: Optional[str] = None, site: Optional[str] = None,
                       funnel: Optional[str] = None, creative_contiene: Optional[str] = None) -> str:
    """Consulta métricas de campañas sociales de Ben & Frank (jul–sep 2026) con KPIs ya calculados
    (cpm, cpc, ctr, tasa_lpv, costo_por_lpv, tasa_conversion, costo_por_cita, view_rate, vtr, cpv, cpcv).
    Fechas en formato YYYY-MM-DD. pais: US, CA o PR. site: facebook, tiktok, snapchat, pinterest, reddit.
    funnel: upper, middle o lower. Usa SIEMPRE esta herramienta para cualquier número."""
    return T.a_texto(T.consultar_metricas(agrupar_por, fecha_inicio, fecha_fin, pais, site, funnel, creative_contiene))


@tool
def ranking(dimension: Dim = "creative", kpi: str = "costo_por_cita", n: int = 5, mejores: bool = True,
            pais: Optional[str] = None, site: Optional[str] = None, funnel: Optional[str] = None,
            fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None) -> str:
    """Devuelve los N mejores (mejores=True) o peores (mejores=False) elementos de una dimensión según un KPI.
    Ya sabe que en costos menor es mejor. Ignora elementos con menos de 10.000 impresiones."""
    return T.a_texto(T.ranking(dimension, kpi, n, mejores, pais=pais, site=site, funnel=funnel,
                               fecha_inicio=fecha_inicio, fecha_fin=fecha_fin))


@tool
def detectar_anomalias(fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                       pais: Optional[str] = None, site: Optional[str] = None) -> str:
    """Detecta anomalías de entrega: gasto sin impresiones, tracking caído (impresiones sin clicks),
    CTR sospechoso (clicks que no llegan a la landing) y sobrecosto de CPM."""
    return T.a_texto(T.detectar_anomalias(fecha_inicio, fecha_fin, pais, site))


@tool
def buscar_conocimiento(consulta: str) -> str:
    """Busca en la base de conocimiento: definiciones de métricas, plataformas, funnel, brief de la
    campaña Q3, convenciones de nombres y benchmarks internos. Úsala para conceptos y contexto, no para números."""
    global _ret
    _ret = _ret or retriever()
    return "\n\n".join(f"[{d.metadata.get('fuente')}] {d.page_content}" for d in _ret.invoke(consulta))


@tool(response_format="content_and_artifact")
def generar_grafico(kpi: str = "media_cost", por: Dim = "semana", desglose: Optional[Dim] = None,
                    fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                    pais: Optional[str] = None, site: Optional[str] = None,
                    funnel: Optional[str] = None) -> tuple[str, str]:
    """Genera un gráfico que se muestra en la interfaz. Úsala cuando el usuario pida un gráfico,
    una tendencia, una evolución en el tiempo o una comparación visual.
    kpi: media_cost, impressions, clicks, subscriptions, cpm, cpc, ctr, tasa_lpv, costo_por_cita, vtr, view_rate, cpv...
    por: eje X. Temporal (fecha, semana, mes) => línea; categórico (site, pais, funnel, creative) => barras.
    desglose: segunda dimensión opcional como color, ej. por='semana', desglose='site' = una línea por plataforma.
    Filtros iguales a consultar_metricas."""
    fig, df = G.grafico(kpi, por, desglose, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin,
                        pais=pais, site=site, funnel=funnel)
    resumen = f"Gráfico '{fig.layout.title.text}' generado y visible para el usuario. Datos graficados:\n"
    return resumen + T.a_texto(df), fig.to_json()


VACIOS = ({}, [], "", "null", "None", "none")


@wrap_tool_call
def limpiar_argumentos(request, handler):
    """Los modelos pequeños mandan {} o "" en los parámetros opcionales que no usan.
    Se eliminan antes de validar para que la herramienta use su valor por defecto (None)."""
    args = {k: v for k, v in request.tool_call["args"].items() if v is not None and v not in VACIOS}
    if isinstance(args.get("agrupar_por"), str):      # "site" -> ["site"]
        args["agrupar_por"] = [args["agrupar_por"]]
    return handler(request.override(tool_call={**request.tool_call, "args": args}))


HERRAMIENTAS = [consultar_metricas, ranking, detectar_anomalias, buscar_conocimiento, generar_grafico]

SYSTEM_PROMPT = """Eres un analista senior de medios digitales para la marca de gafas Ben & Frank.
Datos disponibles: campañas sociales diarias del 2026-07-01 al 2026-09-30 en US, CA y PR.
Reglas:
- SIEMPRE llama al menos una herramienta antes de responder. Tienes acceso completo a los datos de la campaña.
- Todo número debe salir de una herramienta; nunca respondas con conocimiento general ni inventes cifras.
- Solo existen estas plataformas: facebook, tiktok, snapchat, pinterest, reddit.
- Para comparar plataformas, países o creativos usa agrupar_por (ej. agrupar_por=["site"]).
- Meses de 2026: julio = 2026-07-01 a 2026-07-31, agosto = 2026-08-01 a 2026-08-31, septiembre = 2026-09-01 a 2026-09-30.
- Para definiciones, benchmarks o contexto de la campaña usa buscar_conocimiento.
- Si piden un gráfico, tendencia o comparación visual usa generar_grafico. El gráfico ya se muestra
  en pantalla: no escribas código ni tablas para dibujarlo, solo comenta en 2-3 frases lo que se ve.
- Compara funnels con sus KPIs propios (upper: CPM/VTR, middle: CTR/CPC, lower: costo por cita).
- Responde en español, conciso, con las cifras clave y una recomendación cuando aplique."""


def crear_agente(modelo: str = LLM_PRINCIPAL):
    llm = init_chat_model(modelo, temperature=0)
    return create_agent(llm, tools=HERRAMIENTAS, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver(),
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
