"""
Las 5 herramientas del asistente envueltas con @tool.
Los números salen de src/tools.py (DuckDB) y src/graficos.py (Plotly); aquí solo se definen
las firmas que ve el LLM y el formato de salida etiquetado (kpi: valor).

Las usan tanto el agente único (src/agent.py) como los especialistas del multiagente (src/agentes.py).
"""
from typing import Literal, Optional

from langchain.tools import tool

from src import tools as T
from src import graficos as G
from src.rag import retriever

_ret = None


def _texto(df) -> str:
    """Una línea por fila con cada valor etiquetado (kpi: valor).
    Los modelos pequeños confunden columnas al leer tablas anchas; así no hay ambigüedad."""
    if df is None or len(df) == 0:
        return "Sin resultados para esos filtros."
    lineas = []
    for _, fila in df.iterrows():
        partes = []
        for col, val in fila.items():
            if val is None or (isinstance(val, float) and val != val):   # NaN
                continue
            if isinstance(val, float):
                val = f"{val:,.2f}"
            elif isinstance(val, int) and not isinstance(val, bool):
                val = f"{val:,}"
            partes.append(f"{col}: {val}")
        lineas.append("- " + " | ".join(partes))
    return "\n".join(lineas)


Dim = Literal["fecha", "semana", "mes", "pais", "site", "funnel", "creative"]

KPIS_VALIDOS = ["media_cost", "impressions", "clicks", "lpv", "subscriptions", "video_starts", "video_completes",
                "cpm", "cpc", "ctr", "tasa_lpv", "costo_por_lpv", "tasa_conversion", "costo_por_cita",
                "view_rate", "vtr", "cpv", "cpcv"]

MENOR_ES_MEJOR = {"cpm", "cpc", "costo_por_lpv", "costo_por_cita", "cpv", "cpcv"}


@tool
def consultar_metricas(agrupar_por: Optional[list[Dim]] = None, fecha_inicio: Optional[str] = None,
                       fecha_fin: Optional[str] = None, pais: Optional[str] = None, site: Optional[str] = None,
                       funnel: Optional[str] = None, creative_contiene: Optional[str] = None) -> str:
    """Consulta métricas de campañas sociales de Ben & Frank (jul–sep 2026) con KPIs ya calculados
    (cpm, cpc, ctr, tasa_lpv, costo_por_lpv, tasa_conversion, costo_por_cita, view_rate, vtr, cpv, cpcv).
    Fechas en formato YYYY-MM-DD. pais: US, CA o PR. site: facebook, tiktok, snapchat, pinterest, reddit.
    funnel: upper, middle o lower. Usa SIEMPRE esta herramienta para cualquier número."""
    return _texto(T.consultar_metricas(agrupar_por, fecha_inicio, fecha_fin, pais, site, funnel, creative_contiene))


@tool
def ranking(dimension: Dim, kpi: str, orden: Literal["mayor", "menor"], n: int = 5,
            pais: Optional[str] = None, site: Optional[str] = None, funnel: Optional[str] = None,
            fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None) -> str:
    """Ordena elementos por un KPI y devuelve los N primeros: la PRIMERA fila es la respuesta.
    Úsala solo para "cuál tuvo el mayor/menor X". Para un valor único usa consultar_metricas.
    dimension: site = plataformas, pais = países, mes = meses, semana, creative = creativos, funnel.
    orden: "menor" = valores más bajos primero; "mayor" = valores más altos primero.
      Ej.: menor costo por cita -> orden="menor"; más citas -> orden="mayor"; peor VTR -> orden="menor".
    kpi: media_cost, impressions, clicks, subscriptions (citas), cpm, cpc, ctr, tasa_lpv,
      costo_por_cita, view_rate, vtr (tasa de completado), cpv, cpcv."""
    if kpi not in KPIS_VALIDOS:
        return f"KPI '{kpi}' no existe. Usa uno de: {', '.join(KPIS_VALIDOS)}."
    # tools.ranking trabaja con "mejores"; se traduce desde el orden numérico
    mejores = (orden == "menor") if kpi in MENOR_ES_MEJOR else (orden == "mayor")
    try:
        df = T.ranking(dimension, kpi, n, mejores, pais=pais, site=site, funnel=funnel,
                       fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    except Exception as e:
        return f"No se pudo calcular el ranking ({e}). Revisa dimension y filtros."
    return f"Ranking por {kpi}, de {orden} a {'mayor' if orden == 'menor' else 'menor'} (la primera fila es el resultado):\n" + _texto(df)


@tool
def detectar_anomalias(fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                       pais: Optional[str] = None, site: Optional[str] = None) -> str:
    """Detecta anomalías de entrega: gasto sin impresiones, tracking caído (impresiones sin clicks),
    CTR sospechoso (clicks que no llegan a la landing) y sobrecosto de CPM."""
    return _texto(T.detectar_anomalias(fecha_inicio, fecha_fin, pais, site))


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
    return resumen + _texto(df), fig.to_json()


# Todas las herramientas (agente único) y por nombre (para armar especialistas)
HERRAMIENTAS = [consultar_metricas, ranking, detectar_anomalias, buscar_conocimiento, generar_grafico]
POR_NOMBRE = {h.name: h for h in HERRAMIENTAS}
