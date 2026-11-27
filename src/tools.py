"""
Capa determinista de datos: todo número que vea el usuario sale de aquí, nunca del LLM.
Estas funciones devuelven DataFrames; en agent.py se envuelven con @tool para el agente.
"""
from typing import Literal, Optional
import pandas as pd

from src.config import TABLA
from src.ingest import conectar

Dimension = Literal["fecha", "semana", "mes", "pais", "site", "funnel", "creative"]
KPI = Literal["cpm", "cpc", "ctr", "tasa_lpv", "costo_por_lpv", "tasa_conversion", "costo_por_cita",
              "view_rate", "vtr", "cpv", "cpcv"]
BASE = ["media_cost", "impressions", "clicks", "lpv", "subscriptions", "video_starts", "video_completes"]

_DIM_SQL = {
    "fecha": "fecha", "semana": "date_trunc('week', fecha)::DATE AS semana",
    "mes": "strftime(fecha, '%Y-%m') AS mes",
    "pais": "pais", "site": "site", "funnel": "funnel", "creative": "creative",
}


# --------------------------------------------------------------------------- KPIs
def calcular_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega KPIs a un DataFrame con las columnas base. Divisiones por cero -> NaN."""
    d = df.copy()
    div = lambda a, b: (d[a] / d[b].where(d[b] != 0))
    d["cpm"] = div("media_cost", "impressions") * 1000          # costo por mil impresiones
    d["cpc"] = div("media_cost", "clicks")                       # costo por click
    d["ctr"] = div("clicks", "impressions") * 100                # % clicks / impresiones
    d["tasa_lpv"] = div("lpv", "clicks") * 100                   # % clicks que cargan la landing
    d["costo_por_lpv"] = div("media_cost", "lpv")
    d["tasa_conversion"] = div("subscriptions", "lpv") * 100     # % LPV que agendan examen visual
    d["costo_por_cita"] = div("media_cost", "subscriptions")     # CPA
    d["view_rate"] = div("video_starts", "impressions") * 100
    d["vtr"] = div("video_completes", "video_starts") * 100      # tasa de completado
    d["cpv"] = div("media_cost", "video_starts")                 # costo por reproducción
    d["cpcv"] = div("media_cost", "video_completes")             # costo por video completo
    return d.round(4)


def _where(fecha_inicio, fecha_fin, pais, site, funnel, creative_contiene):
    cond, params = [], []
    for col, val in (("pais", pais), ("site", site), ("funnel", funnel)):
        if val:
            cond.append(f"{col} = ?"); params.append(val.lower() if col != "pais" else val.upper())
    if fecha_inicio:
        cond.append("fecha >= ?"); params.append(fecha_inicio)
    if fecha_fin:
        cond.append("fecha <= ?"); params.append(fecha_fin)
    if creative_contiene:
        cond.append("creative ILIKE ?"); params.append(f"%{creative_contiene}%")
    return ("WHERE " + " AND ".join(cond)) if cond else "", params


# --------------------------------------------------------------------------- herramienta 1
def consultar_metricas(agrupar_por: Optional[list[Dimension]] = None, fecha_inicio: Optional[str] = None,
                       fecha_fin: Optional[str] = None, pais: Optional[str] = None, site: Optional[str] = None,
                       funnel: Optional[str] = None, creative_contiene: Optional[str] = None) -> pd.DataFrame:
    """Suma métricas base con filtros y agrupación opcional, y añade KPIs."""
    agrupar_por = agrupar_por or []
    dims = [_DIM_SQL[d] for d in agrupar_por]
    alias = [d for d in agrupar_por]
    where, params = _where(fecha_inicio, fecha_fin, pais, site, funnel, creative_contiene)
    sums = ", ".join(f"SUM({c}) AS {c}" for c in BASE)
    select = ", ".join(dims + [sums])
    group = f"GROUP BY {', '.join(alias)} ORDER BY {', '.join(alias)}" if alias else ""
    with conectar() as con:
        df = con.execute(f"SELECT {select} FROM {TABLA} {where} {group}", params).df()
    return calcular_kpis(df)


# --------------------------------------------------------------------------- herramienta 2
def ranking(dimension: Dimension = "creative", kpi: str = "costo_por_cita", n: int = 5,
            mejores: bool = True, min_impresiones: int = 10_000, **filtros) -> pd.DataFrame:
    """Top/bottom N de una dimensión según un KPI o métrica base."""
    df = consultar_metricas([dimension], **filtros)
    df = df[df.impressions >= min_impresiones].dropna(subset=[kpi])
    menor_es_mejor = kpi in {"cpm", "cpc", "costo_por_lpv", "costo_por_cita", "cpv", "cpcv"}
    asc = menor_es_mejor if mejores else not menor_es_mejor
    return df.sort_values(kpi, ascending=asc).head(n)[[dimension, kpi, "media_cost", "impressions"]]


# --------------------------------------------------------------------------- herramienta 3
def detectar_anomalias(fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                       pais: Optional[str] = None, site: Optional[str] = None) -> pd.DataFrame:
    """Reglas simples sobre datos diarios: tracking caído, gasto sin entrega, CTR sospechoso, sobrecosto de CPM."""
    where, params = _where(fecha_inicio, fecha_fin, pais, site, None, None)
    with conectar() as con:
        d = con.execute(f"SELECT * FROM {TABLA} {where}", params).df()
    hallazgos = []

    # Gasto sin entrega (fila a fila)
    for _, r in d[(d.media_cost > 0) & (d.impressions == 0)].iterrows():
        hallazgos.append(dict(tipo="gasto_sin_entrega", fecha=r.fecha, pais=r.pais, site=r.site,
                              creative=r.creative, detalle=f"${r.media_cost:,.2f} gastados con 0 impresiones"))

    # Agregado por día-site-país
    g = d.groupby(["fecha", "pais", "site"], as_index=False)[BASE].sum()
    for _, r in g[(g.impressions > 5_000) & (g.clicks == 0)].iterrows():
        hallazgos.append(dict(tipo="tracking_caido", fecha=r.fecha, pais=r.pais, site=r.site, creative=None,
                              detalle=f"{r.impressions:,} impresiones y 0 clicks/LPV"))
    g["cpm"] = g.media_cost / g.impressions.where(g.impressions > 0) * 1000
    med = g.groupby(["pais", "site"]).cpm.transform("median")
    for _, r in g[g.cpm > 2.5 * med].iterrows():
        hallazgos.append(dict(tipo="sobrecosto_cpm", fecha=r.fecha, pais=r.pais, site=r.site, creative=None,
                              detalle=f"CPM ${r.cpm:,.2f} vs mediana ${med[_]:,.2f}"))

    # CTR sospechoso: muchos clicks que no llegan a la landing
    s = d[(d.clicks >= 100)]
    for _, r in s[s.lpv / s.clicks < 0.30].iterrows():
        hallazgos.append(dict(tipo="pico_ctr_sospechoso", fecha=r.fecha, pais=r.pais, site=r.site,
                              creative=r.creative,
                              detalle=f"CTR {r.clicks / r.impressions:.1%}, solo {r.lpv / r.clicks:.0%} de clicks llegan a la landing"))
    if not hallazgos:
        return pd.DataFrame()
    out = pd.DataFrame(hallazgos).sort_values(["fecha", "tipo"])
    out["fecha"] = pd.to_datetime(out.fecha).dt.date
    return out.fillna("-")


def a_texto(df: pd.DataFrame, max_filas: int = 30) -> str:
    """Formato compacto para devolver al LLM."""
    if df.empty:
        return "Sin resultados para esos filtros."
    extra = f"\n(... {len(df) - max_filas} filas más)" if len(df) > max_filas else ""
    return df.head(max_filas).to_markdown(index=False, floatfmt=",.2f") + extra
