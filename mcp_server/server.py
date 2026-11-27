"""
Servidor MCP "campanas-q2" — datos de la campaña social Ben & Frank Q2 2026 (abril–junio).

Expone 4 herramientas deterministas sobre data/campanas_q2.duckdb. Toda cifra sale de
DuckDB; el LLM solo decide qué consultar. Corre local por stdio: lo lanza el cliente
(tu app LangChain, VS Code, Claude Desktop o el MCP Inspector).

Probar sin LLM:   uv run mcp dev server.py
IMPORTANTE: no usar print() aquí (stdout es del protocolo). Para depurar: sys.stderr.
"""
from pathlib import Path
from typing import Literal, Optional

import duckdb
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

DUCKDB_PATH = Path(__file__).parent / "data" / "campanas_q2.duckdb"
TABLA = "campanas_q2"

mcp = FastMCP("campanas-q2")

Dimension = Literal["fecha", "semana", "mes", "pais", "site", "funnel", "creative"]
KPI = Literal["cpm", "cpc", "ctr", "tasa_lpv", "costo_por_lpv", "tasa_conversion",
              "costo_por_cita", "view_rate", "vtr", "cpv", "cpcv"]
BASE = ["media_cost", "impressions", "clicks", "lpv", "subscriptions", "video_starts", "video_completes"]
_DIM_SQL = {
    "fecha": "fecha", "semana": "date_trunc('week', fecha)::DATE AS semana",
    "mes": "strftime(fecha, '%Y-%m') AS mes",
    "pais": "pais", "site": "site", "funnel": "funnel", "creative": "creative",
}
# KPIs donde "mejor" es el valor más bajo (costos)
_KPI_COSTO = {"cpm", "cpc", "costo_por_lpv", "costo_por_cita", "cpv", "cpcv"}


# ------------------------------------------------------------------ utilidades internas
def _query(sql: str, params: list | None = None) -> pd.DataFrame:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def _kpis(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    div = lambda a, b: d[a] / d[b].where(d[b] != 0)
    d["cpm"] = div("media_cost", "impressions") * 1000
    d["cpc"] = div("media_cost", "clicks")
    d["ctr"] = div("clicks", "impressions") * 100
    d["tasa_lpv"] = div("lpv", "clicks") * 100
    d["costo_por_lpv"] = div("media_cost", "lpv")
    d["tasa_conversion"] = div("subscriptions", "lpv") * 100
    d["costo_por_cita"] = div("media_cost", "subscriptions")
    d["view_rate"] = div("video_starts", "impressions") * 100
    d["vtr"] = div("video_completes", "video_starts") * 100
    d["cpv"] = div("media_cost", "video_starts")
    d["cpcv"] = div("media_cost", "video_completes")
    num = d.select_dtypes("number").columns
    d[num] = d[num].round(2)
    return d


def _registros(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista de dicts serializable (NaN -> None, fechas -> texto)."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
        elif d[c].dtype == object:
            d[c] = d[c].astype(str)
    return d.replace({np.nan: None}).to_dict(orient="records")


def _where(fecha_inicio, fecha_fin, pais, site, funnel, creative_contiene):
    cond, params = [], []
    for col, val in (("pais", pais), ("site", site), ("funnel", funnel)):
        if val:
            cond.append(f"{col} = ?")
            params.append(val.upper() if col == "pais" else val.lower())
    if fecha_inicio:
        cond.append("fecha >= ?"); params.append(fecha_inicio)
    if fecha_fin:
        cond.append("fecha <= ?"); params.append(fecha_fin)
    if creative_contiene:
        cond.append("creative ILIKE ?"); params.append(f"%{creative_contiene}%")
    return ("WHERE " + " AND ".join(cond)) if cond else "", params


# ------------------------------------------------------------------ herramientas MCP
@mcp.tool()
def resumen_q2() -> dict:
    """Resumen general de la campaña Q2 (abril-junio 2026): rango de fechas, países,
    plataformas, número de creativos y totales con KPIs. Úsala primero para saber qué datos hay."""
    info = _query(f"""
        SELECT MIN(fecha) AS fecha_inicio, MAX(fecha) AS fecha_fin,
               COUNT(DISTINCT creative) AS creativos,
               LIST(DISTINCT pais ORDER BY pais) AS paises,
               LIST(DISTINCT site ORDER BY site) AS plataformas
        FROM {TABLA}""").iloc[0]
    tot = _kpis(_query(f"SELECT {', '.join(f'SUM({c}) AS {c}' for c in BASE)} FROM {TABLA}"))
    return {
        "campana": "Ben & Frank social Q2 2026",
        "fecha_inicio": str(pd.Timestamp(info.fecha_inicio).date()),
        "fecha_fin": str(pd.Timestamp(info.fecha_fin).date()),
        "paises": list(info.paises), "plataformas": list(info.plataformas),
        "creativos": int(info.creativos), "totales": _registros(tot)[0],
    }


@mcp.tool()
def consultar_metricas_q2(agrupar_por: Optional[list[Dimension]] = None,
                          fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                          pais: Optional[str] = None, site: Optional[str] = None,
                          funnel: Optional[str] = None,
                          creative_contiene: Optional[str] = None) -> list[dict]:
    """Métricas y KPIs (cpm, cpc, ctr, tasa_lpv, costo_por_cita, vtr, etc.) de la campaña Q2.
    Filtros opcionales: fechas 'YYYY-MM-DD' entre 2026-04-01 y 2026-06-30, pais (US, CA, PR),
    site (facebook, tiktok, snapchat, pinterest, reddit), funnel (upper, middle, lower).
    agrupar_por: lista con fecha, semana, mes, pais, site, funnel o creative."""
    dims = agrupar_por or []
    where, params = _where(fecha_inicio, fecha_fin, pais, site, funnel, creative_contiene)
    sel = ", ".join([_DIM_SQL[d] for d in dims] + [f"SUM({c}) AS {c}" for c in BASE])
    group = f"GROUP BY {', '.join(map(str, range(1, len(dims) + 1)))} ORDER BY 1" if dims else ""
    df = _kpis(_query(f"SELECT {sel} FROM {TABLA} {where} {group}", params))
    return _registros(df.head(100))  # límite para no saturar el contexto del modelo


@mcp.tool()
def ranking_q2(dimension: Literal["site", "pais", "funnel", "creative"], kpi: KPI,
               n: int = 5, mejores: bool = True, pais: Optional[str] = None,
               site: Optional[str] = None, fecha_inicio: Optional[str] = None,
               fecha_fin: Optional[str] = None) -> list[dict]:
    """Top N (mejores=True) o peores N (mejores=False) de la campaña Q2 según un KPI.
    Para KPIs de costo (cpm, cpc, costo_por_cita, cpv...) 'mejor' es el valor más bajo.
    Ignora filas con muy poco volumen para que el ranking sea confiable."""
    where, params = _where(fecha_inicio, fecha_fin, pais, site, None, None)
    df = _kpis(_query(f"SELECT {dimension}, {', '.join(f'SUM({c}) AS {c}' for c in BASE)} "
                      f"FROM {TABLA} {where} GROUP BY 1", params))
    df = df[(df.impressions >= 10_000) & df[kpi].notna()]
    asc = (kpi in _KPI_COSTO) == mejores
    cols = [dimension, kpi, "media_cost", "impressions", "clicks", "subscriptions"]
    return _registros(df.sort_values(kpi, ascending=asc).head(n)[cols])


@mcp.tool()
def detectar_anomalias_q2() -> list[dict]:
    """Busca problemas en la campaña Q2: tracking caído (impresiones sin clicks),
    gasto sin entrega (costo sin impresiones), video completes en 0 con video starts
    normales, y picos de CPM (más de 2.5 veces la mediana de esa plataforma y país)."""
    hallazgos = []
    diario = _kpis(_query(f"SELECT fecha, pais, site, {', '.join(f'SUM({c}) AS {c}' for c in BASE)} "
                          f"FROM {TABLA} GROUP BY 1, 2, 3"))

    for _, r in diario[(diario.impressions > 5_000) & (diario.clicks == 0)].iterrows():
        hallazgos.append(dict(tipo="tracking_caido", fecha=str(r.fecha.date()), pais=r.pais, site=r.site,
                              impressions=int(r.impressions),
                              detalle="Impresiones normales pero 0 clicks, LPV y citas."))

    sin_entrega = _query(f"SELECT fecha, pais, site, creative, media_cost FROM {TABLA} "
                         f"WHERE media_cost > 20 AND impressions = 0")
    for _, r in sin_entrega.iterrows():
        hallazgos.append(dict(tipo="gasto_sin_entrega", fecha=str(r.fecha.date()), pais=r.pais,
                              site=r.site, creative=r.creative, media_cost=float(r.media_cost),
                              detalle="Costo mayor a 0 con 0 impresiones."))

    for _, r in diario[(diario.video_starts > 1_000) & (diario.video_completes == 0)].iterrows():
        hallazgos.append(dict(tipo="video_completes_en_cero", fecha=str(r.fecha.date()), pais=r.pais,
                              site=r.site, video_starts=int(r.video_starts),
                              detalle="Video starts normales pero 0 video completes."))

    med = diario.groupby(["pais", "site"]).cpm.transform("median")
    for _, r in diario[diario.cpm > 2.5 * med].iterrows():
        hallazgos.append(dict(tipo="pico_cpm", fecha=str(r.fecha.date()), pais=r.pais, site=r.site,
                              cpm=float(r.cpm), media_cost=float(r.media_cost),
                              detalle="CPM muy por encima de lo normal para esa plataforma y país."))

    return sorted(hallazgos, key=lambda h: h["fecha"]) or [{"detalle": "Sin anomalías detectadas."}]


if __name__ == "__main__":
    mcp.run()  # stdio por defecto
