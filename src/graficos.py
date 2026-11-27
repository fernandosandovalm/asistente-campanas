"""
Herramienta 5: gráficos deterministas con Plotly.
Los números salen de tools.consultar_metricas (DuckDB); el LLM solo elige qué graficar.
"""
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import tools as T

TEMPORALES = {"fecha", "semana", "mes"}
MENOR_ES_MEJOR = {"cpm", "cpc", "costo_por_lpv", "costo_por_cita", "cpv", "cpcv"}

ETIQUETAS = {
    "media_cost": "Inversión (USD)", "impressions": "Impresiones", "clicks": "Clicks", "lpv": "LPV",
    "subscriptions": "Citas", "video_starts": "Video starts", "video_completes": "Video completes",
    "cpm": "CPM (USD)", "cpc": "CPC (USD)", "ctr": "CTR (%)", "tasa_lpv": "Tasa LPV (%)",
    "costo_por_lpv": "Costo por LPV (USD)", "tasa_conversion": "Tasa de conversión (%)",
    "costo_por_cita": "Costo por cita (USD)", "view_rate": "View rate (%)", "vtr": "VTR (%)",
    "cpv": "CPV (USD)", "cpcv": "CPCV (USD)",
    "fecha": "Fecha", "semana": "Semana", "mes": "Mes", "pais": "País", "site": "Plataforma",
    "funnel": "Funnel", "creative": "Creativo",
}


def grafico(kpi: str = "media_cost", por: str = "semana", desglose: Optional[str] = None,
            top: int = 15, **filtros) -> tuple[go.Figure, pd.DataFrame]:
    """Devuelve (figura, datos). Línea si `por` es temporal, barras si es categórico.
    `desglose` agrega una segunda dimensión como color (ej. por=semana, desglose=site)."""
    if desglose == por:
        desglose = None
    dims = [por] + ([desglose] if desglose else [])
    df = T.consultar_metricas(dims, **filtros)
    if kpi not in df.columns:
        raise ValueError(f"KPI '{kpi}' no existe. Opciones: {', '.join(ETIQUETAS)}")
    df = df.dropna(subset=[kpi])

    temporal = por in TEMPORALES
    if not temporal and not desglose:
        # barras ordenadas de mejor a peor; para creativos solo el top N
        df = df.sort_values(kpi, ascending=kpi in MENOR_ES_MEJOR)
        if por == "creative":
            df = df[df.impressions >= 10_000].head(top)

    titulo = f"{ETIQUETAS.get(kpi, kpi)} por {ETIQUETAS.get(por, por).lower()}"
    if desglose:
        titulo += f" y {ETIQUETAS.get(desglose, desglose).lower()}"
    filtros_txt = ", ".join(f"{k}={v}" for k, v in filtros.items() if v)
    if filtros_txt:
        titulo += f" ({filtros_txt})"

    kwargs = dict(x=por, y=kpi, color=desglose, title=titulo, labels=ETIQUETAS)
    if temporal:
        fig = px.line(df, markers=True, **kwargs)
    else:
        fig = px.bar(df, barmode="group", **kwargs)
        if por == "creative":
            fig.update_xaxes(tickangle=-45)
    fig.update_layout(template="plotly_white", legend_title_text=ETIQUETAS.get(desglose, ""),
                      margin=dict(t=60, l=10, r=10, b=10))
    return fig, df[[c for c in dims + [kpi] if c in df.columns]]
