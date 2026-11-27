"""
Construye el dataset de evaluación con respuestas de referencia calculadas desde DuckDB
(la verdad sale de los datos, no se escribe a mano).  Uso:  python -m src.construir_eval
"""
import json
from src import tools as T
from src.config import EVAL_PATH


def _top(df, col):
    return df.iloc[0][col]


def construir():
    q = []
    add = lambda pregunta, respuesta, valor=None, herramienta=None, tipo="numerica": q.append(dict(
        inputs={"pregunta": pregunta},
        outputs={"respuesta": respuesta, "valor": valor, "herramienta_esperada": herramienta, "tipo": tipo}))

    tot = T.consultar_metricas().iloc[0]
    add("¿Cuánto se invirtió en total en social durante el Q3?", f"${tot.media_cost:,.2f}",
        round(float(tot.media_cost), 2), "consultar_metricas")
    add("¿Cuántas citas de examen visual se generaron en el trimestre?", f"{int(tot.subscriptions):,}",
        int(tot.subscriptions), "consultar_metricas")

    s = T.consultar_metricas(["site"])
    fila = s.sort_values("costo_por_cita").iloc[0]
    add("¿Qué plataforma tuvo el menor costo por cita en todo el trimestre?",
        f"{fila.site} (${fila.costo_por_cita:,.2f})", fila.site, "ranking")
    fila = s.sort_values("ctr", ascending=False).iloc[0]
    add("¿Qué plataforma tuvo el CTR más alto?", f"{fila.site} ({fila.ctr:.2f}%)", fila.site, "ranking")

    us_ago = T.consultar_metricas(["site"], "2026-08-01", "2026-08-31", pais="US").sort_values("costo_por_cita").iloc[0]
    add("En agosto, en US, ¿qué plataforma tuvo el menor costo por cita?",
        f"{us_ago.site} (${us_ago.costo_por_cita:,.2f})", us_ago.site, "ranking")

    p = T.consultar_metricas(["pais"])
    for _, r in p.iterrows():
        add(f"¿Cuál fue el CPM en {r.pais}?", f"${r.cpm:,.2f}", round(float(r.cpm), 2), "consultar_metricas")

    m = T.consultar_metricas(["mes"]).sort_values("subscriptions", ascending=False).iloc[0]
    add("¿Qué mes generó más citas de examen visual?", f"{m.mes} ({int(m.subscriptions):,} citas)", m.mes,
        "consultar_metricas")

    snap = T.consultar_metricas(site="snapchat").iloc[0]
    add("¿Cuál fue el VTR (tasa de completado) de Snapchat?", f"{snap.vtr:.2f}%", round(float(snap.vtr), 2),
        "consultar_metricas")

    peor = T.ranking("creative", "vtr", 1, mejores=False, site="tiktok").iloc[0]
    add("¿Cuál es el creativo de TikTok con peor VTR?", f"{peor.creative} ({peor.vtr:.2f}%)", peor.creative, "ranking")

    lower = T.consultar_metricas(funnel="lower").iloc[0]
    add("¿El lower funnel cumplió el objetivo de costo por cita del brief?",
        f"Costo por cita lower = ${lower.costo_por_cita:,.2f} vs objetivo $25 -> "
        f"{'sí cumple' if lower.costo_por_cita <= 25 else 'no cumple'}",
        round(float(lower.costo_por_cita), 2), "consultar_metricas+buscar_conocimiento", "mixta")

    add("¿Hubo algún problema de tracking en Facebook?",
        "Sí: el 2026-08-14 en US, Facebook tuvo impresiones normales pero 0 clicks y 0 LPV (posible pixel caído).",
        "2026-08-14", "detectar_anomalias", "anomalia")
    add("¿Detectas gasto sin entrega en septiembre?",
        "Sí: el 2026-09-03 un creativo de TikTok en CA gastó dinero con 0 impresiones.",
        "2026-09-03", "detectar_anomalias", "anomalia")

    add("¿Qué es la tasa LPV y cuándo es preocupante?",
        "LPV/clicks × 100; por debajo de 30 % sugiere tráfico sospechoso o clicks inválidos.",
        None, "buscar_conocimiento", "conceptual")
    add("¿Cuándo es la temporada de regreso a clases en la campaña y qué mensajes se usan?",
        "Del 1 de agosto al 7 de septiembre, con más inversión diaria y mensajes de exámenes visuales para niños.",
        None, "buscar_conocimiento", "conceptual")
    add("¿Qué KPIs debo usar para evaluar el upper funnel?",
        "CPM, impresiones/alcance, view rate, VTR y CPCV; no costo por cita.", None, "buscar_conocimiento", "conceptual")

    EVAL_PATH.parent.mkdir(exist_ok=True)
    EVAL_PATH.write_text(json.dumps(q, indent=2, ensure_ascii=False, default=str))
    print(f"{len(q)} preguntas -> {EVAL_PATH}")


if __name__ == "__main__":
    construir()
