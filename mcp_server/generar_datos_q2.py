"""
Generador de datos sintéticos — Ben & Frank, campaña social Q2 2026 (abril–junio).

Fuente de datos INDEPENDIENTE de Q3: vive dentro de mcp_server/ y solo se consulta
a través del servidor MCP. Mismo esquema de columnas que el CSV de Q3, para que
los KPIs y las comparaciones entre trimestres sean directas.

Salidas (en mcp_server/data/):
  campanas_q2.csv          datos diarios (sin fila de total, como un pull del data lake)
  catalogo_creativos_q2.csv
  anomalias_q2.json        anomalías plantadas (para evaluación en LangSmith)
  campanas_q2.duckdb       tabla `campanas_q2` + `catalogo_creativos_q2`

Uso:  uv run generar_datos_q2.py      (o: python generar_datos_q2.py)
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 2026_02  # distinta a la de Q3 → datos distintos pero reproducibles
FECHA_INICIO, FECHA_FIN = "2026-04-01", "2026-06-30"
PRESUPUESTO_TOTAL = 420_000  # USD en el trimestre

DATA = Path(__file__).parent / "data"
CSV = DATA / "campanas_q2.csv"
CATALOGO = DATA / "catalogo_creativos_q2.csv"
ANOMALIAS = DATA / "anomalias_q2.json"
DUCKDB_PATH = DATA / "campanas_q2.duckdb"
TABLA = "campanas_q2"

PAISES = {"US": 0.50, "CA": 0.25, "PR": 0.25}
CPM_PAIS = {"US": 1.00, "CA": 0.85, "PR": 0.60}

# Parámetros por plataforma: peso del presupuesto, CPM base, CTR base, tasa LPV,
# view rate (video starts / imps) y factor de completado de video
SITES = {
    "facebook":  dict(peso=0.32, cpm=9.5,  ctr=0.011, lpv=0.78, vr=0.30, cf=0.95),
    "tiktok":    dict(peso=0.26, cpm=7.0,  ctr=0.008, lpv=0.62, vr=0.55, cf=0.80),
    "snapchat":  dict(peso=0.16, cpm=5.5,  ctr=0.006, lpv=0.58, vr=0.45, cf=0.85),
    "pinterest": dict(peso=0.16, cpm=6.5,  ctr=0.009, lpv=0.74, vr=0.20, cf=1.00),
    "reddit":    dict(peso=0.10, cpm=4.5,  ctr=0.005, lpv=0.70, vr=0.18, cf=0.90),
}
FUNNEL = {
    "upper":  dict(peso=0.40, cpm=0.85, ctr=0.70, conv=0.020),
    "middle": dict(peso=0.35, cpm=1.00, ctr=1.00, conv=0.045),
    "lower":  dict(peso=0.25, cpm=1.30, ctr=1.45, conv=0.090),
}

# Bancos para construir nombres de creativos con la misma convención que Q3
MENSAJES = [
    "Clear vision all day", "Frames that fit your face", "Book your eye exam today",
    "Blue light, no problem", "Try 5 frames at home", "Your new everyday pair",
    "Lenses made in 24h", "Style meets prescription", "See the difference",
    "Eye exams for the whole family", "Polarized and on point", "Light frames, big look",
    "Find your shape", "Insurance accepted", "Kids frames that last",
    "Glasses in under a week", "Swap your old pair", "Comfort you can see",
]
COLA_SITE = {"facebook": "FB feed", "tiktok": "TikTok", "snapchat": "Snap ads",
             "pinterest": "Pin", "reddit": "Promoted"}
N_CREATIVOS = {"facebook": 22, "tiktok": 18, "snapchat": 14, "pinterest": 14, "reddit": 10}


def construir_catalogo(rng: np.random.Generator) -> pd.DataFrame:
    filas, post = [], 1
    for site, n in N_CREATIVOS.items():
        mensajes = rng.choice(MENSAJES, size=n, replace=n > len(MENSAJES))
        for i, msg in enumerate(mensajes):
            prob_video = {"tiktok": 1.0, "snapchat": 0.85, "facebook": 0.6,
                          "pinterest": 0.35, "reddit": 0.4}[site]
            formato = "video" if rng.random() < prob_video else "estatico"
            dur = int(rng.choice([6, 10, 15, 20, 30])) if formato == "video" else 0
            if site in ("facebook", "pinterest") and i % 2 == 0:
                nombre = f"Post {post} - {msg}" + (f" {dur}s" if dur else "")
                post += 1
            else:
                nombre = f"{msg}" + (f" {dur}s" if dur else "") + f" - {COLA_SITE[site]} v{i + 1}"
            filas.append(dict(
                creative=nombre, site=site,
                funnel=rng.choice(list(FUNNEL), p=[f["peso"] for f in FUNNEL.values()]),
                formato=formato, duracion_s=dur,
                calidad=float(np.clip(rng.lognormal(0, 0.25), 0.5, 1.8)),
                # cada creativo corre en una ventana del trimestre (rotación de piezas)
                inicio=pd.Timestamp(FECHA_INICIO) + pd.Timedelta(days=int(rng.integers(0, 45))),
            ))
    cat = pd.DataFrame(filas).drop_duplicates("creative").reset_index(drop=True)
    cat["fin"] = [min(pd.Timestamp(FECHA_FIN), i + pd.Timedelta(days=int(rng.integers(35, 91))))
                  for i in cat.inicio]
    return cat


def generar():
    rng = np.random.default_rng(SEED)
    cat = construir_catalogo(rng)
    dias = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    # estacionalidad semanal suave (más inversión mar–jue) + rampa leve a lo largo del trimestre
    semana = {0: 1.02, 1: 1.08, 2: 1.08, 3: 1.05, 4: 0.97, 5: 0.90, 6: 0.90}
    rampa = np.linspace(0.92, 1.08, len(dias))
    pesos_dia = np.array([semana[d.dayofweek] for d in dias]) * rampa
    presup_dia = PRESUPUESTO_TOTAL * pesos_dia / pesos_dia.sum()

    registros = []
    for k, dia in enumerate(dias):
        activos = cat[(cat.inicio <= dia) & (cat.fin >= dia)]
        for pais, share in PAISES.items():
            for site, s in SITES.items():
                act = activos[activos.site == site]
                if act.empty:
                    continue
                presupuesto = presup_dia[k] * share * s["peso"]
                w = act.calidad.values * rng.lognormal(0, 0.25, len(act))
                for (_, c), costo in zip(act.iterrows(), presupuesto * w / w.sum()):
                    fu = FUNNEL[c.funnel]
                    if rng.random() < 0.015:  # filas casi vacías, como en el reporte real
                        costo = 0.0
                    cpm = s["cpm"] * CPM_PAIS[pais] * fu["cpm"] * rng.lognormal(0, 0.15)
                    imps = int(costo / cpm * 1000) if costo > 0 else int(rng.integers(0, 6))
                    ctr = s["ctr"] * fu["ctr"] * c.calidad * (1.1 if c.formato == "estatico" else 1.0)
                    clicks = int(rng.binomial(imps, min(ctr, 0.2)))
                    lpv = int(rng.binomial(clicks, min(s["lpv"] * rng.uniform(0.9, 1.1), 1.0)))
                    subs = int(rng.binomial(lpv, min(fu["conv"] * c.calidad, 0.5)))
                    vs = vc = 0
                    if c.formato == "video":
                        vs = int(rng.binomial(imps, min(s["vr"] * rng.uniform(0.85, 1.15), 1.0)))
                        cr = min(0.9, 1.1 * np.exp(-c.duracion_s / 22) * s["cf"])
                        vc = int(rng.binomial(vs, cr))
                    registros.append(dict(
                        fecha=dia.date(), pais=pais, site=site, funnel=c.funnel, creative=c.creative,
                        media_cost=round(float(costo), 2), impressions=imps, clicks=clicks,
                        subscriptions=subs, lpv=lpv, video_starts=vs, video_completes=vc,
                    ))
    df = pd.DataFrame(registros)
    df, anomalias = plantar_anomalias(df)
    df = df.sort_values(["fecha", "pais", "site", "creative"]).reset_index(drop=True)
    return cat, df, anomalias


def plantar_anomalias(df: pd.DataFrame):
    """Anomalías con fecha conocida, distintas a las de Q3."""
    a = []
    f = pd.to_datetime(df.fecha)

    # 1) Tracking caído: Snapchat PR el 22-abr → impresiones normales, 0 clicks/LPV/citas
    m = (f == "2026-04-22") & (df.site == "snapchat") & (df.pais == "PR")
    df.loc[m, ["clicks", "lpv", "subscriptions"]] = 0
    a.append(dict(tipo="tracking_caido", fecha="2026-04-22", site="snapchat", pais="PR",
                  descripcion="Impresiones normales pero clicks, LPV y subscriptions en 0."))

    # 2) Gasto sin entrega: el creativo de mayor gasto de Pinterest US el 13-may
    m = (f == "2026-05-13") & (df.site == "pinterest") & (df.pais == "US") & (df.media_cost > 0)
    idx = df[m].sort_values("media_cost").index[-1]
    df.loc[idx, ["impressions", "clicks", "lpv", "subscriptions", "video_starts", "video_completes"]] = 0
    a.append(dict(tipo="gasto_sin_entrega", fecha="2026-05-13", site="pinterest", pais="US",
                  creative=df.loc[idx, "creative"], media_cost=float(df.loc[idx, "media_cost"]),
                  descripcion="Costo > 0 con 0 impresiones."))

    # 3) Video sin completados: TikTok CA del 03 al 05-jun → starts normales, completes en 0
    m = (f >= "2026-06-03") & (f <= "2026-06-05") & (df.site == "tiktok") & (df.pais == "CA")
    df.loc[m, "video_completes"] = 0
    a.append(dict(tipo="video_completes_en_cero", fecha="2026-06-03 a 2026-06-05", site="tiktok",
                  pais="CA", descripcion="Video starts normales pero video completes en 0 (tracking de video)."))

    # 4) Pico de CPM: Reddit US el 18-jun → costo x4 con la misma entrega
    m = (f == "2026-06-18") & (df.site == "reddit") & (df.pais == "US")
    df.loc[m, "media_cost"] = (df.loc[m, "media_cost"] * 4).round(2)
    a.append(dict(tipo="pico_cpm", fecha="2026-06-18", site="reddit", pais="US",
                  descripcion="Costo 4 veces mayor con impresiones normales (CPM disparado)."))
    return df, a


def ingest():
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLA} AS
        SELECT CAST(fecha AS DATE) AS fecha, pais, site, funnel, creative,
               CAST(media_cost AS DOUBLE) AS media_cost,
               impressions, clicks, subscriptions, lpv, video_starts, video_completes
        FROM read_csv_auto('{CSV.as_posix()}', header=true)
    """)
    con.execute(f"CREATE OR REPLACE TABLE catalogo_creativos_q2 AS "
                f"SELECT * FROM read_csv_auto('{CATALOGO.as_posix()}')")
    n = con.execute(f"SELECT COUNT(*) FROM {TABLA}").fetchone()[0]
    con.close()
    return n


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    cat, df, anomalias = generar()

    # validaciones de coherencia
    assert (df.lpv <= df.clicks).all() and (df.subscriptions <= df.lpv).all()
    assert (df.clicks <= df.impressions).all() and (df.video_completes <= df.video_starts).all()

    df.to_csv(CSV, index=False)
    cat[["creative", "site", "funnel", "formato", "duracion_s"]].to_csv(CATALOGO, index=False)
    ANOMALIAS.write_text(json.dumps(anomalias, indent=2, ensure_ascii=False), encoding="utf-8")
    n = ingest()

    print(f"{len(df):,} filas | {df.creative.nunique()} creativos | ${df.media_cost.sum():,.0f} USD")
    print((df.groupby("pais").media_cost.sum() / df.media_cost.sum()).round(3).to_dict())
    print(f"DuckDB: {n:,} filas en {DUCKDB_PATH.name}:{TABLA}")
