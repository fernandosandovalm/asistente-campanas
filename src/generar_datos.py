"""
Generador de datos sintéticos: Ben & Frank — Social — Q3 (jul–sep), nivel diario.

Simula un pull directo del data lake (Meta y demás plataformas sociales), sin fila de totales.
Las métricas se derivan en cadena para que sean coherentes:
    costo -> impresiones (CPM) -> clicks (CTR) -> LPV -> subscriptions (citas de examen visual)
    impresiones -> video_starts -> video_completes (solo creativos de video)

Uso:  python -m src.generar_datos
"""
import json
import numpy as np
import pandas as pd

from src.config import DATA_RAW, CATALOGO, ANOMALIAS

SEED = 42
rng = np.random.default_rng(SEED)

FECHA_INI, FECHA_FIN = "2026-07-01", "2026-09-30"
PRESUPUESTO_TOTAL = 750_000  # USD en el trimestre
REPARTO_PAIS = {"US": 0.50, "CA": 0.25, "PR": 0.25}

# --- Supuestos por plataforma (sintéticos) -----------------------------------
SITES = {
    #            n creativos, % video, duraciones,    peso $, CPM,  CTR,    view rate, factor completado, LPV/click
    "facebook":  dict(n=25, video=0.60, durs=[10, 15, 29, 30], peso=0.35, cpm=9.0, ctr=0.011, vr=0.35, cf=0.80, lpv=0.75),
    "tiktok":    dict(n=18, video=1.00, durs=[9, 15, 21, 30],  peso=0.25, cpm=6.5, ctr=0.009, vr=0.55, cf=0.90, lpv=0.62),
    "snapchat":  dict(n=16, video=0.85, durs=[6, 10, 15],      peso=0.15, cpm=5.0, ctr=0.006, vr=0.45, cf=1.00, lpv=0.60),
    "pinterest": dict(n=16, video=0.40, durs=[6, 15],          peso=0.15, cpm=5.5, ctr=0.0055, vr=0.25, cf=0.85, lpv=0.78),
    "reddit":    dict(n=12, video=0.45, durs=[10, 15, 30],     peso=0.10, cpm=7.0, ctr=0.004, vr=0.30, cf=0.75, lpv=0.80),
}
FUNNEL = {  # peso $, factor CPM, factor CTR, conversión LPV -> cita
    "upper":  dict(peso=0.45, cpm=0.80, ctr=0.70, conv=0.015),
    "middle": dict(peso=0.30, cpm=1.00, ctr=1.00, conv=0.030),
    "lower":  dict(peso=0.25, cpm=1.35, ctr=1.50, conv=0.060),
}
CPM_PAIS = {"US": 1.0, "CA": 0.8, "PR": 0.6}

TITULOS = [
    "Weather getting nicer", "Can your child see", "An exam is not just for mouses",
    "Back to school vision check", "Blue light blues", "Frames that fit your face",
    "Try 5 frames at home", "Book your eye exam in 2 min", "Sunnies season",
    "Your eyes deserve a checkup", "Prescription sunglasses", "Screen time eye strain",
    "Kids frames under 100", "Optometrist near you", "Free shipping and returns",
    "Style meets clarity", "Reading glasses reimagined", "Polarized for summer",
    "Meet our optometrists", "Insurance accepted", "First exam on us",
    "Vision changes with age", "Glasses in 7 days", "Find your shape",
    "Class photo ready", "See the board clearly", "Sport frames for kids",
]
TITULOS_B2S = {"Can your child see", "Back to school vision check", "Kids frames under 100",
               "Class photo ready", "See the board clearly", "Sport frames for kids"}


def nombre_creativo(site, n, titulo, es_video, dur):
    sufijo = f"{dur}s" if es_video else "Static"
    if site == "facebook":
        return f"Post {n} - {titulo} {sufijo}"
    if site == "snapchat":
        return f"{titulo} {sufijo} - Snap ads"
    pref = {"tiktok": "TT", "pinterest": "Pin", "reddit": "RD"}[site]
    return f"{pref} {n} - {titulo} {sufijo}"


def construir_catalogo():
    filas, dias = [], pd.date_range(FECHA_INI, FECHA_FIN)
    for site, p in SITES.items():
        titulos = rng.choice(TITULOS, size=p["n"], replace=False)
        for i, titulo in enumerate(titulos, start=1):
            es_video = rng.random() < p["video"]
            dur = int(rng.choice(p["durs"])) if es_video else 0
            # video se inclina a upper; estático a lower
            probs = [0.55, 0.30, 0.15] if es_video else [0.15, 0.35, 0.50]
            funnel = rng.choice(list(FUNNEL), p=probs)
            if titulo in TITULOS_B2S:  # campañas de regreso a clases: ago–sep
                ini = pd.Timestamp("2026-07-25") + pd.Timedelta(days=int(rng.integers(0, 15)))
                fin = pd.Timestamp("2026-09-20") + pd.Timedelta(days=int(rng.integers(0, 10)))
            elif rng.random() < 0.25:  # siempre activos
                ini, fin = dias[0], dias[-1]
            else:
                ini = dias[int(rng.integers(0, 67))]
                fin = min(ini + pd.Timedelta(days=int(rng.integers(14, 75))), dias[-1])
            paises = ["US"] + [c for c, pr in (("CA", 0.85), ("PR", 0.75)) if rng.random() < pr]
            filas.append(dict(
                creative=nombre_creativo(site, i, titulo, es_video, dur), site=site, funnel=funnel,
                formato="video" if es_video else "estatico", duracion_s=dur,
                fecha_inicio=ini.date(), fecha_fin=min(fin, dias[-1]).date(),
                paises=",".join(paises), calidad=float(np.round(rng.lognormal(0, 0.3), 3)),
                peso=float(np.round(rng.lognormal(0, 0.6), 3)), b2s=titulo in TITULOS_B2S,
            ))
    return pd.DataFrame(filas)


def curva_diaria():
    dias = pd.date_range(FECHA_INI, FECHA_FIN)
    semana = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.85, 6: 0.90}
    f = np.array([semana[d.weekday()] for d in dias])
    b2s = (dias >= "2026-08-01") & (dias <= "2026-09-07")
    f = f * np.where(b2s, 1.45, 1.0)
    return pd.Series(f / f.sum(), index=dias), pd.Series(b2s, index=dias)


def generar():
    cat = construir_catalogo()
    curva, es_b2s = curva_diaria()
    registros = []
    for dia, frac in curva.items():
        for pais, share in REPARTO_PAIS.items():
            presupuesto = PRESUPUESTO_TOTAL * share * frac
            act = cat[(pd.to_datetime(cat.fecha_inicio) <= dia) & (pd.to_datetime(cat.fecha_fin) >= dia)
                      & cat.paises.str.contains(pais)]
            if act.empty:
                continue
            w = (act.site.map(lambda s: SITES[s]["peso"]) * act.funnel.map(lambda f: FUNNEL[f]["peso"])
                 * act.peso * rng.lognormal(0, 0.25, len(act)))
            costos = presupuesto * w / w.sum()
            for (_, c), costo in zip(act.iterrows(), costos):
                s, fu = SITES[c.site], FUNNEL[c.funnel]
                b2s = bool(es_b2s[dia])
                if rng.random() < 0.015:  # filas casi vacías, como en el reporte real
                    costo = 0.0
                cpm = s["cpm"] * CPM_PAIS[pais] * fu["cpm"] * (1.10 if b2s else 1.0) * rng.lognormal(0, 0.15)
                imps = int(costo / cpm * 1000) if costo > 0 else int(rng.integers(0, 6))
                ctr = s["ctr"] * fu["ctr"] * c.calidad * (1.1 if c.formato == "estatico" else 1.0)
                clicks = int(rng.binomial(imps, min(ctr, 0.2)))
                lpv = int(rng.binomial(clicks, s["lpv"] * rng.uniform(0.9, 1.1)))
                conv = fu["conv"] * (1.5 if (c.b2s and b2s) else 1.0) * c.calidad
                subs = int(rng.binomial(lpv, min(conv, 0.5)))
                vs = vc = 0
                if c.formato == "video":
                    vs = int(rng.binomial(imps, s["vr"] * rng.uniform(0.85, 1.15)))
                    cr = min(0.9, 1.1 * np.exp(-c.duracion_s / 22) * s["cf"])
                    vc = int(rng.binomial(vs, cr))
                registros.append(dict(
                    fecha=dia.date(), pais=pais, site=c.site, funnel=c.funnel, creative=c.creative,
                    media_cost=round(float(costo), 2), impressions=imps, clicks=clicks,
                    subscriptions=subs, lpv=lpv, video_starts=vs, video_completes=vc,
                ))
    df = pd.DataFrame(registros)
    df, anomalias = plantar_anomalias(df)
    df = df.sort_values(["fecha", "pais", "site", "creative"]).reset_index(drop=True)
    return cat, df, anomalias


def plantar_anomalias(df):
    """Anomalías conocidas: sirven para probar la herramienta de detección y para la evaluación."""
    a = []
    f = pd.to_datetime(df.fecha)

    # 1) Pixel caído: Facebook US el 14-ago tiene entrega pero cero clicks / LPV / citas
    m = (f == "2026-08-14") & (df.site == "facebook") & (df.pais == "US")
    df.loc[m, ["clicks", "lpv", "subscriptions"]] = 0
    a.append(dict(tipo="tracking_caido", fecha="2026-08-14", site="facebook", pais="US",
                  descripcion="Impresiones normales pero clicks, LPV y subscriptions en 0 (posible pixel caído)."))

    # 2) Gasto sin entrega: un creativo de TikTok CA el 03-sep
    m = (f == "2026-09-03") & (df.site == "tiktok") & (df.pais == "CA") & (df.media_cost > 0)
    idx = df[m].sort_values("media_cost").index[-1]
    df.loc[idx, ["impressions", "clicks", "lpv", "subscriptions", "video_starts", "video_completes"]] = 0
    a.append(dict(tipo="gasto_sin_entrega", fecha="2026-09-03", site="tiktok", pais="CA",
                  creative=df.loc[idx, "creative"], media_cost=float(df.loc[idx, "media_cost"]),
                  descripcion="Costo > 0 con 0 impresiones."))

    # 3) Pico de CTR sospechoso (posible tráfico bot): un creativo de Pinterest US el 22-jul
    m = (f == "2026-07-22") & (df.site == "pinterest") & (df.pais == "US") & (df.impressions > 500)
    idx = df[m].sort_values("impressions").index[-1]
    df.loc[idx, "clicks"] = int(df.loc[idx, "clicks"] * 8 + 50)
    a.append(dict(tipo="pico_ctr_sospechoso", fecha="2026-07-22", site="pinterest", pais="US",
                  creative=df.loc[idx, "creative"],
                  descripcion="Clicks x8 sin aumento de LPV: la tasa LPV/click se desploma."))

    # 4) Sobregasto: Snapchat PR del 15 al 17-sep, costo x3 con impresiones normales (CPM x3)
    m = (f >= "2026-09-15") & (f <= "2026-09-17") & (df.site == "snapchat") & (df.pais == "PR")
    df.loc[m, "media_cost"] = (df.loc[m, "media_cost"] * 3).round(2)
    a.append(dict(tipo="sobregasto_cpm", fecha="2026-09-15/2026-09-17", site="snapchat", pais="PR",
                  descripcion="Costo triplicado con entrega normal: CPM ~3x."))
    return df, a


if __name__ == "__main__":
    cat, df, anomalias = generar()
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_RAW, index=False)
    cat.drop(columns=["calidad", "peso"]).to_csv(CATALOGO, index=False)
    ANOMALIAS.write_text(json.dumps(anomalias, indent=2, ensure_ascii=False))
    print(f"{len(df):,} filas | {df.creative.nunique()} creativos | costo total ${df.media_cost.sum():,.0f}")
