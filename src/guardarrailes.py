"""
Guardarraíles deterministas que corren ANTES de ejecutar cualquier herramienta (middleware wrap_tool_call).
Corrigen los errores típicos de un LLM local de 7B al llenar argumentos (ver informe v1–v5).

Viven en un archivo propio para que el agente único y todos los especialistas del multiagente
usen exactamente las mismas reglas: un especialista sin este middleware repetiría los errores de v1.
"""
import re
from typing import Optional

from langchain.agents.middleware import wrap_tool_call

from src.herramientas import MENOR_ES_MEJOR

VACIOS = ({}, [], "", "null", "None", "none")

# Palabras que deben aparecer en la conversación para aceptar cada filtro
MENCIONES = {
    "pais": {"US": ["us", "usa", "ee. uu", "ee.uu", "eeuu", "estados unidos"],
             "CA": ["ca", "canadá", "canada"],
             "PR": ["pr", "puerto rico"]},
    "funnel": {"upper": ["upper", "alcance", "awareness"], "middle": ["middle", "consideración", "consideracion"],
               "lower": ["lower", "conversión", "conversion"]},
}
VALIDOS = {
    "pais": {"US", "CA", "PR"},
    "site": {"facebook", "tiktok", "snapchat", "pinterest", "reddit"},
    "funnel": {"upper", "middle", "lower"},
}

# Argumentos que realmente recibió cada herramienta (para depurar).
# Es una sola lista compartida: no la reasignes (usa REGISTRO.clear() para vaciarla).
REGISTRO = []

SINONIMOS_KPI = {
    "citas": "subscriptions", "cita": "subscriptions", "citas de examen visual": "subscriptions",
    "conversiones": "subscriptions", "inversion": "media_cost", "inversión": "media_cost",
    "gasto": "media_cost", "costo": "media_cost", "impresiones": "impressions",
    "tasa de completado": "vtr", "completion rate": "vtr", "costo por cita": "costo_por_cita", "cpa": "costo_por_cita",
}

KPI_EN_PREGUNTA = [   # el orden importa: "costo por cita" antes que "cita"
    ("costo por cita", "costo_por_cita"), ("cpa", "costo_por_cita"), ("costo por lpv", "costo_por_lpv"),
    ("tasa de completado", "vtr"), ("vtr", "vtr"), ("view rate", "view_rate"), ("tasa lpv", "tasa_lpv"),
    ("tasa de conversión", "tasa_conversion"), ("ctr", "ctr"), ("cpm", "cpm"), ("cpc", "cpc"),
    ("cpcv", "cpcv"), ("cpv", "cpv"), ("citas", "subscriptions"), ("inversión", "media_cost"),
    ("invirt", "media_cost"), ("gasto", "media_cost"), ("impresiones", "impressions"), ("clicks", "clicks"),
]


def _texto_usuario(request) -> str:
    """Todo lo que ha escrito el usuario en esta conversación (para preguntas de seguimiento)."""
    msgs = (request.state or {}).get("messages", [])
    return " ".join(m.content for m in msgs if getattr(m, "type", "") == "human" and isinstance(m.content, str)).lower()


def _kpi_de_pregunta(texto: str) -> Optional[str]:
    for palabra, kpi in KPI_EN_PREGUNTA:
        if re.search(rf"(?<![a-z]){re.escape(palabra)}(?![a-z])", texto):
            return kpi
    return None


def _orden_de_pregunta(texto: str, kpi: str) -> Optional[str]:
    """menor/mayor explícitos; mejor/peor se traducen según si el KPI es un costo."""
    if re.search(r"\b(menor|más baj[oa]|mínim[oa]|menos)\b", texto):
        return "menor"
    if re.search(r"\b(mayor|más alt[oa]|máxim[oa]|más)\b", texto):
        return "mayor"
    es_costo = kpi in MENOR_ES_MEJOR
    if re.search(r"\bpeor(es)?\b", texto):
        return "mayor" if es_costo else "menor"
    if re.search(r"\bmejor(es)?\b", texto):
        return "menor" if es_costo else "mayor"
    return None


def _mencionado(valor: str, palabras: list[str], texto: str) -> bool:
    return any(re.search(rf"(?<![a-záéíóúñ]){re.escape(p)}(?![a-záéíóúñ])", texto) for p in palabras + [valor.lower()])


@wrap_tool_call
def limpiar_argumentos(request, handler):
    """Guardarraíl antes de ejecutar cada herramienta:
    1) quita valores vacíos ({} , "", "null") que mandan los modelos pequeños;
    2) normaliza agrupar_por ("site" o {"site": True} -> ["site"]);
    3) traduce sinónimos de KPI (citas -> subscriptions);
    4) descarta filtros de país, plataforma o funnel que el usuario nunca mencionó;
    5) ranking_q2 (MCP): KPI de la pregunta y orden menor/mayor traducido a mejores=True/False."""
    args = {k: v for k, v in request.tool_call["args"].items() if v is not None and v not in VACIOS}

    g = args.get("agrupar_por")
    if isinstance(g, str):
        args["agrupar_por"] = [g]
    elif isinstance(g, dict):
        args["agrupar_por"] = [k for k, v in g.items() if v]

    if isinstance(args.get("kpi"), str):
        k = args["kpi"].strip().lower()
        args["kpi"] = SINONIMOS_KPI.get(k, k.replace(" ", "_"))

    # listas o textos con comas ("US,CA,PR", ["facebook", "tiktok", ...]): el modelo enumera valores
    # cuando quiere decir "todos". Con un solo valor válido se usa ese; con varios se quita el filtro.
    for campo, validos in VALIDOS.items():
        v = args.get(campo)
        if isinstance(v, str) and "," in v:
            v = v.split(",")
        if isinstance(v, (list, tuple)):
            norm = {(str(x).strip().upper() if campo == "pais" else str(x).strip().lower()) for x in v}
            norm &= validos
            if len(norm) == 1:
                args[campo] = norm.pop()
            else:
                args.pop(campo)

    # valores que no existen en los datos ("social", "all", "todas", "Q3"...) se descartan
    for campo, validos in VALIDOS.items():
        v = args.get(campo)
        if isinstance(v, str):
            v = v.strip().upper() if campo == "pais" else v.strip().lower()
            if v in validos:
                args[campo] = v
            else:
                args.pop(campo)

    texto = _texto_usuario(request)

    # ranking: KPI y orden explícitos en la pregunta mandan sobre lo que eligió el modelo
    if request.tool_call["name"] == "ranking" and texto:
        ultima = texto.split("¿")[-1]
        k = _kpi_de_pregunta(ultima)
        if k:
            args["kpi"] = k
        o = _orden_de_pregunta(ultima, args.get("kpi", ""))
        if o:
            args["orden"] = o
        args.pop("mejores", None)

    # ranking_q2 (servidor MCP de Q2): mismo arreglo de KPI, pero el orden se expresa con
    # mejores=True/False. En ranking_q2, "mejor" de un costo es el valor más bajo.
    if request.tool_call["name"] == "ranking_q2" and texto:
        ultima = texto.split("¿")[-1]
        k = _kpi_de_pregunta(ultima)
        if k:
            args["kpi"] = k
        o = args.pop("orden", None) or _orden_de_pregunta(ultima, args.get("kpi", ""))
        if o in ("menor", "mayor"):
            es_costo = args.get("kpi", "") in MENOR_ES_MEJOR
            args["mejores"] = (o == "menor") == es_costo
        if not args.get("dimension"):
            for palabra, dim in (("plataforma", "site"), ("creativo", "creative"), ("país", "pais"),
                                 ("pais", "pais"), ("funnel", "funnel")):
                if palabra in ultima:
                    args["dimension"] = dim
                    break

    # ranking sin dimension: se deduce de la pregunta (por defecto caería en creativos)
    if request.tool_call["name"] == "ranking" and not args.get("dimension"):
        ultima = texto.split("¿")[-1] if texto else ""
        for palabra, dim in (("plataforma", "site"), ("creativo", "creative"), ("mes", "mes"),
                             ("semana", "semana"), ("país", "pais"), ("pais", "pais"), ("funnel", "funnel")):
            if palabra in ultima:
                args["dimension"] = dim
                break

    if texto:
        for campo in ("pais", "funnel"):
            v = args.get(campo)
            if isinstance(v, str):
                palabras = MENCIONES[campo].get(v.upper() if campo == "pais" else v.lower(), [])
                if not _mencionado(v, palabras, texto):
                    args.pop(campo)
        v = args.get("site")
        if isinstance(v, str) and not _mencionado(v, [], texto):
            args.pop("site")

    REGISTRO.append({"herramienta": request.tool_call["name"],
                     "modelo_envio": request.tool_call["args"], "ejecutado": args})
    return handler(request.override(tool_call={**request.tool_call, "args": args}))
