"""
Grafo multiagente (LangGraph). Patrones de diseño (diapositiva 5):
  - ROUTER: el supervisor clasifica la pregunta y la envía al especialista adecuado.
  - PARALELIZACIÓN: las preguntas mixtas (cifra + objetivo) corren Datos y Conocimiento a la vez.
  - EVALUADOR–OPTIMIZADOR: el crítico revisa la respuesta; si falla, el redactor la reescribe
    con su retroalimentación (máximo MAX_RONDAS_CRITICO rondas o hasta el tope de costo).

    pregunta -> supervisor --datos--------> datos ---------\
                           --anomalias----> anomalias -----+-> integrar -> (mixta) redactor -> critico
                           --conocimiento-> conocimiento --/            -> (simple) ----------> critico
                           --mixta--------> datos + conocimiento (en paralelo)
    critico -> aprobado / rondas agotadas / tope  -> entregar
            -> rechazado                          -> redactor -> critico ...

  MCP (datos de Q2, abril-junio): el especialista datos_q2 solo tiene las tools del servidor MCP
  local "campanas-q2" (src/mcp_cliente.py). Después de elegir la ruta, una regla determinista
  detecta el trimestre de la pregunta:
    Q2               -> datos/anomalias pasan a datos_q2; mixta pasa a mixta_q2 (datos_q2 + conocimiento)
    Q2 y Q3 a la vez -> trimestres: datos (Q3) + datos_q2 (Q2) en paralelo -> redactor compara

Uso:
    from src.grafo import crear_multiagente, responder
    grafo = crear_multiagente()
    r = responder(grafo, "¿Qué plataforma tuvo el menor costo por cita?", thread_id="demo")
    r["respuesta"], r["ruta"], r["herramientas"], r["figuras"], r["veredicto"]
"""
import operator
import re
from typing import Annotated, Literal, Optional, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agentes import TAREA_MIXTA, crear_especialistas
from src.config import (CRITICO_LLM, LLM_PRINCIPAL, MAX_RONDAS_CRITICO, MODELO_REFERENCIA_SOMBRA,
                        ROUTER_MODO, TOPE_USD_POR_TAREA)
from src.costos import costo_usd
from src.guardarrailes import limpiar_argumentos
from src.mcp_cliente import herramientas_mcp

Ruta = Literal["datos", "anomalias", "conocimiento", "mixta"]
# Rutas con más de un especialista: siempre pasan por el redactor
RUTAS_MULTIPLES = {"mixta", "mixta_q2", "trimestres"}


# --------------------------------------------------------------------------- mensajes estructurados
class Enrutamiento(BaseModel):
    """Decisión del supervisor."""
    ruta: Ruta = Field(description="datos: cifras, rankings, comparaciones o gráficos. "
                                   "anomalias: problemas de entrega o tracking. "
                                   "conocimiento: definiciones, benchmarks, brief, conceptos. "
                                   "mixta: comparar una cifra contra un objetivo o benchmark.")
    razon: str = Field(description="Una frase corta con el motivo.")


class Veredicto(BaseModel):
    """Decisión del crítico."""
    aprobado: bool = Field(description="True si la respuesta responde la pregunta y es fiel a los datos.")
    problema: str = Field(default="", description="Si no se aprueba: qué corregir, en una frase.")


# --------------------------------------------------------------------------- estado
class Estado(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # conversación: humano + respuestas finales
    turno: int                                             # sube en cada pregunta (lo escribe el supervisor)
    ruta: str
    razon_ruta: str
    resultados: Annotated[list[dict], operator.add]        # salidas de especialistas (se filtran por turno)
    usos: Annotated[list[dict], operator.add]              # tokens por llamada (se filtran por turno)
    respuesta: str
    feedback: str
    rondas: int
    veredicto: str
    herramientas: list[str]
    figuras: list[str]


def _del_turno(estado: Estado, clave: str) -> list[dict]:
    t = estado.get("turno", 0)
    return [x for x in estado.get(clave, []) if x.get("turno") == t]


def _pregunta(estado: Estado) -> str:
    return next((m.content for m in reversed(estado["messages"]) if m.type == "human"), "")


def _uso(agente: str, mensajes, turno: int) -> list[dict]:
    """Tokens de cada AIMessage (Ollama y OpenRouter los reportan en usage_metadata)."""
    out = []
    for m in mensajes:
        u = getattr(m, "usage_metadata", None)
        if m is not None and getattr(m, "type", "") == "ai" and u:
            out.append({"turno": turno, "agente": agente,
                        "entrada": u.get("input_tokens", 0), "salida": u.get("output_tokens", 0)})
    return out


def _costo_turno(estado: Estado) -> float:
    return sum(costo_usd(u["entrada"], u["salida"], MODELO_REFERENCIA_SOMBRA) for u in _del_turno(estado, "usos"))


def _es_error(m) -> bool:
    return getattr(m, "status", "") == "error" or str(m.content).startswith("Error invoking tool")


_NIEGA = re.compile(r"no se (encontr|detect|identific|registr)|no hay (anomal|problema|hallazgo)|sin anomal|ninguna anomal", re.I)


def _etiqueta_trimestre(herramienta: str) -> str:
    if herramienta.endswith("_q2"):
        return " · Q2 abr-jun"
    if herramienta in ("consultar_metricas", "ranking", "detectar_anomalias", "generar_grafico"):
        return " · Q3 jul-sep"
    return ""


def _salidas_texto(resultados: list[dict]) -> str:
    bloques = []
    for r in resultados:
        for h, s in zip(r["herramientas"], r["salidas"]):
            bloques.append(f"[{r['agente']} · {h}{_etiqueta_trimestre(h)}]\n{s}")
    return "\n\n".join(bloques) or "(sin salidas de herramientas)"


# --------------------------------------------------------------------------- router por reglas (respaldo)
def _ruta_por_reglas(texto: str) -> Ruta:
    t = texto.lower()
    if re.search(r"objetivo|benchmark|meta\b|brief", t) and re.search(r"cumpl|alcanz|super|logr|lleg|frente|vs|compar", t):
        return "mixta"
    if re.search(r"anomal|problema|tracking|sin entrega|sobrecosto|raro|extraño|ca[ií]da", t):
        return "anomalias"
    if re.search(r"qu[eé] es\b|qu[eé] significa|c[oó]mo se calcula|qu[eé] kpis?|defin|temporada|convenci|benchmark|brief|objetivo", t):
        return "conocimiento"
    return "datos"


# --------------------------------------------------------------------------- trimestre (regla determinista)
# \b evita falsos positivos: "mayor" contiene "mayo".
_Q2 = re.compile(r"\bq2\b|\babril\b|\bmayo\b|\bjunio\b|segundo trimestre", re.I)
_Q3 = re.compile(r"\bq3\b|\bjulio\b|\bagosto\b|\bseptiembre\b|tercer trimestre", re.I)
_ENTRE = re.compile(r"entre (los )?trimestres|trimestre anterior|trimestre a trimestre|qoq", re.I)


def _trimestre(texto: str) -> str:
    """'q2', 'q3' o 'ambos'. Sin mención de fecha se asume Q3 (la campaña principal)."""
    q2, q3 = bool(_Q2.search(texto)), bool(_Q3.search(texto))
    if (q2 and q3) or _ENTRE.search(texto):
        return "ambos"
    return "q2" if q2 else "q3"


def _ajustar_por_trimestre(ruta: str, texto: str) -> str:
    """El supervisor elige el TIPO de tarea; esta regla elige la FUENTE de datos (nativa Q3 o MCP Q2)."""
    t = _trimestre(texto)
    if ruta == "conocimiento" or t == "q3":
        return ruta
    if t == "ambos":
        return "trimestres"
    return "mixta_q2" if ruta == "mixta" else "datos_q2"   # datos / anomalias de Q2


# Instrucción extra para cada especialista según la ruta (además de TAREA_MIXTA de src.agentes)
TAREAS_Q2 = {
    "mixta_q2": {"datos_q2": "Obtén SOLO la cifra de Q2 que pide la pregunta con tus herramientas; "
                             "el objetivo o benchmark lo busca otro agente."},
    "trimestres": {"datos": "Obtén SOLO la cifra de Q3 (2026-07-01 a 2026-09-30) que pide la pregunta. "
                            "Los datos de Q2 los consulta otro agente.",
                   "datos_q2": "Obtén SOLO la cifra de Q2 (2026-04-01 a 2026-06-30) que pide la pregunta, "
                               "con los mismos filtros. Los datos de Q3 los consulta otro agente."},
}

PROMPT_DATOS_Q2 = """Eres el analista de datos de la campaña social de Ben & Frank del Q2 2026 (abril a junio),
en US, CA y PR, plataformas facebook, tiktok, snapchat, pinterest y reddit.
Tus datos llegan desde un servidor MCP. Reglas:
- SIEMPRE llama al menos una herramienta antes de responder; todo número debe salir de una herramienta.
- Cifras y KPIs: consultar_metricas_q2 (usa agrupar_por para comparar, p. ej. ["site"]).
- "¿Cuál tuvo el menor/mayor X?": ranking_q2 con dimension y kpi exactos; llámala UNA sola vez.
- Problemas de tracking, entrega o costos raros: detectar_anomalias_q2.
- Si no sabes qué datos hay, usa resumen_q2.
- Meses: abril = 2026-04-01 a 2026-04-30, mayo = 2026-05-01 a 2026-05-31, junio = 2026-06-01 a 2026-06-30.
- No agregues filtros que el usuario no pidió. Citas = subscriptions; inversión = media_cost.
- Responde en español, conciso, con las cifras tal como salen de la herramienta."""


PROMPT_SUPERVISOR = """Eres el supervisor de un equipo de análisis de campañas sociales de Ben & Frank.
Clasifica la ÚLTIMA pregunta del usuario en una ruta:
- datos: cifras, totales, KPIs, rankings ("¿cuál tuvo el mayor/menor...?"), comparaciones o gráficos.
- anomalias: problemas de entrega, tracking caído, gasto sin entrega, sobrecostos.
- conocimiento: definiciones, qué KPI usar, benchmarks, objetivos del brief, temporadas, convenciones.
- mixta: comparar una cifra de los datos contra un objetivo o benchmark ("¿cumplió el objetivo...?").
Ejemplos: "¿Cuál fue el CPM en CA?" -> datos. "¿Qué mes generó más citas?" -> datos.
"Grafica la inversión semanal" -> datos. "¿Hubo problemas de tracking en Facebook?" -> anomalias.
"¿Qué es la tasa LPV?" -> conocimiento. "¿El lower funnel cumplió el objetivo de costo por cita?" -> mixta.
Si es una pregunta de seguimiento ("¿y en agosto?"), usa la conversación para entenderla."""

PROMPT_REDACTOR = """Eres el redactor final de un equipo de análisis de campañas de Ben & Frank.
Escribe la respuesta al usuario en español, concisa, usando SOLO las cifras que aparecen en los resultados
de las herramientas (cópialas tal cual, con sus decimales). No inventes cifras ni datos.
Si la pregunta compara una cifra con un objetivo: da la cifra, da el objetivo y di si se cumplió.
En costos (CPM, CPC, costo por cita, CPV), cumplir = quedar por DEBAJO del objetivo.
Cada resultado indica su trimestre (Q2 abr-jun o Q3 jul-sep). Si la pregunta compara trimestres:
da la cifra de Q2 y la de Q3 y di cuál es mayor o menor y si mejoró (en costos, mejorar = bajar).
No calcules porcentajes ni diferencias nuevas: usa solo cifras que aparezcan en los resultados.
Cierra con una recomendación breve si aplica."""

PROMPT_CRITICO = """Eres el crítico de calidad de un equipo de análisis de campañas.
Revisa si la RESPUESTA responde la PREGUNTA y es fiel a los RESULTADOS de las herramientas.
Rechaza si: no responde lo que se preguntó, usa una cifra que no está en los resultados,
o en una comparación contra un objetivo concluye al revés (en costos, cumplir = estar por debajo del objetivo).
Si todo está bien, apruébala. No exijas estilo ni detalles extra."""


# --------------------------------------------------------------------------- verificación determinista de cifras
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numeros(texto: str) -> list[float]:
    vals = []
    for tok in _NUM.findall(texto or ""):
        try:
            vals.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return vals


def cifras_sin_respaldo(respuesta: str, fuentes: str, pregunta: str = "") -> list[float]:
    """Números de la respuesta que no aparecen (±1 %) en las salidas de herramientas ni en la pregunta.
    Se ignoran enteros pequeños (días, 'TT 14', '5 plataformas') y años."""
    base = _numeros(fuentes) + _numeros(pregunta)
    sin = []
    for v in _numeros(respuesta):
        if (v.is_integer() and v <= 31) or v in (2026.0, 2025.0):
            continue
        if not any(abs(v - b) <= 0.01 * max(abs(b), 1.0) for b in base):
            sin.append(v)
    return sin


# --------------------------------------------------------------------------- construcción del grafo
def crear_multiagente(modelo: str = LLM_PRINCIPAL, llm=None, checkpointer=None,
                      router_modo: str = ROUTER_MODO, critico_llm: str = CRITICO_LLM,
                      max_rondas: int = MAX_RONDAS_CRITICO, tope_usd: float = TOPE_USD_POR_TAREA):
    """Compila el grafo. El mismo modelo (Qwen 2.5 7B) se usa en todos los roles, como pide el informe."""
    llm = llm or init_chat_model(modelo, temperature=0)
    especialistas = crear_especialistas(llm=llm)
    # Especialista MCP: solo las tools del servidor local campanas-q2, con los mismos guardarraíles
    especialistas["datos_q2"] = create_agent(llm, tools=herramientas_mcp(), system_prompt=PROMPT_DATOS_Q2,
                                             middleware=[limpiar_argumentos], name="datos_q2")

    def _llm_rol(nombre: str):
        return llm.with_config(metadata={"agente": nombre}, tags=[f"agente:{nombre}"], run_name=nombre)

    # ---------------------------------------------------------------- supervisor (router)
    def supervisor(estado: Estado, config):
        turno = estado.get("turno", 0) + 1
        pregunta = _pregunta(estado)
        ruta, razon, usos = None, "", []
        if router_modo == "llm":
            try:
                historial = [m for m in estado["messages"][-5:]]
                salida = (llm.with_structured_output(Enrutamiento, include_raw=True)
                          .with_config(metadata={"agente": "supervisor"}, tags=["agente:supervisor"], run_name="supervisor")
                          .invoke([SystemMessage(PROMPT_SUPERVISOR)] + historial, config))
                usos = _uso("supervisor", [salida.get("raw")], turno)
                if salida.get("parsed") is not None:
                    ruta, razon = salida["parsed"].ruta, salida["parsed"].razon
            except Exception as e:   # el modelo no devolvió JSON válido u otro fallo
                razon = f"respaldo por reglas ({type(e).__name__})"
        if ruta is None:
            ruta = _ruta_por_reglas(pregunta)
            razon = razon or "clasificado por reglas"
        ruta_final = _ajustar_por_trimestre(ruta, pregunta)
        if ruta_final != ruta:
            razon = f"{razon} | trimestre {_trimestre(pregunta).upper()}: {ruta} -> {ruta_final}"
            ruta = ruta_final
        return {"turno": turno, "ruta": ruta, "razon_ruta": razon, "usos": usos,
                "rondas": 0, "feedback": "", "respuesta": "", "veredicto": "", "herramientas": [], "figuras": []}

    def despachar(estado: Estado):
        ruta = estado["ruta"]
        if ruta == "mixta":
            return ["datos", "conocimiento"]
        if ruta == "mixta_q2":
            return ["datos_q2", "conocimiento"]
        if ruta == "trimestres":
            return ["datos", "datos_q2"]   # paralelización: Q3 nativo + Q2 por MCP
        return ruta

    # ---------------------------------------------------------------- especialistas
    def _nodo_especialista(nombre: str):
        agente = especialistas[nombre]

        def nodo(estado: Estado, config):
            turno = estado["turno"]
            # conversación previa (humano + respuestas finales) para entender seguimientos
            entrada = list(estado["messages"])
            ruta = estado["ruta"]
            if ruta in ("mixta", "mixta_q2") and nombre in TAREA_MIXTA:
                entrada = entrada + [HumanMessage(TAREA_MIXTA[nombre])]
            elif nombre in TAREAS_Q2.get(ruta, {}):
                entrada = entrada + [HumanMessage(TAREAS_Q2[ruta][nombre])]
            out = agente.invoke({"messages": entrada}, config)
            nuevos = out["messages"][len(entrada):]
            herramientas = [m for m in nuevos if m.type == "tool"]
            # un reintento si respondió de memoria o si TODAS las llamadas fallaron (error de validación)
            if not herramientas or all(_es_error(m) for m in herramientas):
                aviso = ("La herramienta devolvió un error; corrige los argumentos (un solo valor por filtro, "
                         "o no envíes el filtro) y vuelve a llamarla." if herramientas
                         else "Usa tus herramientas para responder; no respondas de memoria.")
                entrada = entrada + nuevos + [HumanMessage(aviso)]
                out = agente.invoke({"messages": entrada}, config)
                nuevos = nuevos + out["messages"][len(entrada):]
                herramientas = [m for m in nuevos if m.type == "tool"]
            herramientas_ok = [m for m in herramientas if not _es_error(m)]
            final = next((m.content for m in reversed(nuevos) if m.type == "ai" and m.content), "")
            figuras = [m.artifact for m in herramientas_ok if m.name == "generar_grafico" and getattr(m, "artifact", None)]
            # solo las llamadas exitosas cuentan como datos (las fallidas quedan en la traza de LangSmith)
            resultado = {"turno": turno, "agente": nombre, "respuesta": final,
                         "herramientas": [m.name for m in herramientas_ok],
                         "salidas": [str(m.content) for m in herramientas_ok], "figuras": figuras,
                         "errores": len(herramientas) - len(herramientas_ok)}
            return {"resultados": [resultado], "usos": _uso(nombre, nuevos, turno)}
        return nodo

    # ---------------------------------------------------------------- integrar (fan-in)
    def integrar(estado: Estado):
        res = _del_turno(estado, "resultados")
        if estado["ruta"] in RUTAS_MULTIPLES or not res:
            return {}
        return {"respuesta": res[0]["respuesta"]}

    def tras_integrar(estado: Estado):
        return "redactor" if estado["ruta"] in RUTAS_MULTIPLES or not estado.get("respuesta") else "critico"

    # ---------------------------------------------------------------- redactor
    def redactor(estado: Estado, config):
        res = _del_turno(estado, "resultados")
        partes = [f"PREGUNTA: {_pregunta(estado)}", f"RESULTADOS DE LAS HERRAMIENTAS:\n{_salidas_texto(res)}"]
        if estado.get("respuesta"):
            partes.append(f"BORRADOR ANTERIOR:\n{estado['respuesta']}")
        if estado.get("feedback"):
            partes.append(f"CORRIGE ESTO (crítico): {estado['feedback']}")
        msg = _llm_rol("redactor").invoke([SystemMessage(PROMPT_REDACTOR), HumanMessage("\n\n".join(partes))], config)
        return {"respuesta": msg.content, "usos": _uso("redactor", [msg], estado["turno"])}

    # ---------------------------------------------------------------- crítico (evaluador)
    def critico(estado: Estado, config):
        res = _del_turno(estado, "resultados")
        fuentes = _salidas_texto(res)
        pregunta, respuesta = _pregunta(estado), estado.get("respuesta", "")
        rondas = estado.get("rondas", 0) + 1
        usos = []

        # 1) determinista, 0 tokens: toda cifra de la respuesta debe estar en las salidas
        sin = cifras_sin_respaldo(respuesta, fuentes, pregunta)
        if not any(r["herramientas"] for r in res):
            return {"rondas": rondas, "veredicto": "sin_datos",
                    "feedback": "Ningún especialista obtuvo datos de una herramienta."}
        if sin:
            return {"rondas": rondas, "veredicto": "rechazado",
                    "feedback": f"Estas cifras no aparecen en los resultados: {', '.join(f'{v:g}' for v in sin)}. "
                                "Usa solo cifras copiadas de los resultados."}
        # la herramienta encontró anomalías pero la respuesta dice que no hay
        salidas_anom = [s for r in res for h, s in zip(r["herramientas"], r["salidas"])
                        if h in ("detectar_anomalias", "detectar_anomalias_q2")]
        # nativa: "tipo: ..." | MCP (JSON): '"tipo": "..."'
        hay_hallazgos = any(re.search(r'tipo"?\s*:', s) for s in salidas_anom)
        if hay_hallazgos and _NIEGA.search(respuesta):
            return {"rondas": rondas, "veredicto": "rechazado",
                    "feedback": "La herramienta SÍ encontró anomalías y la respuesta dice que no hay. "
                                "Reporta cada hallazgo con fecha, país, plataforma, tipo y detalle."}

        # 2) LLM: ¿responde la pregunta y concluye bien?
        if critico_llm == "siempre" or (critico_llm == "mixta" and estado["ruta"] in RUTAS_MULTIPLES):
            try:
                salida = (llm.with_structured_output(Veredicto, include_raw=True)
                          .with_config(metadata={"agente": "critico"}, tags=["agente:critico"], run_name="critico")
                          .invoke([SystemMessage(PROMPT_CRITICO),
                                   HumanMessage(f"PREGUNTA: {pregunta}\n\nRESULTADOS:\n{fuentes}\n\nRESPUESTA:\n{respuesta}")],
                                  config))
                usos = _uso("critico", [salida.get("raw")], estado["turno"])
                v = salida.get("parsed")
                if v is not None and not v.aprobado:
                    return {"rondas": rondas, "veredicto": "rechazado", "usos": usos,
                            "feedback": v.problema or "La respuesta no responde bien la pregunta."}
            except Exception:
                pass   # si el crítico falla, no bloquea la entrega
        return {"rondas": rondas, "veredicto": "aprobado", "usos": usos}

    def tras_critico(estado: Estado):
        if estado["veredicto"] in ("aprobado", "sin_datos"):
            return "entregar"
        if estado["rondas"] >= max_rondas or _costo_turno(estado) > tope_usd:
            return "entregar"
        return "redactor"

    # ---------------------------------------------------------------- entregar
    def entregar(estado: Estado):
        res = _del_turno(estado, "resultados")
        veredicto = estado.get("veredicto", "")
        if veredicto == "rechazado":   # salió por rondas o tope
            veredicto = "tope" if _costo_turno(estado) > tope_usd else "rondas_agotadas"
        respuesta = estado.get("respuesta")
        if veredicto == "sin_datos" or not respuesta:
            respuesta = ("No pude obtener los datos: la herramienta falló o no se llamó. "
                         "Intenta reformular la pregunta (por ejemplo, indicando el mes o el país).")
        return {"messages": [AIMessage(respuesta)],
                "herramientas": [h for r in res for h in r["herramientas"]],
                "figuras": [f for r in res for f in r["figuras"]],
                "veredicto": veredicto}

    # ---------------------------------------------------------------- armado
    g = StateGraph(Estado)
    g.add_node("supervisor", supervisor)
    for nombre in especialistas:
        g.add_node(nombre, _nodo_especialista(nombre))
    g.add_node("integrar", integrar)
    g.add_node("redactor", redactor)
    g.add_node("critico", critico)
    g.add_node("entregar", entregar)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", despachar, list(especialistas))
    for nombre in especialistas:
        g.add_edge(nombre, "integrar")
    g.add_conditional_edges("integrar", tras_integrar, ["redactor", "critico"])
    g.add_edge("redactor", "critico")
    g.add_conditional_edges("critico", tras_critico, ["redactor", "entregar"])
    g.add_edge("entregar", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(), name="multiagente")


# --------------------------------------------------------------------------- uso
def responder(grafo, texto: str, thread_id: str = "demo", callbacks: Optional[list] = None) -> dict:
    """Respuesta + metadatos del turno (ruta, herramientas, figuras, veredicto, rondas, costo sombra)."""
    config = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    e = grafo.invoke({"messages": [HumanMessage(texto)]}, config)
    return {"respuesta": e["messages"][-1].content, "ruta": e.get("ruta"), "razon_ruta": e.get("razon_ruta"),
            "herramientas": e.get("herramientas", []), "figuras": e.get("figuras", []),
            "veredicto": e.get("veredicto"), "rondas": e.get("rondas", 0),
            "usd_sombra_turno": round(_costo_turno(e), 6)}


def preguntar(grafo, texto: str, thread_id: str = "demo") -> str:
    """Misma firma que src.agent.preguntar, para comparar ambos sistemas con el mismo código."""
    return responder(grafo, texto, thread_id)["respuesta"]
