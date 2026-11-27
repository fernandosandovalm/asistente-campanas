"""
Paso 6: evaluación comparativa agente único (v5) vs. multiagente en LangSmith.

Reutiliza de src/evaluate.py el dataset, el cliente y `valor_correcto`; agrega:
  - juez_externo: juez de OTRA familia que Qwen (Haiku 5.5 vía OpenRouter, GPT-6 Luna vía OpenAI o
    llama3.1:8b local si no hay llaves), en lugar del Qwen que se calificaba a sí mismo (ver informe).
  - herramienta_correcta_v2: exige TODAS las herramientas en las mixtas ("a+b") y acepta ranking
    donde se esperaba consultar_metricas (ambas resuelven bien esas preguntas).
  - métricas de costo por pregunta: usd_sombra, tokens, llamadas al LLM (ContadorCostos).
  - repeticiones por experimento (num_repetitions) para medir la variabilidad entre corridas.
  - resumen_comparativo: tabla con calidad, latencia p50/p95 y costo por configuración.

Uso en el notebook:
    from src.evaluacion_multi import correr, resumen_comparativo, target_agente_unico, target_multiagente
    r1 = correr("v5-agente-unico", target_agente_unico())
    r2 = correr("multiagente-v1",  target_multiagente())
    resumen_comparativo({"v5": r1, "multiagente": r2})
"""
import os
import uuid

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import LLM_JUEZ, LLM_JUEZ_LOCAL, LLM_JUEZ_OPENAI, LLM_PRINCIPAL, OPENROUTER_BASE_URL
from src.costos import ContadorCostos
from src.evaluate import DATASET, client, valor_correcto  # noqa: F401  (se reutilizan tal cual)

# --------------------------------------------------------------------------- juez externo
CONTADOR_JUEZ = ContadorCostos(agente_por_defecto="juez")   # costo REAL acumulado de evaluar
_juez = None


class Juicio(BaseModel):
    correcto: bool = Field(description="True si la respuesta es correcta y fiel a la referencia.")
    razon: str = Field(description="Una frase con el motivo.")


PROMPT_JUEZ = """Eres un evaluador estricto de un asistente de analítica de campañas publicitarias.
Compara la RESPUESTA del asistente con la RESPUESTA DE REFERENCIA (calculada directamente de la base de datos).
Es correcta si contiene la cifra o entidad clave de la referencia (se aceptan redondeos de ±1 % y formatos
distintos, como "14 de agosto" por "2026-08-14") y no la contradice. En preguntas conceptuales, es correcta
si cubre la idea central de la referencia. No penalices el estilo, la longitud ni recomendaciones adicionales
razonables. Penaliza cifras que contradigan la referencia o conclusiones al revés."""


def _llm_juez():
    """Juez de otra familia que Qwen: OpenRouter (Haiku 5.5) > OpenAI (GPT-6 Luna) > local (llama3.1:8b)."""
    global _juez
    if _juez is None:
        from langchain_openai import ChatOpenAI
        if os.environ.get("OPENROUTER_API_KEY"):
            _juez = ChatOpenAI(model=LLM_JUEZ, base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"],
                               temperature=0, max_tokens=200, extra_body={"usage": {"include": True}})
        elif os.environ.get("OPENAI_API_KEY"):
            # modelo de razonamiento: sin temperature; razonamiento apagado para un juicio corto y barato
            _juez = ChatOpenAI(model=LLM_JUEZ_OPENAI, api_key=os.environ["OPENAI_API_KEY"],
                               reasoning_effort="none", max_tokens=200)
        else:
            # sin llaves: juez local de otra familia (Llama) -> USD 0, todo sigue en la máquina
            from langchain.chat_models import init_chat_model
            _juez = init_chat_model(LLM_JUEZ_LOCAL, temperature=0)
    return _juez


def juez_activo() -> str:
    """Qué modelo está usando el juez (para reportarlo en el informe)."""
    j = _llm_juez()
    return getattr(j, "model_name", None) or getattr(j, "model", "desconocido")


def juez_externo(inputs, outputs, reference_outputs):
    """LLM como juez con un modelo de otra familia (reduce el sesgo de autoevaluación)."""
    msgs = [SystemMessage(PROMPT_JUEZ),
            HumanMessage(f"PREGUNTA: {inputs['pregunta']}\n\n"
                         f"RESPUESTA DE REFERENCIA: {reference_outputs.get('respuesta')}\n\n"
                         f"RESPUESTA DEL ASISTENTE: {outputs.get('respuesta')}")]
    cfg = {"callbacks": [CONTADOR_JUEZ], "metadata": {"agente": "juez"}}
    try:
        j = _llm_juez().with_structured_output(Juicio).invoke(msgs, cfg)
        return {"key": "juez_externo", "score": bool(j.correcto), "comment": j.razon}
    except Exception as e:   # respaldo: texto libre
        txt = _llm_juez().invoke(msgs + [HumanMessage("Responde solo CORRECTO o INCORRECTO.")], cfg).content.upper()
        return {"key": "juez_externo", "score": "INCORRECTO" not in txt and "CORRECTO" in txt,
                "comment": f"respaldo texto ({type(e).__name__})"}


# --------------------------------------------------------------------------- herramienta correcta v2
EQUIVALENTES = {"consultar_metricas": {"consultar_metricas", "ranking"}}


def herramienta_correcta_v2(outputs, reference_outputs):
    """Todas las herramientas requeridas ("a+b" = las dos); ranking vale por consultar_metricas."""
    esperada = reference_outputs.get("herramienta_esperada") or ""
    usadas = set(outputs.get("herramientas") or [])
    requeridas = [h.strip() for h in esperada.split("+") if h.strip()]
    ok = all(usadas & EQUIVALENTES.get(h, {h}) for h in requeridas)
    return {"key": "herramienta_correcta_v2", "score": ok,
            "comment": f"esperadas={requeridas} usadas={sorted(usadas)}"}


# --------------------------------------------------------------------------- métricas de costo
def costo_usd_sombra(outputs):
    return {"key": "costo_usd_sombra", "score": outputs.get("usd_sombra", 0.0)}


def tokens_totales(outputs):
    return {"key": "tokens_totales", "score": outputs.get("tokens_entrada", 0) + outputs.get("tokens_salida", 0)}


def llamadas_llm(outputs):
    return {"key": "llamadas_llm", "score": outputs.get("llamadas", 0)}


EVALUADORES = [valor_correcto, herramienta_correcta_v2, juez_externo, costo_usd_sombra, tokens_totales, llamadas_llm]


def _con_costos(salida: dict, contador: ContadorCostos) -> dict:
    r = contador.resumen()
    salida.update({k: r[k] for k in ("llamadas", "tokens_entrada", "tokens_salida", "usd_sombra", "segundos_llm")})
    salida["agentes"] = r["agentes"]
    return salida


# --------------------------------------------------------------------------- targets
def target_agente_unico(modelo: str = LLM_PRINCIPAL):
    """Agente único v5 con medición de costo. Hilo nuevo por pregunta (sin memoria entre ejemplos)."""
    from src.agent import crear_agente
    agente = crear_agente(modelo)

    def target(inputs: dict) -> dict:
        contador = ContadorCostos(agente_por_defecto="agente_unico")
        out = agente.invoke({"messages": [{"role": "user", "content": inputs["pregunta"]}]},
                            {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}, "callbacks": [contador]})
        msgs = out["messages"]
        inicio = max(i for i, m in enumerate(msgs) if m.type == "human")
        herramientas = [m.name for m in msgs[inicio:] if m.type == "tool"]
        return _con_costos({"respuesta": msgs[-1].content, "herramientas": herramientas}, contador)
    return target


def target_multiagente(modelo: str = LLM_PRINCIPAL, **opciones):
    """Multiagente con medición de costo. `opciones` pasa a crear_multiagente
    (router_modo="reglas", critico_llm="nunca", max_rondas=..., tope_usd=...)."""
    from src.grafo import crear_multiagente, responder
    grafo = crear_multiagente(modelo, **opciones)

    def target(inputs: dict) -> dict:
        contador = ContadorCostos(agente_por_defecto="otro")
        r = responder(grafo, inputs["pregunta"], thread_id=f"eval-{uuid.uuid4()}", callbacks=[contador])
        salida = {"respuesta": r["respuesta"], "herramientas": r["herramientas"],
                  "ruta": r["ruta"], "veredicto": r["veredicto"], "rondas": r["rondas"]}
        return _con_costos(salida, contador)
    return target


# --------------------------------------------------------------------------- correr y resumir
def correr(nombre: str, target, repeticiones: int = 2, evaluadores=None):
    """Un experimento en LangSmith. Cada pregunta se corre `repeticiones` veces."""
    return client.evaluate(target, data=DATASET, evaluators=evaluadores or EVALUADORES,
                           experiment_prefix=nombre, num_repetitions=repeticiones, max_concurrency=1)


def _tasa(serie) -> str:
    s = serie.dropna()
    s = s[s.astype(str).isin(["True", "False", "1", "0", "1.0", "0.0"])].astype(float) if len(s) else s
    return f"{s.mean():.0%} ({int(s.sum())}/{len(s)})" if len(s) else "-"


def resumen_comparativo(resultados: dict) -> pd.DataFrame:
    """{nombre: resultado de correr()} -> tabla comparativa (diapositiva 7: calidad, costo, latencia, robustez)."""
    filas = []
    for nombre, res in resultados.items():
        df = res.to_pandas() if hasattr(res, "to_pandas") else res
        fb = lambda k: df.get(f"feedback.{k}", pd.Series(dtype=float))
        t = df["execution_time"].astype(float)
        filas.append({
            "configuracion": nombre,
            "corridas": len(df),
            "valor_correcto": _tasa(fb("valor_correcto")),
            "herramienta_v2": _tasa(fb("herramienta_correcta_v2")),
            "juez_externo": _tasa(fb("juez_externo")),
            "errores": int(df["error"].notna().sum()) if "error" in df else 0,
            "latencia_p50_s": round(float(np.percentile(t, 50)), 2),
            "latencia_p95_s": round(float(np.percentile(t, 95)), 2),
            "llamadas_llm_media": round(float(fb("llamadas_llm").astype(float).mean()), 2),
            "tokens_media": int(fb("tokens_totales").astype(float).mean()),
            "usd_sombra_por_pregunta": round(float(fb("costo_usd_sombra").astype(float).mean()), 6),
        })
    return pd.DataFrame(filas).set_index("configuracion")


def variabilidad(res, metrica: str = "valor_correcto") -> pd.DataFrame:
    """Preguntas cuyo resultado cambió entre repeticiones (misma pregunta, distinta calificación)."""
    df = res.to_pandas() if hasattr(res, "to_pandas") else res
    col = f"feedback.{metrica}"
    g = df.groupby("inputs.pregunta")[col].agg(lambda s: sorted({str(x) for x in s}))
    return g[g.map(len) > 1].rename("resultados_distintos").to_frame()


def costo_juez() -> dict:
    """Costo de todas las llamadas al juez en esta sesión (USD real si es de pago; sombra si es local)."""
    return CONTADOR_JUEZ.resumen()
