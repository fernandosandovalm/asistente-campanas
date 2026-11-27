"""
Evaluación en LangSmith: sube el dataset y corre experimentos (modelo A vs B, prompt v1 vs v2).
Uso:  python -m src.evaluate
[FASE 3: pendiente de probar]
"""
import json
import re
from pydantic import BaseModel
from langsmith import Client
from langchain.chat_models import init_chat_model

from src.config import EVAL_PATH, LLM_PRINCIPAL, LLM_COMPARACION

JUEZ = LLM_PRINCIPAL  # cambiar a LLM_COMPARACION cuando tengas qwen2.5:7b (mejor juez)
from src.agent import crear_agente

DATASET = "ben-frank-asistente-eval"
client = Client()


def subir_dataset():
    if client.has_dataset(dataset_name=DATASET):
        return
    ds = client.create_dataset(DATASET, description="Preguntas con respuesta calculada desde DuckDB")
    client.create_examples(dataset_id=ds.id, examples=json.loads(EVAL_PATH.read_text()))


def target_factory(modelo):
    agente = crear_agente(modelo)

    def target(inputs: dict) -> dict:
        out = agente.invoke({"messages": [{"role": "user", "content": inputs["pregunta"]}]},
                            {"configurable": {"thread_id": inputs["pregunta"][:40]}})
        usadas = [tc["name"] for m in out["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
        return {"respuesta": out["messages"][-1].content, "herramientas": usadas}
    return target


# ---- Evaluadores
MESES = {"01": "enero", "02": "febrero", "03": "marzo", "04": "abril", "05": "mayo", "06": "junio",
         "07": "julio", "08": "agosto", "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre"}


def _variantes_texto(v: str) -> list[str]:
    """'2026-08' -> agosto; '2026-08-14' -> 14 de agosto, 14/08, agosto 14..."""
    v = str(v).lower()
    m = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", v)
    if not m:
        return [v]
    _, mes, dia = m.groups()
    nombre = MESES[mes]
    if dia is None:
        return [v, nombre]
    d = str(int(dia))
    return [v, f"{d} de {nombre}", f"{nombre} {d}", f"{dia}/{mes}", f"{d}/{int(mes)}", f"{mes}-{dia}"]


def valor_correcto(outputs, reference_outputs):
    """Numéricas: el número de referencia aparece en la respuesta (tolerancia 1 %).
    Texto: aparece el valor o una variante (fechas y meses en español)."""
    v = reference_outputs.get("valor")
    if v is None:
        return {"key": "valor_correcto", "score": None}
    respuesta = outputs["respuesta"]
    if isinstance(v, (int, float)):
        txt = respuesta.replace(",", "")
        nums = [float(n) for n in re.findall(r"\d+\.?\d*", txt)]
        return {"key": "valor_correcto", "score": any(abs(n - v) <= abs(v) * 0.01 for n in nums)}
    txt = respuesta.lower()
    return {"key": "valor_correcto", "score": any(var in txt for var in _variantes_texto(v))}


def herramienta_correcta(outputs, reference_outputs):
    esperadas = reference_outputs["herramienta_esperada"].split("+")
    return {"key": "herramienta_correcta", "score": any(h in outputs["herramientas"] for h in esperadas)}


class Veredicto(BaseModel):
    correcto: bool
    razon: str


juez = init_chat_model(JUEZ, temperature=0).with_structured_output(Veredicto)


def llm_juez(inputs, outputs, reference_outputs):
    v = juez.invoke(f"Pregunta: {inputs['pregunta']}\nReferencia: {reference_outputs['respuesta']}\n"
                    f"Respuesta del asistente: {outputs['respuesta']}\n¿La respuesta es correcta y fiel a la referencia?")
    return {"key": "llm_juez", "score": v.correcto, "comment": v.razon}


if __name__ == "__main__":
    subir_dataset()
    for modelo in [LLM_PRINCIPAL]:  # agrega LLM_COMPARACION tras `ollama pull qwen2.5:7b`
        client.evaluate(target_factory(modelo), data=DATASET,
                        evaluators=[valor_correcto, herramienta_correcta, llm_juez],
                        experiment_prefix=modelo.split(":")[-1], max_concurrency=1)

