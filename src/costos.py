"""
Medición de costo por llamada, por agente y por tarea.

Fórmula (diapositiva 8):  costo_llamada = tokens_entrada × precio_entrada + tokens_salida × precio_salida
                          costo_tarea   = suma de todas las llamadas de todos los agentes
(precios en USD por millón de tokens)

Se reportan tres costos:
  1. usd_real    -> lo que cobra OpenRouter (solo el juez). Ollama = 0.
  2. usd_sombra  -> lo que costarían los tokens locales con un modelo de nube de referencia.
  3. energía     -> kWh y COP de la GPU mientras Ollama genera.

Uso:
    from src.costos import ContadorCostos
    contador = ContadorCostos()
    agente.invoke(entrada, config={"callbacks": [contador], "configurable": {...}})
    contador.tabla()        # DataFrame por agente, como la diapositiva 8
    contador.resumen()      # dict con totales (para app.py y LangSmith)

El nombre del agente se toma, en este orden, de: metadata["agente"], un tag "agente:<nombre>",
o el valor por defecto del contador ("agente_unico").
"""
import threading
import time
from dataclasses import dataclass, asdict

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler

from src.config import (MODELO_REFERENCIA_SOMBRA, POTENCIA_GPU_W, PRECIOS_USD_POR_MILLON,
                        TARIFA_KWH_COP)


class TopeExcedido(Exception):
    """El costo acumulado de la tarea superó el tope permitido."""


def costo_usd(tokens_entrada: int, tokens_salida: int, modelo: str) -> float:
    """USD de una llamada con los precios de PRECIOS_USD_POR_MILLON."""
    p_in, p_out = PRECIOS_USD_POR_MILLON[modelo]
    return (tokens_entrada * p_in + tokens_salida * p_out) / 1_000_000


def energia(segundos: float) -> tuple[float, float]:
    """(kWh, COP) de la GPU local durante `segundos` de generación."""
    kwh = POTENCIA_GPU_W * segundos / 3_600_000
    return kwh, kwh * TARIFA_KWH_COP


@dataclass
class Llamada:
    agente: str
    modelo: str
    local: bool
    entrada: int
    salida: int
    segundos: float
    usd_real: float


def _es_local(modelo: str) -> bool:
    m = modelo.lower()
    return "ollama" in m or "qwen" in m or "llama" in m


class ContadorCostos(BaseCallbackHandler):
    """Callback de LangChain que registra tokens, tiempo y costo de cada llamada al LLM."""

    def __init__(self, agente_por_defecto: str = "agente_unico"):
        self.agente_por_defecto = agente_por_defecto
        self.llamadas: list[Llamada] = []
        self._inicio: dict = {}      # run_id -> (agente, modelo, t0)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ callbacks
    def _agente(self, tags, metadata) -> str:
        if metadata and metadata.get("agente"):
            return metadata["agente"]
        for t in tags or []:
            if isinstance(t, str) and t.startswith("agente:"):
                return t.split(":", 1)[1]
        return self.agente_por_defecto

    def _registrar_inicio(self, serialized, run_id, tags, metadata, kwargs):
        params = kwargs.get("invocation_params") or {}
        modelo = (params.get("model") or params.get("model_name")
                  or (metadata or {}).get("ls_model_name") or (serialized or {}).get("name", "desconocido"))
        if (metadata or {}).get("ls_provider") == "ollama" and "ollama" not in str(modelo):
            modelo = f"ollama:{modelo}"
        with self._lock:
            self._inicio[run_id] = (self._agente(tags, metadata), str(modelo), time.perf_counter())

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs):
        self._registrar_inicio(serialized, run_id, tags, metadata, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, tags=None, metadata=None, **kwargs):
        self._registrar_inicio(serialized, run_id, tags, metadata, kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            agente, modelo, t0 = self._inicio.pop(run_id, (self.agente_por_defecto, "desconocido", time.perf_counter()))
        seg = time.perf_counter() - t0

        entrada = salida = 0
        usd_real = 0.0
        try:
            msg = response.generations[0][0].message
            uso = getattr(msg, "usage_metadata", None) or {}
            entrada, salida = uso.get("input_tokens", 0), uso.get("output_tokens", 0)
            meta = getattr(msg, "response_metadata", {}) or {}
            modelo = meta.get("model_name") or meta.get("model") or modelo
            # OpenRouter devuelve el costo cobrado en usage.cost
            usd_real = float((meta.get("token_usage") or {}).get("cost") or 0.0)
        except (IndexError, AttributeError):
            pass
        if not entrada and response.llm_output:   # respaldo para modelos que lo reportan aquí
            tu = response.llm_output.get("token_usage", {}) or {}
            entrada, salida = tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)
            usd_real = usd_real or float(tu.get("cost") or 0.0)

        local = _es_local(modelo)
        if not local and not usd_real:   # OpenRouter no devolvió el costo: se calcula con la tabla de precios
            clave = next((k for k in PRECIOS_USD_POR_MILLON if k in str(modelo).lower()), None)
            if clave:
                usd_real = costo_usd(entrada, salida, clave)
        with self._lock:
            self.llamadas.append(Llamada(agente, modelo, local, entrada, salida, seg, 0.0 if local else usd_real))

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            self._inicio.pop(run_id, None)

    # ------------------------------------------------------------------ reportes
    def reiniciar(self):
        with self._lock:
            self.llamadas.clear()
            self._inicio.clear()

    def df(self) -> pd.DataFrame:
        """Una fila por llamada."""
        return pd.DataFrame([asdict(l) for l in self.llamadas],
                            columns=["agente", "modelo", "local", "entrada", "salida", "segundos", "usd_real"])

    def usd_sombra(self, modelo_ref: str = MODELO_REFERENCIA_SOMBRA) -> float:
        """Costo de los tokens LOCALES si se hubieran pagado con `modelo_ref`."""
        return sum(costo_usd(l.entrada, l.salida, modelo_ref) for l in self.llamadas if l.local)

    def verificar_tope(self, tope_usd: float, modelo_ref: str = MODELO_REFERENCIA_SOMBRA):
        """Lanza TopeExcedido si (real + sombra) supera el tope. Úsalo en el grafo antes de cada ronda."""
        total = self.usd_sombra(modelo_ref) + sum(l.usd_real for l in self.llamadas)
        if total > tope_usd:
            raise TopeExcedido(f"Costo acumulado USD {total:.5f} > tope {tope_usd:.5f}")

    def tabla(self, modelos_ref: list[str] | None = None) -> pd.DataFrame:
        """Tabla por agente (diapositiva 8): llamadas, tokens, segundos, USD real, USD sombra por modelo y energía."""
        modelos_ref = modelos_ref or list(PRECIOS_USD_POR_MILLON)
        d = self.df()
        if d.empty:
            return d
        for m in modelos_ref:
            d[f"usd_{m}"] = [costo_usd(e, s, m) if loc else 0.0
                             for e, s, loc in zip(d.entrada, d.salida, d.local)]
        d["kwh"] = [energia(s)[0] if loc else 0.0 for s, loc in zip(d.segundos, d.local)]
        d["cop_energia"] = d.kwh * TARIFA_KWH_COP
        num = ["entrada", "salida", "segundos", "usd_real", "kwh", "cop_energia"] + [f"usd_{m}" for m in modelos_ref]
        t = d.groupby("agente").agg(llamadas=("agente", "size"), **{c: (c, "sum") for c in num})
        t.loc["TOTAL"] = t.sum(numeric_only=True)
        t[["llamadas", "entrada", "salida"]] = t[["llamadas", "entrada", "salida"]].astype(int)
        return t.reset_index()

    def resumen(self, modelo_ref: str = MODELO_REFERENCIA_SOMBRA) -> dict:
        """Totales de la tarea en un dict plano (para la interfaz y para feedback en LangSmith)."""
        seg_local = sum(l.segundos for l in self.llamadas if l.local)
        kwh, cop = energia(seg_local)
        return {
            "llamadas": len(self.llamadas),
            "tokens_entrada": sum(l.entrada for l in self.llamadas),
            "tokens_salida": sum(l.salida for l in self.llamadas),
            "segundos_llm": round(sum(l.segundos for l in self.llamadas), 3),
            "usd_real": round(sum(l.usd_real for l in self.llamadas), 6),
            "usd_sombra": round(self.usd_sombra(modelo_ref), 6),
            "modelo_sombra": modelo_ref,
            "kwh": round(kwh, 6),
            "cop_energia": round(cop, 4),
            "agentes": sorted({l.agente for l in self.llamadas}),
        }
