"""
Interfaz Streamlit del asistente.  Uso (desde la raíz, con el .venv activo):  streamlit run app.py
Requiere: haber corrido antes ingest + rag (el notebook lo hace) y el .env con LANGSMITH_API_KEY en la raíz.

Permite elegir entre el agente único (v5) y el multiagente (supervisor + especialistas + crítico),
y muestra por cada respuesta: herramientas usadas, ruta de agentes y costo (tokens, USD sombra, energía).
"""
import json
import os
import uuid

import pandas as pd
import plotly.io as pio
import streamlit as st

from src import tools as T
from src.config import (DUCKDB_PATH, LANGSMITH_PROJECT, LLM_COMPARACION, LLM_PRINCIPAL,
                        MODELO_REFERENCIA_SOMBRA, PRECIOS_USD_POR_MILLON)
from src.costos import ContadorCostos
from src.guardarrailes import REGISTRO
from src.mcp_cliente import herramientas_mcp

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

st.set_page_config(page_title="Asistente de campañas · Ben & Frank", page_icon="👓", layout="wide")

EJEMPLOS = [
    "¿Qué plataforma tuvo el menor costo por cita en US durante agosto?",
    "¿Detectas problemas de tracking o gasto sin entrega?",
    "¿Cuáles son los 3 creativos de TikTok con peor VTR?",
    "¿El lower funnel cumplió el objetivo de costo por cita del brief?",
    "Grafica la evolución semanal del costo por cita por plataforma en US",
    "¿Qué plataforma tuvo el menor costo por cita en Q2?",
    "¿Cómo cambió el costo por cita de TikTok entre Q2 y Q3?",
]
UNICO, MULTI = "Agente único (v6)", "Multiagente"
NOMBRES_AGENTE = {"supervisor": "🧭 Supervisor", "datos": "📊 Datos", "datos_q2": "🔌 Datos Q2 (MCP)",
                  "anomalias": "🚨 Anomalías",
                  "conocimiento": "📚 Conocimiento", "redactor": "✍️ Redactor", "critico": "🔎 Crítico",
                  "agente_unico": "🤖 Agente único"}


# ---------------------------------------------------------------- sistemas (se cargan una vez)
@st.cache_resource(show_spinner="Cargando agente único...")
def agente_unico(modelo):
    from src.agent import crear_agente
    return crear_agente(modelo)


@st.cache_resource(show_spinner="Cargando multiagente...")
def multiagente(modelo, router_modo, critico_llm):
    from src.grafo import crear_multiagente
    return crear_multiagente(modelo, router_modo=router_modo, critico_llm=critico_llm)


@st.cache_data(show_spinner="Consultando servidor MCP...")
def resumen_q2_mcp() -> dict:
    """Llama la tool resumen_q2 del servidor MCP local (stdio) y devuelve su JSON."""
    tool = next(t for t in herramientas_mcp() if t.name == "resumen_q2")
    return json.loads(tool.invoke({}))


def responder(arquitectura, modelo, opciones, texto, thread):
    """Respuesta, pasos (argumentos ejecutados), figuras, ruta y costo de este turno."""
    contador = ContadorCostos(agente_por_defecto="agente_unico" if arquitectura == UNICO else "otro")
    n_registro = len(REGISTRO)
    ruta = None

    if arquitectura == UNICO:
        from src.agent import figuras_del_turno
        out = agente_unico(modelo).invoke({"messages": [{"role": "user", "content": texto}]},
                                          {"configurable": {"thread_id": thread}, "callbacks": [contador]})
        msgs = out["messages"]
        respuesta, figuras = msgs[-1].content, figuras_del_turno(msgs)
    else:
        from src.grafo import responder as responder_multi
        r = responder_multi(multiagente(modelo, **opciones), texto, thread_id=thread, callbacks=[contador])
        respuesta, figuras = r["respuesta"], r["figuras"]
        ruta = {k: r[k] for k in ("ruta", "razon_ruta", "veredicto", "rondas")}

    # argumentos que realmente se ejecutaron (después de los guardarraíles)
    pasos = [{"herramienta": x["herramienta"], "argumentos": x["ejecutado"]} for x in REGISTRO[n_registro:]]
    return {"texto": respuesta, "pasos": pasos, "figuras": figuras, "ruta": ruta,
            "costo": contador.resumen(), "tabla_costo": contador.tabla([MODELO_REFERENCIA_SOMBRA]).to_dict("records")}


# ---------------------------------------------------------------- detalle bajo cada respuesta
def mostrar_detalle(item):
    for f in item.get("figuras", []):
        st.plotly_chart(pio.from_json(f), use_container_width=True)

    c = item.get("costo")
    ruta = item.get("ruta")
    if ruta:
        agentes = " → ".join(NOMBRES_AGENTE.get(a, a) for a in _orden_agentes(c.get("agentes", [])))
        icono = {"aprobado": "✅", "rondas_agotadas": "⚠️", "tope": "💸", "sin_datos": "❌"}.get(ruta["veredicto"], "•")
        st.caption(f"Ruta **{ruta['ruta']}** · {agentes} · crítico {icono} {ruta['veredicto']} "
                   f"({ruta['rondas']} ronda{'s' if ruta['rondas'] != 1 else ''})")
    if c:
        st.caption(f"💰 {c['llamadas']} llamadas al LLM · {c['tokens_entrada'] + c['tokens_salida']:,} tokens · "
                   f"{c['segundos_llm']:.1f} s · USD {c['usd_sombra']:.5f} sombra ({c['modelo_sombra']}) · "
                   f"{c['cop_energia']:.2f} COP de energía")

    if item.get("pasos"):
        with st.expander(f"🛠️ Herramientas usadas ({len(item['pasos'])})"):
            st.json(item["pasos"])
    if item.get("tabla_costo"):
        with st.expander("💰 Costo por agente"):
            st.dataframe(pd.DataFrame(item["tabla_costo"]).round(6), use_container_width=True, hide_index=True)
            if ruta:
                st.caption(f"Motivo del supervisor: {ruta.get('razon_ruta') or '-'}")


def _orden_agentes(agentes):
    orden = ["supervisor", "datos", "datos_q2", "anomalias", "conocimiento", "redactor", "critico", "agente_unico"]
    return sorted(agentes, key=lambda a: orden.index(a) if a in orden else 99)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    arquitectura = st.radio("Arquitectura", [UNICO, MULTI], index=1, horizontal=True)
    modelo = st.selectbox("Modelo local (Ollama)", [LLM_PRINCIPAL, LLM_COMPARACION])
    opciones = {}
    if arquitectura == MULTI:
        with st.expander("Opciones del multiagente"):
            opciones["router_modo"] = st.selectbox("Supervisor", ["llm", "reglas"],
                                                   help="llm: clasifica con el modelo. reglas: palabras clave, 0 tokens.")
            opciones["critico_llm"] = st.selectbox("Crítico con LLM", ["mixta", "siempre", "nunca"],
                                                   help="La verificación determinista de cifras corre siempre.")
        st.caption("Patrones: router · paralelización (mixtas y Q2 vs Q3) · evaluador–optimizador · MCP (Q2)")

    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.subheader("Resumen de campaña")
    trimestre = st.radio("Trimestre", ["Q3 · jul–sep", "Q2 · abr–jun"],
                         horizontal=True, label_visibility="collapsed")
    resumen = None
    if trimestre.startswith("Q3"):
        tot = T.consultar_metricas().iloc[0]
        resumen = (tot.media_cost, tot.subscriptions, tot.costo_por_cita, tot.ctr)
        fuente = f"🗄️ Fuente: DuckDB local (`{DUCKDB_PATH.name}`) · tools nativas del agente"
    else:
        try:
            r = resumen_q2_mcp()["totales"]
            resumen = (r["media_cost"], r["subscriptions"], r["costo_por_cita"], r["ctr"])
            fuente = "🔌 Fuente: servidor MCP local `campanas-q2` (stdio) · `campanas_q2.duckdb`"
        except Exception as e:
            st.warning(f"No se pudo consultar el servidor MCP: {type(e).__name__}: {e}")
    if resumen:
        inversion, citas, costo_cita, ctr = resumen
        c1, c2 = st.columns(2)
        c1.metric("Inversión", f"${inversion/1000:,.0f}K")
        c2.metric("Citas", f"{citas:,.0f}")
        c1.metric("Costo/cita", f"${costo_cita:,.2f}")
        c2.metric("CTR", f"{ctr:.2f}%")
        st.caption(fuente)

    st.divider()
    st.subheader("Costo de la sesión")
    costos = [i["costo"] for i in st.session_state.get("chat", []) if i.get("costo")]
    c1, c2 = st.columns(2)
    c1.metric("Respuestas", len(costos))
    c2.metric("Tokens", f"{sum(c['tokens_entrada'] + c['tokens_salida'] for c in costos):,}")
    c1.metric("USD sombra", f"{sum(c['usd_sombra'] for c in costos):.4f}")
    c2.metric("Energía", f"{sum(c['cop_energia'] for c in costos):.1f} COP")
    st.caption(f"USD sombra = lo que costarían los tokens con {MODELO_REFERENCIA_SOMBRA} "
               f"(USD {PRECIOS_USD_POR_MILLON[MODELO_REFERENCIA_SOMBRA][0]} / "
               f"{PRECIOS_USD_POR_MILLON[MODELO_REFERENCIA_SOMBRA][1]} por millón de tokens de entrada/salida).")

    st.divider()
    st.caption(f"🔍 Trazas en LangSmith → proyecto **{LANGSMITH_PROJECT}**")
    st.caption("🔒 LLM y datos corren 100 % local")

# ---------------------------------------------------------------- chat
st.title("👓 Asistente de campañas sociales — Ben & Frank")
st.caption("Facebook · TikTok · Snapchat · Pinterest · Reddit  |  US · CA · PR  |  "
           "Q2 abr–jun (MCP) · Q3 jul–sep 2026")

st.session_state.setdefault("thread", str(uuid.uuid4()))
st.session_state.setdefault("chat", [])
# memoria separada por arquitectura y configuración (cada sistema tiene su propio checkpointer)
thread = f"{st.session_state.thread}-{arquitectura}-{modelo}-{'-'.join(opciones.values())}"

if not st.session_state.chat:
    st.write("**Prueba con una pregunta de ejemplo:**")
    cols = st.columns(2)
    for i, ej in enumerate(EJEMPLOS):
        if cols[i % 2].button(ej, use_container_width=True, key=f"ej-{i}"):
            st.session_state.pendiente = ej
            st.rerun()

for item in st.session_state.chat:
    with st.chat_message(item["rol"]):
        st.markdown(item["texto"])
        if item["rol"] == "assistant":
            st.caption(f"_{item.get('arquitectura', '')}_")
            mostrar_detalle(item)

prompt = st.chat_input("Pregunta sobre las campañas...") or st.session_state.pop("pendiente", None)
if prompt:
    st.session_state.chat.append({"rol": "user", "texto": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        spinner = "Supervisor enrutando a los especialistas..." if arquitectura == MULTI else "Consultando datos y conocimiento..."
        with st.spinner(spinner):
            try:
                item = responder(arquitectura, modelo, opciones, prompt, thread)
            except Exception as e:
                item = {"texto": f"⚠️ Error: `{e}`\n\n¿Está Ollama corriendo y el modelo descargado?",
                        "pasos": [], "figuras": []}
        item.update({"rol": "assistant", "arquitectura": arquitectura})
        st.markdown(item["texto"])
        st.caption(f"_{arquitectura}_")
        mostrar_detalle(item)
    st.session_state.chat.append(item)
    st.rerun()   # refresca el costo de la sesión en la barra lateral