"""
Cliente MCP: conecta el servidor "campanas-q2" (mcp_server/server.py) y entrega sus
herramientas como tools normales de LangChain para el agente.

- El servidor corre LOCAL como subproceso, por stdio (sin red ni puertos).
- Se lanza con `uv --directory mcp_server run server.py`, así usa su propio entorno.
- langchain-mcp-adapters solo da tools async; el agente y Streamlit usan .invoke()
  síncrono, por eso cada tool se envuelve con una versión síncrona que corre la
  llamada en un event loop de fondo.
"""
import asyncio
import shutil
import sys
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

RAIZ = Path(__file__).resolve().parents[1]
MCP_DIR = RAIZ / "mcp_server"

# Si uv no está en el PATH (raro), se usa el Python actual; en ese caso el entorno
# actual debe tener mcp<2, duckdb, pandas y numpy instalados.
if shutil.which("uv"):
    _CMD, _ARGS = "uv", ["--directory", str(MCP_DIR), "run", "server.py"]
else:
    _CMD, _ARGS = sys.executable, [str(MCP_DIR / "server.py")]

SERVIDORES = {
    "campanas_q2": {"command": _CMD, "args": _ARGS, "transport": "stdio"},
}

# ---------------------------------------------------------------- event loop de fondo
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="mcp-loop").start()


def _correr(coro, timeout: float = 120):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout)


def _a_texto(contenido) -> str:
    """El adaptador devuelve bloques [{'type': 'text', 'text': ...}]; se dejan como texto plano
    para que el crítico, el redactor y las trazas lean las cifras igual que con las tools nativas."""
    if isinstance(contenido, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido)
    return str(contenido)


def _version_sync(tool_async) -> StructuredTool:
    """Misma tool (nombre, descripción, esquema) pero invocable con .invoke()."""
    def _func(**kwargs):
        # se llama la corrutina interna: devuelve (contenido, artefacto) como espera LangChain
        contenido, artefacto = _correr(tool_async.coroutine(**kwargs))
        return _a_texto(contenido), artefacto

    return StructuredTool(
        name=tool_async.name,
        description=tool_async.description,
        args_schema=tool_async.args_schema,
        func=_func,
        coroutine=tool_async.coroutine,
        response_format=getattr(tool_async, "response_format", "content"),
    )


@lru_cache(maxsize=1)
def herramientas_mcp() -> list:
    """Lista las tools del servidor MCP (una sola vez por sesión)."""
    cliente = MultiServerMCPClient(SERVIDORES)
    tools = _correr(cliente.get_tools())
    return [_version_sync(t) for t in tools]


if __name__ == "__main__":
    # Prueba rápida sin LLM:  python -m src.mcp_cliente
    tools = herramientas_mcp()
    print("Tools MCP:", [t.name for t in tools])
    print(next(t for t in tools if t.name == "resumen_q2").invoke({}))
