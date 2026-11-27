"""
RAG local: docs_rag/*.md -> chunks -> embeddings (Ollama) -> Chroma.
Uso:  python -m src.rag   (construye el índice; requiere `ollama pull bge-m3`)
[FASE 3: pendiente de probar con Ollama corriendo]
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.config import DOCS_RAG, CHROMA_DIR, EMBEDDINGS


def _embeddings():
    return OllamaEmbeddings(model=EMBEDDINGS)


def construir_indice():
    import shutil
    from pathlib import Path
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)   # índice limpio en cada construcción
    docs = DirectoryLoader(str(DOCS_RAG), glob="*.md", loader_cls=TextLoader,
                           loader_kwargs={"encoding": "utf-8"}).load()
    por_titulo = MarkdownHeaderTextSplitter([("#", "doc"), ("##", "seccion")], strip_headers=False)
    fino = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = []
    for d in docs:
        for c in por_titulo.split_text(d.page_content):
            c.metadata["fuente"] = Path(d.metadata["source"]).name
            chunks.extend(fino.split_documents([c]))
    Chroma.from_documents(chunks, _embeddings(), persist_directory=str(CHROMA_DIR),
                          collection_name="conocimiento")
    print(f"Índice listo: {len(chunks)} chunks de {len(docs)} documentos")


def retriever(k: int = 4):
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=_embeddings(),
                collection_name="conocimiento")
    return db.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    construir_indice()
