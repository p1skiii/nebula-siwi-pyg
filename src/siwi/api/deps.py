from typing import Any, Dict, List

from siwi.api.settings import Settings
from siwi.rag.doc_loader import DocumentChunk, DocLoader
from siwi.rag.embedder import TfidfEmbedder
from siwi.rag.llm_client import ConfigError, LLMClient, build_llm_client
from siwi.rag.pipeline import RagPipeline
from siwi.rag.retriever import Retriever


def build_dependencies(settings: Settings) -> Dict[str, Any]:
    """Initialize long-lived dependencies for the API."""
    loader = DocLoader(settings.rag_data_dir)
    documents: List[DocumentChunk] = loader.load()

    embedder = TfidfEmbedder()
    retriever = Retriever(embedder=embedder, documents=documents)

    try:
        llm_client: LLMClient = build_llm_client(
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            allow_mock=settings.llm_allow_mock,
            model=settings.llm_model,
        )
        llm_error = None
    except ConfigError as exc:
        llm_client = None
        llm_error = str(exc)

    pipeline = RagPipeline(
        retriever=retriever,
        llm_client=llm_client,
        top_k=settings.rag_top_k,
        llm_error=llm_error,
    )

    return {
        "documents": documents,
        "embedder": embedder,
        "retriever": retriever,
        "llm_client": llm_client,
        "pipeline": pipeline,
        "settings": settings,
        "llm_error": llm_error,
    }
