from typing import Dict, List, Optional

from siwi.rag.doc_loader import DocumentChunk
from siwi.rag.llm_client import LLMClient, MockLLMClient
from siwi.rag.retriever import Retriever


class RagPipeline:
    """Text RAG pipeline: retrieve -> generate answer."""

    def __init__(self, retriever: Retriever, llm_client: Optional[LLMClient], top_k: int = 4, llm_error: Optional[str] = None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.llm_error = llm_error

    def _build_context(self, sources: List[Dict]) -> List[DocumentChunk]:
        # Turn source dicts back into lightweight DocumentChunk for LLM
        chunks: List[DocumentChunk] = []
        for idx, src in enumerate(sources):
            chunks.append(
                DocumentChunk(
                    doc_id=src.get("doc_id", f"src-{idx}"),
                    title=src.get("title", ""),
                    text=src.get("snippet", ""),
                )
            )
        return chunks

    def run(self, query: str) -> Dict:
        sources = self.retriever.retrieve(query, top_k=self.top_k)

        # Graceful fallback if no LLM client is provided
        if self.llm_client is None:
            answer = self._build_fallback_answer(query, sources)
            llm_provider = "unconfigured"
        else:
            context_chunks = self._build_context(sources)
            answer = self.llm_client.generate(query=query, context=context_chunks)
            # Prefix demo notice for mock client
            if isinstance(self.llm_client, MockLLMClient) and not answer.startswith("[DEMO MODE]"):
                answer = "[DEMO MODE] " + answer
            llm_provider = self.llm_client.__class__.__name__

        meta = {
            "mode": "text_rag",
            "top_k": self.top_k,
            "llm_provider": llm_provider,
            "llm_error": self.llm_error,
        }

        if not sources:
            answer = answer or "没有找到相关文档。"

        return {
            "answer": answer,
            "sources": sources,
            "meta": meta,
        }

    def _build_fallback_answer(self, query: str, sources: List[Dict]) -> str:
        if not sources:
            return "没有找到相关文档。"
        snippets = " ".join(src.get("snippet", "") for src in sources)
        return f"(No LLM configured) Query: {query}. Context: {snippets}"
