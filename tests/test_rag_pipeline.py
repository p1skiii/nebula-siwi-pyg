from siwi.rag.doc_loader import DocLoader, DocumentChunk
from siwi.rag.embedder import TfidfEmbedder
from siwi.rag.pipeline import RagPipeline
from siwi.rag.retriever import Retriever


class FakeLLM:
    def generate(self, query, context):
        return "mock answer"


def test_rag_pipeline_runs_and_returns_sources(tmp_path):
    (tmp_path / "a.txt").write_text("Nebula SIWI Bot is a refreshed ChatBot backend.", encoding="utf-8")
    docs = DocLoader(tmp_path).load()
    retriever = Retriever(embedder=TfidfEmbedder(), documents=docs)
    pipeline = RagPipeline(retriever=retriever, llm_client=FakeLLM(), top_k=2)

    result = pipeline.run("What is Nebula SIWI Bot?")

    assert isinstance(result["answer"], str)
    assert len(result["sources"]) >= 1
    assert result["meta"]["mode"] == "text_rag"
    assert "llm_provider" in result["meta"]
    # Ensure sources carry scores and are sorted
    scores = [s["score"] for s in result["sources"]]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)
