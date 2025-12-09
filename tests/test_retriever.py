import tempfile
from pathlib import Path

from siwi.rag.doc_loader import DocLoader
from siwi.rag.embedder import TfidfEmbedder
from siwi.rag.retriever import Retriever


def test_retriever_returns_relevant_chunk():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "a.txt").write_text("Nebula SIWI Bot is a refreshed ChatBot backend.", encoding="utf-8")
        (data_dir / "b.txt").write_text("Totally unrelated content goes here.", encoding="utf-8")

        docs = DocLoader(data_dir).load()
        retriever = Retriever(embedder=TfidfEmbedder(), documents=docs)

        results = retriever.retrieve("What is Nebula SIWI Bot?", top_k=2)
        assert results, "Expected at least one result"
        first = results[0]
        assert "Nebula SIWI Bot".lower() in first["snippet"].lower() or "Nebula SIWI Bot".lower() in first["title"].lower()
        assert isinstance(first["score"], float)
        assert 0 <= first["score"] <= 1
        # Scores should be sorted descending
        scores = [item["score"] for item in results]
        assert scores == sorted(scores, reverse=True)
