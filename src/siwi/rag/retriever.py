from dataclasses import dataclass
from typing import List, Dict
import re

import numpy as np

from siwi.rag.doc_loader import DocumentChunk
from siwi.rag.embedder import TfidfEmbedder
from siwi.rag.index import compute_similarity


@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, embedder: TfidfEmbedder, documents: List[DocumentChunk], snippet_size: int = 200):
        self.embedder = embedder
        self.documents = documents
        self.snippet_size = snippet_size
        self.doc_matrix = self.embedder.fit_transform([doc.text for doc in documents])

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        scores = None
        if self.embedder.available:
            try:
                query_vec = self.embedder.encode_queries([query])
                scores = compute_similarity(query_vec, self.doc_matrix)
            except Exception:
                scores = None

        # If similarity is unavailable or all zeros, fall back to keyword overlap
        if scores is None or len(scores) == 0 or np.allclose(scores, 0):
            scores = self._keyword_overlap_scores(query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            score = float(max(0.0, min(1.0, scores[idx])))
            chunk = self.documents[idx]
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "snippet": chunk.text[: self.snippet_size],
                    "score": score,
                }
            )
        return results

    def _keyword_overlap_scores(self, query: str) -> np.ndarray:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return np.zeros(len(self.documents))

        scores = []
        for doc in self.documents:
            doc_tokens = set(_tokenize(doc.text))
            overlap = len(query_tokens & doc_tokens)
            norm = max(len(query_tokens), len(doc_tokens), 1)
            scores.append(overlap / norm)

        return np.array(scores, dtype=float)
