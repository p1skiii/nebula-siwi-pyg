import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    TfidfVectorizer = None  # type: ignore


class TfidfEmbedder:
    """Lightweight embedder based on TF-IDF, with graceful fallback if sklearn is missing."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english") if TfidfVectorizer else None
        self._fitted = False
        self._fitted_texts = None

    @property
    def available(self) -> bool:
        return self.vectorizer is not None

    def fit_transform(self, texts):
        self._fitted_texts = list(texts)
        if not self.available:
            self._fitted = True
            return None
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix

    def encode_queries(self, queries):
        if not self._fitted:
            raise RuntimeError("Embedder not fitted; call fit_transform first")
        if not self.available:
            return None
        return self.vectorizer.transform(queries)


def normalize(vectors):
    # sklearn outputs sparse matrices; this keeps compatibility
    if hasattr(vectors, "toarray"):
        vectors = vectors.toarray()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors / norms
