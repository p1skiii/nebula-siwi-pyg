import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover - fallback path
    cosine_similarity = None  # type: ignore


def compute_similarity(query_vector, doc_matrix) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a document matrix.

    Works with sparse matrices if sklearn is available; otherwise falls back to manual dense computation.
    """
    if cosine_similarity and query_vector is not None and doc_matrix is not None:
        return cosine_similarity(query_vector, doc_matrix)[0]

    # Fallback: manual dense cosine
    if query_vector is None or doc_matrix is None:
        return np.zeros(0)

    if hasattr(query_vector, "toarray"):
        query_vector = query_vector.toarray()
    if hasattr(doc_matrix, "toarray"):
        doc_matrix = doc_matrix.toarray()

    query_norm = query_vector / (np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-9)
    doc_norm = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9)
    return np.dot(query_norm, doc_norm.T)[0]
