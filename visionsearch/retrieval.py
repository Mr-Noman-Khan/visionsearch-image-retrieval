"""Small reusable cosine-similarity retrieval utilities."""
from __future__ import annotations
import numpy as np

def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms

def top_k_cosine(query: np.ndarray, database: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return (index, cosine score) pairs, sorted by descending similarity."""
    if database.ndim != 2:
        raise ValueError("database must be a 2D array")
    if k < 1:
        raise ValueError("k must be at least 1")
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if database.shape[1] != q.shape[0]:
        raise ValueError("query and database vectors must have matching dimensions")
    qnorm = max(float(np.linalg.norm(q)), 1e-12)
    db = normalize_rows(database)
    scores = db @ (q / qnorm)
    order = np.argsort(scores)[::-1][:min(k, len(scores))]
    return [(int(i), float(scores[i])) for i in order]
