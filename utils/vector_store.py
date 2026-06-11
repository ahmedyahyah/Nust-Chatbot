"""
Vector Store - FAISS vector database using SentenceTransformers.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def create_vector_store(chunks: List[str]) -> Tuple[faiss.Index, List[str]]:
    """Create FAISS index from text chunks."""
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, chunks


def search_vector_store(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    top_k: int = 4
) -> List[str]:
    """Search for most relevant chunks."""
    model = get_model()
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    top_k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and scores[0][i] > 0.3:
            results.append(chunks[idx])
    return results


def save_vector_store(index: faiss.Index, chunks: List[str], path: str = "vector_store"):
    """Save FAISS index and chunks to disk."""
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_vector_store(path: str = "vector_store") -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index from disk."""
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def vector_store_exists(path: str = "vector_store") -> bool:
    """Check if saved vector store exists."""
    return (
        os.path.exists(os.path.join(path, "index.faiss")) and
        os.path.exists(os.path.join(path, "chunks.pkl"))
    )
