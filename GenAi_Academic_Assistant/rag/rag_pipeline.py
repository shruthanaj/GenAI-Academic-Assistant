"""
RAG Pipeline
============
Splits text into chunks, embeds them with a sentence-transformer,
stores in a FAISS index, and retrieves relevant chunks on query.

Requires:
    pip install faiss-cpu sentence-transformers
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from typing import List
from utils.text_preprocessor import chunk_text

# ── Lazy imports so the app doesn't crash if the library isn't installed ──────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _LIBS_AVAILABLE = True
except ImportError:
    _LIBS_AVAILABLE = False


CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 1       # sentence overlap between consecutive chunks
TOP_K = 4               # number of chunks to retrieve
MODEL_NAME = "all-MiniLM-L6-v2"


class RAGPipeline:
    """Build a FAISS index from raw text and retrieve relevant passages."""

    def __init__(self, model_name: str = MODEL_NAME):
        if not _LIBS_AVAILABLE:
            raise RuntimeError(
                "RAG dependencies missing. Run: pip install faiss-cpu sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self.index: "faiss.IndexFlatL2 | None" = None
        self.chunks: List[str] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(self, text: str) -> None:
        """Embed *text* chunks and store in a FAISS flat-L2 index."""
        self.chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap_sentences=CHUNK_OVERLAP)
        if not self.chunks:
            raise ValueError("No valid text chunks generated. Check uploaded material content.")

        embeddings = self.model.encode(self.chunks, show_progress_bar=False, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))

    def save(self, path: str) -> None:
        """Persist index + chunks to *path* directory."""
        if self.index is None:
            raise ValueError("Index is empty. Build or load an index before saving.")

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as fh:
            pickle.dump(self.chunks, fh)

    def load(self, path: str) -> None:
        """Load a previously saved index from *path*."""
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.pkl")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Saved index not found in: {path}")

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as fh:
            self.chunks = pickle.load(fh)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K, limit_tokens: bool = False) -> str:
        """
        Return the *top_k* most relevant chunks joined as a single string.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            limit_tokens: If True, retrieve fewer chunks and truncate to save tokens
        """
        if self.index is None or not self.chunks or not query.strip():
            return ""

        # For large requests, limit context chunks
        if limit_tokens:
            top_k = min(top_k, 2)  # Use only top 2 chunks for large requests

        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        _, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

        context = "\n\n---\n\n".join(results)

        # Truncate if still too long
        if limit_tokens and len(context) > 3000:
            context = context[:3000] + "...\n\n[Context truncated for token limit]"

        return context
