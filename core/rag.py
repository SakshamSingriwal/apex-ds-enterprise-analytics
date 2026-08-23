"""
Retrieval-Augmented Generation (RAG) pipeline.
Supports PDF, DOCX, TXT. Embeds with sentence-transformers + FAISS.
Generates answers via Ollama (mistral).
"""
from __future__ import annotations

import io
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("apex_ds.rag")


def _extract_text(file_obj) -> str:
    """Extract raw text from uploaded file object."""
    name = getattr(file_obj, "name", "")
    data = file_obj.read()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore[import]
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return "pypdf not installed."

    if name.endswith(".docx"):
        try:
            from docx import Document  # type: ignore[import]
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return "python-docx not installed."

    return data.decode("utf-8", errors="ignore")


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


class RAGPipeline:
    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._index = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def load_document(self, file_obj) -> None:
        text = _extract_text(file_obj)
        self._chunks = _chunk_text(text)

    def build_index(self) -> None:
        import faiss  # type: ignore[import]
        embedder = self._get_embedder()
        embeddings = embedder.encode(self._chunks, show_progress_bar=False).astype(np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)

    def query(self, query: str, top_k: int = 4) -> Tuple[str, List[str]]:
        if not self._chunks or self._index is None:
            return "No document indexed.", []

        embedder = self._get_embedder()
        q_emb = embedder.encode([query], show_progress_bar=False).astype(np.float32)
        _, indices = self._index.search(q_emb, min(top_k, len(self._chunks)))
        retrieved = [self._chunks[i] for i in indices[0] if i < len(self._chunks)]
        context = "\n\n".join(retrieved)

        try:
            import ollama  # type: ignore[import]
            prompt = (
                f"You are a helpful data science assistant. Answer the question based on the context.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )
            resp = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            answer = resp.get("message", {}).get("content", str(resp))
        except Exception as exc:
            answer = f"Ollama not available in this environment ({exc}). Retrieved context: {context[:300]}"

        return answer, retrieved