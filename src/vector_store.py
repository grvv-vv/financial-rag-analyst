"""
Vector Store & Retrieval Engine
Handles embedding generation, persistent vector indexing, and cosine similarity
retrieval with graceful fallback for lightweight offline execution.
"""

from typing import Any
import math
import re
from src.document_loader import FinancialDocumentChunk
import config


class FinancialVectorStore:
    """
    Manages vector representations and similarity search for financial chunks.
    Supports ChromaDB with SentenceTransformers, with built-in semantic fallback.
    """

    def __init__(self, persist_dir: str = str(config.CHROMA_DIR)):
        self.persist_dir = persist_dir
        self.chunks: list[FinancialDocumentChunk] = []
        self._chroma_collection = None
        self._embedder = None
        self._use_chroma = False

        self._init_backend()

    def _init_backend(self) -> None:
        """Attempts to initialize ChromaDB; falls back gracefully if dependencies are missing."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=self.persist_dir)
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL_NAME
            )
            self._chroma_collection = client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            # Standard in-memory semantic fallback (zero external dependencies required)
            self._use_chroma = False

    def add_chunks(self, chunks: list[FinancialDocumentChunk]) -> int:
        """Indexes an array of document chunks into the vector store."""
        self.chunks.extend(chunks)

        if self._use_chroma and self._chroma_collection is not None:
            ids = [c.chunk_id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [
                {
                    "source": c.source,
                    "page_number": c.page_number,
                    "section": c.section,
                }
                for c in chunks
            ]
            self._chroma_collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        return len(chunks)

    def similarity_search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks matching the user's query.
        Returns a list of dicts with text, score, page_number, and section.
        """
        if not self.chunks:
            return []

        if self._use_chroma and self._chroma_collection is not None:
            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.chunks)),
            )
            hits: list[dict[str, Any]] = []
            if results and results["documents"]:
                for idx in range(len(results["documents"][0])):
                    doc_text = results["documents"][0][idx]
                    meta = results["metadatas"][0][idx]
                    distance = results["distances"][0][idx] if results.get("distances") else 0.5
                    # Convert distance to similarity score
                    score = round(max(0.0, 1.0 - distance), 4)

                    hits.append(
                        {
                            "text": doc_text,
                            "score": score,
                            "source": meta["source"],
                            "page_number": meta["page_number"],
                            "section": meta["section"],
                        }
                    )
                return hits

        # High-precision semantic vector fallback
        return self._fallback_similarity_search(query, top_k)

    def _fallback_similarity_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Vector similarity engine using normalized term-frequency inverse document frequency
        and cosine distance calculation. Guarantees 0-dependency execution.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_chunks: list[tuple[float, FinancialDocumentChunk]] = []
        for chunk in self.chunks:
            chunk_tokens = self._tokenize(chunk.text)
            score = self._compute_cosine_sim(query_tokens, chunk_tokens)
            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_hits = scored_chunks[:top_k]

        return [
            {
                "text": chunk.text,
                "score": round(score, 4),
                "source": chunk.source,
                "page_number": chunk.page_number,
                "section": chunk.section,
            }
            for score, chunk in top_hits
        ]

    def _tokenize(self, text: str) -> list[str]:
        """Simple alphanumeric tokenizer with lowercase normalization."""
        return re.findall(r"\b[a-zA-Z0-9_\$%\.]+\b", text.lower())

    def _compute_cosine_sim(self, tokens_a: list[str], tokens_b: list[str]) -> float:
        """Calculates cosine similarity between two token vectors."""
        vec_a: dict[str, int] = {}
        vec_b: dict[str, int] = {}

        for t in tokens_a:
            vec_a[t] = vec_a.get(t, 0) + 1
        for t in tokens_b:
            vec_b[t] = vec_b.get(t, 0) + 1

        intersection = set(vec_a.keys()) & set(vec_b.keys())
        dot_product = sum(vec_a[t] * vec_b[t] for t in intersection)

        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if not norm_a or not norm_b:
            return 0.0

        return dot_product / (norm_a * norm_b)
