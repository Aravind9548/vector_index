"""
Phase 6 — Retrieval backend interface for the Due Diligence Agent.

This module defines the abstract interface that the agent's retrieval layer
uses, and provides two concrete implementations:

1. FAISSBackend  — the existing FAISS IndexFlatIP backend (oracle/fallback)
2. VecIdxBackend — our custom C++ engine via gRPC

A feature flag (`RETRIEVAL_BACKEND`) switches between them.  Both backends
must produce identical document IDs so that RRF fusion with BM25 works correctly.

THE SYNC TRAP:
  Document IDs must stay identical between the dense retrieval backend and the
  sparse side (BM25).  If they drift, RRF fuses mismatched results.  The
  VecIdxBackend uses the same string document IDs as FAISS to prevent this.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Feature flag: "faiss" (default) or "vecidx"
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "faiss")


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    doc_id: str
    score: float
    metadata: dict = field(default_factory=dict)


class RetrievalBackend(ABC):
    """Abstract retrieval backend interface.

    Both FAISS and VecIdx must implement this interface so the agent can
    swap backends via feature flag without changing its retrieval code.
    """

    @abstractmethod
    def upsert(
        self,
        doc_id: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert or update a document's embedding."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Search for nearest neighbours."""
        ...

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by ID.  Returns count deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed documents."""
        ...


class FAISSBackend(RetrievalBackend):
    """FAISS IndexFlatIP backend — the oracle and fallback.

    This wraps the agent's existing FAISS usage.  It's the ground truth
    that VecIdxBackend must match.
    """

    def __init__(self, dimension: int = 384):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSBackend.  "
                "Install with: pip install faiss-cpu"
            )
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map: list[str] = []
        self.id_to_row: dict[str, int] = {}
        self.metadata_store: dict[str, dict] = {}

    def upsert(self, doc_id, embedding, metadata=None):
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # L2-normalise for cosine similarity via IP.
        import faiss
        faiss.normalize_L2(vec)

        if doc_id in self.id_to_row:
            row = self.id_to_row[doc_id]
            # FAISS IndexFlat doesn't support in-place update; rebuild.
            # For the oracle this is acceptable.
            self.index.reconstruct(row, vec.ravel())
        else:
            row = len(self.id_map)
            self.id_map.append(doc_id)
            self.id_to_row[doc_id] = row
            self.index.add(vec)

        if metadata:
            self.metadata_store[doc_id] = metadata

    def search(self, query_embedding, top_k=10, filter=None):
        if self.index.ntotal == 0:
            return []

        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        import faiss
        faiss.normalize_L2(vec)

        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.id_map[idx]
            meta = self.metadata_store.get(doc_id, {})
            if filter:
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue
            results.append(RetrievalResult(doc_id=doc_id, score=float(score), metadata=meta))
        return results[:top_k]

    def delete(self, doc_ids):
        # FAISS IndexFlat doesn't support delete.  For the oracle, we skip.
        return 0

    def count(self):
        return self.index.ntotal


class VecIdxBackend(RetrievalBackend):
    """VecIdx C++ engine backend via gRPC.

    Uses the Python gRPC client to talk to the vecidx_server.
    Document IDs are the same strings used by FAISS to prevent
    the RRF sync trap.
    """

    def __init__(self, grpc_target: str = "localhost:50051"):
        from vecidx_client import VecIdxClient
        self.client = VecIdxClient(grpc_target)

    def upsert(self, doc_id, embedding, metadata=None):
        values = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        self.client.upsert(doc_id, values, metadata or {})

    def search(self, query_embedding, top_k=10, filter=None):
        values = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
        raw = self.client.search(values, top_k=top_k, filter=filter)
        return [
            RetrievalResult(doc_id=r["id"], score=r["score"], metadata=r.get("metadata", {}))
            for r in raw
        ]

    def delete(self, doc_ids):
        return self.client.delete(doc_ids)

    def count(self):
        # Would need a Count RPC; for now, search with empty returns size.
        return 0  # Placeholder — add when Count RPC is added to proto.


def get_backend(backend_name: Optional[str] = None) -> RetrievalBackend:
    """Factory: return the configured retrieval backend.

    Args:
        backend_name: "faiss" or "vecidx".  Defaults to RETRIEVAL_BACKEND env var.

    Returns:
        The configured backend instance.

    The agent calls this once at startup.  To compare backends:
        RETRIEVAL_BACKEND=faiss python agent.py   # baseline
        RETRIEVAL_BACKEND=vecidx python agent.py  # experimental
    """
    name = (backend_name or RETRIEVAL_BACKEND).lower()
    if name == "faiss":
        return FAISSBackend()
    elif name == "vecidx":
        return VecIdxBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Use 'faiss' or 'vecidx'.")


# ---------------------------------------------------------------------------
# Eval harness comparison
# ---------------------------------------------------------------------------

def compare_backends(
    queries: list[np.ndarray],
    ground_truth_ids: list[list[str]],
    top_k: int = 10,
) -> dict:
    """Run the same queries through both backends and compare recall.

    This is the Phase 6 gate: recall and faithfulness should match FAISS.
    If they don't, there's a defect, not a feature.

    Returns:
        Dict with recall stats for each backend.
    """
    results = {}
    for backend_name in ["faiss", "vecidx"]:
        try:
            backend = get_backend(backend_name)
        except Exception as e:
            results[backend_name] = {"error": str(e)}
            continue

        recalls = []
        for query, truth in zip(queries, ground_truth_ids):
            retrieved = backend.search(query, top_k=top_k)
            retrieved_ids = {r.doc_id for r in retrieved}
            truth_set = set(truth[:top_k])
            recall = len(retrieved_ids & truth_set) / max(len(truth_set), 1)
            recalls.append(recall)

        results[backend_name] = {
            "avg_recall": float(np.mean(recalls)) if recalls else 0.0,
            "min_recall": float(np.min(recalls)) if recalls else 0.0,
            "max_recall": float(np.max(recalls)) if recalls else 0.0,
            "queries": len(recalls),
        }

    return results
