"""
VecIdx Python client — generated from proto/vecidx.proto.

Usage:
    # Generate stubs first:
    #   python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/vecidx.proto

    from vecidx_client import VecIdxClient

    client = VecIdxClient("localhost:50051")
    client.upsert("doc-1", [0.1] * 1536, {"company_id": "acme"})
    results = client.search([0.1] * 1536, top_k=10, filter={"company_id": "acme"})
    client.delete(["doc-1"])
    client.snapshot("/data/snapshot.vecidx")
    print(client.health("localhost:8080"))
"""

import json
import urllib.request
from typing import Optional


class VecIdxClient:
    """gRPC client for the VecIdx vector index service.

    This is the interface layer.  When proto stubs are generated, this class
    wraps them with a clean Python API.  Until then, it documents the contract.
    """

    def __init__(self, grpc_target: str = "localhost:50051"):
        self.grpc_target = grpc_target
        self._channel = None
        self._stub = None
        try:
            import grpc
            import vecidx_pb2
            import vecidx_pb2_grpc
            self._channel = grpc.insecure_channel(grpc_target)
            self._stub = vecidx_pb2_grpc.VectorIndexStub(self._channel)
            self._pb2 = vecidx_pb2
        except ImportError:
            print("[vecidx_client] gRPC stubs not generated yet.  "
                  "Run: python -m grpc_tools.protoc -I../proto "
                  "--python_out=. --grpc_python_out=. ../proto/vecidx.proto")

    def upsert(
        self,
        id: str,
        values: list[float],
        metadata: Optional[dict[str, str]] = None,
    ) -> int:
        """Upsert a single vector.  Returns upserted count."""
        if self._stub is None:
            raise RuntimeError("gRPC stubs not available")
        vec = self._pb2.Vector(id=id, values=values, metadata=metadata or {})
        req = self._pb2.UpsertRequest(vectors=[vec])
        resp = self._stub.Upsert(req)
        return resp.upserted_count

    def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict[str, str]]],
    ) -> int:
        """Upsert a batch of vectors.  Returns upserted count."""
        if self._stub is None:
            raise RuntimeError("gRPC stubs not available")
        vecs = [
            self._pb2.Vector(id=id, values=vals, metadata=meta)
            for id, vals, meta in vectors
        ]
        req = self._pb2.UpsertRequest(vectors=vecs)
        resp = self._stub.Upsert(req)
        return resp.upserted_count

    def search(
        self,
        query: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """Search for nearest neighbours.  Returns list of {id, score, metadata}."""
        if self._stub is None:
            raise RuntimeError("gRPC stubs not available")
        req = self._pb2.SearchRequest(
            query=query, top_k=top_k, filter=filter or {}
        )
        resp = self._stub.Search(req)
        return [
            {"id": r.id, "score": r.score, "metadata": dict(r.metadata)}
            for r in resp.results
        ]

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by id.  Returns deleted count."""
        if self._stub is None:
            raise RuntimeError("gRPC stubs not available")
        req = self._pb2.DeleteRequest(ids=ids)
        resp = self._stub.Delete(req)
        return resp.deleted_count

    def snapshot(self, path: str) -> dict:
        """Take a snapshot.  Returns {vector_count, size_bytes}."""
        if self._stub is None:
            raise RuntimeError("gRPC stubs not available")
        req = self._pb2.SnapshotRequest(path=path)
        resp = self._stub.Snapshot(req)
        return {"vector_count": resp.vector_count, "size_bytes": resp.size_bytes}

    @staticmethod
    def health(health_url: str = "http://localhost:8080/health") -> dict:
        """Check health via HTTP.  Returns {"status": "ok"/"not_ready"}."""
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def close(self):
        if self._channel:
            self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick connectivity test.
    print("VecIdx client — checking health...")
    print(VecIdxClient.health())
