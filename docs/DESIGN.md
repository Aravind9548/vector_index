# VecIdx — Design Document

## Locked Parameters

| Parameter          | Value                   | Rationale                                                    |
|--------------------|-------------------------|--------------------------------------------------------------|
| Embedding dimension| **384**                | all-MiniLM-L6-v2 (sentence-transformers) output dimension              |
| Distance metric    | **Inner Product**       | On L2-normalised vectors, IP = cosine similarity             |
| Oracle             | FAISS `IndexFlatIP`     | Ground-truth; our results must match exactly (flat) or be measured against it (ANN) |

**Why these are locked:**  A dimension or metric mismatch is the first silent bug.
If the embedding model changes, the index must be rebuilt — there is no in-place
migration.

## Architecture Overview

```
┌──────────────┐         gRPC (.proto)        ┌──────────────┐
│ Python Agent │ ◄──────────────────────────► │  C++ Server  │
│  (retrieval) │                               │  (vecidx)    │
└──────────────┘                               └──────┬───────┘
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    │                 │                 │
                              FlatIndex          HNSWIndex        Snapshot
                             (Phase 1)          (Phase 3)        (Phase 2)
```

## Phases

| Phase | Deliverable | Gate |
|-------|-------------|------|
| 0 | Scaffold: CMake, tests, CI, .proto, this doc | CI green |
| 1 | Exact flat index (naive → SIMD) | top-10 matches FAISS `IndexFlatIP` exactly |
| 2 | Snapshot persistence | write → read → byte-identical |
| 3 | HNSW | recall@10 vs QPS curve against FAISS `IndexHNSWFlat` |
| 4 | Filtered search (company_id) | pre-filter vs post-filter comparison |
| 5 | gRPC service | TSan clean under concurrent load |
| 6 | Agent integration | eval harness: FAISS flag vs vecidx flag |
| 7 | Azure deploy | cold-start readiness probe works |
| 8 | Scale + postmortems | 3+ incidents documented |

## Concurrency Model (Phase 5)

Single writer thread, concurrent readers.  Stated explicitly so TSan can verify.

## Data Layout (Phase 1)

Vectors stored as one contiguous `float[N × 384]` array, row-major.
Not `vector<vector<float>>` — packed for cache locality and SIMD.
