# VecIdx Benchmark Results

**Hardware:** 12th Gen Intel (12 cores, 2496 MHz), L3 12 MiB  
**Compiler:** GCC 15.2.0, -O3  
**OS:** Windows 11 (MSYS2 ucrt64)  
**Dimension:** 1536, Metric: Inner Product  

## Search Latency (engine-only, no gRPC)

| Index | N | ef_search | p50 (ms) | QPS |
|-------|---|-----------|----------|-----|
| Flat (brute-force) | 1,000 | — | 0.88 | 1,138 |
| Flat (brute-force) | 10,000 | — | 9.00 | 112 |
| HNSW | 1,000 | 50 | 4.46 | 235 |
| HNSW | 1,000 | 200 | 5.62 | 188 |

## Dot Product Kernel

| Implementation | Time (ns) |
|---------------|-----------|
| Naive scalar loop (1536 floats) | 4,046 |

## Recall

| Index | N | ef_search | recall@10 |
|-------|---|-----------|-----------|
| Flat (brute-force) | any | — | 1.000 (exact) |
| HNSW | 1,000 | 10 | 0.614 |
| HNSW | 1,000 | 50 | 0.927 (est.) |
| HNSW | 1,000 | 100 | 0.992 |
| HNSW | 1,000 | 200 | 1.000 |

## Filtered Search

| Strategy | Partition (%) | recall vs oracle | Returns k? |
|----------|--------------|-----------------|------------|
| Pre-filter (flat) | 20% | 1.000 | Always |
| Post-filter (HNSW, ef_mult=1) | 20% | — | 0/10 |
| Post-filter (HNSW, ef_mult=4) | 20% | 0.778 | Usually |
| Post-filter (HNSW, ef_mult=10) | 20% | ~0.95 | 10/10 |

## What this does NOT show

- These numbers are engine-only.  End-to-end through gRPC adds ~0.1–0.5ms.
- Nothing here beats FAISS on speed.  FAISS uses BLAS, SIMD, and GPU.
- The pitch is: "I built it, verified it against FAISS, ran it as a real
  service, and can explain every trade-off."

## Before / After Table (Optimisations)

| Optimisation | Before | After | Speedup |
|-------------|--------|-------|---------|
| Contiguous float array (vs vector<vector<float>>) | N/A (started right) | 0.88ms/1K | — |
| Naive dot product (1536 floats) | 4,046 ns | — | Baseline |
| SIMD (AVX2) | Not yet implemented | TBD | TBD |
| HNSW vs flat at 10K | 9.00ms | TBD (10K HNSW build slow) | TBD |
