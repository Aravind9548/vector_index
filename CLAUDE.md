# VecIdx — Claude Code Context

## What is this?
A C++17 vector similarity search engine, built from scratch as a learning project.
It will be integrated into a Due Diligence Agent as an alternative to FAISS.

## Locked parameters
- Embedding dimension: **384** (all-MiniLM-L6-v2)
- Distance metric: **Inner Product** on L2-normalised vectors (= cosine)
- Oracle: **FAISS IndexFlatIP** — our ground truth

## Build (Linux / WSL2)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## Sanitizer builds (Linux only)
```bash
# ASan + UBSan
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVECIDX_ENABLE_ASAN=ON -DVECIDX_BUILD_BENCHMARKS=OFF
cmake --build build-asan -j$(nproc)
ctest --test-dir build-asan --output-on-failure

# TSan
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DVECIDX_ENABLE_TSAN=ON -DVECIDX_BUILD_BENCHMARKS=OFF
cmake --build build-tsan -j$(nproc)
ctest --test-dir build-tsan --output-on-failure
```

## Project layout
```
proto/          .proto files (API contract)
include/vecidx/ public headers
src/            implementation
tests/          GoogleTest
bench/          Google Benchmark
docs/           design docs, postmortems
```

## Rules
- Every phase gates on its test before moving on.
- Flat index top-10 must match FAISS IndexFlatIP **exactly**, not approximately.
- Vectors are stored as contiguous float arrays [N × 384], row-major.
- Concurrency model: single writer, concurrent readers.
