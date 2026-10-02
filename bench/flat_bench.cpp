// Placeholder benchmark — will be replaced with the real flat-index bench
// in Phase 1.
#include <benchmark/benchmark.h>
#include "vecidx/flat_index.h"

static void BM_DimConstant(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(vecidx::kEmbeddingDim);
  }
}
BENCHMARK(BM_DimConstant);
