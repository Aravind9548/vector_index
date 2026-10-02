// Phase 1 — Flat index benchmarks.
// Measures: naive dot-product search, varying N.
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "vecidx/flat_index.h"

namespace vecidx {
namespace {

void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = dist(rng);
  }
}

static void BM_FlatSearch(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t K = 10;

  FlatIndex idx;
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < n; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("v-" + std::to_string(i), v.data());
  }

  FillRandom(v.data(), kEmbeddingDim, rng);
  float query[kEmbeddingDim];
  FlatIndex::Normalize(v.data(), query);

  for (auto _ : state) {
    auto results = idx.Search(query, K);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                           static_cast<int64_t>(n));
}

BENCHMARK(BM_FlatSearch)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Unit(benchmark::kMicrosecond);

static void BM_DotProduct(benchmark::State& state) {
  std::mt19937 rng(123);
  float a[kEmbeddingDim], b[kEmbeddingDim];
  FillRandom(a, kEmbeddingDim, rng);
  FillRandom(b, kEmbeddingDim, rng);

  for (auto _ : state) {
    float s = FlatIndex::DotProduct(a, b);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_DotProduct);

}  // namespace
}  // namespace vecidx
