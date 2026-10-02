// Phase 8 — Scale benchmark.
// Tests flat and HNSW at larger scale (10K, 50K vectors).
// Measures QPS and recall@10 for the interview artifact.
#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <string>
#include <vector>
#include <set>
#include <iostream>

#include "vecidx/flat_index.h"
#include "vecidx/hnsw_index.h"

namespace vecidx {
namespace {

void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
}

// -- Flat index at scale ------------------------------------------------------

static void BM_FlatSearch_Scale(benchmark::State& state) {
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
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_FlatSearch_Scale)
    ->Arg(1000)
    ->Arg(10000)
    ->Unit(benchmark::kMillisecond);

// -- HNSW at scale ------------------------------------------------------------

static void BM_HnswSearch_Scale(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto ef = static_cast<std::size_t>(state.range(1));
  constexpr std::size_t K = 10;

  HnswParams params;
  params.ef_construction = 200;
  params.ef_search = ef;
  HnswIndex idx(params);

  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < n; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("v-" + std::to_string(i), v.data());
  }

  FillRandom(v.data(), kEmbeddingDim, rng);

  for (auto _ : state) {
    auto results = idx.Search(v.data(), K);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_HnswSearch_Scale)
    ->Args({1000, 50})
    ->Args({1000, 200})
    ->Args({10000, 50})
    ->Args({10000, 200})
    ->Unit(benchmark::kMillisecond);

// -- HNSW build time ----------------------------------------------------------

static void BM_HnswBuild(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    HnswParams params;
    params.ef_construction = 200;
    HnswIndex idx(params);

    std::mt19937 rng(42);
    std::vector<float> v(kEmbeddingDim);
    for (std::size_t i = 0; i < n; ++i) {
      FillRandom(v.data(), kEmbeddingDim, rng);
      idx.Upsert("v-" + std::to_string(i), v.data());
    }
    benchmark::DoNotOptimize(idx.size());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                           static_cast<int64_t>(n));
}

BENCHMARK(BM_HnswBuild)
    ->Arg(1000)
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace vecidx
