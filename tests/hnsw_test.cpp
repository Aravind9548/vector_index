// Phase 3 — HNSW index tests.
// Measures recall@10 vs. exact flat index as oracle.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "vecidx/flat_index.h"
#include "vecidx/hnsw_index.h"

namespace vecidx {
namespace {

void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
}

TEST(HnswTest, BasicUpsertAndSearch) {
  HnswIndex idx;
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  for (int i = 0; i < 50; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("doc-" + std::to_string(i), v.data());
  }
  EXPECT_EQ(idx.size(), 50u);

  FillRandom(v.data(), kEmbeddingDim, rng);
  auto results = idx.Search(v.data(), 10);
  EXPECT_EQ(results.size(), 10u);

  // Scores should be descending.
  for (std::size_t i = 1; i < results.size(); ++i) {
    EXPECT_GE(results[i - 1].score, results[i].score);
  }
}

TEST(HnswTest, SearchEmptyIndex) {
  HnswIndex idx;
  float q[kEmbeddingDim] = {};
  auto results = idx.Search(q, 10);
  EXPECT_TRUE(results.empty());
}

TEST(HnswTest, SelfSearchFindsItself) {
  HnswIndex idx;
  std::mt19937 rng(99);
  std::vector<float> v(kEmbeddingDim);
  FillRandom(v.data(), kEmbeddingDim, rng);
  idx.Upsert("only-one", v.data());

  auto results = idx.Search(v.data(), 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].id, "only-one");
  EXPECT_GT(results[0].score, 0.99f);
}

TEST(HnswTest, RecallAt10Against1000Vectors) {
  // Build flat index and HNSW on the same data, compare recall.
  constexpr std::size_t N = 1000;
  constexpr std::size_t K = 10;
  constexpr int NUM_QUERIES = 100;

  FlatIndex flat;
  HnswParams params;
  params.ef_construction = 200;
  params.ef_search = 100;
  HnswIndex hnsw(params);

  std::mt19937 rng(2024);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < N; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    std::string id = "v-" + std::to_string(i);
    flat.Upsert(id, v.data());
    hnsw.Upsert(id, v.data());
  }

  double total_recall = 0.0;
  for (int q = 0; q < NUM_QUERIES; ++q) {
    FillRandom(v.data(), kEmbeddingDim, rng);

    auto exact = flat.Search(v.data(), K);
    auto approx = hnsw.Search(v.data(), K);

    // Compute recall: how many of exact top-K are in approx top-K?
    std::set<std::string> exact_set;
    for (auto& r : exact) exact_set.insert(r.id);

    int hits = 0;
    for (auto& r : approx) {
      if (exact_set.count(r.id)) ++hits;
    }
    total_recall += static_cast<double>(hits) / static_cast<double>(K);
  }

  double avg_recall = total_recall / NUM_QUERIES;
  // HNSW with ef_search=100 should achieve > 90% recall on 1000 random vectors.
  EXPECT_GT(avg_recall, 0.90)
      << "Average recall@10 = " << avg_recall << " (expected > 0.90)";

  // Print for the interview artifact.
  std::cout << "\n=== HNSW Recall@" << K << " ===\n"
            << "  N = " << N << ", queries = " << NUM_QUERIES << "\n"
            << "  M = " << params.M << ", ef_construction = "
            << params.ef_construction
            << ", ef_search = " << params.ef_search << "\n"
            << "  Average recall@" << K << " = " << avg_recall << "\n";
}

TEST(HnswTest, HigherEfSearchImprovesRecall) {
  constexpr std::size_t N = 500;
  constexpr std::size_t K = 10;
  constexpr int NUM_QUERIES = 50;

  FlatIndex flat;
  HnswParams params;
  params.ef_construction = 100;
  HnswIndex hnsw(params);

  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < N; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    std::string id = "v-" + std::to_string(i);
    flat.Upsert(id, v.data());
    hnsw.Upsert(id, v.data());
  }

  auto measure_recall = [&](std::size_t ef) {
    double total = 0.0;
    std::mt19937 qrng(999);  // same queries each time
    for (int q = 0; q < NUM_QUERIES; ++q) {
      FillRandom(v.data(), kEmbeddingDim, qrng);
      auto exact = flat.Search(v.data(), K);
      auto approx = hnsw.SearchWithEf(v.data(), K, ef);

      std::set<std::string> exact_set;
      for (auto& r : exact) exact_set.insert(r.id);
      int hits = 0;
      for (auto& r : approx) {
        if (exact_set.count(r.id)) ++hits;
      }
      total += static_cast<double>(hits) / static_cast<double>(K);
    }
    return total / NUM_QUERIES;
  };

  double recall_low = measure_recall(10);
  double recall_high = measure_recall(200);

  std::cout << "\n=== ef_search sensitivity ===\n"
            << "  ef=10  recall@10 = " << recall_low << "\n"
            << "  ef=200 recall@10 = " << recall_high << "\n";

  EXPECT_GT(recall_high, recall_low)
      << "Higher ef should give better recall";
}

}  // namespace
}  // namespace vecidx
