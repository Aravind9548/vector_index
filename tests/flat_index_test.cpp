// Phase 1 — Flat index tests.
// Gate: top-10 must match FAISS IndexFlatIP exactly on the same data.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "vecidx/flat_index.h"

namespace vecidx {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fill a vector with deterministic pseudo-random floats in [-1, 1].
void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = dist(rng);
  }
}

/// Brute-force reference: compute IP of normalised query against all normalised
/// rows, return top-k indices sorted by descending score.
/// This is the oracle — equivalent to FAISS IndexFlatIP on normalised vectors.
std::vector<std::pair<float, std::size_t>> ReferenceTopK(
    const float* data, std::size_t n, const float* query, std::size_t top_k) {
  // Normalise query.
  float q[kEmbeddingDim];
  FlatIndex::Normalize(query, q);

  std::vector<std::pair<float, std::size_t>> scores;
  scores.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    float s = FlatIndex::DotProduct(q, &data[i * kEmbeddingDim]);
    scores.push_back({s, i});
  }
  // Sort descending by score, then ascending by index for tie-breaking.
  std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) return a.first > b.first;
              return a.second < b.second;
            });
  if (scores.size() > top_k) scores.resize(top_k);
  return scores;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(FlatIndexTest, UpsertAndSize) {
  FlatIndex idx;
  EXPECT_EQ(idx.size(), 0u);

  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);
  FillRandom(v.data(), kEmbeddingDim, rng);

  idx.Upsert("doc-1", v.data());
  EXPECT_EQ(idx.size(), 1u);

  // Upsert same id replaces, size stays 1.
  idx.Upsert("doc-1", v.data());
  EXPECT_EQ(idx.size(), 1u);

  idx.Upsert("doc-2", v.data());
  EXPECT_EQ(idx.size(), 2u);
}

TEST(FlatIndexTest, DeleteReducesSize) {
  FlatIndex idx;
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  for (int i = 0; i < 5; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("doc-" + std::to_string(i), v.data());
  }
  EXPECT_EQ(idx.size(), 5u);

  EXPECT_TRUE(idx.Delete("doc-2"));
  EXPECT_EQ(idx.size(), 4u);

  // Deleting non-existent id returns false.
  EXPECT_FALSE(idx.Delete("doc-99"));
  EXPECT_EQ(idx.size(), 4u);
}

TEST(FlatIndexTest, DeleteSwapsCorrectly) {
  FlatIndex idx;
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  // Insert 3 docs.
  for (int i = 0; i < 3; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("doc-" + std::to_string(i), v.data());
  }

  // Delete the first one; last row should move into its slot.
  idx.Delete("doc-0");
  EXPECT_EQ(idx.size(), 2u);

  // "doc-2" (was last) should now be at row 0.
  EXPECT_EQ(idx.RowOf("doc-2"), 0u);
  EXPECT_EQ(idx.RowOf("doc-1"), 1u);
  EXPECT_EQ(idx.RowOf("doc-0"), static_cast<std::size_t>(-1));
}

TEST(FlatIndexTest, NormalizationIsCorrect) {
  std::mt19937 rng(123);
  std::vector<float> v(kEmbeddingDim);
  FillRandom(v.data(), kEmbeddingDim, rng);

  float out[kEmbeddingDim];
  FlatIndex::Normalize(v.data(), out);

  // L2 norm of output should be ~1.0.
  float norm = 0.0f;
  for (std::size_t i = 0; i < kEmbeddingDim; ++i) {
    norm += out[i] * out[i];
  }
  EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-5f);
}

TEST(FlatIndexTest, SearchReturnsCorrectTopK) {
  constexpr std::size_t N = 100;
  constexpr std::size_t K = 10;

  FlatIndex idx;
  std::mt19937 rng(7);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < N; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("doc-" + std::to_string(i), v.data());
  }

  // Generate a query.
  FillRandom(v.data(), kEmbeddingDim, rng);

  auto results = idx.Search(v.data(), K);
  ASSERT_EQ(results.size(), K);

  // Compare against the oracle.
  auto ref = ReferenceTopK(idx.raw_data(), idx.size(), v.data(), K);
  ASSERT_EQ(ref.size(), K);

  for (std::size_t i = 0; i < K; ++i) {
    EXPECT_EQ(results[i].index, ref[i].second)
        << "Mismatch at rank " << i
        << ": got row " << results[i].index
        << ", expected row " << ref[i].second;
    EXPECT_NEAR(results[i].score, ref[i].first, 1e-5f);
  }
}

TEST(FlatIndexTest, SearchMatchesFAISSOracleOnLargerSet) {
  // 1000 vectors, top-10 — this is the Phase 1 gate.
  constexpr std::size_t N = 1000;
  constexpr std::size_t K = 10;

  FlatIndex idx;
  std::mt19937 rng(2024);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < N; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("vec-" + std::to_string(i), v.data());
  }

  // Run 50 random queries and check every one.
  for (int q = 0; q < 50; ++q) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    auto results = idx.Search(v.data(), K);
    auto ref = ReferenceTopK(idx.raw_data(), idx.size(), v.data(), K);

    ASSERT_EQ(results.size(), K) << "query " << q;
    ASSERT_EQ(ref.size(), K) << "query " << q;

    for (std::size_t i = 0; i < K; ++i) {
      EXPECT_EQ(results[i].index, ref[i].second)
          << "query=" << q << " rank=" << i;
      EXPECT_NEAR(results[i].score, ref[i].first, 1e-5f)
          << "query=" << q << " rank=" << i;
    }
  }
}

TEST(FlatIndexTest, SearchEmptyIndex) {
  FlatIndex idx;
  float q[kEmbeddingDim] = {};
  auto results = idx.Search(q, 10);
  EXPECT_TRUE(results.empty());
}

TEST(FlatIndexTest, SearchWithTopKLargerThanCount) {
  FlatIndex idx;
  std::mt19937 rng(99);
  std::vector<float> v(kEmbeddingDim);

  for (int i = 0; i < 3; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("x-" + std::to_string(i), v.data());
  }

  FillRandom(v.data(), kEmbeddingDim, rng);
  auto results = idx.Search(v.data(), 100);
  EXPECT_EQ(results.size(), 3u);
}

TEST(FlatIndexTest, UpsertOverwriteChangesVector) {
  FlatIndex idx;
  std::mt19937 rng(55);
  std::vector<float> v1(kEmbeddingDim), v2(kEmbeddingDim);
  FillRandom(v1.data(), kEmbeddingDim, rng);
  FillRandom(v2.data(), kEmbeddingDim, rng);

  idx.Upsert("same-id", v1.data());
  auto r1 = idx.Search(v1.data(), 1);

  idx.Upsert("same-id", v2.data());
  auto r2 = idx.Search(v2.data(), 1);

  EXPECT_EQ(r1[0].id, "same-id");
  EXPECT_EQ(r2[0].id, "same-id");
  // After overwrite, searching with v2 should give a perfect score (~1.0).
  EXPECT_GT(r2[0].score, 0.99f);
}

}  // namespace
}  // namespace vecidx
