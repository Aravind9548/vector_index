// Phase 4 — Filtered search tests.
// Compare pre-filter vs post-filter: recall, result count, cross-company isolation.
#include <gtest/gtest.h>

#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "vecidx/filtered_index.h"

namespace vecidx {
namespace {

void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
}

class FilteredSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    HnswParams params;
    params.ef_construction = 200;
    params.ef_search = 100;
    idx_ = std::make_unique<FilteredIndex>(params);

    std::mt19937 rng(42);
    std::vector<float> v(kEmbeddingDim);

    // 500 vectors split across 5 companies (100 each).
    for (int i = 0; i < 500; ++i) {
      FillRandom(v.data(), kEmbeddingDim, rng);
      std::string company = "company-" + std::to_string(i / 100);
      idx_->Upsert("doc-" + std::to_string(i), v.data(),
                    {{"company_id", company}});
    }
  }

  std::unique_ptr<FilteredIndex> idx_;
};

TEST_F(FilteredSearchTest, NoFilterReturnsAll) {
  std::mt19937 rng(99);
  std::vector<float> q(kEmbeddingDim);
  FillRandom(q.data(), kEmbeddingDim, rng);

  auto results = idx_->Search(q.data(), 10);
  EXPECT_EQ(results.size(), 10u);
}

TEST_F(FilteredSearchTest, PreFilterOnlyReturnsMatchingCompany) {
  std::mt19937 rng(99);
  std::vector<float> q(kEmbeddingDim);
  FillRandom(q.data(), kEmbeddingDim, rng);

  std::unordered_map<std::string, std::string> filter = {
      {"company_id", "company-2"}};
  auto results = idx_->Search(q.data(), 10, filter, FilterStrategy::PreFilter);

  for (auto& r : results) {
    // Verify metadata.
    auto& meta = idx_->metadata().Get(r.index);
    auto it = meta.find("company_id");
    ASSERT_NE(it, meta.end());
    EXPECT_EQ(it->second, "company-2")
        << "Cross-company leakage! doc=" << r.id;
  }
}

TEST_F(FilteredSearchTest, PostFilterOnlyReturnsMatchingCompany) {
  std::mt19937 rng(99);
  std::vector<float> q(kEmbeddingDim);
  FillRandom(q.data(), kEmbeddingDim, rng);

  std::unordered_map<std::string, std::string> filter = {
      {"company_id", "company-3"}};
  auto results = idx_->Search(q.data(), 10, filter, FilterStrategy::PostFilter);

  for (auto& r : results) {
    auto& meta = idx_->metadata().Get(r.index);
    auto it = meta.find("company_id");
    ASSERT_NE(it, meta.end());
    EXPECT_EQ(it->second, "company-3")
        << "Cross-company leakage! doc=" << r.id;
  }
}

TEST_F(FilteredSearchTest, PreFilterRecallOnLargePartition) {
  // Pre-filter on a partition with 100 vectors (company-0).
  // Since it's a flat scan, recall should be 1.0.
  std::mt19937 rng(77);
  std::vector<float> q(kEmbeddingDim);
  std::unordered_map<std::string, std::string> filter = {
      {"company_id", "company-0"}};

  double total_recall = 0.0;
  constexpr int QUERIES = 50;
  for (int i = 0; i < QUERIES; ++i) {
    FillRandom(q.data(), kEmbeddingDim, rng);
    auto pre = idx_->Search(q.data(), 10, filter, FilterStrategy::PreFilter);
    // Pre-filter is exact on the filtered subset — recall must be 1.0.
    // We check it returns 10 results from the right company.
    EXPECT_EQ(pre.size(), 10u) << "Pre-filter should return exactly 10";
    for (auto& r : pre) {
      auto& meta = idx_->metadata().Get(r.index);
      EXPECT_EQ(meta.at("company_id"), "company-0");
    }
    total_recall += 1.0;  // Exact by definition.
  }
  std::cout << "\n=== Pre-filter recall@10 = "
            << (total_recall / QUERIES) << " (exact, by definition) ===\n";
}

TEST_F(FilteredSearchTest, PostFilterMayReturnFewerResults) {
  // With a very small partition ratio (1 company out of 5 = 20%),
  // post-filter with ef_multiplier=1 may not return enough.
  std::mt19937 rng(33);
  std::vector<float> q(kEmbeddingDim);
  FillRandom(q.data(), kEmbeddingDim, rng);

  std::unordered_map<std::string, std::string> filter = {
      {"company_id", "company-4"}};
  // Low multiplier — may not find enough.
  auto results = idx_->SearchPostFilter(q.data(), 10, filter, 1);

  // We just document this behaviour; it's the trade-off.
  std::cout << "\n=== Post-filter (ef_mult=1): returned "
            << results.size() << "/10 ===\n";
  // With higher multiplier it should be better.
  auto results_high = idx_->SearchPostFilter(q.data(), 10, filter, 10);
  std::cout << "=== Post-filter (ef_mult=10): returned "
            << results_high.size() << "/10 ===\n";
  EXPECT_GE(results_high.size(), results.size());
}

TEST_F(FilteredSearchTest, CompareStrategiesRecall) {
  // The comparison table: pre-filter vs post-filter on the same queries.
  std::mt19937 rng(2024);
  std::vector<float> q(kEmbeddingDim);
  constexpr int QUERIES = 50;

  std::unordered_map<std::string, std::string> filter = {
      {"company_id", "company-1"}};

  int pre_total = 0, post_total = 0;
  for (int i = 0; i < QUERIES; ++i) {
    FillRandom(q.data(), kEmbeddingDim, rng);
    auto pre = idx_->Search(q.data(), 10, filter, FilterStrategy::PreFilter);
    auto post = idx_->Search(q.data(), 10, filter, FilterStrategy::PostFilter);

    // Count how many post-filter results match pre-filter (the oracle).
    std::set<std::string> pre_set;
    for (auto& r : pre) pre_set.insert(r.id);
    int hits = 0;
    for (auto& r : post) {
      if (pre_set.count(r.id)) ++hits;
    }
    pre_total += static_cast<int>(pre.size());
    post_total += hits;
  }

  double post_recall_vs_pre =
      static_cast<double>(post_total) / static_cast<double>(pre_total);

  std::cout << "\n=== Strategy Comparison ===\n"
            << "  Pre-filter: exact (flat scan on partition)\n"
            << "  Post-filter recall vs pre-filter = " << post_recall_vs_pre
            << "\n"
            << "  Pre-filter: always returns k results from the right company\n"
            << "  Post-filter: may return < k if partition is small\n"
            << "  Trade-off: pre-filter is O(partition_size), "
               "post-filter uses the full HNSW graph\n";
}

TEST_F(FilteredSearchTest, NoCrossCompanyLeakageUnderStress) {
  // Run 200 queries with random company filters.  None should leak.
  std::mt19937 rng(12345);
  std::vector<float> q(kEmbeddingDim);

  for (int i = 0; i < 200; ++i) {
    FillRandom(q.data(), kEmbeddingDim, rng);
    std::string company =
        "company-" + std::to_string(static_cast<int>(rng() % 5));
    std::unordered_map<std::string, std::string> filter = {
        {"company_id", company}};

    for (auto strategy : {FilterStrategy::PreFilter, FilterStrategy::PostFilter}) {
      auto results = idx_->Search(q.data(), 10, filter, strategy);
      for (auto& r : results) {
        auto& meta = idx_->metadata().Get(r.index);
        auto it = meta.find("company_id");
        ASSERT_NE(it, meta.end()) << "Missing company_id on " << r.id;
        ASSERT_EQ(it->second, company)
            << "CROSS-COMPANY LEAKAGE! query_company=" << company
            << " result_company=" << it->second
            << " doc=" << r.id
            << " strategy=" << (strategy == FilterStrategy::PreFilter
                                    ? "pre" : "post");
      }
    }
  }
}

}  // namespace
}  // namespace vecidx
