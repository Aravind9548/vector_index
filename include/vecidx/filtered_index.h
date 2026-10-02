// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
//
// Filtered search: search within a company_id (or any metadata key-value).
//
// Two strategies, both implemented and compared:
//
// 1. PRE-FILTER: Restrict the candidate set before search.
//    Pro: Never returns wrong-company results.
//    Con: Wrecks HNSW graph connectivity — recall drops on small partitions.
//         Falls back to flat scan on the filtered subset.
//
// 2. POST-FILTER: Full HNSW search, then drop non-matching results.
//    Pro: Uses the full graph, so recall on the matching subset stays high.
//    Con: Can return fewer than top_k results if the match ratio is low.
//         Must over-fetch to compensate (ef multiplier).
//
// This is the depth-skill phase.  The trade-off maps directly to the agent's
// cross-company citation contamination bug.
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vecidx/flat_index.h"
#include "vecidx/hnsw_index.h"

namespace vecidx {

/// Metadata store: node_id → {key: value} map.
class MetadataStore {
 public:
  void Set(std::size_t node,
           const std::unordered_map<std::string, std::string>& meta) {
    if (node >= meta_.size()) meta_.resize(node + 1);
    meta_[node] = meta;
  }

  const std::unordered_map<std::string, std::string>&
  Get(std::size_t node) const {
    static const std::unordered_map<std::string, std::string> empty;
    if (node >= meta_.size()) return empty;
    return meta_[node];
  }

  /// Check if a node's metadata contains ALL filter key-value pairs.
  bool Matches(std::size_t node,
               const std::unordered_map<std::string, std::string>& filter) const {
    if (filter.empty()) return true;
    if (node >= meta_.size()) return false;
    const auto& m = meta_[node];
    for (const auto& [k, v] : filter) {
      auto it = m.find(k);
      if (it == m.end() || it->second != v) return false;
    }
    return true;
  }

  std::size_t size() const { return meta_.size(); }

 private:
  std::vector<std::unordered_map<std::string, std::string>> meta_;
};

/// Filtered search strategies.
enum class FilterStrategy {
  PreFilter,   // flat scan on filtered subset
  PostFilter,  // HNSW search then drop non-matching
};

/// Filtered index: wraps FlatIndex + HnswIndex + MetadataStore.
class FilteredIndex {
 public:
  explicit FilteredIndex(HnswParams params = {})
      : flat_(), hnsw_(params) {}

  void Upsert(const std::string& id, const float* vec,
              const std::unordered_map<std::string, std::string>& metadata = {}) {
    flat_.Upsert(id, vec);
    hnsw_.Upsert(id, vec);
    // Node IDs are assigned in insertion order for both indexes,
    // so flat_.RowOf(id) == hnsw_.NodeOf(id) when no deletes have occurred.
    std::size_t node = hnsw_.NodeOf(id);
    meta_.Set(node, metadata);
  }

  // -- Pre-filter search (flat scan on filtered subset) -----------------------

  std::vector<ScoredResult> SearchPreFilter(
      const float* query, std::size_t top_k,
      const std::unordered_map<std::string, std::string>& filter) const {
    if (flat_.size() == 0 || top_k == 0) return {};

    // Normalise query.
    float q[kEmbeddingDim];
    FlatIndex::Normalize(query, q);

    // Scan all vectors, skip non-matching.
    using Pair = std::pair<float, std::size_t>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

    for (std::size_t i = 0; i < flat_.size(); ++i) {
      if (!meta_.Matches(i, filter)) continue;
      float score = FlatIndex::DotProduct(q, &flat_.raw_data()[i * kEmbeddingDim]);
      if (heap.size() < top_k) {
        heap.push({score, i});
      } else if (score > heap.top().first) {
        heap.pop();
        heap.push({score, i});
      }
    }

    std::vector<ScoredResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
      auto [s, idx] = heap.top();
      heap.pop();
      results.push_back({idx, flat_.IdOf(idx), s});
    }
    std::sort(results.begin(), results.end(),
              [](const ScoredResult& a, const ScoredResult& b) {
                return a.score > b.score;
              });
    return results;
  }

  // -- Post-filter search (HNSW then drop) ------------------------------------

  std::vector<ScoredResult> SearchPostFilter(
      const float* query, std::size_t top_k,
      const std::unordered_map<std::string, std::string>& filter,
      std::size_t ef_multiplier = 4) const {
    if (hnsw_.size() == 0 || top_k == 0) return {};

    // Over-fetch to compensate for filtering.
    std::size_t ef = top_k * ef_multiplier;
    auto candidates = hnsw_.SearchWithEf(query, ef, ef);

    // Filter.
    std::vector<ScoredResult> filtered;
    for (auto& r : candidates) {
      std::size_t node = hnsw_.NodeOf(r.id);
      if (meta_.Matches(node, filter)) {
        filtered.push_back(r);
        if (filtered.size() >= top_k) break;
      }
    }
    return filtered;
  }

  // -- Convenience: auto-select strategy (post-filter by default) -------------

  std::vector<ScoredResult> Search(
      const float* query, std::size_t top_k,
      const std::unordered_map<std::string, std::string>& filter = {},
      FilterStrategy strategy = FilterStrategy::PostFilter) const {
    if (filter.empty()) {
      // No filter — use HNSW directly.
      return hnsw_.Search(query, top_k);
    }
    switch (strategy) {
      case FilterStrategy::PreFilter:
        return SearchPreFilter(query, top_k, filter);
      case FilterStrategy::PostFilter:
        return SearchPostFilter(query, top_k, filter);
    }
    return {};  // unreachable
  }

  std::size_t size() const { return flat_.size(); }
  const FlatIndex& flat() const { return flat_; }
  const HnswIndex& hnsw() const { return hnsw_; }
  const MetadataStore& metadata() const { return meta_; }

 private:
  FlatIndex flat_;
  HnswIndex hnsw_;
  MetadataStore meta_;
};

}  // namespace vecidx
