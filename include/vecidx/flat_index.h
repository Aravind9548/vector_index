// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vecidx {

/// Locked contract — do not change without a migration.
constexpr std::size_t kEmbeddingDim = 1536;

/// Distance metric.
enum class Metric : std::uint8_t {
  InnerProduct = 0,
};

/// A scored search result.
struct ScoredResult {
  std::size_t index;      // internal row index
  std::string id;         // external string id
  float score;            // inner-product score (higher = more similar)
};

/// Flat (brute-force) vector index.
///
/// Storage: one contiguous float array, row-major [N × kEmbeddingDim].
/// Search: dot product against all rows, top-k via min-heap.
class FlatIndex {
 public:
  explicit FlatIndex(Metric metric = Metric::InnerProduct)
      : metric_(metric) {}

  // -- Mutation ---------------------------------------------------------------

  /// Insert or replace a vector.  `vec` must point to exactly kEmbeddingDim
  /// floats.  The vector is L2-normalised before storage.
  void Upsert(const std::string& id, const float* vec) {
    auto it = id_to_row_.find(id);
    if (it != id_to_row_.end()) {
      // Replace in-place.
      float* dst = &data_[it->second * kEmbeddingDim];
      CopyAndNormalize(vec, dst);
    } else {
      std::size_t row = count_;
      id_to_row_[id] = row;
      row_to_id_.push_back(id);
      data_.resize((count_ + 1) * kEmbeddingDim);
      float* dst = &data_[row * kEmbeddingDim];
      CopyAndNormalize(vec, dst);
      ++count_;
    }
  }

  /// Delete a vector by id.  Returns true if it existed.
  bool Delete(const std::string& id) {
    auto it = id_to_row_.find(id);
    if (it == id_to_row_.end()) return false;

    std::size_t row = it->second;
    std::size_t last = count_ - 1;

    if (row != last) {
      // Swap with last row.
      std::memcpy(&data_[row * kEmbeddingDim],
                  &data_[last * kEmbeddingDim],
                  kEmbeddingDim * sizeof(float));
      const std::string& moved_id = row_to_id_[last];
      id_to_row_[moved_id] = row;
      row_to_id_[row] = moved_id;
    }

    id_to_row_.erase(it);
    row_to_id_.pop_back();
    data_.resize(last * kEmbeddingDim);
    --count_;
    return true;
  }

  // -- Query ------------------------------------------------------------------

  /// Brute-force top-k search.  `query` must point to kEmbeddingDim floats.
  /// Returns up to `top_k` results sorted by descending score.
  std::vector<ScoredResult> Search(const float* query, std::size_t top_k) const {
    if (count_ == 0 || top_k == 0) return {};

    // Normalise query into a local buffer.
    float q[kEmbeddingDim];
    Normalize(query, q);

    // Min-heap of (score, row_index) — keeps the top-k largest scores.
    using Pair = std::pair<float, std::size_t>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

    for (std::size_t i = 0; i < count_; ++i) {
      float score = DotProduct(q, &data_[i * kEmbeddingDim]);
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
      results.push_back({idx, row_to_id_[idx], s});
    }
    // Sort descending by score.
    std::sort(results.begin(), results.end(),
              [](const ScoredResult& a, const ScoredResult& b) {
                return a.score > b.score;
              });
    return results;
  }

  // -- Accessors --------------------------------------------------------------

  std::size_t size() const { return count_; }
  Metric metric() const { return metric_; }
  const float* raw_data() const { return data_.data(); }
  std::size_t raw_data_size() const { return data_.size(); }

  /// Look up the row index for an id, or -1 if not found.
  std::size_t RowOf(const std::string& id) const {
    auto it = id_to_row_.find(id);
    return it != id_to_row_.end() ? it->second : static_cast<std::size_t>(-1);
  }

  const std::string& IdOf(std::size_t row) const { return row_to_id_.at(row); }

  // -- Static helpers ---------------------------------------------------------

  /// Dot product of two kEmbeddingDim-length vectors.  Naive scalar loop.
  static float DotProduct(const float* a, const float* b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < kEmbeddingDim; ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /// L2-normalise `src` into `dst` (both kEmbeddingDim floats).
  static void Normalize(const float* src, float* dst) {
    float norm = 0.0f;
    for (std::size_t i = 0; i < kEmbeddingDim; ++i) {
      norm += src[i] * src[i];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-12f) {
      std::memset(dst, 0, kEmbeddingDim * sizeof(float));
      return;
    }
    float inv = 1.0f / norm;
    for (std::size_t i = 0; i < kEmbeddingDim; ++i) {
      dst[i] = src[i] * inv;
    }
  }

 private:
  void CopyAndNormalize(const float* src, float* dst) {
    Normalize(src, dst);
  }

  Metric metric_;
  std::size_t count_ = 0;
  std::vector<float> data_;                              // [N × kEmbeddingDim]
  std::unordered_map<std::string, std::size_t> id_to_row_;
  std::vector<std::string> row_to_id_;
};

}  // namespace vecidx
