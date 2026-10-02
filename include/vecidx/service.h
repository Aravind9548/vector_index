// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
//
// VecIdx service layer — the business logic behind the gRPC server.
// Concurrency model: single writer thread, concurrent readers.
//
// This layer is transport-agnostic; it can be driven by gRPC, HTTP, or tests.
#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vecidx/filtered_index.h"
#include "vecidx/snapshot.h"

namespace vecidx {

/// Thread-safe service wrapping FilteredIndex.
/// Reads acquire a shared lock; writes acquire an exclusive lock.
class VecIdxService {
 public:
  explicit VecIdxService(HnswParams params = {})
      : index_(params), ready_(false) {}

  // -- Upsert (write) ---------------------------------------------------------

  struct UpsertResult {
    uint32_t upserted_count;
  };

  UpsertResult Upsert(
      const std::vector<std::tuple<std::string, std::vector<float>,
                                    std::unordered_map<std::string, std::string>>>& vectors) {
    std::unique_lock lock(mu_);
    uint32_t count = 0;
    for (auto& [id, vals, meta] : vectors) {
      if (vals.size() != kEmbeddingDim) continue;
      index_.Upsert(id, vals.data(), meta);
      ++count;
    }
    return {count};
  }

  // -- Search (read) ----------------------------------------------------------

  std::vector<ScoredResult> Search(
      const float* query, std::size_t top_k,
      const std::unordered_map<std::string, std::string>& filter = {},
      FilterStrategy strategy = FilterStrategy::PostFilter) const {
    std::shared_lock lock(mu_);
    return index_.Search(query, top_k, filter, strategy);
  }

  // -- Delete (write) ---------------------------------------------------------

  struct DeleteResult {
    uint32_t deleted_count;
  };

  // Note: delete is not yet implemented in the underlying indexes.
  // This is a placeholder for Phase 5 completeness.
  DeleteResult Delete(const std::vector<std::string>& /*ids*/) {
    std::unique_lock lock(mu_);
    // TODO: implement when FlatIndex/HnswIndex support concurrent delete.
    return {0};
  }

  // -- Snapshot (write — blocks readers during snapshot) -----------------------

  struct SnapshotResult {
    uint64_t vector_count;
    uint64_t size_bytes;
  };

  SnapshotResult TakeSnapshot(const std::string& path) const {
    std::shared_lock lock(mu_);
    uint64_t bytes = SnapshotWrite(index_.flat(), path);
    return {static_cast<uint64_t>(index_.size()), bytes};
  }

  // -- Health / readiness -----------------------------------------------------

  bool IsReady() const { return ready_.load(std::memory_order_acquire); }
  void SetReady(bool r) { ready_.store(r, std::memory_order_release); }

  std::size_t size() const {
    std::shared_lock lock(mu_);
    return index_.size();
  }

  // -- Load snapshot on startup -----------------------------------------------

  void LoadSnapshot(const std::string& path) {
    std::unique_lock lock(mu_);
    auto flat = SnapshotLoad(path);
    // Re-insert into the filtered index from the loaded flat index.
    for (std::size_t i = 0; i < flat.size(); ++i) {
      index_.Upsert(flat.IdOf(i), &flat.raw_data()[i * kEmbeddingDim]);
    }
  }

 private:
  mutable std::shared_mutex mu_;
  FilteredIndex index_;
  std::atomic<bool> ready_;
};

}  // namespace vecidx
