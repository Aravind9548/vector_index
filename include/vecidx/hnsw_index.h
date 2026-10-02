// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
//
// HNSW (Hierarchical Navigable Small World) index.
// Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
//            neighbor search using Hierarchical Navigable Small World graphs",
//            IEEE TPAMI 2020.
//
// Single-threaded implementation.  Concurrency comes later (Phase 5).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vecidx/flat_index.h"  // for kEmbeddingDim, Metric, Normalize, DotProduct

namespace vecidx {

/// HNSW configuration.
struct HnswParams {
  std::size_t M              = 16;   // max edges per node per layer
  std::size_t M0             = 32;   // max edges at layer 0 (typically 2*M)
  std::size_t ef_construction = 200; // beam width during insertion
  std::size_t ef_search       = 50;  // beam width during search (tunable)
};

/// HNSW index — approximate nearest-neighbour search.
class HnswIndex {
 public:
  explicit HnswIndex(HnswParams params = {},
                     Metric metric = Metric::InnerProduct,
                     uint32_t seed = 42)
      : params_(params),
        metric_(metric),
        rng_(seed),
        ml_(1.0 / std::log(static_cast<double>(params.M))) {}

  // -- Mutation ---------------------------------------------------------------

  void Upsert(const std::string& id, const float* vec) {
    auto it = id_to_node_.find(id);
    if (it != id_to_node_.end()) {
      // Replace vector in-place (does NOT rebuild graph edges — acceptable
      // for small drift; full rebuild needed for large changes).
      std::size_t node = it->second;
      float* dst = &vectors_[node * kEmbeddingDim];
      FlatIndex::Normalize(vec, dst);
      return;
    }

    std::size_t new_id = node_count_;
    id_to_node_[id] = new_id;
    node_to_id_.push_back(id);

    // Store normalised vector.
    vectors_.resize((node_count_ + 1) * kEmbeddingDim);
    FlatIndex::Normalize(vec, &vectors_[new_id * kEmbeddingDim]);

    // Assign a random level.
    int level = RandomLevel();
    node_levels_.push_back(level);

    // Ensure adjacency lists exist for this node at all its levels.
    while (static_cast<int>(adj_.size()) <= level) {
      adj_.emplace_back();
    }
    for (int l = 0; l <= level; ++l) {
      if (adj_[static_cast<std::size_t>(l)].size() <= new_id) {
        adj_[static_cast<std::size_t>(l)].resize(new_id + 1);
      }
    }

    ++node_count_;

    if (node_count_ == 1) {
      // First node — it becomes the entry point.
      entry_point_ = new_id;
      max_level_ = level;
      return;
    }

    // Insert into the graph (Algorithm 1 from the paper).
    std::size_t ep = entry_point_;

    // Phase 1: greedily descend from the top level to level+1.
    for (int l = max_level_; l > level; --l) {
      ep = GreedyClosest(new_id, ep, static_cast<std::size_t>(l));
    }

    // Phase 2: insert at each layer from min(level, max_level_) down to 0.
    for (int l = std::min(level, max_level_); l >= 0; --l) {
      auto layer = static_cast<std::size_t>(l);
      std::size_t max_edges = (l == 0) ? params_.M0 : params_.M;

      // Search for ef_construction nearest neighbours at this layer.
      auto candidates = SearchLayer(new_id, ep, params_.ef_construction, layer);

      // Select M nearest as neighbours.
      auto neighbours = SelectNeighbours(candidates, max_edges);

      // Add bidirectional edges.
      for (auto [score, nb] : neighbours) {
        adj_[layer][new_id].push_back(nb);
        adj_[layer][nb].push_back(new_id);

        // Prune neighbour's edges if over limit.
        if (adj_[layer][nb].size() > max_edges) {
          PruneEdges(nb, layer, max_edges);
        }
      }

      // Use the closest found as entry point for the next layer down.
      if (!candidates.empty()) {
        // candidates is a max-heap; find the element with highest score.
        float best = -std::numeric_limits<float>::infinity();
        for (auto& [s, n] : candidates) {
          if (s > best) { best = s; ep = n; }
        }
      }
    }

    // Update entry point if new node has a higher level.
    if (level > max_level_) {
      entry_point_ = new_id;
      max_level_ = level;
    }
  }

  // -- Query ------------------------------------------------------------------

  /// Search for top-k approximate nearest neighbours.
  std::vector<ScoredResult> Search(const float* query, std::size_t top_k) const {
    return SearchWithEf(query, top_k, params_.ef_search);
  }

  /// Search with a custom ef value (for recall tuning).
  std::vector<ScoredResult> SearchWithEf(const float* query, std::size_t top_k,
                                          std::size_t ef) const {
    if (node_count_ == 0 || top_k == 0) return {};

    // Normalise query.
    float q[kEmbeddingDim];
    FlatIndex::Normalize(query, q);

    // Store query temporarily so Distance() can use it.
    query_vec_ = q;

    std::size_t ep = entry_point_;

    // Greedy descent from top to layer 1.
    for (int l = max_level_; l > 0; --l) {
      ep = GreedyClosest(q, ep, static_cast<std::size_t>(l));
    }

    // Search at layer 0 with the given ef.
    auto candidates = SearchLayerQuery(q, ep, std::max(ef, top_k), 0);

    // Extract top-k.
    // candidates is a vector of (score, node). Sort descending.
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (candidates.size() > top_k) candidates.resize(top_k);

    std::vector<ScoredResult> results;
    results.reserve(candidates.size());
    for (auto& [score, node] : candidates) {
      results.push_back({node, node_to_id_[node], score});
    }
    return results;
  }

  // -- Accessors --------------------------------------------------------------

  std::size_t size() const { return node_count_; }
  void set_ef_search(std::size_t ef) { params_.ef_search = ef; }
  std::size_t ef_search() const { return params_.ef_search; }
  const HnswParams& params() const { return params_; }
  const float* raw_data() const { return vectors_.data(); }

  const std::string& IdOf(std::size_t node) const {
    return node_to_id_.at(node);
  }

  std::size_t NodeOf(const std::string& id) const {
    auto it = id_to_node_.find(id);
    return it != id_to_node_.end() ? it->second
                                   : static_cast<std::size_t>(-1);
  }

 private:
  // -- Distance ---------------------------------------------------------------

  float Distance(std::size_t a, std::size_t b) const {
    return FlatIndex::DotProduct(&vectors_[a * kEmbeddingDim],
                                 &vectors_[b * kEmbeddingDim]);
  }

  float Distance(const float* q, std::size_t b) const {
    return FlatIndex::DotProduct(q, &vectors_[b * kEmbeddingDim]);
  }

  // -- Random level -----------------------------------------------------------

  int RandomLevel() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    return static_cast<int>(-std::log(r) * ml_);
  }

  // -- Greedy search helpers --------------------------------------------------

  /// Greedy walk to the closest node to `target` starting from `ep` at `layer`.
  std::size_t GreedyClosest(std::size_t target, std::size_t ep,
                             std::size_t layer) const {
    float best = Distance(target, ep);
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t nb : adj_[layer][ep]) {
        float d = Distance(target, nb);
        if (d > best) {
          best = d;
          ep = nb;
          changed = true;
        }
      }
    }
    return ep;
  }

  /// Greedy walk for a raw query vector.
  std::size_t GreedyClosest(const float* q, std::size_t ep,
                             std::size_t layer) const {
    float best = Distance(q, ep);
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t nb : adj_[layer][ep]) {
        float d = Distance(q, nb);
        if (d > best) {
          best = d;
          ep = nb;
          changed = true;
        }
      }
    }
    return ep;
  }

  // -- Layer search (Algorithm 2 from paper) ----------------------------------

  using ScoreNode = std::pair<float, std::size_t>;

  /// Search at a layer for ef nearest to node `target`.
  std::vector<ScoreNode> SearchLayer(std::size_t target, std::size_t ep,
                                      std::size_t ef,
                                      std::size_t layer) const {
    float d_ep = Distance(target, ep);
    // candidates: max-heap (largest score = closest in IP)
    // visited set
    std::unordered_set<std::size_t> visited;
    visited.insert(ep);

    // W: result set, min-heap (smallest score on top — the worst result)
    std::priority_queue<ScoreNode, std::vector<ScoreNode>,
                        std::greater<ScoreNode>> W;
    W.push({d_ep, ep});

    // C: candidate set, max-heap (largest score on top — best candidate)
    std::priority_queue<ScoreNode> C;
    C.push({d_ep, ep});

    while (!C.empty()) {
      auto [c_score, c_node] = C.top();
      C.pop();

      // If the best candidate is worse than the worst result, stop.
      if (c_score < W.top().first && W.size() >= ef) break;

      // Expand neighbours.
      if (layer < adj_.size() && c_node < adj_[layer].size()) {
        for (std::size_t nb : adj_[layer][c_node]) {
          if (visited.count(nb)) continue;
          visited.insert(nb);

          float d = Distance(target, nb);
          if (W.size() < ef || d > W.top().first) {
            C.push({d, nb});
            W.push({d, nb});
            if (W.size() > ef) W.pop();
          }
        }
      }
    }

    // Extract W into a vector.
    std::vector<ScoreNode> result;
    result.reserve(W.size());
    while (!W.empty()) {
      result.push_back(W.top());
      W.pop();
    }
    return result;
  }

  /// Search at a layer for ef nearest to raw query vector.
  std::vector<ScoreNode> SearchLayerQuery(const float* q, std::size_t ep,
                                           std::size_t ef,
                                           std::size_t layer) const {
    float d_ep = Distance(q, ep);
    std::unordered_set<std::size_t> visited;
    visited.insert(ep);

    std::priority_queue<ScoreNode, std::vector<ScoreNode>,
                        std::greater<ScoreNode>> W;
    W.push({d_ep, ep});

    std::priority_queue<ScoreNode> C;
    C.push({d_ep, ep});

    while (!C.empty()) {
      auto [c_score, c_node] = C.top();
      C.pop();

      if (c_score < W.top().first && W.size() >= ef) break;

      if (layer < adj_.size() && c_node < adj_[layer].size()) {
        for (std::size_t nb : adj_[layer][c_node]) {
          if (visited.count(nb)) continue;
          visited.insert(nb);

          float d = Distance(q, nb);
          if (W.size() < ef || d > W.top().first) {
            C.push({d, nb});
            W.push({d, nb});
            if (W.size() > ef) W.pop();
          }
        }
      }
    }

    std::vector<ScoreNode> result;
    result.reserve(W.size());
    while (!W.empty()) {
      result.push_back(W.top());
      W.pop();
    }
    return result;
  }

  // -- Neighbour selection (simple: just keep closest M) ----------------------

  std::vector<ScoreNode> SelectNeighbours(
      const std::vector<ScoreNode>& candidates, std::size_t M) const {
    auto sorted = candidates;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (sorted.size() > M) sorted.resize(M);
    return sorted;
  }

  // -- Edge pruning -----------------------------------------------------------

  void PruneEdges(std::size_t node, std::size_t layer, std::size_t max_edges) {
    auto& edges = adj_[layer][node];
    if (edges.size() <= max_edges) return;

    // Score each neighbour and keep the closest max_edges.
    std::vector<ScoreNode> scored;
    scored.reserve(edges.size());
    for (std::size_t nb : edges) {
      scored.push_back({Distance(node, nb), nb});
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    scored.resize(max_edges);

    edges.clear();
    edges.reserve(max_edges);
    for (auto& [s, n] : scored) {
      edges.push_back(n);
    }
  }

  // -- State ------------------------------------------------------------------

  HnswParams params_;
  Metric metric_;
  std::mt19937 rng_;
  double ml_;  // level generation factor = 1/ln(M)

  std::size_t node_count_ = 0;
  std::size_t entry_point_ = 0;
  int max_level_ = 0;

  std::vector<float> vectors_;  // [N × kEmbeddingDim]
  std::vector<int> node_levels_;
  std::unordered_map<std::string, std::size_t> id_to_node_;
  std::vector<std::string> node_to_id_;

  // adj_[layer][node] = vector of neighbour node ids
  std::vector<std::vector<std::vector<std::size_t>>> adj_;

  // Temporary for query — mutable so Search can be const.
  mutable const float* query_vec_ = nullptr;
};

}  // namespace vecidx
