// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace vecidx {

/// Locked contract — do not change without a migration.
constexpr std::size_t kEmbeddingDim = 1536;

/// Distance metric.  We store L2-normalised vectors and compute inner product,
/// which equals cosine similarity.  Higher = more similar.
enum class Metric : std::uint8_t {
  InnerProduct = 0,
};

}  // namespace vecidx
