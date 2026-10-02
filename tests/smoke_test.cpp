// Smoke test — proves the toolchain, GoogleTest, and the core header all link.
#include <gtest/gtest.h>
#include "vecidx/flat_index.h"

TEST(SmokeTest, DimensionIsLocked) {
  EXPECT_EQ(vecidx::kEmbeddingDim, 1536u);
}

TEST(SmokeTest, MetricIsInnerProduct) {
  EXPECT_EQ(static_cast<int>(vecidx::Metric::InnerProduct), 0);
}
