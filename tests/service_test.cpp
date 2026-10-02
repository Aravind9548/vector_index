// Phase 5 — Service layer tests.
// Tests: thread-safe upsert/search, concurrent readers, snapshot.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "vecidx/service.h"

namespace vecidx {
namespace {

void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
}

TEST(ServiceTest, UpsertAndSearch) {
  VecIdxService svc;
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  std::vector<std::tuple<std::string, std::vector<float>,
                          std::unordered_map<std::string, std::string>>> batch;
  for (int i = 0; i < 50; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    batch.push_back({"doc-" + std::to_string(i),
                     std::vector<float>(v.begin(), v.end()),
                     {{"company_id", "acme"}}});
  }

  auto result = svc.Upsert(batch);
  EXPECT_EQ(result.upserted_count, 50u);
  EXPECT_EQ(svc.size(), 50u);

  FillRandom(v.data(), kEmbeddingDim, rng);
  auto results = svc.Search(v.data(), 10);
  EXPECT_EQ(results.size(), 10u);
}

TEST(ServiceTest, ConcurrentReaders) {
  VecIdxService svc;
  std::mt19937 rng(99);

  // Insert some data.
  std::vector<std::tuple<std::string, std::vector<float>,
                          std::unordered_map<std::string, std::string>>> batch;
  std::vector<float> v(kEmbeddingDim);
  for (int i = 0; i < 100; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    batch.push_back({"v-" + std::to_string(i),
                     std::vector<float>(v.begin(), v.end()), {}});
  }
  svc.Upsert(batch);

  // Spawn multiple reader threads.
  std::atomic<int> errors{0};
  constexpr int NUM_THREADS = 8;
  constexpr int QUERIES_PER_THREAD = 50;

  std::vector<std::thread> threads;
  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&svc, &errors, t]() {
      std::mt19937 trng(static_cast<unsigned>(t * 1000));
      std::vector<float> q(kEmbeddingDim);
      for (int i = 0; i < QUERIES_PER_THREAD; ++i) {
        FillRandom(q.data(), kEmbeddingDim, trng);
        auto results = svc.Search(q.data(), 5);
        if (results.size() != 5) {
          errors.fetch_add(1);
        }
      }
    });
  }

  for (auto& t : threads) t.join();
  EXPECT_EQ(errors.load(), 0)
      << "Concurrent reader failures: " << errors.load();
}

TEST(ServiceTest, ReadinessProbe) {
  VecIdxService svc;
  EXPECT_FALSE(svc.IsReady());
  svc.SetReady(true);
  EXPECT_TRUE(svc.IsReady());
  svc.SetReady(false);
  EXPECT_FALSE(svc.IsReady());
}

TEST(ServiceTest, SnapshotRoundTrip) {
  VecIdxService svc;
  std::mt19937 rng(77);
  std::vector<float> v(kEmbeddingDim);

  std::vector<std::tuple<std::string, std::vector<float>,
                          std::unordered_map<std::string, std::string>>> batch;
  for (int i = 0; i < 20; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    batch.push_back({"snap-" + std::to_string(i),
                     std::vector<float>(v.begin(), v.end()), {}});
  }
  svc.Upsert(batch);

  std::string path = "test_service_snapshot.vecidx";
  auto snap = svc.TakeSnapshot(path);
  EXPECT_EQ(snap.vector_count, 20u);
  EXPECT_GT(snap.size_bytes, 0u);

  // Load into a new service.
  VecIdxService svc2;
  svc2.LoadSnapshot(path);
  EXPECT_EQ(svc2.size(), 20u);

  std::remove(path.c_str());
}

TEST(ServiceTest, ConcurrentReadAndWrite) {
  // One writer + multiple readers.  TSan should catch data races.
  VecIdxService svc;

  // Pre-populate.
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);
  {
    std::vector<std::tuple<std::string, std::vector<float>,
                            std::unordered_map<std::string, std::string>>> batch;
    for (int i = 0; i < 50; ++i) {
      FillRandom(v.data(), kEmbeddingDim, rng);
      batch.push_back({"init-" + std::to_string(i),
                       std::vector<float>(v.begin(), v.end()), {}});
    }
    svc.Upsert(batch);
  }

  std::atomic<bool> stop{false};
  std::atomic<int> read_errors{0};
  std::atomic<int> writes{0};

  // Writer thread.
  std::thread writer([&]() {
    std::mt19937 wrng(1000);
    for (int i = 0; i < 20 && !stop; ++i) {
      FillRandom(v.data(), kEmbeddingDim, wrng);
      std::vector<std::tuple<std::string, std::vector<float>,
                              std::unordered_map<std::string, std::string>>> batch;
      batch.push_back({"write-" + std::to_string(i),
                       std::vector<float>(v.begin(), v.end()), {}});
      svc.Upsert(batch);
      writes.fetch_add(1);
    }
  });

  // Reader threads.
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&, t]() {
      std::mt19937 rrng(static_cast<unsigned>(t * 100));
      std::vector<float> q(kEmbeddingDim);
      for (int i = 0; i < 30; ++i) {
        FillRandom(q.data(), kEmbeddingDim, rrng);
        try {
          auto results = svc.Search(q.data(), 5);
          // Results may vary as writer adds vectors — that's fine.
        } catch (...) {
          read_errors.fetch_add(1);
        }
      }
    });
  }

  writer.join();
  for (auto& r : readers) r.join();

  EXPECT_EQ(read_errors.load(), 0);
  EXPECT_GT(writes.load(), 0);
}

}  // namespace
}  // namespace vecidx
