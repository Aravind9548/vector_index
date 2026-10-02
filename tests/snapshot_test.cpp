// Phase 2 — Snapshot persistence tests.
// Gate: write → read → byte-identical.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "vecidx/snapshot.h"

namespace vecidx {
namespace {

void FillRandom(float* v, std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < dim; ++i) v[i] = dist(rng);
}

class SnapshotTest : public ::testing::Test {
 protected:
  void TearDown() override {
    std::remove(path_.c_str());
    std::remove((path_ + ".tmp").c_str());
  }
  std::string path_ = "test_snapshot.vecidx";
};

TEST_F(SnapshotTest, WriteAndLoadRoundTrip) {
  constexpr std::size_t N = 50;
  FlatIndex original;
  std::mt19937 rng(42);
  std::vector<float> v(kEmbeddingDim);

  for (std::size_t i = 0; i < N; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    original.Upsert("doc-" + std::to_string(i), v.data());
  }

  uint64_t bytes = SnapshotWrite(original, path_);
  EXPECT_GT(bytes, sizeof(SnapshotHeader));

  FlatIndex loaded = SnapshotLoad(path_);
  EXPECT_EQ(loaded.size(), original.size());

  // Every vector must match exactly.
  for (std::size_t i = 0; i < original.size(); ++i) {
    const std::string& orig_id = original.IdOf(i);
    std::size_t loaded_row = loaded.RowOf(orig_id);
    ASSERT_NE(loaded_row, static_cast<std::size_t>(-1))
        << "Missing id: " << orig_id;

    const float* orig_vec = &original.raw_data()[i * kEmbeddingDim];
    const float* load_vec = &loaded.raw_data()[loaded_row * kEmbeddingDim];
    for (std::size_t d = 0; d < kEmbeddingDim; ++d) {
      EXPECT_EQ(orig_vec[d], load_vec[d])
          << "Mismatch at id=" << orig_id << " dim=" << d;
    }
  }
}

TEST_F(SnapshotTest, ByteIdenticalSnapshots) {
  // Property test: write twice, files must be byte-identical.
  FlatIndex idx;
  std::mt19937 rng(99);
  std::vector<float> v(kEmbeddingDim);

  for (int i = 0; i < 20; ++i) {
    FillRandom(v.data(), kEmbeddingDim, rng);
    idx.Upsert("id-" + std::to_string(i), v.data());
  }

  std::string path1 = path_ + ".1";
  std::string path2 = path_ + ".2";
  SnapshotWrite(idx, path1);
  SnapshotWrite(idx, path2);

  // Read both files and compare.
  auto readFile = [](const std::string& p) -> std::vector<char> {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
  };
  auto d1 = readFile(path1);
  auto d2 = readFile(path2);
  EXPECT_EQ(d1.size(), d2.size());
  EXPECT_EQ(d1, d2);

  std::remove(path1.c_str());
  std::remove(path2.c_str());
}

TEST_F(SnapshotTest, EmptyIndexRoundTrip) {
  FlatIndex empty;
  SnapshotWrite(empty, path_);
  FlatIndex loaded = SnapshotLoad(path_);
  EXPECT_EQ(loaded.size(), 0u);
}

TEST_F(SnapshotTest, CorruptMagicThrows) {
  FlatIndex idx;
  std::mt19937 rng(1);
  std::vector<float> v(kEmbeddingDim);
  FillRandom(v.data(), kEmbeddingDim, rng);
  idx.Upsert("x", v.data());
  SnapshotWrite(idx, path_);

  // Corrupt the magic bytes.
  {
    std::fstream f(path_, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(0);
    f.write("CORRUPT!", 8);
  }
  EXPECT_THROW(SnapshotLoad(path_), std::runtime_error);
}

TEST_F(SnapshotTest, TruncatedFileThrows) {
  FlatIndex idx;
  std::mt19937 rng(2);
  std::vector<float> v(kEmbeddingDim);
  FillRandom(v.data(), kEmbeddingDim, rng);
  idx.Upsert("x", v.data());
  SnapshotWrite(idx, path_);

  // Truncate the file to just the header.
  {
    std::ofstream f(path_, std::ios::binary | std::ios::trunc);
    SnapshotHeader hdr{};
    std::memcpy(hdr.magic, kSnapshotMagic, 8);
    hdr.version = kSnapshotVersion;
    hdr.dimension = static_cast<uint32_t>(kEmbeddingDim);
    hdr.count = 1;  // Claims 1 vector but no data follows.
    hdr.metric = 0;
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  }
  EXPECT_THROW(SnapshotLoad(path_), std::runtime_error);
}

TEST_F(SnapshotTest, AtomicWriteNoCorruptionOnRename) {
  // Simulates the scenario: if .tmp exists from a crashed write,
  // a new write should succeed by overwriting it.
  {
    std::ofstream f(path_ + ".tmp", std::ios::binary);
    f << "garbage from a crashed write";
  }

  FlatIndex idx;
  std::mt19937 rng(3);
  std::vector<float> v(kEmbeddingDim);
  FillRandom(v.data(), kEmbeddingDim, rng);
  idx.Upsert("survive", v.data());

  EXPECT_NO_THROW(SnapshotWrite(idx, path_));
  FlatIndex loaded = SnapshotLoad(path_);
  EXPECT_EQ(loaded.size(), 1u);
  EXPECT_NE(loaded.RowOf("survive"), static_cast<std::size_t>(-1));
}

}  // namespace
}  // namespace vecidx
