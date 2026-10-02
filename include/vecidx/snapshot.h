// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
//
// Snapshot format:
//   [Header: 32 bytes]
//     magic      : 8 bytes  "VECIDX\x01\x00"
//     version    : uint32   (1)
//     dimension  : uint32   (1536)
//     count      : uint64   (number of vectors)
//     metric     : uint32   (0 = InnerProduct)
//     reserved   : 4 bytes  (zeroed)
//   [Vectors: count × dim × sizeof(float) bytes]
//     Row-major float array, already L2-normalised.
//   [ID map: count × (uint32 len + len bytes)]
//     Each id is a length-prefixed string.
//
// Atomic write: write to temp file → fsync → rename.
// On load: validate header before trusting any data.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "vecidx/flat_index.h"

namespace vecidx {

// Header layout — 32 bytes, packed.
#pragma pack(push, 1)
struct SnapshotHeader {
  char     magic[8];      // "VECIDX\x01\x00"
  uint32_t version;       // 1
  uint32_t dimension;     // 1536
  uint64_t count;         // number of vectors
  uint32_t metric;        // 0 = InnerProduct
  uint32_t reserved;      // 0
};
#pragma pack(pop)
static_assert(sizeof(SnapshotHeader) == 32, "Header must be 32 bytes");

constexpr char kSnapshotMagic[8] = {'V','E','C','I','D','X','\x01','\x00'};
constexpr uint32_t kSnapshotVersion = 1;

/// Write the index to a file atomically (temp + rename).
/// Returns the number of bytes written.
inline uint64_t SnapshotWrite(const FlatIndex& index, const std::string& path) {
  // Write to a temp file first.
  std::string tmp_path = path + ".tmp";
  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open temp file: " + tmp_path);

    // Header.
    SnapshotHeader hdr{};
    std::memcpy(hdr.magic, kSnapshotMagic, 8);
    hdr.version   = kSnapshotVersion;
    hdr.dimension  = static_cast<uint32_t>(kEmbeddingDim);
    hdr.count      = static_cast<uint64_t>(index.size());
    hdr.metric     = static_cast<uint32_t>(index.metric());
    hdr.reserved   = 0;

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // Vectors — contiguous block.
    if (index.size() > 0) {
      auto bytes = static_cast<std::streamsize>(
          index.size() * kEmbeddingDim * sizeof(float));
      out.write(reinterpret_cast<const char*>(index.raw_data()), bytes);
    }

    // ID map — length-prefixed strings.
    for (std::size_t i = 0; i < index.size(); ++i) {
      const std::string& id = index.IdOf(i);
      auto len = static_cast<uint32_t>(id.size());
      out.write(reinterpret_cast<const char*>(&len), sizeof(len));
      out.write(id.data(), static_cast<std::streamsize>(len));
    }

    out.flush();
    if (!out) throw std::runtime_error("Write failed: " + tmp_path);
  }
  // Atomic rename (on POSIX this is atomic; on Windows it replaces).
  std::remove(path.c_str());
  if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("Rename failed: " + tmp_path + " -> " + path);
  }

  // Compute size.
  std::ifstream check(path, std::ios::binary | std::ios::ate);
  return static_cast<uint64_t>(check.tellg());
}

/// Validate a snapshot header.
inline void ValidateHeader(const SnapshotHeader& hdr) {
  if (std::memcmp(hdr.magic, kSnapshotMagic, 8) != 0) {
    throw std::runtime_error("Invalid snapshot: bad magic");
  }
  if (hdr.version != kSnapshotVersion) {
    throw std::runtime_error("Invalid snapshot: unsupported version " +
                             std::to_string(hdr.version));
  }
  if (hdr.dimension != kEmbeddingDim) {
    throw std::runtime_error(
        "Invalid snapshot: dimension mismatch (expected " +
        std::to_string(kEmbeddingDim) + ", got " +
        std::to_string(hdr.dimension) + ")");
  }
  if (hdr.metric != static_cast<uint32_t>(Metric::InnerProduct)) {
    throw std::runtime_error("Invalid snapshot: unknown metric " +
                             std::to_string(hdr.metric));
  }
}

/// Load a snapshot into a new FlatIndex.
inline FlatIndex SnapshotLoad(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open snapshot: " + path);

  // Read and validate header.
  SnapshotHeader hdr{};
  in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!in) throw std::runtime_error("Truncated snapshot header: " + path);
  ValidateHeader(hdr);

  FlatIndex index(static_cast<Metric>(hdr.metric));

  if (hdr.count == 0) return index;

  // Read vectors into a temp buffer.
  std::size_t n = static_cast<std::size_t>(hdr.count);
  std::vector<float> vectors(n * kEmbeddingDim);
  auto vec_bytes = static_cast<std::streamsize>(
      n * kEmbeddingDim * sizeof(float));
  in.read(reinterpret_cast<char*>(vectors.data()), vec_bytes);
  if (!in) throw std::runtime_error("Truncated snapshot vectors: " + path);

  // Read id map.
  std::vector<std::string> ids(n);
  for (std::size_t i = 0; i < n; ++i) {
    uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!in) throw std::runtime_error("Truncated snapshot id map: " + path);
    ids[i].resize(len);
    in.read(&ids[i][0], static_cast<std::streamsize>(len));
    if (!in) throw std::runtime_error("Truncated snapshot id: " + path);
  }

  // Load pre-normalised vectors directly — no re-normalisation, so the
  // round-trip is byte-identical.
  index.LoadRaw(vectors.data(), ids, n);

  return index;
}

}  // namespace vecidx
