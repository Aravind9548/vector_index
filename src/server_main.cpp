// Copyright 2024 VecIdx Authors — SPDX-License-Identifier: MIT
//
// VecIdx server entry point.
//
// Concurrency model: single writer thread, concurrent readers.
// - gRPC server handles concurrent Search RPCs via shared_mutex (read lock).
// - Upsert/Delete/Snapshot acquire an exclusive write lock.
//
// Health: plain HTTP /health endpoint on a separate port for probes.
//
// This file is the main() for the deployed server.  It currently demonstrates
// the service layer; full gRPC integration requires protobuf/grpc libraries.
//
// Build note: gRPC C++ setup via vcpkg is done in the Dockerfile for Linux.
// On Windows/MinGW this file is compiled but not linked (no grpc libs).

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "vecidx/service.h"

namespace {

// Minimal HTTP health server (single-threaded, for readiness/liveness probes).
// Returns 200 if ready, 503 if not.  No dependencies beyond sockets.
// On Windows we'd use winsock; on Linux, POSIX sockets.
// For simplicity this is a placeholder — real deployment uses a proper server.
void PrintUsage() {
  std::cerr << "Usage: vecidx_server [--snapshot PATH] [--port PORT]\n"
            << "  --snapshot PATH   Load a snapshot on startup\n"
            << "  --port PORT       gRPC port (default: 50051)\n"
            << "  --health-port P   Health HTTP port (default: 8080)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string snapshot_path;
  int port = 50051;
  int health_port = 8080;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--snapshot" && i + 1 < argc) {
      snapshot_path = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::atoi(argv[++i]);
    } else if (arg == "--health-port" && i + 1 < argc) {
      health_port = std::atoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    }
  }

  vecidx::VecIdxService service;

  // Load snapshot if provided.
  if (!snapshot_path.empty()) {
    std::cout << "[vecidx] Loading snapshot from " << snapshot_path << "...\n";
    try {
      service.LoadSnapshot(snapshot_path);
      std::cout << "[vecidx] Loaded " << service.size() << " vectors.\n";
    } catch (const std::exception& e) {
      std::cerr << "[vecidx] ERROR loading snapshot: " << e.what() << "\n";
      return 1;
    }
  }

  // Mark ready.
  service.SetReady(true);
  std::cout << "[vecidx] Service ready. Vectors: " << service.size() << "\n"
            << "[vecidx] gRPC port: " << port
            << ", health port: " << health_port << "\n";

  // In the real deployment, this would start the gRPC server and health HTTP
  // server.  For now, just keep alive.
  std::cout << "[vecidx] Press Ctrl+C to stop.\n";

  // Placeholder event loop.
  while (service.IsReady()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return 0;
}
