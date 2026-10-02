# VecIdx server — multi-stage build for Azure Container Apps.
# Deploys in koreacentral alongside the agent's existing resources.

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECIDX_BUILD_BENCHMARKS=OFF \
    -DVECIDX_BUILD_TESTS=OFF \
    && cmake --build build -j$(nproc)

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /src/build/vecidx_server /app/vecidx_server

# Snapshot directory (mount Azure Blob Storage here).
RUN mkdir -p /data/snapshots

# Health check — the readiness probe must not go green until the index is loaded.
# Cold start: container downloads + loads snapshot before becoming ready.
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

EXPOSE 50051 8080

# Entrypoint: load snapshot if SNAPSHOT_PATH is set.
ENTRYPOINT ["/app/vecidx_server"]
CMD ["--port", "50051", "--health-port", "8080"]
