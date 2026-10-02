# Postmortem #2: Cold Start — Stateful Index in Stateless Infra

## Summary
A fresh container must download and load the snapshot before serving queries.
If the readiness probe goes green before the index is loaded, the service
returns empty results, which propagates through the agent as "no relevant
documents found" — a silent failure.

## Timeline
1. Azure Container Apps scales up a new replica (or restarts after a crash).
2. Container starts, health endpoint comes up on port 8080.
3. **BUG (original):** Health returns 200 immediately.
4. Load balancer routes gRPC traffic to the new replica.
5. Queries arrive.  Index is empty (snapshot still loading).
6. All searches return 0 results.
7. Agent tells the user: "No relevant documents found for your query."
8. User sees no error — just wrong answers.  **Silent failure.**

## Root Cause
The readiness probe did not distinguish between "process is alive" and
"index is loaded and ready to serve."

A vector index is stateful — it needs data loaded into memory.  Container
infra is stateless — it assumes a container is ready once the process starts.
This mismatch is the core of the problem.

## Fix

### 1. Readiness probe gates on index load
```cpp
// Service starts with ready=false.
VecIdxService service;
// ready_ is initially false.

// Load snapshot.
service.LoadSnapshot(snapshot_path);

// NOW mark ready.
service.SetReady(true);
```

The `/health` endpoint returns:
- **503** if `IsReady() == false` (index not loaded)
- **200** if `IsReady() == true` (ready to serve)

### 2. Container Apps readiness probe config
```yaml
probes:
  - type: readiness
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 5
    failureThreshold: 60  # 5 minutes for large snapshots
```

The `failureThreshold: 60` × `periodSeconds: 5` = 300 seconds.  A 100K-vector
snapshot at 1536 dimensions is ~600 MB; on Azure Files with a Premium share,
download + load takes ~30–60 seconds.  The 5-minute budget handles larger
indexes or slower storage.

### 3. Liveness probe is separate
The liveness probe runs on a longer interval (30s) and checks only if the
process is alive — not if the index is loaded.  This prevents the container
from being killed during a long snapshot load.

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| Cold start with 100K vectors | Serves empty results for ~30s | Returns 503 until loaded |
| User experience | "No documents found" (wrong) | Query waits for ready replica |
| Scale-up event | Silent data loss window | Zero-downtime with min-replicas=1 |

## Lesson
A stateful service in stateless infra must gate readiness on state load.
The readiness probe is not "is the process running" — it's "can this replica
serve correct answers."

## Code
- `VecIdxService::IsReady()` / `SetReady()` in [service.h](../include/vecidx/service.h)
- Container config in [azure-container-app.yaml](../deploy/azure-container-app.yaml)
