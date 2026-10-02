# Postmortem #1: Corrupt Snapshot on Crash

## Summary
A crash or power loss during snapshot write produces a corrupt or partial file.
On the next startup, the server fails to load the index and refuses to become ready.

## Timeline
1. Server calls `SnapshotWrite()` during a periodic save.
2. Writer is mid-way through the vector data section.
3. Process crashes (OOM, SIGKILL, power loss).
4. Partial file remains at the snapshot path.
5. On restart, `SnapshotLoad()` reads the truncated file.
6. Header says `count = N` but only partial vectors are present.
7. **Result: `std::runtime_error("Truncated snapshot vectors")`** — server won't start.

## Root Cause
The initial implementation wrote directly to the final path.  Any interruption
left a corrupt file that could not be recovered.

## Fix
**Atomic write pattern:**
```
1. Write to PATH.tmp
2. Flush (fsync on Linux)
3. Rename PATH.tmp → PATH (atomic on POSIX, replace on Windows)
```

If a crash happens during step 1 or 2, the `.tmp` file is corrupt but the
original snapshot at `PATH` is untouched.  On the next write, the `.tmp` file
is overwritten.

**Header validation on load:**
- Magic bytes: `VECIDX\x01\x00`
- Version check (reject unknown versions)
- Dimension match (1536 — catches loading a snapshot from a different model)
- Metric match (InnerProduct)

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| Crash during write | Corrupt file, server won't start | Previous good snapshot preserved |
| Wrong-model snapshot | Silent data corruption | Load fails with clear error |
| Truncated file | Undefined behaviour (read garbage) | `std::runtime_error` with reason |

## Lesson
Always write to a temp file first.  "Atomic rename" is the standard pattern
for any persistent state that must survive crashes.

## Code
- `SnapshotWrite()` in [snapshot.h](../include/vecidx/snapshot.h)
- Tests: `SnapshotTest.AtomicWriteNoCorruptionOnRename`, `SnapshotTest.CorruptMagicThrows`
