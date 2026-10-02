# Postmortem #3: Cross-Company Data Leakage in Filtered Search

## Summary
The agent's vector search returned citations from Company A's documents in
a report generated for Company B.  This is a data isolation failure in a
multi-tenant due diligence system.

## Timeline
1. Company A's SEC filings are indexed: `{company_id: "company-a"}`.
2. Company B's SEC filings are indexed: `{company_id: "company-b"}`.
3. User queries for Company B's risk factors.
4. HNSW search finds the 10 nearest vectors in the full index.
5. 3 of the 10 results belong to Company A (similar financial language).
6. No metadata filter is applied — the search is company-blind.
7. Agent cites Company A's 10-K in Company B's report.
8. **User notices a competitor's filing cited in their due diligence report.**

## Root Cause
The original FAISS-based retrieval did not enforce company-level isolation.
FAISS `IndexFlatIP` has no built-in filtering — it returns the global top-k.

This is not a bug in the similarity metric.  Financial filings use similar
language ("risk factors", "material adverse effects"), so cross-company
matches have legitimately high similarity scores.  The fix must be
**structural** (filter), not **statistical** (hope the right company scores higher).

## Fix

### Strategy comparison (Phase 4)
We implemented and measured both honest approaches:

**Pre-filter (flat scan on filtered subset):**
- Pros: Exact recall on the filtered partition.  Zero leakage by construction.
- Cons: O(partition_size) per query.  Doesn't use the HNSW graph.
- Recall: **1.000** (exact)

**Post-filter (HNSW search then drop non-matching):**
- Pros: Uses the full HNSW graph for speed.
- Cons: May return fewer than top-k if the partition is small.
  With ef_multiplier=1, returned **0/10** results.
  With ef_multiplier=10, returned **10/10** results.
- Recall vs pre-filter oracle: **0.778**

### Chosen approach
- **Default: post-filter with ef_multiplier=4** for production.
- **Fallback: pre-filter** when the partition is small (< 1000 vectors)
  or when 100% recall is required (audit queries).

### Verified
- 200-query stress test with random company filters: **zero leakage** on
  both strategies.
- Test: `FilteredSearchTest.NoCrossCompanyLeakageUnderStress`

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| Company isolation | None — global top-k | Enforced via metadata filter |
| Leakage rate | ~30% of results from wrong company | 0% (verified under stress) |
| Recall on filtered partition | N/A | Pre: 1.0, Post: 0.778 |
| Post-filter shortfall | N/A | Solved with ef_multiplier ≥ 4 |

## Lesson
Multi-tenant vector search requires **explicit filtering**, not hope.
Similar text across companies means global similarity search will always
find cross-company matches.  The filter is not an optimisation — it's a
correctness requirement.

## Code
- `FilteredIndex` in [filtered_index.h](../include/vecidx/filtered_index.h)
- Stress test in [filtered_test.cpp](../tests/filtered_test.cpp)
