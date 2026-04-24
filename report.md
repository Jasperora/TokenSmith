**Project:** Cache Retention Policy for `EmbeddingCache` in TokenSmith  

[Github repo link](https://github.com/Jasperora/TokenSmith)

**1) Re-iteration of Proposed Goals and Progress to Date**

My proposal defined three target levels:

- **75% goal:** implement a cache retention policy in `EmbeddingCache` so cache size does not grow unbounded.
- **100% goal:** validate correctness and evaluate performance impact.
- **125% goal:** extend beyond static policy with more advanced retention strategies.

**Progress against goals**

- **75% goal: completed.**  
  I implemented retention controls in `EmbeddingCache`:
  - `ttl_days` for time-based expiration
  - `max_rows` for capacity bound
  - `prune_every_writes` for demand-triggered pruning  
  Pruning now runs on initialization and periodically on write path.

- **100% goal: completed.**  
  I added correctness tests and a repeatable performance benchmark:
  - `tests/test_embedding_cache.py`
  - `tests/test_cache_config.py`
  - `tests/test_cache_retention_perf.py`  
  The benchmark now uses fixed random seeds and reports aggregate statistics over multiple trials.

- **125% goal: partially completed.**  
  I added an additional policy dimension and stricter semantics:
  - `cache_strict_ttl_on_read` (prevents stale entries from being returned during reads)
  - `cache_eviction_policy` with `fifo` and `lru` modes  
  Full adaptive policies (for example LFU or auto-tuned hybrid policies) are not yet implemented.

In summary, the planned core implementation and evaluation are complete, and part of the stretch objective has been delivered.

**2) How I Tested Correctness**

I tested correctness with focused unit tests that validate cache behavior directly at the data level.

**A. Functional correctness tests**

- **Read/write roundtrip:** verifies that an embedding inserted into cache is retrieved correctly as the same `float32` vector.
- **TTL prune behavior:** inserts both old and fresh rows and confirms pruning removes only expired entries.
- **Max-row prune behavior:** inserts above capacity and confirms final row count is capped.
- **Strict TTL on read:** verifies stale entries are rejected by `get()` when strict read filtering is enabled.
- **LRU eviction behavior:** manipulates access times and confirms recently accessed entries are retained under `lru`.
- **Config validation:** verifies cache config is mapped correctly from `RAGConfig` and invalid eviction policy values raise an error.

**B. Test run and status**

I executed:

```bash
python3 -m pytest -q \
  tests/test_cache_config.py \
  tests/test_embedding_cache.py \
  tests/test_cache_retention_perf.py \
  --output-mode terminal
```

Result: **8 passed** (all cache-focused tests passing).

These tests provide evidence that the implemented policy behaves as intended under controlled conditions.

**3) Experimental Results Collected to Evaluate the Implementation**

To evaluate impact, I compared:

- **Baseline:** no TTL, no row cap (effectively no retention)
- **Policy:** `ttl_days=30`, `max_rows=6000`, `prune_every_writes=500`

Workload setup:

- inserts: 10,000 unique entries
- reads: 5,000 mixed hot/cold accesses
- embedding dimension: 384
- trials with fixed seeds: 7, 11, 23

**Aggregate results (mean +/- std across 3 trials)**

| Metric | Baseline | Policy |
|---|---:|---:|
| Hit rate | 0.8097 +/- 0.0056 | 0.5015 +/- 0.0065 |
| Relative hit rate (policy/baseline) | - | 0.6193 |
| Rows after prune | 10,000 | 2,000 |
| DB size after prune | 21,708,800 bytes | 14,155,776 bytes |
| Insert time | 3.5301 +/- 0.0855 s | 3.5356 +/- 0.1015 s |
| Read time | 1.3878 +/- 0.0830 s | 1.0435 +/- 0.0371 s |

**Interpretation**

The retention policy achieved the primary system goal: bounded cache growth with significantly lower stored data volume.

- Row count reduced from 10,000 to 2,000 (**80% reduction**).
- File size reduced from ~21.7 MB to ~14.2 MB (**about 35% reduction**).
- Insert latency remained similar to baseline, indicating retention overhead is manageable.

The main tradeoff is hit rate reduction (0.81 to 0.50) under this synthetic workload. This is expected when enforcing stronger retention constraints and indicates policy tuning should be workload-dependent. The result is still useful because it quantifies the explicit memory/utility tradeoff and provides a foundation for better policy tuning.

**4) Concrete Future Work to Expand/Improve the Implementation**

1. **Adaptive retention tuning**  
   Automatically adjust `ttl_days` and `max_rows` based on observed hit rate and storage targets, rather than fixed static values.

2. **Additional eviction policies**  
   Implement LFU and hybrid TTL+LRU/LFU policies, then compare them against current `fifo` and `lru` using the same benchmark harness.
