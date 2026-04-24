import sqlite3
import time
from pathlib import Path
import random
import statistics

import numpy as np

from src.embedder import EmbeddingCache


def _db_stats(db_path: Path) -> tuple[int, int]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    size = db_path.stat().st_size if db_path.exists() else 0
    return rows, size


def _age_old_rows(db_path: Path, n_rows_to_age: int, days_old: int = 90) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE embeddings
            SET timestamp = datetime('now', ?)
            WHERE rowid IN (
                SELECT rowid
                FROM embeddings
                ORDER BY rowid ASC
                LIMIT ?
            )
            """,
            (f"-{int(days_old)} days", int(n_rows_to_age)),
        )
        conn.commit()


def _run_case(
    cache: EmbeddingCache,
    model_path: str,
    n_insert: int,
    n_get: int,
    embedding_dim: int,
    hot_window: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    vec = np.ones((embedding_dim,), dtype=np.float32)

    # 1) Insert unique keys
    t0 = time.perf_counter()
    for i in range(n_insert):
        cache.set(model_path, f"q_{i}", vec)
    t1 = time.perf_counter()

    rows_before_prune, size_before_prune = _db_stats(cache.db_path)

    # 2) Age only stale tail, keep newest hot_window rows fresh
    hot_window = min(hot_window, rows_before_prune)
    stale_rows = max(0, rows_before_prune - hot_window)
    if stale_rows > 0:
        _age_old_rows(cache.db_path, stale_rows, days_old=90)

    # 3) Explicit prune timing
    t2 = time.perf_counter()
    cache.prune()
    t3 = time.perf_counter()

    rows_after_prune, size_after_prune = _db_stats(cache.db_path)

    # 4) Read workload: hits target recent/hot keys that should survive TTL
    hits = 0
    base_key = max(0, n_insert - hot_window)
    hot_keys = [f"q_{base_key + i}" for i in range(hot_window)]
    cold_keys = [f"q_{i}" for i in range(base_key)] + [f"miss_{i}" for i in range(n_get)]
    t4 = time.perf_counter()
    for i in range(n_get):
        if rng.random() < 0.5 and hot_keys:
            q = rng.choice(hot_keys)  # likely hit
        else:
            q = rng.choice(cold_keys) if cold_keys else f"miss_{i}"  # likely miss
        got = cache.get(model_path, q)
        if got is not None:
            hits += 1
    t5 = time.perf_counter()

    return {
        "insert_s": t1 - t0,
        "prune_s": t3 - t2,
        "get_s": t5 - t4,
        "rows_before_prune": rows_before_prune,
        "rows_after_prune": rows_after_prune,
        "size_before_prune": size_before_prune,
        "size_after_prune": size_after_prune,
        "hit_rate": hits / n_get if n_get else 0.0,
    }


def test_cache_retention_policy_vs_baseline(tmp_path):
    """
    Performance comparison:
      - baseline: no TTL, no max_rows
      - policy: TTL + max_rows + periodic prune
    """
    model_path = "/tmp/fake_model.gguf"
    n_insert = 10_000
    n_get = 5_000
    emb_dim = 384
    hot_window = 2_000

    seeds = [7, 11, 23]
    trial_results = []

    for seed in seeds:
        baseline = EmbeddingCache(
            cache_dir=str(tmp_path / f"baseline_{seed}"),
            ttl_days=None,
            max_rows=None,
            prune_every_writes=10**9,  # effectively disabled
        )
        policy = EmbeddingCache(
            cache_dir=str(tmp_path / f"policy_{seed}"),
            ttl_days=30,
            max_rows=6_000,
            prune_every_writes=500,
        )

        base = _run_case(baseline, model_path, n_insert, n_get, emb_dim, hot_window, seed=seed)
        pol = _run_case(policy, model_path, n_insert, n_get, emb_dim, hot_window, seed=seed)
        trial_results.append((seed, base, pol))

        print(f"\n=== Trial seed={seed} ===")
        print(f"baseline: {base}")
        print(f"policy:   {pol}")

    base_hits = [r[1]["hit_rate"] for r in trial_results]
    pol_hits = [r[2]["hit_rate"] for r in trial_results]
    relative_hits = [p / b if b else 0.0 for b, p in zip(base_hits, pol_hits)]
    base_rows = [r[1]["rows_after_prune"] for r in trial_results]
    pol_rows = [r[2]["rows_after_prune"] for r in trial_results]
    base_sizes = [r[1]["size_after_prune"] for r in trial_results]
    pol_sizes = [r[2]["size_after_prune"] for r in trial_results]
    base_insert = [r[1]["insert_s"] for r in trial_results]
    pol_insert = [r[2]["insert_s"] for r in trial_results]

    print("\n=== Aggregate ===")
    print(f"baseline hit mean/std: {statistics.mean(base_hits):.4f}/{statistics.pstdev(base_hits):.4f}")
    print(f"policy   hit mean/std: {statistics.mean(pol_hits):.4f}/{statistics.pstdev(pol_hits):.4f}")
    print(f"relative hit mean:     {statistics.mean(relative_hits):.4f}")
    print(f"baseline rows mean:    {statistics.mean(base_rows):.1f}")
    print(f"policy rows mean:      {statistics.mean(pol_rows):.1f}")
    print(f"baseline size mean:    {statistics.mean(base_sizes):.1f}")
    print(f"policy size mean:      {statistics.mean(pol_sizes):.1f}")

    # Storage/capacity effectiveness
    assert statistics.mean(pol_rows) < statistics.mean(base_rows)
    assert statistics.mean(pol_sizes) < statistics.mean(base_sizes)

    # Hit-rate quality checks: absolute and relative to baseline
    assert statistics.mean(pol_hits) >= 0.45
    assert statistics.mean(relative_hits) >= 0.55

    # Policy should not add extreme insertion overhead
    assert statistics.mean(pol_insert) <= statistics.mean(base_insert) * 3.0
