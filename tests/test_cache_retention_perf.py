import sqlite3
import time
from pathlib import Path
import random

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
) -> dict:
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
        if random.random() < 0.5 and hot_keys:
            q = random.choice(hot_keys)  # likely hit
        else:
            q = random.choice(cold_keys) if cold_keys else f"miss_{i}"  # likely miss
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

    baseline = EmbeddingCache(
        cache_dir=str(tmp_path / "baseline"),
        ttl_days=None,
        max_rows=None,
        prune_every_writes=10**9,  # effectively disabled
    )
    policy = EmbeddingCache(
        cache_dir=str(tmp_path / "policy"),
        ttl_days=30,
        max_rows=6_000,
        prune_every_writes=500,
    )

    base = _run_case(baseline, model_path, n_insert, n_get, emb_dim, hot_window)
    pol = _run_case(policy, model_path, n_insert, n_get, emb_dim, hot_window)

    print("\n=== Baseline ===")
    for k, v in base.items():
        print(f"{k}: {v}")

    print("\n=== Policy ===")
    for k, v in pol.items():
        print(f"{k}: {v}")

    assert pol["rows_after_prune"] <= 8_000
    assert base["rows_after_prune"] >= pol["rows_after_prune"]
    assert pol["hit_rate"] >= 0.45  # should keep hot set
    assert pol["size_after_prune"] <= pol["size_before_prune"]
    assert pol["insert_s"] <= base["insert_s"] * 3.0