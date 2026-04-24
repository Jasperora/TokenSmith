import sqlite3
import hashlib
import numpy as np
from src.embedder import EmbeddingCache


def _count(db):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


def _queries(db):
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT query_text FROM embeddings ORDER BY query_text").fetchall()
    return [r[0] for r in rows]


def test_roundtrip(tmp_path):
    cache = EmbeddingCache(cache_dir=str(tmp_path), ttl_days=None, max_rows=None)
    emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set("/tmp/model.gguf", "q1", emb)
    got = cache.get("/tmp/model.gguf", "q1")
    assert got is not None
    assert np.allclose(got, emb)

def test_ttl_prune(tmp_path):
    cache = EmbeddingCache(cache_dir=str(tmp_path), ttl_days=7, max_rows=None)
    with sqlite3.connect(cache.db_path) as c:
        c.execute("INSERT INTO embeddings (model_name, model_hash, query_text, embedding, timestamp) VALUES (?,?,?,?,datetime('now','-10 days'))",
                  ("m", "h", "old", np.array([1], dtype=np.float32).tobytes()))
        c.execute("INSERT INTO embeddings (model_name, model_hash, query_text, embedding, timestamp) VALUES (?,?,?,?,datetime('now','-1 day'))",
                  ("m", "h", "new", np.array([2], dtype=np.float32).tobytes()))
    cache.prune()
    assert _count(cache.db_path) == 1

def test_max_rows_prune(tmp_path):
    cache = EmbeddingCache(cache_dir=str(tmp_path), ttl_days=None, max_rows=2)
    for i in range(5):
        cache.set("/tmp/model.gguf", f"q{i}", np.array([i], dtype=np.float32))
    cache.prune()
    assert _count(cache.db_path) == 2


def test_strict_ttl_on_read(tmp_path):
    cache = EmbeddingCache(
        cache_dir=str(tmp_path),
        ttl_days=7,
        max_rows=None,
        strict_ttl_on_read=True,
    )
    model_path = "/tmp/model.gguf"
    model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
    with sqlite3.connect(cache.db_path) as c:
        c.execute(
            """
            INSERT INTO embeddings (
                model_name, model_hash, query_text, embedding, timestamp, last_accessed
            )
            VALUES (?, ?, ?, ?, datetime('now','-10 days'), datetime('now','-10 days'))
            """,
            ("m", model_hash, "old", np.array([1], dtype=np.float32).tobytes()),
        )
    # stale value should not be returned when strict_ttl_on_read=True
    assert cache.get(model_path, "old") is None


def test_lru_eviction_keeps_recently_accessed(tmp_path):
    cache = EmbeddingCache(
        cache_dir=str(tmp_path),
        ttl_days=None,
        max_rows=2,
        eviction_policy="lru",
    )
    model_path = "/tmp/model.gguf"
    for q in ("q1", "q2", "q3"):
        cache.set(model_path, q, np.array([1], dtype=np.float32))

    with sqlite3.connect(cache.db_path) as c:
        c.execute("UPDATE embeddings SET last_accessed=datetime('now','-3 days') WHERE query_text='q1'")
        c.execute("UPDATE embeddings SET last_accessed=datetime('now','-1 days') WHERE query_text='q2'")
        c.execute("UPDATE embeddings SET last_accessed=datetime('now','-2 days') WHERE query_text='q3'")
        c.commit()

    # q1 becomes most recently used due to read access.
    assert cache.get(model_path, "q1") is not None
    cache.prune()

    assert _count(cache.db_path) == 2
    assert _queries(cache.db_path) == ["q1", "q2"]
