import sqlite3
import numpy as np
from src.embedder import EmbeddingCache

def _count(db):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

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