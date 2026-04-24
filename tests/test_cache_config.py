from src.config import RAGConfig
from src.embedder import EmbeddingCache


def test_rag_config_exposes_embedding_cache_settings():
    cfg = RAGConfig(
        cache_dir="tmp/cache",
        cache_ttl_days=14,
        cache_max_rows=1234,
        cache_prune_every_writes=77,
        cache_strict_ttl_on_read=True,
        cache_eviction_policy="lru",
    )

    cache_cfg = cfg.get_embedding_cache_config()
    assert cache_cfg["cache_dir"] == "tmp/cache"
    assert cache_cfg["ttl_days"] == 14
    assert cache_cfg["max_rows"] == 1234
    assert cache_cfg["prune_every_writes"] == 77
    assert cache_cfg["strict_ttl_on_read"] is True
    assert cache_cfg["eviction_policy"] == "lru"


def test_embedding_cache_rejects_unknown_eviction_policy(tmp_path):
    try:
        EmbeddingCache(cache_dir=str(tmp_path), eviction_policy="unknown")
    except ValueError as exc:
        assert "eviction_policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid eviction_policy")
