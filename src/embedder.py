import sqlite3
import hashlib
import multiprocessing
import multiprocessing.pool
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Literal, Any, TYPE_CHECKING
from tqdm import tqdm

# Import only for type checkers; runtime import is lazy so cache-only code paths
# can run without llama.cpp installed.
if TYPE_CHECKING:
    from llama_cpp import Llama

# Global variables for worker processes
_worker_model: Optional[Any] = None
_worker_embedding_dim: int = 0


def _get_llama_class():
    try:
        from llama_cpp import Llama as _Llama
    except ImportError as exc:
        raise ImportError(
            "llama_cpp is required for embedding generation. "
            "Install project dependencies (e.g., `make build`) to enable this feature."
        ) from exc
    return _Llama

def _init_worker(model_path: str, n_ctx: int, n_threads: int):
    """
    Initializes the model inside a worker process.
    """
    global _worker_model, _worker_embedding_dim
    Llama = _get_llama_class()

    _worker_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=False,
        use_mmap=True # Allows OS to share model weights across processes
    )
    
    # Cache dimension
    test_emb = _worker_model.create_embedding("test")['data'][0]['embedding']
    _worker_embedding_dim = len(test_emb)

def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Encodes a batch of text using the worker's local model instance.
    """
    global _worker_model, _worker_embedding_dim
    if _worker_model is None:
        return []
        
    embeddings = []
    for text in texts:
        try:
            # Create embedding
            emb = _worker_model.create_embedding(text)['data'][0]['embedding']
            embeddings.append(emb)
        except Exception:
            # Return zero vector on failure
            embeddings.append([0.0] * _worker_embedding_dim)
            
    return embeddings

class SentenceTransformer:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        """
        Initialize with a local GGUF model file path.
        
        Args:
            model_path: Path to your local .gguf file
            n_ctx: Context window size (increased to match Qwen3 training context)
            n_threads: Number of threads to use (None = auto-detect)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        Llama = _get_llama_class()
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            embedding=True,
            verbose=True,
            use_mmap=True,
            n_gpu_layers=-1 # use GPU if available
        )
        self._embedding_dimension = None
        
        _ = self.embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension

    def encode(self, 
           texts: Union[str, List[str]], 
           batch_size: int = 16,  # Adjusted for 4B model
           normalize: bool = False,
           show_progress_bar: bool = False,
           **kwargs) -> np.ndarray:

        """
        Encode texts to embeddings with batch processing.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings
            show_progress_bar: Whether to show progress bar
            Returns:
            numpy.ndarray: Float32 embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        # Process in batches
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            try:
                # IMPORTANT CHANGE: Pass the entire LIST to the model at once.
                # This triggers the native C++/Metal batch processing logic.
                response = self.model.create_embedding(batch_texts)
                
                # Extract the list of embedding vectors from the response
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch: {e}")
                # Fallback: encode one by one if batch fails, or append zeros
                for _ in batch_texts:
                    embeddings.append([0.0] * self.embedding_dimension)
                
        vecs = np.array(embeddings, dtype=np.float32)
        
        if normalize: # do L2 normalization
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)
            
        return vecs

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (compatibility method)."""
        return self.embedding_dimension

    def start_multi_process_pool(self, num_workers: int = None) -> multiprocessing.pool.Pool:
        """
        Starts a pool of worker processes.
        """
        if num_workers:
            workers = num_workers
        else:
            # Default to CPU count - 2 (leave room for OS/Main process)
            workers = max(1, multiprocessing.cpu_count() - 2)

        print(f"Creating {workers} worker processes...")
        
        # Use 1 thread per worker to avoid CPU thrashing
        worker_threads = 1
        
        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.model_path, self.n_ctx, worker_threads)
        )
        return pool

    def encode_multi_process(self, texts: List[str], pool: multiprocessing.pool.Pool, batch_size: int = 32) -> np.ndarray:
        """
        Distributes work across the pool.
        """
        # Sort by length to minimize padding/processing waste
        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]

        # Create batches
        chunks = [sorted_texts[i : i + batch_size] for i in range(0, len(sorted_texts), batch_size)]

        # Process with progress bar
        results = []
        print(f"Dispatching {len(chunks)} batches to pool...")
        for batch_result in tqdm(
            pool.imap(_encode_batch_worker, chunks), 
            total=len(chunks), 
            desc="Parallel Encoding"
        ):
            results.append(batch_result)

        flat_embeddings = [emb for batch in results for emb in batch]

        # Restore original order
        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        ordered_embeddings = [flat_embeddings[i] for i in inverse_indices]
        
        return np.array(ordered_embeddings, dtype=np.float32)

    @staticmethod
    def stop_multi_process_pool(pool: multiprocessing.pool.Pool):
        pool.close()
        pool.join()


class EmbeddingCache:
    """Persistent SQLite cache for embeddings."""
    
    def __init__(
        self,
        cache_dir: str = "index/cache",
        ttl_days: Optional[int] = 30,
        max_rows: Optional[int] = 200_000,
        prune_every_writes: int = 500,
        strict_ttl_on_read: bool = False,
        eviction_policy: Literal["fifo", "lru"] = "fifo",
    ):
        self.db_path = Path(cache_dir) / "embeddings.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.ttl_days = ttl_days
        self.max_rows = max_rows
        self.prune_every_writes = max(1, prune_every_writes)
        self.strict_ttl_on_read = strict_ttl_on_read
        self.eviction_policy = eviction_policy.lower()
        if self.eviction_policy not in {"fifo", "lru"}:
            raise ValueError("eviction_policy must be one of: fifo, lru")
        self._writes_since_prune = 0

        self._init_db()
        self.prune()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT,
                    model_hash TEXT,
                    query_text TEXT,
                    embedding BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_hash, query_text)
                )
            """)
            # Backward-compatible migrations for pre-existing DBs.
            try:
                conn.execute("ALTER TABLE embeddings ADD COLUMN last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON embeddings(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON embeddings(last_accessed)")
    
    def get(self, model_path: str, query: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if it exists."""
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            if self.strict_ttl_on_read and self.ttl_days is not None:
                row = conn.execute(
                    """
                    SELECT embedding
                    FROM embeddings
                    WHERE model_hash=? AND query_text=?
                      AND timestamp >= datetime('now', ?)
                    """,
                    (model_hash, query, f"-{int(self.ttl_days)} days"),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT embedding FROM embeddings WHERE model_hash=? AND query_text=?",
                    (model_hash, query),
                ).fetchone()
            
            if row:
                conn.execute(
                    """
                    UPDATE embeddings
                    SET last_accessed=CURRENT_TIMESTAMP
                    WHERE model_hash=? AND query_text=?
                    """,
                    (model_hash, query),
                )
                return np.frombuffer(row[0], dtype=np.float32)
        return None

    def set(self, model_path: str, query: str, embedding: np.ndarray):
        """Store embedding in cache."""
        model_name = Path(model_path).stem
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        blob = embedding.astype(np.float32).tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO embeddings (
                    model_name,
                    model_hash,
                    query_text,
                    embedding,
                    timestamp,
                    last_accessed
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(model_hash, query_text)
                DO UPDATE SET
                    model_name=excluded.model_name,
                    embedding=excluded.embedding,
                    timestamp=CURRENT_TIMESTAMP,
                    last_accessed=CURRENT_TIMESTAMP
                """,
                (model_name, model_hash, query, blob),
            )
        self._prune_if_needed()

    def prune(self):
        """Retention policy: TTL + max row cap"""
        with sqlite3.connect(self.db_path) as conn:
            if self.ttl_days is not None:
                conn.execute(
                    "DELETE FROM embeddings WHERE timestamp < datetime('now', ?)",
                    (f"-{int(self.ttl_days)} days",),
                )
            
            if self.max_rows is not None:
                order_col = "last_accessed" if self.eviction_policy == "lru" else "timestamp"
                conn.execute(
                    f"""
                    DELETE FROM embeddings
                    WHERE rowid IN (
                        SELECT rowid
                        FROM embeddings
                        ORDER BY {order_col} DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (int(self.max_rows),),
                )
    
    def _prune_if_needed(self):
        self._writes_since_prune += 1
        if self._writes_since_prune >= self.prune_every_writes:
            self.prune()
            self._writes_since_prune = 0


class CachedEmbedder:
    """
    Wrapper around SentenceTransformer that caches query embeddings.
    Drop-in replacement for SentenceTransformer.
    """
    
    def __init__(
        self,
        model_path: str,
        cache_dir: str = "index/cache",
        ttl_days: Optional[int] = 30,
        max_rows: Optional[int] = 200_000,
        prune_every_writes: int = 500,
        strict_ttl_on_read: bool = False,
        eviction_policy: Literal["fifo", "lru"] = "fifo",
        **kwargs,
    ):
        self.embedder = SentenceTransformer(model_path, **kwargs)
        self.cache = EmbeddingCache(
            cache_dir=cache_dir,
            ttl_days=ttl_days,
            max_rows=max_rows,
            prune_every_writes=prune_every_writes,
            strict_ttl_on_read=strict_ttl_on_read,
            eviction_policy=eviction_policy,
        )
        self.model_path = model_path
    
    def encode(self, texts, **kwargs):
        """Encode texts with caching support."""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        to_compute = []
        to_compute_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(self.model_path, text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)
        
        # Compute missing embeddings
        if to_compute:
            computed = self.embedder.encode(to_compute, **kwargs)
            for idx, text, emb in zip(to_compute_indices, to_compute, computed):
                self.cache.set(self.model_path, text, emb)
                results.append((idx, emb))
        
        # Restore original order
        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])
    
    def __getattr__(self, name):
        """Delegate other methods to wrapped embedder."""
        return getattr(self.embedder, name)
