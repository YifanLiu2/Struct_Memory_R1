"""
Embedding cache - stores computed embeddings in the artifacts folder.

Cache keys include:
- model_name: Identifies the embedding model (e.g., qwen/qwen3-embedding-8b)
- config_digest: For API clients, hashes base_url and other non-secret config
- normalize: Whether L2 normalization was applied
- text: The input text being embedded

This ensures correct cache hits only when all parameters match.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Project root: client/embedding_cache.py -> project root is parent of client
_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "artifacts" / "embedding_cache"


def _model_slug(model_name: str) -> str:
    """Create filesystem-safe slug from model name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:80]


def _config_digest(openai_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a digest of API config for cache key.
    Excludes api_key for security; includes base_url, model overrides, etc.
    """
    if not openai_config:
        return "local"
    # Include only fields that affect the embedding output
    safe = {
        k: v for k, v in openai_config.items()
        if k != "api_key" and v is not None
    }
    if not safe:
        return "default"
    canonical = json.dumps(safe, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _cache_key(
    model_name: str,
    config_digest: str,
    normalize: bool,
    text: str,
) -> str:
    """Compute cache key for a single embedding."""
    payload = f"{model_name}|{config_digest}|{normalize}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Return the embedding cache directory, creating it if needed."""
    path = cache_dir or DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


class EmbeddingCache:
    """
    Disk-backed cache for embeddings.
    
    Storage layout: {cache_dir}/{model_slug}/{key}.npy
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = get_cache_dir(cache_dir)

    def _path(self, model_name: str, key: str) -> Path:
        slug = _model_slug(model_name)
        return self.cache_dir / slug / f"{key}.npy"

    def get(self, model_name: str, config_digest: str, normalize: bool, text: str) -> Optional[np.ndarray]:
        """Return cached embedding if present, else None."""
        key = _cache_key(model_name, config_digest, normalize, text)
        path = self._path(model_name, key)
        if path.exists():
            try:
                arr = np.load(path)
                return arr
            except Exception as e:
                logger.warning(f"Embedding cache read failed for {path}: {e}")
        return None

    def set(self, model_name: str, config_digest: str, normalize: bool, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = _cache_key(model_name, config_digest, normalize, text)
        path = self._path(model_name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(path, embedding.astype(np.float32))
        except Exception as e:
            logger.warning(f"Embedding cache write failed for {path}: {e}")


class CachingEmbeddingClient:
    """
    Wrapper that adds disk caching to any embedding client.
    
    The delegate must have:
    - get_embedding(text, normalize) -> np.ndarray
    - get_embeddings(texts, normalize, batch_size) -> np.ndarray
    - model_name: str
    - embedding_dim: Optional[int] (can be None until first call)
    """

    def __init__(
        self,
        delegate: Any,
        cache_dir: Optional[Path] = None,
        openai_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            delegate: Underlying embedding client (TASBClient or OpenAIEmbeddingClient).
            cache_dir: Override cache directory. Default: artifacts/embedding_cache.
            openai_config: For API clients, the openai config dict (used for config_digest).
                          For local TASB, pass None.
        """
        self._delegate = delegate
        self._cache = EmbeddingCache(cache_dir)
        self._config_digest = _config_digest(openai_config)

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def embedding_dim(self) -> Optional[int]:
        return getattr(self._delegate, "embedding_dim", None)

    def get_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Get embedding, using cache when available."""
        cached = self._cache.get(
            self.model_name,
            self._config_digest,
            normalize,
            text,
        )
        if cached is not None:
            return cached
        emb = self._delegate.get_embedding(text, normalize=normalize)
        self._cache.set(
            self.model_name,
            self._config_digest,
            normalize,
            text,
            emb,
        )
        return emb

    def get_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get embeddings for multiple texts, using cache when available."""
        if not texts:
            dim = self.embedding_dim or 0
            return np.zeros((0, dim), dtype=np.float32)

        results = []
        to_fetch = []  # (index, text)
        for i, text in enumerate(texts):
            cached = self._cache.get(
                self.model_name,
                self._config_digest,
                normalize,
                text,
            )
            if cached is not None:
                results.append((i, cached))
            else:
                to_fetch.append((i, text))

        # Build output array in order
        if not to_fetch:
            out = np.vstack([r[1] for r in sorted(results, key=lambda x: x[0])])
            return out

        # Fetch missing embeddings from delegate (in batches)
        fetch_indices = [p[0] for p in to_fetch]
        fetch_texts = [p[1] for p in to_fetch]
        fetched = self._delegate.get_embeddings(
            fetch_texts,
            normalize=normalize,
            batch_size=batch_size,
        )

        # Cache and collect
        for j, (idx, text) in enumerate(to_fetch):
            emb = fetched[j]
            self._cache.set(
                self.model_name,
                self._config_digest,
                normalize,
                text,
                emb,
            )
            results.append((idx, emb))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.vstack([r[1] for r in results])

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity (uses cached embeddings)."""
        emb1 = self.get_embedding(text1, normalize=True)
        emb2 = self.get_embedding(text2, normalize=True)
        return float(np.dot(emb1, emb2))
