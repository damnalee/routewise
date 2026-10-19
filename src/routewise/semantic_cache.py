"""Semantic cache: reuse LLM answers for queries that mean the same thing.

Design notes:
  * The caller embeds the query once and passes the vector to `lookup` and `store`,
    so a miss followed by a store costs a single embedding.
  * Entries are scoped by `namespace` (e.g. tenant or knowledge-base id). A RAG answer
    is only valid for the corpus it was generated from, so namespaces never share hits.
  * An optional `verifier(query, cached_query) -> bool` runs on every candidate above the
    threshold and can veto it (see query_guard.py). Vetoed candidates count as rejections.
  * Eviction is LRU once `max_entries` is exceeded; entries also expire after a TTL.
  * All public methods are thread-safe.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from routewise.embeddings import FloatArray
from routewise.logging import get_logger
from routewise.vector_index import VectorIndex

log = get_logger(__name__)

_DUPLICATE_SCORE = 0.999  # a stored query this close is treated as the same query
_SEARCH_DEPTH = 5  # candidates inspected per lookup (skips expired or vetoed ones)

Verifier = Callable[[str, str], bool]


@dataclass(slots=True)
class CacheEntry:
    namespace: str
    entry_id: int
    query: str
    response: str
    model: str
    created_at: float
    expires_at: float
    hit_count: int = 0


@dataclass(frozen=True, slots=True)
class CacheHit:
    entry: CacheEntry
    score: float


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0
    rejections: int = 0  # candidates above threshold vetoed by the verifier

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class SemanticCache:
    def __init__(
        self,
        dim: int,
        threshold: float,
        ttl_s: int,
        max_entries: int,
        clock: Callable[[], float] = time.time,
        verifier: Verifier | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if ttl_s <= 0 or max_entries <= 0:
            raise ValueError("ttl_s and max_entries must be positive")
        self._dim = dim
        self._threshold = threshold
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._verifier = verifier
        self._indexes: dict[str, VectorIndex] = {}
        self._entries: OrderedDict[tuple[str, int], CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    @property
    def threshold(self) -> float:
        return self._threshold

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def lookup(
        self,
        vector: FloatArray,
        namespace: str = "default",
        query: str | None = None,
    ) -> CacheHit | None:
        """Return the best non-expired, verified entry at or above the threshold, else None.

        `query` is the raw text and is required when the cache has a verifier.
        """
        verifier = self._verifier
        if verifier is not None and query is None:
            raise ValueError("query text is required when a verifier is configured")

        with self._lock:
            now = self._clock()
            index = self._indexes.get(namespace)
            if index is not None:
                candidates = index.search(vector, k=_SEARCH_DEPTH, min_score=self._threshold)
                for cand in candidates:
                    key = (namespace, cand.id)
                    entry = self._entries[key]
                    if entry.expires_at <= now:
                        self._remove(key)
                        self.stats.expirations += 1
                        continue
                    if (
                        verifier is not None
                        and query is not None
                        and not verifier(query, entry.query)
                    ):
                        self.stats.rejections += 1
                        continue
                    entry.hit_count += 1
                    self._entries.move_to_end(key)
                    self.stats.hits += 1
                    return CacheHit(entry=entry, score=cand.score)
            self.stats.misses += 1
            return None

    def store(
        self,
        vector: FloatArray,
        query: str,
        response: str,
        model: str,
        namespace: str = "default",
        ttl_s: int | None = None,
    ) -> CacheEntry:
        """Insert an answer. Re-storing a near-identical query refreshes it in place."""
        with self._lock:
            now = self._clock()
            ttl = self._ttl_s if ttl_s is None else ttl_s
            if ttl <= 0:
                raise ValueError("ttl_s must be positive")
            index = self._indexes.get(namespace)
            if index is None:
                index = self._indexes[namespace] = VectorIndex(self._dim)

            duplicates = index.search(vector, k=1, min_score=_DUPLICATE_SCORE)
            if duplicates:
                key = (namespace, duplicates[0].id)
                existing = self._entries[key]
                # Embedding-identical is not text-identical ("A to B" vs "B to A"),
                # so only refresh in place when the verifier agrees they match.
                if self._verifier is None or self._verifier(query, existing.query):
                    existing.response = response
                    existing.model = model
                    existing.created_at = now
                    existing.expires_at = now + ttl
                    self._entries.move_to_end(key)
                    self.stats.stores += 1
                    return existing

            entry_id = index.add(vector)
            entry = CacheEntry(
                namespace=namespace,
                entry_id=entry_id,
                query=query,
                response=response,
                model=model,
                created_at=now,
                expires_at=now + ttl,
            )
            self._entries[(namespace, entry_id)] = entry
            self.stats.stores += 1

            while len(self._entries) > self._max_entries:
                oldest = next(iter(self._entries))
                self._remove(oldest)
                self.stats.evictions += 1
            return entry

    def purge_expired(self) -> int:
        """Drop all expired entries. Returns how many were removed."""
        with self._lock:
            now = self._clock()
            expired = [k for k, e in self._entries.items() if e.expires_at <= now]
            for key in expired:
                self._remove(key)
            self.stats.expirations += len(expired)
            return len(expired)

    def clear(self, namespace: str | None = None) -> int:
        """Invalidate one namespace (e.g. after a corpus update) or everything."""
        with self._lock:
            keys = [k for k in self._entries if namespace is None or k[0] == namespace]
            for key in keys:
                self._remove(key)
            return len(keys)

    def snapshot(self) -> dict[str, float | int]:
        """Point-in-time numbers for /metrics and the dashboard."""
        with self._lock:
            s = self.stats
            return {
                "size": len(self._entries),
                "hits": s.hits,
                "misses": s.misses,
                "stores": s.stores,
                "evictions": s.evictions,
                "expirations": s.expirations,
                "rejections": s.rejections,
                "hit_rate": s.hit_rate,
            }

    def _remove(self, key: tuple[str, int]) -> None:
        namespace, entry_id = key
        self._entries.pop(key, None)
        index = self._indexes.get(namespace)
        if index is not None:
            index.remove(entry_id)
            if len(index) == 0:
                del self._indexes[namespace]
