"""Thread-safe in-memory vector index using exact cosine search.

Exact search over a contiguous float32 matrix is faster and simpler than an ANN
library at this scale (<= ~100k vectors) and gives perfect recall. IDs are
monotonically increasing and never reused, so stale references cannot alias a
newer entry.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import numpy as np

from routewise.embeddings import FloatArray, l2_normalize


class SearchHit(NamedTuple):
    id: int
    score: float


class VectorIndex:
    def __init__(self, dim: int, initial_capacity: int = 1024) -> None:
        if dim <= 0 or initial_capacity <= 0:
            raise ValueError("dim and initial_capacity must be positive")
        self._dim = dim
        self._matrix: FloatArray = np.zeros((initial_capacity, dim), dtype=np.float32)
        self._active = np.zeros(initial_capacity, dtype=bool)
        self._id_to_slot: dict[int, int] = {}
        self._slot_to_id: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._high_water = 0  # slots in [0, high_water) have been used at least once
        self._next_id = 0
        self._lock = threading.RLock()

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._id_to_slot)

    def __contains__(self, item_id: int) -> bool:
        return item_id in self._id_to_slot

    def _prepare(self, vector: FloatArray) -> FloatArray:
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self._dim,):
            raise ValueError(f"expected vector of shape ({self._dim},), got {v.shape}")
        return l2_normalize(v)

    def _grow(self) -> None:
        capacity = self._matrix.shape[0]
        self._matrix = np.vstack([self._matrix, np.zeros_like(self._matrix)])
        self._active = np.concatenate([self._active, np.zeros(capacity, dtype=bool)])

    def add(self, vector: FloatArray) -> int:
        """Insert a vector and return its permanent id."""
        v = self._prepare(vector)
        with self._lock:
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                if self._high_water == self._matrix.shape[0]:
                    self._grow()
                slot = self._high_water
                self._high_water += 1
            self._matrix[slot] = v
            self._active[slot] = True
            item_id = self._next_id
            self._next_id += 1
            self._id_to_slot[item_id] = slot
            self._slot_to_id[slot] = item_id
            return item_id

    def remove(self, item_id: int) -> bool:
        """Delete a vector. Returns False if the id was unknown."""
        with self._lock:
            slot = self._id_to_slot.pop(item_id, None)
            if slot is None:
                return False
            del self._slot_to_id[slot]
            self._active[slot] = False
            self._free_slots.append(slot)
            return True

    def search(
        self, query: FloatArray, k: int = 1, min_score: float | None = None
    ) -> list[SearchHit]:
        """Top-k cosine matches, best first, optionally above `min_score`."""
        if k <= 0:
            raise ValueError("k must be positive")
        q = self._prepare(query)
        with self._lock:
            if not self._id_to_slot:
                return []
            n = self._high_water
            scores = self._matrix[:n] @ q
            scores = np.where(self._active[:n], scores, -np.inf)
            k = min(k, len(self._id_to_slot))
            if k < n:
                top = np.argpartition(-scores, k - 1)[:k]
            else:
                top = np.arange(n)
            top = top[np.argsort(-scores[top])]
            hits = [
                SearchHit(self._slot_to_id[int(s)], float(scores[s]))
                for s in top
                if np.isfinite(scores[s])
            ]
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]
        return hits
