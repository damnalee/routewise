"""Text embedding backends.

`Embedder` is the interface the rest of the system depends on. Production uses
`SentenceTransformerEmbedder`; tests use `HashEmbedder`, which is deterministic,
instant, and needs no model download.

Thread safety: the gateway calls `aembed` through asyncio.to_thread, so several requests can
reach the model from different threads at once. PyTorch models are not safe for that (on
Apple MPS it corrupted memory and killed the process), so `SentenceTransformerEmbedder`
serialises every encode call behind a lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from routewise.logging import get_logger

log = get_logger(__name__)

FloatArray = NDArray[np.float32]


def l2_normalize(x: FloatArray) -> FloatArray:
    """Row-wise L2 normalisation so dot product equals cosine similarity."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


class Embedder(ABC):
    """Turns text into unit-length float32 vectors."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Return an array of shape (len(texts), dim), L2-normalised."""

    async def aembed(self, texts: Sequence[str]) -> FloatArray:
        """Run the (blocking) embedding off the event loop."""
        return await asyncio.to_thread(self.embed, texts)

    def embed_one(self, text: str) -> FloatArray:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model. Lazy-loaded, encode calls are serialised."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        if device not in ("auto", "cpu", "mps"):
            raise ValueError("device must be 'auto', 'cpu' or 'mps'")
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model = None  # type: ignore[var-annotated]
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    from sentence_transformers import SentenceTransformer

                    device = self._resolve_device()
                    log.info("embedder.loading", model=self._model_name, device=device)
                    self._model = SentenceTransformer(self._model_name, device=device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> FloatArray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._encode_lock:
            vectors = model.encode(
                list(texts),
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder for tests.

    Texts sharing words get higher cosine similarity, which is enough to test
    cache and routing logic without loading a neural model.
    """

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> FloatArray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                out[row, int.from_bytes(digest, "little") % self._dim] += 1.0
        return l2_normalize(out)
