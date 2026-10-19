import threading
import time
from typing import Any

import numpy as np
import pytest

from routewise.embeddings import SentenceTransformerEmbedder


class SlowStubModel:
    """Stands in for a torch model. Records how many threads are inside encode at once."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._guard:
            self.active -= 1
        return np.ones((len(texts), 4), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return 4


def test_concurrent_encodes_are_serialised() -> None:
    embedder = SentenceTransformerEmbedder("stub-model")
    stub = SlowStubModel()
    embedder._model = stub  # skip the real model download

    threads = [threading.Thread(target=embedder.embed, args=(["hello"],)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stub.max_active == 1


def test_empty_input_returns_empty_matrix() -> None:
    embedder = SentenceTransformerEmbedder("stub-model")
    embedder._model = SlowStubModel()
    assert embedder.embed([]).shape == (0, 4)


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder("stub-model", device="cuda")
