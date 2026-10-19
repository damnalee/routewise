import numpy as np
import pytest

from routewise.embeddings import HashEmbedder
from routewise.vector_index import VectorIndex


def make_index() -> tuple[VectorIndex, HashEmbedder]:
    emb = HashEmbedder()
    return VectorIndex(dim=emb.dim, initial_capacity=2), emb


def test_exact_match_scores_one() -> None:
    idx, emb = make_index()
    item = idx.add(emb.embed_one("what is semantic caching"))
    hits = idx.search(emb.embed_one("what is semantic caching"), k=1)
    assert hits[0].id == item
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_returns_best_match_first_and_respects_k() -> None:
    idx, emb = make_index()
    ids = [idx.add(emb.embed_one(t)) for t in ["red apple pie", "blue ocean wave", "red apple tart"]]
    hits = idx.search(emb.embed_one("red apple pie"), k=2)
    assert [h.id for h in hits][0] == ids[0]
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_grows_beyond_initial_capacity() -> None:
    idx, emb = make_index()
    for i in range(50):
        idx.add(emb.embed_one(f"document number {i}"))
    assert len(idx) == 50


def test_remove_excludes_item_and_ids_are_never_reused() -> None:
    idx, emb = make_index()
    first = idx.add(emb.embed_one("alpha beta"))
    assert idx.remove(first) is True
    assert idx.remove(first) is False
    second = idx.add(emb.embed_one("alpha beta"))
    assert second != first
    hits = idx.search(emb.embed_one("alpha beta"), k=5)
    assert [h.id for h in hits] == [second]


def test_min_score_filters_weak_matches() -> None:
    idx, emb = make_index()
    idx.add(emb.embed_one("completely unrelated sentence"))
    assert idx.search(emb.embed_one("zebra"), k=1, min_score=0.9) == []


def test_empty_index_and_bad_input() -> None:
    idx, emb = make_index()
    assert idx.search(emb.embed_one("anything")) == []
    with pytest.raises(ValueError):
        idx.add(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        idx.search(emb.embed_one("x"), k=0)
