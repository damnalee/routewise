import pytest

from routewise.embeddings import HashEmbedder
from routewise.query_guard import queries_compatible
from routewise.semantic_cache import SemanticCache

# Same words in a different order: HashEmbedder gives these identical vectors (cosine 1.0),
# reproducing the real Celsius/Fahrenheit failure.
A = "convert celsius to fahrenheit"
B = "convert fahrenheit to celsius"


@pytest.fixture
def emb() -> HashEmbedder:
    return HashEmbedder(dim=4096)


def make_cache(emb: HashEmbedder, guarded: bool) -> SemanticCache:
    return SemanticCache(
        dim=emb.dim,
        threshold=0.8,
        ttl_s=60,
        max_entries=100,
        verifier=queries_compatible if guarded else None,
    )


def test_without_guard_order_swap_returns_wrong_answer(emb: HashEmbedder) -> None:
    cache = make_cache(emb, guarded=False)
    cache.store(emb.embed_one(A), A, "C*9/5+32", "small")
    hit = cache.lookup(emb.embed_one(B))
    assert hit is not None  # the bug the guard exists to prevent


def test_guard_rejects_order_swap(emb: HashEmbedder) -> None:
    cache = make_cache(emb, guarded=True)
    cache.store(emb.embed_one(A), A, "C*9/5+32", "small")
    assert cache.lookup(emb.embed_one(B), query=B) is None
    assert cache.stats.rejections == 1
    assert cache.stats.misses == 1 and cache.stats.hits == 0


def test_guard_still_serves_true_paraphrases(emb: HashEmbedder) -> None:
    cache = make_cache(emb, guarded=True)
    q = "how do i reset my password"
    paraphrase = "how do i reset my password quickly"
    cache.store(emb.embed_one(q), q, "use the reset link", "small")
    hit = cache.lookup(emb.embed_one(paraphrase), query=paraphrase)
    assert hit is not None and hit.entry.response == "use the reset link"
    assert cache.stats.rejections == 0


def test_conflicting_queries_coexist_instead_of_overwriting(emb: HashEmbedder) -> None:
    cache = make_cache(emb, guarded=True)
    cache.store(emb.embed_one(A), A, "C to F answer", "small")
    cache.store(emb.embed_one(B), B, "F to C answer", "small")
    assert len(cache) == 2
    hit_a = cache.lookup(emb.embed_one(A), query=A)
    hit_b = cache.lookup(emb.embed_one(B), query=B)
    assert hit_a is not None and hit_a.entry.response == "C to F answer"
    assert hit_b is not None and hit_b.entry.response == "F to C answer"


def test_query_text_is_required_when_guarded(emb: HashEmbedder) -> None:
    cache = make_cache(emb, guarded=True)
    with pytest.raises(ValueError):
        cache.lookup(emb.embed_one(A))
