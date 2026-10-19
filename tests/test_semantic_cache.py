import pytest

from routewise.embeddings import HashEmbedder
from routewise.semantic_cache import SemanticCache

QUERY = "how do i reset my password"
PARAPHRASE = "how do i reset my password quickly"  # cosine ~0.93 with QUERY
UNRELATED = "quarterly revenue forecast spreadsheet"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def emb() -> HashEmbedder:
    return HashEmbedder(dim=4096)  # large dim keeps hash collisions negligible


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_cache(
    emb: HashEmbedder,
    clock: FakeClock,
    threshold: float = 0.8,
    ttl_s: int = 60,
    max_entries: int = 100,
) -> SemanticCache:
    return SemanticCache(
        dim=emb.dim, threshold=threshold, ttl_s=ttl_s, max_entries=max_entries, clock=clock
    )


def test_miss_on_empty_cache(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock)
    assert cache.lookup(emb.embed_one(QUERY)) is None
    assert cache.stats.misses == 1


def test_paraphrase_hits_and_unrelated_misses(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock)
    cache.store(emb.embed_one(QUERY), QUERY, "use the reset link", "small")

    hit = cache.lookup(emb.embed_one(PARAPHRASE))
    assert hit is not None
    assert hit.entry.response == "use the reset link"
    assert hit.score > 0.8
    assert cache.lookup(emb.embed_one(UNRELATED)) is None
    assert cache.stats.hits == 1 and cache.stats.misses == 1
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_entries_expire_after_ttl(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock, ttl_s=60)
    cache.store(emb.embed_one(QUERY), QUERY, "answer", "small")
    clock.advance(61)
    assert cache.lookup(emb.embed_one(QUERY)) is None
    assert cache.stats.expirations == 1
    assert len(cache) == 0


def test_purge_expired_removes_only_stale_entries(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock, ttl_s=60)
    cache.store(emb.embed_one("alpha one"), "alpha one", "a", "small")
    clock.advance(40)
    cache.store(emb.embed_one("bravo two"), "bravo two", "b", "small")
    clock.advance(30)  # alpha is 70s old (expired), bravo is 30s old
    assert cache.purge_expired() == 1
    assert len(cache) == 1


def test_lru_eviction_keeps_recently_used(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock, max_entries=2)
    cache.store(emb.embed_one("alpha one"), "alpha one", "a", "small")
    cache.store(emb.embed_one("bravo two"), "bravo two", "b", "small")
    assert cache.lookup(emb.embed_one("alpha one")) is not None  # touch alpha
    cache.store(emb.embed_one("charlie three"), "charlie three", "c", "small")

    assert cache.stats.evictions == 1
    assert cache.lookup(emb.embed_one("bravo two")) is None  # least recently used, evicted
    assert cache.lookup(emb.embed_one("alpha one")) is not None
    assert cache.lookup(emb.embed_one("charlie three")) is not None


def test_namespaces_do_not_share_hits(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock)
    cache.store(emb.embed_one(QUERY), QUERY, "kb1 answer", "small", namespace="kb1")
    assert cache.lookup(emb.embed_one(QUERY), namespace="kb2") is None
    hit = cache.lookup(emb.embed_one(QUERY), namespace="kb1")
    assert hit is not None and hit.entry.response == "kb1 answer"


def test_restoring_same_query_updates_in_place(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock)
    cache.store(emb.embed_one(QUERY), QUERY, "old", "small")
    cache.store(emb.embed_one(QUERY), QUERY, "new", "large")
    assert len(cache) == 1
    hit = cache.lookup(emb.embed_one(QUERY))
    assert hit is not None
    assert hit.entry.response == "new" and hit.entry.model == "large"


def test_clear_by_namespace(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock)
    cache.store(emb.embed_one("alpha one"), "alpha one", "a", "small", namespace="kb1")
    cache.store(emb.embed_one("bravo two"), "bravo two", "b", "small", namespace="kb2")
    assert cache.clear("kb1") == 1
    assert len(cache) == 1
    assert cache.lookup(emb.embed_one("bravo two"), namespace="kb2") is not None


def test_hit_count_and_snapshot(emb: HashEmbedder, clock: FakeClock) -> None:
    cache = make_cache(emb, clock)
    cache.store(emb.embed_one(QUERY), QUERY, "answer", "small")
    cache.lookup(emb.embed_one(QUERY))
    hit = cache.lookup(emb.embed_one(QUERY))
    assert hit is not None and hit.entry.hit_count == 2
    snap = cache.snapshot()
    assert snap["size"] == 1 and snap["hits"] == 2 and snap["hit_rate"] == 1.0


@pytest.mark.parametrize(
    ("threshold", "ttl_s", "max_entries"),
    [(0.0, 60, 10), (1.5, 60, 10), (0.9, 0, 10), (0.9, 60, 0)],
)
def test_invalid_config_is_rejected(threshold: float, ttl_s: int, max_entries: int) -> None:
    with pytest.raises(ValueError):
        SemanticCache(dim=8, threshold=threshold, ttl_s=ttl_s, max_entries=max_entries)
