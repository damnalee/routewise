from routewise.config import Settings


def test_defaults_are_sane() -> None:
    s = Settings(_env_file=None)
    assert 0.0 <= s.cache_similarity_threshold <= 1.0
    assert s.small_price_in < s.large_price_in
    assert s.small_model != s.large_model
