import pytest

from routewise.query_guard import content_tokens, queries_compatible


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("How do I convert Celsius to Fahrenheit?", "How do I convert Fahrenheit to Celsius?"),
        ("Reset 2 devices", "Reset 3 devices"),
        ("Does it support streaming?", "Does it not support streaming?"),
        ("Why can't I log in?", "Why can I log in?"),
        ("Top 10 movies of the year", "Best movies of the year"),
    ],
)
def test_vetoes_what_embeddings_cannot_distinguish(a: str, b: str) -> None:
    assert queries_compatible(a, b) is False


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("How can I reset my password?", "What's the way to reset my password?"),
        ("What is semantic caching?", "Can you explain what semantic caching is?"),
        ("Explain how a stack works", "Can you explain how stacks work?"),
        ("How many calories are in an apple?", "What is the calorie count of one apple?"),
        ("Write a Python function to sort a list", "Give me a Python function that sorts a list"),
    ],
)
def test_allows_genuine_paraphrases(a: str, b: str) -> None:
    assert queries_compatible(a, b) is True


def test_guard_is_symmetric() -> None:
    a, b = "convert celsius to fahrenheit", "convert fahrenheit to celsius"
    assert queries_compatible(a, b) == queries_compatible(b, a)


def test_word_substitution_is_left_to_the_threshold() -> None:
    # Documented non-goal: "iPhone" vs "Android" differ by one word, so the guard
    # passes it and the similarity threshold must reject it instead.
    assert queries_compatible("dark mode on iPhone", "dark mode on Android") is True


def test_tokenizer_drops_stopwords_and_stems_plurals() -> None:
    assert content_tokens("What are the symptoms of the flu?") == ["what", "symptom", "flu"]
