import pytest

from routewise.complexity import HeuristicScorer

FENCE = "`" * 3
scorer = HeuristicScorer()


def test_trivial_lookup_scores_near_zero() -> None:
    assert scorer.score("What is the capital of France?").score < 0.05


def test_empty_query_scores_zero() -> None:
    assert scorer.score("   ").score == 0.0


def test_architecture_question_scores_high() -> None:
    query = (
        "Design a scalable architecture for a multi-tenant SaaS platform. Compare the "
        "trade-offs between a shared database and database-per-tenant, analyze the failure "
        "modes, and justify your recommendation step by step."
    )
    result = scorer.score(query)
    assert result.signals["reasoning"] == pytest.approx(0.45)
    assert result.score >= 0.45


def test_fenced_code_pushes_score_up() -> None:
    query = f"Why does this fail?\n{FENCE}python\ndef f(x):\n    return x[0]\n{FENCE}"
    result = scorer.score(query)
    assert result.signals["code"] == pytest.approx(0.40)
    assert result.score >= 0.5


def test_short_snippet_request_is_not_hard() -> None:
    assert scorer.score("Write a Python function to sort a list").score < 0.30


def test_math_keyword_and_expression_detected() -> None:
    result = scorer.score("Solve the integral of x^2 sin(x) dx and show each step.")
    assert result.signals["math"] == pytest.approx(0.40)


def test_context_size_raises_score() -> None:
    query = "Summarize the main points of the document."
    assert scorer.score(query, context_chars=20_000).score > scorer.score(query).score


def test_score_is_clamped_to_unit_interval() -> None:
    huge = "design architecture compare analyze justify " * 200 + FENCE
    result = scorer.score(huge, context_chars=10**6)
    assert 0.0 <= result.score <= 1.0


def test_negative_context_is_rejected() -> None:
    with pytest.raises(ValueError):
        scorer.score("hello", context_chars=-1)
