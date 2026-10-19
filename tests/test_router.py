import pytest

from routewise.complexity import ComplexityResult
from routewise.config import Settings
from routewise.router import Router, Tier


class FixedScorer:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, query: str, context_chars: int = 0) -> ComplexityResult:
        return ComplexityResult(score=self._score, signals={"fixed": self._score})


def make_router(score: float, threshold: float = 0.30, weight: float = 0.20) -> Router:
    return Router(FixedScorer(score), "small-m", "large-m", threshold, weight)


def test_routes_by_threshold() -> None:
    assert make_router(0.20).route("q").tier is Tier.SMALL
    assert make_router(0.40).route("q").tier is Tier.LARGE
    assert make_router(0.40).route("q").model == "large-m"
    assert make_router(0.20).route("q").model == "small-m"


def test_score_equal_to_threshold_goes_large() -> None:
    assert make_router(0.30).route("q").tier is Tier.LARGE


def test_cost_pressure_shifts_borderline_queries_to_small() -> None:
    router = make_router(0.40)
    assert router.route("q", cost_pressure=0.0).tier is Tier.LARGE
    decision = router.route("q", cost_pressure=1.0)
    assert decision.tier is Tier.SMALL
    assert decision.threshold == pytest.approx(0.50)


def test_very_hard_queries_stay_large_under_max_pressure() -> None:
    assert make_router(0.80).route("q", cost_pressure=1.0).tier is Tier.LARGE


def test_pressure_is_clamped() -> None:
    router = make_router(0.40)
    assert router.route("q", cost_pressure=5.0).threshold == pytest.approx(0.50)
    assert router.route("q", cost_pressure=-3.0).threshold == pytest.approx(0.30)


def test_force_tier_overrides_score() -> None:
    decision = make_router(0.05).route("q", force_tier=Tier.LARGE)
    assert decision.tier is Tier.LARGE and decision.forced is True
    assert decision.score == pytest.approx(0.05)


def test_decision_exposes_signals() -> None:
    assert make_router(0.40).route("q").signals == {"fixed": 0.40}


@pytest.mark.parametrize(
    ("threshold", "weight"), [(-0.1, 0.2), (1.1, 0.2), (0.3, -0.1), (0.3, 1.5)]
)
def test_invalid_config_is_rejected(threshold: float, weight: float) -> None:
    with pytest.raises(ValueError):
        make_router(0.5, threshold=threshold, weight=weight)


def test_from_settings_with_real_scorer() -> None:
    router = Router.from_settings(Settings(_env_file=None))
    easy = router.route("What is the capital of France?")
    hard = router.route(
        "Design a scalable architecture for a multi-tenant SaaS platform and compare "
        "the trade-offs between a shared database and database-per-tenant."
    )
    assert easy.tier is Tier.SMALL
    assert hard.tier is Tier.LARGE
