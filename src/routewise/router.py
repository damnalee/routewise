"""Cost-aware model router.

Decision rule: send a query to the LARGE model when its complexity score is at or above the
effective threshold, otherwise to the SMALL model.

Cost awareness: `cost_pressure` in [0, 1] (derived from recent spend vs. budget by the cost
tracker in a later step) raises the threshold by up to `pressure_weight`, shifting borderline
queries to the small model. Because the shift is bounded, very hard queries (score above
threshold + pressure_weight) always stay on the large model, so quality has a floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from routewise.complexity import ComplexityScorer, HeuristicScorer
from routewise.config import Settings
from routewise.logging import get_logger

log = get_logger(__name__)


class Tier(StrEnum):
    SMALL = "small"
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    tier: Tier
    model: str
    score: float
    threshold: float  # effective threshold used, includes any cost pressure
    signals: dict[str, float]
    forced: bool = False


class Router:
    def __init__(
        self,
        scorer: ComplexityScorer,
        small_model: str,
        large_model: str,
        threshold: float = 0.30,
        pressure_weight: float = 0.20,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= pressure_weight <= 1.0:
            raise ValueError("pressure_weight must be in [0, 1]")
        self._scorer = scorer
        self._small_model = small_model
        self._large_model = large_model
        self._threshold = threshold
        self._pressure_weight = pressure_weight

    @classmethod
    def from_settings(cls, settings: Settings, scorer: ComplexityScorer | None = None) -> Router:
        return cls(
            scorer=scorer or HeuristicScorer(),
            small_model=settings.small_model,
            large_model=settings.large_model,
            threshold=settings.router_threshold,
            pressure_weight=settings.router_pressure_weight,
        )

    def route(
        self,
        query: str,
        context_chars: int = 0,
        force_tier: Tier | None = None,
        cost_pressure: float = 0.0,
    ) -> RouteDecision:
        pressure = min(max(cost_pressure, 0.0), 1.0)
        threshold = min(self._threshold + self._pressure_weight * pressure, 1.0)
        result = self._scorer.score(query, context_chars)

        if force_tier is not None:
            tier, forced = force_tier, True
        else:
            tier = Tier.LARGE if result.score >= threshold else Tier.SMALL
            forced = False

        model = self._large_model if tier is Tier.LARGE else self._small_model
        log.debug(
            "router.decision",
            tier=str(tier),
            model=model,
            score=result.score,
            threshold=threshold,
            forced=forced,
        )
        return RouteDecision(
            tier=tier,
            model=model,
            score=result.score,
            threshold=threshold,
            signals=result.signals,
            forced=forced,
        )
