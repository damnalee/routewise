"""Prometheus metrics. One registry per instance so tests never collide."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from routewise.semantic_cache import SemanticCache
from routewise.tracker import CostTracker

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "routewise_requests",
            "Requests by outcome (cache_hit, small, large, error).",
            ["outcome"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "routewise_request_latency_seconds",
            "End-to-end request latency by outcome.",
            ["outcome"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.tokens = Counter(
            "routewise_llm_tokens",
            "LLM tokens consumed.",
            ["model", "direction"],
            registry=self.registry,
        )
        self.cost = Counter(
            "routewise_cost_usd", "Actual LLM spend in USD.", registry=self.registry
        )
        self.savings = Counter(
            "routewise_savings_usd",
            "Spend avoided versus always using the large model, in USD.",
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "routewise_fallbacks",
            "Large-model failures that fell back to the small model.",
            registry=self.registry,
        )
        self.cache_entries = Gauge(
            "routewise_cache_entries", "Entries in the semantic cache.", registry=self.registry
        )
        self.cache_rejections = Gauge(
            "routewise_cache_rejections",
            "Cache candidates vetoed by the lexical guard.",
            registry=self.registry,
        )
        self.cost_pressure = Gauge(
            "routewise_cost_pressure",
            "Recent spend as a fraction of the budget, 0 to 1.",
            registry=self.registry,
        )

    def bind(self, cache: SemanticCache, tracker: CostTracker) -> None:
        """Make gauges read live values at scrape time."""
        self.cache_entries.set_function(lambda: len(cache))
        self.cache_rejections.set_function(lambda: cache.stats.rejections)
        self.cost_pressure.set_function(tracker.cost_pressure)

    def observe(
        self,
        outcome: str,
        latency_s: float,
        model: str | None,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        saved_usd: float,
    ) -> None:
        self.requests.labels(outcome=outcome).inc()
        self.latency.labels(outcome=outcome).observe(latency_s)
        if model is not None:
            self.tokens.labels(model=model, direction="in").inc(tokens_in)
            self.tokens.labels(model=model, direction="out").inc(tokens_out)
        self.cost.inc(max(cost_usd, 0.0))
        self.savings.inc(max(saved_usd, 0.0))
