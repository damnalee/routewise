from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from routewise.app import create_app
from routewise.config import Settings
from routewise.embeddings import HashEmbedder
from routewise.gateway import Gateway
from routewise.llm import (
    ChatMessage,
    FakeLLMClient,
    LLMBadResponseError,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
)
from routewise.llm_groq import LLMAuthError, LLMRateLimitError
from routewise.metrics import Metrics
from routewise.query_guard import queries_compatible
from routewise.router import Router
from routewise.semantic_cache import SemanticCache
from routewise.tracker import CostModel, CostTracker

EASY = "How do I reset my password"
EASY_PARAPHRASE = "How do I reset my password quickly"
HARD = (
    "Design a scalable architecture for a multi-tenant SaaS platform and compare the "
    "trade-offs between a shared database and database-per-tenant."
)


class LargeDownLLM(FakeLLMClient):
    """Fails every call to the large model with a transient error."""

    def __init__(self, error: LLMError | None = None) -> None:
        super().__init__(reply="small model answer")
        self._error = error or LLMUnavailableError("large model down")

    async def generate(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if model == "large-m":
            self.calls.append((model, list(messages)))
            raise self._error
        return await super().generate(model, messages, temperature, max_tokens)


@dataclass
class Env:
    client: httpx.AsyncClient
    gateway: Gateway
    llm: FakeLLMClient
    tracker: CostTracker


@asynccontextmanager
async def running(llm: FakeLLMClient | None = None, api_key: str = "") -> AsyncIterator[Env]:
    settings = Settings(
        _env_file=None, small_model="small-m", large_model="large-m", gateway_api_key=api_key
    )
    embedder = HashEmbedder(dim=4096)
    cache = SemanticCache(
        dim=embedder.dim,
        threshold=settings.cache_similarity_threshold,
        ttl_s=60,
        max_entries=100,
        verifier=queries_compatible,
    )
    tracker = CostTracker(":memory:", CostModel.from_settings(settings))
    fake = llm or FakeLLMClient(reply="canned answer")
    gateway = Gateway(
        settings, embedder, cache, Router.from_settings(settings), fake, tracker, Metrics()
    )
    app = create_app(gateway, settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield Env(client, gateway, fake, tracker)
    tracker.close()


async def ask(env: Env, query: str, **extra: Any) -> httpx.Response:
    return await env.client.post("/v1/query", json={"query": query, **extra})


async def test_miss_then_semantic_hit() -> None:
    async with running() as env:
        first = (await ask(env, EASY)).json()
        assert first["source"] == "small" and first["model"] == "small-m"
        assert first["answer"] == "canned answer" and first["cost_usd"] > 0

        second = (await ask(env, EASY_PARAPHRASE)).json()
        assert second["source"] == "cache"
        assert second["answer"] == "canned answer"
        assert second["cost_usd"] == 0 and second["saved_usd"] > 0
        assert second["cache_score"] >= 0.85 and second["tokens_estimated"] is True
        assert len(env.llm.calls) == 1


async def test_hard_query_routes_to_large_model() -> None:
    async with running() as env:
        data = (await ask(env, HARD)).json()
        assert data["source"] == "large" and data["model"] == "large-m"
        assert data["complexity_score"] >= 0.30


async def test_force_tier_overrides_router() -> None:
    async with running() as env:
        data = (await ask(env, EASY, force_tier="large")).json()
        assert data["source"] == "large"


async def test_use_cache_false_skips_lookup_and_store() -> None:
    async with running() as env:
        await ask(env, EASY, use_cache=False)
        await ask(env, EASY, use_cache=False)
        assert len(env.llm.calls) == 2
        assert len(env.gateway.cache) == 0


async def test_namespaces_are_isolated() -> None:
    async with running() as env:
        await ask(env, EASY, namespace="kb1")
        other = (await ask(env, EASY, namespace="kb2")).json()
        same = (await ask(env, EASY, namespace="kb1")).json()
        assert other["source"] == "small" and same["source"] == "cache"


async def test_guard_blocks_order_swapped_query() -> None:
    async with running() as env:
        await ask(env, "convert celsius to fahrenheit")
        swapped = (await ask(env, "convert fahrenheit to celsius")).json()
        assert swapped["source"] == "small"
        assert len(env.llm.calls) == 2
        assert env.gateway.cache.stats.rejections == 1


async def test_context_is_sent_to_the_model() -> None:
    async with running() as env:
        await ask(env, "What is the policy", context=["Refunds take 5 days."])
        _, messages = env.llm.calls[0]
        assert messages[0].role == "system"
        assert "[1] Refunds take 5 days." in messages[1].content
        assert "Question: What is the policy" in messages[1].content


async def test_large_failure_falls_back_to_small_and_is_not_cached() -> None:
    async with running(LargeDownLLM()) as env:
        first = (await ask(env, HARD)).json()
        assert first["source"] == "small" and first["degraded"] is True
        second = (await ask(env, HARD)).json()
        assert second["source"] != "cache"
        metrics_text = (await env.client.get("/metrics")).text
        assert "routewise_fallbacks_total 2.0" in metrics_text


async def test_forced_large_does_not_fall_back() -> None:
    async with running(LargeDownLLM()) as env:
        resp = await ask(env, EASY, force_tier="large")
        assert resp.status_code == 503


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMUnavailableError("down"), 503, "upstream_unavailable"),
        (LLMTimeoutError("slow"), 504, "upstream_timeout"),
        (LLMRateLimitError("limit"), 429, "upstream_rate_limited"),
        (LLMAuthError("bad key secret-123"), 502, "upstream_error"),
        (LLMBadResponseError("garbage"), 502, "upstream_error"),
    ],
)
async def test_llm_errors_map_to_http_and_are_recorded(
    error: LLMError, status: int, code: str
) -> None:
    async with running(FakeLLMClient(fail_with=error)) as env:
        resp = await ask(env, EASY)
        assert resp.status_code == status
        assert resp.json()["error"]["code"] == code
        assert "secret-123" not in resp.text
        if status == 429:
            assert resp.headers["retry-after"] == "5"
        assert env.tracker.summary()["errors"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "ok", "namespace": "bad namespace!"},
        {"query": "ok", "unknown_field": 1},
        {"query": "ok", "temperature": 9},
        {"query": "x" * 8001},
    ],
)
async def test_invalid_requests_are_rejected(payload: dict[str, Any]) -> None:
    async with running() as env:
        resp = await env.client.post("/v1/query", json=payload)
        assert resp.status_code == 422
        assert env.llm.calls == []


async def test_api_key_protects_v1_but_not_probes() -> None:
    async with running(api_key="secret") as env:
        assert (await ask(env, EASY)).status_code == 401
        denied = await env.client.post(
            "/v1/query", json={"query": EASY}, headers={"Authorization": "Bearer nope"}
        )
        assert denied.status_code == 401
        assert denied.json()["error"]["code"] == "unauthorized"

        bearer = await env.client.post(
            "/v1/query", json={"query": EASY}, headers={"Authorization": "Bearer secret"}
        )
        assert bearer.status_code == 200
        header = await env.client.post(
            "/v1/query", json={"query": EASY}, headers={"X-API-Key": "secret"}
        )
        assert header.status_code == 200

        assert (await env.client.get("/v1/stats")).status_code == 401
        assert (await env.client.get("/healthz")).status_code == 200
        assert (await env.client.get("/metrics")).status_code == 200


async def test_health_and_readiness() -> None:
    async with running() as env:
        assert (await env.client.get("/healthz")).json()["status"] == "ok"
        assert (await env.client.get("/readyz")).status_code == 200
    async with running(FakeLLMClient(fail_with=LLMUnavailableError("down"))) as env:
        assert (await env.client.get("/readyz")).status_code == 503


async def test_metrics_endpoint_exposes_prometheus_text() -> None:
    async with running() as env:
        await ask(env, EASY)
        await ask(env, EASY_PARAPHRASE)
        resp = await env.client.get("/metrics")
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/plain")
        text = resp.text
        assert 'routewise_requests_total{outcome="small"} 1.0' in text
        assert 'routewise_requests_total{outcome="cache_hit"} 1.0' in text
        assert "routewise_cache_entries 1.0" in text
        assert "routewise_cost_usd_total" in text and "routewise_savings_usd_total" in text
        assert "routewise_request_latency_seconds_bucket" in text


async def test_stats_endpoint_summarises_traffic() -> None:
    async with running() as env:
        await ask(env, EASY)
        await ask(env, EASY_PARAPHRASE)
        data = (await env.client.get("/v1/stats")).json()
        assert data["summary"]["requests"] == 2 and data["summary"]["cache_hits"] == 1
        assert data["cache"]["size"] == 1 and 0.0 <= data["cost_pressure"] <= 1.0
        windowed = (await env.client.get("/v1/stats", params={"window_s": 3600})).json()
        assert windowed["summary"]["requests"] == 2


async def test_cache_clear_endpoint_invalidates_entries() -> None:
    async with running() as env:
        await ask(env, EASY)
        assert (await env.client.post("/v1/cache/clear")).json() == {"cleared": 1}
        again = (await ask(env, EASY)).json()
        assert again["source"] == "small"


async def test_request_id_is_echoed_or_generated() -> None:
    async with running() as env:
        echoed = await env.client.get("/healthz", headers={"X-Request-ID": "abc123"})
        assert echoed.headers["x-request-id"] == "abc123"
        generated = await env.client.get("/healthz")
        assert len(generated.headers["x-request-id"]) == 32
