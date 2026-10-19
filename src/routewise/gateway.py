"""Request pipeline: cache lookup, routing, LLM call with fallback, accounting.

Flow for one request:
  1. embed the query
  2. semantic cache lookup (namespaced, verified by the lexical guard)  -> hit: done
  3. read cost pressure, route to small or large model
  4. call the LLM; if the large model fails with a transient error, fall back to small
  5. cache the answer (not when degraded), record cost and latency, update metrics

Framework independent: app.py is a thin HTTP layer on top of this class.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from routewise.config import Settings
from routewise.embeddings import Embedder, FloatArray, SentenceTransformerEmbedder
from routewise.llm import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    estimate_tokens,
)
from routewise.llm_factory import build_llm_client
from routewise.llm_groq import LLMAuthError, LLMRateLimitError
from routewise.logging import get_logger
from routewise.metrics import Metrics
from routewise.query_guard import queries_compatible
from routewise.router import Router, Tier
from routewise.schemas import QueryRequest, QueryResponse
from routewise.semantic_cache import CacheHit, SemanticCache
from routewise.tracker import CostTracker, Outcome, RequestRecord

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise, concise assistant. If context passages are provided, answer from "
    "them and say clearly when they do not contain the answer."
)

# Transient provider problems: worth trying the small model instead of failing.
_FALLBACK_ERRORS = (LLMRateLimitError, LLMUnavailableError, LLMTimeoutError)


class GatewayError(Exception):
    """An error that maps directly to an HTTP response."""

    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def map_llm_error(exc: LLMError) -> GatewayError:
    """Translate provider errors without leaking provider details to clients."""
    if isinstance(exc, LLMRateLimitError):
        return GatewayError(
            429, "upstream_rate_limited", "LLM provider is rate limited", {"Retry-After": "5"}
        )
    if isinstance(exc, LLMTimeoutError):
        return GatewayError(504, "upstream_timeout", "LLM provider timed out")
    if isinstance(exc, LLMUnavailableError):
        return GatewayError(503, "upstream_unavailable", "LLM provider is unavailable")
    return GatewayError(502, "upstream_error", "LLM provider error, see gateway logs")


def build_messages(query: str, context: list[str]) -> list[ChatMessage]:
    if context:
        passages = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context, start=1))
        user = f"Context:\n{passages}\n\nQuestion: {query}"
    else:
        user = query
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]


class Gateway:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        cache: SemanticCache,
        router: Router,
        llm: LLMClient,
        tracker: CostTracker,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.cache = cache
        self.router = router
        self.llm = llm
        self.tracker = tracker
        self.metrics = metrics
        metrics.bind(cache, tracker)

    @classmethod
    def build(cls, settings: Settings) -> Gateway:
        """Wire the production components from settings."""
        embedder = SentenceTransformerEmbedder(settings.embedding_model, device=settings.embedding_device)
        cache = SemanticCache(
            dim=embedder.dim,  # loads the embedding model once, at startup
            threshold=settings.cache_similarity_threshold,
            ttl_s=settings.cache_ttl_s,
            max_entries=settings.cache_max_entries,
            verifier=queries_compatible,
        )
        return cls(
            settings=settings,
            embedder=embedder,
            cache=cache,
            router=Router.from_settings(settings),
            llm=build_llm_client(settings),
            tracker=CostTracker.from_settings(settings, store_queries=settings.store_queries),
            metrics=Metrics(),
        )

    async def aclose(self) -> None:
        await self.llm.aclose()
        await asyncio.to_thread(self.tracker.close)

    async def answer(self, req: QueryRequest) -> QueryResponse:
        start = time.perf_counter()
        vector = (await self.embedder.aembed([req.query]))[0]

        if req.use_cache:
            hit = self.cache.lookup(vector, namespace=req.namespace, query=req.query)
            if hit is not None:
                return await self._serve_cached(req, hit, start)
        return await self._serve_fresh(req, vector, start)

    async def _serve_cached(self, req: QueryRequest, hit: CacheHit, start: float) -> QueryResponse:
        entry = hit.entry
        tokens_in = estimate_tokens(" ".join([req.query, *req.context]))
        tokens_out = estimate_tokens(entry.response)
        saved = self.tracker.cost_model.baseline_for("cache_hit", tokens_in, tokens_out)
        latency = time.perf_counter() - start

        await self._record(
            RequestRecord(
                ts=time.time(),
                outcome="cache_hit",
                latency_s=latency,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                namespace=req.namespace,
                query=req.query,
                model=entry.model,
            )
        )
        self.metrics.observe("cache_hit", latency, None, 0, 0, 0.0, saved)
        return QueryResponse(
            answer=entry.response,
            source="cache",
            model=entry.model,
            cache_score=round(hit.score, 4),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_estimated=True,
            cost_usd=0.0,
            saved_usd=saved,
            latency_ms=latency * 1000,
        )

    async def _serve_fresh(
        self, req: QueryRequest, vector: FloatArray, start: float
    ) -> QueryResponse:
        pressure = await asyncio.to_thread(self.tracker.cost_pressure)
        decision = self.router.route(
            req.query,
            context_chars=sum(len(c) for c in req.context),
            force_tier=req.force_tier,
            cost_pressure=pressure,
        )
        messages = build_messages(req.query, req.context)

        attempts: list[tuple[Tier, str]] = [(decision.tier, decision.model)]
        if decision.tier is Tier.LARGE and not decision.forced:
            attempts.append((Tier.SMALL, self.settings.small_model))

        llm_resp: LLMResponse | None = None
        used_tier = decision.tier
        last_error: LLMError | None = None
        for index, (tier, model) in enumerate(attempts):
            try:
                llm_resp = await self.llm.generate(
                    model, messages, temperature=req.temperature, max_tokens=req.max_tokens
                )
            except _FALLBACK_ERRORS as exc:
                last_error = exc
                if index + 1 < len(attempts):
                    log.warning("gateway.fallback", failed_model=model, error=str(exc))
                    self.metrics.fallbacks.inc()
                continue
            except LLMError as exc:
                last_error = exc
                break
            used_tier = tier
            break

        if llm_resp is None:
            assert last_error is not None
            log.error("gateway.llm_failed", model=decision.model, error=str(last_error))
            await self._record_error(req, decision.model, decision.score, start, last_error)
            raise map_llm_error(last_error)

        degraded = used_tier is not decision.tier
        outcome: Outcome = "small" if used_tier is Tier.SMALL else "large"
        source: Literal["small", "large"] = "small" if used_tier is Tier.SMALL else "large"
        cost_model = self.tracker.cost_model
        cost = cost_model.cost_for(outcome, llm_resp.tokens_in, llm_resp.tokens_out)
        baseline = cost_model.baseline_for(outcome, llm_resp.tokens_in, llm_resp.tokens_out)
        saved = max(baseline - cost, 0.0)

        if req.use_cache and not degraded:
            self.cache.store(
                vector, req.query, llm_resp.text, llm_resp.model, namespace=req.namespace
            )

        latency = time.perf_counter() - start
        await self._record(
            RequestRecord(
                ts=time.time(),
                outcome=outcome,
                latency_s=latency,
                tokens_in=llm_resp.tokens_in,
                tokens_out=llm_resp.tokens_out,
                namespace=req.namespace,
                query=req.query,
                model=llm_resp.model,
                score=decision.score,
            )
        )
        self.metrics.observe(
            outcome, latency, llm_resp.model, llm_resp.tokens_in, llm_resp.tokens_out, cost, saved
        )
        return QueryResponse(
            answer=llm_resp.text,
            source=source,
            model=llm_resp.model,
            complexity_score=decision.score,
            degraded=degraded,
            tokens_in=llm_resp.tokens_in,
            tokens_out=llm_resp.tokens_out,
            tokens_estimated=llm_resp.estimated_tokens,
            cost_usd=cost,
            saved_usd=saved,
            latency_ms=latency * 1000,
        )

    async def _record(self, rec: RequestRecord) -> None:
        """Persist a record. A metrics DB failure must never fail the user request."""
        try:
            await asyncio.to_thread(self.tracker.record, rec)
        except Exception:
            log.exception("gateway.record_failed")

    async def _record_error(
        self, req: QueryRequest, model: str, score: float, start: float, exc: LLMError
    ) -> None:
        latency = time.perf_counter() - start
        await self._record(
            RequestRecord(
                ts=time.time(),
                outcome="error",
                latency_s=latency,
                namespace=req.namespace,
                query=req.query,
                model=model,
                score=score,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        self.metrics.observe("error", latency, None, 0, 0, 0.0, 0.0)
