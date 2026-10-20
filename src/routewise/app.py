"""HTTP layer. Run with:  uvicorn routewise.app:build_app --factory --port 8000"""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, Query, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from routewise import __version__
from routewise.config import Settings, get_settings
from routewise.gateway import Gateway, GatewayError
from routewise.logging import configure_logging, get_logger
from routewise.schemas import QueryRequest, QueryResponse

log = get_logger(__name__)


def create_app(gateway: Gateway, settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info("gateway.started", provider=settings.llm_provider, version=__version__)
        yield
        await gateway.aclose()
        log.info("gateway.stopped")

    app = FastAPI(
        title="RouteWise",
        version=__version__,
        description="LLM gateway with semantic caching and cost-aware routing.",
        lifespan=lifespan,
        docs_url=None if settings.env == "prod" else "/docs",
        redoc_url=None,
    )
    expected_key = settings.gateway_api_key

    async def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not expected_key:
            return
        supplied = x_api_key
        if supplied is None and authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not supplied or not hmac.compare_digest(supplied.encode(), expected_key.encode()):
            raise GatewayError(401, "unauthorized", "missing or invalid API key")

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message}},
            headers=exc.headers,
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:64]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response

    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
    async def query(req: QueryRequest) -> QueryResponse:
        return await gateway.answer(req)

    @app.get("/v1/stats", dependencies=[Depends(require_api_key)])
    async def stats(window_s: int | None = Query(default=None, ge=1)) -> dict[str, Any]:
        since = time.time() - window_s if window_s else None
        summary = await asyncio.to_thread(gateway.tracker.summary, since)
        pressure = await asyncio.to_thread(gateway.tracker.cost_pressure)
        return {
            "summary": summary,
            "cache": gateway.cache.snapshot(),
            "cost_pressure": pressure,
        }

    @app.post("/v1/cache/clear", dependencies=[Depends(require_api_key)])
    async def clear_cache(
        namespace: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,64}$"),
    ) -> dict[str, int]:
        return {"cleared": gateway.cache.clear(namespace)}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = await gateway.llm.healthy()
        return JSONResponse(
            {"status": "ready" if ready else "llm_unavailable"},
            status_code=200 if ready else 503,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(gateway.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def build_app() -> FastAPI:
    """Application factory for uvicorn."""
    settings = get_settings()
    configure_logging(settings)
    return create_app(Gateway.build(settings), settings)
