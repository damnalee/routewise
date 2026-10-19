"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from routewise.router import Tier

MAX_QUERY_CHARS = 8_000
MAX_CONTEXT_ITEMS = 20
MAX_CONTEXT_CHARS = 60_000


class QueryRequest(BaseModel):
    """A question, optionally with retrieved RAG passages.

    `namespace` scopes the semantic cache, for example one per knowledge base or tenant.
    Cache hits are keyed on the query text within a namespace, NOT on `context`. If the same
    question must get different answers for different retrieved context, use a different
    namespace or set use_cache=false.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    context: list[str] = Field(default_factory=list, max_length=MAX_CONTEXT_ITEMS)
    namespace: str = Field("default", pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    use_cache: bool = True
    force_tier: Tier | None = None
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=16, le=8192)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def _context_size(self) -> QueryRequest:
        if sum(len(c) for c in self.context) > MAX_CONTEXT_CHARS:
            raise ValueError(f"total context must be <= {MAX_CONTEXT_CHARS} characters")
        return self


class QueryResponse(BaseModel):
    answer: str
    source: Literal["cache", "small", "large"]
    model: str | None = None
    cache_score: float | None = None
    complexity_score: float | None = None
    degraded: bool = False  # True when the large model failed and the small one answered
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool = False  # True for cache hits and providers without token counts
    cost_usd: float
    saved_usd: float  # versus answering with the large model
    latency_ms: float
