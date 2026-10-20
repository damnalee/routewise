"""Centralised, typed configuration loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ROUTEWISE_",
        extra="ignore",
    )

    # --- app ---
    env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # --- gateway ---
    # If set, /v1/* endpoints require this key via "Authorization: Bearer <key>" or "X-API-Key".
    gateway_api_key: str = ""
    # Store a short preview of each query in the metrics DB. Turn off for sensitive data.
    store_queries: bool = True

    # --- LLM provider ---
    llm_provider: Literal["ollama", "groq"] = "groq"
    small_model: str = "openai/gpt-oss-20b"
    large_model: str = "openai/gpt-oss-120b"
    request_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_retry_backoff_s: float = Field(default=0.5, ge=0)

    # Ollama (local)
    llm_base_url: str = "http://localhost:11434"

    # Groq (hosted). Comma-separated keys, rotated round-robin with failover on 429.
    groq_api_keys: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_reasoning_effort: Literal["low", "medium", "high"] = "low"

    # --- embeddings + semantic cache ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # cpu is the safe default: torch on Apple MPS crashed under concurrent requests.
    embedding_device: Literal["auto", "cpu", "mps"] = "cpu"
    cache_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    cache_ttl_s: int = Field(default=86_400, gt=0)
    cache_max_entries: int = Field(default=10_000, gt=0)

    # --- routing ---
    # Queries scoring >= router_threshold go to the large model.
    router_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    # Max amount the threshold is raised under full budget pressure, see Router.
    router_pressure_weight: float = Field(default=0.20, ge=0.0, le=1.0)

    # --- cost budget ---
    # cost_pressure = spend in the last budget_window_s divided by budget_usd, clamped to [0, 1].
    budget_usd: float = Field(default=1.0, gt=0)
    budget_window_s: int = Field(default=3600, gt=0)

    # --- storage ---
    db_path: Path = Path("data/routewise.db")

    # --- pricing, USD per 1M tokens, approximate Groq list prices ---
    # Verify at console.groq.com. Used for cost tracking and savings estimates.
    small_price_in: float = 0.075
    small_price_out: float = 0.30
    large_price_in: float = 0.15
    large_price_out: float = 0.60

    @property
    def groq_keys(self) -> list[str]:
        return [k.strip() for k in self.groq_api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
