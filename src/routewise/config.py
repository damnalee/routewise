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

    # --- LLM provider ---
    llm_provider: Literal["ollama", "groq"] = "groq"
    small_model: str = "openai/gpt-oss-20b"
    large_model: str = "openai/gpt-oss-120b"
    request_timeout_s: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=10)
    llm_retry_backoff_s: float = Field(0.5, ge=0)

    # Ollama (local)
    llm_base_url: str = "http://localhost:11434"

    # Groq (hosted). Comma-separated keys, rotated round-robin with failover on 429.
    groq_api_keys: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_reasoning_effort: Literal["low", "medium", "high"] = "low"

    # --- embeddings + semantic cache ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)
    cache_ttl_s: int = Field(86_400, gt=0)
    cache_max_entries: int = Field(10_000, gt=0)

    # --- routing ---
    # Queries scoring >= router_threshold go to the large model.
    router_threshold: float = Field(0.30, ge=0.0, le=1.0)
    # Max amount the threshold is raised under full budget pressure (see Router).
    router_pressure_weight: float = Field(0.20, ge=0.0, le=1.0)

    # --- storage ---
    db_path: Path = Path("data/routewise.db")

    # --- pricing (USD per 1M tokens), approximate Groq list prices ---
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
