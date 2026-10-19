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

    # --- LLM provider (Ollama by default) ---
    llm_base_url: str = "http://localhost:11434"
    small_model: str = "llama3.2:3b"
    large_model: str = "qwen2.5:7b"
    request_timeout_s: float = Field(60.0, gt=0)

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

    # --- reference pricing (USD per 1M tokens) ---
    # Local models are free, so we price them like comparable hosted tiers
    # to make the cost-savings numbers meaningful.
    small_price_in: float = 0.15
    small_price_out: float = 0.60
    large_price_in: float = 2.50
    large_price_out: float = 10.00


@lru_cache
def get_settings() -> Settings:
    return Settings()
