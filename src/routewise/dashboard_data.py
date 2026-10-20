"""Pure data shaping for the dashboard. No Streamlit imports, so it is unit-testable."""

from __future__ import annotations

from datetime import tzinfo
from typing import Any

import pandas as pd

WINDOWS: dict[str, int] = {
    "Last 15 minutes": 900,
    "Last hour": 3_600,
    "Last 6 hours": 21_600,
    "Last 24 hours": 86_400,
    "Last 7 days": 604_800,
}
TARGET_POINTS = 60
MIN_BUCKET_S = 5

OUTCOME_ORDER = ("cache_hit", "small", "large", "error")
LATENCY_ORDER = ("all", *OUTCOME_ORDER)

TIMESERIES_COLUMNS = [
    "time",
    "requests",
    "cache_hits",
    "llm_calls",
    "errors",
    "cost_usd",
    "baseline_usd",
    "avg_latency_s",
    "cum_cost_usd",
    "cum_baseline_usd",
]
RECENT_COLUMNS = [
    "time",
    "outcome",
    "model",
    "score",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "query",
    "error",
]
LATENCY_COLUMNS = ["outcome", "requests", "avg_ms", "p50_ms", "p95_ms"]


def bucket_for_window(window_s: int) -> int:
    """Chart bucket size giving about TARGET_POINTS points for the window."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    return max(window_s // TARGET_POINTS, MIN_BUCKET_S)


def _to_time(epoch_seconds: pd.Series, tz: tzinfo | None) -> pd.Series:
    times = pd.to_datetime(epoch_seconds, unit="s", utc=True)
    return times.dt.tz_convert(tz) if tz is not None else times


def timeseries_frame(rows: list[dict[str, Any]], tz: tzinfo | None = None) -> pd.DataFrame:
    """Per-bucket frame with LLM-call counts and cumulative cost vs always-large baseline."""
    if not rows:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)
    df = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    df["time"] = _to_time(df["bucket"], tz)
    df["llm_calls"] = df["requests"] - df["cache_hits"] - df["errors"]
    df["cum_cost_usd"] = df["cost_usd"].cumsum()
    df["cum_baseline_usd"] = df["baseline_usd"].cumsum()
    return df[TIMESERIES_COLUMNS]


def kpis(summary: dict[str, Any]) -> dict[str, float]:
    """Headline numbers from CostTracker.summary()."""
    overall = summary["latency_s"].get("all", {})
    return {
        "requests": summary["requests"],
        "errors": summary["errors"],
        "hit_rate": summary["hit_rate"],
        "large_share": summary["large_share"],
        "cost_usd": summary["cost_usd"],
        "baseline_usd": summary["baseline_usd"],
        "savings_usd": summary["savings_usd"],
        "savings_pct": summary["savings_pct"],
        "p50_ms": overall.get("p50", 0.0) * 1000,
        "p95_ms": overall.get("p95", 0.0) * 1000,
    }


def outcome_frame(summary: dict[str, Any]) -> pd.DataFrame:
    counts = {
        "cache_hit": summary["cache_hits"],
        "small": summary["small"],
        "large": summary["large"],
        "error": summary["errors"],
    }
    return pd.DataFrame(
        {"outcome": list(OUTCOME_ORDER), "requests": [counts[o] for o in OUTCOME_ORDER]}
    )


def latency_frame(summary: dict[str, Any]) -> pd.DataFrame:
    """Latency table in milliseconds, one row per outcome that has data."""
    latency = summary["latency_s"]
    rows = []
    for outcome in LATENCY_ORDER:
        stats = latency.get(outcome)
        if not stats or stats["count"] == 0:
            continue
        rows.append(
            {
                "outcome": outcome,
                "requests": stats["count"],
                "avg_ms": round(stats["avg"] * 1000, 1),
                "p50_ms": round(stats["p50"] * 1000, 1),
                "p95_ms": round(stats["p95"] * 1000, 1),
            }
        )
    return pd.DataFrame(rows, columns=LATENCY_COLUMNS)


def recent_frame(rows: list[dict[str, Any]], tz: tzinfo | None = None) -> pd.DataFrame:
    """Recent requests table, newest first (input order is kept)."""
    if not rows:
        return pd.DataFrame(columns=RECENT_COLUMNS)
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "time": _to_time(df["ts"], tz),
            "outcome": df["outcome"],
            "model": df["model"],
            "score": pd.to_numeric(df["score"]).round(3),
            "tokens_in": df["tokens_in"],
            "tokens_out": df["tokens_out"],
            "cost_usd": df["cost_usd"].round(6),
            "latency_ms": (df["latency_s"] * 1000).round(1),
            "query": df["query_preview"],
            "error": df["error"],
        }
    )
    return out[RECENT_COLUMNS]
