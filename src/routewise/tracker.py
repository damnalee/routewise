"""Per-request cost and latency tracking backed by SQLite.

Every gateway request is recorded with its outcome:

  cache_hit   answered from the semantic cache, no LLM cost
  small       answered by the small model
  large       answered by the large model
  error       provider failure, no cost

Savings are measured against a baseline: what the same request would have cost if the
large model had answered it. Cache hits and small-model answers both count as savings.

Also provides `cost_pressure()`, the recent spend as a fraction of the budget, which the
router uses to push borderline queries toward the small model.

SQLite details: WAL mode, one connection guarded by a lock, schema versioned through
PRAGMA user_version. Calls are short and blocking, so async code should wrap them in
asyncio.to_thread.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from routewise.config import Settings

Outcome = Literal["cache_hit", "small", "large", "error"]
_OUTCOMES = frozenset(get_args(Outcome))

_SCHEMA_VERSION = 1
_PREVIEW_CHARS = 120
_ERROR_CHARS = 300

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    namespace     TEXT    NOT NULL,
    query_preview TEXT    NOT NULL,
    outcome       TEXT    NOT NULL
                  CHECK (outcome IN ('cache_hit', 'small', 'large', 'error')),
    model         TEXT,
    score         REAL,
    tokens_in     INTEGER NOT NULL,
    tokens_out    INTEGER NOT NULL,
    cost_usd      REAL    NOT NULL,
    baseline_usd  REAL    NOT NULL,
    latency_s     REAL    NOT NULL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


@dataclass(frozen=True, slots=True)
class CostModel:
    """Prices in USD per 1M tokens."""

    small_in: float
    small_out: float
    large_in: float
    large_out: float

    @classmethod
    def from_settings(cls, settings: Settings) -> CostModel:
        return cls(
            small_in=settings.small_price_in,
            small_out=settings.small_price_out,
            large_in=settings.large_price_in,
            large_out=settings.large_price_out,
        )

    def small_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.small_in + tokens_out * self.small_out) / 1_000_000

    def large_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.large_in + tokens_out * self.large_out) / 1_000_000

    def cost_for(self, outcome: str, tokens_in: int, tokens_out: int) -> float:
        """Actual cost of serving a request with this outcome."""
        if outcome == "small":
            return self.small_cost(tokens_in, tokens_out)
        if outcome == "large":
            return self.large_cost(tokens_in, tokens_out)
        return 0.0

    def baseline_for(self, outcome: str, tokens_in: int, tokens_out: int) -> float:
        """What the request would have cost on the large model."""
        return 0.0 if outcome == "error" else self.large_cost(tokens_in, tokens_out)


@dataclass(frozen=True, slots=True)
class RequestRecord:
    ts: float
    outcome: Outcome
    latency_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    namespace: str = "default"
    query: str = ""
    model: str | None = None
    score: float | None = None
    error: str | None = None


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


class CostTracker:
    def __init__(
        self,
        db_path: Path | str,
        cost_model: CostModel,
        budget_usd: float = 1.0,
        budget_window_s: int = 3600,
        store_queries: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if budget_usd <= 0 or budget_window_s <= 0:
            raise ValueError("budget_usd and budget_window_s must be positive")
        self._cost_model = cost_model
        self._budget_usd = budget_usd
        self._budget_window_s = budget_window_s
        self._store_queries = store_queries
        self._clock = clock
        self._lock = threading.RLock()

        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    @classmethod
    def from_settings(cls, settings: Settings, store_queries: bool = True) -> CostTracker:
        return cls(
            db_path=settings.db_path,
            cost_model=CostModel.from_settings(settings),
            budget_usd=settings.budget_usd,
            budget_window_s=settings.budget_window_s,
            store_queries=store_queries,
        )

    @property
    def cost_model(self) -> CostModel:
        return self._cost_model

    def _init_db(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema v{version} is newer than supported v{_SCHEMA_VERSION}"
                )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._conn.commit()

    # ------------------------------------------------------------------ writes

    def record(self, rec: RequestRecord) -> None:
        if rec.outcome not in _OUTCOMES:
            raise ValueError(f"unknown outcome: {rec.outcome!r}")
        if rec.tokens_in < 0 or rec.tokens_out < 0 or rec.latency_s < 0:
            raise ValueError("tokens and latency must be non-negative")

        cost = self._cost_model.cost_for(rec.outcome, rec.tokens_in, rec.tokens_out)
        baseline = self._cost_model.baseline_for(rec.outcome, rec.tokens_in, rec.tokens_out)
        preview = " ".join(rec.query.split())[:_PREVIEW_CHARS] if self._store_queries else ""
        error = rec.error[:_ERROR_CHARS] if rec.error else None

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO requests (ts, namespace, query_preview, outcome, model, score, "
                "tokens_in, tokens_out, cost_usd, baseline_usd, latency_s, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.ts,
                    rec.namespace,
                    preview,
                    rec.outcome,
                    rec.model,
                    rec.score,
                    rec.tokens_in,
                    rec.tokens_out,
                    cost,
                    baseline,
                    rec.latency_s,
                    error,
                ),
            )

    def prune(self, older_than_ts: float) -> int:
        """Delete rows older than the timestamp. Returns the number removed."""
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM requests WHERE ts < ?", (older_than_ts,))
            return cursor.rowcount

    # ------------------------------------------------------------------- reads

    @staticmethod
    def _where(since: float | None) -> tuple[str, list[float]]:
        return ("WHERE ts >= ?", [since]) if since is not None else ("", [])

    def summary(self, since: float | None = None) -> dict[str, Any]:
        """Totals, savings vs always-large, and latency stats per outcome."""
        where, params = self._where(since)
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(outcome = 'cache_hit'), 0) AS hits, "
                "COALESCE(SUM(outcome = 'small'), 0) AS small, "
                "COALESCE(SUM(outcome = 'large'), 0) AS large, "
                "COALESCE(SUM(outcome = 'error'), 0) AS errors, "
                "COALESCE(SUM(tokens_in), 0) AS tokens_in, "
                "COALESCE(SUM(tokens_out), 0) AS tokens_out, "
                "COALESCE(SUM(cost_usd), 0.0) AS cost, "
                "COALESCE(SUM(baseline_usd), 0.0) AS baseline "
                f"FROM requests {where}",
                params,
            ).fetchone()
            latency_rows = self._conn.execute(
                f"SELECT outcome, latency_s FROM requests {where}", params
            ).fetchall()

        by_outcome: dict[str, list[float]] = defaultdict(list)
        for r in latency_rows:
            by_outcome[r["outcome"]].append(r["latency_s"])
        latency = {outcome: _latency_stats(vals) for outcome, vals in by_outcome.items()}
        latency["all"] = _latency_stats(
            [v for outcome, vals in by_outcome.items() if outcome != "error" for v in vals]
        )

        total, errors = row["n"], row["errors"]
        served = total - errors
        llm_calls = row["small"] + row["large"]
        cost, baseline = float(row["cost"]), float(row["baseline"])
        savings = baseline - cost
        return {
            "requests": total,
            "served": served,
            "errors": errors,
            "cache_hits": row["hits"],
            "small": row["small"],
            "large": row["large"],
            "hit_rate": row["hits"] / served if served else 0.0,
            "large_share": row["large"] / llm_calls if llm_calls else 0.0,
            "tokens_in": row["tokens_in"],
            "tokens_out": row["tokens_out"],
            "cost_usd": cost,
            "baseline_usd": baseline,
            "savings_usd": savings,
            "savings_pct": savings / baseline if baseline > 0 else 0.0,
            "latency_s": latency,
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent requests, newest first."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def timeseries(self, bucket_s: int, since: float | None = None) -> list[dict[str, Any]]:
        """Per-bucket counts, cost and latency for dashboard charts."""
        if bucket_s <= 0:
            raise ValueError("bucket_s must be positive")
        where, params = self._where(since)
        with self._lock:
            rows = self._conn.execute(
                "SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, "
                "COUNT(*) AS requests, "
                "SUM(outcome = 'cache_hit') AS cache_hits, "
                "SUM(outcome = 'error') AS errors, "
                "SUM(cost_usd) AS cost_usd, "
                "SUM(baseline_usd) AS baseline_usd, "
                "AVG(latency_s) AS avg_latency_s "
                f"FROM requests {where} GROUP BY bucket ORDER BY bucket",
                [bucket_s, bucket_s, *params],
            ).fetchall()
        return [dict(r) for r in rows]

    def cost_pressure(self, now: float | None = None) -> float:
        """Recent spend as a fraction of the budget, clamped to [0, 1]."""
        current = self._clock() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM requests WHERE ts >= ?",
                (current - self._budget_window_s,),
            ).fetchone()
        return min(max(float(row[0]) / self._budget_usd, 0.0), 1.0)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
