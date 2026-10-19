from datetime import timedelta, timezone
from typing import Any

import pytest

from routewise.dashboard_data import (
    LATENCY_COLUMNS,
    RECENT_COLUMNS,
    TIMESERIES_COLUMNS,
    bucket_for_window,
    kpis,
    latency_frame,
    outcome_frame,
    recent_frame,
    timeseries_frame,
)
from routewise.tracker import CostModel, CostTracker, RequestRecord

# 1000 in / 500 out: small = 0.002, large = 0.02
MODEL = CostModel(small_in=1.0, small_out=2.0, large_in=10.0, large_out=20.0)


def rec(outcome: Any, ts: float, latency: float, **kwargs: Any) -> RequestRecord:
    return RequestRecord(
        ts=ts, outcome=outcome, latency_s=latency, tokens_in=1000, tokens_out=500, **kwargs
    )


def seeded() -> CostTracker:
    tracker = CostTracker(":memory:", MODEL)
    tracker.record(rec("small", 0.0, 1.0, query="first"))
    tracker.record(rec("cache_hit", 30.0, 0.01))
    tracker.record(rec("large", 70.0, 2.0))
    return tracker


def test_bucket_for_window() -> None:
    assert bucket_for_window(900) == 15
    assert bucket_for_window(3600) == 60
    assert bucket_for_window(60) == 5
    with pytest.raises(ValueError):
        bucket_for_window(0)


def test_timeseries_frame_counts_and_cumulative_sums() -> None:
    frame = timeseries_frame(seeded().timeseries(bucket_s=60))
    assert list(frame.columns) == TIMESERIES_COLUMNS
    assert frame["requests"].tolist() == [2, 1]
    assert frame["cache_hits"].tolist() == [1, 0]
    assert frame["llm_calls"].tolist() == [1, 1]
    assert frame["cum_cost_usd"].tolist() == pytest.approx([0.002, 0.022])
    assert frame["cum_baseline_usd"].tolist() == pytest.approx([0.04, 0.06])


def test_timeseries_frame_converts_timezone() -> None:
    tz = timezone(timedelta(hours=5))
    frame = timeseries_frame(seeded().timeseries(bucket_s=60), tz)
    assert frame["time"].iloc[0].hour == 5


def test_empty_frames_keep_columns() -> None:
    assert list(timeseries_frame([]).columns) == TIMESERIES_COLUMNS
    assert list(recent_frame([]).columns) == RECENT_COLUMNS
    assert len(timeseries_frame([])) == 0


def test_kpis() -> None:
    tracker = seeded()
    tracker.record(rec("error", 80.0, 0.5))
    k = kpis(tracker.summary())
    assert k["requests"] == 4 and k["errors"] == 1
    assert k["hit_rate"] == pytest.approx(1 / 3)
    assert k["cost_usd"] == pytest.approx(0.022)
    assert k["savings_usd"] == pytest.approx(0.038)
    assert k["p50_ms"] == pytest.approx(1000.0)
    assert k["p95_ms"] == pytest.approx(1900.0)


def test_kpis_on_empty_summary() -> None:
    k = kpis(CostTracker(":memory:", MODEL).summary())
    assert k["requests"] == 0 and k["p50_ms"] == 0.0 and k["hit_rate"] == 0.0


def test_latency_frame_order_and_units() -> None:
    frame = latency_frame(seeded().summary())
    assert list(frame.columns) == LATENCY_COLUMNS
    assert frame["outcome"].tolist() == ["all", "cache_hit", "small", "large"]
    small = frame[frame["outcome"] == "small"].iloc[0]
    assert small["p50_ms"] == 1000.0 and small["requests"] == 1


def test_outcome_frame_counts() -> None:
    frame = outcome_frame(seeded().summary())
    assert frame["outcome"].tolist() == ["cache_hit", "small", "large", "error"]
    assert frame["requests"].tolist() == [1, 1, 1, 0]


def test_recent_frame_newest_first_with_units() -> None:
    frame = recent_frame(seeded().recent(10))
    assert list(frame.columns) == RECENT_COLUMNS
    assert frame["outcome"].tolist() == ["large", "cache_hit", "small"]
    assert frame["latency_ms"].iloc[0] == 2000.0
    assert frame["query"].iloc[2] == "first"
