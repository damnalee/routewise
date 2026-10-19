import threading
from pathlib import Path
from typing import Any

import pytest

from routewise.config import Settings
from routewise.tracker import CostModel, CostTracker, RequestRecord

# USD per 1M tokens. With 1000 in / 500 out: small = 0.002, large = 0.02.
MODEL = CostModel(small_in=1.0, small_out=2.0, large_in=10.0, large_out=20.0)


def make_tracker(**kwargs: Any) -> CostTracker:
    return CostTracker(":memory:", MODEL, **kwargs)


def rec(outcome: Any, ts: float = 1000.0, latency: float = 1.0, **kwargs: Any) -> RequestRecord:
    kwargs.setdefault("tokens_in", 1000)
    kwargs.setdefault("tokens_out", 500)
    return RequestRecord(ts=ts, outcome=outcome, latency_s=latency, **kwargs)


def test_cost_model_costs_and_baseline() -> None:
    assert MODEL.cost_for("small", 1000, 500) == pytest.approx(0.002)
    assert MODEL.cost_for("large", 1000, 500) == pytest.approx(0.02)
    assert MODEL.cost_for("cache_hit", 1000, 500) == 0.0
    assert MODEL.cost_for("error", 1000, 500) == 0.0
    assert MODEL.baseline_for("small", 1000, 500) == pytest.approx(0.02)
    assert MODEL.baseline_for("cache_hit", 1000, 500) == pytest.approx(0.02)
    assert MODEL.baseline_for("error", 1000, 500) == 0.0


def test_summary_totals_and_savings() -> None:
    tracker = make_tracker()
    for outcome in ("cache_hit", "small", "large", "error"):
        tracker.record(rec(outcome))
    s = tracker.summary()

    assert (s["requests"], s["served"], s["errors"]) == (4, 3, 1)
    assert (s["cache_hits"], s["small"], s["large"]) == (1, 1, 1)
    assert s["hit_rate"] == pytest.approx(1 / 3)
    assert s["large_share"] == pytest.approx(0.5)
    assert s["cost_usd"] == pytest.approx(0.022)
    assert s["baseline_usd"] == pytest.approx(0.06)
    assert s["savings_usd"] == pytest.approx(0.038)
    assert s["savings_pct"] == pytest.approx(0.038 / 0.06)
    assert s["tokens_in"] == 4000 and s["tokens_out"] == 2000


def test_empty_summary_is_all_zero() -> None:
    s = make_tracker().summary()
    assert s["requests"] == 0 and s["hit_rate"] == 0.0 and s["savings_pct"] == 0.0
    assert s["latency_s"]["all"]["count"] == 0


def test_latency_percentiles_per_outcome() -> None:
    tracker = make_tracker()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        tracker.record(rec("small", latency=value))
    tracker.record(rec("cache_hit", latency=0.01))
    tracker.record(rec("error", latency=9.0))

    latency = tracker.summary()["latency_s"]
    assert latency["small"]["count"] == 5
    assert latency["small"]["avg"] == pytest.approx(3.0)
    assert latency["small"]["p50"] == pytest.approx(3.0)
    assert latency["small"]["p95"] == pytest.approx(4.8)
    assert latency["cache_hit"]["p50"] == pytest.approx(0.01)
    assert latency["all"]["count"] == 6  # errors excluded


def test_since_filters_rows() -> None:
    tracker = make_tracker()
    tracker.record(rec("small", ts=100.0))
    tracker.record(rec("small", ts=200.0))
    assert tracker.summary(since=150.0)["requests"] == 1
    assert tracker.summary()["requests"] == 2


def test_cost_pressure_uses_window_and_clamps() -> None:
    tracker = make_tracker(budget_usd=0.04, budget_window_s=3600, clock=lambda: 10_000.0)
    assert tracker.cost_pressure() == 0.0
    tracker.record(rec("large", ts=9_000.0))
    assert tracker.cost_pressure() == pytest.approx(0.5)
    tracker.record(rec("large", ts=1_000.0))  # outside the window
    assert tracker.cost_pressure() == pytest.approx(0.5)
    tracker.record(rec("large", ts=9_500.0))
    assert tracker.cost_pressure() == pytest.approx(1.0)
    tracker.record(rec("large", ts=9_600.0))
    assert tracker.cost_pressure() == pytest.approx(1.0)


def test_cache_hits_and_small_answers_do_not_add_pressure_beyond_their_cost() -> None:
    tracker = make_tracker(budget_usd=0.02, clock=lambda: 1000.0)
    tracker.record(rec("cache_hit"))
    assert tracker.cost_pressure() == 0.0
    tracker.record(rec("small"))
    assert tracker.cost_pressure() == pytest.approx(0.1)


def test_timeseries_buckets() -> None:
    tracker = make_tracker()
    tracker.record(rec("small", ts=0.0, latency=1.0))
    tracker.record(rec("cache_hit", ts=30.0, latency=0.0))
    tracker.record(rec("large", ts=70.0, latency=2.0))
    buckets = tracker.timeseries(bucket_s=60)
    assert [b["bucket"] for b in buckets] == [0, 60]
    assert [b["requests"] for b in buckets] == [2, 1]
    assert buckets[0]["cache_hits"] == 1
    assert buckets[0]["avg_latency_s"] == pytest.approx(0.5)
    assert buckets[1]["cost_usd"] == pytest.approx(0.02)


def test_recent_is_newest_first_and_previews_are_cleaned() -> None:
    tracker = make_tracker()
    tracker.record(rec("small", query="a   b\n c"))
    tracker.record(rec("large", query="x" * 500))
    rows = tracker.recent(limit=2)
    assert rows[0]["outcome"] == "large" and len(rows[0]["query_preview"]) == 120
    assert rows[1]["query_preview"] == "a b c"


def test_store_queries_false_keeps_no_text() -> None:
    tracker = make_tracker(store_queries=False)
    tracker.record(rec("small", query="secret customer data"))
    assert tracker.recent()[0]["query_preview"] == ""


def test_error_rows_keep_truncated_message() -> None:
    tracker = make_tracker()
    tracker.record(rec("error", error="boom " * 200))
    row = tracker.recent()[0]
    assert row["cost_usd"] == 0.0 and len(row["error"]) == 300


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "metrics.db"
    first = CostTracker(db, MODEL)
    first.record(rec("small"))
    first.close()
    second = CostTracker(db, MODEL)
    assert second.summary()["requests"] == 1
    second.close()


def test_prune_removes_old_rows() -> None:
    tracker = make_tracker()
    for ts in (100.0, 200.0, 300.0):
        tracker.record(rec("small", ts=ts))
    assert tracker.prune(older_than_ts=250.0) == 2
    assert tracker.summary()["requests"] == 1


def test_invalid_input_is_rejected() -> None:
    tracker = make_tracker()
    with pytest.raises(ValueError):
        tracker.record(rec("small", latency=-1.0))
    with pytest.raises(ValueError):
        tracker.record(rec("small", tokens_in=-5))
    with pytest.raises(ValueError):
        tracker.record(rec("bogus"))
    with pytest.raises(ValueError):
        tracker.timeseries(bucket_s=0)
    with pytest.raises(ValueError):
        tracker.recent(limit=0)
    with pytest.raises(ValueError):
        make_tracker(budget_usd=0)
    with pytest.raises(ValueError):
        make_tracker(budget_window_s=0)


def test_concurrent_writes_are_safe() -> None:
    tracker = make_tracker()

    def worker() -> None:
        for _ in range(50):
            tracker.record(rec("small"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.summary()["requests"] == 400


def test_from_settings_uses_configured_prices() -> None:
    settings = Settings(_env_file=None, db_path=":memory:", budget_usd=2.0)  # type: ignore[arg-type]
    tracker = CostTracker.from_settings(settings)
    assert tracker.cost_model.small_in == settings.small_price_in
    assert tracker.cost_model.large_out == settings.large_price_out
