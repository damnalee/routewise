"""Streamlit dashboard for RouteWise.

Run from the project root so the relative DB path resolves:
    python -m streamlit run src/routewise/dashboard.py

Metrics come straight from the SQLite file the gateway writes (WAL mode allows a concurrent
reader). Cache internals (entries, guard rejections) come from the gateway's /v1/stats when
it is reachable.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import httpx
import streamlit as st

from routewise.config import get_settings
from routewise.dashboard_data import (
    WINDOWS,
    bucket_for_window,
    kpis,
    latency_frame,
    outcome_frame,
    recent_frame,
    timeseries_frame,
)
from routewise.tracker import CostTracker

st.set_page_config(page_title="RouteWise", layout="wide")


@st.cache_resource
def get_tracker() -> CostTracker:
    return CostTracker.from_settings(get_settings())


def fetch_gateway_stats(base_url: str, api_key: str) -> dict[str, Any] | None:
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/stats", headers=headers, timeout=1.5)
        return response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def render(window_s: int, bucket_s: int, gateway_url: str) -> None:
    settings = get_settings()
    tracker = get_tracker()
    tz = datetime.now().astimezone().tzinfo
    since = time.time() - window_s

    summary = tracker.summary(since)
    kp = kpis(summary)
    if kp["requests"] == 0:
        st.info("No requests in this window. Send traffic: python scripts/simulate_traffic.py")
        return

    top = st.columns(5)
    top[0].metric("Requests", f"{int(kp['requests']):,}")
    top[1].metric("Cache hit rate", f"{kp['hit_rate']:.0%}")
    top[2].metric("Spend", f"${kp['cost_usd']:.4f}")
    top[3].metric("Saved vs always-large", f"${kp['savings_usd']:.4f}", f"{kp['savings_pct']:.0%}")
    top[4].metric("Errors", f"{int(kp['errors']):,}")

    bottom = st.columns(4)
    bottom[0].metric("Latency p50", f"{kp['p50_ms']:.0f} ms")
    bottom[1].metric("Latency p95", f"{kp['p95_ms']:.0f} ms")
    bottom[2].metric("Large-model share", f"{kp['large_share']:.0%}")
    bottom[3].metric("Cost pressure", f"{tracker.cost_pressure():.1%}")

    gateway = fetch_gateway_stats(gateway_url, settings.gateway_api_key)
    if gateway is not None:
        cache = gateway["cache"]
        st.caption(
            f"Gateway cache: {cache['size']} entries, {cache['rejections']} guard rejections, "
            f"{cache['evictions']} evictions"
        )
    else:
        st.caption("Gateway not reachable. Cache internals hidden, metrics still read from SQLite.")

    series = timeseries_frame(tracker.timeseries(bucket_s, since=since), tz)
    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative spend vs always-large")
        st.line_chart(
            series.set_index("time")[["cum_cost_usd", "cum_baseline_usd"]].rename(
                columns={"cum_cost_usd": "RouteWise", "cum_baseline_usd": "Always large model"}
            )
        )
    with right:
        st.subheader("Requests per interval")
        st.bar_chart(
            series.set_index("time")[["cache_hits", "llm_calls", "errors"]].rename(
                columns={"cache_hits": "Cache hits", "llm_calls": "LLM calls", "errors": "Errors"}
            )
        )

    a, b = st.columns(2)
    with a:
        st.subheader("Latency by outcome")
        st.dataframe(latency_frame(summary), hide_index=True)
    with b:
        st.subheader("Outcome mix")
        st.bar_chart(outcome_frame(summary).set_index("outcome"))

    st.subheader("Recent requests")
    st.dataframe(recent_frame(tracker.recent(50), tz), hide_index=True)
    st.caption(
        f"Metrics DB: {settings.db_path}. Prices are reference values from config, verify at "
        "console.groq.com."
    )


def main() -> None:
    st.title("RouteWise")
    st.caption("Semantic caching and cost-aware routing: live cost and latency.")

    with st.sidebar:
        st.header("View")
        label = st.selectbox("Time window", list(WINDOWS), index=1)
        auto = st.checkbox("Auto-refresh", value=True)
        interval = st.slider("Refresh every (s)", 2, 60, 5, disabled=not auto)
        st.header("Gateway")
        gateway_url = st.text_input("Gateway URL", "http://localhost:8000")
        st.button("Refresh now")

    window_s = WINDOWS[label]
    bucket_s = bucket_for_window(window_s)
    if auto:
        st.fragment(run_every=float(interval))(render)(window_s, bucket_s, gateway_url)
    else:
        render(window_s, bucket_s, gateway_url)


main()
